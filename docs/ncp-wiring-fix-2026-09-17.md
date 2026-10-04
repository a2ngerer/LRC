# NCP wiring fix (2026-09-17)

## What was wrong

`src/wirings/ncp.py::NCPWiring` implemented "NCP wiring" as **three stacked RNN
layers** (inter -> command -> motor). Each layer was internally *fully*
recurrent (dense recurrent weights); the only sparsity was a `SparseLinear`
mask *between* the layers, and `elapsed_time` (dt) reached the first layer
only — the command and motor layers always integrated with dt = 1.

That is not the Neural Circuit Policy of Lechner/Hasani. It measures network
depth plus inter-layer sparsity.

## Reference semantics (ncps)

`.venv/.../ncps/keras/ltc_cell.py` (wired LTC) is **one cell**:

* the hidden state holds inter + command + motor neurons at once (ncps orders
  them `[motor | command | inter]`), and the "sensory neurons" are simply the
  input features;
* `sparsity_mask = |wiring.adjacency_matrix|` multiplies the recurrent synapse
  weights `w`, `sensory_sparsity_mask = |wiring.sensory_adjacency_matrix|`
  multiplies `sensory_w`;
* recurrence therefore exists only where the NCP adjacency allows it
  (command <-> command); motor neurons receive from command neurons only;
* one synchronous state update per timestep, so `elapsed_time` reaches every
  neuron.

`wired_cfc_cell.py` masks CfC the same way: a `[input_sparsity; recurrent_sparsity]`
row-concatenated mask applied to the `[x, h] -> units` kernels.

## What changed

* **`NCPWiring` (key `'ncp'`)** is now one cell with
  `units = inter + command + motor`, one hidden vector, synchronous update,
  `elapsed_time` every step. The graph comes from `ncps.wirings.NCP(...)` with
  the same fanout defaults and seed as before, built with the **real input
  dimension** (the old code built it with `input_shape=0`, so it had no
  sensory adjacency at all). Model output = the motor slice of the hidden
  state, i.e. the leading `motor_neurons` columns, shape
  `(batch, T, motor_neurons)`.
* **Masks** are constants, applied by multiplication on every forward pass, so
  gradients of masked-off weights are exactly zero. They are passed to the cell
  as the optional `sparsity_mask` / `sensory_mask` constructor arguments
  (`BaseCell`); `None` (the default) is byte-identical to the old dense
  behaviour.
* **Per cell family**
  * `ltc`, `lrc`, `lrc_ar`: the explicit synapse tensors are masked like ncps —
    `w`, `h` (and `erev`, in LTC) by the adjacency mask, `sensory_w`,
    `sensory_h` by the sensory mask. The `mu`/`sigma` shape parameters are not
    masked (they only multiply an already-masked synapse) but are counted as
    switched off, see below. `lrc_ar` is autoregressive (its input *is* its
    state), so it has no sensory synapses and stays unusable under `ncp`, as
    before.
  * `ctrnn`: `W_x` by the sensory mask, `W_h` by the adjacency mask.
  * `cfc`, `cfc_lrc` (+ their mixed-memory variants): the `[x, h] -> units`
    maps — `backbone_0` and the elastance head — carry
    `concat([sensory_mask; adjacency_mask])`. The downstream heads read
    backbone features, not `[x, h]`, and are therefore dense (as in ncps). If
    `backbone_units != units` the cell raises a `ValueError` under `ncp`
    (affects `cfc_pm`, whose wider backbone is its defining property).
  * `gru`, `lstm`, mixed memory: the Keras cell's `kernel` and
    `recurrent_kernel` are masked per gate block (3 resp. 4 blocks).
* **`NCPStackedWiring` (key `'ncp_stacked'`)** is the old implementation,
  marked deprecated. `experiments/run_benchmark.py --wiring ncp_stacked` and
  campaign configs can still select it to reproduce pre-2026-09-17 runs. The
  frozen legacy campaign profiles keep emitting `wiring: 'ncp'` so the golden
  equivalence tests and the on-disk result filenames are unchanged.
* `src/wirings/cncp.py` is behaviourally untouched; its nodes are intentionally
  *composite* (whole sub-cells, one per cortical lamina), which is a documented
  design choice, not the bug fixed here.

## Status of pre-2026-09-17 'ncp' runs

All runs before 2026-09-17 with `wiring: ncp` are **invalid as NCP evidence**.
They must not be cited as NCP results and must not be used as a precondition
for new results. What actually ran was three stacked, internally fully
recurrent RNN layers with sparse NCP-fanout transitions between them: a deep
dense network, not an NCP, and `elapsed_time` reached only the first layer.

They are not useless. They produced methodological findings that shaped the
final design and remain as development history:

* solver stability / stiffness of the numerical ODE cells,
* the teacher-forcing leak,
* the parameter-matched `cfc_pm` capacity control,
* closed-form LRC beating the numerical LRC.

All experiments will be re-run in the final suite with the corrected `ncp`.
Reproducing an old run requires selecting `ncp_stacked` explicitly. See also
`results/README.md`.

## Signal propagation within one input step (2026-10-03)

A true NCP routes sensory -> inter -> command -> motor. The single-step
neural-ODE task (`experiments/run_benchmark.py`) calls the model once per
timestep from a zero state, so the motor neurons must see the input within
ONE call. The two cell families follow the two ncps reference cells:

* **ODE cells** (`ltc`, `lrc` + its eps variants, `ctrnn`, `mm_ltc`, `mm_lrc`):
  one synchronous masked cell, as ncps' wired `LTCCell`. Each of the
  `ode_unfolds` sub-steps updates every neuron and moves a signal one synapse,
  so >= 3 sub-steps reach the motor neurons. Verified: with `ode_unfolds=1` the
  gradient of the motor state w.r.t. the input is exactly zero, with 6 it is
  not (`tests/wirings/test_ncp_wiring.py`). The previously reported
  input-independent motor output had two causes: `lrc` defaults to
  `ode_unfolds=1`, and `ctrnn` had a single Euler step with no sub-steps.
  Fixes: `CTRNN_Cell` takes `ode_unfolds` (default 1 = old behaviour), and
  `run_benchmark.build_model` sets `ode_unfolds=6` (ncps default) for
  single-step ODE cells under `ncp` unless the spec sets it; an explicit value
  < 3 emits a `RuntimeWarning` (affects the eps `uf=1` ncp arms). `ltc`
  already defaults to 6. Caveat: at initialisation the `lrc` motor
  sensitivity is ~1e-8 (three sigmoid synapses in series, small `vleak`);
  it is non-zero but weak.
* **Closed-form and discrete cells** (`cfc`, `cfc_lrc`, `cfc_lrc_outer`,
  `gru`, `lstm`, `cfc_mm_lrc`, `cfc_mm_ltc`): no sub-steps, so a synchronous
  update would move the signal one layer per input step (`gru`/`lstm` emitted
  exactly zero). They run `NCPLayeredCell`, the ncps `WiredCfCCell` semantics:
  within one call inter <- input, command <- new inter + old command, motor <-
  new command. Each layer is a masked sub-cell; its input mask is the NCP
  adjacency from the previous layer, its recurrent mask the intra-layer
  adjacency (empty for inter and motor, command <-> command for command).
  Every layer gets the same `elapsed_time`. Deviation from ncps: ncps defaults
  to `fully_recurrent=True` (dense intra-layer recurrence); here it is masked
  by the adjacency. The hidden state is still one concatenated vector per state
  slot in ncps order `[motor | command | inter]`, so the motor slice, state
  handling and `effective_param_count` are unchanged.

  Resolved 2026-10-04 (see below): CfC-family sub-cells run with
  `backbone_layers=0`, so there is no lateral mixing inside a layer.

The `RuntimeWarning` for every `ncp` build on the ODE task is removed.

## Effective parameter count and budget matching

`src/wirings/ncp.py::effective_param_count(model)` returns the trainable
parameter count minus every entry switched off by a wiring mask (masked synapse
weights, masked kernels, the per-synapse `mu`/`sigma`/`erev` entries that can
only ever multiply a masked synapse, and the masked entries of `SparseLinear` /
`SignedSparseLinear` in `ncp_stacked` and `cncp`). For unmasked models it
equals `count_params()`. The model must have run one forward pass before
counting, because some masks are applied inside `call()`.

Parameter budgets are matched per arm on effective counts with
`match_param_budget` (unit-step scan of each arm's own width knob, stops at
the first size above the budget):

* `experiments/run_lotka_volterra_hpo.py::size_for_param_budget` (dense / ncp /
  cncp / tbt_cncp*),
* `experiments/run_benchmark.py::budget_size`, used when a spec carries
  `param_budget` (new campaigns only; filename token `_pb<n>`). Arms: dense
  (`units`), ncp / ncp_stacked (inter=size, command=size//2, motor=2), cncp
  (lamina widths scaled by size/16). Specs without `param_budget` -- every
  legacy campaign -- keep the frozen sizes, so the golden equivalence tests
  are untouched.

## Follow-ups (2026-10-04)

* **No lateral mixing in CfC NCP layers.** Under `ncp`, `NCPLayeredCell`
  builds the `cfc` / `cfc_lrc` / `cfc_lrc_outer` / `cfc_mm_lrc` /
  `cfc_mm_ltc` sub-cells with `backbone_layers=0`
  (`src/wirings/ncp.py::ncp_cell_kwargs`, an explicit spec value wins), as
  ncps `WiredCfCCell`. The heads `ff1`, `ff2`, `time_a`, `time_b` then act
  directly on `[x, h]` and each carries the concat mask (`CfC_Cell` /
  `CfC_LRC_Cell` route the heads through `_masked_dense` when there is no
  backbone; the elastance head already read raw `[x, h]`). Dense models keep
  `backbone_layers=1` and are byte-identical. `cfc_pm` under `ncp` still raises
  (its wider backbone is its defining property). Budget-matched sizes at
  `param_budget=4000` (effective params): ODE task (`run_benchmark.budget_size`)
  `cfc` size 57 (4016), `cfc_lrc` 51 (3970); person_activity `cfc` 44 (4077),
  `cfc_lrc` 39 (3845); Lotka-Volterra `cfc` 48 (4170), `cfc_lrc` 43 (3984).
  Test: `tests/wirings/test_ncp_wiring.py::test_cfc_ncp_subcells_have_no_lateral_kernel`.
* **Provenance.** `run_benchmark.build_model` attaches `effective_config`
  (effective `cell_kwargs` incl. injected `ode_unfolds` / `backbone_layers`,
  resolved `size`, `units`, `ncp` sizes, `lamina_units`) to the model;
  `run_one` writes it into the result `config` under `effective`, and the
  top-level `config.ode_unfolds` now holds the effective value.
* **Parameter reporting.** `src/wirings/ncp.py::param_counts(model)` returns
  `params_effective` (primary, `effective_param_count`) and `params_raw`
  (`count_params()`). All runners write both (`run_benchmark` in `config`,
  `run_lotka_volterra_hpo`, and the person_activity, data_efficiency,
  active_sensing, active_glimpse, voting, committee, lotka_volterra runners,
  which no longer write the ambiguous `params`). Readers
  (`plot_person_activity`, `plot_data_efficiency`, `plot_committee`,
  `plot_lotka_volterra`, `aggregate_hpo`) use `params_effective` and fall back
  to the legacy `params` / `matched_params` for old files (legacy `params` is
  the raw count).

### eps ablation: dense only (2026-10-04)

The `eps` and `eps_pilot` campaigns run on the `dense` wiring only; the `ncp` arms
are removed (eps 3600 -> 1800 runs, eps_pilot 640 -> 320). The ablation asks a
question about the cell, and its LRCU arms use `ode_unfolds=1`. Under the corrected
single-cell NCP wiring a one-unfold arm is structurally blind on the single-step
harness: the signal needs three sub-steps to reach the motor neurons. The
interaction of elastance and wiring is tested in the final suite via `cfc_lrc` vs
`cfc_pm`. Neither campaign had been run, so this changes the planned matrix only.
`_build_eps_model` still accepts any wiring and keeps the RuntimeWarning for
`ncp` with `ode_unfolds < 3`.
