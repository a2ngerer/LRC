# Running a campaign

A **campaign** is one benchmark matrix described by a single YAML in
`configs/campaigns/`. The config defines the matrix (cells × wirings × systems ×
seeds, plus extras), optionally the cluster resources, and how to analyse the
results. The pipeline expands the config into a deterministic spec list, submits
it as a SLURM array, fetches the JSONs back, and aggregates — so a new benchmark
means *editing a config*, not editing Python.

The pipeline runs the same way whether driven by an agent or by hand; this page
documents both.

## Defining a campaign

Copy an existing config (`v5.yaml` for an `axes` matrix, `v3.yaml` for `concat`,
`eps.yaml` for `explicit`) and edit it. The field reference and one worked
example per mode live in [`configs/campaigns/_schema.md`](../configs/campaigns/_schema.md).

Validate and inspect the matrix locally before submitting — no cluster needed:

```bash
uv run python experiments/run_campaign.py --config X --count   # number of specs
uv run python experiments/run_campaign.py --config X --list    # every spec + index
```

`--config` takes a profile name (resolved to `configs/campaigns/X.yaml`) or a
path. A malformed config fails here (unknown cell, empty axis, bad `range`),
before any cluster time is spent.

## Agent-driven (default)

Ask the agent to add or run a campaign; it edits the YAML, runs the
`--count`/`--list` checks, submits, fetches, and analyses, gating on the golden
equivalence test (`uv run pytest tests/benchmark/test_equivalence.py`) so the
spec order of existing profiles stays byte-identical.

## By hand

The full manual path, one campaign `X`:

```bash
# 1. Edit the matrix.
$EDITOR configs/campaigns/X.yaml

# 2. Inspect the expanded run list (local, deterministic).
uv run python experiments/run_campaign.py --config X --count
uv run python experiments/run_campaign.py --config X --list

# 3. Submit the SLURM array (rsyncs the code tree, then sbatch).
#    Pass the YAML path, not just the name. TU Wien VPN must be up.
./cluster/submit_campaign.sh configs/campaigns/X.yaml
#    Forward per-run overrides after `--`, e.g.:
#    ./cluster/submit_campaign.sh configs/campaigns/X.yaml -- --iters 4000

# 4. Watch / fetch results when the array finishes.
ssh datalab squeue -u \$USER
./cluster/fetch_results.sh

# 5. Aggregate + plot.
uv run python experiments/analyze_campaign.py --config X
```

`submit_campaign.sh` computes the array size from `run_campaign.py --config X
--count`, applies any `cluster:` resource overrides from the YAML (env vars still
win), submits `cluster/campaign_array.sbatch` (which packs `runs_per_gpu` runs
onto each GPU and respects the dataLAB 8-GPU throttle), and on a successful
`sbatch` appends a work-log entry via `cluster/log_campaign.sh` (see below).

Resources can be overridden per invocation without touching the YAML:

```bash
CLUSTER_MEM=64G CLUSTER_RUNS_PER_GPU=6 ./cluster/submit_campaign.sh configs/campaigns/X.yaml
```

## Auto work-log

After a successful submit, `submit_campaign.sh` calls

```
cluster/log_campaign.sh <date|-> <campaign> <slurm_job_id> <run_count>
```

which appends one bullet to the thesis work-documentation log in the Obsidian
vault (`obsidian_master_thesis/Thesis/work-documentation.md`):

```
- 2026-06-30  campaign `X` submitted — SLURM job 123456, 1980 runs
```

Pass `-` (or nothing) as the date to use today's UTC date. The script is a
**no-op** when the vault log is absent on the host (e.g. on the cluster or a
code-only clone), so the submit path never fails because of it. Override the
target with `CAMPAIGN_WORKLOG=/some/file.md`. The call is guarded, so deleting
`log_campaign.sh` only drops the logging — it does not break submission.

## Files

| File | Role |
|------|------|
| `configs/campaigns/X.yaml` | the campaign definition (matrix + cluster + analysis) |
| `configs/campaigns/_schema.md` | field reference + per-mode worked examples |
| `experiments/run_campaign.py` | expand config → spec list; run a spec (`--index`) |
| `cluster/submit_campaign.sh` | compute array size, sync, sbatch, auto-log |
| `cluster/campaign_array.sbatch` | the generic array job (sources `_common.sh`) |
| `cluster/log_campaign.sh` | append the work-log entry (no-op without the vault) |
| `experiments/analyze_campaign.py` | aggregate + plot from the result dir |
