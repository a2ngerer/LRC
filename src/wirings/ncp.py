import numpy as np
import tensorflow as tf
from ncps.wirings import NCP
from .base_wiring import BaseWiring
from src.neurons.base_cell import BaseCell
from src.neurons import (GRU_Cell, LSTM_Cell, CfC_Cell, CfC_LRC_Cell,
                         CfC_MM_LRC_Cell, CfC_MM_LTC_Cell)

# CfC-family sub-cells under 'ncp' default to backbone_layers=0 (ncps
# WiredCfCCell): the ff1/ff2/time_a/time_b heads act directly on the masked
# [x, h] input, so there is no instantaneous lateral mixing inside a layer.
CFC_NCP_CELLS = (CfC_Cell, CfC_LRC_Cell, CfC_MM_LRC_Cell, CfC_MM_LTC_Cell)


def ncp_cell_kwargs(cell_cls, cell_kwargs):
    """Effective sub-cell kwargs under 'ncp' (explicit values win)."""
    kw = dict(cell_kwargs)
    if issubclass(cell_cls, CFC_NCP_CELLS):
        kw.setdefault('backbone_layers', 0)
    return kw

# Closed-form and discrete cells have no ODE sub-steps, so one synchronous
# update would move a signal only one synapse per input step. Under 'ncp' they
# run the ncps WiredCfCCell semantics instead (NCPLayeredCell): the layers are
# computed sequentially within one step. ODE cells (ltc, lrc, ctrnn, mm_ltc,
# mm_lrc) keep the synchronous single cell of ncps' wired LTCCell, where the
# ode_unfolds sub-steps propagate the signal through the layers.
SEQUENTIAL_NCP_CELLS = (GRU_Cell, LSTM_Cell, CfC_Cell, CfC_LRC_Cell,
                        CfC_MM_LRC_Cell, CfC_MM_LTC_Cell)


def ncp_graph(inter_neurons, command_neurons, motor_neurons, seed, input_dim):
    """Build the ncps NCP graph with this project's fanout defaults.

    The fanouts are the ones the pre-2026-09-17 wiring used, so a given
    (sizes, seed, input_dim) always yields the same graph.
    """
    wiring = NCP(
        inter_neurons=inter_neurons,
        command_neurons=command_neurons,
        motor_neurons=motor_neurons,
        sensory_fanout=max(1, inter_neurons // 2),
        inter_fanout=max(1, command_neurons // 2),
        recurrent_command_synapses=max(1, command_neurons // 2),
        motor_fanin=max(1, command_neurons // 2),
        seed=seed,
    )
    wiring.build(input_dim)
    return wiring


class SparseLinear(tf.keras.layers.Layer):
    """Dense layer with a fixed binary connectivity mask.

    The mask determines which connections exist. Weights at masked-off
    positions are zeroed each forward pass (W * mask), so gradients there
    are also zero — the sparsity is permanent throughout training.

    Args:
        units:  output dimension
        mask:   numpy bool/int array of shape (input_dim, units), 1 = connected
    """

    def __init__(self, units, mask, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self._mask_np = np.array(mask, dtype=np.float32)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W', shape=(input_shape[-1], self.units),
            dtype=tf.float32, initializer='glorot_uniform',
        )
        self.mask = tf.constant(self._mask_np, dtype=tf.float32)
        self.built = True

    def call(self, x):
        return x @ (self.W * self.mask)


class NCPWiring(BaseWiring):
    """Neural Circuit Policy wiring: ONE recurrent cell, sparse synapses.

    Follows Lechner/Hasani (ncps ``wired`` cells, see
    ``ncps/keras/ltc_cell.py``): the hidden state holds all inter + command +
    motor neurons at once, the sensory "neurons" are the input features, and
    the NCP graph is imposed as fixed binary masks on the synapse weights:

        * ``adjacency_mask`` (units, units) masks the state -> state map,
        * ``sensory_mask`` (input_dim, units) masks the input -> state map.

    Recurrence therefore exists only where the NCP adjacency allows it
    (command <-> command); inter neurons never feed back into themselves and
    motor neurons receive only from command neurons. There is ONE synchronous
    state update per step, so ``elapsed_time`` reaches every neuron.

    The model output is the motor slice of the hidden state — ncps orders the
    neurons ``[motor | command | inter]``, so that is the leading
    ``motor_neurons`` columns.

    The masks depend on the input dimension (the sensory adjacency), which is
    only known at build time, so they are handed to the cell as callables and
    materialized in ``BaseCell._build_masks``.

    Args:
        cell_cls:         BaseCell subclass (e.g. LRC_Cell, LSTM_Cell)
        inter_neurons:    number of inter neurons
        command_neurons:  number of command neurons
        motor_neurons:    number of motor neurons (= model output size)
        seed:             random seed for NCP wiring generation (default 42)
        **cell_kwargs:    forwarded to the cell constructor
    """

    def __init__(self, cell_cls, inter_neurons, command_neurons, motor_neurons,
                 seed=42, **cell_kwargs):
        super().__init__(cell=None)
        self.cell_cls = cell_cls
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.units = inter_neurons + command_neurons + motor_neurons
        self.seed = seed
        self.cell_kwargs = cell_kwargs
        self._graphs = {}

    def graph(self, input_dim):
        """The ncps NCP graph for ``input_dim`` sensory features (cached)."""
        if input_dim not in self._graphs:
            self._graphs[int(input_dim)] = ncp_graph(
                self.inter_neurons, self.command_neurons, self.motor_neurons,
                self.seed, int(input_dim))
        return self._graphs[int(input_dim)]

    def adjacency_mask(self, input_dim):
        """Binary state -> state mask, shape (units, units)."""
        return (np.abs(self.graph(input_dim).adjacency_matrix) > 0
                ).astype(np.float32)

    def sensory_mask(self, input_dim):
        """Binary input -> state mask, shape (input_dim, units)."""
        return (np.abs(self.graph(input_dim).sensory_adjacency_matrix) > 0
                ).astype(np.float32)

    def make_cell(self):
        """One cell holding the whole circuit (layered for closed-form/discrete
        cells, see SEQUENTIAL_NCP_CELLS)."""
        if issubclass(self.cell_cls, SEQUENTIAL_NCP_CELLS):
            return NCPLayeredCell(self)
        return self.cell_cls(units=self.units,
                             sparsity_mask=self.adjacency_mask,
                             sensory_mask=self.sensory_mask,
                             **self.cell_kwargs)

    def motor_slice(self):
        """Layer selecting the motor neurons from the hidden state."""
        motor = self.motor_neurons
        return tf.keras.layers.Lambda(lambda h: h[..., :motor],
                                      name='motor_neurons')

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(self.make_cell(), return_sequences=True),
            self.motor_slice(),
        ])


class NCPLayeredCell(BaseCell):
    """NCP as one RNN cell with a sequential layer pass (ncps WiredCfCCell).

    Within ONE call: inter <- input, command <- new inter + old command,
    motor <- new command. Each layer is a masked sub-cell of ``cell_cls``:
    its input mask is the NCP adjacency from the previous layer (the sensory
    adjacency for inter), its recurrent mask the intra-layer adjacency, which
    is empty for inter and motor and the command <-> command synapses for
    command. Every layer receives the same ``elapsed_time``. Unlike ncps
    (``fully_recurrent=True`` default) the intra-layer recurrence is masked.
    CfC-family sub-cells default to ``backbone_layers=0`` (see CFC_NCP_CELLS).

    The hidden state stays one concatenated vector per state slot in ncps
    order ``[motor | command | inter]``, so the motor slice, state handling and
    ``effective_param_count`` see the same layout as the synchronous cell.
    """

    def __init__(self, wiring, **kwargs):
        super().__init__(wiring.units, **kwargs)
        self._wiring = wiring
        self._n_states = len(np.atleast_1d(
            wiring.cell_cls(units=1, **wiring.cell_kwargs).state_size))

    @property
    def state_size(self):
        return self.units if self._n_states == 1 else [self.units] * self._n_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        zeros = [tf.zeros([batch_size, self.units], dtype=dtype or tf.float32)
                 for _ in range(self._n_states)]
        return zeros[0] if self._n_states == 1 else zeros

    def build(self, input_shape):
        if isinstance(input_shape[0], (tuple, list, tf.TensorShape)):
            input_shape = input_shape[0]
        input_dim = int(input_shape[-1])
        g = self._wiring.graph(input_dim)
        adj = (np.abs(g.adjacency_matrix) > 0).astype(np.float32)
        sens = (np.abs(g.sensory_adjacency_matrix) > 0).astype(np.float32)
        w = self._wiring
        kw = ncp_cell_kwargs(w.cell_cls, w.cell_kwargs)
        self._slices, self._cells = [], []    # compute order: inter, command, motor
        prev = None
        for idx in (g._inter_neurons, g._command_neurons, g._motor_neurons):
            idx = np.asarray(idx)
            assert np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)), \
                'ncps neuron ids per layer are expected to be contiguous'
            in_mask = sens[:, idx] if prev is None else adj[np.ix_(prev, idx)]
            cell = w.cell_cls(units=len(idx),
                              sparsity_mask=adj[np.ix_(idx, idx)],
                              sensory_mask=in_mask, **kw)
            self._slices.append(slice(int(idx[0]), int(idx[-1]) + 1))
            self._cells.append(cell)
            prev = idx
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            x, elapsed_time = inputs
            feed = lambda v: (v, elapsed_time)
        else:
            x, feed = inputs, (lambda v: v)
        states = list(states) if isinstance(states, (list, tuple)) else [states]
        outs, new = [], []
        for sl, cell in zip(self._slices, self._cells):
            out, ns = cell(feed(x), [s[:, sl] for s in states])
            outs.append(out)
            new.append(list(ns) if isinstance(ns, (list, tuple)) else [ns])
            x = out
        # compute order is inter, command, motor; ncps order is the reverse.
        output = tf.concat(outs[::-1], axis=-1)
        new_states = [tf.concat([n[i] for n in new[::-1]], axis=-1)
                      for i in range(self._n_states)]
        return output, new_states


class NCPStackedWiring(BaseWiring):
    """DEPRECATED legacy stacked-layer approximation; NOT an NCP.

    Three stacked RNN layers (inter -> command -> motor), each internally
    fully recurrent, with a SparseLinear mask only BETWEEN the layers and
    ``elapsed_time`` reaching the first layer only. This measures depth plus
    inter-layer sparsity, not NCP wiring (see
    ``docs/ncp-wiring-fix-2026-09-17.md``). Retained only to reproduce
    pre-2026-09-17 runs and for the golden equivalence tests; registry key
    ``'ncp_stacked'``. New work uses :class:`NCPWiring`.

    Note: The ncps.wirings.NCP class uses neuron ordering [motor | command | inter]
    internally, so neuron index lists are extracted via w._inter_neurons,
    w._command_neurons, w._motor_neurons before computing masks.

    Args:
        cell_cls:         BaseCell subclass (e.g. LRC_Cell, LSTM_Cell)
        inter_neurons:    number of inter neurons
        command_neurons:  number of command neurons
        motor_neurons:    number of motor neurons (= model output size)
        seed:             random seed for NCP wiring generation (default 42)
        **cell_kwargs:    forwarded to each cell constructor
    """

    def __init__(self, cell_cls, inter_neurons, command_neurons, motor_neurons,
                 seed=42, **cell_kwargs):
        super().__init__(cell=None)
        self.cell_cls = cell_cls
        self.inter_neurons = inter_neurons
        self.command_neurons = command_neurons
        self.motor_neurons = motor_neurons
        self.seed = seed
        self.cell_kwargs = cell_kwargs

        # Legacy behaviour: the graph is built with input_shape=0, i.e. without
        # sensory synapses, because only the internal adjacency was used.
        wiring = ncp_graph(inter_neurons, command_neurons, motor_neurons,
                           seed, 0)
        A = wiring.adjacency_matrix

        inter_idx = wiring._inter_neurons    # list of inter neuron indices
        cmd_idx = wiring._command_neurons    # list of command neuron indices
        mot_idx = wiring._motor_neurons      # list of motor neuron indices

        # Extract inter->command and command->motor connectivity masks.
        # A[src, dst] != 0 means a connection from src to dst exists.
        self._inter_to_command = (
            A[np.ix_(inter_idx, cmd_idx)] != 0
        ).astype(np.float32)   # shape: (inter_neurons, command_neurons)

        self._command_to_motor = (
            A[np.ix_(cmd_idx, mot_idx)] != 0
        ).astype(np.float32)   # shape: (command_neurons, motor_neurons)

        assert self._inter_to_command.any(), (
            "NCPStackedWiring: inter→command mask is all-zeros (disconnected "
            "layer). Increase inter_neurons or command_neurons.")
        assert self._command_to_motor.any(), (
            "NCPStackedWiring: command→motor mask is all-zeros (disconnected "
            "layer). Increase command_neurons or motor_neurons.")

    def build_model(self) -> tf.keras.Sequential:
        return tf.keras.Sequential([
            tf.keras.layers.RNN(
                self.cell_cls(units=self.inter_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
            SparseLinear(self.command_neurons, self._inter_to_command),
            tf.keras.layers.RNN(
                self.cell_cls(units=self.command_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
            SparseLinear(self.motor_neurons, self._command_to_motor),
            tf.keras.layers.RNN(
                self.cell_cls(units=self.motor_neurons, **self.cell_kwargs),
                return_sequences=True,
            ),
        ])


def effective_param_count(model) -> int:
    """Trainable parameters minus the entries switched off by NCP masks.

    Masked-off weights exist as variables but are multiplied by 0 on every
    forward pass, so they neither influence the output nor receive gradient.
    Counting them would inflate an NCP model against a dense one, so the
    parameter-budget matching uses this number.

    The model must have run at least one forward pass (masks are applied
    inside ``call``); for unmasked models this is plain ``count_params()``.
    """
    total = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
    masked = {}                       # variable ref -> mask (deduplicated)
    seen = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, BaseCell):
            if obj.is_masked and not obj._masked_vars:
                raise ValueError(
                    f"{type(obj).__name__} is masked but no masked weights were "
                    "recorded -- run one forward pass before counting params.")
            masked.update(obj._masked_vars)
        if isinstance(getattr(obj, 'W', None), tf.Variable) and hasattr(obj, 'mask'):
            # SparseLinear / SignedSparseLinear (ncp_stacked, cncp edges).
            masked[obj.W.ref()] = obj.mask
        if isinstance(obj, tf.keras.layers.Layer):
            stack.extend(obj._flatten_layers(include_self=False, recursive=False))
            stack.extend(v for v in vars(obj).values()
                         if isinstance(v, tf.keras.layers.Layer))
    off = sum(np.count_nonzero(np.asarray(mask) == 0.0)
              for ref, mask in masked.items() if ref.deref().trainable)
    return total - int(off)


def param_counts(model) -> dict:
    """Both parameter counts every result file reports.

    ``params_effective`` (masked-off weights excluded) is the primary number
    for any parameter-matched comparison; ``params_raw`` is ``count_params()``.
    """
    return {'params_effective': int(effective_param_count(model)),
            'params_raw': int(model.count_params())}


def match_param_budget(count_fn, target, sizes):
    """Return ``(size, params)`` whose ``count_fn(size)`` is closest to ``target``.

    ``count_fn`` maps a width knob to an EFFECTIVE parameter count
    (``effective_param_count``) and must be non-decreasing in ``size``; the scan
    stops at the first size above ``target``, which is then provably the last
    candidate that can be closer. Sizes that fail to build (too small for the
    wiring) are skipped.
    """
    best = None
    for size in sizes:
        try:
            n = int(count_fn(size))
        except Exception:                       # size too small for this arm
            continue
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (size, n)
        if n > target:
            break
    if best is None:
        raise ValueError(f"no size in {sizes!r} builds a model")
    return best
