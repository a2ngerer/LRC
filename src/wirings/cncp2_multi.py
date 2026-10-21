# cNCP v2 with K independent columns (registry key 'cncp2_multi', spec
# docs/superpowers/specs/2026-10-21-cncp-v2-abstraction.md, B.7 and Part C V4).
#
# K copies of the four-node column (src/wirings/cncp2.py) with their OWN
# weights and masks, an optional lateral S -> S coupling between the columns,
# and a readout over the column outputs d_k. Implementation style: a Python
# loop over K CorticalColumnV2Cell instances. The column's sub-cells are the
# pluggable BaseCell classes, which have no stacked (K, in, out) form, so
# stacking would mean re-implementing every sub-cell family; the loop costs
# about K times the single-column ops per step (B.7 "with a Python loop").
# The lateral edge is ONE SparseLinear (K u -> K u) with a block mask (zero
# diagonal blocks, Bernoulli(d_lat) off-diagonal blocks), i.e. the K (K - 1)
# independent maps L_{c'c} of B.7 in one matmul; effective_param_count counts
# it through its mask like every other SparseLinear.

import warnings

import numpy as np
import tensorflow as tf

from .cncp import _make_mask
from .cncp2 import CorticalColumnV2Cell, NODE_ORDER_V2
from .ncp import SparseLinear

PARTITIONS = ('same', 'region')
COUPLINGS = ('none', 'lateral')
READOUTS = ('concat', 'mean', 'median')

_S_IDX = NODE_ORDER_V2.index('S')
_N_NODES = len(NODE_ORDER_V2)
# RNG stream of the lateral mask; columns use the seeds seed, [seed, 1], ...
_LATERAL_STREAM = 2 ** 20


def _column_seed(seed, k):
    """Mask seed of column k: column 0 keeps the single-column seed (K = 1 is
    the cncp2 column), the others get independent streams."""
    return seed if k == 0 else [int(seed), int(k)]


class MultiColumnV2Cell(tf.keras.layers.AbstractRNNCell):
    """K independent cNCP v2 columns with optional lateral coupling (spec B.7).

    Per-step input (x, dt). x is either (B, F) and routed by ``partition``, or
    (B, K, F) per-column views (the committee runner), where column k reads
    x[:, k]. State: K copies of [g, s, d, r], flat in column-major order.
    Output: [d_1 ... d_K] (B, K u); the readout over the columns is
    ColumnReadout, applied after the RNN.

    Args:
        columns:         K.
        cell_cls, units: sub-cell family and column width u (as cncp2).
        seed:            mask seed (column k: see _column_seed).
        partition:       'same': every column reads the full x. 'region': with
                         ``region_tags`` = T, x = [one-hot(tag, T), values]
                         and column k reads [m_k, m_k * values], m_k = x[:, k]
                         (Person Activity: T = 4 tags, values = xyz, F_in 4;
                         requires K == T). Without region_tags the feature
                         axis is split into K contiguous groups (warning).
        coupling:        'none' or 'lateral': S_c(t) additionally receives
                         sum_{c' != c} L_{c'c} tanh(s_{c'}(t-1)), L a
                         SparseLinear(u, u) of density ``lateral_density``.
                         No-op for K = 1.
        lateral_density: d_lat (default 0.25).
        region_tags:     number of leading one-hot tag features (see partition).
        (rest)           forwarded to every CorticalColumnV2Cell (column flags
                         and sub-cell kwargs).
    """

    def __init__(self, columns=2, cell_cls='cfc_lrc', units=8, seed=42,
                 partition='same', coupling='none', lateral_density=0.25,
                 region_tags=None, dt=1.0, **column_kwargs):
        super().__init__()
        if partition not in PARTITIONS:
            raise ValueError(f"partition must be one of {PARTITIONS}, "
                             f"got {partition!r}")
        if coupling not in COUPLINGS:
            raise ValueError(f"coupling must be one of {COUPLINGS}, "
                             f"got {coupling!r}")
        if int(columns) < 1:
            raise ValueError(f"columns must be >= 1, got {columns}")
        self.K = int(columns)
        self.units = int(units)
        self._seed = seed
        self._partition = partition
        self._lateral_on = coupling == 'lateral' and self.K > 1
        self._density = float(lateral_density)
        self._tags = region_tags
        self._dt = dt
        self.cols = [CorticalColumnV2Cell(
            cell_cls=cell_cls, units=units, seed=_column_seed(seed, k), dt=dt,
            extra_s_edges=int(self._lateral_on), **column_kwargs)
            for k in range(self.K)]

    @property
    def state_size(self):
        return list(self.cols[0].state_size) * self.K

    @property
    def output_size(self):
        return self.K * self.units

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [s for c in self.cols
                for s in c.get_initial_state(batch_size=batch_size,
                                             dtype=dtype)]

    def _routing(self, x_shape):
        """Per-column input widths and a function x -> list of K inputs."""
        K = self.K
        if len(x_shape) == 3:                           # committee views
            if self._partition != 'same':
                warnings.warn("per-column views (B, K, F) are routed as given; "
                              f"partition={self._partition!r} is ignored",
                              stacklevel=2)
            if int(x_shape[1]) != K:
                raise ValueError(f"views axis has {x_shape[1]} columns, "
                                 f"the cell has {K}")
            return [int(x_shape[-1])] * K, lambda x: [x[:, k] for k in range(K)]
        F = int(x_shape[-1])
        if self._partition == 'same':
            return [F] * K, lambda x: [x] * K
        if self._tags is not None:
            T = int(self._tags)
            if K != T or F <= T:
                raise ValueError(
                    f"partition='region' with {T} sensor tags needs columns == "
                    f"{T} and more than {T} features, got columns={K}, F={F}")

            def split(x):
                vals = x[:, T:]
                return [tf.concat([x[:, k:k + 1], x[:, k:k + 1] * vals], -1)
                        for k in range(K)]
            return [1 + F - T] * K, split
        if K > F:
            raise ValueError(f"partition='region' cannot split {F} features "
                             f"over {K} columns")
        groups = np.array_split(np.arange(F), K)       # sizes differ by <= 1
        warnings.warn(
            "partition='region': input has no sensor structure "
            f"(region_tags=None); splitting the {F} features into {K} "
            "contiguous groups of (near-)equal size.", stacklevel=2)
        bounds = [(int(g[0]), int(g[-1]) + 1) for g in groups]
        return ([b - a for a, b in bounds],
                lambda x: [x[:, a:b] for a, b in bounds])

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(
            input_shape[0], (tuple, list, tf.TensorShape)) else input_shape
        x_shape = tf.TensorShape(x_shape)
        dims, self._split = self._routing(x_shape)
        self.column_input_dims = dims
        for col, d_in in zip(self.cols, dims):
            col.build((tf.TensorShape([None, d_in]), tf.TensorShape([None, 1])))
        self.lateral = None
        if self._lateral_on:
            u, K = self.units, self.K
            rng = np.random.default_rng([int(self._seed), _LATERAL_STREAM])
            mask = np.zeros((K * u, K * u), dtype=np.float32)
            for src in range(K):                    # rows: source column c'
                for dst in range(K):                # cols: target column c
                    if src != dst:
                        mask[src * u:(src + 1) * u, dst * u:(dst + 1) * u] = (
                            _make_mask(rng, u, u, self._density))
            self.lateral = SparseLinear(K * u, mask, name='lateral')
            self.lateral.build((None, K * u))
            # R2: glorot of one u x u block (the (Ku, Ku) glorot limit is
            # sqrt(K) smaller), then 1/sqrt(n) with n the edges summed into S
            # (E_GS, E_DS if feedback, lateral as one edge).
            n = 1 + int(self.cols[0]._feedback) + 1
            self.lateral.W.assign(self.lateral.W * np.sqrt(K) / np.sqrt(n))
        self.built = True

    def call(self, inputs, states):
        if isinstance(inputs, (tuple, list)):
            x, dt = inputs
        else:
            x, dt = inputs, self._dt
        xs = self._split(x)
        per_col = [list(states[k * _N_NODES:(k + 1) * _N_NODES])
                   for k in range(self.K)]
        lat = None
        if self.lateral is not None:
            lat = self.lateral(tf.concat(
                [tf.tanh(st[_S_IDX]) for st in per_col], axis=-1))
        u = self.units
        outs, new_states = [], []
        for k, col in enumerate(self.cols):
            inp = ((xs[k], dt) if lat is None
                   else (xs[k], dt, lat[:, k * u:(k + 1) * u]))
            d, ns = col(inp, per_col[k])
            outs.append(d)
            new_states.extend(ns)
        return tf.concat(outs, axis=-1), new_states


class ColumnReadout(tf.keras.layers.Layer):
    """Readout over the K column outputs [d_1 ... d_K] (spec B.7).

    'concat': Dense(n_out) on the concatenation, (K u + 1) n_out parameters.
    'mean':   Dense(n_out) on the mean of the d_k, (u + 1) n_out.
    'median': one head Dense(n_out) per column (a stacked (K, u, n_out) kernel
              plus (K, n_out) bias, K (u + 1) n_out), then the coordinate-wise
              lower median over the K output vectors (index (K - 1) // 2).
    """

    def __init__(self, n_out, columns, readout='concat', **kwargs):
        super().__init__(**kwargs)
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, "
                             f"got {readout!r}")
        self.n_out, self.K, self.readout = int(n_out), int(columns), readout

    def build(self, input_shape):
        width = int(input_shape[-1])
        if width % self.K:
            raise ValueError(f"input width {width} is not a multiple of "
                             f"columns={self.K}")
        self.u = width // self.K
        if self.readout == 'median':
            lim = np.sqrt(6.0 / (self.u + self.n_out))   # glorot per column head
            self.kernel = self.add_weight(
                name='kernel', shape=(self.K, self.u, self.n_out),
                initializer=tf.keras.initializers.RandomUniform(-lim, lim))
            self.bias = self.add_weight(name='bias', shape=(self.K, self.n_out),
                                        initializer='zeros')
        else:
            self.head = tf.keras.layers.Dense(self.n_out, name='head')
            self.head.build(tf.TensorShape(
                [None, width if self.readout == 'concat' else self.u]))
        self.built = True

    def _per_column(self, h):
        shape = tf.concat([tf.shape(h)[:-1], [self.K, self.u]], axis=0)
        return tf.reshape(h, shape)                     # (..., K, u)

    def call(self, h):
        if self.readout == 'concat':
            return self.head(h)
        hk = self._per_column(h)
        if self.readout == 'mean':
            return self.head(tf.reduce_mean(hk, axis=-2))
        z = tf.einsum('...ku,kuo->...ko', hk, self.kernel) + self.bias
        return tf.sort(z, axis=-2)[..., (self.K - 1) // 2, :]

    def get_config(self):
        return {**super().get_config(), 'n_out': self.n_out,
                'columns': self.K, 'readout': self.readout}


def multi_column_head(inputs, n_out, head_name, columns=2, cell_cls='cfc_lrc',
                      units=8, seed=42, partition='same', coupling='none',
                      readout='concat', **kwargs):
    """RNN(MultiColumnV2Cell) on the (x, time) tuple plus ColumnReadout: the
    task builders' cncp2_multi branch, returning the head output."""
    cell = MultiColumnV2Cell(columns=columns, cell_cls=cell_cls, units=units,
                             seed=seed, partition=partition, coupling=coupling,
                             **kwargs)
    h = tf.keras.layers.RNN(cell, return_sequences=True)(inputs)
    return ColumnReadout(n_out, columns, readout, name=head_name)(h)
