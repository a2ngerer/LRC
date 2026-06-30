import tensorflow as tf
from src.neurons import (LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell,
                         LTC_Cell, GRU_Cell, CfC_Cell, CfC_LRC_Cell,
                         MM_LTC_Cell, MM_LRC_Cell, CfC_MM_LRC_Cell,
                         CfC_MM_LTC_Cell)
from src.wirings import NCPWiring

_CELL_REGISTRY = {
    "lrc":    LRC_Cell,
    "lrc_ar": LRC_AR_Cell,
    "ctrnn": CTRNN_Cell,
    "lstm": LSTM_Cell,
    "ltc": LTC_Cell,
    "gru": GRU_Cell,
    "cfc": CfC_Cell,
    # cfc_lrc: closed-form LRC (CfC + liquid-elastance gate). cfc_pm: plain CfC with
    # a wider backbone -- the parameter-matched capacity control for the v3.1
    # cfc_lrc ablation (backbone width set via CELL_KWARGS in run_benchmark.py).
    # See docs/superpowers/specs/2026-06-19-cfc-lrc-v3_1-design.md
    "cfc_lrc": CfC_LRC_Cell,
    "cfc_pm": CfC_Cell,
    "mm_ltc": MM_LTC_Cell,
    "mm_lrc": MM_LRC_Cell,
    # cfc_mm_lrc: mixed-memory wrapper around the closed-form CfC_LRC inner cell.
    # Completes the v3.2 2x2 {numerical, closed-form} x {plain, mixed-memory}.
    "cfc_mm_lrc": CfC_MM_LRC_Cell,
    # cfc_mm_ltc: mixed-memory wrapper around plain CfC (= closed-form LTC). The
    # LTC-family partner of cfc_mm_lrc; completes the cross-family 2x2 tested in
    # benchmark v4 (does the LRC architecture-fix story generalize to LTC?).
    "cfc_mm_ltc": CfC_MM_LTC_Cell,
    # lrc_pm: plain numerical LRC widened (units / NCP set via CELL_UNITS /
    # CELL_NCP in run_benchmark.py) to the largest fixed cell's parameter count --
    # the v3.3 capacity control isolating mechanism vs. pure capacity.
    "lrc_pm": LRC_Cell,
    # eps-ablation: the 8 elastance conditions are all plain LRC_Cell, differing
    # only in constructor kwargs (elastance_type / ode_solver / freeze_elastance /
    # pm_pad / pm_pad_extra), supplied via CELL_KWARGS in run_benchmark.py.
    "lrc_interp": LRC_Cell,        # A: conductance-only tau baseline
    "lrc_asym": LRC_Cell,          # B: asymmetric multiplicative gate
    "lrc_sym": LRC_Cell,           # C: symmetric two-sided bump
    "lrc_frozen": LRC_Cell,        # D: frozen-structure control
    "lrc_pmctrl": LRC_Cell,        # E: same-budget additive-residual control for B
    "lrc_pmctrl_c": LRC_Cell,      # E_C: same-budget control for C
    "lrc_asym_hybrid": LRC_Cell,   # F: asymmetric + semi-implicit solver
    "lrc_interp_hybrid": LRC_Cell, # G: tau + semi-implicit solver
}


def make_dense_model(neuron_type, units, num_layers=1, output_neurons=None, **cell_kwargs):
    """Build a stacked Dense-wired RNN model.

    Args:
        neuron_type:    str key ("lrc", "lrc_ar", "ctrnn", "lstm") or BaseCell subclass
        units:          neurons per RNN layer (all layers the same size)
        num_layers:     number of stacked RNN layers (default 1)
        output_neurons: if given, appends a Dense(output_neurons) projection layer
        **cell_kwargs:  forwarded to each cell constructor

    Returns:
        tf.keras.Sequential, always return_sequences=True on every RNN layer
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    layers = []
    for _ in range(num_layers):
        cell = cell_cls(units=units, **cell_kwargs)
        layers.append(tf.keras.layers.RNN(cell, return_sequences=True))
    if output_neurons is not None:
        layers.append(tf.keras.layers.Dense(output_neurons))
    return tf.keras.Sequential(layers)


def make_ncp_model(neuron_type, inter_neurons, command_neurons, motor_neurons,
                   seed=42, **cell_kwargs):
    """Build a three-layer NCP-wired RNN model.

    Layers: inter -> (sparse) -> command -> (sparse) -> motor
    Output shape: (batch, timesteps, motor_neurons)

    Args:
        neuron_type:      str key ("lrc", "lrc_ar", "ctrnn", "lstm") or BaseCell subclass
        inter_neurons:    neurons in inter layer
        command_neurons:  neurons in command layer
        motor_neurons:    neurons in motor layer (= output size)
        seed:             NCP wiring seed (default 42)
        **cell_kwargs:    forwarded to each cell constructor

    Returns:
        tf.keras.Sequential
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    wiring = NCPWiring(cell_cls, inter_neurons, command_neurons, motor_neurons,
                       seed=seed, **cell_kwargs)
    return wiring.build_model()
