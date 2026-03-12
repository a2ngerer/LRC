import tensorflow as tf
from src.neurons import LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell
from src.wirings import NCPWiring

_CELL_REGISTRY = {
    "lrc":    LRC_Cell,
    "lrc_ar": LRC_AR_Cell,
    "ctrnn": CTRNN_Cell,
    "lstm": LSTM_Cell,
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
