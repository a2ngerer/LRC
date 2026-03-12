import tensorflow as tf
from src.neurons import LRC_Cell, LRC_AR_Cell, CTRNN_Cell, LSTM_Cell
from src.wirings import DenseWiring

_CELL_REGISTRY = {
    'lrc':    LRC_Cell,
    'lrc_ar': LRC_AR_Cell,
    'ctrnn': CTRNN_Cell,
    'lstm': LSTM_Cell,
}

_WIRING_REGISTRY = {
    'dense': DenseWiring,
}


def make_model(neuron_type, wiring_type, units, **kwargs):
    """Build a bare RNN Sequential model for the given neuron + wiring combination.

    Args:
        neuron_type: str key ('lrc', 'lrc_ar', 'ctrnn', 'lstm') or a BaseCell subclass
        wiring_type: str key ('dense') or a BaseWiring subclass
        units:       number of RNN units (forwarded to cell constructor)
        **kwargs:    additional keyword args forwarded to the cell constructor
                     (e.g. ode_unfolds=5, elastance_type='symmetric')

    Returns:
        tf.keras.Sequential wrapping the cell in the specified wiring.
        Always return_sequences=True — all timesteps are returned.

    Raises:
        KeyError: if neuron_type or wiring_type is an unknown string key.
    """
    cell_cls = _CELL_REGISTRY[neuron_type] if isinstance(neuron_type, str) else neuron_type
    cell = cell_cls(units=units, **kwargs)

    wiring_cls = _WIRING_REGISTRY[wiring_type] if isinstance(wiring_type, str) else wiring_type
    wiring = wiring_cls(cell)

    return wiring.build_model()
