import pytest
import tensorflow as tf
from src.neurons import BaseCell, LRC_Cell, LRC_AR_Cell


# --- BaseCell ---

def test_basecell_is_abstract():
    """BaseCell cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCell(units=4)


def test_basecell_is_abstract_rnn_cell():
    assert issubclass(BaseCell, tf.keras.layers.AbstractRNNCell)


# --- LRC_Cell ---

def test_lrc_cell_is_subclass_of_basecell():
    assert issubclass(LRC_Cell, BaseCell)


def test_lrc_cell_state_size():
    cell = LRC_Cell(units=32)
    assert cell.state_size == 32


def test_lrc_cell_output_size():
    cell = LRC_Cell(units=16)
    assert cell.output_size == 16


def test_lrc_cell_units_stored():
    cell = LRC_Cell(units=8)
    assert cell.units == 8


def test_lrc_cell_forward_pass():
    """Full forward pass through one time step."""
    units = 4
    batch = 2
    input_dim = 3
    cell = LRC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)


# --- LRC_AR_Cell ---

def test_lrc_ar_cell_is_subclass_of_basecell():
    assert issubclass(LRC_AR_Cell, BaseCell)


def test_lrc_ar_cell_state_size():
    cell = LRC_AR_Cell(units=32)
    assert cell.state_size == 32


def test_lrc_ar_cell_forward_pass():
    units = 4
    batch = 2
    input_dim = 4  # AR cell: input_dim == units
    cell = LRC_AR_Cell(units=units, output_mapping=None, input_mapping=None)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)


# --- CTRNN_Cell ---

def test_ctrnn_cell_is_subclass_of_basecell():
    from src.neurons import CTRNN_Cell
    from src.neurons.base_cell import BaseCell
    assert issubclass(CTRNN_Cell, BaseCell)


def test_ctrnn_forward_pass_shape():
    from src.neurons import CTRNN_Cell
    cell = CTRNN_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, new_states = cell(x, state)
    assert output.shape == (3, 8)
    assert new_states[0].shape == (3, 8)


def test_ctrnn_irregular_sampling():
    from src.neurons import CTRNN_Cell
    cell = CTRNN_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, _ = cell((x, 0.5), state)
    assert output.shape == (3, 8)


def test_ctrnn_make_model_and_gradient_flow():
    from src.models import make_model
    model = make_model('ctrnn', 'dense', 4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)  # ensure weights built
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)


# --- LSTM_Cell ---

def test_lstm_cell_is_subclass_of_basecell():
    from src.neurons import LSTM_Cell
    from src.neurons.base_cell import BaseCell
    assert issubclass(LSTM_Cell, BaseCell)


def test_lstm_state_size():
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    assert cell.state_size == [8, 8]


def test_lstm_forward_pass_shape():
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8]), tf.zeros([3, 8])]
    output, new_states = cell(x, states)
    assert output.shape == (3, 8)
    assert len(new_states) == 2
    assert new_states[0].shape == (3, 8)
    assert new_states[1].shape == (3, 8)


def test_lstm_irregular_sampling_ignored():
    """LSTM discards elapsed_time (discrete cell)."""
    from src.neurons import LSTM_Cell
    cell = LSTM_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8]), tf.zeros([3, 8])]
    output_reg, _ = cell(x, states)
    output_irr, _ = cell((x, 0.5), states)
    assert tf.reduce_all(output_reg == output_irr).numpy()


def test_lstm_make_model_and_gradient_flow():
    from src.models import make_model
    model = make_model('lstm', 'dense', 4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)
