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
    from src.models import make_dense_model
    model = make_dense_model('ctrnn', units=4)
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
    from src.models import make_dense_model
    model = make_dense_model('lstm', units=4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)


# --- LTC_Cell ---

def test_ltc_cell_is_subclass_of_basecell():
    from src.neurons import LTC_Cell
    assert issubclass(LTC_Cell, BaseCell)


def test_ltc_cell_state_size():
    from src.neurons import LTC_Cell
    cell = LTC_Cell(units=32)
    assert cell.state_size == 32


def test_ltc_forward_pass_shape():
    from src.neurons import LTC_Cell
    cell = LTC_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, new_states = cell(x, state)
    assert output.shape == (3, 8)
    assert new_states[0].shape == (3, 8)


def test_ltc_irregular_sampling():
    from src.neurons import LTC_Cell
    cell = LTC_Cell(units=8)
    x = tf.zeros([3, 5])
    state = [tf.zeros([3, 8])]
    output, _ = cell((x, 0.5), state)
    assert output.shape == (3, 8)


def test_ltc_state_stays_finite():
    """Fused solver must not blow up over many steps (stiffness guard)."""
    from src.neurons import LTC_Cell
    cell = LTC_Cell(units=8)
    x = tf.random.normal([3, 5])
    state = [tf.zeros([3, 8])]
    for _ in range(50):
        _, state = cell(x, state)
    assert tf.reduce_all(tf.math.is_finite(state[0])).numpy()


def test_ltc_make_model_and_gradient_flow():
    from src.models import make_dense_model
    model = make_dense_model('ltc', units=4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)


# --- GRU_Cell ---

def test_gru_cell_is_subclass_of_basecell():
    from src.neurons import GRU_Cell
    assert issubclass(GRU_Cell, BaseCell)


def test_gru_state_size():
    from src.neurons import GRU_Cell
    cell = GRU_Cell(units=8)
    assert cell.state_size == 8


def test_gru_forward_pass_shape():
    from src.neurons import GRU_Cell
    cell = GRU_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8])]
    output, new_states = cell(x, states)
    assert output.shape == (3, 8)
    assert new_states[0].shape == (3, 8)


def test_gru_irregular_sampling_ignored():
    """GRU discards elapsed_time (discrete cell)."""
    from src.neurons import GRU_Cell
    cell = GRU_Cell(units=8)
    x = tf.zeros([3, 5])
    states = [tf.zeros([3, 8])]
    output_reg, _ = cell(x, states)
    output_irr, _ = cell((x, 0.5), states)
    assert tf.reduce_all(output_reg == output_irr).numpy()


def test_gru_make_model_and_gradient_flow():
    from src.models import make_dense_model
    model = make_dense_model('gru', units=4)
    assert isinstance(model, tf.keras.Sequential)
    x = tf.zeros([2, 5, 3])
    assert model(x).shape == (2, 5, 4)
    model(x)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(model(x))
    grads = tape.gradient(loss, model.trainable_variables)
    assert len(model.trainable_variables) > 0
    assert any(g is not None and tf.reduce_any(g != 0).numpy() for g in grads)


# --- CfC_Cell ---

def test_cfc_cell_is_subclass_of_basecell():
    from src.neurons import CfC_Cell
    assert issubclass(CfC_Cell, BaseCell)


def test_cfc_cell_state_size():
    from src.neurons import CfC_Cell
    cell = CfC_Cell(units=32)
    assert cell.state_size == 32


def test_cfc_forward_pass_shape():
    from src.neurons import CfC_Cell
    units, batch, input_dim = 4, 2, 3
    cell = CfC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert new_state[0].shape == (batch, units)


def test_cfc_irregular_sampling():
    """CfC uses elapsed_time: different dt -> different next state."""
    from src.neurons import CfC_Cell
    tf.random.set_seed(0)
    cell = CfC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert not tf.reduce_all(tf.abs(s1[0] - s2[0]) < 1e-7)


def test_cfc_state_stays_finite():
    """Closed-form update is saturated -> no blow-up over many steps."""
    from src.neurons import CfC_Cell
    tf.random.set_seed(0)
    cell = CfC_Cell(units=4)
    x = tf.random.normal([2, 3]) * 10.0
    state = [tf.zeros([2, 4])]
    for _ in range(100):
        _, state = cell(x, state)
    assert bool(tf.reduce_all(tf.math.is_finite(state[0])))


def test_cfc_make_model_and_gradient_flow():
    from src.models import make_dense_model
    tf.random.set_seed(0)
    model = make_dense_model('cfc', units=8, output_neurons=2)
    x = tf.random.normal((2, 10, 3))
    with tf.GradientTape() as tape:
        y = model(x)
        loss = tf.reduce_mean(tf.square(y))
    grads = tape.gradient(loss, model.trainable_variables)
    assert y.shape == (2, 10, 2)
    assert all(g is not None for g in grads)


# --- MixedMemoryCell (MM_LTC, MM_LRC) ---

def test_mm_cells_are_subclasses_of_basecell():
    from src.neurons import MM_LTC_Cell, MM_LRC_Cell, MixedMemoryCell
    assert issubclass(MixedMemoryCell, BaseCell)
    assert issubclass(MM_LTC_Cell, MixedMemoryCell)
    assert issubclass(MM_LRC_Cell, MixedMemoryCell)


def test_mm_state_size_is_h_and_c():
    from src.neurons import MM_LTC_Cell
    cell = MM_LTC_Cell(units=8)
    assert cell.state_size == [8, 8]


def test_mm_ltc_forward_pass_shape():
    from src.neurons import MM_LTC_Cell
    units, batch, input_dim = 4, 2, 3
    cell = MM_LTC_Cell(units=units)
    inputs = tf.zeros([batch, input_dim])
    state = [tf.zeros([batch, units]), tf.zeros([batch, units])]
    output, new_state = cell(inputs, state)
    assert output.shape == (batch, units)
    assert len(new_state) == 2
    assert new_state[0].shape == (batch, units)
    assert new_state[1].shape == (batch, units)


def test_mm_lrc_forwards_elastance_kwarg():
    from src.neurons import MM_LRC_Cell
    cell = MM_LRC_Cell(units=4, elastance_type='asymmetric')
    inputs = tf.zeros([2, 3])
    state = [tf.zeros([2, 4]), tf.zeros([2, 4])]
    output, _ = cell(inputs, state)
    assert cell._inner._elastance_type == 'asymmetric'
    assert output.shape == (2, 4)


def test_mm_ltc_irregular_sampling():
    """elapsed_time reaches the inner ODE cell: different dt -> different state."""
    from src.neurons import MM_LTC_Cell
    tf.random.set_seed(0)
    cell = MM_LTC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4]), tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert not tf.reduce_all(tf.abs(s1[0] - s2[0]) < 1e-7)


def test_mm_memory_path_isolated_from_ode():
    """The c path is pure LSTM gating: same x and (h, c) but different dt
    must yield the *same* new c (only h goes through the ODE)."""
    from src.neurons import MM_LTC_Cell
    tf.random.set_seed(0)
    cell = MM_LTC_Cell(units=4)
    x = tf.random.normal([2, 3])
    state = [tf.random.normal([2, 4]), tf.random.normal([2, 4])]
    _, s1 = cell((x, 1.0), state)
    _, s2 = cell((x, 0.1), state)
    assert bool(tf.reduce_all(tf.abs(s1[1] - s2[1]) < 1e-7))


def test_mm_make_models_and_gradient_flow():
    from src.models import make_dense_model
    for key in ('mm_ltc', 'mm_lrc'):
        tf.random.set_seed(0)
        model = make_dense_model(key, units=8, output_neurons=2)
        x = tf.random.normal((2, 10, 3))
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        assert y.shape == (2, 10, 2)
        assert all(g is not None for g in grads), key
