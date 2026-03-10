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
