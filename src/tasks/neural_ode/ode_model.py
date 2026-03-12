import tensorflow as tf
from tensorflow.keras.layers import Dense
from src.models import make_dense_model


class ODEFuncModel(tf.keras.Model):
    def __init__(self, neuron_type, units, features, **cell_kwargs):
        """ODE function model: Dense(units) -> RNN core -> Dense(features).

        Wraps make_dense_model as the RNN core and adds input/output projections
        so the model maps state (batch, 1, features) -> derivative (batch, 1, features).

        Args:
            neuron_type:  passed to make_dense_model ("lrc_ar" for ODE tasks)
            units:        RNN cell width
            features:     input/output feature dimension (2 for all 6 ODE systems)
            **cell_kwargs: forwarded to make_dense_model -> cell constructor
        """
        super().__init__()
        self.dense_in = Dense(units)
        self.rnn = make_dense_model(neuron_type, units=units, **cell_kwargs)
        self.dense_out = Dense(features)

    def call(self, t, state):
        # t is unused (autonomous ODE) but kept for euler_odeint compatibility
        # state: (batch, 1, features)
        h = self.dense_in(state)    # (batch, 1, units)
        dh = self.rnn(h)            # (batch, 1, units)
        dy = self.dense_out(dh)     # (batch, 1, features)
        return dy
