import abc


class BaseWiring(abc.ABC):
    """Abstract base for wiring strategies.

    A wiring defines how a cell is wrapped into a Keras model.
    Subclass this and implement build_model().

    Phase 2 will add NCPWiring here — no changes to make_model required.
    """

    def __init__(self, cell):
        self.cell = cell

    @abc.abstractmethod
    def build_model(self) -> "tf.keras.Sequential":
        """Wrap self.cell and return a tf.keras.Sequential."""
        pass
