"""Parameter groups for the RQ4 diagnostics (issue #30).

``group_parameters(model)`` assigns every trainable variable of a task model to
exactly one functional group:

    timescale  time-constant / gate parameters: what sets how fast the state
               moves (leak, capacitance, gating synapses, elastance, CfC time
               heads, GRU/LSTM gate kernels)
    recurrent  state -> state drive (recurrent kernels, reversal potentials,
               CfC ff heads)
    input      input -> state maps (sensory synapses, input affine)
    backbone   CfC-family backbone layers
    readout    the cell output affine of LTC/LRC and every layer outside the
               RNN layers (the task head)
    other      anything not in the table (cNCP edges, ablation pads, ...)

Keras drops the sub-layer scope from the variable names of the Dense heads that
the CfC cells build eagerly (all of them read ``.../cf_c__cell/kernel:0``), so
the lookup key is the name of the layer that owns the variable for Dense
sub-layers, and the variable's own ``add_weight`` name otherwise.
"""
import tensorflow as tf

# key (add_weight name, or owning Dense sub-layer name) -> group, per cell class
_TABLE = {
    # CTRNN_Cell
    "tau": "timescale", "W_h": "recurrent", "b": "recurrent", "W_x": "input",
    # LTC_Cell / LRC_Cell / LRC_AR_Cell (also the inner cell of mm_*): the
    # recurrent w/sigma/mu synapses set the conductance, i.e. the time constant
    # (LTC) or the forget gate (LRC); erev / h / vleak set the drive.
    "cm": "timescale", "gleak": "timescale", "w": "timescale",
    "sigma": "timescale", "mu": "timescale",
    "erev": "recurrent", "h": "recurrent", "vleak": "recurrent",
    "sensory_w": "input", "sensory_sigma": "input", "sensory_mu": "input",
    "sensory_erev": "input", "sensory_h": "input",
    "input_w": "input", "input_b": "input",
    "output_w": "readout", "output_b": "readout",
    # LRC elastance head (also CfC_LRC_Cell), symmetric-elastance shift
    "elastance_mapping": "timescale", "distr_shift": "timescale",
    # CfC_Cell / CfC_LRC_Cell heads (backbone_<i> -> backbone, see _group)
    "time_a": "timescale", "time_b": "timescale",
    "ff1": "recurrent", "ff2": "recurrent",
}
GROUPS = ("timescale", "recurrent", "input", "backbone", "readout", "other")


def _group(owner, var):
    # Keras GRU/LSTM fuse gates and candidate into one kernel each, so all
    # three variables count as gate kernels.
    if isinstance(owner, (tf.keras.layers.GRUCell, tf.keras.layers.LSTMCell)):
        return "timescale"
    if isinstance(owner, tf.keras.layers.Dense):
        key = owner.name
    else:
        key = var.name.split("/")[-1].split(":")[0]
    if key.startswith("backbone_"):
        return "backbone"
    return _TABLE.get(key, "other")


def group_parameters(model) -> dict:
    """``{group: [variables]}`` over ``model.trainable_variables``; every
    variable lands in exactly one group, empty groups are omitted."""
    groups = {g: [] for g in GROUPS}
    seen = set()

    def put(group, var):
        if var.trainable and var.ref() not in seen:
            seen.add(var.ref())
            groups[group].append(var)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.RNN):
            for owner in layer.cell._flatten_layers(include_self=True):
                for var in owner._trainable_weights:
                    put(_group(owner, var), var)
        else:
            for var in layer.trainable_variables:
                put("readout", var)
    for var in model.trainable_variables:      # not reached above
        put("other", var)
    return {g: vs for g, vs in groups.items() if vs}
