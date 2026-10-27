"""Per-fit weight checkpoints of the sequence runners (LV, PA, committee).

Every fit writes ``<stem>.weights.h5`` next to its result ``<stem>.json``; the
JSON carries ``weights_path`` (relative to the JSON's folder, so a fetched run
folder stays loadable) and ``config.effective``:

    task         'lotka_volterra' | 'person_activity' | 'committee' (BUILDERS key)
    build        keyword arguments of that task builder, as the runner called it
    cell_kwargs  forwarded to the cell constructors
    ...          context the builder does not take (param budget, effective
                 backbone depth, LV system and data size, committee views)

so post-hoc analyses rebuild the trained model without a refit.

Records written before the NCP self-recurrence flag (#23) lack
build.self_recurrent: their NCP had none (self_recurrent=False), and as the
closed-form backbone default changed in between, their CfC depth is taken
from config.effective.backbone_layers (the depth the fit actually had).
"""
import hashlib
import json
import os

import tensorflow as tf

from src.tasks.committee.model import build_committee_model
from src.tasks.lotka_volterra.model import build_lotka_volterra_model
from src.tasks.person_activity.model import (backbone_override,
                                             build_person_activity_model)

BUILDERS = {"lotka_volterra": build_lotka_volterra_model,
            "person_activity": build_person_activity_model,
            "committee": build_committee_model}


def _layers_only(model):
    # An uncompiled view on the same layers: the file then holds the layer
    # weights only, no optimizer slots or metric state (issue #18 decision).
    return tf.keras.Model(model.inputs, model.outputs)


def save_weights(model, json_path, enabled=True):
    """Write ``<stem>.weights.h5`` next to ``json_path``; return its file name
    (the JSON's ``weights_path``), or None when disabled (--no-save-weights)."""
    if not enabled:
        return None
    path = json_path[:-len(".json")] + ".weights.h5"
    _layers_only(model).save_weights(path)
    return os.path.basename(path)


def rebuild_from_result(json_path) -> tf.keras.Model:
    """Rebuild the compiled model of a result JSON and load its trained weights."""
    with open(json_path) as f:
        record = json.load(f)
    eff = record["config"]["effective"]
    build, cell_kwargs = dict(eff["build"]), dict(eff["cell_kwargs"])
    if (eff["task"] in ("lotka_volterra", "person_activity")
            and "self_recurrent" not in build):                 # pre-#23 fit
        build["self_recurrent"] = False
        if "backbone_layers" not in cell_kwargs:
            cell_kwargs.update(backbone_override(
                build["cell"], eff.get("backbone_layers")))
    model = BUILDERS[eff["task"]](**build, **cell_kwargs)
    _layers_only(model).load_weights(
        os.path.join(os.path.dirname(json_path), record["weights_path"]))
    return model


def config_hash(effective) -> str:
    """sha256 of ``config.effective`` as canonical JSON (sorted keys, no
    whitespace): post-hoc outputs store it to name the exact config they were
    computed from."""
    return hashlib.sha256(json.dumps(effective, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()
