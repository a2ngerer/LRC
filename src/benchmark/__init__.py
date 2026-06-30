"""Config-driven benchmark engine.

Replaces the scattered ``build_specs_<profile>()`` matrix generators in
``experiments/run_benchmark.py`` with declarative YAML campaign configs. The
training run path (``run_one`` / ``build_model`` / ``train`` / ``result_filename``)
is still imported from the legacy module -- only spec GENERATION moves here.

  load_config(path) -> CampaignConfig      # parse + validate a campaign YAML
  expand(config)    -> list[dict]          # byte-identical legacy spec list
  write_manifest(...)                      # provenance sidecar for a run
"""
from .config import (
    CampaignConfig, ConfigError, Block, DropRule, ExtraAxis, load_config,
)
from .expand import expand
from .manifest import build_manifest, write_manifest

__all__ = [
    'CampaignConfig', 'ConfigError', 'Block', 'DropRule', 'ExtraAxis',
    'load_config', 'expand', 'build_manifest', 'write_manifest',
]
