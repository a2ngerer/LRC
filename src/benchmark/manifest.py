"""Provenance manifest for a benchmark campaign run.

``write_manifest`` records the git SHA, SLURM ids, environment (python + TF
version, uv.lock hash) and a snapshot of the config into ``<outdir>/manifest.json``
so any result dir can be traced back to the exact code/config that produced it.

Every probe is wrapped: off-cluster the SLURM env vars are absent (-> None),
``git`` may be missing or the dir may not be a repo (-> None), and TF / uv.lock
may be unavailable. Nothing here raises; a manifest is best-effort metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    # src/benchmark/manifest.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(_repo_root()),
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        pass
    return None


def _slurm_ids() -> dict:
    return {
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
        'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
    }


def _uv_lock_sha256() -> str | None:
    try:
        lock = _repo_root() / 'uv.lock'
        if lock.exists():
            return hashlib.sha256(lock.read_bytes()).hexdigest()
    except Exception:
        pass
    return None


def _tensorflow_version() -> str | None:
    try:
        import tensorflow as tf
        return tf.__version__
    except Exception:
        return None


def _env_info() -> dict:
    return {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'tensorflow': _tensorflow_version(),
        'uv_lock_sha256': _uv_lock_sha256(),
    }


def build_manifest(config_path, config_dict: dict, spec_count: int) -> dict:
    """Assemble (but do not write) the manifest dict."""
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'config_path': str(config_path),
        'spec_count': int(spec_count),
        'git_sha': _git_sha(),
        'slurm': _slurm_ids(),
        'env': _env_info(),
        'config': config_dict,
    }


def write_manifest(outdir, config_path, config_dict: dict, spec_count: int) -> str:
    """Write ``<outdir>/manifest.json`` and return its path."""
    os.makedirs(outdir, exist_ok=True)
    manifest = build_manifest(config_path, config_dict, spec_count)
    path = os.path.join(outdir, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path
