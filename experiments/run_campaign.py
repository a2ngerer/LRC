# experiments/run_campaign.py
"""Config-driven campaign runner -- the thin CLI over the benchmark engine.

Same operational contract as ``run_benchmark.py`` (``--list`` / ``--count`` /
``--index`` / ``--all``; ``--index`` is the SLURM array contract), but the run
matrix comes from a declarative campaign config instead of a hard-coded
``build_specs_<profile>()``:

    # number of specs (for --array=0-$((N-1)))
    uv run python experiments/run_campaign.py --config v5 --count

    # list every spec with its index
    uv run python experiments/run_campaign.py --config v5 --list

    # run spec i (SLURM array task) -- writes results + a provenance manifest
    uv run python experiments/run_campaign.py --config v5 --index $SLURM_ARRAY_TASK_ID

``--config`` accepts a profile name (resolved to ``configs/campaigns/<name>.yaml``)
or a path to a YAML file. Spec GENERATION is the engine's; TRAINING is reused
verbatim from ``run_benchmark`` (``run_one`` / ``save_result``), so a campaign run
is bit-identical to the legacy runner for the same spec.
"""
import argparse
import sys
from pathlib import Path

# Make both `experiments.*` and `src.*` importable regardless of how the script
# is launched (direct path, SLURM, cluster venv).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark import load_config, expand, write_manifest      # noqa: E402
from src.benchmark.config import ConfigError                       # noqa: E402
# Reuse the legacy training path verbatim (minimal-invasion rule).
from experiments.run_benchmark import (                            # noqa: E402
    run_one, save_result, result_filename, DEFAULTS,
)

CONFIG_DIR = _REPO_ROOT / 'configs' / 'campaigns'


def _resolve_config(arg: str) -> Path:
    """Resolve --config to a YAML path (literal path or a profile name)."""
    p = Path(arg)
    if p.suffix in ('.yaml', '.yml') and p.exists():
        return p
    candidate = CONFIG_DIR / f'{arg}.yaml'
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    raise SystemExit(
        f'config not found: {arg!r} (looked for a file and for '
        f'{CONFIG_DIR / (arg + ".yaml")})')


def _apply_cli_filters(specs, args):
    """Post-construction matrix filters (mirror run_benchmark.py)."""
    if args.cells:
        keep = set(args.cells.split(','))
        specs = [s for s in specs if s['cell'] in keep]
    if args.wirings:
        keep = set(args.wirings.split(','))
        specs = [s for s in specs if s['wiring'] in keep]
    if args.systems:
        keep = set(args.systems.split(','))
        specs = [s for s in specs if s['system'] in keep]
    if args.seeds:
        keep = {int(s) for s in args.seeds.split(',') if s != ''}
        specs = [s for s in specs if s['seed'] in keep]
    return specs


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Config-driven benchmark campaign runner')
    p.add_argument('--config', required=True,
                   help='campaign profile name (configs/campaigns/<name>.yaml) '
                        'or a path to a YAML config')

    sel = p.add_argument_group('run selection')
    sel.add_argument('--list', action='store_true',
                     help='print all run specs and exit')
    sel.add_argument('--count', action='store_true',
                     help='print number of run specs and exit')
    sel.add_argument('--index', type=int, default=None,
                     help='run spec by index (SLURM array task)')
    sel.add_argument('--all', action='store_true',
                     help='run all specs sequentially')

    mat = p.add_argument_group('matrix filters (apply before indexing)')
    mat.add_argument('--cells', default=None, help='comma-separated cell subset')
    mat.add_argument('--wirings', default=None, help='comma-separated wiring subset')
    mat.add_argument('--systems', default=None, help='comma-separated system subset')
    mat.add_argument('--seeds', default=None, help='comma-separated seeds')

    tr = p.add_argument_group('training config')
    tr.add_argument('--iters', type=int, default=DEFAULTS['n_iters'])
    tr.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    tr.add_argument('--batch-time', type=int, default=DEFAULTS['batch_time'])
    tr.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    tr.add_argument('--loss', choices=['mse', 'mae'], default=DEFAULTS['loss'])
    tr.add_argument('--grad-log-every', type=int, default=DEFAULTS['grad_log_every'])
    tr.add_argument('--data-size', type=int, default=DEFAULTS['data_size'])
    tr.add_argument('--deterministic', action='store_true',
                    help='enable TF op determinism (bit-exact, slower)')
    tr.add_argument('--outdir', default=None,
                    help='override the config outdir')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = _resolve_config(args.config)
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f'invalid config: {exc}', file=sys.stderr)
        return 2

    specs = _apply_cli_filters(expand(cfg), args)
    outdir = args.outdir or cfg.outdir

    if args.count:
        print(len(specs))
        return 0
    if args.list:
        for i, s in enumerate(specs):
            extra = ''
            for key in ('stress', 'ncp_wiring_seed', 'ode_unfolds', 'batch_time',
                        'eps_jitter'):
                if key in s:
                    extra += f' {key}={s[key]}'
            print(f"{i:4d}  {s['cell']:<18} {s['wiring']:<5} {s['system']:<22} "
                  f"seed={s['seed']} clip={s['clip_norm']}{extra}")
        return 0

    run_cfg = dict(
        n_iters=args.iters, batch_size=args.batch_size, batch_time=args.batch_time,
        lr=args.lr, loss=args.loss, grad_log_every=args.grad_log_every,
        data_size=args.data_size, deterministic=args.deterministic,
    )

    if args.index is not None:
        if not (0 <= args.index < len(specs)):
            print(f'Index {args.index} out of range [0, {len(specs)})',
                  file=sys.stderr)
            return 1
        todo = [specs[args.index]]
    elif args.all:
        todo = specs
    else:
        print('Select runs via --index, --all (or use --list/--count).',
              file=sys.stderr)
        return 1

    # Provenance sidecar: written on a real run (not for --list/--count).
    manifest_path = write_manifest(outdir, str(config_path), cfg.raw, len(specs))
    print(f'manifest -> {manifest_path}')

    for i, spec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {spec['cell']} x {spec['wiring']} x "
              f"{spec['system']} (seed {spec['seed']}) -> {result_filename(spec)}")
        result = run_one(spec, run_cfg)
        path = save_result(result, outdir)
        ev = result['evaluation']
        print(f"  done in {result['training']['duration_s']:.1f}s -- "
              f"final_loss={result['training']['final_loss']:.6f} "
              f"traj MSE={ev['mse']:.6f} NRMSE={ev['nrmse']:.4f} -> {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
