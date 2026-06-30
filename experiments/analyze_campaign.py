# experiments/analyze_campaign.py
"""Config-driven analysis entry point -- the analysis twin of run_campaign.py.

Where ``run_campaign.py`` reads a campaign YAML and PRODUCES result JSONs, this
reads the same YAML and CONSUMES them: it takes ``outdir`` plus an ``analysis:``
section and emits the requested aggregation (metric + statistical tests) and
figures (loss / phase / gradient-flow), into ``<outdir>``.

    uv run python experiments/analyze_campaign.py --config configs/campaigns/v5.yaml
    uv run python experiments/analyze_campaign.py --config v5            # profile name
    uv run python experiments/analyze_campaign.py --config v5 --compare results/runs_v6b

It replaces the hand-rolled, dir-hard-coded ``analyze_v5.py`` / ``analyze_v6.py``
launchers with one config-driven launcher. Two reuse rules keep it a thin shell
over the legacy code (no logic is reimplemented):

  * Aggregation + statistics are imported VERBATIM from ``aggregate_results``
    (``load_runs`` / ``summary_tables`` / ``statistics``). The requested
    ``metric`` is injected by setting ``aggregate_results.METRIC`` before the
    call, the same global those functions already read.
  * Plotting reuses ``plot_results``'s plot functions, but feeds them runs
    labelled through the consolidated ``src.benchmark.variants`` so the
    +stress / +w / +lvl variants the drifted ``plot_results.load_runs`` dropped
    are now kept (one suffix implementation for tables AND plots).

The ``analysis:`` block (all keys optional; sensible defaults if the block or a
key is absent):

    analysis:
      metric: nrmse                  # primary comparison metric
      tests:  [wilcoxon, cohen_d]    # empty/absent -> skip the statistics block
      plots:  [loss, phase, gradflow]
      compare_dirs: [results/runs_v4]  # extra dirs loaded alongside outdir
                                       # (generalizes analyze_v5/v6 hard-coded lists)
"""
import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

# Make both `experiments.*` and `src.*` importable regardless of launch context.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark import load_config                               # noqa: E402
from src.benchmark.config import ConfigError                        # noqa: E402
from src.benchmark.variants import cell_label                       # noqa: E402
# Reuse the legacy aggregation + statistics path verbatim (do not reimplement).
import experiments.aggregate_results as agg                         # noqa: E402
from experiments import plot_results                                # noqa: E402

CONFIG_DIR = _REPO_ROOT / 'configs' / 'campaigns'

DEFAULT_METRIC = 'nrmse'
DEFAULT_TESTS = ['wilcoxon', 'cohen_d']
DEFAULT_PLOTS = ['loss', 'phase', 'gradflow']
KNOWN_PLOTS = set(DEFAULT_PLOTS)


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


def _analysis_section(cfg) -> dict:
    """The campaign's ``analysis:`` block (from the raw YAML), defaults applied."""
    section = dict(cfg.raw.get('analysis') or {})
    metric = section.get('metric', DEFAULT_METRIC)
    tests = section.get('tests', DEFAULT_TESTS)
    plots = section.get('plots', DEFAULT_PLOTS)
    compare = section.get('compare_dirs', []) or []
    if tests is None:
        tests = []
    if plots is None:
        plots = []
    unknown = [p for p in plots if p not in KNOWN_PLOTS]
    if unknown:
        raise SystemExit(f'analysis.plots: unknown plot kind(s) {unknown}; '
                         f'known: {sorted(KNOWN_PLOTS)}')
    return {'metric': metric, 'tests': list(tests), 'plots': list(plots),
            'compare_dirs': list(compare)}


def load_labelled_runs(runs_dirs) -> list:
    """Load raw run JSONs and stamp the consolidated variant label on each.

    Unlike ``aggregate_results.load_runs`` (which returns a flat DataFrame), the
    plot functions need the full run dicts (loss_history / trajectory_pred /
    gradient_flow). The cell label is rewritten through ``variants.cell_label``
    so plots group by the SAME variant key the aggregation tables use.
    """
    runs = []
    for runs_dir in runs_dirs:
        for path in sorted(glob(os.path.join(runs_dir, '*.json'))):
            with open(path, encoding='utf-8') as f:
                r = json.load(f)
            r['run']['cell'] = cell_label(r)
            runs.append(r)
    return runs


def build_report(df, name: str, metric: str, tests, runs_dirs) -> str:
    """Assemble the markdown report by reusing aggregate_results' table builders."""
    parts = [
        f'# Campaign {name} -- Analysis\n',
        f'Runs: {len(df)} | Dirs: {runs_dirs} | Metric: {metric} | '
        f'Cells: {sorted(df.cell.unique())} | '
        f'Wirings: {sorted(df.wiring.unique())}\n',
        agg.summary_tables(df),
    ]
    if tests:
        parts.append(f'<!-- requested tests: {", ".join(tests)} -->\n')
        parts.append(agg.statistics(df))
    return '\n'.join(parts)


def make_plots(runs, figdir: str, plots, seed: int = 0) -> None:
    """Render the requested figures, reusing plot_results' plot functions."""
    os.makedirs(figdir, exist_ok=True)
    if 'loss' in plots:
        print('Loss curves:')
        plot_results.plot_loss_curves(runs, figdir)
    if 'phase' in plots:
        print('Phase portraits:')
        try:
            plot_results.plot_phase_portraits(runs, figdir, seed=seed)
        except Exception as exc:   # phase needs a generatable dataset per system
            print(f'  skipped phase portraits: {exc}', file=sys.stderr)
    if 'gradflow' in plots:
        print('Gradient flow (RQ4):')
        plot_results.plot_gradient_flow(runs, figdir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Config-driven benchmark analysis (aggregation + plots)')
    p.add_argument('--config', required=True,
                   help='campaign profile name (configs/campaigns/<name>.yaml) '
                        'or a path to a YAML config')
    p.add_argument('--out', default=None,
                   help='output dir for report + figures (default: the '
                        "config's outdir)")
    p.add_argument('--compare', nargs='+', default=None,
                   help='extra result dirs to load alongside outdir, appended '
                        'to analysis.compare_dirs (paired clean-vs-stress etc.)')
    p.add_argument('--metric', default=None,
                   help='override analysis.metric (e.g. nrmse, mse, final_loss)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed used for phase portraits')
    p.add_argument('--skip-plots', action='store_true',
                   help='aggregation + statistics only, no figures')
    p.add_argument('--cells', default=None,
                   help='comma-separated cell-variant subset (e.g. '
                        '"cfc,cfc+noise,gru+noise")')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = _resolve_config(args.config)
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(f'invalid config: {exc}', file=sys.stderr)
        return 2

    analysis = _analysis_section(cfg)
    metric = args.metric or analysis['metric']
    out = args.out or cfg.outdir

    runs_dirs = [cfg.outdir] + analysis['compare_dirs']
    if args.compare:
        runs_dirs += list(args.compare)

    # --- aggregation + statistics (reuse aggregate_results verbatim) ---------
    df = agg.load_runs(runs_dirs)
    if args.cells:
        df = df[df['cell'].isin(args.cells.split(','))]
    if df.empty:
        print(f'No run JSONs found in {runs_dirs}', file=sys.stderr)
        return 1
    if metric not in df.columns:
        print(f'metric {metric!r} not in run records '
              f'(have: {[c for c in df.columns if df[c].dtype != object]})',
              file=sys.stderr)
        return 2

    agg.METRIC = metric   # the global summary_tables / statistics read
    report = build_report(df, cfg.name, metric, analysis['tests'], runs_dirs)

    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, 'summary.csv')
    md_path = os.path.join(out, f'analysis_{cfg.name}.md')
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f'\nWritten: {md_path} and {csv_path}')

    # --- figures (reuse plot_results' plot functions, correct labels) --------
    if not args.skip_plots and analysis['plots']:
        figdir = os.path.join(out, 'figures')
        runs = load_labelled_runs(runs_dirs)
        if args.cells:
            keep = set(args.cells.split(','))
            runs = [r for r in runs if r['run']['cell'] in keep]
        make_plots(runs, figdir, analysis['plots'], seed=args.seed)
        print(f'Figures -> {figdir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
