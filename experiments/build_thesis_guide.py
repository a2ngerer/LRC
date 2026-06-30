#!/usr/bin/env python3
"""Build the self-contained thesis refresh guide.

Reads ``experiments/thesis_guide_template.html``, inlines the curated benchmark
figures as base64 data URIs (so the output is a single portable file that works
offline and when copied elsewhere), and writes ``results/thesis_guide.html``.

Re-run after the OOM-backfill (job 481470) completes and the figures have been
regenerated, to embed the final ltc/ncp-complete plots:

    uv run python experiments/aggregate_results.py --runs results/runs results/runs_v2 --out results
    uv run python experiments/plot_results.py --runs results/runs --out results/figures_v1
    uv run python experiments/plot_results.py --runs results/runs_v2 --out results/figures_v2
    uv run python experiments/compare_v1_v2.py
    uv run python experiments/build_thesis_guide.py
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # code/
TEMPLATE = REPO / "experiments" / "thesis_guide_template.html"
OUT = REPO / "results" / "thesis_guide.html"

# Placeholder token -> figure file (relative to results/). Curated so each
# figure carries one didactic point; see the captions in the template.
FIGURES = {
    "FIG_NRMSE_BOTH": "figures_v1_vs_v2/nrmse_overview.png",
    "FIG_NRMSE_V1": "figures_v1/nrmse_overview.png",
    "FIG_NRMSE_V2": "figures_v2/nrmse_overview.png",
    "FIG_PHASE_LV": "figures_v1/phase_periodic_predator_prey.png",
    "FIG_PHASE_DUFFING": "figures_v2/phase_duffing.png",
    "FIG_GRADFLOW_LRC": "figures_v1/gradflow_lrc_spiral.png",
    "FIG_GRADFLOW_MMLRC": "figures_v2/gradflow_mm_lrc_spiral.png",
}

# Inline SVG fallback so a missing figure never breaks the build.
_MISSING = "data:image/svg+xml;base64," + base64.b64encode(
    b'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="200">'
    b'<rect width="100%" height="100%" fill="#efeee8"/>'
    b'<text x="50%" y="50%" font-family="monospace" font-size="16" '
    b'fill="#b7791f" text-anchor="middle" dominant-baseline="middle">'
    b"figure not found - run plot_results.py</text></svg>"
).decode("ascii")


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> int:
    if not TEMPLATE.exists():
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    results_dir = REPO / "results"
    embedded, missing = 0, []

    for key, rel in FIGURES.items():
        token = "{{" + key + "}}"
        fig = results_dir / rel
        if fig.exists():
            html = html.replace(token, data_uri(fig))
            embedded += 1
        else:
            html = html.replace(token, _MISSING)
            missing.append(rel)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    size_mb = OUT.stat().st_size / 1e6

    print(f"wrote {OUT}  ({size_mb:.1f} MB, {embedded}/{len(FIGURES)} figures embedded)")
    if missing:
        print("WARNING: missing figures (placeholder inserted):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
