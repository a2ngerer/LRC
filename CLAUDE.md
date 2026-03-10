# LRC Benchmark Codebase

Thesis benchmark: 4 neuron types × 2 wirings (8 combinations) on Neural ODE + control tasks.

Detailed workflow: `claude_instructions/code-workflow.md`
Progress log: `obsidian_master_thesis/Thesis/work-documentation.md`

## Core Rules

- **Every task = its own branch**: `phase{N}/step{M}-short-name` (e.g. `phase1/step1-migrate-lrc-cells`)
- **Commits**: small, atomic — commit per logical sub-step, not per file
- **After each branch**: merge to `main` with `--no-ff`, then push branch + main to origin
- **After each session**: update `obsidian_master_thesis/Thesis/work-documentation.md`
- **Python**: always `uv` — never `pip`, `conda`, `python` directly (use `uv run python`)

## Repo Structure (target)

```
src/neurons/   src/wirings/   src/models/   src/tasks/   src/evaluation/
experiments/   notebooks/     results/ (gitignored)       pyproject.toml
```
