# Verify LRC Results Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `experiments/verify_neural_ode.py` that trains LRC_AR + Dense on all 6 ODE systems for 2000 iterations, saves a structured JSON baseline to `results/`, and exits non-zero if any system fails to halve its loss.

**Architecture:** Single verification script with a testable `run_verification()` function and a `check_convergence()` helper. The CLI entry point calls these in sequence. Tests use `run_verification(systems=['spiral'], n_iters=5)` for speed.

**Tech Stack:** uv, TensorFlow 2.15, scipy, pytest

---

## Chunk 1: Branch + Test Scaffold

### Task 1: Create feature branch

**Files:**
- (none — branch setup only)

- [ ] **Step 1: Create feature branch from main**

```bash
cd /Users/angeral/Repositories/master_thesis/code
git checkout main && git pull
git checkout -b phase1/step6-verify-lrc-results
```

Expected: now on branch `phase1/step6-verify-lrc-results`

---

### Task 2: Test scaffold + failing tests

**Files:**
- Create: `tests/experiments/__init__.py`
- Create: `tests/experiments/test_verify_script.py`

- [ ] **Step 1: Create test directory**

Create `tests/experiments/__init__.py` as an empty file.

- [ ] **Step 2: Create `tests/experiments/test_verify_script.py`**

```python
import pytest
from experiments.verify_neural_ode import run_verification, check_convergence


def test_run_verification_output_shape():
    """run_verification returns dict with correct structure."""
    result = run_verification(systems=['spiral'], n_iters=5)

    assert set(result.keys()) == {'timestamp', 'config', 'systems'}
    assert 'spiral' in result['systems']

    sys_data = result['systems']['spiral']
    assert set(sys_data.keys()) == {'initial_loss', 'final_loss', 'loss_history'}
    assert len(sys_data['loss_history']) == 5
    assert sys_data['initial_loss'] > 0
    assert sys_data['final_loss'] > 0
    assert all(isinstance(v, float) for v in sys_data['loss_history'])


def test_check_convergence_passes():
    """check_convergence returns True when all final losses are < 0.5 * initial."""
    results = {
        'systems': {
            'spiral': {'initial_loss': 1.0, 'final_loss': 0.3},
            'duffing': {'initial_loss': 0.8, 'final_loss': 0.2},
        }
    }
    assert check_convergence(results) is True


def test_check_convergence_fails():
    """check_convergence returns False when any final loss >= 0.5 * initial."""
    results = {
        'systems': {
            'spiral': {'initial_loss': 1.0, 'final_loss': 0.3},
            'duffing': {'initial_loss': 0.8, 'final_loss': 0.6},
        }
    }
    assert check_convergence(results) is False
```

- [ ] **Step 3: Run tests — verify they FAIL**

```bash
uv run pytest tests/experiments/test_verify_script.py -v
```

Expected: `ModuleNotFoundError: No module named 'experiments.verify_neural_ode'`

This confirms the test setup is correct.

- [ ] **Step 4: Commit test scaffold**

```bash
git add tests/experiments/
git commit -m "test(experiments): add failing tests for verify_neural_ode"
```

---

## Chunk 2: Implementation

### Task 3: Implement verify_neural_ode.py

**Files:**
- Create: `experiments/verify_neural_ode.py`

- [ ] **Step 1: Create `experiments/verify_neural_ode.py`**

```python
import argparse
import json
import os
import sys
from datetime import datetime

from src.tasks.neural_ode.datasets import generate_dataset
from src.tasks.neural_ode.ode_model import ODEFuncModel
from src.tasks.neural_ode.trainer import train

ALL_SYSTEMS = [
    'spiral',
    'duffing',
    'periodic_sinusoidal',
    'periodic_predator_prey',
    'limited_predator_prey',
    'nonlinear_predator_prey',
]

DEFAULT_CONFIG = {
    'neuron': 'lrc_ar',
    'wiring': 'dense',
    'units': 16,
    'niters': 2000,
    'batch_size': 16,
    'batch_time': 16,
    'lr': 1e-3,
}


def run_verification(systems=ALL_SYSTEMS, n_iters=None, config=None):
    """Run training on each ODE system and return results dict.

    Args:
        systems:  list of system name strings to run
        n_iters:  overrides config['niters'] if provided
        config:   dict of hyperparameters; defaults to DEFAULT_CONFIG

    Returns:
        dict with keys 'timestamp', 'config', 'systems'.
        Each entry in 'systems' has 'initial_loss', 'final_loss', 'loss_history'.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if n_iters is not None:
        cfg['niters'] = n_iters

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'config': cfg,
        'systems': {},
    }

    for name in systems:
        print(f'\n=== {name} ===')
        t, y = generate_dataset(name)
        model = ODEFuncModel(cfg['neuron'], cfg['wiring'], cfg['units'], features=2)
        losses = train(model, t, y,
                       n_iters=cfg['niters'],
                       batch_size=cfg['batch_size'],
                       batch_time=cfg['batch_time'],
                       lr=cfg['lr'])
        initial = losses[0]
        final = losses[-1]
        pct = (initial - final) / initial * 100
        print(f'{name}: {initial:.4f} → {final:.4f} ({pct:.1f}% decrease)')
        results['systems'][name] = {
            'initial_loss': initial,
            'final_loss': final,
            'loss_history': losses,
        }

    return results


def check_convergence(results):
    """Return True if all systems have final_loss < 0.5 * initial_loss."""
    all_converged = True
    for name, data in results['systems'].items():
        if data['final_loss'] >= 0.5 * data['initial_loss']:
            print(f'FAIL: {name} did not converge '
                  f'(initial={data["initial_loss"]:.4f}, '
                  f'final={data["final_loss"]:.4f})')
            all_converged = False
    return all_converged


def save_results(results, path='results/neural_ode_lrc_baseline.json'):
    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {path}')


def main():
    parser = argparse.ArgumentParser(
        description='Verify LRC_AR + Dense on all 6 Neural ODE systems')
    parser.add_argument('--output', default='results/neural_ode_lrc_baseline.json',
                        help='Output JSON path')
    args = parser.parse_args()

    results = run_verification()
    save_results(results, args.output)

    if check_convergence(results):
        print('\nAll systems converged. ✓')
        sys.exit(0)
    else:
        print('\nSome systems failed to converge.')
        sys.exit(1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run tests — verify they PASS**

```bash
uv run pytest tests/experiments/test_verify_script.py -v
```

Expected: all 3 tests PASS.

Troubleshooting:
- `test_run_verification_output_shape` fails with import error: ensure `experiments/__init__.py` exists (it does — it was created in phase1/step5 and is already in the repo).
- `test_run_verification_output_shape` fails with key error: verify `run_verification` returns exactly `{'timestamp', 'config', 'systems'}` at the top level and each system entry has exactly `{'initial_loss', 'final_loss', 'loss_history'}`.
- `test_check_convergence_fails` returns True instead of False: verify the condition is `>=` (not `>`).

- [ ] **Step 3: Run full test suite — check for regressions**

```bash
uv run pytest -v
```

Expected: all tests PASS (neurons + wirings + models + tasks + experiments).

- [ ] **Step 4: Commit implementation**

```bash
git add experiments/verify_neural_ode.py
git commit -m "feat(experiments): add verify_neural_ode runner for all 6 ODE systems"
```

---

### Task 4: Add results/ to .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add `results/` entry to `.gitignore`**

Open `.gitignore` and add this line in the "project-specific" section (near the bottom, above the Claude Code section):

```
results/
```

- [ ] **Step 2: Commit .gitignore update**

```bash
git add .gitignore
git commit -m "chore: gitignore results/ directory"
```

---

## Chunk 3: Merge + Work Log

### Task 5: Merge to main and push

- [ ] **Step 1: Merge branch to main**

```bash
git checkout main
git merge --no-ff phase1/step6-verify-lrc-results -m "merge: phase1/step6-verify-lrc-results — verify LRC_AR+Dense on 6 ODE systems"
```

- [ ] **Step 2: Push branch and main to origin**

```bash
git push origin phase1/step6-verify-lrc-results
git push origin main
```

---

### Task 6: Update Obsidian work log and project status

- [ ] **Step 1: Prepend entry to `Thesis/work-documentation.md`**

Read the current content of `Thesis/work-documentation.md` via Obsidian MCP, then prepend the following entry (use `wholeFile/overwrite` after prepending):

```markdown
## 2026-03-10 — Phase 1 / Step 6: Verify LRC Results

**Branch:** `phase1/step6-verify-lrc-results` (merged ✅)

### What was done
- Created `experiments/verify_neural_ode.py` with `run_verification()` and `check_convergence()`
- Runs LRC_AR + Dense on all 6 ODE systems for 2000 iterations each
- Saves structured JSON baseline to `results/neural_ode_lrc_baseline.json`
- Exit 0 if all systems halve their loss, exit 1 otherwise
- Added `results/` to `.gitignore`
- 3 new tests

### Files changed
- `experiments/verify_neural_ode.py` (new)
- `tests/experiments/__init__.py` (new)
- `tests/experiments/test_verify_script.py` (new)
- `.gitignore` (modified)

### Next
`phase1/step7` — TBD (see roadmap)

---
```

- [ ] **Step 2: Update `Meta/claude-project-status.md`**

Mark `phase1/step6-verify-lrc-results` as done (`[x]`) and add the next step from the roadmap as pending.
