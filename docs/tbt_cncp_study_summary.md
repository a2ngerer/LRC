# tbt_cNCP improvement study — results summary (2026-07-06)

A thesis-ready condensation of a 9-iteration controlled study asking: does the
bio-inspired cortical wiring (cNCP) and its Thousand-Brains extensions (tbt_cNCP)
outperform simpler baselines at MATCHED parameters? Full journal:
`docs/superpowers/specs/2026-07-05-tbt-cncp-improvement-loop.md`.

## Headline finding

**Under fair, parameter-matched, control-augmented evaluation, the tbt/cNCP
cortical machinery does not beat simple baselines, and cNCP ≈ NCP ≈ dense across
every property tested.** This is a robust, mechanistically-explained negative
result, not a run of bad luck.

## Evidence (each row: the tbt/cNCP mechanism vs a FAIR baseline)

| # | Mechanism (property) | Fair comparison | Result |
|---|---|---|---|
| 1–3 | FiLM location gating | vs. concat(location) | equal |
| 2 | Multi-column voting | vs. concat | voting worse |
| 3 | Graded occlusion | FiLM best of the tbt variants | intra-family only |
| 5–6 | L5 motor / active sensing | vs. dense hidden-state motor | worse + fragile (bimodal) |
| 7 | Partial-view committee (robustness) | vs. **dropout-augmented monolith** | dominated (−0.2 @15% dropout) |
| 7d | Committee, corruption-agnostic | vs. correctly-augmented monolith | no reliable advantage |
| 8 | Data efficiency (learning curves) | cNCP vs NCP (both 45k, matched) | no low-data advantage |
| 9 | Multi-timescale laminae (architecture) | cNCP-plain vs cNCP-cortical, 5 seeds | −0.001 (null) |

## Why the negative is fundamental (mechanistic)

The per-node cell is a closed-form liquid network (cfc_lrc) that already learns its
own time constants and rich dynamics. At matched parameters, an expressive cell
inside any of the three wirings converges to similar performance on these tasks,
because the tasks do not contain structure that specifically rewards the cortical
topology. Iteration 9 makes this concrete: imposing a per-lamina timescale prior
changes nothing (−0.001), because the cell already learns timescales. Structural
priors on quantities the cell can learn add no value.

## Methodology worth defending (the reason this is a credible negative)

- **Parameter matching** enforced and re-checked (Iteration 8 caught a 26k-vs-45k
  mismatch that would have inflated a false positive).
- **Adversarial / control baselines**, not just the naive one — the
  dropout-augmented monolith (Iter 7b) is what killed the committee headline;
  weaker studies omit it.
- **≥5 seeds with 95% CIs** for every claim; single-seed go/no-go checks produced
  false positives twice (committee +0.05, data-efficiency +0.19) that evaporated
  at 5 seeds. Rule adopted: never conclude from <3 seeds.
- **Honest priors stated before running** and negatives reported in full.

## Recommendation

Treat this as the thesis's honest OUTLOOK/negative-result chapter. The positive
core deliverable is the separate, not-yet-run Farsang matrix (dense / NCP / cNCP ×
Lotka-Volterra / MuJoCo / IMDB, parameter-matched, with Optuna HPO). Further
single-mechanism tbt tweaks are not warranted — the null is explained by cell
expressiveness under matched parameters, which such tweaks do not change.
