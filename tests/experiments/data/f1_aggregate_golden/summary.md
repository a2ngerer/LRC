# Benchmark Summary

Runs: 10 | Cells: ['cfc', 'ltc'] | Wirings: ['dense'] | Seeds/config: 5-5

## Mean +/- std over seeds (NRMSE, full-trajectory rollout)

### spiral

| Cell | Dense | NCP |
|------|-------|-----|
| cfc | 0.4200 +/- 0.0158 (n=5) | — |
| ltc | 0.5200 +/- 0.0158 (n=5) | — |

## Aggregated over all systems (mean NRMSE)

| Cell | Dense | NCP |
|------|-------|-----|
| cfc | 0.4200 | nan |
| ltc | 0.5200 | nan |

## Cell comparisons — dense wiring (RQ1) (Wilcoxon signed-rank, alpha=0.05, metric=nrmse, Bonferroni m=1)

| Comparison | n pairs | excluded | median diff | p raw | p Bonferroni | alpha adj | significant | Cohen's d_z | P(A>B) [95% CI] |
|------------|---------|----------|-------------|-------|--------------|-----------|-------------|--------------|-----------------|
| cfc vs ltc (dense) | 5 | 0 | -0.1000 | 0.03389 | 0.03389 | 0.05 | YES | -3997036155235652.500 | 1.00 [0.478, 1.000] |

P(A>B): paired win fraction of A (lower nrmse wins, ties 0.5, a diverged run loses, two diverged runs tie), exact Clopper-Pearson 95% CI; reference gamma = 0.75 for orientation only (the decision is the Wilcoxon).

## Cell comparisons — ncp wiring (RQ1) (Wilcoxon signed-rank, alpha=0.05, metric=nrmse, Bonferroni m=1)

| Comparison | n pairs | excluded | median diff | p raw | p Bonferroni | alpha adj | significant | Cohen's d_z | P(A>B) [95% CI] |
|------------|---------|----------|-------------|-------|--------------|-----------|-------------|--------------|-----------------|
| cfc vs ltc (ncp) | 0 | — | — | — | — | 0.05 | too few pairs | — | — |

P(A>B): paired win fraction of A (lower nrmse wins, ties 0.5, a diverged run loses, two diverged runs tie), exact Clopper-Pearson 95% CI; reference gamma = 0.75 for orientation only (the decision is the Wilcoxon).

## Wiring effect per cell (RQ3) (Wilcoxon signed-rank, alpha=0.05, metric=nrmse, Bonferroni m=2)

| Comparison | n pairs | excluded | median diff | p raw | p Bonferroni | alpha adj | significant | Cohen's d_z | P(A>B) [95% CI] |
|------------|---------|----------|-------------|-------|--------------|-----------|-------------|--------------|-----------------|
| cfc: dense vs ncp | 0 | — | — | — | — | 0.025 | too few pairs | — | — |
| ltc: dense vs ncp | 0 | — | — | — | — | 0.025 | too few pairs | — | — |

P(A>B): paired win fraction of A (lower nrmse wins, ties 0.5, a diverged run loses, two diverged runs tie), exact Clopper-Pearson 95% CI; reference gamma = 0.75 for orientation only (the decision is the Wilcoxon).

## HPO fragility (hpo_selection.json)

| Task | Arm | lr | lr_at_grid_edge | grid_extended |
|------|-----|----|-----------------|---------------|
| pa | cfc_lrc/ncp | 0.001 | False | False |
| pa | gru/ncp | 0.001 | False | False |
| periodic_predator_prey | cfc_lrc/dense3 | 0.001 | False | False |
| periodic_predator_prey | cfc_lrc/ncp | 0.001 | True | True |

| Family | Task | A | B | sign best | sign 2nd | sign_stable_2nd |
|--------|------|---|---|-----------|----------|-----------------|
| H-c | periodic_predator_prey | cfc_lrc/ncp | cfc_lrc/dense3 | +1 | -1 | NO |
| RQ1 | pa | cfc_lrc/ncp | gru/ncp | +1 | -1 | NO |

## Person Activity arm means (test macro-F1, bootstrap over seed)

| Arm | n seeds | mean | 95% CI |
|-----|---------|------|--------|
| cfc_lrc_cncp | 10 | 0.7143 | [0.7143, 0.7143] |
| cfc_lrc_dense3 | 10 | 0.6050 | [0.4957, 0.7143] |
| cfc_lrc_ncp | 10 | 0.8646 | [0.8605, 0.8687] |
| ctrl_k1/cfc_lrc_dense | 5 | 0.8000 | [0.8000, 0.8000] |
| gru_ncp | 10 | 0.7891 | [0.7438, 0.8351] |
| lrc_ncp | 10 | 0.6414 | [0.5321, 0.7143] |

## Rollout outcome classes (test trajectories, share [95% Clopper-Pearson CI])

| System | Arm | fits | trajectories | tau_orb | ok | bounded_wrong | unbounded | non-finite fits |
|--------|-----|------|--------------|---------|----|----|----|-----------------|
| duffing | duffing/cfc_lrc_ncp_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/cfc_lrc_ncp_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/cfc_ncp_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/cfc_ncp_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/cfc_static_ncp_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/cfc_static_ncp_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/ctrl_combo/cfc_static_ncp_free | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| duffing | duffing/ctrl_combo/cfc_static_ncp_free | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| duffing | duffing/ctrl_k1/lrc_ncp_mixed | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| duffing | duffing/ctrl_k1/lrc_ncp_mixed | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| duffing | duffing/gru_dense3_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/gru_dense3_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| duffing | duffing/lrc_ncp_mixed | 10 | 20 | 0.15 | 0.90 [0.68, 0.99] | 0.00 [0.00, 0.17] | 0.10 [0.01, 0.32] | at least 1 of 10 fits non-finite (1/10 [0.00, 0.45]) |
| duffing | duffing/lrc_ncp_mixed | 10 | 20 | 0.3 | 0.90 [0.68, 0.99] | 0.00 [0.00, 0.17] | 0.10 [0.01, 0.32] | at least 1 of 10 fits non-finite (1/10 [0.00, 0.45]) |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_cncp_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_cncp_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_dense3_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_dense3_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_ncp_mixed | 10 | 20 | 0.15 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_ncp_mixed | 10 | 20 | 0.3 | 1.00 [0.83, 1.00] | 0.00 [0.00, 0.17] | 0.00 [0.00, 0.17] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_dt1/cfc_lrc_ncp_mixed | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_dt1/cfc_lrc_ncp_mixed | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_tf/cfc_lrc_ncp_teacher | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_tf/cfc_lrc_ncp_teacher | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_w/cfc_static_ncp_mixed | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/ctrl_w/cfc_static_ncp_mixed | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/gru_ncp_mixed | 10 | 20 | 0.15 | 0.90 [0.68, 0.99] | 0.00 [0.00, 0.17] | 0.10 [0.01, 0.32] | — |
| periodic_predator_prey | periodic_predator_prey/gru_ncp_mixed | 10 | 20 | 0.3 | 0.90 [0.68, 0.99] | 0.00 [0.00, 0.17] | 0.10 [0.01, 0.32] | — |
| periodic_predator_prey | periodic_predator_prey/pilot/gru_dense3_mixed | 5 | 10 | 0.15 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |
| periodic_predator_prey | periodic_predator_prey/pilot/gru_dense3_mixed | 5 | 10 | 0.3 | 1.00 [0.69, 1.00] | 0.00 [0.00, 0.31] | 0.00 [0.00, 0.31] | — |


Left out 1 tuning-seed fit(s) (seed outside 0..9).

## Floor lines (multi-anchor VPT median, test split)

| System | data seed | persistence | AR(1) | floor |
|--------|-----------|-------------|-------|-------|
| duffing | 20260702 | 2 | 12.5 | 12.5 |
| duffing | 20260703 | 2.5 | 9.5 | 9.5 |
| duffing | 20260704 | 2 | 2.5 | 2.5 |
| duffing | 20260705 | 1.5 | 9.5 | 9.5 |
| duffing | 20260706 | 2.5 | 9.5 | 9.5 |
| duffing | 20260707 | 1.5 | 5 | 5 |
| duffing | 20260708 | 4.5 | 11 | 11 |
| duffing | 20260709 | 2.5 | 6.5 | 6.5 |
| duffing | 20260710 | 2 | 12.5 | 12.5 |
| duffing | 20260711 | 1.5 | 7.5 | 7.5 |
| periodic_predator_prey | 20260702 | 1 | 5.5 | 5.5 |
| periodic_predator_prey | 20260703 | 1 | 2.5 | 2.5 |
| periodic_predator_prey | 20260704 | 1 | 3 | 3 |
| periodic_predator_prey | 20260705 | 1 | 4.5 | 4.5 |
| periodic_predator_prey | 20260706 | 1 | 5.5 | 5.5 |
| periodic_predator_prey | 20260707 | 1 | 4 | 4 |
| periodic_predator_prey | 20260708 | 1 | 4.5 | 4.5 |
| periodic_predator_prey | 20260709 | 1 | 4 | 4 |
| periodic_predator_prey | 20260710 | 1 | 3 | 3 |
| periodic_predator_prey | 20260711 | 1 | 4.5 | 4.5 |
| periodic_predator_prey | 20260802 | 1 | 4.5 | 4.5 |

| System | Arm | fits | seeds below floor | above floor |
|--------|-----|------|-------------------|-------------|
| duffing | duffing/cfc_lrc_ncp_mixed | 10 | — | yes |
| duffing | duffing/cfc_ncp_mixed | 10 | [0, 8] | yes |
| duffing | duffing/cfc_static_ncp_mixed | 10 | [0, 8] | yes |
| duffing | duffing/ctrl_combo/cfc_static_ncp_free | 5 | [0, 1, 3, 4] | NO (not learned) |
| duffing | duffing/ctrl_k1/lrc_ncp_mixed | 5 | [0, 1, 3, 4] | NO (not learned) |
| duffing | duffing/gru_dense3_mixed | 10 | [0, 1, 3, 4, 6, 8] | NO (not learned) |
| duffing | duffing/lrc_ncp_mixed | 10 | [0, 1, 3, 4, 6, 8] | NO (not learned) |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_cncp_mixed | 10 | — | yes |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_dense3_mixed | 10 | [0, 1, 2] | NO (not learned) |
| periodic_predator_prey | periodic_predator_prey/cfc_lrc_ncp_mixed | 10 | — | yes |
| periodic_predator_prey | periodic_predator_prey/ctrl_dt1/cfc_lrc_ncp_mixed | 5 | — | yes |
| periodic_predator_prey | periodic_predator_prey/ctrl_tf/cfc_lrc_ncp_teacher | 5 | — | yes |
| periodic_predator_prey | periodic_predator_prey/ctrl_w/cfc_static_ncp_mixed | 5 | [0] | yes |
| periodic_predator_prey | periodic_predator_prey/gru_ncp_mixed | 10 | — | yes |
| periodic_predator_prey | periodic_predator_prey/pilot/gru_dense3_mixed | 5 | [0, 4] | NO (not learned) |

| Family | Task | A | B | floor_bound |
|--------|------|---|---|-------------|
| H-a | duffing | duffing/cfc_ncp_mixed | duffing/cfc_static_ncp_mixed | no |
| H-b | duffing | duffing/cfc_lrc_ncp_mixed | duffing/lrc_ncp_mixed | YES |
| H-c | periodic_predator_prey | periodic_predator_prey/cfc_lrc_ncp_mixed | periodic_predator_prey/cfc_lrc_dense3_mixed | YES |
| H-c | periodic_predator_prey | periodic_predator_prey/cfc_lrc_cncp_mixed | periodic_predator_prey/cfc_lrc_dense3_mixed | YES |
| RQ1 | periodic_predator_prey | periodic_predator_prey/cfc_lrc_ncp_mixed | periodic_predator_prey/gru_ncp_mixed | no |

Seed rule: not learned if below max(persistence, AR(1)) in more than one of five seeds (two of ten).

## Sequence-suite arms (final seeds; metric: multi-anchor VPT median on the rollouts, test macro-F1 on Person Activity)

| Task | Arm | fits | median metric | diverged [95% CP] | budget above tolerance |
|------|-----|------|---------------|-------------------|------------------------|
| duffing | cfc/ncp | 10 | 12.5 | 0/10 [0.00, 0.31] | no |
| duffing | cfc_lrc/ncp | 10 | 16.25 | 0/10 [0.00, 0.31] | no |
| duffing | cfc_static/ncp | 10 | 12 | 0/10 [0.00, 0.31] | no |
| duffing | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | 5 | 3 | 0/5 [0.00, 0.52] | no |
| duffing | gru/dense3 | 10 | 9 | 0/10 [0.00, 0.31] | YES |
| duffing | lrc/ncp | 10 | 7.125 | 1/10 [0.00, 0.45] | no |
| duffing | lrc/ncp [k=1] | 5 | 7 | 0/5 [0.00, 0.52] | no |
| pa | cfc_lrc/cncp | 10 | 0.7143 | 0/10 [0.00, 0.31] | no |
| pa | cfc_lrc/dense | 5 | 0.8 | 0/5 [0.00, 0.52] | no |
| pa | cfc_lrc/dense3 | 10 | 0.7143 | 0/10 [0.00, 0.31] | no |
| pa | cfc_lrc/ncp | 10 | 0.8661 | 0/10 [0.00, 0.31] | no |
| pa | gru/ncp | 10 | 0.7857 | 0/10 [0.00, 0.31] | no |
| pa | lrc/ncp | 10 | 0.7143 | 0/10 [0.00, 0.31] | no |
| periodic_predator_prey | cfc_lrc/cncp | 10 | 9.5 | 0/10 [0.00, 0.31] | no |
| periodic_predator_prey | cfc_lrc/dense3 | 10 | 10.25 | 0/10 [0.00, 0.31] | no |
| periodic_predator_prey | cfc_lrc/ncp | 10 | 14.5 | 0/10 [0.00, 0.31] | no |
| periodic_predator_prey | cfc_lrc/ncp [dt_scale=1] | 5 | 11 | 0/5 [0.00, 0.52] | no |
| periodic_predator_prey | cfc_lrc/ncp [forcing=teacher] | 5 | 8 | 0/5 [0.00, 0.52] | no |
| periodic_predator_prey | cfc_static/ncp [fixed width] | 5 | 6 | 0/5 [0.00, 0.52] | no |
| periodic_predator_prey | gru/dense3 | 5 | 5 | 0/5 [0.00, 0.52] | no |
| periodic_predator_prey | gru/ncp | 10 | 8.5 | 1/10 [0.00, 0.45] | no |

## Pre-registered contrasts (paired over seeds, two-sided Wilcoxon, alpha=0.05, Bonferroni within the family; n_pairs < 8: undecidable)

| Family | m | Task | A | B | n pairs | status | method | p raw | p adj | significant | d_z | P(A>B) [95% CI] | median diff [95% CI] | diverged A | diverged B | floor_bound | sign_stable_tuning |
|--------|---|------|---|---|---------|--------|--------|-------|-------|-------------|-----|-----------------|----------------------|------------|------------|-------------|--------------------|
| H-a | 3 | multitimescale_lib | cfc/ncp | cfc_static/ncp | 0 | missing | — | — | — | — | — | — | — | — | — | — | — |
| H-a | 3 | duffing | cfc/ncp | cfc_static/ncp | 10 | tested | approx | 0.02535 | 0.07604 | no | +0.949 | 0.75 [0.394, 0.956] | +0.5 [+0, +1] (seed) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | False | — |
| H-a | 3 | pa | cfc/dense | cfc_static/dense | 0 | missing | — | — | — | — | — | — | — | — | — | — | — |
| H-b | 2 | duffing | cfc_lrc/ncp | lrc/ncp | 10 | tested | exact | 0.001953 | 0.003906 | YES | +4.953 | 1.00 [0.692, 1.000] | +9.375 [+8.625, +10] (seed) | 0/10 [0.00, 0.31] | 1/10 [0.00, 0.45] | True | — |
| H-b | 2 | pa | cfc_lrc/ncp | lrc/ncp | 10 | tested | approx | 0.004809 | 0.009617 | YES | +1.466 | 1.00 [0.692, 1.000] | +0.1553 [+0.1404, +0.3323] (seed x recording) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | False | — |
| H-c | 2 | periodic_predator_prey | cfc_lrc/ncp | cfc_lrc/dense3 | 10 | tested | exact | 0.001953 | 0.003906 | YES | +2.218 | 1.00 [0.692, 1.000] | +5.75 [+4.5, +10] (seed) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | True | False |
| H-c | 2 | periodic_predator_prey | cfc_lrc/cncp | cfc_lrc/dense3 | 10 | tested | approx | 0.9591 | 1 | no | +0.322 | 0.30 [0.067, 0.652] | -0.5 [-2, +9] (seed) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | True | — |
| H-c | 2 | duffing | cfc_lrc/ncp | cfc_lrc/dense3 | 0 | missing | — | — | — | — | — | — | — | — | — | — | — |
| H-c | 2 | duffing | cfc_lrc/cncp | cfc_lrc/dense3 | 0 | missing | — | — | — | — | — | — | — | — | — | — | — |
| H-c | 2 | pa | cfc_lrc/ncp | cfc_lrc/dense3 | 10 | tested | approx | 0.004892 | 0.009784 | YES | +1.469 | 1.00 [0.692, 1.000] | +0.1553 [+0.1404, +0.5085] (seed x recording) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | True | — |
| H-c | 2 | pa | cfc_lrc/cncp | cfc_lrc/dense3 | 10 | tested | approx | 0.08326 | 0.1665 | no | +0.621 | 0.65 [0.304, 0.907] | +0 [+0, +0.3643] (seed x recording) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | True | — |
| RQ1 | 3 | periodic_predator_prey | cfc_lrc/ncp | gru/ncp | 10 | tested | approx | 0.004644 | 0.01393 | YES | +1.568 | 1.00 [0.692, 1.000] | +6.5 [+2.5, +10.5] (seed) | 0/10 [0.00, 0.31] | 1/10 [0.00, 0.45] | False | — |
| RQ1 | 3 | duffing | cfc_lrc/ncp | gru/ncp | 0 | missing | — | — | — | — | — | — | — | — | — | — | — |
| RQ1 | 3 | pa | cfc_lrc/ncp | gru/ncp | 10 | tested | approx | 0.04123 | 0.1237 | no | +0.947 | 0.75 [0.394, 0.956] | +0.07143 [+0, +0.1558] (seed x recording) | 0/10 [0.00, 0.31] | 0/10 [0.00, 0.31] | False | True |

H-c conjunction (significant on every task that is not floor-bound):
- cfc_lrc/cncp vs cfc_lrc/dense3: **not supported**
- cfc_lrc/ncp vs cfc_lrc/dense3: **not supported**

A diverged fit (non-finite, or every test rollout unbounded) enters the pairing with the worst value (VPT 0, macro-F1 0); in the H-c robustness family divergence is that of the robustness set (diverged_robust: every OOD rollout unbounded or a non-finite one, non-finite noisy macro-F1). Median difference: 10,000-resample percentile bootstrap. sign_stable_tuning: the sign of A - B on the validation scores of the selected configurations agrees between the two tuning seeds.

## Person Activity chance marking (#80: test macro-F1 at most 0.01 above the one-class value of the test split; floor_bound of the contrasts)

| Task | Arm | fits | one-class | seeds at chance | above chance |
|------|-----|------|-----------|-----------------|--------------|
| pa | cfc_lrc/cncp | 10 | 0.3333 | — | yes |
| pa | cfc_lrc/dense | 5 | 0.3333 | — | yes |
| pa | cfc_lrc/dense3 | 10 | 0.3333/0.3500 | [0, 1, 2] | NO (not learned) |
| pa | cfc_lrc/ncp | 10 | 0.3333/0.3478/0.3600 | — | yes |
| pa | gru/ncp | 10 | 0.3333/0.3478/0.3600 | — | yes |
| pa | lrc/ncp | 10 | 0.3333/0.3500 | [0, 1] | yes |

Seed rule (decision f): not learned if at chance in more than one of five seeds (two of ten).

## Descriptive pairs (no significance claim)

| Task | A | B | n pairs | median diff | P(A>B) [95% CI] |
|------|---|---|---------|-------------|-----------------|
| duffing | cfc/ncp | cfc_lrc/ncp | 10 | -3.75 | 0.00 [0.000, 0.308] |
| duffing | cfc/ncp | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | 5 | +9 | 1.00 [0.478, 1.000] |
| duffing | cfc/ncp | gru/dense3 | 10 | +3.5 | 1.00 [0.692, 1.000] |
| duffing | cfc/ncp | lrc/ncp | 10 | +5.5 | 1.00 [0.692, 1.000] |
| duffing | cfc/ncp | lrc/ncp [k=1] | 5 | +5 | 1.00 [0.478, 1.000] |
| duffing | cfc_lrc/ncp | cfc_static/ncp | 10 | +4.25 | 1.00 [0.692, 1.000] |
| duffing | cfc_lrc/ncp | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | 5 | +12 | 1.00 [0.478, 1.000] |
| duffing | cfc_lrc/ncp | gru/dense3 | 10 | +7.25 | 1.00 [0.692, 1.000] |
| duffing | cfc_lrc/ncp | lrc/ncp [k=1] | 5 | +8 | 1.00 [0.478, 1.000] |
| duffing | cfc_static/ncp | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | 5 | +9 | 1.00 [0.478, 1.000] |
| duffing | cfc_static/ncp | gru/dense3 | 10 | +3 | 1.00 [0.692, 1.000] |
| duffing | cfc_static/ncp | lrc/ncp | 10 | +4.875 | 1.00 [0.692, 1.000] |
| duffing | cfc_static/ncp | lrc/ncp [k=1] | 5 | +5 | 1.00 [0.478, 1.000] |
| duffing | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | gru/dense3 | 5 | -6 | 0.00 [0.000, 0.522] |
| duffing | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | lrc/ncp | 5 | -3.25 | 0.20 [0.005, 0.716] |
| duffing | cfc_static/ncp [k=1, dt_scale=1, fixed width, forcing=free] | lrc/ncp [k=1] | 5 | -4 | 0.00 [0.000, 0.522] |
| duffing | gru/dense3 | lrc/ncp | 10 | +1.875 | 1.00 [0.692, 1.000] |
| duffing | gru/dense3 | lrc/ncp [k=1] | 5 | +2 | 0.90 [0.371, 1.000] |
| duffing | lrc/ncp | lrc/ncp [k=1] | 5 | -1.25 | 0.40 [0.053, 0.853] |
| pa | cfc_lrc/cncp | cfc_lrc/dense | 5 | -0.08571 | 0.00 [0.000, 0.522] |
| pa | cfc_lrc/cncp | cfc_lrc/ncp | 10 | -0.1518 | 0.00 [0.000, 0.308] |
| pa | cfc_lrc/cncp | gru/ncp | 10 | -0.07143 | 0.25 [0.044, 0.606] |
| pa | cfc_lrc/cncp | lrc/ncp | 10 | +0 | 0.60 [0.262, 0.878] |
| pa | cfc_lrc/dense | cfc_lrc/dense3 | 5 | +0.45 | 1.00 [0.478, 1.000] |
| pa | cfc_lrc/dense | cfc_lrc/ncp | 5 | -0.06607 | 0.00 [0.000, 0.522] |
| pa | cfc_lrc/dense | gru/ncp | 5 | +0.08571 | 0.60 [0.147, 0.947] |
| pa | cfc_lrc/dense | lrc/ncp | 5 | +0.08571 | 1.00 [0.478, 1.000] |
| pa | cfc_lrc/dense3 | gru/ncp | 10 | -0.1473 | 0.15 [0.011, 0.503] |
| pa | cfc_lrc/dense3 | lrc/ncp | 10 | +0 | 0.45 [0.153, 0.776] |
| pa | gru/ncp | lrc/ncp | 10 | +0.1429 | 0.80 [0.444, 0.975] |
| periodic_predator_prey | cfc_lrc/cncp | cfc_lrc/ncp | 10 | -4.75 | 0.00 [0.000, 0.308] |
| periodic_predator_prey | cfc_lrc/cncp | cfc_lrc/ncp [dt_scale=1] | 5 | -1 | 0.10 [0.000, 0.629] |
| periodic_predator_prey | cfc_lrc/cncp | cfc_lrc/ncp [forcing=teacher] | 5 | +2 | 0.70 [0.209, 0.977] |
| periodic_predator_prey | cfc_lrc/cncp | cfc_static/ncp [fixed width] | 5 | +4 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/cncp | gru/dense3 | 5 | +4.5 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/cncp | gru/ncp | 10 | +0.75 | 0.70 [0.348, 0.933] |
| periodic_predator_prey | cfc_lrc/dense3 | cfc_lrc/ncp [dt_scale=1] | 5 | -9 | 0.00 [0.000, 0.522] |
| periodic_predator_prey | cfc_lrc/dense3 | cfc_lrc/ncp [forcing=teacher] | 5 | -6 | 0.30 [0.023, 0.791] |
| periodic_predator_prey | cfc_lrc/dense3 | cfc_static/ncp [fixed width] | 5 | -4 | 0.40 [0.053, 0.853] |
| periodic_predator_prey | cfc_lrc/dense3 | gru/dense3 | 5 | -5 | 0.40 [0.053, 0.853] |
| periodic_predator_prey | cfc_lrc/dense3 | gru/ncp | 10 | +1.25 | 0.60 [0.262, 0.878] |
| periodic_predator_prey | cfc_lrc/ncp | cfc_lrc/ncp [dt_scale=1] | 5 | +1 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp | cfc_lrc/ncp [forcing=teacher] | 5 | +4 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp | cfc_static/ncp [fixed width] | 5 | +6 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp | gru/dense3 | 5 | +7 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [dt_scale=1] | cfc_lrc/ncp [forcing=teacher] | 5 | +3 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [dt_scale=1] | cfc_static/ncp [fixed width] | 5 | +5 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [dt_scale=1] | gru/dense3 | 5 | +6 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [dt_scale=1] | gru/ncp | 5 | +1.5 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [forcing=teacher] | cfc_static/ncp [fixed width] | 5 | +2 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [forcing=teacher] | gru/dense3 | 5 | +3 | 1.00 [0.478, 1.000] |
| periodic_predator_prey | cfc_lrc/ncp [forcing=teacher] | gru/ncp | 5 | -1.5 | 0.20 [0.005, 0.716] |
| periodic_predator_prey | cfc_static/ncp [fixed width] | gru/dense3 | 5 | +1 | 0.70 [0.209, 0.977] |
| periodic_predator_prey | cfc_static/ncp [fixed width] | gru/ncp | 5 | -3.5 | 0.20 [0.005, 0.716] |
| periodic_predator_prey | gru/dense3 | gru/ncp | 5 | -3.5 | 0.20 [0.005, 0.716] |
