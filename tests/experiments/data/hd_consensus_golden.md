## H-d consensus family (K=30, one-sided Wilcoxon over seeds, Bonferroni m=4, alpha=0.05)

| Cell | Wiring | Part | Contrast | n pairs | median diff | p one-sided | p Bonferroni | method | significant |
|------|--------|------|----------|---------|-------------|-------------|--------------|--------|-------------|
| cfc_lrc | ncp | i | consensus vs avg_in, macro-F1 m=15 | 10 | +0.1450 | 0.0009766 | 0.003906 | exact | YES |
| cfc_lrc | ncp | ii | consensus vs lambda=0, per-position AUC m in {0, 15} | 10 | +0.1175 | 0.0009766 | 0.003906 | exact | YES |
| cfc_lrc | cncp | i | consensus vs avg_in, macro-F1 m=15 | 5 | +0.0800 | — | — | — | undecidable |
| cfc_lrc | cncp | ii | consensus vs lambda=0, per-position AUC m in {0, 15} | 5 | -0.0500 | — | — | — | undecidable |

Attribution (a part-i effect counts as consensus only if learned lambda also beats lambda=0 at m=15, one-sided, alpha=0.05; no family test):
- cfc_lrc/ncp: learned lambda vs lambda=0 at m=15, n=10, p one-sided 0.0009766 -> attribution: **consensus**
- cfc_lrc/cncp: learned lambda vs lambda=0 at m=15, n=5, p one-sided — -> attribution: **late fusion**
