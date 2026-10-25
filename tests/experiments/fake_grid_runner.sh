#!/usr/bin/env bash
# Fake task runner for test_run_grid_hpo.py. Parses the flags the HPO driver
# passes, appends "<lr> <clip> <seed> <all args>" to $FAKE_LOG and writes
# <outdir>/fake.json whose metrics come from the $FAKE_SCORES row "<lr> <clip>
# <seed> <score> <tie_break>" (clip "off" = no --clipnorm); a missing row fails
# the fit. "system" echoes --system (null without it, as the PA runner).
set -euo pipefail
clip=off; system=null; all="$*"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lr) lr="$2"; shift ;;
        --clipnorm) clip="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --outdir) outdir="$2"; shift ;;
        --system) system="\"$2\""; shift ;;
    esac
    shift
done
echo "$lr $clip $seed $all" >> "$FAKE_LOG"
row="$(awk -v l="$lr" -v c="$clip" -v s="$seed" \
    '$1 == l && $2 == c && $3 == s {print $4, $5}' "$FAKE_SCORES")"
[[ -n "$row" ]] || { echo "no score for $lr $clip $seed" >&2; exit 1; }
read -r score tie <<<"$row"
mkdir -p "$outdir"
cat > "$outdir/fake.json" <<JSON
{"system": $system, "data_seed": $((1000 + seed)), "val_macro_f1": $score,
 "rollout_metrics_val": {"vpt_median": $score, "nrmse_32_median": $tie}}
JSON
