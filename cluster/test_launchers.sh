#!/usr/bin/env bash
# Smoke test for the task launchers: every submit script in dry-run mode
# against a fake host must resolve the run folder under the benchmarks tree,
# pass RUN_DIR and the log paths to sbatch, and never touch the network.
#   ./cluster/test_launchers.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export CLUSTER_HOST="fakehost.invalid"
export CLUSTER_REPO_DIR="wk/code"
export CLUSTER_BENCH_DIR="wk/benchmarks"
today="$(date -u +%F)"
fail=0

check() {  # check <name> <output> <expected substring>...
    local name="$1" out="$2"; shift 2
    for want in "$@"; do
        if ! grep -qF -- "$want" <<<"$out"; then
            echo "FAIL [$name]: missing '$want'"; fail=1
        fi
    done
    if grep -q "^Submitted\|^rsync " <<<"$out"; then
        echo "FAIL [$name]: dry run reached rsync/sbatch"; fail=1
    fi
}

out="$(LV_CELLS=cfc_lrc LV_SEEDS=0 ./cluster/submit_lotka_volterra.sh -n smoke 2>&1)"
check lv "$out" "run folder: ${CLUSTER_HOST}:wk/benchmarks/01_lotka_volterra/runs/${today}_smoke" \
    "RUN_DIR=\$HOME/wk/benchmarks/01_lotka_volterra/runs/${today}_smoke sbatch" \
    "--output=\$HOME/wk/benchmarks/01_lotka_volterra/runs/${today}_smoke/logs/%x-%j.out" \
    "LV_CELLS='cfc_lrc' LV_SEEDS='0'" "cluster/lotka_volterra.sbatch" "fits: 6"

out="$(LV_EXTRA="--columns 2 --partition same" LV_PARAM_BUDGET=0 SBATCH_EXTRA="--time=00:10:00" ./cluster/submit_lotka_volterra.sh -n smoke 2>&1)"
check extra "$out" "LV_EXTRA='--columns 2 --partition same'" "LV_PARAM_BUDGET='0'" "--export=ALL --time=00:10:00"

out="$(LV_SYSTEM=duffing ./cluster/submit_lotka_volterra.sh -n smoke 2>&1)"
check duffing "$out" "02_duffing/runs/${today}_smoke"

out="$(LV_SYSTEM=multitimescale LV_TSPAN="0 10" ./cluster/submit_lotka_volterra.sh -n smoke 2>&1)"
check tspan "$out" "LV_SYSTEM='multitimescale' LV_TSPAN='0 10'"

out="$(BENCH=00_canary ./cluster/submit_lotka_volterra.sh -n lv-smoke 2>&1)"
check canary "$out" "00_canary/runs/${today}_lv-smoke"
if grep -q "test -f" <<<"$out"; then
    echo "FAIL [canary]: LV launcher checks the PA dataset"; fail=1
fi

out="$(PA_EPOCHS=1 ./cluster/submit_person_activity.sh -n smoke 2>&1)"
check pa "$out" "03_person_activity/runs/${today}_smoke" "PA_EPOCHS='1'" "cluster/person_activity.sbatch" \
    "--delete --copy-links --filter=P data/" \
    "[dry-run] ssh ${CLUSTER_HOST} test -f wk/code/data/person/ConfLongDemo_JSI.txt"

out="$(CO_KS='1 4' ./cluster/submit_committee.sh -n smoke 2>&1)"
check committee "$out" "04_column_voting/runs/${today}_smoke" "CO_KS='1 4'" "cluster/committee.sbatch" \
    "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

out="$(BENCH=00_canary GH_TASK=lv GH_CELL=cfc_lrc GH_WIRING=dense3 GH_EXTRA="--epochs 2 --grid-lr 1e-3 3e-3 --grid-clip off" ./cluster/submit_grid_hpo.sh --dry-run canary 2>&1)"
check grid_hpo "$out" "00_canary/runs/${today}_canary" "GH_TASK='lv'" \
    "GH_EXTRA='--epochs 2 --grid-lr 1e-3 3e-3 --grid-clip off'" "cluster/grid_hpo.sbatch"

out="$(GH_TASK=pa GH_CELL=gru GH_WIRING=ncp ./cluster/submit_grid_hpo.sh -n smoke 2>&1)"
check grid_hpo_pa "$out" "03_person_activity/runs/${today}_smoke" \
    "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

# sync_to_cluster.sh has no dry run: check the filter statically.
grep -qF -- "--filter='P data/'" cluster/sync_to_cluster.sh \
    || { echo "FAIL [sync_to_cluster]: data/ not protected from --delete"; fail=1; }

out="$(BENCH=00_canary ./cluster/submit_suite.sh -n --wandb configs/suite/canary_lv_hpo.yaml canary 2>&1)"
check suite "$out" "00_canary/runs/${today}_canary" "fits: 4 manifest lines" \
    "cat > \$HOME/wk/benchmarks/00_canary/runs/${today}_canary/manifest_heavy.jsonl" \
    "SUITE_GROUP=default N_SPECS=1 RUNS_PER_GPU=4 RUN_DIR=" "--array=0-0%8" \
    "SUITE_GROUP=heavy N_SPECS=3 RUNS_PER_GPU=2 RUN_DIR=" "--array=0-1%8" "--mem=32G" \
    "--time=01:00:00" "cluster/suite_array.sbatch"

out="$(./cluster/submit_suite.sh -n --resume configs/suite/canary_lv_hpo.yaml 2026-10-20_lv-final 2>&1)"
check suite_resume "$out" "01_lotka_volterra/runs/2026-10-20_lv-final"

if out="$(./cluster/submit_suite.sh -n missing.yaml canary 2>&1)"; then
    echo "FAIL [suite]: accepted a missing config"; fail=1
fi
if out="$(./cluster/submit_committee.sh -n 'bad label' 2>&1)"; then
    echo "FAIL [label]: accepted a label with a space"; fail=1
fi
if out="$(./cluster/submit_committee.sh -n 2>&1)"; then
    echo "FAIL [usage]: accepted a missing label"; fail=1
fi

for f in cluster/lotka_volterra.sbatch cluster/person_activity.sbatch cluster/committee.sbatch \
         cluster/grid_hpo.sbatch cluster/suite_array.sbatch; do
    if grep -q 'outputs/slurm\|results/\(person_activity\|lotka_volterra\|committee\)' "$f"; then
        echo "FAIL [$f]: still writes under the code tree"; fail=1
    fi
    grep -q 'RUN_DIR:?' "$f" || { echo "FAIL [$f]: no RUN_DIR guard"; fail=1; }
done

if [[ $fail -eq 0 ]]; then echo "launchers: OK"; else exit 1; fi
