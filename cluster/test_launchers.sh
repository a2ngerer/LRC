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

out="$(CO_KS='1 4' CO_WANDB=1 ./cluster/submit_committee.sh -n smoke 2>&1)"
check committee "$out" "04_column_voting/runs/${today}_smoke" "CO_KS='1 4'" "cluster/committee.sbatch" \
    "test -f wk/code/data/person/ConfLongDemo_JSI.txt" "CO_WANDB='1'" \
    "RUN_GIT_SHA=$(git rev-parse HEAD)"

out="$(BENCH=00_canary GH_TASK=lv GH_CELL=cfc_lrc GH_WIRING=dense3 GH_EXTRA="--epochs 2 --grid-lr 1e-3 3e-3 --grid-clip off" ./cluster/submit_grid_hpo.sh --dry-run canary 2>&1)"
check grid_hpo "$out" "00_canary/runs/${today}_canary" "GH_TASK='lv'" \
    "GH_EXTRA='--epochs 2 --grid-lr 1e-3 3e-3 --grid-clip off'" "cluster/grid_hpo.sbatch"

out="$(GH_TASK=pa GH_CELL=gru GH_WIRING=ncp ./cluster/submit_grid_hpo.sh -n smoke 2>&1)"
check grid_hpo_pa "$out" "03_person_activity/runs/${today}_smoke" \
    "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

out="$(DP_LV_ARGS="--epochs 3" ./cluster/submit_determinism_probe.sh -n smoke 2>&1)"
check determinism "$out" "00_canary/runs/${today}_smoke" "DP_LV_ARGS='--epochs 3'" \
    "cluster/determinism_probe.sbatch" "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

out="$(EX_MODE=ablation EX_RESULTS='../benchmarks/a/runs/r/results/cfc_lrc_cncp_seed*.json ../b/*.json' EX_DATA_PATH=data/x.txt ./cluster/submit_eval_x.sh -n f3-x1-ablation 2>&1)"
check eval_x_ablation "$out" "00_canary/runs/${today}_f3-x1-ablation" "cluster/eval_x.sbatch" \
    "EX_MODE='ablation' EX_RESULTS='../benchmarks/a/runs/r/results/cfc_lrc_cncp_seed*.json ../b/*.json' EX_DATA_PATH='data/x.txt' RUN_DIR=" \
    "fits: 1 job, eval-only" "output: results/cncp_ablation.csv (--out)" "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

out="$(BENCH=01_lotka_volterra EX_MODE=gains EX_RESULTS='../x/*_seed*.json' ./cluster/submit_eval_x.sh -n f3-x2-gains 2>&1)"
check eval_x_gains "$out" "01_lotka_volterra/runs/${today}_f3-x2-gains" "cluster/eval_x.sbatch" \
    "EX_MODE='gains' EX_RESULTS='../x/*_seed*.json' RUN_DIR=" "results/cncp_paths.csv"

eval_x_rejects() {  # eval_x_rejects <name> <message> <env assignment>... [-- <label>]
    local name="$1" msg="$2" out label=smoke; shift 2
    local env_args=()
    while [[ $# -gt 0 && "$1" != -- ]]; do env_args+=("$1"); shift; done
    [[ "${1:-}" == -- ]] && label="${2:-}"
    # shellcheck disable=SC2086  # an empty label must vanish from the call
    if out="$(env "${env_args[@]}" ./cluster/submit_eval_x.sh -n $label 2>&1)" \
            || grep -q "run folder" <<<"$out" || ! grep -q "$msg" <<<"$out"; then
        echo "FAIL [eval_x_$name]: accepted ${env_args[*]}, got past the check or no '$msg'"; fail=1
    fi
}
eval_x_rejects mode "must be ablation or gains" EX_MODE=bogus EX_RESULTS=x.json
eval_x_rejects empty "is empty" EX_MODE=gains "EX_RESULTS= "
eval_x_rejects quote "single quote" EX_MODE=gains "EX_RESULTS=a'b.json"
eval_x_rejects label "label is required" EX_MODE=gains EX_RESULTS=x.json -- setup
eval_x_rejects data_path "ablation only" EX_MODE=gains EX_RESULTS=x.json EX_DATA_PATH=d.txt

out="$(GH_TASK=co GH_CELL=cfc_lrc GH_WIRING=ncp GH_EXTRA="--grid-clip off --n-columns 3" ./cluster/submit_grid_hpo.sh -n smoke 2>&1)"
check grid_hpo_co "$out" "04_column_voting/runs/${today}_smoke" "GH_TASK='co'" \
    "test -f wk/code/data/person/ConfLongDemo_JSI.txt"

# sync_to_cluster.sh has no dry run: check the filter statically.
grep -qF -- "--filter='P data/'" cluster/sync_to_cluster.sh \
    || { echo "FAIL [sync_to_cluster]: data/ not protected from --delete"; fail=1; }

out="$(BENCH=00_canary ./cluster/submit_suite.sh -n --wandb configs/suite/canary_lv_hpo.yaml canary 2>&1)"
check suite "$out" "00_canary/runs/${today}_canary" "fits: 4 manifest lines" \
    "cat > \$HOME/wk/benchmarks/00_canary/runs/${today}_canary/manifest_heavy.jsonl" \
    "SUITE_GROUP=default N_SPECS=1 RUNS_PER_GPU=4 SUITE_CAMPAIGN='canary_lv_hpo' SUITE_STAGE='hpo' RUN_DIR=" "--array=0-0%8" \
    "SUITE_GROUP=heavy N_SPECS=3 RUNS_PER_GPU=2 SUITE_CAMPAIGN='canary_lv_hpo' SUITE_STAGE='hpo' RUN_DIR=" \
    "--array=0-1%8" "--mem=32G" "--time=01:00:00" "cluster/suite_array.sbatch" "RUN_GIT_SHA="

out="$(./cluster/submit_suite.sh -n --resume configs/suite/canary_lv_hpo.yaml 2026-10-20_lv-final 2>&1)"
check suite_resume "$out" "01_lotka_volterra/runs/2026-10-20_lv-final" \
    "[dry-run] ssh ${CLUSTER_HOST} if [ -f \$HOME/wk/benchmarks/01_lotka_volterra/runs/2026-10-20_lv-final/manifest_default.jsonl ]" \
    "[dry-run] resume guard: group default not checked" \
    "[dry-run] resume guard: group heavy not checked"
if grep -q "resume guard" <<<"$(./cluster/submit_suite.sh -n configs/suite/canary_lv_hpo.yaml canary 2>&1)"; then
    echo "FAIL [suite]: the resume guard runs without --resume"; fail=1
fi

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
         cluster/grid_hpo.sbatch cluster/suite_array.sbatch cluster/determinism_probe.sbatch \
         cluster/eval_x.sbatch; do
    if grep -q 'outputs/slurm\|results/\(person_activity\|lotka_volterra\|committee\)' "$f"; then
        echo "FAIL [$f]: still writes under the code tree"; fail=1
    fi
    grep -q 'RUN_DIR:?' "$f" || { echo "FAIL [$f]: no RUN_DIR guard"; fail=1; }
done

# Every sbatch that can run a --wandb fit logs offline into the run folder.
for f in cluster/lotka_volterra.sbatch cluster/person_activity.sbatch cluster/committee.sbatch \
         cluster/grid_hpo.sbatch cluster/suite_array.sbatch; do
    grep -q 'WANDB_MODE=offline' "$f" && grep -q 'WANDB_DIR="$RUN_DIR/wandb"' "$f" \
        || { echo "FAIL [$f]: wandb not offline into the run folder"; fail=1; }
done

# wandb_sync.sh <run folder>: every offline-run-* below it (a fake uv echoes
# the call), nothing outside it.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/run/wandb/wandb/offline-run-a/files" \
    "$tmp/run/results/hpo/x/wandb/offline-run-b" "$tmp/other/offline-run-c"
printf '#!/usr/bin/env bash\necho "uv $*"\n' >"$tmp/bin/uv" && chmod +x "$tmp/bin/uv"
out="$(PATH="$tmp/bin:$PATH" ./cluster/wandb_sync.sh "$tmp/run" 2>&1)"
check wandb_sync "$out" "syncing 2 offline run(s)" "uv run wandb sync" \
    "run/wandb/wandb/offline-run-a" "run/results/hpo/x/wandb/offline-run-b"
if grep -q "offline-run-c\|offline-run-a/files" <<<"$out"; then
    echo "FAIL [wandb_sync]: synced a directory that is no offline run of the folder"; fail=1
fi
# a symlinked folder (results/ and outputs/ point to the eSSD)
ln -s "$tmp/run" "$tmp/link"
out="$(PATH="$tmp/bin:$PATH" ./cluster/wandb_sync.sh "$tmp/link" 2>&1)"
check wandb_sync_symlink "$out" "syncing 2 offline run(s)" "link/wandb/wandb/offline-run-a"
if out="$(PATH="$tmp/bin:$PATH" ./cluster/wandb_sync.sh "$tmp/bin" 2>&1)"; then
    echo "FAIL [wandb_sync]: accepted a folder without offline runs"; fail=1
fi

# suite_array.sbatch slice exit code (#91): a fake uv stands in for the runner
# (prints a skip line for index 3 and exits 0), no SLURM involved.
suite_task() {  # suite_task <expected rc> <name> <task id> <N_SPECS> <rows> <K>
    local want="$1" name="$2" task="$3" n="$4" rows="$5" k="$6" rc=0 i
    local rd="$tmp/suite_$name" st="$tmp/stub"
    mkdir -p "$rd" "$st" "$tmp/home"
    : >"$rd/manifest_default.jsonl"
    for ((i = 0; i < rows; i++)); do echo "{\"id\": $i}" >>"$rd/manifest_default.jsonl"; done
    echo >>"$rd/manifest_default.jsonl"   # a trailing blank line is no row
    cat >"$st/uv" <<'STUB'
#!/usr/bin/env bash
case "$*" in *suite_manifest.py*) ;; *) exit 0 ;; esac
args="$*"; idx="${args##*--index }"; idx="${idx%% *}"
[[ "$idx" == 3 ]] && echo "[skip] line 3: done exists"
exit 0
STUB
    chmod +x "$st/uv"
    HOME="$tmp/home" PATH="$st:$PATH" RUN_DIR="$rd" SUITE_GROUP=default N_SPECS="$n" \
        RUNS_PER_GPU="$k" SLURM_ARRAY_TASK_ID="$task" SLURM_ARRAY_JOB_ID=7 \
        bash cluster/suite_array.sbatch >"$rd/out" 2>&1 || rc=$?
    if [[ "$rc" -ne "$want" ]]; then
        echo "FAIL [suite_slice_$name]: rc=$rc, expected $want"; sed 's/^/    /' "$rd/out"; fail=1
    fi
    if [[ "$want" -ne 0 ]] && ! grep -q "N_SPECS=${n}" "$rd/out"; then
        echo "FAIL [suite_slice_$name]: message does not name N_SPECS"; fail=1
    fi
}
suite_task 0 normal      1 4 4 2    # slice 2..3 starts both runners
suite_task 0 empty       3 4 4 2    # slice 6..7 beyond the 4 rows
suite_task 0 skipdone    1 4 4 2    # runner 3 prints a skip line and exits 0
suite_task 1 nspecs_zero 0 0 4 2    # non-empty slice, N_SPECS=0
suite_task 1 nspecs_low  1 3 4 2    # slice 2..3, only index 2 < N_SPECS
suite_task 0 partial     1 3 3 2    # last slice holds one row, N_SPECS = rows
grep -q "skip\] line 3" "$tmp/suite_skipdone/logs/run-7_idx3.log" \
    || { echo "FAIL [suite_slice_skipdone]: runner was not started"; fail=1; }

# eval_x.sbatch (#156): a fake uv logs every call and prints the eval call one
# argument per bracket; the globs reach the script unexpanded (all lines of
# EX_RESULTS), a bad EX_MODE or EX_RESULTS stops before any uv call.
mkdir -p "$tmp/ex/bin" "$tmp/ex/home"
printf '#!/usr/bin/env bash\necho "$*" >>"%s"\n[[ "$*" == *eval_cncp* ]] || exit 0\nprintf "[%%s]" "$@"; echo\n' \
    "$tmp/ex/uv.log" >"$tmp/ex/bin/uv"
chmod +x "$tmp/ex/bin/uv"
eval_x_job() {  # eval_x_job <env assignment>...: runs the sbatch, prints its output
    rm -f "$tmp/ex/uv.log"
    env HOME="$tmp/ex/home" PATH="$tmp/ex/bin:$PATH" RUN_DIR="$tmp/ex/run" "$@" \
        bash cluster/eval_x.sbatch 2>&1
}
out="$(eval_x_job EX_MODE=ablation "EX_RESULTS=cluster/*.sh  cluster/*.sbatch" EX_DATA_PATH=d.txt)"
check eval_x_job_ablation "$out" "[experiments/eval_cncp_ablation.py][--results][cluster/*.sh][cluster/*.sbatch][--out][$tmp/ex/run/results/cncp_ablation.csv][--data-path][d.txt]"
out="$(eval_x_job EX_MODE=ablation "EX_RESULTS=cluster/*.sh")"
check eval_x_job_ablation_nodata "$out" "[experiments/eval_cncp_ablation.py][--results][cluster/*.sh][--out][$tmp/ex/run/results/cncp_ablation.csv]"
grep -q -- "--data-path" <<<"$out" && { echo "FAIL [eval_x_job_ablation_nodata]: --data-path without EX_DATA_PATH"; fail=1; }
out="$(eval_x_job EX_MODE=gains "EX_RESULTS=cluster/*.sh
cluster/*.sbatch")"
check eval_x_job_gains "$out" "[experiments/eval_cncp_gains.py][--results][cluster/*.sh][cluster/*.sbatch][--out][$tmp/ex/run/results/cncp_gains.csv][--paths][$tmp/ex/run/results/cncp_paths.csv]"
for bad in "EX_MODE=bogus EX_RESULTS=x.json" "EX_MODE=gains EX_RESULTS="; do
    # shellcheck disable=SC2086  # the assignments split on the space on purpose
    if out="$(eval_x_job $bad)" || [[ -e "$tmp/ex/uv.log" ]] || ! grep -q FATAL <<<"$out"; then
        echo "FAIL [eval_x_job]: '$bad' ran, called uv, exited 0 or printed no FATAL"; fail=1
    fi
done

if [[ $fail -eq 0 ]]; then echo "launchers: OK"; else exit 1; fi
