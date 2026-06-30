#!/usr/bin/env bash
# Backfill for the v3 ncp heavy runs (ode_unfolds=24 and batch_time=64 on the
# ncp wiring) that OOMed in the main v3 sweep (job 482023). The main sweep ran
# 6 runs/GPU at 64G; the ncp + 4x-intervention computation graph peaked at ~67G
# and SLURM killed those tasks (OUT_OF_MEMORY). This backfill packs FEWER runs
# per GPU and DOUBLES the memory so each heavy graph has ample headroom.
#
# Unlike benchmark_array.sbatch (contiguous --index block) this uses an EXPLICIT
# v3 spec-index list: the missing runs are scattered (ncp halves of the solver
# and horizon blocks), so a contiguous block cannot target them. Each entry is
# `--profile v3 --index <i>`; the runner writes to results/runs_v3/ with the
# correct _unfolds24 / _bt64 filename suffix.
#
# Submit with (RUNS_PER_GPU=2 => ceil(88/2)=44 array tasks):
#   ssh datalab "cd thesis-benchmark && mkdir -p outputs/slurm && \
#     sbatch --array=0-43%8 --export=ALL,RUNS_PER_GPU=2 cluster/backfill_v3_ncp.sbatch"
#
#SBATCH --job-name=thesis-bf-v3ncp
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail

# Virtual nodes have a non-writable /tmp — redirect temp paths into $HOME.
export TMPDIR="$HOME/tmp"
export TEST_TMPDIR="$HOME/tmp"
export TMP="$HOME/tmp"
export TEMP="$HOME/tmp"
mkdir -p "$TMPDIR"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# TF 2.15 needs LD_LIBRARY_PATH to find the pip-installed CUDA libs.
VENV_SITE="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [[ -d "${VENV_SITE}/nvidia" ]]; then
    CUDA_LD="$(find "${VENV_SITE}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
    export LD_LIBRARY_PATH="${CUDA_LD}${LD_LIBRARY_PATH:-}"
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=1

# Fewer runs/GPU than the main sweep (6 -> OOM at 64G); 2 at 128G gives ~64 GB
# headroom per heavy ncp graph.
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
CPUS_TOTAL="${SLURM_CPUS_PER_TASK:-12}"
THREADS_PER_RUN=$(( CPUS_TOTAL / RUNS_PER_GPU ))
[[ "${THREADS_PER_RUN}" -lt 1 ]] && THREADS_PER_RUN=1
export OMP_NUM_THREADS="${THREADS_PER_RUN}"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS="${THREADS_PER_RUN}"

mkdir -p outputs/slurm results/runs_v3

# v3 spec indices of the missing ncp heavy runs (ncp + unfolds=24 or bt=64).
# 28 solver-block (unfolds24) + 60 horizon-block (bt64) = 88 runs.
INDICES=(
  513 515 516 519 522 523 526 528 530 531 534 536 539
  570 572 574 576 577 580 584 585 587 589 590 593 594 597 599
  630 631 632 633 634 635 636 637 638 639 640 641 642 643 644
  645 646 647 648 649 650 651 652 653 654 655 656 657 658 659
  690 691 692 693 694 695 696 697 698 699 700 701 702 703 704
  705 706 707 708 709 710 711 712 713 714 715 716 717 718 719
)

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_GPU ))
echo "=== v3-ncp backfill task ${SLURM_ARRAY_TASK_ID} on $(hostname) — $(date -u +%FT%TZ) ==="
echo "    entries ${START}..$(( START + RUNS_PER_GPU - 1 )) of ${#INDICES[@]}, ${THREADS_PER_RUN} threads/run, mem 128G"

pids=()
for off in $(seq 0 $(( RUNS_PER_GPU - 1 ))); do
    i=$(( START + off ))
    [[ "${i}" -ge "${#INDICES[@]}" ]] && break
    IDX="${INDICES[$i]}"
    RUN_LOG="outputs/slurm/bf-v3ncp-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_idx${IDX}.log"
    echo "    -> --profile v3 --index ${IDX}  (log ${RUN_LOG})"
    uv run python experiments/run_benchmark.py --profile v3 --index "${IDX}" > "${RUN_LOG}" 2>&1 &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
done
echo "=== done (rc=${rc}) — $(date -u +%FT%TZ) ==="
exit "${rc}"
