#!/usr/bin/env bash
# cluster/_common.sh -- shared SLURM array boilerplate for the benchmark engine.
#
# Sourceable functions that factor the four blocks copy-pasted across the legacy
# array sbatch scripts (benchmark_array.sbatch, backfill_array.sbatch,
# backfill_v3_ncp.sbatch, mujoco_array.sbatch):
#
#   cc_setup_tmpdir       TMPDIR/TMP/TEMP -> $HOME/tmp (virtual nodes have a
#                         read-only /tmp); prepend ~/.local/bin to PATH.
#   cc_setup_cuda_ld      discover the pip-installed CUDA libs under the venv's
#                         nvidia/*/lib and export LD_LIBRARY_PATH (TF 2.15 does
#                         not auto-find them on all cluster distros).
#   cc_setup_tf_env K     TF env (GPU memory growth, log level, unbuffered) and
#                         the per-process CPU-thread split for K runs/GPU.
#   cc_start_gpu_logger   background nvidia-smi CSV utilization logger.
#   cc_launch_bundle ...  the RUNS_PER_GPU parallel-launch loop -- THE bundling
#                         contract (see below).
#
# Designed for `source cluster/_common.sh` from an sbatch script that runs with
# CWD = repo root under `set -euo pipefail`.

# --- 1. temp dirs ------------------------------------------------------------
# Virtual nodes (hostnames i-*) have a non-writable /tmp -- redirect all temp
# paths into $HOME (same workaround as arc-jax-rl train.sbatch).
cc_setup_tmpdir() {
    export TMPDIR="$HOME/tmp"
    export TEST_TMPDIR="$HOME/tmp"
    export TMP="$HOME/tmp"
    export TEMP="$HOME/tmp"
    mkdir -p "$TMPDIR"
    export PATH="$HOME/.local/bin:$PATH"
}

# --- 2. CUDA lib discovery ---------------------------------------------------
# tensorflow[and-cuda] installs CUDA libs under
# .venv/lib/python*/site-packages/nvidia/*/lib/. TF 2.15 does not auto-discover
# these without LD_LIBRARY_PATH on all cluster distros.
cc_setup_cuda_ld() {
    local venv_site
    venv_site="$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
    if [[ -d "${venv_site}/nvidia" ]]; then
        local cuda_ld
        cuda_ld="$(find "${venv_site}/nvidia" -maxdepth 2 -name 'lib' -type d 2>/dev/null | tr '\n' ':')"
        export LD_LIBRARY_PATH="${cuda_ld}${LD_LIBRARY_PATH:-}"
    fi
}

# --- 3. TF env + per-process CPU-thread split --------------------------------
# Splits the allocated CPUs across the RUNS_PER_GPU concurrent runs so the
# per-process TF thread pools do not oversubscribe the cores. Exports
# CC_THREADS_PER_RUN for the task banner.
cc_setup_tf_env() {
    local runs_per_gpu="${1:?cc_setup_tf_env: runs_per_gpu (K) required}"
    # Let multiple runs share one GPU: each process allocates only what it needs
    # (~1.5 GB) instead of pre-grabbing all of VRAM.
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_CPP_MIN_LOG_LEVEL=1
    # Unbuffered stdout/stderr so each run's progress streams live into its log.
    export PYTHONUNBUFFERED=1

    local cpus_total threads
    cpus_total="${SLURM_CPUS_PER_TASK:-24}"
    threads=$(( cpus_total / runs_per_gpu ))
    [[ "${threads}" -lt 1 ]] && threads=1
    export OMP_NUM_THREADS="${threads}"
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS="${threads}"
    CC_THREADS_PER_RUN="${threads}"
}

# --- 4. background GPU utilization logger ------------------------------------
# Writes a CSV (30s polling) next to the SLURM logs.
cc_start_gpu_logger() {
    local job_name="${1:-${SLURM_JOB_NAME:-thesis-benchmark}}"
    local gpu_idx="${CUDA_VISIBLE_DEVICES:-0}"
    local gpu_log="outputs/slurm/${job_name}-${SLURM_JOB_ID}.gpu.csv"
    nohup nvidia-smi -i "${gpu_idx}" \
      --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total \
      --format=csv -l 30 > "${gpu_log}" 2>&1 &
}

# --- 5. the RUNS_PER_GPU parallel-launch loop (the bundling contract) --------
# usage: cc_launch_bundle <runner-prefix-token>...
#   env in: RUNS_PER_GPU, N_SPECS, SLURM_ARRAY_TASK_ID,
#           RUN_EXTRA_ARGS[] (optional forwarded runner args).
#
# Array task t owns the contiguous block of spec indices
#   [t*K, ..., t*K + K-1]   (K = RUNS_PER_GPU), capped at N_SPECS,
# launched in parallel on the single allocated GPU as
#   <runner-prefix> --index IDX <RUN_EXTRA_ARGS...>
# Waits for all; returns 1 if any run failed. This reproduces the index
# bundling of legacy benchmark_array.sbatch byte-for-byte; only the runner
# prefix differs (run_campaign.py --config <cfg> vs run_benchmark.py).
cc_launch_bundle() {
    local runner=( "$@" )
    local k="${RUNS_PER_GPU:?cc_launch_bundle: RUNS_PER_GPU required}"
    local n="${N_SPECS:?cc_launch_bundle: N_SPECS required}"
    local task="${SLURM_ARRAY_TASK_ID:?cc_launch_bundle: SLURM_ARRAY_TASK_ID required}"

    # Forwarded runner args (e.g. --regimes noise). Tolerate unset under set -u
    # and drop the lone empty-string placeholder it would otherwise inject.
    local extra=( "${RUN_EXTRA_ARGS[@]:-}" )
    [[ ${#extra[@]} -eq 1 && -z "${extra[0]}" ]] && extra=()

    local start=$(( task * k ))
    echo "=== task ${task} on $(hostname) -- $(date -u +%FT%TZ) ==="
    echo "    indices ${start}..$(( start + k - 1 )) (cap ${n}), ${CC_THREADS_PER_RUN:-?} threads/run"

    local pids=() off idx run_log
    for off in $(seq 0 $(( k - 1 ))); do
        idx=$(( start + off ))
        [[ "${idx}" -ge "${n}" ]] && break
        run_log="outputs/slurm/run-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_idx${idx}.log"
        echo "    -> launch index ${idx} -> ${run_log}"
        "${runner[@]}" --index "${idx}" "${extra[@]}" > "${run_log}" 2>&1 &
        pids+=("$!")
    done

    # Wait for all runs; surface a non-zero exit if any of them failed.
    local rc=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || rc=1
    done
    echo "=== done (rc=${rc}) -- $(date -u +%FT%TZ) ==="
    return "${rc}"
}
