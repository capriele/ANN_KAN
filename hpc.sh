#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Simple batch submitter.
#
# Usage examples:
#   ./hpc.sh --results-root results_auv_big AUV_DATASET2 "7 10 1 6 15 1 0 40 3" kan chebyshev fractional
#   ./hpc.sh NLTankNLF5Affine_CAPACITY "5 1 1 6 10 1 0 40 3" kan_cm chebyshev_cm fractional_cm
#   ./hpc.sh --cancel-current AUV_DATASET2 "7 10 1 6 15 1 0 0" chebyshev fractional
#
# Parameter format:
#   "FIT DATASET NONLIN STATE STRIDE AFFINE REG N_NEURONS N_LAYERS"
# The legacy eight-value format remains supported.
#
# Environment overrides:
#   RESULTS_ROOT_PATH=results_auv_big PARTITION=gprod_gssi CPUS=64 MEM=120GB ./hpc.sh ...
# -----------------------------------------------------------------------------

PARTITION="${PARTITION:-gprod_gssi}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
CPUS="${CPUS:-32}"
MEM="${MEM:-32GB}"
TIME="${TIME:-}"
GRES="${GRES:-}"
DEPEND_CHAIN="${DEPEND_CHAIN:-0}"
RESULTS_ROOT_PATH="${RESULTS_ROOT_PATH:-results_auv_big}"

cancel_current_jobs() {
    local user jobs
    user="$(whoami)"
    jobs="$(squeue -u "$user" -h -o "%A" || true)"
    if [[ -z "$jobs" ]]; then
        echo "No jobs found for user $user."
        return 0
    fi
    echo "Cancelling jobs for $user: $jobs"
    for job in $jobs; do
        scancel "$job"
    done
}

model_code() {
    case "$1" in
        ann) echo 0 ;;
        kan|spline) echo 1 ;;
        koopman) echo 2 ;;
        kan_koopman) echo 3 ;;
        mamba) echo 4 ;;
        mixed) echo 5 ;;
        chebyshev|cheby|chebyshev_kan) echo 6 ;;
        fractional|jacobi|fractional_kan) echo 7 ;;
        kan_cm|spline_cm|capacity_kan) echo 8 ;;
        kan_koopman_cm|capacity_kan_koopman) echo 9 ;;
        chebyshev_cm|cheby_cm|capacity_chebyshev) echo 10 ;;
        fractional_cm|jacobi_cm|capacity_fractional) echo 11 ;;
        *) echo "ERROR: unknown model '$1'" >&2; return 2 ;;
    esac
}

submit_one() {
    local exp_prefix="$1"
    local base_params="$2"
    local model_name="$3"
    local results_root_path="$4"
    local code exp_name params job_name jobid


    code="$(model_code "$model_name")"
    exp_name="${exp_prefix}"
    params="${base_params} ${code}"
    job_name="$(slurm_job_name "$exp_name" "$model_name")"
    echo "exp_prefix: $exp_prefix"
    echo "base_params: $base_params"
    echo "model_name: $model_name"
    echo "code: $code"
    echo "exp_name: $exp_name"
    echo "job_name: $job_name"
    echo "params: $params"
    echo "results_root_path: $results_root_path"

    local sbatch_args=(--job-name="$job_name" --partition="$PARTITION" -N "$NODES" --ntasks="$NTASKS" --cpus-per-task="$CPUS" --mem="$MEM")
    [[ -n "$TIME" ]] && sbatch_args+=(--time="$TIME")
    [[ -n "$GRES" ]] && sbatch_args+=(--gres="$GRES")
    if [[ "$DEPEND_CHAIN" == "1" && -n "${LAST_JOB_ID:-}" ]]; then
        sbatch_args+=(--dependency="afterok:${LAST_JOB_ID}")
    fi

    jobid="$(sbatch "${sbatch_args[@]}" hpc_cluster_run.sh "$exp_name" "$params" "$job_name" "$results_root_path" | awk '{print $4}')"
    #bash hpc_cluster_run.sh "$exp_name" "$params" "$job_name"
    #jobid="0"
    LAST_JOB_ID="$jobid"
    echo "Submitted $model_name as job $jobid ($job_name) with params: $params"
}

validate_base_params() {
    local params_string="$1"
    local values
    read -r -a values <<< "$params_string"

    if [[ ${#values[@]} -ne 8 && ${#values[@]} -ne 9 ]]; then
        echo "ERROR: expected 8 legacy values or 9 values including N_NEURONS and N_LAYERS; got ${#values[@]}" >&2
        return 2
    fi
    for value in "${values[@]}"; do
        if [[ ! "$value" =~ ^-?[0-9]+$ ]]; then
            echo "ERROR: all parameters must be integers; got '$value'" >&2
            return 2
        fi
    done
    if [[ ${#values[@]} -eq 9 && ( ${values[7]} -lt 1 || ${values[8]} -lt 1 ) ]]; then
        echo "ERROR: N_NEURONS and N_LAYERS must be greater than zero" >&2
        return 2
    fi
}

slurm_job_name() {
    local exp_name="$1"
    local model_name="$2"
    local raw="${exp_name}_${model_name}"
    local sanitized="${raw//[^[:alnum:]_.-]/_}"
    echo "${sanitized:0:128}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cancel-current)
            cancel_current_jobs
            shift
            ;;
        --results-root)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --results-root requires a path" >&2
                exit 2
            fi
            RESULTS_ROOT_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Check if at least two arguments are provided else execute default behavior
if [[ $# -lt 2 ]]; then
    for i in {1..14}; do 
        # mamba) echo 4 ;;
        # mixed) echo 5 ;;
        # kan_cm kan_koopman_cm chebyshev_cm fractional_cm
        bash hpc.sh AUV "7 7 1 6 15 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh AUV_DATASET "7 8 1 6 15 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh datasetTankNLF5Affine "5 4 0 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh HWSystemF5Affine "5 2 1 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh NLTankLinF5Affine "5 1 0 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
        bash hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0" ann kan koopman kan_koopman chebyshev fractional
    done
    exit 0
fi

exp_prefix="${1:-AUV_DATASET2}"
base_params="${2:-7 10 1 6 15 1 0 0}"
validate_base_params "$base_params"
shift $(( $# >= 2 ? 2 : $# ))

if [[ $# -eq 0 ]]; then
    set -- ann kan chebyshev fractional
fi

for model in "$@"; do
    echo "Submitting model: $model" 
    submit_one "$exp_prefix" "$base_params" "$model" "$RESULTS_ROOT_PATH"
done
