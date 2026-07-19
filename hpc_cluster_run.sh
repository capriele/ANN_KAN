#!/bin/bash
# Default for direct sbatch calls; hpc.sh overrides this with --job-name.
#SBATCH --job-name=pykan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=96:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# One-job launcher for the project.
#
# Backward compatible call:
#   sbatch hpc_cluster_run.sh EXP_NAME "fit dataset nonlin state stride affine reg neurons layers model_code" [JOB_NAME] [RESULTS_ROOT_PATH]
#
# Example:
#   sbatch --job-name=AUV_DATASET2_chebyshev --partition=gprod_gssi -N 1 \
#     --ntasks=1 --cpus-per-task=64 --mem=120GB \
#     hpc_cluster_run.sh AUV_DATASET2 "7 10 1 6 15 1 0 40 3 6" AUV_DATASET2_chebyshev results_auv_big
#
# The last value in PARAMS is the model code. main.py receives a clean set
# of named arguments, so the Python side no longer depends on positional magic.
# -----------------------------------------------------------------------------

# User environment. Keep these lines if your cluster relies on this local stack.
export HOME="${HOME:-/home/petruccia}"
export PATH="$HOME/bin/bin:$HOME/bin:$HOME/.pyenv/bin:$HOME/local/bin:$HOME/local/python3.11-sqlite/bin:$PATH"
if command -v pyenv >/dev/null 2>&1; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

export PATH="$HOME/local/gcc-15.1.0/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$HOME/local/python3.11-sqlite/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$HOME/local/python3.11-sqlite/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:${LIBRARY_PATH:-}"
export C_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:${CPLUS_INCLUDE_PATH:-}"
export PKG_CONFIG_PATH="$HOME/local/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPATH="$HOME/local/include:${CPATH:-}"

export PYKAN_CPU_THREADS=32
export PYKAN_TORCH_COMPILE=1 

usage() {
    cat <<'EOF'
Usage:
  hpc_cluster_run.sh EXP_NAME "FIT DATASET NONLIN STATE STRIDE AFFINE REG N_NEURONS N_LAYERS [MODEL_CODE]" [JOB_NAME] [RESULTS_ROOT_PATH]

Fields:
  FIT          fit horizon
  DATASET      dataset/system id used by main.py
  NONLIN       1/0 non-linear input characteristic
  STATE        latent state size
  STRIDE       stride length n_a
  AFFINE       1/0 affine structure
  REG          0 disabled, 1 group lasso/state reduction, 2 group lasso/no state reduction
  N_NEURONS    number of neurons per hidden layer
  N_LAYERS     number of hidden layers
  MODEL_CODE   optional legacy code: 0 ann, 1 kan, 2 koopman, 3 kan_koopman,
               4 mamba, 5 mixed, 6 chebyshev, 7 fractional,
               8/9/10/11 capacity matched variants
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

exp_name="$1"
params_string="$2"
job_name="${3:-${SLURM_JOB_NAME:-$exp_name}}"
results_root_path="${4:-${RESULTS_ROOT_PATH:-results_auv_big}}"
echo "exp_name: $exp_name"
echo "job_name: $job_name"
echo "params_string: $params_string"
echo "results_root_path: $results_root_path"
read -r -a params <<< "$params_string"

if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_NAME:-}" != "$job_name" ]] && command -v scontrol >/dev/null 2>&1; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="$job_name" || true
fi

if [[ ${#params[@]} -lt 7 ]]; then
    echo "ERROR: expected at least 7 numeric params, got: '$params_string'" >&2
    usage >&2
    exit 2
fi

fit_horizon="${params[0]}"
dataset_id="${params[1]}"
nonlinear_input="${params[2]}"
state_size="${params[3]}"
stride_len="${params[4]}"
affine_struct="${params[5]}"
regularizer_mode="${params[6]}"
n_neurons=""
n_layers=""
if [[ ${#params[@]} -ge 10 ]]; then
    n_neurons="${params[7]}"
    n_layers="${params[8]}"
    model_code="${params[9]:-0}"
elif [[ ${#params[@]} -eq 9 && ${params[7]} -gt 0 && ${params[8]} -gt 0 ]]; then
    # Extended direct call without MODEL_CODE.
    n_neurons="${params[7]}"
    n_layers="${params[8]}"
    model_code=0
else
    # Legacy format: the eighth value is reserved and the ninth is MODEL_CODE.
    model_code="${params[8]:-0}"
fi

if [[ -n "$n_neurons" && ( "$n_neurons" -lt 1 || "$n_layers" -lt 1 ) ]]; then
    echo "ERROR: N_NEURONS and N_LAYERS must be greater than zero" >&2
    exit 2
fi

model_kind_from_code() {
    case "$1" in
        1|8) echo "kan" ;;
        2) echo "koopman" ;;
        3|9) echo "kan_koopman" ;;
        4) echo "mamba" ;;
        5) echo "mixed" ;;
        6|10) echo "chebyshev_kan" ;;
        7|11) echo "fractional_kan" ;;
        0|"") echo "ann" ;;
        *) echo "ann" ;;
    esac
}

unique_exp_name() {
    local root="$1"
    local name="$2"
    local candidate="$name"
    local counter=1
    while [[ -d "$root/$candidate" ]]; do
        candidate="${name}_${counter}"
        counter=$((counter + 1))
    done
    echo "$candidate"
}

model_kind="$(model_kind_from_code "$model_code")"
results_root="$results_root_path/${model_kind}"
exp_name="$(unique_exp_name "$results_root" "$exp_name")"
base_dir="${results_root}/${exp_name}"
log_file="${base_dir}/log.txt"

mkdir -p "$base_dir/open_loop" "$base_dir/closed_loop" "dumps/${model_kind}"
: > "$log_file"

{
    echo "Experiment: $exp_name"
    echo "SLURM job name: $job_name"
    echo "SLURM job id: ${SLURM_JOB_ID:-local}"
    echo "Model kind: $model_kind"
    echo "Model code: $model_code"
    echo "Params: fit=$fit_horizon dataset=$dataset_id nonlin=$nonlinear_input state=$state_size stride=$stride_len affine=$affine_struct reg=$regularizer_mode neurons=${n_neurons:-default} layers=${n_layers:-default}"
    echo "Results: $base_dir"
    echo "Started: $(date -Is)"
} | tee -a "$log_file"

network_args=()
if [[ -n "$n_neurons" ]]; then
    network_args+=(--n-neurons "$n_neurons" --n-layers "$n_layers")
fi

python3 -u main.py \
    --fit-horizon "$fit_horizon" \
    --dataset-id "$dataset_id" \
    --nonlinear-input "$nonlinear_input" \
    --state-size "$state_size" \
    --stride-len "$stride_len" \
    --affine-struct "$affine_struct" \
    --regularizer-mode "$regularizer_mode" \
    "${network_args[@]}" \
    --model-code "$model_code" \
    --test-name "$exp_name" \
    --results-path "$results_root_path"  | tee -a "$log_file"

# Collect generated artifacts without failing if a pattern has no matches.
shopt -s nullglob
for file in "dumps/dump_${exp_name}.mat"; do 
    mv "$file" "${base_dir}/dump.mat" || true
done
for file in "dumps/model_${exp_name}.mat"; do 
    mv "$file" "${base_dir}/model.mat" || true
done
for file in ./open_loop_*.png; do 
    mv "$file" "${base_dir}/open_loop/$file" || true
done
for file in ./closed_loop_*.png; do 
    mv "$file" "${base_dir}/closed_loop/$file" || true
done
for file in ./slurm-${SLURM_JOB_ID:-}.*; do
    [[ -f "$file" ]] && cp "$file" "$base_dir/$file" || true
    [[ -f "$file" ]] && rm "$file" || true
done
shopt -u nullglob

echo "Finished: $(date -Is)" | tee -a "$log_file" 
echo "Results stored in $base_dir" | tee -a "$log_file"
