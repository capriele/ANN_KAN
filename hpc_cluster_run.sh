#!/bin/bash
#SBATCH --job-name=ann
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=96:00:00
#-SBATCH --output=slurm-%j.out
#-SBATCH --error=slurm-%j.err
#-SBATCH --time=96:00:00
#-SBATCH --nodes=1
#-SBATCH --ntasks=1
#-SBATCH --cpus-per-task=36
#-SBATCH --mem=100GB
#-SBATCH --partition=serial
# --partition=lprod

# | Name       | Priority | GraceTime | Flags                                   | UsageFactor | GrpTRES       | MaxTRES | MaxTRESPerNode | MaxTRESMins | MaxWall | MaxTRESPU | MinTRES    |
# |------------|----------|-----------|-----------------------------------------|-------------|---------------|---------|----------------|-------------|---------|-----------|------------|
# | normal     | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           |            |
# | serial_pa+ | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes   | 1.000.000   | cpu=36,mem=1+ | cpu=36  |                |             |         |           |            |
# | lprod_part | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes,+ | 1.000.000   | cpu=64,mem=3+ |         |                |             |         |           | cpu=1      |
# | bprod_part | 0        | 00:00:00  | cluster DenyOnLimit,OverPartQOS         | 1.000.000   | cpu=256,mem=+ | cpu=256 | cpu=36         | cpu=256     | 100     | 120       | cpu=65     |
# | debug_part | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes   | 1.000.000   | cpu=4,mem=12+ |         |                |             | 2       | 8         |            |
# | default_q+ | 1000     | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             | 1000    | 1200      |            |
# | staff_gpu+ | 1100     | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           |            |
# | gprod_part | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           | gres/gpu=0 |
# | high_job   | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             | 1000    | 1200      |            |


# User specific aliases and functions
export HOME="/home/petruccia"
export PATH="$HOME/bin/bin:$HOME/bin:$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export PATH="$HOME/local/bin:$PATH"
export PATH="$HOME/local/python3.11/bin/:$PATH"

# GCC 15.1.0
export PATH="$HOME/local/gcc-15.1.0/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$LIBRARY_PATH"
export C_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:$CPLUS_INCLUDE_PATH"

# TMUX
export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/python3.11/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$HOME/local/include:$CPATH
export LIBRARY_PATH=$HOME/local/python3.11/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:$LIBRARY_PATH

# Assign default values to $9 and ${10} if they are not provided
arg9="${9:-0}"
arg10="${10:-0}"

##./batchRun.sh $1 $2 $3 $4 $5 $6 $7 $8 $arg9 $arg10

# Extract the last argument from the input string
last_arg="$(echo "${2}" | grep -oE '[^ ]+$')"
echo "LAST ARG: $last_arg"
mkdir -p dumps
exp_name="${1}"

# Function to generate a new experiment name if the directory exists
generate_new_exp_name() {
    local original_name="$1"
    local model_type="$2"
    local new_name="$original_name"
    local counter=1

    while [ -d "results/${model_type}/${new_name}" ]; do
        new_name="${original_name}_${counter}"
        ((counter++))
    done

    #echo "$original_name"
    echo "$new_name"
}

# Determine model type and create folders
if [ "$last_arg" = "1" ]; then
    model_type="kan"
    # Check if directory exists and generate new name if needed
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"

elif [ "$last_arg" = "2" ]; then
    model_type="koopman"
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"

elif [ "$last_arg" = "3" ]; then
    model_type="kan_koopman"
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"

elif [ "$last_arg" = "4" ]; then
    model_type="mamba"
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"

elif [ "$last_arg" = "5" ]; then
    model_type="mixed"
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"

else
    model_type="ann"
    if [ -d "results/${model_type}/${exp_name}" ]; then
        exp_name=$(generate_new_exp_name "$exp_name" "$model_type")
    fi
    mkdir -p "results/${model_type}/${exp_name}"
    mkdir -p "results/${model_type}/${exp_name}/open_loop"
    mkdir -p "results/${model_type}/${exp_name}/closed_loop"
    filename="results/${model_type}/${exp_name}/log.txt"
fi

echo "Using experiment name: $exp_name"
echo "Log will be written to: $filename"


# Pick method based on last_arg
case "$last_arg" in
    1) method="kan" ;;
    2) method="koopman" ;;
    3) method="kan_koopman" ;;
    4) method="mamba" ;;
    5) method="mixed" ;;
    *) method="ann" ;;
esac

for arg in "$@"; do
  echo "Arg $n: $arg"
  ((n++))
done
echo ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10}
i=1
rm "$filename"
touch "$filename"
python3 -u main.py $i ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${arg9} ${arg10} ${exp_name} | tee -a "$filename"

base_dir="results/${method}/${exp_name}"

# Ensure target dirs exist
mkdir -p "${base_dir}/open_loop" "${base_dir}/closed_loop" "dumps/${method}"

# ---- Move dump file ----
for file in dumps/dump_${exp_name}.mat; do
    if [ -f "$file" ]; then
        mv "$file" "${base_dir}/dump.mat"
        break  # remove this line if you want to move *all* dumps
    fi
done

# ---- Move model file ----
for file in dumps/model_${exp_name}.mat; do
    if [ -f "$file" ]; then
        mv "$file" "${base_dir}/model.mat"
        break  # remove this line if you want to move *all* dumps
    fi
done

# ---- Move open loop plots ----
for file in ./open_loop_*.png; do
    [ -f "$file" ] && mv "$file" "${base_dir}/open_loop/"
done

# ---- Move closed loop plots ----
for file in ./closed_loop_*.png; do
    [ -f "$file" ] && mv "$file" "${base_dir}/closed_loop/"
done

# ---- Move slurm outputs (for all methods) ----
# for file in ./slurm-*.err; do
#     [ -f "$file" ] && mv "$file" "${base_dir}/job.err"
# done
# for file in ./slurm-*.out; do
#     [ -f "$file" ] && mv "$file" "${base_dir}/job.out"
# done