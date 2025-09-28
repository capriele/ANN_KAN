#!/bin/bash

rm *.err
rm *.out

user=$(whoami)
for job in $(squeue -u "$user" -h -o "%A"); do
    scancel "$job"
    echo $job
done

# Change state size (Just Koopman)
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_3_3 "5 1 1 3 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_6_6 "5 1 1 6 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_9_9 "5 1 1 9 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_12_12 "5 1 1 12 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_15_15 "5 1 1 15 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine_18_18 "5 1 1 18 10 1 0 0 5 0.0"

# Change alpha (0 => Koopman, 1 => Normal)
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.1"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.3"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.4"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.5"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.6"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.7"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.8"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 0.9"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run_mixed.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 5 1.0"


sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.0"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.1"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.3"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.4"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.5"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.6"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.7"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.8"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 0.9"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=64 --mem=120GB hpc_cluster_run.sh AUV_DATASET "7 8 1 6 15 1 0 0 5 1.0"