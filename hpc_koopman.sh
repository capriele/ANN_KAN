#!/bin/bash

# rm *.err
# rm *.out

# user=$(whoami)
# for job in $(squeue -u "$user" -h -o "%A"); do
#     scancel "$job"
#     echo $job
# done

sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5Affine "5 1 1 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF2Affine "2 1 1 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5NonAffine "5 1 1 6 10 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF2NonAffine "2 1 1 6 10 0 0 0 2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF5Affine "5 1 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF2Affine "2 1 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF5NonAffine "5 1 0 6 10 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF2NonAffine "2 1 0 6 10 0 0 0 2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh HWSystemF5Affine "5 2 1 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh HWSystemF2Affine "2 2 1 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh HWSystemF5NonAffine "5 2 1 6 10 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh HWSystemF2NonAffine "2 2 1 6 10 0 0 0 2"

# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetRHF2Affine "2 3 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetRHF2NonAffine "2 3 0 6 10 0 0 0 2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetRHF5Affine "5 3 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetRHF5NonAffine "5 3 0 6 10 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetTankNLF2Affine "2 4 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetTankNLF2NonAffine "2 4 0 6 10 0 0 0 2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetTankNLF5Affine "5 4 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetTankNLF5NonAffine "5 4 0 6 10 0 0 0 2"
sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetSilverNLF2Affine "2 5 0 6 10 1 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetSilverNLF5NonAffine "5 5 0 6 10 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh datasetSilverNLF2NonAffine "2 5 0 6 10 0 0 0 2"

# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoState "5 3 0 6 10 0 1 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh RHReducedF5NONAffineGroupLassoInput "5 3 0 6 10 0 2 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh RHReducedF5NonAffine2 "5 3 0 2 2 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh RHReducedF5NonAffine4 "5 3 0 4 4 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoState "5 1 0 6 10 0 1 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinReducedF5NONAffineGroupLassoInput "5 1 0 6 10 0 2 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF5NonAffine2 "5 1 0 2 2 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankLinF5NonAffine4 "5 1 0 4 4 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoState "5 4 0 6 10 0 1 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh DSTankReducedF5NONAffineGroupLassoInput "5 4 0 6 10 0 2 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh DSTankReducedF5NonAffine2 "5 4 0 2 2 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh DSTankReducedF5NonAffine4 "5 4 0 4 4 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh DSTankReducedF5NonAffine24 "5 4 0 2 4 0 0 0 2"

# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoState "5 1 1 6 10 0 1 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLReducedF5NONAffineGroupLassoInput "5 1 1 6 10 0 2 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5NonAffine25 "5 1 1 2 5 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5NonAffine3 "5 1 1 3 3 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5NonAffine5 "5 1 1 5 5 0 0 0 2"
# sbatch --partition=gprod_gssi -N 1 --ntasks=1 --cpus-per-task=32 --mem=64GB hpc_cluster_run.sh NLTankNLF5NonAffine35 "5 1 1 3 5 0 0 0 2"
