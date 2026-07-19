#!/bin/bash

for i in {1..10}; do
    echo "Esecuzione numero $i"
    ########## KAN-Koopman SYMBOLIFICATION ##########
    RESULTS_ROOT_PATH=results_auv_symbolic PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 8 1 4 2 1 0 8 5" kan_koopman
    RESULTS_ROOT_PATH=results_auv_symbolic PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 8 1 4 2 1 0 8 5" kan_koopman
    RESULTS_ROOT_PATH=results_auv_symbolic PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 8 1 4 2 1 0 8 5" kan_koopman

    ########## CLASSICAL MODELS ##########
    ## BIG 
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 7 2" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 30 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 7 2" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 30 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 7 2" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_big PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 30 3" ann koopman

    ## MEDIUM
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 18 3" koopman
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 18 3" koopman
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 18 3" koopman

    ## SMALL
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 4 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 15 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 4 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 15 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 4 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 15 3" ann koopman

    ## VERY SMALL
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 7 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetRHF5Affine "5 3 0 6 10 1 0 7 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 7 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh datasetSilverNLF5Affine "5 5 0 6 10 1 0 7 3" ann koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 7 3" kan kan_koopman
    RESULTS_ROOT_PATH=results_standard_small PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh NLTankNLF5Affine "5 1 1 6 10 1 0 7 3" ann koopman

    ########## AUV MODELS ########## 
    ## TEST AUV NORMAL
    RESULTS_ROOT_PATH=results_auv_model_tmp PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 2 1 0 7 2" kan
    RESULTS_ROOT_PATH=results_auv_model PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 2 1 0 7 2" kan_koopman
    RESULTS_ROOT_PATH=results_auv_model PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET "7 8 1 6 2 1 0 7 2" kan kan_koopman
    RESULTS_ROOT_PATH=results_auv_model_tmp PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET2 "7 10 1 6 2 1 0 7 2" kan
    RESULTS_ROOT_PATH=results_auv_model PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET2 "7 10 1 6 2 1 0 7 2" kan_koopman
    RESULTS_ROOT_PATH=results_auv_model PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 14 1 0 7 3" chebyshev fractional
    RESULTS_ROOT_PATH=results_auv_model PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET "7 8 1 6 14 1 0 7 3" chebyshev fractional
    RESULTS_ROOT_PATH=results_auv_model_tmp PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET2 "7 10 1 6 14 1 0 7 3" chebyshev fractional

    # TEST AUV MEDIUM
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 15 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 15 1 0 18 3" koopman
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 15 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 15 1 0 18 3" koopman
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET2 "7 10 1 6 15 1 0 22 3" ann
    RESULTS_ROOT_PATH=results_standard_medium PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET2 "7 10 1 6 15 1 0 18 3" koopman

done