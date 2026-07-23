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

    # Equal parameters
    RESULTS_ROOT_PATH=results_equal_parameters PARTITION=gprod_gssi CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 14 1 0 19 3" ann
    RESULTS_ROOT_PATH=results_equal_parameters_tmp PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 13 1 0 12 3" koopman
    RESULTS_ROOT_PATH=results_equal_parameters2 PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 13 1 0 17 3" ann
    RESULTS_ROOT_PATH=results_equal_parameters2 PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV "7 7 1 6 13 1 0 11 3" koopman

    RESULTS_ROOT_PATH=results_equal_parameters PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 13 1 0 16 3" ann
    RESULTS_ROOT_PATH=results_equal_parameters PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 10 1 0 11 3" koopman
    RESULTS_ROOT_PATH=results_equal_parameters2 PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 17 1 0 17 3" ann
    RESULTS_ROOT_PATH=results_equal_parameters2 PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET "7 8 1 6 13 1 0 11 3" koopman

    RESULTS_ROOT_PATH=results_auv_model_equal_parameters PARTITION=gprod_gssi CPUS=64 MEM=64GB ./hpc.sh AUV_DATASET2 "7 10 1 6 2 1 0 4 3" kan
    RESULTS_ROOT_PATH=results_auv_model_parameters2 PARTITION=lprod CPUS=36 MEM=120GB ./hpc.sh AUV_DATASET2 "7 10 1 6 2 1 0 4 3" kan
    RESULTS_ROOT_PATH=results_equal_parameters3 PARTITION=gprod_gssi CPUS=36 MEM=32GB ./hpc.sh AUV_DATASET2 "7 10 1 6 21 1 0 14 2" ann
    RESULTS_ROOT_PATH=results_equal_parameters PARTITION=lprod CPUS=32 MEM=32GB ./hpc.sh AUV_DATASET2 "7 10 1 6 13 1 0 11 3" koopman
done

# | PARTITION  | nodes=NODES | time=TIMELIMIT  | size=JOB_SIZE   | features=AVAIL_FEATURES | gres=GRES       |
# |-debug------|-nodes=4-----|-time=1:00:00----|-size=1-infinite-|-features=rack37---------|-gres=(null)-----|
# | serial     | nodes=72    | time=4-00:00:00 | size=1          | features=rack41         | gres=(null)     |
# | serial     | nodes=61    | time=4-00:00:00 | size=1          | features=rack37         | gres=(null)     |
# | serial     | nodes=5     | time=4-00:00:00 | size=1          | features=(null)         | gres=(null)     |
# | serial     | nodes=72    | time=4-00:00:00 | size=1          | features=rack39         | gres=(null)     |
# | serial     | nodes=72    | time=4-00:00:00 | size=1          | features=rack40         | gres=(null)     |
# | serial     | nodes=36    | time=4-00:00:00 | size=1          | features=rack38         | gres=(null)     |
# | lprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack41         | gres=(null)     |
# | lprod      | nodes=61    | time=4-00:00:00 | size=1-infinite | features=rack37         | gres=(null)     |
# | lprod      | nodes=3     | time=4-00:00:00 | size=1-infinite | features=(null)         | gres=(null)     |
# | lprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack39         | gres=(null)     |
# | lprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack40         | gres=(null)     |
# | lprod      | nodes=36    | time=4-00:00:00 | size=1-infinite | features=rack38         | gres=(null)     |
# | bprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack41         | gres=(null)     |
# | bprod      | nodes=61    | time=4-00:00:00 | size=1-infinite | features=rack37         | gres=(null)     |
# | bprod      | nodes=3     | time=4-00:00:00 | size=1-infinite | features=(null)         | gres=(null)     |
# | bprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack39         | gres=(null)     |
# | bprod      | nodes=72    | time=4-00:00:00 | size=1-infinite | features=rack40         | gres=(null)     |
# | bprod      | nodes=36    | time=4-00:00:00 | size=1-infinite | features=rack38         | gres=(null)     |
# | gprod      | nodes=5     | time=6-00:00:00 | size=1-infinite | features=(null)         | gres=gpu:h100:4 |
# | gprod      | nodes=1     | time=6-00:00:00 | size=1-infinite | features=(null)         | gres=gpu:a100:4 |
# | gprod_gssi | nodes=1     | time=4-00:00:00 | size=1-infinite | features=(null)         | gres=gpu:a100:8 |
# | gprod_gssi | nodes=2     | time=4-00:00:00 | size=1-infinite | features=(null)         | gres=gpu:l40:1  |
# | gprod_gssi | nodes=3     | time=4-00:00:00 | size=1-infinite | features=(null)         | gres=gpu:h100:4 |

