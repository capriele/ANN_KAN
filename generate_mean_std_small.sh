#!/bin/bash

## BIG PARAMETERS

# Risultati datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan_koopman --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind ann --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind koopman --block-size 10 --experiment-name datasetRHF5Affine

# Risultati datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan_koopman --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind ann --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind koopman --block-size 10 --experiment-name datasetSilverNLF5Affine

# Risultati NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind kan_koopman --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind ann --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_orig --model-kind koopman --block-size 10 --experiment-name NLTankNLF5Affine

# Risultati AUV
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind kan --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV

# Risultati AUV_DATASET
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV_DATASET

# Risultati AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind kan --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV_DATASET2

## MEDIUM PARAMETERS
# Risultati datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name datasetRHF5Affine

# Risultati datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name datasetSilverNLF5Affine

# Risultati NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name NLTankNLF5Affine

# Risultati AUV
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name AUV

# Risultati AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name AUV_DATASET

# Risultati AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind ann --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_medium --model-kind koopman --block-size 10 --experiment-name AUV_DATASET2

## SMALL PARAMETERS

# Risultati datasetRHF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan --block-size 10 --experiment-name datasetRHF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan_koopman --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name datasetRHF5Affine

# Risultati datasetSilverNLF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan --block-size 10 --experiment-name datasetSilverNLF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan_koopman --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name datasetSilverNLF5Affine

# Risultati NLTankNLF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan --block-size 10 --experiment-name NLTankNLF5Affine
#python3 analyze_result.py --results-dir ./results_small --model-kind kan_koopman --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name NLTankNLF5Affine

# Risultati AUV
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name AUV

# Risultati AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name AUV_DATASET

# Risultati AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_small --model-kind ann --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_small --model-kind koopman --block-size 10 --experiment-name AUV_DATASET2


## VERY SMALL PARAMETERS

# Risultati datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name datasetRHF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name datasetRHF5Affine

# Risultati datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name datasetSilverNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name datasetSilverNLF5Affine

# Risultati NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name NLTankNLF5Affine
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name NLTankNLF5Affine

# Risultati AUV
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name AUV

# Risultati AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name AUV_DATASET

# Risultati AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind ann --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_standard_very_small --model-kind koopman --block-size 10 --experiment-name AUV_DATASET2

# Generate latex table
python3 generate_latex_table.py mean_std_small.txt table_results_small.tex