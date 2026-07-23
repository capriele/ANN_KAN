#!/bin/bash

# Risultati test simbolici
# python3 analyze_result.py --results-dir ./results_auv_symbolic --model-kind kan_koopman_orig --block-size 10 --experiment-name AUV
# python3 analyze_result.py --results-dir ./results_auv_symbolic --model-kind kan_koopman_smooth --block-size 10 --experiment-name AUV
# python3 analyze_result.py --results-dir ./results_auv_symbolic --model-kind kan_koopman_wide --block-size 10 --experiment-name AUV

# # Risultati datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind chebyshev_kan --block-size 10 --experiment-name datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind fractional_kan --block-size 10 --experiment-name datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan --block-size 10 --experiment-name datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan_koopman --block-size 10 --experiment-name datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind ann --block-size 10 --experiment-name datasetRHF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind koopman --block-size 10 --experiment-name datasetRHF5Affine

# # Risultati datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind chebyshev_kan --block-size 10 --experiment-name datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind fractional_kan --block-size 10 --experiment-name datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan --block-size 10 --experiment-name datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan_koopman --block-size 10 --experiment-name datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind ann --block-size 10 --experiment-name datasetSilverNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind koopman --block-size 10 --experiment-name datasetSilverNLF5Affine

# # Risultati NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind chebyshev_kan --block-size 10 --experiment-name NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind fractional_kan --block-size 10 --experiment-name NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan --block-size 10 --experiment-name NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind kan_koopman --block-size 10 --experiment-name NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind ann --block-size 10 --experiment-name NLTankNLF5Affine
# python3 analyze_result.py --results-dir ./results_ann_qlpv --model-kind koopman --block-size 10 --experiment-name NLTankNLF5Affine

# Risultati AUV (High Stress)
python3 analyze_result.py --results-dir ./results_auv_model --model-kind chebyshev_kan --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model --model-kind fractional_kan --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind kan --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model_equal_parameters --model-kind kan --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_equal_parameters2 --model-kind ann --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_equal_parameters2 --model-kind koopman --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model --model-kind ann --block-size 10 --experiment-name AUV
python3 analyze_result.py --results-dir ./results_auv_model --model-kind koopman --block-size 10 --experiment-name AUV

# Risultati AUV_DATASET (High Fidelity)
python3 analyze_result.py --results-dir ./results_auv_model --model-kind chebyshev_kan --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_auv_model --model-kind fractional_kan --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_equal_parameters2 --model-kind ann --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_equal_parameters2 --model-kind koopman --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_orig --model-kind ann --block-size 10 --experiment-name AUV_DATASET
python3 analyze_result.py --results-dir ./results_orig --model-kind koopman --block-size 10 --experiment-name AUV_DATASET

# Risultati AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind chebyshev_kan --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind fractional_kan --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model_tmp --model-kind kan --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model_parameters2 --model-kind kan --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model --model-kind kan_koopman --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_equal_parameters3 --model-kind ann --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_equal_parameters --model-kind koopman --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model --model-kind ann --block-size 10 --experiment-name AUV_DATASET2
python3 analyze_result.py --results-dir ./results_auv_model --model-kind koopman --block-size 10 --experiment-name AUV_DATASET2

# Generate latex table
python3 generate_latex_table.py mean_std.txt table_results.tex