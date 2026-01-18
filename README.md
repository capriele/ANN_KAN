# ANN, KAN & Koopman for Nonlinear State-Space Models

This repository contains the implementation of various neural network architectures for learning nonlinear state-space models from data. The supported architectures include:
- **ANN**: Standard Multi-Layer Perceptrons.
- **KAN**: Kolmogorov-Arnold Networks.
- **Koopman**: Koopman models.
- **Mixed**: Hybrid architectures combining the above.

The framework is designed for system identification tasks and supports multiple dynamical systems (simulated and datasets).

## Folder Structure

### Core Scripts
- `main_mixed.py`: Main entry point for training and evaluating models. It handles argument parsing, model initialization, training loops, and validation.
- `hpc_cluster_run.sh`: SLURM script for submitting experiments to an HPC cluster. It manages environment setup and job execution.

### Model Implementation
- `AdvAutoencoder.py`: Implements the adversarial autoencoder architecture used for state-space modeling. It integrates different encoder/decoder/bridge networks.

### Dynamical Systems & Datasets
- `DynamicalSystem.py`: Base class for dynamical systems and implementation of linear systems.
- `TwoTanks.py`: Simulation of a two-tank system.
- `SpacecraftCW.py`: Simulation of spacecraft relative motion (Clohessy-Wiltshire equations).
- `AUV.py`: Simulation of an Autonomous Underwater Vehicle.
- `AUVDataset.py`: Loader for AUV datasets generated from OpenMAUVe.
- `AUVDataset2.py`: Loader for AUV datasets alternative generated from OpenMAUVe.

### Miscellaneous
- `symbolic.tex`: Example of symbolic representations generated for Koopman states.

## Requirements

- **Python version:** 3.11 (Recommended)
- **Dependencies:**
  - `torch`
  - `numpy`
  - `matplotlib`
  - `scipy`

Install dependencies using pip:

```sh
pip install -r requirements.txt
```

### Usage
To reproduce all experiments, use:

```sh
bash reproducibility.sh
```

## Notes

- The code is tested with Python 3.10.8.
- For questions or issues, please open an issue on GitHub.
