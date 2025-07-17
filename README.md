# Kolmogorov-Arnold Networks (KANs) Reproducibility Package

This repository contains code and scripts for reproducing experiments with Kolmogorov-Arnold Networks (KANs), including system identification tasks and autoencoder models.

## Folder Structure
rep_package/
├── .gitignore 
├── ANNmodel.py 
├── batchRun.sh 
├── DummyModel.py 
├── DynamicalSystem.py 
├── l21.py 
├── main.py 
├── README.md 
├── reproducibility.sh 
├── TwoTanks.py

## Requirements

- **Python version:** 3.10.8

- Install dependencies using pip:

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
