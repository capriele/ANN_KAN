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


## Datasets
1. The dataset used for the "AH1 AUV high-fidelity simulator" case study is available, with relevant explanations, at this [link](AUV https://github.com/grande-dev/public_datasets/main/1-FTC-IS-Sat)  

2. The dataset used for the "Underwater glider" case study is available, with relevant explanations, at this [link](https://github.com/grande-dev/public_datasets/main/2-KAN-state-space-models-AUV)  


## Citation
 This work can be cited with the following BibTeX entry:  

```bibtex
@article{masti2026learning,
  title={Learning State-Space Models with Kolmogorov-Arnold Networks: an Autonomous Underwater Engineering perspective},
  author={Daniele Masti, Alberto Petrucci, Davide Grande, Francesco Basciani, Patrizio Pelliccione},
  journal={ISA Transactions},
  volume={},
  pages={},
  year={2026},
  publisher={Elsevier},
  doi={}
}
``` 

