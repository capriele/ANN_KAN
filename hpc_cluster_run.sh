#!/bin/bash
#SBATCH --job-name=ann_kan
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:8

module purge
module load python/3.10.8  # Load the appropriate Python module if on a system with module system

# Create and activate a virtualenv in the job's scratch directory
VENV_DIR="$SLURM_TMPDIR/myenv"
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# cloning the repo
git clone https://github.com/capriele/ANN_KAN.git ann_kan
cd ann_kan

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Run your python task (replace script.py with your script name)
./batchRun.sh NLTankNLF5Affine 5 1 1 6 10 1 0

# (The environment will be deactivated and cleaned up at job end)
