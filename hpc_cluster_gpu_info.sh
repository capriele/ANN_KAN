#!/bin/bash
#SBATCH --job-name=ann_kan_2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=96:00:00
#-SBATCH --output=slurm-%j.out
#-SBATCH --error=slurm-%j.err
#-SBATCH --time=96:00:00
#-SBATCH --nodes=1
#-SBATCH --ntasks=1
#-SBATCH --cpus-per-task=36
#-SBATCH --mem=100GB
#-SBATCH --partition=serial
# --partition=lprod

# | Name       | Priority | GraceTime | Flags                                   | UsageFactor | GrpTRES       | MaxTRES | MaxTRESPerNode | MaxTRESMins | MaxWall | MaxTRESPU | MinTRES    |
# |------------|----------|-----------|-----------------------------------------|-------------|---------------|---------|----------------|-------------|---------|-----------|------------|
# | normal     | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           |            |
# | serial_pa+ | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes   | 1.000.000   | cpu=36,mem=1+ | cpu=36  |                |             |         |           |            |
# | lprod_part | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes,+ | 1.000.000   | cpu=64,mem=3+ |         |                |             |         |           | cpu=1      |
# | bprod_part | 0        | 00:00:00  | cluster DenyOnLimit,OverPartQOS         | 1.000.000   | cpu=256,mem=+ | cpu=256 | cpu=36         | cpu=256     | 100     | 120       | cpu=65     |
# | debug_part | 0        | 00:00:00  | cluster DenyOnLimit,PartitionMaxNodes   | 1.000.000   | cpu=4,mem=12+ |         |                |             | 2       | 8         |            |
# | default_q+ | 1000     | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             | 1000    | 1200      |            |
# | staff_gpu+ | 1100     | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           |            |
# | gprod_part | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             |         |           | gres/gpu=0 |
# | high_job   | 0        | 00:00:00  | cluster DenyOnLimit                     | 1.000.000   |               |         |                |             | 1000    | 1200      |            |


# User specific aliases and functions
export HOME="/home/petruccia"
export PATH="$HOME/bin/bin:$HOME/bin:$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

export PATH="$HOME/local/bin:$PATH"
export PATH="$HOME/local/python3.11/bin/:$PATH"

# GCC 15.1.0
export PATH="$HOME/local/gcc-15.1.0/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$HOME/local/gcc-15.1.0/lib64:$LIBRARY_PATH"
export C_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="$HOME/local/gcc-15.1.0/include:$CPLUS_INCLUDE_PATH"

# TMUX
export PATH=$HOME/local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/python3.11/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$HOME/local/include:$CPATH
export LIBRARY_PATH=$HOME/local/python3.11/lib:$HOME/local/lib64:$HOME/local/lib/pkgconfig:$HOME/local/lib:$LIBRARY_PATH

# cd $HOME/local/src
# cd Python-3.11.6

# ./configure --prefix=$HOME/local/python3.11 --enable-shared --enable-optimizations CPPFLAGS="-I$HOME/local/include" LDFLAGS="-L$HOME/local/lib -L$HOME/local/lib64" PKG_CONFIG_PATH="$HOME/local/lib/pkgconfig"

# make -j
# make install

# Try with minimal build
#PYTHON_CONFIGURE_OPTS="--enable-shared --with-ensurepip=install" pyenv install 3.11.6
#pyenv virtualenv 3.11.6 test_3_11
#pyenv activate test_3_11

nvidia-smi
