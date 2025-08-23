#!/bin/bash
#SBATCH --job-name=ann_kan_2
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=96:00:00

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

# Run your python task (replace script.py with your script name)
#cd $HOME/pykan/ann_kan/
#pip install -r requirements.txt
./batchRun.sh $1 $2 $3 $4 $5 $6 $7 $8

#pyenv deactivate