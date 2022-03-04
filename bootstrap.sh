#!/usr/bin/env bash
set -euo pipefail

# Directories to use
SRC_DIR="$(cd $(dirname "$0") && pwd)"
ENV_DIR="$(pwd)/venv"
NUMPROC=$(nproc)

# Specific commits to checkout
SBLU_COMMIT=65c6348
PRODY_VERSION=2.0.1

# Setup conda env
if [ ! -d "${ENV_DIR}" ]; then
    conda env create -f "${SRC_DIR}/conda-env.yml" --prefix "${ENV_DIR}"
fi

# Create conda environment in the current directory
set +u  # conda references PS1 variable that is not set in scripts
source activate "${ENV_DIR}"
set -u

# Setting env variables
set +u
export PKG_CONFIG_PATH="${ENV_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export PATH="${ENV_DIR}/bin:${ENV_DIR}/lib:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH}"
set -u

#pip install pytorch-lightning==${PYTORCH_LIGHTNING_VERSION}

# for model parallelism
pip install deepspeed fairscale

# Install ProDy
pip install pyparsing
pip install prody==${PRODY_VERSION}

# for cif parsing
pip install gemmi

# Install ray tune
pip install ray tensorboard hyperopt
pip install ray[tune]
pip install networkx

# Install sb-lab-utils
git clone https://bitbucket.org/bu-structure/sb-lab-utils.git
cd sb-lab-utils
git checkout ${SBLU_COMMIT}
pip install -r requirements/pipeline.txt
python setup.py install
cd ../
rm -rf sb-lab-utils

pip install -e $SRC_DIR