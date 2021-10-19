#!/usr/bin/env bash
set -euo pipefail

# Directories to use
SRC_DIR="$(cd $(dirname "$0") && pwd)"
ENV_DIR="$(pwd)/venv"
NUMPROC=$(nproc)

module reset
set +u
module load open-ce/1.4.0-py38-0
set -u

# Setup conda env
if [ ! -d "${ENV_DIR}" ]; then
    conda create --prefix "${ENV_DIR}" --clone open-ce-1.4.0-py38-0
fi

# Create conda environment in the current directory
set +u  # conda references PS1 variable that is not set in scripts
conda activate "${ENV_DIR}"
set -u

# Setting env variables
set +u
export PKG_CONFIG_PATH="${ENV_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}"
export PATH="${ENV_DIR}/bin:${ENV_DIR}/lib:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH}"
set -u

# Packages
pip install pytest seaborn tqdm path.py

# We can't install rdkit on summit

# Install ProDy
wget https://github.com/prody/ProDy/archive/refs/tags/v1.10.11.tar.gz
tar xvf v1.10.11.tar.gz
cd ProDy-1.10.11
cp "${SRC_DIR}/summit_deps/prody_setup.py" setup.py
python setup.py build
python setup.py install
cd -
rm -rf v1.10.11.tar.gz ProDy-1.10.11

pip install -e $SRC_DIR
