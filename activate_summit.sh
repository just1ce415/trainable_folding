#!/usr/bin/env bash

ROOT="$(cd "$(dirname ${BASH_SOURCE})" && pwd)"

module reset
module load open-ce/1.4.0-py38-0
conda activate ${ROOT}/venv

export PATH="${ROOT}/venv/bin:${ROOT}/scripts:${PATH}"