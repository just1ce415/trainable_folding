#!/usr/bin/env bash

ROOT="$(cd "$(dirname ${BASH_SOURCE})" && pwd)"

module reset
module load open-ce/1.2.0-py38-0
source activate ${ROOT}/venv
