#!/usr/bin/env bash

ROOT="$(cd "$(dirname ${BASH_SOURCE})" && pwd)"
source activate ${ROOT}/venv

export PATH="${ROOT}/venv/bin:${ROOT}/scripts:${PATH}"
