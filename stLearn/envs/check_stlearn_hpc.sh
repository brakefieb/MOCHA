#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"

echo "Checking stLearn Python packages..."
"${PYTHON_BIN}" - <<'PY'
import anndata
import igraph
import leidenalg
import natsort
import numpy
import pandas
import PIL
import pyarrow
import scanpy
import sklearn
import spatialdata
import stlearn
import torch
import torchvision
print("stLearn Python deps OK")
print("stlearn version =", getattr(stlearn, "__version__", "unknown"))
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

echo "All checks passed."

