#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"

PY_ENV_ROOT="${1:-${PY_ENV_ROOT:-${MAMBA_ROOT}/envs/mocha_env}}"
PROJECT_ROOT="${2:-${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  echo "Run: bash envs/setup_stgcl_hpc.sh"
  exit 1
fi

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/stGCL:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_stgcl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_stgcl}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import anndata
import cv2
import glob2
import munkres
import numpy
import pandas
import PIL
import pyarrow
import scanpy
import sklearn
import torch
import torch_geometric
import torch_sparse
import torchvision
import yaml
import stGCL
from stGCL.modules import extract_model

print("stGCL dependencies OK")
print("torch =", torch.__version__)
print("torch.cuda.is_available =", torch.cuda.is_available())
print("stGCL source =", stGCL.__file__)
PY
