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
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/ResST:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/path/to/micromamba}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-/path/to/micromamba/pkgs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_PKGS_DIRS}}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${MAMBA_PKGS_DIRS}"

echo "Checking ResST Python packages..."
"${PYTHON_BIN}" - <<'PY'
import anndata
import igraph
import leidenalg
import numpy
import pandas
import PIL
import scanorama
import scanpy
import sklearn
import torch
import torch_geometric
import torch_sparse
import torchvision
import resst
from resst.model_ST_utils import trainer, priori_cluster
from resst.get_adata import get_data, refine
from resst.preprocess import get_enhance_feature
print("ResST Python deps OK")
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

echo "All checks passed."

