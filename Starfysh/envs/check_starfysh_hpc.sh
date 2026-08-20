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
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/path/to/micromamba}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-/path/to/micromamba/pkgs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_PKGS_DIRS}}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}" "${MAMBA_PKGS_DIRS}"

echo "Checking Starfysh Python packages..."
"${PYTHON_BIN}" - <<PY
import sys, types
sys.path.insert(0, "${PROJECT_ROOT}/methods/starfysh")
sys.modules.setdefault("histomicstk", types.ModuleType("histomicstk"))
import anndata
import numpy
import pandas
import pyarrow
import scanpy
import skdim
import sklearn
import cv2
import torch
import torchvision
import umap
from py_pcha import PCHA
import starfysh
from starfysh import AA, utils
print("Starfysh Python deps OK")
cuda_ok = torch.cuda.is_available()
print("torch.cuda.is_available =", cuda_ok)
if "${STARFYSH_REQUIRE_CUDA:-0}" not in {"0", "false", "False", "no", "NO"} and not cuda_ok:
    raise SystemExit("ERROR: STARFYSH_REQUIRE_CUDA=1 but PyTorch cannot see CUDA.")
PY

echo "All checks passed."

