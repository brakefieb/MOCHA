#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/DeepST:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

echo "Checking Python packages..."
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
"${PYTHON_BIN}" - <<'PY'
import anndata
import igraph
import leidenalg
import louvain
import numpy
import pandas
import PIL
import psutil
import pyarrow
import scanpy
import sklearn
import torch
import torch_geometric
import torchvision
import yaml
import deepstkit
print("DeepST Python deps OK")
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

echo "All checks passed."

