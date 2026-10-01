#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
TORCH_VARIANT="${3:-cpu}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
READY_MARKER="${PY_ENV_ROOT}/.resst_env_ready_${TORCH_VARIANT}"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}"
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

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "ResST environment marker found: ${READY_MARKER}"
  echo "Skipping package installation."
  echo "Set FORCE_SETUP=1 to run setup again."
  exit 0
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Torch variant: ${TORCH_VARIANT}"

"${PYTHON_BIN}" -m pip --version
"${PYTHON_BIN}" -m pip install "setuptools<81"
"${PYTHON_BIN}" -m pip install wheel

# ResST upstream ships a full Windows/conda export with local file:// entries.
# Keep this additive so ResST can share the DeepST environment without downgrading it.
"${PYTHON_BIN}" -m pip install \
  anndata \
  h5py \
  imageio \
  igraph \
  leidenalg \
  matplotlib \
  natsort \
  networkx \
  "numpy<2" \
  pandas \
  pillow \
  pyarrow \
  pyyaml \
  scanpy \
  scanorama \
  scikit-image \
  scikit-learn \
  scipy \
  tqdm \
  "zarr<3"

if "${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "torchvision", "torch_geometric", "torch_sparse"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing_torch_stack=", missing)
raise SystemExit(0 if not missing else 1)
PY
then
  echo "Existing torch/PyG stack detected; keeping current install."
else
  case "${TORCH_VARIANT}" in
    cpu)
      "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
      "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cpu torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
      ;;
    cu118)
      "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
      "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cu118 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
      ;;
    cu121)
      "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
      "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cu121 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
      ;;
    *)
      echo "ERROR: Unsupported TORCH_VARIANT=${TORCH_VARIANT}. Choose cpu, cu118, or cu121."
      exit 1
      ;;
  esac
  "${PYTHON_BIN}" -m pip install torch_geometric==2.3.1
fi

echo "Verifying ResST imports..."
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
print("ResST Python deps OK")
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

touch "${READY_MARKER}"
echo "ResST environment setup complete. Marker written to ${READY_MARKER}"

