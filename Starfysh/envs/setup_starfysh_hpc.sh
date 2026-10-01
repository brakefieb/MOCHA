#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
TORCH_VARIANT="${3:-cpu}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
READY_MARKER="${PY_ENV_ROOT}/.starfysh_overlay_ready"
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

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  if [ "${TORCH_VARIANT}" = "cpu" ]; then
    echo "Starfysh environment marker found: ${READY_MARKER}"
    echo "Skipping package installation."
    echo "Set FORCE_SETUP=1 to run setup again."
    exit 0
  fi
  if "${PYTHON_BIN}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    echo "Starfysh environment marker found and CUDA torch is available: ${READY_MARKER}"
    echo "Skipping package installation."
    echo "Set FORCE_SETUP=1 to run setup again."
    exit 0
  fi
  echo "Starfysh marker exists, but CUDA torch is not available; repairing torch install."
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
"${PYTHON_BIN}" -m pip install wheel "setuptools<81"

# Install the MOCHA base stack plus Starfysh-specific extras. This also works
# when mocha_env was freshly created on /work and has only python/pip.
"${PYTHON_BIN}" -m pip install \
  anndata \
  matplotlib \
  networkx \
  "numpy<2" \
  opencv-python-headless \
  pandas \
  pillow \
  py_pcha \
  pyarrow \
  pyyaml \
  scanpy \
  scikit-dimension \
  scikit-image \
  scikit-learn \
  scipy \
  seaborn \
  threadpoolctl \
  tqdm \
  umap-learn \
  "zarr<3"

if "${PYTHON_BIN}" - <<PY
import importlib.util
mods = ["torch", "torchvision"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("missing_torch_stack=", missing)
    raise SystemExit(1)
if "${TORCH_VARIANT}" == "cpu":
    print("Existing torch/torchvision detected for CPU run; keeping current install.")
    raise SystemExit(0)
import torch
cuda_ok = torch.cuda.is_available()
print("existing_torch_cuda_available=", cuda_ok)
raise SystemExit(0 if cuda_ok else 1)
PY
then
  echo "Existing torch/torchvision install is compatible; keeping current install."
else
  if [ "${TORCH_VARIANT}" = "cpu" ]; then
    "${PYTHON_BIN}" -m pip install \
      --upgrade --force-reinstall \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cpu
  else
    "${PYTHON_BIN}" -m pip install \
      --upgrade --force-reinstall \
      torch torchvision torchaudio \
      --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"
  fi
fi

"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}/methods/starfysh" --no-deps

echo "Verifying Starfysh imports..."
"${PYTHON_BIN}" - <<'PY'
import sys, types
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
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

touch "${READY_MARKER}"
echo "Starfysh environment setup complete. Marker written to ${READY_MARKER}"

