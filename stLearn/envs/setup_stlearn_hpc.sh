#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
READY_MARKER="${PY_ENV_ROOT}/.stlearn_env_ready"
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
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "stLearn environment marker found: ${READY_MARKER}"
  echo "Skipping package installation."
  echo "Set FORCE_SETUP=1 to run setup again."
  exit 0
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

# Avoid self-upgrading pip inside the shared micromamba env on HPC.
# We have seen pip self-upgrades leave the interpreter in a broken state.
"${PYTHON_BIN}" -m pip --version
"${PYTHON_BIN}" -m pip install wheel "setuptools<81"

# Practical dependency set for MOCHA + stLearn runner on HPC.
# stLearn itself requires Python >= 3.12 and newer scanpy/anndata than mocha_env.
"${PYTHON_BIN}" -m pip install \
  "anndata>=0.12.0" \
  "scanpy>=1.12.0" \
  "numpy>=2,<3" \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  "pillow>=11,<12" \
  pyarrow \
  pyyaml \
  tqdm \
  natsort \
  numba \
  "igraph>=1.0.0" \
  "leidenalg>=0.11.0" \
  imageio \
  scikit-image \
  zarr \
  bokeh \
  click \
  dask \
  geopandas \
  shapely \
  spatialdata \
  spatialdata-io \
  spatialdata-plot

# stLearn morphology extraction uses torchvision pretrained models.
# Reuse an existing torch install if present; otherwise install CPU wheels.
if "${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "torchvision"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing_torch_stack=", missing)
raise SystemExit(0 if not missing else 1)
PY
then
  echo "Existing torch/torchvision detected; keeping current install."
else
  "${PYTHON_BIN}" -m pip install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu
fi

# Install local stLearn source into the target environment.
"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}/methods/stLearn" --no-deps

echo "Verifying stLearn imports..."
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
import stlearn
import torch
import torchvision
print("stLearn Python deps OK")
print("stlearn version =", getattr(stlearn, "__version__", "unknown"))
print("torch.cuda.is_available =", torch.cuda.is_available())
PY

touch "${READY_MARKER}"
echo "stLearn environment setup complete. Marker written to ${READY_MARKER}"

