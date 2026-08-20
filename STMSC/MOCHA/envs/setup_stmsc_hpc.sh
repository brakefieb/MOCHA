#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_ENV_ROOT="${1:-${PY_ENV_ROOT:-}}"
PROJECT_ROOT="${2:-${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}}"
READY_MARKER="${PY_ENV_ROOT}/.stmsc_env_ready_mocha"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ -z "${PY_ENV_ROOT}" ]; then
  echo "ERROR: Set PY_ENV_ROOT or pass the existing MOCHA environment as argument 1."
  echo "Usage: bash envs/setup_stmsc_hpc.sh /path/to/mocha_env [/path/to/MOCHA]"
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}/methods/STMSC" ]; then
  echo "ERROR: STMSC source not found under ${PROJECT_ROOT}/methods/STMSC"
  exit 1
fi
if [ ! -x "${PY_ENV_ROOT}/bin/python" ]; then
  echo "ERROR: Environment Python not found: ${PY_ENV_ROOT}/bin/python"
  echo "Pass the correct environment path as argument 1. This script does not create environments."
  exit 1
fi
if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "STMSC environment is already marked ready: ${READY_MARKER}"
  exit 0
fi

PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
echo "Using existing MOCHA environment: ${PY_ENV_ROOT}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip install "pip<26" "setuptools<81" wheel

# Add STMSC dependencies without replacing the existing MOCHA NumPy/PyTorch
# stack. NumPy stays below 2 for compatibility with DeepST/ResST/stLearn.
# Headless OpenCV avoids libGL errors on headless compute nodes.
"${PYTHON_BIN}" -m pip install \
  "numpy<2" \
  anndata \
  "opencv-python-headless==4.11.0.86" \
  pandas \
  POT \
  scanpy \
  scikit-learn \
  scikit-misc \
  scipy \
  torch \
  tqdm \
  h5py igraph leidenalg matplotlib pillow pyarrow pyyaml "zarr<3"

"${PYTHON_BIN}" -m pip install --no-deps --ignore-requires-python -e "${PROJECT_ROOT}/methods/STMSC"
bash "${PROJECT_ROOT}/envs/check_stmsc_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
touch "${READY_MARKER}"
echo "STMSC environment ready: ${PY_ENV_ROOT}"
