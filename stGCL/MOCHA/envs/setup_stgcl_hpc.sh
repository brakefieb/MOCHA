#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"

PY_ENV_ROOT="${1:-${PY_ENV_ROOT:-${MAMBA_ROOT}/envs/mocha_env}}"
PROJECT_ROOT="${2:-${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}}"
TORCH_VARIANT="${3:-cu121}"
READY_MARKER="${PY_ENV_ROOT}/.stgcl_packages_ready_${TORCH_VARIANT}"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "stGCL environment is already prepared: ${READY_MARKER}"
  echo "Set FORCE_SETUP=1 to reinstall."
  exit 0
fi

if [ ! -x "${PY_ENV_ROOT}/bin/python" ]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    echo "ERROR: micromamba is required to create ${PY_ENV_ROOT}"
    exit 1
  fi
  micromamba create -y -p "${PY_ENV_ROOT}" python=3.10 pip
fi

PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/stGCL:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_stgcl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_stgcl}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

"${PYTHON_BIN}" -m pip install --upgrade pip wheel
"${PYTHON_BIN}" -m pip install "setuptools<81"
"${PYTHON_BIN}" -m pip install \
  anndata \
  glob2 \
  h5py \
  matplotlib \
  munkres \
  "numpy<2" \
  opencv-python-headless \
  pandas \
  pillow \
  psutil \
  pyarrow \
  pyyaml \
  scanpy==1.10.3 \
  scikit-learn \
  scikit-misc \
  scipy \
  seaborn \
  tqdm \
  "zarr<3"

case "${TORCH_VARIANT}" in
  cpu)
    "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
    "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cpu torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    ;;
  cu118)
    "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
    "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cu118 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    ;;
  cu121)
    "${PYTHON_BIN}" -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
    "${PYTHON_BIN}" -m pip install pyg_lib==0.3.1+pt21cu121 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    ;;
  *)
    echo "ERROR: TORCH_VARIANT must be cpu, cu118, or cu121"
    exit 1
    ;;
esac

"${PYTHON_BIN}" -m pip install torch_geometric==2.3.1
"${PYTHON_BIN}" -m pip install --no-deps -e "${PROJECT_ROOT}/methods/stGCL"

bash "${PROJECT_ROOT}/envs/check_stgcl_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
touch "${READY_MARKER}"
echo "stGCL environment setup complete: ${PY_ENV_ROOT}"
