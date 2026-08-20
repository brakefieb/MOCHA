#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
TORCH_VARIANT="${3:-cpu}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
READY_MARKER="${PY_ENV_ROOT}/.deepst_env_ready_${TORCH_VARIANT}"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/DeepST:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "DeepST environment marker found: ${READY_MARKER}"
  echo "Skipping package installation."
  echo "Set FORCE_SETUP=1 to run setup again."
  exit 0
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Torch variant: ${TORCH_VARIANT}"

"${PYTHON_BIN}" -m pip install --upgrade pip wheel
"${PYTHON_BIN}" -m pip install "setuptools<81"
"${PYTHON_BIN}" -m pip install \
  anndata \
  matplotlib \
  "numpy<2" \
  pandas \
  pillow \
  psutil \
  pyarrow \
  pyyaml \
  scanpy==1.10.3 \
  scikit-learn \
  scipy \
  "setuptools<81" \
  tqdm \
  "zarr<3" \
  igraph==0.11.8 \
  leidenalg \
  louvain

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

echo "Verifying setuptools/pkg_resources and louvain..."
"${PYTHON_BIN}" - <<'PY'
import pkg_resources
import louvain
print("pkg_resources OK")
print("louvain OK")
PY

touch "${READY_MARKER}"
echo "DeepST environment setup complete. Marker written to ${READY_MARKER}"

