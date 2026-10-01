#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
TORCH_VARIANT="${3:-cu121}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
READY_MARKER="${PY_ENV_ROOT}/.spacontds_env_ready_${TORCH_VARIANT}"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  echo "Create it first, for example:"
  echo "  micromamba create -p ${PY_ENV_ROOT} -y python=3.11 pip"
  exit 1
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset MAMBA_DEFAULT_ENV
PATH_CLEAN="$(printf '%s' "${PATH}" | tr ':' '\n' | awk 'index($0, "/mamba_envs/") == 0 && index($0, "/micromamba/envs/") == 0 {print}' | paste -sd: -)"
export PATH="${PY_ENV_ROOT}/bin:${PATH_CLEAN}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/code_Xin:${PROJECT_ROOT}/methods/SpaConTDS"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/path/to/micromamba}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-/path/to/micromamba/pkgs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_PKGS_DIRS}}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${MAMBA_PKGS_DIRS}"

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "SpaConTDS environment marker found: ${READY_MARKER}"
  echo "Skipping package installation."
  echo "Set FORCE_SETUP=1 to run setup again."
  exit 0
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
echo "Torch variant: ${TORCH_VARIANT}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install "setuptools<81" wheel

if "${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = ["torch", "torchvision", "torch_geometric", "torch_sparse", "torch_scatter"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing_torch_stack=", missing)
raise SystemExit(0 if not missing else 1)
PY
then
  echo "Existing torch/PyG stack detected in mocha_env; keeping current install."
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
  "${PYTHON_BIN}" -m pip install torch-geometric==2.3.1
fi

"${PYTHON_BIN}" -m pip install \
  anndata \
  h5py \
  igraph \
  leidenalg \
  matplotlib \
  "numpy<2" \
  opencv-python \
  pandas \
  pillow \
  POT \
  pyarrow \
  pyyaml \
  scanpy \
  scikit-learn \
  scikit-misc \
  scipy \
  tqdm \
  "zarr<3"

echo "Verifying SpaConTDS imports..."
bash "${PROJECT_ROOT}/envs/check_spacontds_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"

touch "${READY_MARKER}"
echo "SpaConTDS environment setup complete. Marker written to ${READY_MARKER}"

