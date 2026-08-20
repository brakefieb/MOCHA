#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="${1:-/path/to/micromamba/envs/mocha_env}"
PROJECT_ROOT="${2:-/path/to/MOCHA}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
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
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}"

echo "Checking SpaConTDS Python packages..."
env -i \
  HOME="${HOME}" \
  USER="${USER}" \
  PATH="${PATH}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
  PYTHONSAFEPATH="${PYTHONSAFEPATH}" \
  PYTHONPATH="${PYTHONPATH}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
  TORCH_HOME="${TORCH_HOME}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

py_env_root = os.environ.get("PY_ENV_ROOT_EXPECTED", "")
print("sys.executable =", sys.executable)
print("sys.path =", sys.path)
bad_paths = [p for p in sys.path if "/mamba_envs/mof_env/" in p or p.endswith("/mamba_envs/mof_env")]
if bad_paths:
    print("ERROR: mof_env paths leaked into sys.path:")
    for path in bad_paths:
        print("  -", path)
    raise SystemExit(1)

required = {
    "anndata": "anndata",
    "cv2": "opencv-python",
    "igraph": "igraph",
    "leidenalg": "leidenalg",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "ot": "POT",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scanpy": "scanpy",
    "skmisc.loess": "scikit-misc",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "torch_sparse": "torch-sparse",
    "torch_scatter": "torch-scatter",
    "torchvision": "torchvision",
}

missing = [
    f"{module} (pip: {package})"
    for module, package in required.items()
    if importlib.util.find_spec(module.split(".")[0]) is None
]
if missing:
    print("Missing SpaConTDS dependencies:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)
PY
env -i \
  HOME="${HOME}" \
  USER="${USER}" \
  PATH="${PATH}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
  PYTHONSAFEPATH="${PYTHONSAFEPATH}" \
  PYTHONPATH="${PYTHONPATH}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
  TORCH_HOME="${TORCH_HOME}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  PY_ENV_ROOT_EXPECTED="${PY_ENV_ROOT}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path
import anndata
import cv2
import igraph
import leidenalg
import matplotlib
import numpy
import ot
import pandas
import pyarrow
import scanpy
import skmisc.loess
import scipy
import sklearn
import torch
import torch_geometric
import torch_sparse
import torch_scatter
import torchvision
from models import MultiModalEnc, TupleCL, GCNDecoder
from utils import TupleDataset, collate_fn
spacontds_main_path = Path(os.environ["PROJECT_ROOT"]) / "methods" / "SpaConTDS" / "main.py"
spec = importlib.util.spec_from_file_location("_spacontds_check_main", spacontds_main_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"ERROR: could not load SpaConTDS main.py from {spacontds_main_path}")
spacontds_main = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = spacontds_main
spec.loader.exec_module(spacontds_main)
if not hasattr(spacontds_main, "train"):
    raise SystemExit(f"ERROR: SpaConTDS main.py has no train function: {spacontds_main_path}")
print("SpaConTDS Python deps OK")
print("sys.executable =", sys.executable)
print("torch.__file__ =", torch.__file__)
print("torch.__version__ =", torch.__version__)
torch_path = Path(torch.__file__).resolve()
if "/mamba_envs/mof_env/" in str(torch_path):
    raise SystemExit(f"ERROR: imported torch from mof_env: {torch_path}")
expected_root = Path(sys.executable).resolve().parents[1]
if expected_root not in torch_path.parents:
    raise SystemExit(f"ERROR: torch is not from active env {expected_root}: {torch_path}")
print("torch.cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.get_device_name =", torch.cuda.get_device_name(0))
PY

echo "All checks passed."

