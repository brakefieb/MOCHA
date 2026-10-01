#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY_ENV_ROOT="${1:-${PY_ENV_ROOT:-}}"
PROJECT_ROOT="${2:-${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

if [ -z "${PY_ENV_ROOT}" ]; then
  echo "ERROR: Set PY_ENV_ROOT or pass the MOCHA environment as argument 1."
  echo "Usage: bash envs/check_stmsc_hpc.sh /path/to/mocha_env [/path/to/MOCHA]"
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi
export PYTHONPATH="${PROJECT_ROOT}/code_Xin:${PROJECT_ROOT}/methods/STMSC:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_stmsc_check}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_stmsc_check}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import anndata, cv2, numpy, ot, pandas, pyarrow, scanpy, scipy, sklearn, torch
import STMSC
from STMSC.load_data_preprocess import extract_histology_features, preprocess
from STMSC.train import learn_mapping_matrix, train_stmsc_model
from STMSC.utils import construct_combined_graph
print("STMSC imports OK")
print("numpy=", numpy.__version__)
print("torch=", torch.__version__)
print("torch.cuda.is_available=", torch.cuda.is_available())
if os.environ.get("REQUIRE_CUDA", "0") == "1" and not torch.cuda.is_available():
    raise RuntimeError(
        "A GPU STMSC job was requested, but the existing mocha_env PyTorch cannot see CUDA."
    )
PY
