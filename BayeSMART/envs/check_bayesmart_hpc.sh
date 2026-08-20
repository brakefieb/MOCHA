#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
PROJECT_ROOT="/path/to/MOCHA"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
RSCRIPT_BIN="${PY_ENV_ROOT}/bin/Rscript"

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export R_HOME="${PY_ENV_ROOT}/lib/R"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export CC="${PY_ENV_ROOT}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${PY_ENV_ROOT}/bin/x86_64-conda-linux-gnu-c++"

echo "Checking Python packages..."
"${PYTHON_BIN}" - <<'PY'
import scanpy, anndata, pandas, numpy, yaml, pyarrow
print("Python deps OK")
PY

echo "Checking R packages..."
"${RSCRIPT_BIN}" -e 'library(Rcpp); library(RcppArmadillo); library(RcppDist); library(scater); library(scran); library(harmony); library(jsonlite); library(STdeconvolve); library(SPARK); cat("R deps OK\n")'

echo "All checks passed."
