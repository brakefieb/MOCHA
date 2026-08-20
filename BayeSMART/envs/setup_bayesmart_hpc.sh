#!/bin/bash
set -euo pipefail

PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
PROJECT_ROOT="/path/to/MOCHA"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
RSCRIPT_BIN="${PY_ENV_ROOT}/bin/Rscript"
READY_MARKER="${PY_ENV_ROOT}/.baysmart_env_ready"
FORCE_SETUP="${FORCE_SETUP:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -x "${RSCRIPT_BIN}" ]; then
  echo "ERROR: Rscript not found: ${RSCRIPT_BIN}"
  exit 1
fi

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export R_HOME="${PY_ENV_ROOT}/lib/R"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export CC="${PY_ENV_ROOT}/bin/x86_64-conda-linux-gnu-cc"
export CXX="${PY_ENV_ROOT}/bin/x86_64-conda-linux-gnu-c++"

if [ -f "${READY_MARKER}" ] && [ "${FORCE_SETUP}" != "1" ]; then
  echo "BayesSMART environment marker found: ${READY_MARKER}"
  echo "Skipping package installation."
  echo "Set FORCE_SETUP=1 to run setup again."
  exit 0
fi

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "Using Rscript: ${RSCRIPT_BIN}"
"${RSCRIPT_BIN}" --version

echo "Installing/checking Python packages..."
"${PYTHON_BIN}" "${PROJECT_ROOT}/envs/install_bayesmart_python.py"

echo "Installing/checking R packages..."
"${RSCRIPT_BIN}" "${PROJECT_ROOT}/envs/install_bayesmart_packages.R"

touch "${READY_MARKER}"
echo "Environment setup complete. Marker written to ${READY_MARKER}"
