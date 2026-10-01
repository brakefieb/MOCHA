#!/bin/bash
#SBATCH --job-name=stlearn
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stlearn_%j.out
#SBATCH --error=logs/stlearn_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "AUTO_INSTALL: ${AUTO_INSTALL}"

PROJECT_ROOT="${PROJECT_ROOT:-/path/to/MOCHA}"
PY_ENV_ROOT="${PY_ENV_ROOT:-/path/to/micromamba/envs/mocha_env}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

module purge

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}"

"${PYTHON_BIN}" --version

if bash "${PROJECT_ROOT}/envs/check_stlearn_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"; then
  echo "stLearn environment check passed."
else
  echo "stLearn environment check failed."
  if [ "${AUTO_INSTALL}" = "1" ]; then
    echo "AUTO_INSTALL=1, running setup_stlearn_hpc.sh..."
    bash "${PROJECT_ROOT}/envs/setup_stlearn_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
    bash "${PROJECT_ROOT}/envs/check_stlearn_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
  else
    echo "AUTO_INSTALL=0, aborting."
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" code_Xin/main.py \
  --method stLearn \
  --cohort "${COHORT}"

echo "===== Job End ====="
date

