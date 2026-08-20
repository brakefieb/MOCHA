#!/bin/bash
#SBATCH --job-name=deepst
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/deepst_%j.out
#SBATCH --error=logs/deepst_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
TORCH_VARIANT="${2:-cpu}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "Torch variant: ${TORCH_VARIANT}"

PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
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
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/DeepST:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}"

cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p results
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

"${PYTHON_BIN}" --version

# ---------------------------------
# Option A: only check environment
# ---------------------------------
bash envs/check_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"

# ---------------------------------
# Option B: if needed, install missing packages
# Uncomment only when preparing environment, not for every production run
# ---------------------------------
# bash envs/setup_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" "${TORCH_VARIANT}"

# ---------------------------------
# Run method
# ---------------------------------
"${PYTHON_BIN}" code_Xin/main.py \
  --method DeepST \
  --cohort "${COHORT}"

echo "===== Job End ====="
date

