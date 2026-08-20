#!/bin/bash
#SBATCH --job-name=bayesmart
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bayesmart_%j.out
#SBATCH --error=logs/bayesmart_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"

PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
RSCRIPT_BIN="${PY_ENV_ROOT}/bin/Rscript"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "ERROR: Python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -x "${RSCRIPT_BIN}" ]; then
  echo "ERROR: Rscript not found: ${RSCRIPT_BIN}"
  exit 1
fi

module purge

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export R_HOME="${PY_ENV_ROOT}/lib/R"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "RSCRIPT_BIN=${RSCRIPT_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "R_HOME=${R_HOME}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}"
echo "MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"

cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p results

"${PYTHON_BIN}" --version
"${RSCRIPT_BIN}" --version

# ---------------------------------
# Option A: only check environment
# ---------------------------------
bash envs/check_bayesmart_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"

# ---------------------------------
# Option B: if needed, install missing packages
# Uncomment only when preparing environment, not for every production run
# ---------------------------------
# bash envs/setup_bayesmart_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"

# ---------------------------------
# Run method
# ---------------------------------
"${PYTHON_BIN}" code_Xin/main.py \
  --method BayeSMART \
  --cohort "${COHORT}" \
  --rscript "${RSCRIPT_BIN}"

echo "===== Job End ====="
date
