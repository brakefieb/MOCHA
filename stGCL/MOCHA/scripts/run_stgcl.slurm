#!/bin/bash
#SBATCH --job-name=stgcl
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stgcl_%x_%j.out
#SBATCH --error=logs/stgcl_%x_%j.err

set -euo pipefail

COHORT="${1:?Provide a cohort, e.g. DLPFC_10x}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-${HOME}/micromamba}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
PY_ENV_ROOT="${PY_ENV_ROOT:-${MAMBA_ROOT}/envs/mocha_env}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

module purge
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/stGCL:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_stgcl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_stgcl_${SLURM_JOB_ID}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cd "${PROJECT_ROOT}"
echo "===== stGCL start ====="
date
hostname
echo "cohort=${COHORT}"
echo "project=${PROJECT_ROOT}"
echo "python=${PYTHON_BIN}"
nvidia-smi

bash envs/check_stgcl_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
"${PYTHON_BIN}" code_Xin/main.py --method stGCL --cohort "${COHORT}"

echo "===== stGCL end ====="
date
