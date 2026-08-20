#!/bin/bash
#SBATCH --job-name=stmsc
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=logs/stmsc_%x_%A_%a.out
#SBATCH --error=logs/stmsc_%x_%A_%a.err

set -euo pipefail

COHORT="${1:?Provide a cohort, e.g. DLPFC_10x}"
GROUP_IDX="${2:-${SLURM_ARRAY_TASK_ID:-}}"
if [ -z "${GROUP_IDX}" ]; then
  echo "ERROR: provide group index as argument 2 or submit this script as an array"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY_ENV_ROOT="${PY_ENV_ROOT:?Set PY_ENV_ROOT to the existing MOCHA Python environment}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

module purge
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}/code_Xin:${PROJECT_ROOT}/methods/STMSC:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2
export PYTHONHASHSEED=2024
export REQUIRE_CUDA=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_stmsc_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_stmsc_${SLURM_JOB_ID}"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${TORCH_HOME}"

cd "${PROJECT_ROOT}"
echo "===== STMSC start ====="
date
hostname
echo "cohort=${COHORT} group_idx=${GROUP_IDX} python=${PYTHON_BIN}"
nvidia-smi
bash envs/check_stmsc_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
"${PYTHON_BIN}" code_Xin/main.py --method STMSC --cohort "${COHORT}" --group_idx "${GROUP_IDX}"
echo "===== STMSC end ====="
date
