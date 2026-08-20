#!/bin/bash
#SBATCH --job-name=deepst_gpu
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=logs/deepst_gpu_%j.out
#SBATCH --error=logs/deepst_gpu_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
TORCH_VARIANT="${2:-cu121}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "Torch variant: ${TORCH_VARIANT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

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

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

bash envs/check_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"

# Uncomment only when preparing a fresh GPU environment.
# bash envs/setup_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" "${TORCH_VARIANT}"

"${PYTHON_BIN}" code_Xin/main.py \
  --method DeepST \
  --cohort "${COHORT}"

echo "===== Job End ====="
date

