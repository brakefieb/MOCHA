#!/bin/bash
#SBATCH --job-name=resst_gpu
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/resst_gpu_%j.out
#SBATCH --error=logs/resst_gpu_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
TORCH_VARIANT="${2:-cu121}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"

echo "===== ResST GPU Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "Torch variant: ${TORCH_VARIANT}"
echo "AUTO_INSTALL: ${AUTO_INSTALL}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"

PROJECT_ROOT="/path/to/MOCHA"
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
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/ResST:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MALLOC_ARENA_MAX=2
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID}"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/path/to/micromamba}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-/path/to/micromamba/pkgs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_PKGS_DIRS}}"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${MAMBA_PKGS_DIRS}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true
"${PYTHON_BIN}" --version

if bash "${PROJECT_ROOT}/envs/check_resst_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"; then
  echo "ResST environment check passed."
else
  echo "ResST environment check failed."
  if [ "${AUTO_INSTALL}" = "1" ]; then
    echo "AUTO_INSTALL=1, running setup_resst_hpc.sh..."
    bash "${PROJECT_ROOT}/envs/setup_resst_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}" "${TORCH_VARIANT}"
    bash "${PROJECT_ROOT}/envs/check_resst_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
  else
    echo "AUTO_INSTALL=0, aborting."
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" code_Xin/main.py \
  --method ResST \
  --cohort "${COHORT}"

echo "===== ResST GPU Job End ====="
date

