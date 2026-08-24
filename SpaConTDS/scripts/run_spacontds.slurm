#!/bin/bash
#SBATCH --job-name=spacontds
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/spacontds_%j.out
#SBATCH --error=logs/spacontds_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
TORCH_VARIANT="${2:-cu121}"
AUTO_INSTALL="${AUTO_INSTALL:-0}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "Torch variant: ${TORCH_VARIANT}"
echo "AUTO_INSTALL: ${AUTO_INSTALL}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"

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

unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset MAMBA_DEFAULT_ENV
PATH_CLEAN="$(printf '%s' "${PATH}" | tr ':' '\n' | awk 'index($0, "/mamba_envs/") == 0 && index($0, "/micromamba/envs/") == 0 {print}' | paste -sd: -)"
export PATH="${PY_ENV_ROOT}/bin:${PATH_CLEAN}"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/code_Xin:${PROJECT_ROOT}/methods/SpaConTDS"
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

"${PYTHON_BIN}" --version
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "PATH=${PATH}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

if bash "${PROJECT_ROOT}/envs/check_spacontds_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"; then
  echo "SpaConTDS environment check passed."
else
  echo "SpaConTDS environment check failed."
  if [ "${AUTO_INSTALL}" = "1" ]; then
    echo "AUTO_INSTALL=1, running setup_spacontds_hpc.sh..."
    bash "${PROJECT_ROOT}/envs/setup_spacontds_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}" "${TORCH_VARIANT}"
    bash "${PROJECT_ROOT}/envs/check_spacontds_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
  else
    echo "AUTO_INSTALL=0, aborting."
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"
env -i \
  HOME="${HOME}" \
  USER="${USER}" \
  PATH="${PATH}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  PYTHONNOUSERSITE="${PYTHONNOUSERSITE}" \
  PYTHONSAFEPATH="${PYTHONSAFEPATH}" \
  PYTHONPATH="${PYTHONPATH}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
  NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
  MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX}" \
  MPLCONFIGDIR="${MPLCONFIGDIR}" \
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
  TORCH_HOME="${TORCH_HOME}" \
  PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
  MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS}" \
  CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  "${PYTHON_BIN}" code_Xin/main.py \
  --method SpaConTDS \
  --cohort "${COHORT}"

echo "===== Job End ====="
date

