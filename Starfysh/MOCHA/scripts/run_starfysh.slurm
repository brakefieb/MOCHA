#!/bin/bash
#SBATCH --job-name=starfysh
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/starfysh_%j.out
#SBATCH --error=logs/starfysh_%j.err

set -euo pipefail

COHORT="${1:?Please provide cohort name, e.g. DLPFC_10x}"
TARGET="${2:-}"
AUTO_INSTALL="${AUTO_INSTALL:-1}"
TORCH_VARIANT="${TORCH_VARIANT:-cu121}"

echo "===== Job Start ====="
date
hostname
pwd
echo "Cohort: ${COHORT}"
echo "Target: ${TARGET:-ALL}"
echo "AUTO_INSTALL: ${AUTO_INSTALL}"
echo "TORCH_VARIANT: ${TORCH_VARIANT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [ -z "${PROJECT_ROOT:-}" ]; then
  if [ -d "/path/to/MOCHA" ]; then
    PROJECT_ROOT="/path/to/MOCHA"
  elif [ -d "/path/to/MOCHA" ]; then
    PROJECT_ROOT="/path/to/MOCHA"
  else
    PROJECT_ROOT="$(pwd)"
  fi
fi

if [ -z "${PY_ENV_ROOT:-}" ]; then
  if [ -x "/path/to/micromamba/envs/mocha_env/bin/python" ]; then
    PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
  else
    PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
  fi
fi
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
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/path/to/micromamba}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-/path/to/micromamba/pkgs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${MAMBA_PKGS_DIRS}}"
export STARFYSH_REQUIRE_CUDA="${STARFYSH_REQUIRE_CUDA:-1}"
if [[ "${TARGET}" == shard:* ]]; then
  SHARD_SPEC="${TARGET#shard:}"
  export STARFYSH_GROUP_SHARD_INDEX="${SHARD_SPEC%/*}"
  export STARFYSH_GROUP_SHARD_TOTAL="${SHARD_SPEC#*/}"
elif [ -n "${TARGET}" ]; then
  export STARFYSH_ONLY_GROUP="${TARGET}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}"

cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p results
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${PIP_CACHE_DIR}" "${MAMBA_PKGS_DIRS}"

"${PYTHON_BIN}" --version

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if bash "${PROJECT_ROOT}/envs/check_starfysh_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"; then
  echo "Starfysh environment check passed."
else
  echo "Starfysh environment check failed."
  if [ "${AUTO_INSTALL}" = "1" ]; then
    echo "AUTO_INSTALL=1, running setup_starfysh_hpc.sh..."
    bash "${PROJECT_ROOT}/envs/setup_starfysh_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}" "${TORCH_VARIANT}"
    bash "${PROJECT_ROOT}/envs/check_starfysh_hpc.sh" "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
  else
    echo "AUTO_INSTALL=0, aborting."
    exit 1
  fi
fi

"${PYTHON_BIN}" code_Xin/main.py \
  --method Starfysh \
  --cohort "${COHORT}"

echo "===== Job End ====="
date

