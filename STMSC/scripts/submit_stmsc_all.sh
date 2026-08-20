#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PY_ENV_ROOT="${PY_ENV_ROOT:?Set PY_ENV_ROOT to the existing MOCHA Python environment}"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
COHORTS=(BC_10x BC_HER2+_ST BC_HP_10x BC_TNBC_ST CRC_CMS_10x DLPFC_10x KC_TLS_10x LC_TLS_10x MOB_ST RCC_TLS_10x)
SKIP_COHORTS="${SKIP_COHORTS:-}"

should_skip() {
  local cohort="$1"
  local skipped
  for skipped in ${SKIP_COHORTS//,/ }; do
    if [ "${cohort}" = "${skipped}" ]; then
      return 0
    fi
  done
  return 1
}

cd "${PROJECT_ROOT}"
for cohort in "${COHORTS[@]}"; do
  if should_skip "${cohort}"; then
    echo "Skipping ${cohort} (SKIP_COHORTS=${SKIP_COHORTS})"
    continue
  fi
  n_groups="$("${PYTHON_BIN}" scripts/list_stmsc_groups.py --cohort "${cohort}" --count)"
  last=$((n_groups - 1))
  echo "Submitting ${cohort}: ${n_groups} groups"
  sbatch --job-name="stmsc_${cohort}" --array="0-${last}%${MAX_CONCURRENT}" scripts/run_stmsc.slurm "${cohort}"
done
