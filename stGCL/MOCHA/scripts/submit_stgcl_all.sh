#!/bin/bash
set -euo pipefail

COHORTS=(
  BC_10x
  BC_HER2+_ST
  BC_HP_10x
  BC_TNBC_ST
  CRC_CMS_10x
  DLPFC_10x
  KC_TLS_10x
  LC_TLS_10x
  MOB_ST
  RCC_TLS_10x
)

mkdir -p logs
for cohort in "${COHORTS[@]}"; do
  sbatch --job-name="stgcl_${cohort}" scripts/run_stgcl.slurm "${cohort}"
done

