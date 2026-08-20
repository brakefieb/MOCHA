# BayeSMART Runner Guide

This document describes the MOCHA integration of BayeSMART, including its
inputs, Python/R environment, execution, and standardized outputs.

## Integration files

```text
methods/BayeSMART/                       upstream method source
code_Xin/runners/run_bayesmart.py        Python orchestration and data export
code_Xin/runners/run_bayesmart.R         R inference backend
configs/methods/bayesmart.yaml           method defaults
configs/experiments/BayeSMART/*.yaml     cohort-specific overrides
envs/setup_bayesmart_hpc.sh              dependency setup
envs/check_bayesmart_hpc.sh              environment validation
scripts/run_bayesmart.slurm              Slurm job example
```

## Inputs

Each sample is loaded from one `.h5ad` file. The runner requires:

- an expression matrix in `adata.X`;
- gene names in `adata.var_names`;
- spot identifiers in `adata.obs_names`;
- two-dimensional coordinates in supported `obs` columns or in
  `obsm['spatial']`, `obsm['X_spatial']`, or `obsm['S']`.

Ground-truth labels are optional and are used only for evaluation. BayeSMART
does not require a pathology image.

The Python runner exports counts, coordinates, labels, and a sample manifest to
an intermediate directory. The R backend performs method-specific feature
selection and posterior inference.

## Configuration

Defaults are defined in `configs/methods/bayesmart.yaml`. Important settings
include:

```yaml
runtime:
  mcmc_iter: 5000
  store_thin: 100
  preprocessing_workers: 4

parameters:
  k: 7
  w: 0.1
  n_neighbor: 6
  f_val: 1

preprocessing:
  gene_select: sparkx
  n_gene: 2000
  pcn: 3
```

Cohort-specific values, including the requested number of domains, live under
`configs/experiments/BayeSMART/`.

## Environment setup

BayeSMART requires both Python and R in the same environment. Use portable
placeholders rather than committing a username or cluster-specific path:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"

bash envs/setup_bayesmart_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
bash envs/check_bayesmart_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
```

Confirm that `${PY_ENV_ROOT}/bin/python` and `${PY_ENV_ROOT}/bin/Rscript` are
available before starting a full run.

## Running

Run one cohort through the unified entry point:

```bash
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PY_ENV_ROOT}/bin/python" code_Xin/main.py \
  --method BayeSMART \
  --cohort DLPFC_10x \
  --rscript "${PY_ENV_ROOT}/bin/Rscript"
```

Submit an HPC job after setting paths for the cluster:

```bash
PROJECT_ROOT="/path/to/MOCHA" \
PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env" \
sbatch --export=ALL scripts/run_bayesmart.slurm DLPFC_10x
```

BayeSMART uses MCMC, so production runs can take substantially longer than a
small environment check. Use cohort overrides for smoke testing rather than
interpreting shortened chains as benchmark results.

## Outputs

Outputs are written under `results/BayeSMART/<cohort>/`:

```text
predictions.csv
predictions.parquet
performance.csv
performance.parquet
evaluation_summary.csv
run_metadata.json
stdout.log
stderr.log
figures/<sample>_true_vs_pred.png
```

Optional posterior and intermediate artifacts are controlled by the `outputs`
section of `configs/methods/bayesmart.yaml`.
> **Branch layout:** This branch contains the reproducibility bundle for BayeSMART only. After cloning or switching to this branch, run `cd BayeSMART/MOCHA` before using the commands below. Input data and generated results are intentionally excluded.
