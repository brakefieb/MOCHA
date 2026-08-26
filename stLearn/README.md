# stLearn Runner Guide

This document describes the MOCHA integration of stLearn, including image
alignment, expression-only fallback, environment setup, execution, and outputs.

## Integration files

```text
methods/stLearn/                         upstream stLearn source
code_Xin/runners/run_stlearn.py          MOCHA runner
code_Xin/spatial_alignment.py            shared image-alignment logic
configs/methods/stlearn.yaml             method defaults
configs/experiments/stLearn/*.yaml       cohort-specific overrides
envs/setup_stlearn_hpc.sh                dependency setup
envs/check_stlearn_hpc.sh                environment validation
scripts/run_stlearn.slurm                Slurm job example
```

## Inputs

Each sample requires an `.h5ad` expression matrix and two-dimensional spatial
coordinates. Coordinates may come from supported `adata.obs` columns or from
`adata.obsm['spatial']`, `adata.obsm['X_spatial']`, or `adata.obsm['S']`.

A pathology image is optional. When a valid image and pixel alignment are
available, the runner enables stLearn's morphology branch. If alignment cannot
be validated and `fallback_to_expression_only: true`, the run continues with
expression and spatial information, and records the reason in metadata.

Ground-truth labels are optional and are used only for evaluation and, when
`n_domains: auto`, to infer the requested cluster count.

## Configuration

Defaults live in `configs/methods/stlearn.yaml`. Important values include:

```yaml
parameters:
  n_domains: auto
  use_morphology: auto
  fallback_to_expression_only: true
  pre_pca_n_comps: 50
  post_pca_n_comps: 50
  cnn_base: resnet50
  morphology_n_components: 50
  tiling_crop_size: 40
  tiling_target_size: 299
  kmeans_use_data: X_pca
```

Cohort-specific overrides are stored under `configs/experiments/stLearn/`.

## Environment setup

The local stLearn source requires a compatible Python and PyTorch stack. Define
paths outside version-controlled files:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/micromamba/envs/mocha_stlearn"

bash envs/setup_stlearn_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
bash envs/check_stlearn_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
```

The setup installs the vendored source from `methods/stLearn` and creates an
environment readiness marker. Set `FORCE_SETUP=1` only when dependencies need
to be refreshed.

## Running

Run one cohort locally or in an interactive allocation:

```bash
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PY_ENV_ROOT}/bin/python" code_Xin/main.py \
  --method stLearn \
  --cohort DLPFC_10x
```

Submit one cohort with Slurm:

```bash
PROJECT_ROOT="/path/to/MOCHA" \
PY_ENV_ROOT="/path/to/micromamba/envs/mocha_stlearn" \
AUTO_INSTALL=0 \
sbatch --export=ALL scripts/run_stlearn.slurm DLPFC_10x
```

Adjust partition, memory, time, and GPU directives to match the target cluster.
Keeping `AUTO_INSTALL=0` for production jobs prevents an unexpected dependency
installation on a compute node.

## Outputs

Outputs are written under `results/stLearn/<cohort>/`:

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

Alignment overlays and other optional artifacts are stored in the corresponding
cohort result directory when enabled.
> **Branch layout:** This branch contains the reproducibility bundle for stLearn only. After cloning or switching to this branch, run `cd stLearn` before using the commands below. Input data and generated results are intentionally excluded.
