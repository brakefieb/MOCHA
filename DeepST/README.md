# DeepST Runner Guide

This document describes how `DeepST` is integrated into this repository, what
inputs it expects, how image alignment is handled, and how to run it locally or
on an HPC cluster.

## Overview

The repository provides a cohort-level `DeepST` runner with:

- standardized outputs for benchmarking
- optional pathology-image support
- shared spot-to-image alignment logic
- CPU and GPU batch-script examples

Relevant files:

- `code_Xin/runners/run_deepst.py`
- `code_Xin/spatial_alignment.py`
- `configs/methods/deepst.yaml`
- `configs/experiments/DeepST/*.yaml`
- `configs/cohorts/*.yaml`
- `envs/setup_deepst_hpc.sh`
- `envs/check_deepst_hpc.sh`
- `scripts/run_deepst.slurm`
- `scripts/run_deepst_gpu.slurm`

## Inputs

Each sample is loaded from one `.h5ad` file.

Required:

- expression matrix in `adata.X`
- spatial coordinates from one of:
  - `adata.obs[['x', 'y']]`
  - `adata.obs[['imagecol', 'imagerow']]`
  - `adata.obsm['spatial']`
  - `adata.obsm['X_spatial']`
  - `adata.obsm['S']`

Optional:

- pathology image matched through the cohort config
- labels from one of:
  - `annotation`
  - `Classification`
  - `ground_truth`
  - `layer_guess`
  - `manual_label`
  - `label`
  - `region`
  - `z`

## Image Alignment

Formal `DeepST` runs and pre-run checks use the same alignment code in
`code_Xin/spatial_alignment.py`.

Three alignment modes are supported:

1. `direct`
   - use coordinates already stored in the `.h5ad`
   - intended for datasets whose coordinates already match image pixel space

2. `visium_sidecar`
   - use Visium-style sidecars such as:
     - `*_tissue_positions_list.csv`
     - `*_tissue_positions.csv`
     - `*_scalefactors_json.json`

3. `st_transform_matrix`
   - use a legacy ST `3x3` transform matrix
   - intended for older Spatial Transcriptomics datasets such as `MOB_ST`

If image alignment cannot be resolved and the cohort config requests
`fallback_mode: disable_image`, the runner falls back to expression-only mode.

## Pre-run Alignment Check

You can inspect alignment before a formal run with the shared logic used by the
runner.

Notebook option:

- `code_Xin/test.ipynb`
- section: `Test image alignment with shared pipeline logic`

Script option:

```bash
python scripts/check_deepst_image_alignment.py --cohort DLPFC_10x --sample 151507
python scripts/check_deepst_image_alignment.py --cohort MOB_ST --sample 1
```

Generated overlays are written to:

```text
results/alignment_checks/<cohort>/<sample>__overlay.png
```

Formal `DeepST` runs also save overlays to:

```text
results/DeepST/<cohort>/alignment_overlays/<sample>__overlay.png
```

## Environment Setup

The helper scripts assume an existing Python or micromamba environment and take
the environment path as an argument.

Example placeholders:

```bash
PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/env"
```

CPU setup:

```bash
chmod +x envs/setup_deepst_hpc.sh
chmod +x envs/check_deepst_hpc.sh

bash envs/setup_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cpu
bash envs/check_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
```

GPU setup:

```bash
bash envs/setup_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cu121
# or
bash envs/setup_deepst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cu118
```

Notes:

- `cpu`, `cu118`, and `cu121` refer to the PyTorch wheel family
- the setup script pins key packages for compatibility, including:
  - `numpy<2`
  - `setuptools<81`
  - `zarr<3`
- the runtime scripts prepend `${PY_ENV_ROOT}/lib` to `LD_LIBRARY_PATH` so the
  environment's C++ runtime is preferred over older system libraries

## Running DeepST

Direct run:

```bash
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/DeepST:${PYTHONPATH:-}"

"${PYTHON_BIN}" code_Xin/main.py --method DeepST --cohort DLPFC_10x
"${PYTHON_BIN}" code_Xin/main.py --method DeepST --cohort BC_HER2+_ST
```

CPU batch run:

```bash
sbatch scripts/run_deepst.slurm DLPFC_10x
sbatch scripts/run_deepst.slurm MOB_ST
```

GPU batch run:

```bash
sbatch scripts/run_deepst_gpu.slurm DLPFC_10x cu121
sbatch scripts/run_deepst_gpu.slurm BC_TNBC_ST cu121
```

Cluster-specific notes:

- `scripts/run_deepst.slurm` is a CPU example
- `scripts/run_deepst_gpu.slurm` is a GPU example
- you may need to adjust:
  - partition name
  - memory
  - walltime
  - environment path
- large image-enabled cohorts may require substantially more RAM than small or
  subgroup-based runs

## Outputs

DeepST writes cohort-level results to:

```text
results/DeepST/<cohort>/
```

Main outputs:

- `predictions.csv`
- `predictions.parquet`
- `performance.csv`
- `performance.parquet`
- `evaluation_summary.csv`
- `run_metadata.json`
- `stdout.log`
- `stderr.log`
- `figures/<sample>_true_vs_pred.png`

Optional group-level output:

- `results/DeepST/<cohort>/<group_name>/predictions.csv`

The runner does not keep per-sample prediction directories.

## Parameter Defaults

The current defaults stay close to the reference `DeepST` examples:

- `pre_epochs: 500`
- `epochs: 500`
- `adjacent_weight: 0.3`
- `neighbour_k: 4`
- `spatial_k: 30`
- `pca_n_comps: 200`
- `conv_type: GATConv`
- `cnn_type: ResNet50`

Repository-specific choices:

- some cohorts run as subgroup-level integrations instead of one
  cohort-wide integration
- `10x`-style cohorts generally use `hexagon`
- older `ST`-style cohorts generally use `square`
- `n_domains` can be inferred from available labels when configured as `auto`

## Practical Notes

- GPU usually gives a large speedup for image-enabled runs
- CPU can be acceptable for small cohorts, smoke tests, or subgroup-level runs
- very large cohorts may hit RAM limits before GPU becomes the main bottleneck
- for large cohorts, inspect `stdout.log`, `stderr.log`, and `run_metadata.json`
  together when diagnosing failures
> **Branch layout:** This branch contains the reproducibility bundle for DeepST only. After cloning or switching to this branch, run `cd DeepST` before using the commands below. Input data and generated results are intentionally excluded.
