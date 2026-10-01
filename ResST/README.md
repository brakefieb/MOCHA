# ResST Runner Guide

This document describes how `ResST` is integrated into MOCHA, how to set up the
runtime environment, how histology images are handled, and how to run the method
locally or on an HPC cluster.

## Overview

The repository provides a MOCHA-compatible `ResST` runner with:

- standardized benchmark outputs
- optional histology-image features through the original ResST image branch
- shared spot-to-image alignment logic with the other image-aware methods
- resumable preprocessing and group-level checkpoints for long HPC runs
- CPU and GPU SLURM examples

Relevant files:

- `methods/ResST/`
- `code_Xin/runners/run_resst.py`
- `code_Xin/spatial_alignment.py`
- `configs/methods/resst.yaml`
- `configs/experiments/ResST/*.yaml`
- `configs/cohorts/*.yaml`
- `envs/setup_resst_hpc.sh`
- `envs/check_resst_hpc.sh`
- `scripts/run_resst.slurm`
- `scripts/run_resst_gpu.slurm`

## Source Code

The ResST source is vendored under `methods/ResST`.

To download the upstream method into a fresh MOCHA checkout:

```bash
cd /path/to/MOCHA
git clone https://github.com/StickTaTa/ResST_main.git methods/ResST
rm -rf methods/ResST/.git
```

If the repository was first cloned as `methods/ResST_main`, rename it:

```bash
mv methods/ResST_main methods/ResST
rm -rf methods/ResST/.git
```

Removing the nested `.git` directory keeps the vendored method under the main
MOCHA repository history.

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

The original ResST demo can use histology image features with
`cnnType='ResNet50'`, then combine morphology, expression, and spatial
neighbors before training.

MOCHA follows that behavior when image alignment is available. Before using an
image, the runner resolves spot-to-image coordinates with the shared alignment
code in `code_Xin/spatial_alignment.py`, matching the behavior used by the other
image-aware MOCHA methods.

If no image is available, or if image feature extraction fails, the runner falls
back to expression plus spatial neighbors via `weights_matrix_nomd`. The fallback
is recorded in:

```text
results/ResST/<cohort>/run_metadata.json
```

This MOCHA runner does not call the original ResST `get_data()` reader, so the
benchmark path does not require `stlearn`. ResST still uses deep learning:
PyTorch/PyG for the graph model and torchvision ResNet50 for image features when
images are enabled.

Formal ResST runs save alignment overlays to:

```text
results/ResST/<cohort>/alignment_overlays/<sample>__overlay.png
```

## Environment Setup

The helper scripts assume an existing Python or micromamba environment and take
the environment path as an argument.

Example placeholders:

```bash
PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/env"
```

Create a fresh micromamba environment if needed:

```bash
micromamba create -p "${PY_ENV_ROOT}" -y python=3.11 pip
```

CPU setup:

```bash
chmod +x envs/setup_resst_hpc.sh
chmod +x envs/check_resst_hpc.sh

bash envs/setup_resst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cpu
bash envs/check_resst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
```

GPU setup:

```bash
bash envs/setup_resst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cu121
# or
bash envs/setup_resst_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cu118
```

Notes:

- `cpu`, `cu118`, and `cu121` refer to the PyTorch wheel family
- ResST can share a DeepST-style environment because both use the same
  PyTorch/PyG and scanpy/anndata stack
- the setup script is additive and keeps an existing torch/PyG installation when
  those imports already work
- the setup script pins or constrains compatibility-sensitive packages such as
  `numpy<2`, `setuptools<81`, and `zarr<3`
- for HPC systems with small home quotas, place environments and package caches
  on a project, scratch, or work filesystem when available

## Running ResST

Direct run:

```bash
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"
cd "${PROJECT_ROOT}"

export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/methods/ResST:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"

"${PYTHON_BIN}" code_Xin/main.py --method ResST --cohort DLPFC_10x
```

CPU batch run:

```bash
sbatch scripts/run_resst.slurm DLPFC_10x
sbatch scripts/run_resst.slurm MOB_ST
```

GPU batch run:

```bash
sbatch scripts/run_resst_gpu.slurm DLPFC_10x cu121
sbatch scripts/run_resst_gpu.slurm BC_TNBC_ST cu121
```

Cluster-specific notes:

- `scripts/run_resst.slurm` is a CPU example
- `scripts/run_resst_gpu.slurm` is a GPU example
- you may need to adjust:
  - partition name
  - GPU type
  - memory
  - walltime
  - project path
  - environment path
- image-enabled ResST runs benefit from GPU acceleration
- large cohorts can require long walltimes and substantial memory

## Resume and Progress

ResST writes progress while running:

- `results/ResST/<cohort>/stdout.log`
- `results/ResST/<cohort>/stderr.log`
- `results/ResST/<cohort>/predictions.partial.csv`
- `results/ResST/<cohort>/run_metadata.partial.json`

Each completed integration group writes:

- `results/ResST/<cohort>/<group>/checkpoint_predictions.csv`
- `results/ResST/<cohort>/<group>/group_metadata.json`
- `results/ResST/<cohort>/<group>/_SUCCESS`

Each preprocessed sample writes:

- `results/ResST/<cohort>/<group>/<sample>/resst_enhanced.h5ad`
- `results/ResST/<cohort>/<group>/<sample>/resst_enhanced_metadata.json`

When a job is resubmitted, completed groups are skipped and cached enhanced
sample data are reused. This keeps final benchmark outputs unchanged while
making walltime retries less wasteful.

For very large cohorts, `integration_scope: sample` or `integration_scope:
subgroup` can be used in the cohort-specific experiment config to reduce the
amount of work in a single training group.

## Outputs

ResST writes cohort-level results to:

```text
results/ResST/<cohort>/
```

Main outputs follow the standard MOCHA layout:

- `predictions.csv`
- `predictions.parquet`
- `performance.csv`
- `performance.parquet`
- `evaluation_summary.csv`
- `run_metadata.json`
- `stdout.log`
- `stderr.log`
- `figures/<sample>_true_vs_pred.png`

Additional ResST progress and cache files may be present under group and sample
subdirectories, as described above.

## Parameter Defaults

The current defaults stay close to the reference ResST demo:

- `epochs: 1000`
- `pca_n_comps: 50`
- `rad_cutoff: 150`
- `graph_dist_type: BallTree`
- `cluster_type: leiden`
- `refine: true`
- `refine_shape: auto`
- `use_image: true`
- `cnn_type: ResNet50`

Repository-specific choices:

- each cohort sets `n_domains` to match the corresponding benchmark setup used
  by the other MOCHA methods
- some cohorts run as subgroup-level or sample-level integrations instead of one
  cohort-wide integration
- `10x`-style cohorts generally use `hexagon`
- older `ST`-style cohorts generally use `square`

## Practical Notes

- use GPU jobs for formal image-enabled runs when available
- CPU jobs are mainly useful for smoke tests or small cohorts
- inspect `stdout.log`, `stderr.log`, and `run_metadata.json` together when
  diagnosing failures
- if image-feature extraction failed in an earlier run and cached
  `resst_enhanced.h5ad` files were written, delete those cached files before
  rerunning if the goal is to regenerate image-aware features
> **Branch layout:** This branch contains the reproducibility bundle for ResST only. After cloning or switching to this branch, run `cd ResST` before using the commands below. Input data and generated results are intentionally excluded.
