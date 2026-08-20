# SpaConTDS

This document describes the MOCHA wrapper for
[SpaConTDS](https://github.com/ChengXQ-lab/SpaConTDS), including environment
setup, benchmark configuration, image handling, and expected outputs.

The upstream SpaConTDS source is vendored under:

```text
methods/SpaConTDS/
```

If the source is not present, clone it into that location before applying the
MOCHA compatibility patches:

```bash
git clone https://github.com/ChengXQ-lab/SpaConTDS.git methods/SpaConTDS
```

## Environment

The original SpaConTDS README used Python 3.9 with an older PyTorch/PyG stack.
The MOCHA wrapper is tested with a Python 3.11 environment and a CUDA-enabled
PyTorch/PyG stack. Use a project or scratch/work filesystem for the environment
on HPC systems when home-directory quota is limited.

Create an environment:

```bash
micromamba create -p /path/to/spacontds_env -y python=3.11 pip
```

Install and verify SpaConTDS dependencies:

```bash
PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/spacontds_env"

bash envs/setup_spacontds_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}" cu121
bash envs/check_spacontds_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
```

Use `cpu`, `cu118`, or `cu121` as the third setup argument depending on the
target PyTorch build. If an existing compatible torch/PyG stack is already
installed, the setup script keeps it instead of reinstalling it.

The setup, check, and run scripts intentionally clear inherited `PYTHONPATH`,
`PYTHONHOME`, `LD_LIBRARY_PATH`, and active conda/mamba environment variables
before starting Python. This avoids mixing PyTorch from one environment with
PyG extension wheels from another environment. The check script prints
`sys.executable` and `torch.__file__`; both should resolve under the same
environment root.

## Running

Run a cohort through SLURM:

```bash
sbatch scripts/run_spacontds.slurm DLPFC_10x cu121
```

The provided SLURM script requests one GPU, 8 CPU threads, 128 GB memory, and a
48-hour walltime:

```text
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
```

SpaConTDS uses PyTorch and benefits from a CUDA GPU. A single modern GPU is the
recommended default. The script also sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` from the
SLURM CPU allocation so CPU-side preprocessing can use the requested threads.

For a direct run without SLURM:

```bash
PROJECT_ROOT="/path/to/MOCHA"
PY_ENV_ROOT="/path/to/spacontds_env"
PYTHON_BIN="${PY_ENV_ROOT}/bin/python"

cd "${PROJECT_ROOT}"
unset PYTHONHOME
unset PYTHONPATH
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PATH="${PY_ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${PY_ENV_ROOT}/lib"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/code_Xin:${PROJECT_ROOT}/methods/SpaConTDS"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"

"${PYTHON_BIN}" code_Xin/main.py \
  --method SpaConTDS \
  --cohort DLPFC_10x
```

## MOCHA Adapter

Upstream SpaConTDS contains dataset-specific readers. MOCHA adds a `MOCHA`
reader branch in `methods/SpaConTDS/utils.py` and prepares one temporary h5ad
plus one image per sample under:

```text
results/SpaConTDS/<cohort>/<sample_id>/
```

Each sample also gets an isolated upstream working directory:

```text
results/SpaConTDS/<cohort>/<sample_id>/workdir/
```

This is necessary because upstream SpaConTDS writes temporary model files to
relative `./model` and `./Decoder` paths. Per-sample working directories keep
those checkpoints from leaking across samples in cohort-level benchmark jobs.

The wrapper also patches several compatibility issues for modern Python,
AnnData, and PyG environments:

- loads SpaConTDS `main.py` by absolute path to avoid name conflicts with
  `code_Xin/main.py`;
- avoids deprecated private AnnData sparse view classes;
- replaces PyG neighbor sampling with a local sampler when binary PyG
  extensions are unavailable or incompatible;
- guards singleton batches in BatchNorm and contrastive-loss shape handling;
- saves temporary model snapshots with deterministic names instead of floating
  point reward values;
- falls back from `seurat_v3` highly-variable-gene selection to `cell_ranger`
  or variance ranking if `skmisc.loess` fails on numerically ill-conditioned
  samples.

## Benchmark Parameters

The runner keeps the original SpaConTDS demo defaults where possible and maps
the benchmark domain count `K` to both upstream `valid_cluster` and
`pseudo_cluster`.

| Cohort | K |
| --- | ---: |
| BC_10x | 6 |
| BC_HER2+_ST | 4 |
| BC_HP_10x | 6 |
| BC_TNBC_ST | 4 |
| CRC_CMS_10x | 6 |
| DLPFC_10x | 7 |
| KC_TLS_10x | 3 |
| LC_TLS_10x | 3 |
| MOB_ST | 5 |
| RCC_TLS_10x | 3 |

Method-level defaults are stored in:

```text
configs/methods/spacontds.yaml
```

Cohort-specific overrides are stored in:

```text
configs/experiments/SpaConTDS/<cohort>.yaml
```

## Pathology Image Handling

SpaConTDS requires an image path and extracts one pathology-image patch per
spot. The MOCHA runner therefore uses the shared spot-to-image alignment helper
also used by image-aware methods such as stLearn and ResST. It converts h5ad
coordinates into image pixel coordinates before writing the SpaConTDS input
h5ad.

If a sample has no usable image, or if spot-to-image alignment fails confidence
checks, the runner records that in metadata and creates a blank fallback image
so upstream SpaConTDS can still complete. In that case
`pathology_image_used=false` appears in `sample_metadata.json`, and the result
should be interpreted as a non-morphology fallback run rather than a true
pathology-image run.

## Resume Behavior

SpaConTDS can be slow for large cohorts. The wrapper checkpoints each completed
sample, so a cohort job can be resubmitted after walltime expiration without
rerunning successful samples.

Resume files:

- `results/SpaConTDS/<cohort>/predictions.partial.csv`
- `results/SpaConTDS/<cohort>/run_metadata.partial.json`
- `results/SpaConTDS/<cohort>/<sample_id>/checkpoint_predictions.csv`
- `results/SpaConTDS/<cohort>/<sample_id>/sample_metadata.json`
- `results/SpaConTDS/<cohort>/<sample_id>/_SUCCESS`

Do not delete these files when resuming a cohort. The runner skips samples only
when both `_SUCCESS` and `checkpoint_predictions.csv` are present.

## Outputs

Final outputs follow the standard MOCHA layout:

- `results/SpaConTDS/<cohort>/predictions.csv`
- `results/SpaConTDS/<cohort>/predictions.parquet`
- `results/SpaConTDS/<cohort>/performance.csv`
- `results/SpaConTDS/<cohort>/performance.parquet`
- `results/SpaConTDS/<cohort>/evaluation_summary.csv`
- `results/SpaConTDS/<cohort>/figures/<sampleID>_true_vs_pred.png`
- `results/SpaConTDS/<cohort>/run_metadata.json`
- `results/SpaConTDS/<cohort>/stdout.log`
- `results/SpaConTDS/<cohort>/stderr.log`
> **Branch layout:** This branch contains the reproducibility bundle for SpaConTDS only. After cloning or switching to this branch, run `cd SpaConTDS/MOCHA` before using the commands below. Input data and generated results are intentionally excluded.
