# stGCL in the MOCHA Benchmark

This document describes the MOCHA integration of
[stGCL](https://github.com/RuiGaolab/stGCL), a graph contrastive learning method
for spatial transcriptomics. The integration provides standardized data loading,
image-to-spot alignment, cohort-specific configuration, checkpointing, HPC
execution, and benchmark-compatible outputs.

## Method overview

stGCL combines three sources of information:

- gene expression;
- spatial coordinates;
- histology image features, when a correctly aligned H&E image is available.

The image branch extracts one H-ViT representation per spatial spot. Gene and
image representations are fused through a graph attention autoencoder and a
contrastive learning objective. The resulting embedding is clustered into
spatial domains and optionally refined using spatial neighbors.

The upstream source is vendored under `methods/stGCL/`. Its repository and exact
commit are recorded in `methods/stGCL/UPSTREAM_COMMIT`.

## MOCHA integration files

```text
methods/stGCL/                         upstream stGCL source
code_Xin/runners/run_stgcl.py          MOCHA runner
configs/methods/stgcl.yaml             method defaults
configs/experiments/stGCL/*.yaml       cohort-specific K values
envs/setup_stgcl_hpc.sh                dependency setup
envs/check_stgcl_hpc.sh                environment validation
scripts/check_stgcl_image_alignment.py image-alignment audit
scripts/run_stgcl.slurm                 single-cohort Slurm job
scripts/submit_stgcl_all.sh             all-cohort submission helper
```

## Input data

Each sample requires an AnnData file and spatial coordinates. An H&E image is
optional at the file-format level, but is required to run the multimodal image
branch.

A typical cohort layout is:

```text
data/<cohort>/
├── data/
│   ├── SCE_<sample>.h5ad
│   └── ...
└── images/
    ├── HE_<sample>.png
    ├── <sample>_tissue_positions_list.csv
    ├── <sample>_scalefactors_json.json
    └── ...
```

The corresponding cohort definition belongs in
`configs/cohorts/<cohort>.yaml`. For example:

```yaml
cohort_name: Example_10x
has_he_image: true

data_dir: data/Example_10x/data
image_dir: data/Example_10x/images
h5ad_glob: "SCE_*.h5ad"
image_glob: "HE_*.*"
match_by_stem: false

spatial_alignment:
  mode: visium_sidecar
  fallback_mode: disable_image
  save_overlay: true
  visium_scale_preference: hires
```

Spatial coordinates may be read from supported columns in `adata.obs` or from
`adata.obsm['spatial']`, `adata.obsm['X_spatial']`, or `adata.obsm['S']`.

## Image-to-spot matching

Image alignment is validated before H-ViT feature extraction. The runner:

1. matches each AnnData sample to its pathology image;
2. resolves pixel coordinates using Visium sidecars, direct pixel coordinates,
   or a configured transformation matrix;
3. writes an alignment overlay for visual inspection;
4. creates one `256 x 256` image patch per spot using the convention
   `x = image column` and `y = image row`;
5. preserves AnnData row order and verifies that the image feature rows match
   the gene-expression rows.

Alignment overlays are written to:

```text
results/stGCL/<cohort>/alignment_overlays/
```

Audit alignment before a full training run:

```bash
python scripts/check_stgcl_image_alignment.py --cohort DLPFC_10x
```

To make an alignment failure return a nonzero exit status:

```bash
python scripts/check_stgcl_image_alignment.py \
  --cohort DLPFC_10x \
  --require-image
```

When `fallback_to_expression_only: true`, a missing or unusable image causes the
runner to use the expression-and-spatial branch. The reason is recorded in the
sample metadata; the fallback is not silent.

## Configuration

Method defaults are defined in `configs/methods/stgcl.yaml`. Important settings
include:

```yaml
runtime:
  level: cohort
  seed: 0
  device: auto
  image_batch_size: 64

parameters:
  n_domains: auto
  use_image: auto
  fallback_to_expression_only: true
  top_genes: 3000
  image_crop_size: 256
  vit_patch_size: 64
  image_pca_n_comps: 50
  graph_model: KNN
  graph_k: 6
  hidden_dims: [100, 30]
  n_epochs: auto
  refine_neighbors: 50
```

The included benchmark cohorts use the same K values as the other MOCHA
methods:

| Cohort | K |
|---|---:|
| `BC_10x` | 6 |
| `BC_HER2+_ST` | 4 |
| `BC_HP_10x` | 6 |
| `BC_TNBC_ST` | 4 |
| `CRC_CMS_10x` | 6 |
| `DLPFC_10x` | 7 |
| `KC_TLS_10x` | 3 |
| `LC_TLS_10x` | 3 |
| `MOB_ST` | 5 |
| `RCC_TLS_10x` | 3 |

## Environment setup

stGCL uses the same `mocha_env` environment as the other benchmark methods.
Define local paths rather than editing scripts with user- or cluster-specific
values:

```bash
export PROJECT_ROOT=/path/to/MOCHA
export MOCHA_ENV=/path/to/micromamba/envs/mocha_env
cd "$PROJECT_ROOT"
```

Install the required packages into that environment:

```bash
bash envs/setup_stgcl_hpc.sh "$MOCHA_ENV" "$PROJECT_ROOT" cu121
```

Supported PyTorch variants are `cpu`, `cu118`, and `cu121`. Select the variant
compatible with the compute node and installed CUDA driver.

Validate the environment:

```bash
bash envs/check_stgcl_hpc.sh "$MOCHA_ENV" "$PROJECT_ROOT"
```

The setup script writes an stGCL-specific readiness marker inside `mocha_env`.
Set `FORCE_SETUP=1` to reinstall or refresh the stGCL dependencies:

```bash
FORCE_SETUP=1 bash envs/setup_stgcl_hpc.sh \
  "$MOCHA_ENV" "$PROJECT_ROOT" cu121
```

## Running locally

Activate the shared environment and run one cohort through the unified MOCHA
entry point:

```bash
export PATH="$MOCHA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/methods/stGCL:${PYTHONPATH:-}"

python code_Xin/main.py --method stGCL --cohort DLPFC_10x
```

## Running with Slurm

The supplied Slurm script requests one GPU, 8 CPU cores, 360 GB of memory, and a
48-hour time limit. Adjust the partition, GPU resource syntax, memory, and time
directives in `scripts/run_stgcl.slurm` to match the target cluster.

The script accepts `PROJECT_ROOT` and `PY_ENV_ROOT` as environment variables:

```bash
export PROJECT_ROOT=/path/to/MOCHA
export PY_ENV_ROOT=/path/to/micromamba/envs/mocha_env

sbatch --export=ALL scripts/run_stgcl.slurm DLPFC_10x
```

Submit all configured cohorts as separate jobs:

```bash
export PROJECT_ROOT=/path/to/MOCHA
export PY_ENV_ROOT=/path/to/micromamba/envs/mocha_env

bash scripts/submit_stgcl_all.sh
```

Cluster administrators may require different values for `--partition`,
`--gres`, or `--mem`. Those settings are scheduling policy, not method
requirements.

## Checkpointing and restart behavior

The runner saves per-sample predictions and H-ViT PCA features. A restarted
cohort skips a completed sample only when its checkpoint signature still
matches the input files, method parameters, and random seed. Changed inputs or
configuration invalidate the old checkpoint automatically.

Per-sample checkpoints are stored under:

```text
results/stGCL/<cohort>/<sample_id>/
```

## Outputs

Benchmark-level outputs follow the same schema as the other MOCHA methods:

```text
results/stGCL/<cohort>/
├── predictions.csv
├── predictions.parquet
├── performance.csv
├── performance.parquet
├── evaluation_summary.csv
├── run_metadata.json
├── stdout.log
├── stderr.log
├── alignment_overlays/
└── figures/
    └── <sample_id>_true_vs_pred.png
```

`predictions.csv` and `predictions.parquet` include the standard benchmark
columns:

```text
cohort, sampleID, spotID, x, y, method, z, z_pred
```

They also include stGCL-specific fields such as `stGCL_domain`,
`stGCL_refined`, `image_used`, and `alignment_mode`.

## Differences from the upstream tutorial

The MOCHA runner keeps the upstream preprocessing, graph construction, stGCL
training, and spatial refinement workflow, with the following integration
choices:

- input is read from benchmark AnnData files instead of tutorial-specific
  directories;
- image alignment is checked before patches are extracted;
- H-ViT features and sample predictions are checkpointed;
- clustering uses a tied-covariance Gaussian mixture, corresponding to the
  shared covariance structure of `mclust` model `EEE`, without requiring an
  R/rpy2 runtime;
- benchmark labels are used only for evaluation, not for training or
  cluster-label matching; K is supplied by the experiment configuration;
- outputs conform to the common MOCHA result schema.

## Troubleshooting

### Image branch is disabled

Inspect the overlay and the `image` section of:

```text
results/stGCL/<cohort>/<sample_id>/sample_metadata.json
```

Confirm that the image, tissue-position sidecar, scale-factor file, and AnnData
barcodes belong to the same sample.

### CUDA out-of-memory error

Reduce `runtime.image_batch_size` in `configs/methods/stgcl.yaml` or in a
cohort-specific experiment override. H-ViT extraction and graph training have
different memory profiles, so monitor both stages.

### A job reaches the scheduler time limit

Resubmit the same cohort. Valid per-sample checkpoints are reused. For cohorts
with many samples, submitting cohorts independently limits the amount of work
lost when a job is interrupted.

## Citation and license

Please cite the original stGCL work when using this integration. See the
upstream [stGCL repository](https://github.com/RuiGaolab/stGCL) for its citation
information and license. The vendored upstream source retains its original
license file under `methods/stGCL/LICENSE`.
