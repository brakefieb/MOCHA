# STMSC integration for MOCHA

This document describes the MOCHA integration of
[`bliulab/STMSC`](https://github.com/bliulab/STMSC), including its input
adaptations, configuration, image alignment, local smoke tests, Slurm execution,
checkpoint recovery, and standardized outputs.

STMSC runs through the common MOCHA entry point:

```bash
python code_Xin/main.py --method STMSC --cohort <cohort>
```

The integration writes the same benchmark deliverables as the other MOCHA spatial
domain methods.

## Method source and environment

The upstream source is expected at `methods/STMSC`. If it is not already included
in the checkout, retrieve it from the official repository:

```bash
git clone https://github.com/bliulab/STMSC.git methods/STMSC
```

STMSC is installed into the existing MOCHA Python environment. Define paths for
your installation before running the setup scripts:

```bash
export PROJECT_ROOT=/path/to/MOCHA
export PY_ENV_ROOT=/path/to/mocha_env

cd "$PROJECT_ROOT"
bash envs/setup_stmsc_hpc.sh "$PY_ENV_ROOT" "$PROJECT_ROOT"
bash envs/check_stmsc_hpc.sh "$PY_ENV_ROOT" "$PROJECT_ROOT"
```

The setup is additive: it retains the environment's existing PyTorch stack, keeps
`numpy<2` for compatibility with other integrated methods, installs missing STMSC
dependencies, and installs the upstream package with `--no-deps`.

The supplied Slurm scripts read `PROJECT_ROOT` and `PY_ENV_ROOT` from the
environment. Set them to paths visible on every compute node.

## Inputs and image use

Each section requires:

- an `.h5ad` expression matrix;
- spatial coordinates in a supported `obs` or `obsm` field;
- a cohort configuration under `configs/cohorts/`.

An H&E image is optional, but STMSC **does use pathology image information when a
valid image alignment is available**. The released `extract_histology_features()`
function extracts a local RGB patch at every spot (`49 x 49` pixels by default)
and converts its colour signal into a third spatial coordinate.

Before STMSC sees an image, the MOCHA runner uses the shared
`code_Xin/spatial_alignment.py` resolver to:

1. match each `.h5ad` section to its image;
2. obtain image-pixel coordinates from Visium sidecars, direct pixel coordinates,
   or a configured ST transformation matrix;
3. generate a spot-on-image overlay;
4. extract image patches only if the alignment passes the configured confidence
   checks.

Missing or low-confidence image alignment is recorded in metadata and falls back
to expression plus spatial coordinates. The runner does not extract patches from
unverified coordinates.

### Validate image alignment

Run the alignment checker before a full benchmark when images or spatial metadata
have changed:

```bash
cd "$PROJECT_ROOT"
"$PY_ENV_ROOT/bin/python" scripts/check_stmsc_image_alignment.py \
  --cohort DLPFC_10x
```

Inspect:

```text
results/STMSC/DLPFC_10x/image_alignment_check.csv
results/STMSC/DLPFC_10x/alignment_overlays/*.png
```

## Reference adaptation

The released STMSC preprocessing API requires a labelled scRNA-seq reference. The
ten MOCHA cohorts do not share a compatible reference, and the released latent
training network does not consume the computed reference basis.

The MOCHA runner therefore constructs a deterministic, label-free
pseudo-reference from a subsample of the current execution group. It is used only
to satisfy the upstream highly-variable-gene selection interface. Benchmark
ground-truth annotations are never used to construct this reference. The
adaptation is recorded in each `group_metadata.json` and in `run_metadata.json`.

## Execution groups and memory

STMSC constructs dense pairwise matrices and therefore has approximately
quadratic memory growth in the number of spots. MOCHA avoids combining unrelated
patients and supports resumable Slurm execution by defining groups as follows:

- sections in the same configured subject/donor subgroup are integrated together;
- sections without pairing metadata run independently;
- each group writes an independent checkpoint and success marker;
- the final completed group safely aggregates cohort-level outputs under a file
  lock.

List the deterministic group indices for a cohort:

```bash
"$PY_ENV_ROOT/bin/python" scripts/list_stmsc_groups.py \
  --cohort DLPFC_10x
```

The default dense-graph guard is 18,000 spots per group. A cohort-specific config
may raise it when sufficient memory has been provisioned. Do not increase this
limit blindly: host RAM and GPU memory requirements grow rapidly.

## Configuration

The method defaults are in:

```text
configs/methods/stmsc.yaml
```

Cohort-specific overrides are in:

```text
configs/experiments/STMSC/<cohort>.yaml
```

Important defaults include:

- 5,000 mapping epochs;
- 5,000 latent-training epochs;
- automatic CUDA selection when a GPU is available;
- Gaussian mixture clustering with covariance regularization and K-means fallback;
- resumable mapping and latent checkpoints;
- optional histology with alignment validation.

### Cohort domain counts

| Cohort | K |
|---|---:|
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

These values match the corresponding MOCHA experiment configs for the other
spatial-domain methods.

## Smoke test

A two-epoch run can validate imports, preprocessing, CUDA access, and the output
path without performing a benchmark-quality training run:

```bash
cd "$PROJECT_ROOT"
"$PY_ENV_ROOT/bin/python" code_Xin/main.py \
  --method STMSC \
  --cohort DLPFC_10x \
  --group_idx 0 \
  --mapping_epochs 2 \
  --epochs 2
```

Do not compare smoke-test predictions with full benchmark results. Use a separate
results checkout or remove the smoke-test group artifacts before starting the full
run, because completed groups are intentionally resumed.

## Slurm execution

The provided job script requests one GPU, 8 CPUs, 360 GB host RAM, and a 48-hour
wall-time. Its default partition may need to be changed for a different cluster.
Slurm options can also be overridden at submission time.

Export installation paths before submission:

```bash
export PROJECT_ROOT=/shared/path/to/MOCHA
export PY_ENV_ROOT=/shared/path/to/mocha_env
cd "$PROJECT_ROOT"
```

Submit one group:

```bash
sbatch scripts/run_stmsc.slurm DLPFC_10x 0
```

Submit all three DLPFC donor groups with at most three concurrent tasks:

```bash
sbatch --array=0-2%3 scripts/run_stmsc.slurm DLPFC_10x
```

To determine an array range without hard-coding it:

```bash
n_groups=$("$PY_ENV_ROOT/bin/python" \
  scripts/list_stmsc_groups.py --cohort DLPFC_10x --count)
sbatch --array="0-$((n_groups - 1))%3" \
  scripts/run_stmsc.slurm DLPFC_10x
```

### Submit all configured cohorts

```bash
MAX_CONCURRENT=4 bash scripts/submit_stmsc_all.sh
```

Skip cohorts that are already complete:

```bash
SKIP_COHORTS=BC_10x MAX_CONCURRENT=4 \
  bash scripts/submit_stmsc_all.sh
```

Multiple exclusions may be comma- or space-separated:

```bash
SKIP_COHORTS=BC_10x,DLPFC_10x MAX_CONCURRENT=4 \
  bash scripts/submit_stmsc_all.sh
```

The submission helper queries the current cohort configs for group counts. A
failed array task can be resubmitted with the same cohort and group index.

## Checkpoints and recovery

A group is considered complete only when both files exist:

```text
results/STMSC/<cohort>/groups/<group>/checkpoint_predictions.csv
results/STMSC/<cohort>/groups/<group>/_SUCCESS
```

Resubmitting a complete group skips training. Incomplete groups resume from any
valid mapping or latent checkpoint that was written before the failure. It is safe
to resubmit the entire array when the failed group indices are unknown; completed
groups are skipped.

## Outputs

After all groups in a cohort finish, the runner creates:

```text
results/STMSC/<cohort>/
├── predictions.csv
├── predictions.parquet
├── performance.csv
├── performance.parquet
├── evaluation_summary.csv
├── run_metadata.json
├── _SUCCESS
├── figures/
├── alignment_overlays/
└── groups/<group>/
    ├── checkpoint_predictions.csv
    ├── group_metadata.json
    └── _SUCCESS
```

`performance.runtime` is the sum of group runtimes, while `performance.memory` is
the maximum observed group peak RSS. This keeps the standardized fields meaningful
for array-based execution.

## Troubleshooting

- Scanpy/AnnData `FutureWarning` messages are informational and do not indicate a
  failed run. Diagnose the final traceback instead.
- If the environment check reports that CUDA is unavailable inside a GPU job,
  verify that the environment contains a CUDA-enabled PyTorch build compatible
  with the cluster driver.
- If the dense-graph guard is exceeded, inspect group membership and spot counts.
  Prefer biologically meaningful grouping over raising the limit.
- If a job ends during final clustering or export, resubmit the same group; the
  mapping and latent checkpoints prevent unnecessary retraining when available.

## Upstream project

STMSC is developed by the authors of:

> D. Zhang, R. Qi, X. Lan, and B. Liu. A Novel Multi-Slice Framework for Precision
> 3D Spatial Domain Reconstruction and Disease Pathology Analysis. Genome Research
> (2025). DOI: 10.1101/gr.280281.124.

See the [upstream STMSC repository](https://github.com/bliulab/STMSC) for the
original implementation, installation requirements, tutorials, and license.
