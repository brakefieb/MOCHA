# Starfysh Runner

This document describes the Starfysh integration for the unified MOCHA
benchmark pipeline. The runner follows the same output contract as the other
benchmark methods, including BayeSMART, DeepST, and stLearn.

## Method Summary

Starfysh is a reference-free spatial transcriptomics method that learns
cell-state or cell-type composition features from spatial expression data. This
runner uses pathology image information when spot-to-image alignment is
available.

The workflow is:

1. Load one cohort, subgroup, sample, or shard of `.h5ad` sections.
2. Harmonize genes across sections in the current integration group.
3. Preprocess raw counts with Starfysh-compatible defaults.
4. Discover reference-free signature factors with Starfysh archetypal analysis.
5. Match spots to pathology images with the shared MOCHA spatial alignment
   helper used by image-aware methods.
6. Train Starfysh with the PoE image branch when every section in the group has
   a usable matched image; otherwise run expression-only Starfysh and record the
   reason in `run_metadata.json`.
7. Cluster inferred `qc_m` composition features with KMeans using the
   cohort-specific `K`.
8. Write standardized benchmark outputs under `results/Starfysh/<cohort>/`.

If archetypal analysis fails for a small or numerically unstable group, the
runner falls back to KMeans-derived marker signatures so the run can complete.
The fallback is recorded in `run_metadata.json`.

For benchmark consistency, `parameters.use_image: auto` is the default. Set it
to `false` only for an explicit expression-only ablation.

## Configuration

Default Starfysh parameters live in:

```bash
configs/methods/starfysh.yaml
```

Cohort-specific overrides live in:

```bash
configs/experiments/Starfysh/*.yaml
```

The configured `n_domains` values match the corresponding benchmark settings
used by the other methods:

| Cohort | K | Integration Scope |
| --- | ---: | --- |
| BC_10x | 6 | cohort |
| BC_HER2+_ST | 4 | cohort |
| BC_HP_10x | 6 | subgroup |
| BC_TNBC_ST | 4 | sample |
| CRC_CMS_10x | 6 | subgroup |
| DLPFC_10x | 7 | subgroup |
| KC_TLS_10x | 3 | cohort |
| LC_TLS_10x | 3 | cohort |
| MOB_ST | 5 | sample |
| RCC_TLS_10x | 3 | subgroup |

Subgroup scope follows each cohort YAML `subgroup_map`, matching the multi-case
handling used by the rest of the benchmark pipeline.

## Environment

Use the provided setup and check scripts to prepare a Python environment:

```bash
bash envs/setup_starfysh_hpc.sh <PY_ENV_ROOT> <PROJECT_ROOT> <TORCH_VARIANT>
bash envs/check_starfysh_hpc.sh <PY_ENV_ROOT> <PROJECT_ROOT>
```

Example:

```bash
bash envs/setup_starfysh_hpc.sh /path/to/env /path/to/MOCHA cu121
bash envs/check_starfysh_hpc.sh /path/to/env /path/to/MOCHA
```

`TORCH_VARIANT` can be set to a CUDA wheel tag such as `cu121`, or to `cpu` for
CPU-only runs. GPU execution is recommended for image-aware Starfysh runs.

The setup script installs Starfysh-specific dependencies and the local Starfysh
source tree:

```bash
pip install -e <PROJECT_ROOT>/methods/starfysh --no-deps
```

## Running

Submit one cohort with the provided Slurm script:

```bash
sbatch scripts/run_starfysh.slurm DLPFC_10x
```

Submit all configured cohorts:

```bash
for cohort in BC_10x BC_HER2+_ST BC_HP_10x BC_TNBC_ST CRC_CMS_10x DLPFC_10x KC_TLS_10x LC_TLS_10x MOB_ST RCC_TLS_10x; do
  sbatch scripts/run_starfysh.slurm "$cohort"
done
```

The Slurm script accepts an optional second argument to run one integration
group or one shard:

```bash
sbatch scripts/run_starfysh.slurm RCC_TLS_10x subject22
sbatch scripts/run_starfysh.slurm BC_TNBC_ST "shard:0/8"
```

Useful runtime environment variables:

```bash
AUTO_INSTALL=0 sbatch scripts/run_starfysh.slurm DLPFC_10x
TORCH_VARIANT=cpu sbatch scripts/run_starfysh.slurm DLPFC_10x
PROJECT_ROOT=/path/to/MOCHA PY_ENV_ROOT=/path/to/env sbatch scripts/run_starfysh.slurm DLPFC_10x
```

## Large Cohorts And Partial Runs

Large Starfysh cohorts can exceed common cluster wall-time limits. The runner
supports resumable group-level checkpoints and shard-based partial jobs.

List integration groups for a cohort:

```bash
python scripts/list_starfysh_groups.py RCC_TLS_10x --project_root /path/to/MOCHA
python scripts/list_starfysh_groups.py BC_TNBC_ST --project_root /path/to/MOCHA --shard 0/8
```

Submit one group:

```bash
sbatch scripts/run_starfysh.slurm RCC_TLS_10x subject22
```

Submit shard jobs:

```bash
for shard in 0 1 2 3; do
  sbatch scripts/run_starfysh.slurm BC_TNBC_ST "shard:${shard}/8"
done
```

Resume a timed-out shard by submitting the same shard specification again:

```bash
sbatch scripts/run_starfysh.slurm BC_TNBC_ST "shard:0/8"
```

Completed groups are skipped when their group-level `predictions.csv` checkpoint
already exists. A group that was interrupted before writing this checkpoint will
be rerun from the beginning.

Check partial progress:

```bash
python scripts/check_starfysh_progress.py BC_TNBC_ST --project_root /path/to/MOCHA --shard 0/8
```

Merge partial outputs after all groups or shards finish:

```bash
python scripts/merge_starfysh_partials.py --cohort BC_TNBC_ST --project_root /path/to/MOCHA
```

The merge script recursively collects group-level partial outputs under
`results/Starfysh/<cohort>/_partials/` and writes the final standardized output
tables to `results/Starfysh/<cohort>/`.

## Outputs

Primary outputs are written to:

```bash
results/Starfysh/<cohort>/
```

The standardized prediction schema is:

```text
cohort, sampleID, spotID, x, y, method, z, z_pred
```

Additional provenance columns may include:

```text
group_name, ground_truth, starfysh_cluster, cluster_key
```

The runner also writes:

```text
predictions.csv
predictions.parquet
performance.csv
performance.parquet
evaluation_summary.csv
run_metadata.json
```

Alignment overlays, group checkpoints, gene signatures, loss files, and other
optional intermediate artifacts are stored under the relevant Starfysh result
directory when enabled by configuration.
