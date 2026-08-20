# HPC Environment Migration

Home quota errors often come from Python environments and pip caches stored
under a user's home directory. Prefer a shared work or scratch filesystem when
the cluster provides one.

Do not move or delete an environment while a job using it is still running.
Wait for active jobs to finish before migrating.

## Define portable paths

Set these values for the target cluster. The examples intentionally contain no
usernames, hostnames, or personal filesystem paths.

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export OLD_MAMBA_ROOT="/path/to/current/micromamba"
export WORK_MAMBA_ROOT="/path/to/work/micromamba"
export WORK_ENV_DIR="${WORK_MAMBA_ROOT}/envs"

mkdir -p "${WORK_ENV_DIR}" "${PROJECT_ROOT}/.cache/pip"
```

## Clone an existing environment

Try a micromamba clone first:

```bash
micromamba create \
  -p "${WORK_ENV_DIR}/mocha_env" \
  --clone "${OLD_MAMBA_ROOT}/envs/mocha_env"

micromamba create \
  -p "${WORK_ENV_DIR}/mocha_stlearn" \
  --clone "${OLD_MAMBA_ROOT}/envs/mocha_stlearn"
```

If cloning is unavailable, recreate the environment from an explicit spec:

```bash
micromamba list \
  -p "${OLD_MAMBA_ROOT}/envs/mocha_env" \
  --explicit > "${PROJECT_ROOT}/mocha_env.explicit.txt"

micromamba create \
  -p "${WORK_ENV_DIR}/mocha_env" \
  --file "${PROJECT_ROOT}/mocha_env.explicit.txt"
```

Repeat the export and recreation steps for `mocha_stlearn` if it is used.

## Verify the migrated environments

```bash
bash "${PROJECT_ROOT}/envs/check_resst_hpc.sh" \
  "${WORK_ENV_DIR}/mocha_env" "${PROJECT_ROOT}"

bash "${PROJECT_ROOT}/envs/check_stlearn_hpc.sh" \
  "${WORK_ENV_DIR}/mocha_stlearn" "${PROJECT_ROOT}"
```

Run a small smoke test before submitting a full benchmark job:

```bash
PY_ENV_ROOT="${WORK_ENV_DIR}/mocha_env" \
PROJECT_ROOT="${PROJECT_ROOT}" \
sbatch --export=ALL scripts/run_resst.slurm DLPFC_10x
```

## Update job configuration

Prefer passing installation paths through environment variables rather than
hard-coding them in scripts:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/work/micromamba/envs/mocha_env"
sbatch --export=ALL scripts/run_deepst.slurm DLPFC_10x
```

## Clean up the old copy

Only remove an old environment after the migrated copy passes validation and no
running or queued job references the old path. Keep the old copy temporarily if
rollback space is available.
