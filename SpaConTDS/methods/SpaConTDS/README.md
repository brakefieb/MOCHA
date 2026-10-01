# SpaConTDS: MOCHA reproducibility guide

## Running this method in MOCHA

This directory contains the method source used by MOCHA. Run it from the repository root through the unified entry point:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
cd "${PROJECT_ROOT}"

bash envs/setup_spacontds_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
bash envs/check_spacontds_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
"${PY_ENV_ROOT}/bin/python" code_Xin/main.py --method SpaConTDS --cohort DLPFC_10x
```

MOCHA integration files:

- Runner: `code_Xin/runners/run_spacontds.py`
- Method defaults: `configs/methods/spacontds.yaml`
- Cohort overrides: `configs/experiments/SpaConTDS/`
- Environment setup/check: `envs/setup_spacontds_hpc.sh`, `envs/check_spacontds_hpc.sh`
- HPC job: `scripts/run_spacontds.slurm`
- Detailed guide: [docs/spacontds_hpc.md](../../docs/spacontds_hpc.md)

Input data is intentionally not included. Configure local data paths in `configs/cohorts/*.yaml`. Outputs are written under `results/SpaConTDS/<cohort>/` and are also excluded from Git.

---

## Upstream method documentation

# SpaConTDS

SpaConTDS: A multimodal contrastive learning framework for identifying spatial domains by applying tuple disturbing strategy


## Overview

**SpaConTDS** is a method that combines reinforcement learning with self-supervised multimodal contrastive learning. It constructs positive/negative samples using data augmentation and pseudo-label tuple perturbation strategy, to learn fused representations that capture global semantics and interactions between modalities. It also adapts the mode’s hyperparameters through reinforcement learning. Additionally, it is capable of integrating multiple tissue sections and correcting batch effects without prior alignment.  SpaConTDS can effectively learn the fused represen-tations of multimodal data, providing researchers with a versatile analytical tool that is suitable for a wide range of tasks.
 

## Installation
1. Create conda environment. 
```conda create -n SpaConTDS python=3.9.23```
```conda activate SpaConTDS```

2. Install pytorch & pyG
Install according to your own CUDA version.
-```pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html```
-```pip install torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-2.0.0+cu117.html```
-```pip install torch-sparse==0.6.16+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```
-```pip install torch-scatter==2.1.0+pt113cu117 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```

3. Install required package
```pip install -r ./requirements.txt```


## Get Started

see `./Her2ST_tutorial.ipynb`

We provide the experiment scripts of SpaConTDS under the folder `./scripts`. You can reproduce the experiment results by using `.JSON` file in `./scripts`.

