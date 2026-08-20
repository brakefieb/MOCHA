# STMSC: MOCHA reproducibility guide

## Running this method in MOCHA

This directory contains the method source used by MOCHA. Run it from the repository root through the unified entry point:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
cd "${PROJECT_ROOT}"

bash envs/setup_stmsc_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
bash envs/check_stmsc_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
"${PY_ENV_ROOT}/bin/python" code_Xin/main.py --method STMSC --cohort DLPFC_10x
```

MOCHA integration files:

- Runner: `code_Xin/runners/run_stmsc.py`
- Method defaults: `configs/methods/stmsc.yaml`
- Cohort overrides: `configs/experiments/STMSC/`
- Environment setup/check: `envs/setup_stmsc_hpc.sh`, `envs/check_stmsc_hpc.sh`
- HPC job: `scripts/run_stmsc.slurm`
- Detailed guide: [docs/stmsc_hpc.md](../../docs/stmsc_hpc.md)

Input data is intentionally not included. Configure local data paths in `configs/cohorts/*.yaml`. Outputs are written under `results/STMSC/<cohort>/` and are also excluded from Git.

---

## Upstream method documentation

# STMSC
A novel multi-slice framework for precision 3D spatial domain reconstruction and disease pathology analysis
![image text](https://github.com/bliulab/STMSC/blob/main/Figures.png)
**Reference**: D. Zhang, R. Qi, X. Lan, and B. Liu, "A Novel Multi-Slice Framework for Precision 3D Spatial Domain Reconstruction and Disease Pathology Analysis," *Genome Research*,  DOI: 10.1101/gr.280281.124, 2025.
# Installation
The STMSC package is developed based on Python and supports GPU acceleration (recommended) and CPU execution.
## Step 1: Clone the Repository
```
git clone https://github.com/bliulab/STMSC.git
cd STMSC
```
## Step 2: Create a Conda Environment
We recommend creating a separate environment for running STMSC:
```
# Create a conda environment named env_STMSC with Python 3.10
conda create -n env_STMSC python=3.10
# Activate the environment
conda activate env_STMSC
```
## Step 3: Install Required Packages
For Linux:
```
pip install -r requirements.txt
```
## Step 4: Install STMSC
```
python setup.py build
python setup.py install
```
## Tutorials and reproducibility
We provided codes for reproducing the experiments of the paper "A novel multi-slice framework for precision 3D spatial domain reconstruction and disease pathology analysis", and comprehensive tutorials for using STMSC. Please check [the tutorial website](https://stmsc-tutorial.readthedocs.io/en/latest/) for more details.
## Parameter Settings

In STMSC, several hyperparameters are used to control the loss weighting across different components of the framework. Below we explain the meaning of each parameter and provide the settings used for various datasets.

### Parameter Definitions

- **`lam`**: Weight of the loss term in the **deconvolution** module.
- **`bl`**: Weight for incorporating deconvolution-informed features during the **graph correction** step.
- **`bll`**: Weight for enforcing alignment across adjacent slices during **graph correction**. 

> ℹ️ **Tip**: These parameters may need to be adjusted depending on tissue type, slice resolution, or biological heterogeneity.

### Dataset-Specific Settings

```text
# General training configuration
Train_model: epoch=5000, lr=0.01

# Dataset-specific hyperparameter settings
LIBD-151507-151510:       lam=5, bl=0.5, bll=0.1
LIBD-151669-151672:       lam=1, bl=0.6, bll=0.1
LIBD-151673-151676:       lam=7, bl=0.1, bll=0.1
Human breast cancer:      lam=7, bl=0.1
Mouse brain:              lam=3, bl=0.6
Human HER2 breast cancer: lam=9, bl=0.2, bll=0.1
```
## Hardware specifications
1. Intel(R) Xeon(R) w5-3435X, NVIDIA RTX A6000
2. 13th Gen Intel(R) Core(TM) i9-13900KF, NVIDIA GeForce RTX 4090

