# stLearn: MOCHA reproducibility guide

## Running this method in MOCHA

This directory contains the method source used by MOCHA. Run it from the repository root through the unified entry point:

```bash
export PROJECT_ROOT="/path/to/MOCHA"
export PY_ENV_ROOT="/path/to/micromamba/envs/mocha_env"
cd "${PROJECT_ROOT}"

bash envs/setup_stlearn_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
bash envs/check_stlearn_hpc.sh "${PY_ENV_ROOT}" "${PROJECT_ROOT}"
"${PY_ENV_ROOT}/bin/python" code_Xin/main.py --method stLearn --cohort DLPFC_10x
```

MOCHA integration files:

- Runner: `code_Xin/runners/run_stlearn.py`
- Method defaults: `configs/methods/stlearn.yaml`
- Cohort overrides: `configs/experiments/stLearn/`
- Environment setup/check: `envs/setup_stlearn_hpc.sh`, `envs/check_stlearn_hpc.sh`
- HPC job: `scripts/run_stlearn.slurm`
- Detailed guide: [docs/stlearn_hpc.md](../../docs/stlearn_hpc.md)

Input data is intentionally not included. Configure local data paths in `configs/cohorts/*.yaml`. Outputs are written under `results/stLearn/<cohort>/` and are also excluded from Git.

---

## Upstream method documentation

<p align="center">
  <img src="https://i.imgur.com/yfXlCYO.png"
    alt="deepreg_logo" title="DeepReg" width="300"/>
</p>

<table align="center">
  <tr>
    <td>
      <b>Package</b>
    </td>
    <td>
      <a href="https://pypi.python.org/pypi/stlearn/">
      <img src="https://img.shields.io/pypi/v/stlearn.svg" alt="PyPI Version">
      </a>
      <a href="https://pepy.tech/project/stlearn">
      <img src="https://static.pepy.tech/personalized-badge/stlearn?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"
        alt="PyPI downloads">
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <b>Documentation</b>
    </td>
    <td>
      <a href="https://stlearn.readthedocs.io/en/latest/">
      <img src="https://readthedocs.org/projects/stlearn/badge/?version=latest" alt="Documentation Status">
      </a>
    </td>
  </tr>
  <tr>
    <td>
     <b>Paper</b>
    </td>
    <td>
      <a href="https://doi.org/10.1038/s41467-023-43120-6"><img src="https://zenodo.org/badge/DOI/10.1038/s41467-023-43120-6.svg"
        alt="DOI"></a>
    </td>
  </tr>
  <tr>
    <td>
      <b>License</b>
    </td>
    <td>
      <a href="https://github.com/BiomedicalMachineLearning/stLearn/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD-blue.svg"
        alt="LICENSE"></a>
    </td>
  </tr>
</table>


# stLearn - A downstream analysis toolkit for Spatial Transcriptomic data

**stLearn** is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, spatial trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample.

---

## Getting Started

- [Documentation and Tutorials](https://stlearn.readthedocs.io/en/latest/)

## Citing stLearn

If you have used stLearn in your research, please consider citing us:

> Pham, Duy, et al. "Robust mapping of spatiotemporal trajectories and cell–cell interactions in healthy and diseased tissues."
> Nature Communications 14.1 (2023): 7739.
> [https://doi.org/10.1101/2020.05.31.125658](https://doi.org/10.1038/s41467-023-43120-6)

