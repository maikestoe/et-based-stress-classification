# Stress Classification from Eye-Tracking Time Series

This repository contains the code accompanying the manuscript:

**Classifying Mental Stress from Eye Tracking Data: Deep Learning Approaches for Out-of-the-Lab Conditions**
The code supports preprocessing, feature-based baseline training, deep-learning
training, evaluation, and supplementary analyses for two datasets:

- **VR goalkeeper dataset** (publicly available)
- **ForDigitStress dataset** (restricted access)

The datasets themselves are not redistributed in this repository and must be
obtained from their original providers.

Dataset sources:

- **VR goalkeeper dataset** (public):
  Zenodo, DOI [10.5281/zenodo.17972964](https://doi.org/10.5281/zenodo.17972964)

- **ForDigitStress dataset** (restricted):
  Heimerl et al., DOI [10.1109/TAFFC.2024.3501400](https://doi.org/10.1109/TAFFC.2024.3501400)  
  Access must be requested via [hcai.eu/fordigitstress/](https://hcai.eu/fordigitstress/)  
  and is subject to the dataset’s EULA (non-commercial use, no redistribution).
- 
## Data Access and License Notes

This repository contains code only and does not include raw data, processed data,
trained models, or dataset-derived artefacts.

Users are responsible for complying with the licenses and access conditions of
the respective datasets. The VR goalkeeper dataset is publicly available and
should be cited via its Zenodo record. 

The ForDigitStress dataset requires
authorized access and must be used in accordance with its EULA. Do not share or 
publish raw data, derived datasets, trained models, or any other
artefacts that would enable redistribution of the datasets or bypass their access 
restrictions.

When using the ForDigitStress dataset, please acknowledge it according to its
EULA, for example:

> (Portions of) the research in this work use the ForDigitStress Dataset
> collected for the ForDigitHealth project.

## Data Access Procedures

The repository provides preprocessing scripts documenting how each dataset is
loaded and transformed.

- **VR goalkeeper dataset** (public):
  - download from Zenodo,
  - place files in `data/vr_goalkeeper/raw/`,
  - run:
    ```bash
    python scripts/preprocessing/preprocess_vr_goalkeeper.py
    ```
  - example:
    ```python
    import pandas as pd
    df = pd.read_csv("data/vr_goalkeeper/raw/LogID_0_base1.csv", sep=";")
    ```

- **ForDigitStress dataset** (restricted):
  - request access via https://hcai.eu/fordigitstress/,
  - place authorized files in `data/fordigitstress/raw/`,
  - run:
    ```bash
    python scripts/preprocessing/preprocess_fordigitstress.py
    ```
  - example:
    ```python
    import pandas as pd
    df = pd.read_csv("data/fordigitstress/raw/VP1/stress.csv", sep=";")
    ```

The preprocessing scripts define the expected structure and processing steps for
both datasets.

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── data/
│   └── README.md
├── configs/
│   └── examples/
├── batch_scripts/
│   └── examples/
├── scripts/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── analysis/
└── src/
```

- `src/` contains the reusable preprocessing, model-training, evaluation, and plotting code.
- `scripts/preprocessing/` contains dataset preprocessing entry points.
- `scripts/training/` contains training entry points for deep-learning models and the feature-based baseline.
- `scripts/evaluation/` contains prediction recovery and replotting entry points.
- `scripts/analysis/` contains supplementary analyses such as confidence intervals, calibration-related plots, statistical testing, attribution plots, noise/error analyses, and model-comparison figures.
- `configs/examples/` contains representative example configurations only. The full experiment matrix used in the paper consisted of many closely related configurations and is intentionally not duplicated one-by-one here.
- `batch_scripts/examples/` contains a generic SLURM example showing how to launch a configuration on a cluster.

## Environment

The original analyses used TensorFlow/Keras, scikit-learn, Optuna, pandas,
NumPy, SciPy, matplotlib, seaborn, imbalanced-learn, and related utilities.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Depending on your platform, TensorFlow installation may require a platform-specific
package selection. If installation through `pyproject.toml` is not suitable for
your machine, install the dependencies manually following the versions listed
there.

## Data Layout

Create the following folder structure after downloading the public datasets:

```text
data/
├── vr_goalkeeper/
│   ├── raw/
│   └── dataframes/
│       ├── DL_out.pkl
│       └── features_out.pkl
└── fordigitstress/
    ├── raw/
    └── dataframes/
        └── DL_out.pkl
```

The preprocessing scripts generate the `dataframes/` files used by the training
and evaluation code. If you already have preprocessed files, you can place them
directly in the expected locations and start from training or evaluation.

## Model Checkpoints

The repository enables reproduction through dataset access, preprocessing,
training code, configuration files, and evaluation scripts.

- **VR goalkeeper dataset (public)**:  
  All model checkpoints were generated during the study. However, the full set
  comprises hundreds of fold-specific models (30 configurations × 27 folds),
  which are not included due to their size and limited practical usefulness.
  Representative checkpoints can be provided by the authors upon request.

- **ForDigitStress dataset (restricted)**:  
  Model checkpoints are not distributed. Due to the dataset’s restricted-access
  EULA, derivative artefacts such as trained models are not shared through this
  repository.

## Reproducing the Paper Results

The full paper results were produced with nested leave-one-subject-out
cross-validation and Optuna-based hyperparameter optimization. This is
computationally expensive. The commands below show the reproducibility workflow
using representative example configurations.

For a quick local sanity check before launching full runs, see
[`docs_test_matrix.md`](docs_test_matrix.md). The test matrix covers
preprocessing entry points, RF baseline training, deep-learning smoke tests for
both datasets, and representative analysis scripts. The deep-learning smoke
tests use only two outer test participants, one Optuna trial, and two epochs to
verify that installation, data paths, and model training work locally.

For a mapping between internal experiment IDs such as `exp1`, `exp41`, or
`exp201` and the paper-facing model/input-signal names, see
[`docs_experiment_overview.md`](docs_experiment_overview.md).

### 1. Preprocess the Datasets

VR goalkeeper dataset:

```bash
python scripts/preprocessing/preprocess_vr_goalkeeper.py \
  --raw_data_path data/vr_goalkeeper/raw \
  --output_path data/vr_goalkeeper/dataframes
```

ForDigitStress dataset:

```bash
python scripts/preprocessing/preprocess_fordigitstress.py \
  --raw_data_path data/fordigitstress/raw \
  --output_path data/fordigitstress/dataframes
```

### 2. Train Deep-Learning Models

Example: CNN with pupil-diameter input on the VR goalkeeper dataset:

```bash
python scripts/training/train_deep_learning.py \
  --config configs/examples/vr_goalkeeper_cnn_pd.json
```

Example: CNN with pupil-diameter input on the ForDigitStress dataset:

```bash
python scripts/training/train_deep_learning.py \
  --config configs/examples/fordigitstress_cnn_pd_recovery.json
```

Example: ConvLSTM-3 with asymptotic-model input on the VR goalkeeper dataset:

```bash
python scripts/training/train_deep_learning.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json
```

### 3. Train the Feature-Based Baseline

```bash
python scripts/training/train_rf_baseline_vr_goalkeeper.py
```

The feature-based baseline uses task-agnostic pupil-diameter statistics and
fixation characteristics. The supplementary feature-selection plot is generated
from the stored baseline results.

### 4. Recover Predictions and Attribution Maps

After model training, recover held-out predictions and optional saliency or
occlusion maps:

```bash
python scripts/evaluation/recover_predictions.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json
```

Attribution plots can then be regenerated, for example:

```bash
python scripts/evaluation/replot_attribution_maps.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json \
  --methods saliency occlusion \
  --colormap blue_yellow_inverted \
  --single-limits local \
  --mean-limits local
```

### 5. Recreate Evaluation and Supplementary Figures

Confusion matrices:

```bash
python scripts/evaluation/replot_confusion_matrices.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json
```

ROC and precision-recall curves:

```bash
python scripts/analysis/plot_roc_pr_curves.py \
  --config-list configs/examples/vr_goalkeeper_curve_comparison.json
```

Model-comparison bar plots:

```bash
python scripts/analysis/plot_model_comparison.py --dataset both
```

Statistical testing and confidence intervals:

```bash
python scripts/analysis/statistical_significance.py
python scripts/analysis/confidence_intervals.py
```

Subject-level and noise-related error analyses:

```bash
python scripts/analysis/error_analysis_subjects.py
python scripts/analysis/noise_error_analysis.py
```

## Notes on Reproducibility

- The provided configurations are examples that document representative settings
  used in the manuscript. They are intended as templates for reproducing the full
  experiment matrix.
- Random seeds, subject-wise splits, and Optuna search spaces are defined in the
  configuration files and training utilities.
- Some analyses require trained outer-fold models and recovered prediction files.
  Run training and recovery before replotting these results.
- Cluster execution depends on local infrastructure. The SLURM script in
  `batch_scripts/examples/` is a template and should be adapted to local paths,
  modules, and resource limits.
