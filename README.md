# Stress Classification from Eye-Tracking Time Series

This repository contains the code accompanying the manuscript:

**Classifying Mental Stress from Eye Tracking Data: Deep Learning Approaches for Out-of-the-Lab Conditions**
The code supports preprocessing, feature-based baseline training, deep-learning
training, evaluation, and supplementary analyses for two datasets:

- **VR goalkeeper dataset** (publicly available)
- **ForDigitStress dataset** (restricted access)

The datasets themselves are not redistributed in this repository and must be
obtained from their original providers.

## Data Availability

The VR Goalkeeper dataset generated during the current study is publicly
available via Zenodo at [https://zenodo.org/records/17972964](https://zenodo.org/records/17972964).
For convenience, this repository also includes the processed VR Goalkeeper
dataframes used by the training scripts:
`data/vr_goalkeeper/dataframes/DL_out.pkl` and
`data/vr_goalkeeper/dataframes/features_out.pkl`.

The ForDigitStress dataset is publicly available for research and
non-commercial use. Access to the dataset can be requested at
[https://hcai.eu/fordigitstress](https://hcai.eu/fordigitstress), and the
dataset should be cited as Heimerl et al.,
[https://doi.org/10.1109/TAFFC.2024.3501400](https://doi.org/10.1109/TAFFC.2024.3501400).
Use of the ForDigitStress dataset is subject to the dataset's EULA; this
repository therefore does not redistribute raw data, processed data, trained
models, or other dataset-derived artefacts for ForDigitStress. When using
ForDigitStress, please follow the acknowledgement requirements in its EULA.

## Code Availability

The complete code used for preprocessing, feature extraction, model training,
evaluation, and supplementary analysis is publicly available at
[https://github.com/maikestoe/et-based-stress-classification](https://github.com/maikestoe/et-based-stress-classification).
For publication or archival submission, cite the GitHub release, commit, or
repository archive that corresponds to the manuscript version.

## Third-Party Method Acknowledgements

Parts of the preprocessing and feature-extraction code are Python adaptations
or implementations of published eye-tracking methods. `src/pd_utils.py` adapts
pupil-size preprocessing from Kret and Sjak-Shie (2019). `src/fix_utils.py`
implements/adapts fixation and eye-movement procedures from Duchowski et al.
(2002). `src/IPA_utils.py` implements IPA and LHIPA metrics from Duchowski et
al. (2018, 2020). These implementations were modified for the datasets and
workflow used in this study.

Relevant references:

- Kret, M. E., & Sjak-Shie, E. E. (2019). Preprocessing pupil size data:
  Guidelines and code. *Behavior Research Methods*, 51, 1336-1342.
  [https://doi.org/10.3758/s13428-018-1075-y](https://doi.org/10.3758/s13428-018-1075-y)
- Duchowski, A. T., Medlin, E., Cournia, N., Murphy, H., Gramopadhye, A.,
  Nair, S., Vorah, J., & Melloy, B. (2002). 3-D eye movement analysis.
  *Behavior Research Methods, Instruments, & Computers*, 34(4), 573-591.
  [https://doi.org/10.3758/BF03195486](https://doi.org/10.3758/BF03195486)
- Duchowski, A. T., Krejtz, K., Krejtz, I., Biele, C., Niedzielska, A.,
  Kiefer, P., Raubal, M., & Giannopoulos, I. (2018). The Index of Pupillary
  Activity. *Proceedings of CHI 2018*.
  [https://doi.org/10.1145/3173574.3173856](https://doi.org/10.1145/3173574.3173856)
- Duchowski, A. T., Krejtz, K., Gehrer, N. A., Bafna, T., & Baekgaard, P.
  (2020). The Low/High Index of Pupillary Activity. *Proceedings of CHI 2020*.
  [https://doi.org/10.1145/3313831.3376394](https://doi.org/10.1145/3313831.3376394)

## Data Preparation

Place authorized dataset files under `data/` before running preprocessing. The
expected folder layout and dataset-specific access notes are documented in
[`data/README.md`](data/README.md), including minimal examples for reading one
raw file from each dataset. The preprocessing commands used for reproduction
are listed below.

The included VR Goalkeeper dataframes allow users to start directly from
training or evaluation for that public dataset. To verify preprocessing from raw
VR Goalkeeper files, rerun the preprocessing command below; it will regenerate
the same expected dataframe paths.

This repository also includes a small set of recovered VR Goalkeeper result
artefacts under [`results/`](results/README.md). These files allow selected
confusion-matrix and ROC/precision-recall figures to be recreated without
rerunning training or storing model checkpoints.

The final VR Goalkeeper model-comparison source table is generated from the
saved `final_summary.txt` training logs with:

```bash
python scripts/analysis/export_vr_source_data.py
```

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── data/
│   └── README.md
├── configs/
│   ├── examples/
│   └── paper/
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
- `configs/examples/` contains runnable configurations for the main paper
  examples and local smoke tests.
- `configs/paper/` contains a compact experiment matrix mapping every final
  paper experiment ID to its dataset, architecture, and input signal. The full
  paper matrix consists of systematic combinations of these fields with the
  shared hyperparameter search spaces shown in `configs/examples/`.
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

## Model Checkpoints

The repository enables reproduction through dataset access, preprocessing,
training code, configuration files, and evaluation scripts.

- **VR goalkeeper dataset (public)**:  
  All model checkpoints were generated during the study. However, the full set
  comprises hundreds of fold-specific models (30 configurations × 27 folds),
  which are not included due to their size and limited practical usefulness.
  The checkpoints can be regenerated from the public data, code, configuration
  matrix, and training procedure described here. The original training was
  performed with a single NVIDIA RTX 3080 GPU with 10 GB VRAM and Intel Xeon
  CPUs.

- **ForDigitStress dataset (restricted)**:  
  Model checkpoints are not distributed. Due to the dataset’s restricted-access
  EULA, derivative artefacts such as trained models are not shared through this
  repository. They can be regenerated by authorized users with access to the
  dataset.

## Reproducing the Paper Results

The full paper results were produced with nested leave-one-subject-out
cross-validation and Optuna-based hyperparameter optimization. This is
computationally expensive. The commands below show the reproducibility workflow
using the main runnable configurations and the compact paper matrix.

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

### 5. Recreate Main Manuscript Figures

The manuscript contains a mixture of conceptual/data-illustration figures and
figures generated from model outputs. Figure 1 is a schematic overview and is
not generated by an analysis script. Figures 2 and 3 show representative
preprocessed time-series examples from the datasets; these can be recreated by
reading the processed dataframes described above and plotting selected signal
columns, but they are not tied to a fixed model-output script.

The main manuscript figures generated from the repository's evaluation outputs
are recreated as follows.

Model-comparison bar plot (main manuscript Figure 4):

```bash
python scripts/analysis/export_vr_source_data.py
python scripts/analysis/plot_model_comparison.py --dataset both --no-latex
```

The first command regenerates the compact VR Goalkeeper source-data table in
`results/source_data/` from saved `final_summary.txt` logs when those logs are
present. The plotting script uses that table for the VR Goalkeeper panels and
the reported ForDigitStress summary values embedded in the plotting script,
because ForDigitStress result artefacts are not redistributed.

Representative confusion matrices (main manuscript Figure 5):

```bash
python scripts/evaluation/replot_confusion_matrices.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json \
  --no-latex

python scripts/evaluation/replot_confusion_matrices.py \
  --config configs/examples/vr_goalkeeper_cnn_pd.json \
  --no-latex

python scripts/evaluation/replot_confusion_matrices.py \
  --config configs/examples/fordigitstress_cnn_pd_recovery.json \
  --no-latex
```

The included `results/` folder contains compact recovered VR Goalkeeper outputs
for the first two commands. The ForDigitStress confusion matrix can be recreated
by authorized users after generating the corresponding recovered result files.

Compact PD CNN occlusion comparison (main manuscript Figure 6):

```bash
python scripts/analysis/plot_main_pd_occlusion_comparison.py
```

By default, this script expects recovered mean occlusion payloads at:

```text
results/vr_goalkeeper/DL/2024-08-09/CNN/1/recovery/occlusion/correct_stress_mean_occlusion.npz
results/fordigitstress/DL/2024-08-09/CNN/201/recovery/occlusion/correct_stress_mean_occlusion.npz
```

These payloads are created by the attribution recovery workflow. The
ForDigitStress payload is not redistributed because it is a dataset-derived
artefact. If the payloads are stored elsewhere, pass them explicitly:

```bash
python scripts/analysis/plot_main_pd_occlusion_comparison.py \
  --vr-payload path/to/vr/correct_stress_mean_occlusion.npz \
  --fordigit-payload path/to/fordigit/correct_stress_mean_occlusion.npz
```

### 6. Recreate Supplementary Material Figures and Analyses

Training-loss curves (Supplementary Fig. S1) are generated during model
training and saved in the corresponding outer-fold result folders. Rerun the
training commands above to regenerate these fold-level plots.

Random-forest feature-selection frequencies (Supplementary Fig. S2):

```bash
python scripts/analysis/plot_rf_feature_selection.py --no-latex
```

Aggregated saliency and occlusion case plots for the four prediction cases
(correct non-stress, correct stress, wrong non-stress, wrong stress) can be
recreated from recovered attribution payloads with:

```bash
python scripts/evaluation/replot_attribution_maps.py \
  --config configs/examples/vr_goalkeeper_convlstm3_asymptotic_recovery.json \
  --methods saliency occlusion \
  --payload-kind mean \
  --mean-limits local \
  --no-latex
```

Use `--payload-kind single` to recreate individual-example attribution plots, or
`--payload-kind all` to recreate both individual-example and aggregated plots.
The same command structure applies to other recovery configs after the
corresponding attribution payloads have been generated.

ROC and precision-recall curves for the VR Goalkeeper dataset (Supplementary
Fig. S7):

```bash
python scripts/analysis/plot_roc_pr_curves.py \
  --config-list configs/examples/vr_goalkeeper_curve_comparison.json \
  --no-latex
```

ROC and precision-recall curves for the ForDigitStress dataset (Supplementary
Fig. S8):

```bash
python scripts/analysis/plot_roc_pr_curves.py \
  --config-list configs/examples/fordigitstress_curve_comparison.json \
  --no-latex
```

The VR Goalkeeper curve command can use the compact recovered arrays included
for selected models. The ForDigitStress curve command requires authorized local
ForDigitStress-derived recovery files.

Statistical testing and confidence intervals reported in the supplementary
materials:

```bash
python scripts/analysis/statistical_significance.py
python scripts/analysis/confidence_intervals.py
```

Subject-level, noise-related, sample-order, cognitive-task, and computational
complexity analyses:

```bash
python scripts/analysis/error_analysis_subjects.py
python scripts/analysis/noise_error_analysis.py
python scripts/analysis/model_complexity.py
python src/exp41_sample_order_error_analysis.py
python src/exp41_cognitive_task_error_analysis.py
```

These supplementary analyses require the corresponding recovered per-fold or
per-sample result files. If only the compact public artefacts in `results/` are
available, rerun training and recovery first for analyses that report missing
inputs.

## Notes on Reproducibility

- The full paper experiment set is listed in
  `configs/paper/paper_experiment_matrix.csv`. The rows specify the final
  experiment IDs, datasets, architectures, and input signals reported in the
  manuscript.
- The runnable JSON files in `configs/examples/` document the shared training
  settings, participant splits, Optuna search spaces, and recovery settings used
  for the main examples. The remaining paper experiments use the same structure
  with the model/input fields listed in the compact matrix.
- Random seeds, subject-wise splits, and Optuna search spaces are defined in the
  configuration files and training utilities.
- Some analyses require trained outer-fold models and recovered prediction files.
  Run training and recovery before replotting these results.
- Cluster execution depends on local infrastructure. The SLURM script in
  `batch_scripts/examples/` is a template and should be adapted to local paths,
  modules, and resource limits.
