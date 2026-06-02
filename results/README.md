# Result Artefacts

This folder contains a small set of recovered VR Goalkeeper result artefacts
that allow selected publication figures to be recreated without rerunning model
training or storing model checkpoints.

The folder also contains `source_data/`, where compact CSV tables are generated
from the saved VR Goalkeeper training summaries. These tables make the final
reported metrics inspectable without reading Python plotting constants.

Included artefacts:

- `results/vr_goalkeeper/DL/2024-08-09/CNN/1/recovery/`
  contains aggregated recovered outputs for the CNN with pupil-diameter input.
- `results/vr_goalkeeper/DL/2024-08-09/ConvLSTM-3/41/recovery/`
  contains aggregated recovered outputs for the ConvLSTM-3 with asymptotic-model
  input.

Each recovery folder includes:

- `confusion_matrix_outer_aggregated.csv`
- `outer_metrics_recovered.csv`
- `roc_curve_outer_aggregated.csv`
- `pr_curve_outer_aggregated.csv`
- `y_true_outer_aggregated.npy`
- `y_score_outer_aggregated.npy`

The `.npy` arrays are used by `scripts/analysis/plot_roc_pr_curves.py`. The CSV
files provide compact source data for confusion matrices, aggregate metrics, and
curve exports. Trained model weights, Optuna studies, recovered attribution
payloads, and ForDigitStress result artefacts are intentionally not included.
