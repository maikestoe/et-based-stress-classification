# Local Test Matrix

This checklist is intended for testing the public repository before release. It
contains short local runs that verify preprocessing entry points, RF baseline
training, deep-learning training for both datasets, and representative analysis
scripts. These tests are not intended to reproduce the final paper metrics.

Run all commands from the repository root:

```bash
cd /path/to/et-stress-classification
export MPLCONFIGDIR="$PWD/.matplotlib_cache"
```

## 1. Environment

Poetry setup:

```bash
poetry env use python3.13
poetry install --extras tensorflow
```

Fallback with pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Preprocessing Entry Points

Check the command-line interfaces:

```bash
poetry run python scripts/preprocessing/preprocess_vr_goalkeeper.py --help
poetry run python scripts/preprocessing/preprocess_fordigitstress.py --help
```

If raw datasets are available, run preprocessing:

```bash
poetry run python scripts/preprocessing/preprocess_vr_goalkeeper.py \
  --raw_data_path data/vr_goalkeeper/raw/ \
  --output_path data/vr_goalkeeper/dataframes/
```

```bash
poetry run python scripts/preprocessing/preprocess_fordigitstress.py \
  --raw_data_path data/fordigitstress/raw/ \
  --output_path data/fordigitstress/dataframes/
```

Expected outputs:

```text
data/vr_goalkeeper/dataframes/DL_out.pkl
data/vr_goalkeeper/dataframes/features_out.pkl
data/fordigitstress/dataframes/DL_out.pkl
```

For a faster local preprocessing smoke test, write outputs to a temporary
folder and process only one participant:

```bash
poetry run python scripts/preprocessing/preprocess_vr_goalkeeper.py \
  --raw_data_path data/vr_goalkeeper/raw/ \
  --output_path /tmp/et-stress-preprocess-smoke/vr_goalkeeper/ \
  --excluded-ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --target-fs 90
```

```bash
poetry run python scripts/preprocessing/preprocess_fordigitstress.py \
  --raw_data_path data/fordigitstress/raw/ \
  --output_path /tmp/et-stress-preprocess-smoke/fordigitstress/ \
  --invalid-ids 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
```

## 3. RF Baseline Smoke Test

The RF baseline requires:

```text
data/vr_goalkeeper/dataframes/features_out.pkl
```

Run a short combined-feature RF test:

```bash
poetry run python scripts/training/train_rf_baseline_vr_goalkeeper.py \
  --data-path data/vr_goalkeeper/dataframes/ \
  --output-dir results/local_smoke/vr_goalkeeper/rf_baseline/ \
  --subset combined \
  --test-ids 0 1 \
  --smoke-test
```

Expected result: output files are written under
`results/local_smoke/vr_goalkeeper/rf_baseline/combined/RF/`.

## 4. Deep-Learning Smoke Tests

The VR goalkeeper smoke test requires:

```text
data/vr_goalkeeper/dataframes/DL_out.pkl
```

```bash
poetry run python scripts/training/train_deep_learning.py \
  --config configs/examples/local_smoke_vr_goalkeeper_cnn_pd.json
```

The ForDigitStress smoke test requires:

```text
data/fordigitstress/dataframes/DL_out.pkl
```

```bash
poetry run python scripts/training/train_deep_learning.py \
  --config configs/examples/local_smoke_fordigitstress_cnn_pd.json
```

Expected result: both commands start nested LOSO-CV training with two held-out
participants, one Optuna trial, and two epochs. If optional Plotly/Kaleido
dependencies are unavailable, Optuna visualization files are skipped while
training and evaluation continue.

## 5. Representative Analysis Scripts

These commands test analysis/plotting entry points.

Model-comparison plot, no trained models required:

```bash
poetry run python scripts/analysis/plot_model_comparison.py \
  --dataset both \
  --output-dir results/local_smoke/model_comparison/ \
  --no-latex
```

Confidence intervals require recovered per-fold result files:

```bash
poetry run python scripts/analysis/confidence_intervals.py --help
```

ROC/PR plotting requires recovered prediction arrays:

```bash
poetry run python scripts/analysis/plot_roc_pr_curves.py --help
```

Attribution-map replotting requires recovered saliency or occlusion files:

```bash
poetry run python scripts/evaluation/replot_attribution_maps.py --help
```

## 6. Full Reproduction

After the smoke tests pass, use the full example configs in `configs/examples/`
and the complete participant lists for paper-level reproduction. The full runs
are computationally expensive because they use nested LOSO-CV and Optuna
hyperparameter optimization.

For the mapping between internal experiment IDs and the models/input signals
reported in the paper, see
[`docs_experiment_overview.md`](docs_experiment_overview.md).
