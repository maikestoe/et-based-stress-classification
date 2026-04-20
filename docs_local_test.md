# Local Smoke Test

This document describes a small local test run. It is intended to verify that
the repository, dependencies, data paths, and training loop work on a local
machine. It is **not** intended to reproduce the paper-level results.

## 1. Create a Poetry Environment

The `pyproject.toml` uses standard PEP 621 metadata. With recent Poetry
versions, create and install the environment with:

```bash
cd /path/to/et-stress-classification
poetry env use python3.13
poetry install --extras tensorflow
```

If Poetry cannot resolve TensorFlow on your platform, use the already working
TensorFlow environment directly:

```bash
cd /path/to/et-stress-classification
python -m pip install -r requirements.txt
```

## 2. Prepare the Minimal Data File

The smoke-test configuration expects the preprocessed VR goalkeeper dataframe at:

```text
data/vr_goalkeeper/dataframes/DL_out.pkl
```

If the file already exists in another local folder, link or copy it into the
publication repository:

```bash
mkdir -p data/vr_goalkeeper/dataframes
ln -s /path/to/preprocessed/vr_goalkeeper/DL_out.pkl \
  data/vr_goalkeeper/dataframes/DL_out.pkl
```

## 3. Run a Minimal Training Test

```bash
poetry run python scripts/training/train_deep_learning.py \
  --config configs/examples/local_smoke_vr_goalkeeper_cnn_pd.json
```

Equivalent command without Poetry:

```bash
python scripts/training/train_deep_learning.py \
  --config configs/examples/local_smoke_vr_goalkeeper_cnn_pd.json
```

The smoke test uses:

- two outer test participants,
- one Optuna trial,
- two training epochs,
- a small CNN,
- pupil diameter as input,
- local output under `results/local_smoke/`.

Expected result: the script starts nested leave-one-subject-out training, writes
fold outputs under `results/local_smoke/vr_goalkeeper/DL/local-smoke-test/`, and
finishes without import, path, or model-construction errors.

## 4. What This Test Does Not Prove

This run is deliberately too small for scientific interpretation. It does not
reproduce the reported F1-scores and should not be used for manuscript results.
Use the full example configurations, complete participant lists, and full Optuna
search settings for reproduction of the paper analyses.
