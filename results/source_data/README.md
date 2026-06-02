# Source Data

This folder contains compact CSV source-data tables derived from saved training
logs and recovered output artefacts.

- `vr_model_comparison_metrics.csv` is generated from the VR Goalkeeper
  `final_summary.txt` logs with:

  ```bash
  python scripts/analysis/export_vr_source_data.py
  ```

The CSV is intended as figure/table source data for the final VR Goalkeeper
model-comparison metrics. It is not a replacement for the raw data or model
training pipeline; it records the final aggregate metrics reported by the saved
training summaries.
