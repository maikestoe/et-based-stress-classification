# Paper Experiment Matrix

This folder keeps the paper-level experiment mapping compact. The final
manuscript matrix contains many systematic combinations of architecture and
input signal, so the repository stores the matrix as a table instead of adding
dozens of near-identical JSON files.

- `paper_experiment_matrix.csv` maps each final experiment ID to the dataset,
  model architecture, and input signal used in the paper.
- The runnable JSON files in `configs/examples/` show the shared structure:
  data paths, participant IDs, output paths, training settings, Optuna search
  spaces, and recovery settings.
- To reproduce a non-example row, start from the matching example JSON for the
  dataset and update `experiment_id`, `DL.model`, and `DL.input_cols` according
  to the matrix row.

