# Experiment Overview

This document maps the internal experiment identifiers used in configuration
files, result folders, and plotting scripts to the model and input-signal names
reported in the paper. The `exp` numbers are legacy reproducibility identifiers,
not scientific terminology. In text and figures, prefer the paper-facing names
(`CNN`, `LSTM-1`, `LSTM-3`, `ConvLSTM-1`, `ConvLSTM-3`) and use the experiment
IDs only when referring to configs or saved result folders.

The public repository contains runnable example configurations in
`configs/examples/` and a compact machine-readable paper matrix in
`configs/paper/paper_experiment_matrix.csv`. The full paper matrix can be
recreated by combining the model, input signal, and dataset settings shown here
with the hyperparameter ranges in the example configs.

## Naming Conventions

| Internal config field | Paper-facing name |
| --- | --- |
| `CNN` | CNN |
| `LSTM-1` | LSTM-1 |
| `LSTM-3` | LSTM-3 |
| `ConvLSTM-1` | ConvLSTM-1 |
| `ConvLSTM-3` | ConvLSTM-3 |
| `meanDia_corrected` | PD |
| `velocity` | Angular velocity |
| `acceleration` | Angular acceleration |
| `position` | Visual angle |
| `asymptotic_model` | Asymptotic model |
| `fix_array` | Fixations |

## VR Goalkeeper Dataset

The VR goalkeeper deep-learning experiment matrix combines five architectures
with six input signals. The IDs not listed in the table were exploratory or
internal configurations and are not part of the final paper matrix.

| Experiment ID | Model | Input signal | Main use in paper/reproduction |
| --- | --- | --- | --- |
| `exp1` | CNN | PD | PD-based CNN, confusion matrix, attribution analysis, statistical comparison with RF PD baseline |
| `exp2` | CNN | Angular velocity | Model comparison |
| `exp3` | CNN | Angular acceleration | Model comparison |
| `exp4` | CNN | Visual angle | Model comparison, best architecture for visual angle in ROC/PR and reduced subject-difficulty analysis |
| `exp5` | CNN | Asymptotic model | Model comparison, complexity analysis for CNN on asymptotic-model input |
| `exp6` | CNN | Fixations | Model comparison, best architecture for fixations in ROC/PR and reduced subject-difficulty analysis |
| `exp10` | LSTM-1 | PD | Model comparison |
| `exp11` | LSTM-1 | Angular velocity | Model comparison |
| `exp12` | LSTM-1 | Angular acceleration | Model comparison |
| `exp13` | LSTM-1 | Visual angle | Model comparison |
| `exp14` | LSTM-1 | Asymptotic model | Model comparison, complexity analysis for LSTM-1 on asymptotic-model input |
| `exp15` | LSTM-1 | Fixations | Model comparison, best input for LSTM-1 in reduced subject-difficulty analysis |
| `exp19` | ConvLSTM-1 | PD | Model comparison |
| `exp20` | ConvLSTM-1 | Angular velocity | Model comparison, best input for ConvLSTM-1 in reduced subject-difficulty analysis |
| `exp21` | ConvLSTM-1 | Angular acceleration | Model comparison, best architecture for angular acceleration in ROC/PR and reduced subject-difficulty analysis |
| `exp22` | ConvLSTM-1 | Visual angle | Model comparison |
| `exp23` | ConvLSTM-1 | Asymptotic model | Model comparison, complexity analysis for ConvLSTM-1 on asymptotic-model input |
| `exp24` | ConvLSTM-1 | Fixations | Model comparison |
| `exp28` | LSTM-3 | PD | Model comparison |
| `exp29` | LSTM-3 | Angular velocity | Model comparison |
| `exp30` | LSTM-3 | Angular acceleration | Model comparison |
| `exp31` | LSTM-3 | Visual angle | Model comparison |
| `exp32` | LSTM-3 | Asymptotic model | Model comparison, complexity analysis for LSTM-3 on asymptotic-model input, best input for LSTM-3 in reduced subject-difficulty analysis |
| `exp33` | LSTM-3 | Fixations | Model comparison |
| `exp37` | ConvLSTM-3 | PD | Model comparison |
| `exp38` | ConvLSTM-3 | Angular velocity | Model comparison, best architecture for angular velocity in ROC/PR and reduced subject-difficulty analysis |
| `exp39` | ConvLSTM-3 | Angular acceleration | Model comparison |
| `exp40` | ConvLSTM-3 | Visual angle | Model comparison |
| `exp41` | ConvLSTM-3 | Asymptotic model | Best overall model, confusion matrix, attribution analysis, statistical comparison with RF combined baseline, ROC/PR, complexity analysis, sample-order and cognitive-task error analyses |
| `exp42` | ConvLSTM-3 | Fixations | Model comparison |

## ForDigitStress Dataset

Only PD was available as a directly comparable eye-tracking signal for the
ForDigitStress deep-learning experiments.

| Experiment ID | Model | Input signal | Main use in paper/reproduction |
| --- | --- | --- | --- |
| `exp201` | CNN | PD | Best macro-F1 model, confusion matrix, attribution analysis, ROC/PR, calibration, noise-related analysis |
| `exp202` | LSTM-1 | PD | Best weighted-F1 model, ROC/PR, calibration, noise-related analysis |
| `exp203` | ConvLSTM-1 | PD | Model comparison, ROC/PR |
| `exp204` | LSTM-3 | PD | Model comparison, ROC/PR |
| `exp205` | ConvLSTM-3 | PD | Model comparison, ROC/PR |

## Supplementary Analyses Using Experiment IDs

Several plotting and analysis scripts retain experiment IDs because they locate
saved result folders by `timestamp/model/experiment_id`. The main mappings are:

| Analysis | Experiments |
| --- | --- |
| Model-comparison bar plots | VR goalkeeper: all final matrix experiments listed above; ForDigitStress: `exp201`-`exp205` |
| Confusion matrices | `exp41`, `exp1`, `exp201` |
| Temporal attribution plots | `exp41`, `exp1`, `exp201`; representative individual example from `exp41` |
| ROC and precision-recall curves | VR goalkeeper: best model per input signal (`exp1`, `exp38`, `exp21`, `exp4`, `exp41`, `exp6`); ForDigitStress: `exp201`-`exp205` |
| Statistical significance analysis | `exp41` vs. RF combined baseline; `exp1` vs. RF PD baseline |
| Complexity analysis on asymptotic-model input | `exp5`, `exp14`, `exp23`, `exp32`, `exp41` |
| Subject-level difficulty analysis | Full matrices above; reduced VR panels use best input per architecture (`exp4`, `exp15`, `exp20`, `exp32`, `exp41`) and best architecture per input signal (`exp1`, `exp38`, `exp21`, `exp4`, `exp41`, `exp6`) |
| Noise-related error analysis | VR goalkeeper PD/asymptotic model analyses, with selected model emphasis on `exp41`; ForDigitStress PD models, with selected emphasis on `exp201` and `exp202` |
| VR sample-order and cognitive-task error analyses | `exp41` |
