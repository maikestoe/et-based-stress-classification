"""Train the feature-based random-forest baseline for the VR goalkeeper dataset."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_module import run


if __name__ == "__main__":
    run("train_rf_baseline_vr_goalkeeper")

