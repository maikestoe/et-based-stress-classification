"""Run subject-level error analyses and create the corresponding figures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_module import run


if __name__ == "__main__":
    run("error_analysis_subjects")

