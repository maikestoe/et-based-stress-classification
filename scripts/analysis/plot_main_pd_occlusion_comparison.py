"""Plot the compact main-manuscript PD CNN occlusion comparison."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _run_module import run


if __name__ == "__main__":
    run("plot_main_pd_occlusion_comparison")
