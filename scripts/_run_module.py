"""Run modules from the public release `src` folder.

The original analysis code uses flat module imports. This helper keeps the
public repository layout readable while preserving those imports.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    runpy.run_module(module_name, run_name="__main__")

