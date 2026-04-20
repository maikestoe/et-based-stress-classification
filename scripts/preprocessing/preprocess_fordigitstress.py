"""Preprocess the ForDigitStress dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_path", default="data/fordigitstress/raw/")
    parser.add_argument("--output_path", default="data/fordigitstress/dataframes/")
    parser.add_argument(
        "--invalid-ids",
        nargs="*",
        type=int,
        default=[12, 15, 16, 18, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from process_fordigitstress import main as preprocess
    from process_fordigitstress import settings

    output_path = args.output_path
    if not output_path.endswith("/"):
        output_path += "/"

    preprocess(args.raw_data_path, output_path, args.invalid_ids, settings)


if __name__ == "__main__":
    main()
