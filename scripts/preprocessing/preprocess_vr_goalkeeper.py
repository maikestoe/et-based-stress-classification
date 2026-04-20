"""Preprocess the VR goalkeeper dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_path", default="data/vr_goalkeeper/raw/")
    parser.add_argument("--output_path", default="data/vr_goalkeeper/dataframes/")
    parser.add_argument("--excluded-ids", nargs="*", type=int, default=[5, 25, 28])
    parser.add_argument("--target-fs", nargs="*", type=float, default=[15, 25, 30, 45, 90])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))

    from process_vr_goalkeeper import process_participants

    output_path = args.output_path
    if not output_path.endswith("/"):
        output_path += "/"

    process_participants(
        raw_data_path=args.raw_data_path,
        output_path=output_path,
        excluded_IDs=args.excluded_ids,
        target_fs=args.target_fs,
    )


if __name__ == "__main__":
    main()
