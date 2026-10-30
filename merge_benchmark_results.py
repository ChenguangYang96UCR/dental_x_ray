#!/usr/bin/env python3
"""Merge per-model benchmark outputs produced by run_multi_gpu.sh."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_named_files(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/{name}"))
    if not paths:
        raise FileNotFoundError(f"No {name} files found below {root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = load_named_files(args.input_root, "fold_metrics.csv")
    predictions = load_named_files(args.input_root, "predictions.csv")
    summaries = load_named_files(args.input_root, "summary.csv")

    # Guard against accidentally merging the same model twice.
    fold_metrics = fold_metrics.drop_duplicates(subset=["model", "fold"])
    predictions = predictions.drop_duplicates(
        subset=["model", "fold", "image_path"]
    )
    summaries = summaries.drop_duplicates(subset=["model"])
    summaries = summaries.sort_values("auroc_mean", ascending=False)

    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    summaries.to_csv(args.output_dir / "summary.csv", index=False)

    # All workers use the same deterministic folds. Keep one canonical copy.
    split_paths = sorted(args.input_root.glob("*/splits.csv"))
    if split_paths:
        pd.read_csv(split_paths[0]).to_csv(args.output_dir / "splits.csv", index=False)

    print(summaries.to_string(index=False))
    print(f"Merged results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

