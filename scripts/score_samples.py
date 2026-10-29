"""
Score an arbitrary samples.csv with the existing TabDiff metrics (density, C2ST).

This bypasses TabDiff sampling: you supply the generated samples plus the
corresponding real train/test/val splits and info.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch

# Ensure repo root is on sys.path so we can import tabdiff.*
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tabdiff.metrics import TabMetrics  # noqa: E402
from eval.visualize_density import plot_density  # noqa: E402


def load_info(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_column_alignment(df: pd.DataFrame, info: Dict[str, Any]) -> pd.DataFrame:
    """Reorder/validate columns to match info.json schema."""
    cols = info.get("column_names")
    if not cols:
        return df
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Samples missing columns required by info.json: {missing}")
    return df[cols]


def sanitize_unknown_categories(
    samples: pd.DataFrame, info: Dict[str, Any], real_train_path: str
) -> pd.DataFrame:
    """Clamp categorical values to the domain observed in real train data.

    Any unseen category is replaced with the mode of that column to avoid
    downstream encoder failures (e.g., sdmetrics LogisticDetection).
    """

    if "cat_col_idx" not in info:
        return samples
    real = pd.read_csv(real_train_path)
    real = ensure_column_alignment(real, info)

    cat_idx = info["cat_col_idx"]
    col_names = info.get("column_names", list(real.columns))
    for idx in cat_idx:
        col = col_names[idx]
        allowed = set(str(x) for x in real[col].astype(str).unique())
        mode = real[col].mode().iloc[0]
        samples[col] = samples[col].astype(str).apply(
            lambda v: v if v in allowed else mode
        )
    return samples


def save_extras(extras: Dict[str, Any], out_dir: Path) -> None:
    for name, extra in extras.items():
        if isinstance(extra, pd.DataFrame):
            extra.to_csv(out_dir / f"{name}.csv", index=False)
        elif isinstance(extra, dict):
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(extra, f, indent=2)
        else:
            raise NotImplementedError(
                f"Don't know how to save extras of type {type(extra)} for key {name}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score samples.csv with TabDiff metrics (density, C2ST)."
    )
    parser.add_argument("--samples", required=True, help="Path to samples.csv to score")
    parser.add_argument("--info", required=True, help="Path to info.json")
    parser.add_argument("--real_train", required=True, help="Path to real/train.csv")
    parser.add_argument("--real_test", required=True, help="Path to real/test.csv")
    parser.add_argument(
        "--real_val", default=None, help="Optional path to real/val.csv (if available)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write all_results.json and any extra outputs",
    )
    parser.add_argument(
        "--drop_nan_samples",
        action="store_true",
        help="Drop rows with NaNs before scoring (matches Trainer behaviour).",
    )
    parser.add_argument(
        "--plot_density",
        action="store_true",
        help="Also save density_plots.png (same helper used in Trainer).",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    info_path = Path(args.info)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = load_info(info_path)
    samples = pd.read_csv(samples_path)
    samples = ensure_column_alignment(samples, info)
    samples = sanitize_unknown_categories(samples, info, args.real_train)
    if args.drop_nan_samples:
        before = len(samples)
        samples = samples.dropna().reset_index(drop=True)
        dropped = before - len(samples)
        if dropped:
            print(f"Dropped {dropped} rows containing NaNs before scoring.")

    # Initialise metrics object on CPU; metric_list restricted to density + c2st.
    metrics = TabMetrics(
        real_data_path=args.real_train,
        test_data_path=args.real_test,
        val_data_path=args.real_val,
        info=info,
        device=torch.device("cpu"),
        metric_list=["density", "c2st"],
    )

    results, extras = metrics.evaluate(samples)

    # Save primary metrics.
    all_results_path = out_dir / "all_results.json"
    with open(all_results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save any tabular extras (e.g., shapes/trends from SDMetrics QualityReport).
    save_extras(extras, out_dir)

    # Optional density plots using the existing helper.
    if args.plot_density:
        real_df = pd.read_csv(args.real_train)
        real_df = ensure_column_alignment(real_df, info)
        img = plot_density(samples.copy(), real_df.copy(), info)
        img_path = out_dir / "density_plots.png"
        img.save(img_path)
        print(f"Saved density plots to {img_path}")

    print(f"Saved all_results.json to {all_results_path}")
    for name in extras:
        maybe_csv = out_dir / f"{name}.csv"
        maybe_json = out_dir / f"{name}.json"
        if maybe_csv.exists():
            print(f"Saved {name} details to {maybe_csv}")
        if maybe_json.exists():
            print(f"Saved {name} details to {maybe_json}")


if __name__ == "__main__":
    main()
