"""Generate sparsity diagnostics for IPUMS prototype datasets.

Metrics:
- Per-column cardinality, entropy, and proportion of rows in rare categories.
- Pairwise sparsity for selected variable combinations.

Outputs:
- JSON summary (metrics + rankings)
- Human-readable text report
- Optional pairwise contingency tables (CSV)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# Variables and pairs of interest for MWEM-style evaluation
TARGET_COLS = ["PUMA", "OCC", "IND", "EDUCD", "AGE", "SEX", "RACE", "HISPAN"]
PAIR_LIST = [
    ("PUMA", "RACE"),
    ("PUMA", "HISPAN"),
    ("PUMA", "EDUCD"),
    ("OCC", "SEX"),
    ("OCC", "RACE"),
    ("OCC", "EDUCD"),
    ("IND", "RACE"),
    ("IND", "EDUCD"),
    ("AGE", "SEX"),
    ("AGE", "RACE"),
]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IPUMS sparsity diagnostics")
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Cleaned IPUMS CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write summary/report (and tables if requested)",
    )
    parser.add_argument(
        "--rare_threshold",
        type=float,
        default=0.01,
        help="Categories with proportion <= threshold are considered rare",
    )
    parser.add_argument(
        "--save_tables",
        action="store_true",
        help="If set, write pairwise contingency tables to CSV",
    )
    return parser.parse_args(argv)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def entropy_from_counts(counts: pd.Series) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return -float((probs * (probs + 1e-15).map(math.log)).sum())


def per_column_stats(df: pd.DataFrame, rare_threshold: float) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        total = counts.sum()
        card = int(counts.size)
        ent = entropy_from_counts(counts)
        rare_prop = float(counts[counts / total <= rare_threshold].sum() / total) if total else 0.0
        stats[col] = {
            "cardinality": card,
            "entropy": ent,
            "rare_proportion": rare_prop,
        }
    return stats


def pairwise_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for a, b in PAIR_LIST:
        if a not in df.columns or b not in df.columns:
            continue
        cross = pd.crosstab(df[a], df[b], dropna=False)
        observed = int((cross > 0).sum().sum())
        domain = cross.shape[0] * cross.shape[1]
        occupancy = observed / domain if domain else 0.0
        sparsity = 1.0 - occupancy
        key = f"{a}__{b}"
        results[key] = {
            "var_a": a,
            "var_b": b,
            "cardinality_a": cross.shape[0],
            "cardinality_b": cross.shape[1],
            "domain_size": domain,
            "observed_cells": observed,
            "occupancy_ratio": occupancy,
            "sparsity_ratio": sparsity,
        }
        results[key]["table"] = cross if isinstance(cross, pd.DataFrame) else None
    return results


def build_rankings(col_stats: Dict[str, Dict[str, float]], pair_stats: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
    col_rank = sorted(
        ((col, vals["cardinality"]) for col, vals in col_stats.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    pair_rank = sorted(
        ((pair, vals["sparsity_ratio"]) for pair, vals in pair_stats.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {"by_cardinality": col_rank, "pairs_by_sparsity": pair_rank}


def write_outputs(
    output_dir: Path,
    col_stats: Dict[str, Dict[str, float]],
    pair_stats: Dict[str, Dict[str, float]],
    rankings: Dict[str, List[Tuple[str, float]]],
    save_tables: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    json_safe_pairs = {
        k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in pair_stats.items()
    }
    summary = {
        "columns": col_stats,
        "pairs": json_safe_pairs,
        "rankings": rankings,
    }
    (output_dir / "sparsity_summary.json").write_text(json.dumps(summary, indent=2))

    # Text report
    lines: List[str] = []
    lines.append("Per-column stats")
    for col, vals in col_stats.items():
        lines.append(
            f"- {col}: cardinality={vals['cardinality']}, entropy={vals['entropy']:.4f}, rare_prop={vals['rare_proportion']:.4f}"
        )
    lines.append("")
    lines.append("Pairwise sparsity")
    for name, vals in pair_stats.items():
        lines.append(
            f"- {name}: domain={vals['domain_size']}, observed={vals['observed_cells']}, occupancy={vals['occupancy_ratio']:.4f}, sparsity={vals['sparsity_ratio']:.4f}"
        )
    lines.append("")
    lines.append("Rankings")
    lines.append("  Columns by cardinality:")
    lines.extend([f"  - {col}: {card}" for col, card in rankings["by_cardinality"]])
    lines.append("  Pairs by sparsity:")
    lines.extend([f"  - {pair}: {sparsity:.4f}" for pair, sparsity in rankings["pairs_by_sparsity"]])
    (output_dir / "sparsity_report.txt").write_text("\n".join(lines))

    # Optional tables
    if save_tables:
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, vals in pair_stats.items():
            table = vals.get("table")
            if table is not None:
                table.to_csv(tables_dir / f"{name}.csv")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_data(args.input_csv)
        col_stats = per_column_stats(df, args.rare_threshold)
        pair_stats = pairwise_stats(df)
        rankings = build_rankings(col_stats, pair_stats)
        write_outputs(args.output_dir, col_stats, pair_stats, rankings, args.save_tables)
    except Exception as exc:
        print(f"Sparsity report failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
