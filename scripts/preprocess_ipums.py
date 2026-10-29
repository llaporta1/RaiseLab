"""Preprocess IPUMS ACS extract for Prototype #2.

Steps
-----
1. Load the raw CSV produced by ``scripts/load_ipums_dat.py``.
2. Optionally filter to a target YEAR and/or SAMPLE (e.g., ACS 2024 sample 202403).
3. Retain only modeling variables and treat them as categorical strings.
4. Handle missing/NIU-like values using a configurable strategy (label or drop).
5. Optionally collapse categories to top-k for PUMA, OCC, and EDUCD.
6. Save cleaned data, per-variable frequency tables, metadata, and diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


MODELING_VARS = ["PUMA", "OCC", "EDUCD", "AGE", "SEX", "RACE", "HISPAN"]
KEEP_COLS = ["YEAR", "SAMPLE", *MODELING_VARS]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IPUMS ACS extract")
    parser.add_argument("--input_csv", required=True, type=Path, help="Raw IPUMS CSV")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory to write cleaned data, metadata, diagnostics, frequencies",
    )
    parser.add_argument("--year_filter", type=str, default=None, help="Filter YEAR to this value")
    parser.add_argument(
        "--sample_filter", type=str, default=None, help="Filter SAMPLE to this value"
    )
    parser.add_argument("--topk_puma", type=int, default=None, help="Keep top-k PUMA codes; collapse the rest to OTHER")
    parser.add_argument("--topk_occ", type=int, default=None, help="Keep top-k OCC codes; collapse the rest to OTHER")
    parser.add_argument("--topk_educd", type=int, default=None, help="Keep top-k EDUCD codes; collapse the rest to OTHER")
    parser.add_argument(
        "--drop_missing",
        action="store_true",
        help="If set, drop rows with missing/NIU values in modeling variables",
    )
    parser.add_argument(
        "--missing_strategy",
        choices=["drop", "label"],
        default="label",
        help="How to handle missing/NIU values (ignored if --drop_missing is set)",
    )
    return parser.parse_args(argv)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def apply_filters(df: pd.DataFrame, year: str | None, sample: str | None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if year is not None:
        mask &= df["YEAR"].astype(str) == str(year)
    if sample is not None and "SAMPLE" in df.columns:
        mask &= df["SAMPLE"].astype(str) == str(sample)
    return df[mask].copy()


def to_categorical_strings(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
        else:
            out[col] = ""
    return out


def is_missingish(value: str) -> bool:
    if value is None:
        return True
    v = str(value).strip()
    if v == "":
        return True
    lower = v.lower()
    if lower in {"na", "n/a", "nan", "none"}:
        return True
    if v.isdigit() and len(v) > 0 and len(set(v)) == 1 and v[0] in {"0", "9"}:
        # Handles NIU patterns like 0000 or 9999
        return True
    return False


def clean_missing(series: pd.Series, strategy: str) -> tuple[pd.Series, pd.Series]:
    missing_mask = series.apply(is_missingish)
    if strategy == "drop":
        cleaned = series[~missing_mask]
    else:
        cleaned = series.copy()
        cleaned[missing_mask] = "MISSING"
    return cleaned, missing_mask


def collapse_topk(series: pd.Series, k: int | None) -> pd.Series:
    if k is None or k <= 0:
        return series
    counts = series.value_counts(dropna=False)
    keep = set(counts.nlargest(k).index)
    return series.where(series.isin(keep), "OTHER")


def frequency_table(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts(dropna=False)
    total = counts.sum() if counts.sum() else 1
    freq_df = counts.reset_index()
    freq_df.columns = ["value", "count"]
    freq_df["proportion"] = freq_df["count"] / total
    return freq_df


def compute_metadata(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    row_count_before: int,
    row_count_after_filter: int,
    row_count_after_clean: int,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "row_count_before_filter": row_count_before,
        "row_count_after_filter": row_count_after_filter,
        "row_count_after_cleaning": row_count_after_clean,
        "columns": {},
    }
    for col in KEEP_COLS:
        raw_card = raw_df[col].nunique(dropna=False) if col in raw_df.columns else 0
        clean_card = clean_df[col].nunique(dropna=False) if col in clean_df.columns else 0
        meta["columns"][col] = {
            "raw_cardinality": int(raw_card),
            "clean_cardinality": int(clean_card),
        }
    return meta


def write_outputs(
    output_dir: Path,
    cleaned_df: pd.DataFrame,
    freq_tables: Dict[str, pd.DataFrame],
    metadata: Dict[str, object],
    diagnostics: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "ipums_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    diag_path = output_dir / "diagnostics.txt"
    diag_path.write_text(diagnostics)

    for col, table in freq_tables.items():
        table.to_csv(output_dir / f"freq_{col.lower()}.csv", index=False)


def build_diagnostics(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    missing_counts: Dict[str, int],
    row_count_before: int,
    row_count_after_filter: int,
    row_count_after_clean: int,
) -> str:
    lines: List[str] = []
    lines.append(f"Rows before filtering: {row_count_before}")
    lines.append(f"Rows after filtering:  {row_count_after_filter}")
    lines.append(f"Rows after cleaning:   {row_count_after_clean}")
    lines.append("")
    for col in KEEP_COLS:
        raw_unique = raw_df[col].nunique(dropna=False) if col in raw_df.columns else 0
        clean_unique = cleaned_df[col].nunique(dropna=False) if col in cleaned_df.columns else 0
        lines.append(f"[{col}]")
        lines.append(f"  raw unique:   {raw_unique}")
        lines.append(f"  clean unique: {clean_unique}")
        miss = missing_counts.get(col, 0)
        lines.append(f"  missing flagged: {miss}")
        top = cleaned_df[col].value_counts().head(10)
        top_str = ", ".join([f"{idx}: {val}" for idx, val in top.items()])
        lines.append(f"  top categories: {top_str}")
        lines.append("")
    return "\n".join(lines)


def preprocess(args: argparse.Namespace) -> None:
    df = load_data(args.input_csv)
    row_count_before = len(df)

    # Ensure required columns exist
    missing_cols = [c for c in KEEP_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required columns: {missing_cols}")

    filtered_df = apply_filters(df, args.year_filter, args.sample_filter)
    row_count_after_filter = len(filtered_df)

    # Keep only required columns
    filtered_df = filtered_df[KEEP_COLS].copy()
    raw_subset = filtered_df.copy()

    # Cast to strings and trim
    filtered_df = to_categorical_strings(filtered_df, KEEP_COLS)

    # Handle missing values and top-k collapsing
    missing_strategy = "drop" if args.drop_missing else args.missing_strategy
    cleaned_df = filtered_df.copy()
    missing_counts: Dict[str, int] = {}

    for col in MODELING_VARS + ["YEAR", "SAMPLE"]:
        series = cleaned_df[col]
        cleaned_series, missing_mask = clean_missing(series, missing_strategy)
        missing_counts[col] = int(missing_mask.sum())
        # Apply top-k where requested
        if col == "PUMA":
            cleaned_series = collapse_topk(cleaned_series, args.topk_puma)
        elif col == "OCC":
            cleaned_series = collapse_topk(cleaned_series, args.topk_occ)
        elif col == "EDUCD":
            cleaned_series = collapse_topk(cleaned_series, args.topk_educd)
        cleaned_df[col] = cleaned_series

    # If dropping missing, remove rows with MISSING markers
    if missing_strategy == "drop":
        cleaned_df = cleaned_df.dropna()
        for col in MODELING_VARS:
            cleaned_df = cleaned_df[~cleaned_df[col].apply(is_missingish)]

    # Frequency tables
    freq_tables: Dict[str, pd.DataFrame] = {}
    for col in MODELING_VARS:
        freq_tables[col] = frequency_table(cleaned_df[col])

    row_count_after_clean = len(cleaned_df)

    metadata = compute_metadata(
        raw_subset, cleaned_df, row_count_before, row_count_after_filter, row_count_after_clean
    )
    diagnostics = build_diagnostics(
        raw_subset,
        cleaned_df,
        missing_counts,
        row_count_before,
        row_count_after_filter,
        row_count_after_clean,
    )

    write_outputs(args.output_dir, cleaned_df, freq_tables, metadata, diagnostics)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preprocess(args)
    except Exception as exc:  # Fail loudly for pipeline visibility
        print(f"Preprocessing failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
