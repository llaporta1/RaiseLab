"""Build MWEM workload definitions for the IPUMS Prototype #2 dataset.

The script reads a cleaned IPUMS CSV, measures per-column cardinality, and
constructs three workload JSON files (easy/medium/hard) that list candidate
1-way/2-way marginals along with their estimated domain sizes. Any marginal
whose domain size exceeds a configurable threshold is skipped to keep the
workload tractable. A plain-text summary is also written describing what was
included or skipped.

Default input: ``data/ipums/processed/prototype2_00002/cleaned.csv``. If that
file is missing but ``ipums_cleaned.csv`` exists in the same directory, the
script will automatically use it instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


# Modeling variables for Prototype #2
MODELING_VARS = ["PUMA", "OCC", "EDUCD", "AGE", "SEX", "RACE", "HISPAN"]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MWEM workloads for IPUMS Prototype #2")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data/ipums/processed/prototype2_00002/cleaned.csv"),
        help="Cleaned IPUMS CSV (defaults to prototype2 cleaned path)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Where to write workload JSONs; defaults to the input file's directory",
    )
    parser.add_argument(
        "--max_domain_size",
        type=int,
        default=10_000,
        help="Skip marginals whose estimated domain exceeds this size",
    )
    return parser.parse_args(argv)


def resolve_input(path: Path) -> Path:
    """Prefer the provided path; fall back to ipums_cleaned.csv if needed."""

    if path.exists():
        return path
    alt = path.parent / "ipums_cleaned.csv"
    if alt.exists():
        print(f"[info] {path} not found; using {alt} instead")
        return alt
    raise FileNotFoundError(f"Input CSV not found: {path}")


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def cardinalities(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    cards: Dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        cards[col] = int(df[col].nunique(dropna=False))
    return cards


def combo_domain_size(combo: Sequence[str], card_map: Dict[str, int]) -> int | None:
    size = 1
    for col in combo:
        if col not in card_map:
            return None
        size *= card_map[col]
    return size


def dedupe_preserve(ordering: Iterable[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    seen = set()
    out: List[Tuple[str, ...]] = []
    for combo in ordering:
        if combo in seen:
            continue
        seen.add(combo)
        out.append(combo)
    return out


def build_workload(
    name: str,
    combos: List[Tuple[str, ...]],
    card_map: Dict[str, int],
    max_domain: int,
) -> Tuple[Dict[str, object], List[str]]:
    included = []
    skipped: List[str] = []

    for combo in combos:
        size = combo_domain_size(combo, card_map)
        if size is None:
            missing = [c for c in combo if c not in card_map]
            skipped.append(f"{' x '.join(combo)} (missing columns: {', '.join(missing)})")
            continue
        if size > max_domain:
            skipped.append(f"{' x '.join(combo)} (domain {size:,} > {max_domain:,})")
            continue
        included.append({"columns": list(combo), "domain_size": int(size)})

    payload: Dict[str, object] = {
        "name": name,
        "max_domain_size": int(max_domain),
        "marginals": included,
        "cardinality": card_map,
    }
    return payload, skipped


def format_combo(combo: Sequence[str]) -> str:
    return " x ".join(combo)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_summary(
    path: Path,
    card_map: Dict[str, int],
    reports: List[Tuple[str, Dict[str, object], List[str]]],
) -> None:
    lines: List[str] = []
    lines.append("IPUMS Prototype #2 MWEM workloads")
    lines.append("")
    lines.append("Per-variable cardinalities:")
    for col in MODELING_VARS:
        val = card_map.get(col, "-")
        lines.append(f"- {col}: {val}")
    lines.append("")

    for name, payload, skipped in reports:
        marginals = payload.get("marginals", [])
        lines.append(f"[{name.upper()}]")
        lines.append(f"  kept: {len(marginals)} marginals (max domain {payload['max_domain_size']:,})")
        if marginals:
            combos_str = ", ".join([format_combo(m["columns"]) + f" (|Ω|={m['domain_size']:,})" for m in marginals])
            lines.append(f"  marginals: {combos_str}")
        if skipped:
            lines.append(f"  skipped ({len(skipped)}): {', '.join(skipped)}")
        lines.append("")

    path.write_text("\n".join(lines))


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    input_csv = resolve_input(args.input_csv)
    output_dir = args.output_dir or input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_csv)
    card_map = cardinalities(df, MODELING_VARS)

    one_way = [(v,) for v in MODELING_VARS]

    workload_defs: Dict[str, List[Tuple[str, ...]]] = {
        "easy": dedupe_preserve(one_way + [("AGE", "SEX"), ("OCC", "SEX")]),
        "medium": dedupe_preserve(
            one_way
            + [
                ("PUMA", "RACE"),
                ("PUMA", "HISPAN"),
                ("OCC", "RACE"),
                ("AGE", "RACE"),
            ]
        ),
        "hard": dedupe_preserve(
            one_way
            + [
                ("OCC", "EDUCD"),
                ("PUMA", "EDUCD"),
                ("PUMA", "RACE"),
                ("OCC", "RACE"),
            ]
        ),
    }

    reports: List[Tuple[str, Dict[str, object], List[str]]] = []
    for name, combos in workload_defs.items():
        payload, skipped = build_workload(name, combos, card_map, args.max_domain_size)
        out_path = output_dir / f"workload_{name}.json"
        write_json(out_path, payload)
        reports.append((name, payload, skipped))
        print(f"[ok] Wrote {out_path} ({len(payload['marginals'])} marginals; skipped {len(skipped)})")

    summary_path = output_dir / "workload_summary.txt"
    write_summary(summary_path, card_map, reports)
    print(f"[ok] Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
