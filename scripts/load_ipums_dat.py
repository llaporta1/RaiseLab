"""Utility to load IPUMS fixed-width extracts into CSV.

Parses IPUMS-provided metadata (prefers .sas INPUT block, falls back to
DDI .xml) to build pandas ``colspecs`` for ``read_fwf``. Fails loudly if no
schema metadata is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


ColSpec = Tuple[int, int]


def parse_sas_schema(path: Path) -> tuple[list[ColSpec], list[str]]:
    """Parse an IPUMS-style SAS script to extract column specs.

    The INPUT block typically looks like either of these forms:

    INPUT
      VAR1       1-5
      VAR2       6-10
      VAR3     @11 VAR3 $5.
      ...
    ;

    Returns ``(colspecs, names)`` where colspecs are 0-based ``(start, end)``
    tuples compatible with ``pandas.read_fwf`` (end is exclusive).
    """

    text = path.read_text(errors="ignore").splitlines()

    # Find the INPUT block boundaries.
    start_idx = None
    for idx, line in enumerate(text):
        if line.strip().upper().startswith("INPUT"):
            start_idx = idx + 1
            break

    if start_idx is None:
        raise ValueError("Could not find INPUT block in SAS script")

    colspecs: list[ColSpec] = []
    names: list[str] = []

    for line in text[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            break

        # Pattern 1: VAR 1-5
        parts = stripped.replace("\t", " ").split()
        if len(parts) >= 2 and "-" in parts[1]:
            try:
                start_str, end_str = parts[1].split("-")
                start = int(start_str) - 1  # pandas is 0-based
                end = int(end_str)
                var = parts[0]
                names.append(var)
                colspecs.append((start, end))
                continue
            except ValueError:
                pass

        # Pattern 2: @1 VAR $5.  or  @12 VAR 3.
        if stripped.startswith("@"):
            tokens = stripped.split()
            try:
                pointer = int(tokens[0].lstrip("@"))
                var = tokens[1]
                # Width token may include trailing '.'
                width_token = None
                for tok in tokens[2:]:
                    clean = tok.strip("$").rstrip(".")
                    if clean.isdigit():
                        width_token = clean
                        break
                if width_token is None:
                    raise ValueError
                width = int(width_token)
                start = pointer - 1
                end = start + width
                names.append(var)
                colspecs.append((start, end))
                continue
            except (ValueError, IndexError):
                pass

    if not colspecs:
        raise ValueError("Parsed SAS file but found no column specifications")

    # Ensure ordering by start position.
    pairs = sorted(zip(colspecs, names), key=lambda item: item[0][0])
    colspecs, names = zip(*pairs)
    return list(colspecs), list(names)


def parse_xml_schema(path: Path) -> tuple[list[ColSpec], list[str]]:
    """Parse IPUMS DDI XML to extract column specs.

    Looks for ``location`` elements with ``startPos`` and ``width`` (or
    ``endPos``) under ``var`` nodes.
    """

    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    colspecs: list[ColSpec] = []
    names: list[str] = []

    # IPUMS namespaces are usually present; ignore by checking tag suffix.
    def tag_name(elem: ET.Element) -> str:
        return elem.tag.split("}")[-1]

    for var in root.iter():
        if tag_name(var) != "var":
            continue
        var_name = var.attrib.get("name") or var.attrib.get("ID")
        if not var_name:
            continue
        loc = next((child for child in var if tag_name(child) == "location"), None)
        if loc is None:
            continue

        start = loc.attrib.get("startPos") or loc.attrib.get("StartPos")
        width = loc.attrib.get("width") or loc.attrib.get("Width")
        end = loc.attrib.get("endPos") or loc.attrib.get("EndPos")

        try:
            if start is not None and width is not None:
                start_i = int(start) - 1
                end_i = start_i + int(width)
            elif start is not None and end is not None:
                start_i = int(start) - 1
                end_i = int(end)
            else:
                continue
        except ValueError:
            continue

        names.append(var_name)
        colspecs.append((start_i, end_i))

    if not colspecs:
        raise ValueError("Parsed XML but found no column specifications")

    pairs = sorted(zip(colspecs, names), key=lambda item: item[0][0])
    colspecs, names = zip(*pairs)
    return list(colspecs), list(names)


def load_fixed_width(dat_path: Path, colspecs: List[ColSpec], names: List[str], nrows: int | None) -> pd.DataFrame:
    """Load the fixed-width file using pandas.read_fwf with the provided schema."""

    return pd.read_fwf(dat_path, colspecs=colspecs, names=names, dtype=str, nrows=nrows)


def find_schema(dat_path: Path, meta_path: Path | None) -> tuple[Path, str]:
    """Resolve which metadata file to use and return its path and type."""

    if meta_path:
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        kind = meta_path.suffix.lower()
        if kind not in {".sas", ".xml"}:
            raise ValueError("Metadata file must be .sas or .xml")
        return meta_path, kind

    base = dat_path.with_suffix("")
    sas_candidate = base.with_suffix(".sas")
    xml_candidate = base.with_suffix(".xml")

    if sas_candidate.exists():
        return sas_candidate, ".sas"
    if xml_candidate.exists():
        return xml_candidate, ".xml"

    raise FileNotFoundError(
        "No metadata file found. Provide --meta_path pointing to .sas or .xml."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load IPUMS fixed-width extract")
    parser.add_argument(
        "--dat_path",
        default="TabDiff/usa_00001.dat",
        type=Path,
        help="Path to the IPUMS fixed-width .dat file",
    )
    parser.add_argument(
        "--meta_path",
        type=Path,
        default=None,
        help="Optional path to .sas or .xml metadata file (defaults to sibling of dat)",
    )
    parser.add_argument(
        "--output_csv",
        default="data/ipums/processed/ipums_raw.csv",
        type=Path,
        help="Destination CSV path",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit for quick testing",
    )

    args = parser.parse_args(argv)

    dat_path: Path = args.dat_path
    meta_path: Path | None = args.meta_path
    output_csv: Path = args.output_csv

    if not dat_path.exists():
        print(f"Data file not found: {dat_path}", file=sys.stderr)
        return 1

    try:
        schema_path, ext = find_schema(dat_path, meta_path)
    except Exception as exc:  # Fail loudly when schema missing/misconfigured
        print(f"Schema resolution error: {exc}", file=sys.stderr)
        return 1

    try:
        if ext == ".sas":
            colspecs, names = parse_sas_schema(schema_path)
        else:
            colspecs, names = parse_xml_schema(schema_path)
    except Exception as exc:
        print(f"Failed to parse schema from {schema_path}: {exc}", file=sys.stderr)
        return 1

    try:
        df = load_fixed_width(dat_path, colspecs, names, args.nrows)
    except Exception as exc:
        print(f"Failed to load fixed-width data: {exc}", file=sys.stderr)
        return 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {', '.join(df.columns)}")
    print("Head:\n", df.head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
