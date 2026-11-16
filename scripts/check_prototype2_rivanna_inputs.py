#!/usr/bin/env python3
"""Verify the exact historical Prototype-2 Rivanna pilot before comparison.

This read-only preflight requires the recorded original NPZ and schema hashes,
then validates their schema, encoded shapes, and public categorical domains.
It prints only verification status, hashes, and counts; it never prints records.
There is no hash-override or public-fixture bypass in this command.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from prototype2_evaluation import FEATURES, load_pilot


EXPECTED_DATA_SHA256 = "677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906"
EXPECTED_SCHEMA_SHA256 = "3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73"
EXPECTED_FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
EXPECTED_CATEGORIES = [3, 9, 6, 44, 531, 266]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(data_npz, schema_path):
    """Return verification metadata, or fail before a mechanism can be fitted."""
    data_npz = Path(data_npz).expanduser().resolve(strict=True)
    schema_path = Path(schema_path).expanduser().resolve(strict=True)
    observed_data = sha256_file(data_npz)
    observed_schema = sha256_file(schema_path)
    mismatches = []
    if observed_data != EXPECTED_DATA_SHA256:
        mismatches.append(
            f"NPZ SHA256 {observed_data}; expected {EXPECTED_DATA_SHA256}")
    if observed_schema != EXPECTED_SCHEMA_SHA256:
        mismatches.append(
            f"schema SHA256 {observed_schema}; expected {EXPECTED_SCHEMA_SHA256}")
    if mismatches:
        raise ValueError("Historical input hash mismatch. " + ". ".join(mismatches))

    train, holdout, categories, schema = load_pilot(data_npz, schema_path)
    if FEATURES != EXPECTED_FEATURES or schema.get("features") != EXPECTED_FEATURES:
        raise ValueError("Feature count/order differs from the seven historical features")
    if train.shape != (2048, 7) or holdout.shape != (512, 7):
        raise ValueError(
            f"Expected train/holdout shapes (2048, 7)/(512, 7); found {train.shape}/{holdout.shape}")
    if categories.tolist() != EXPECTED_CATEGORIES:
        raise ValueError("Categorical cardinalities differ from [3, 9, 6, 44, 531, 266]")
    if (schema.get("public_fixture", False) is not False
            or schema.get("data_origin") == "programmatically_generated_no_real_records"):
        raise ValueError("A public generated fixture cannot pass a research-data preflight")
    return {
        "status": "verified_exact_historical_inputs",
        "data_npz_sha256": observed_data,
        "schema_sha256": observed_schema,
        "training_shape": list(train.shape),
        "holdout_shape": list(holdout.shape),
        "feature_count": len(EXPECTED_FEATURES),
        "categorical_cardinalities": categories.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = check_inputs(args.data_npz, args.schema)
    except (OSError, ValueError, KeyError) as error:
        print(f"Input preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
