#!/usr/bin/env python3
"""Create generated, non-IPUMS data for software validation only.

The declared categorical cardinalities match the pilot, including its large
OCC x IND table. Values and relationships are invented test fixtures. Scores
from these records must never be presented as results on the research data.
"""

import argparse
import json
from pathlib import Path

import numpy as np


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
CATEGORIES = np.array([3, 9, 6, 44, 531, 266], dtype=np.int64)


def generated_records(count, rng):
    latent = rng.integers(0, 8, count)
    age = rng.integers(18, 66, count)
    result = np.column_stack([
        (age - 50.0) / 50.0,
        1 + latent % 2,
        1 + latent % 3,
        (latent % 3 == 1).astype(int),
        (age // 6 + latent) % 44,
        10 + 7 * latent + rng.integers(0, 3, count),
        3 + 5 * latent,
    ])
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory")
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--holdout-rows", type=int, default=128)
    parser.add_argument("--sample-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if min(args.train_rows, args.holdout_rows, args.sample_rows) < 16:
        parser.error("Fixture table sizes must be at least 16")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    train = generated_records(args.train_rows, rng)
    holdout = generated_records(args.holdout_rows, rng)
    np.savez_compressed(args.out / "pilot_data.npz", X_train=train,
                        X_test=holdout, categories=CATEGORIES)
    schema = {
        "features": FEATURES,
        "category_domains": {name: list(range(int(size)))
                             for name, size in zip(FEATURES[1:], CATEGORIES)},
        "age_transform": "(AGE - 50) / 50",
        "public_fixture": True,
        "fixture_name": "generated_software_validation_only",
        "data_origin": "programmatically_generated_no_real_records",
        "fixture_seed": args.seed,
        "warning": "Invented records and category meanings; not IPUMS research results.",
    }
    (args.out / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    samples = []
    for draw in range(2):
        sample = generated_records(args.sample_rows, rng)
        for arm in ("A_diag", "B_private", "E_matched", "C"):
            values = sample.copy()
            if arm != "A_diag":
                values[:, 0] += rng.normal(0, 1.2, len(values)).astype(np.float32)
            if arm in ("B_private", "E_matched"):
                for col in (2, 3, 5, 6):
                    values[:, col] = rng.permutation(values[:, col])
            if arm == "C":
                # Exercise the race/Hispanic joint-error diagnostic deliberately.
                values[:, 3] = 1 - values[:, 3]
            filename = f"fixture_{arm}_draw_{draw + 1}.npy"
            np.save(args.out / filename, values, allow_pickle=False)
            samples.append({"arm": arm, "fit_id": "fixture_fit_1",
                            "target_id": "fixture_target_1" if arm == "C" else None,
                            "draw_id": draw + 1,
                            "path": filename})
    manifest = {"public_fixture": True,
                "warning": "Arm names label artificial test cases, not fitted models.",
                "samples": samples}
    (args.out / "samples_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created generated software-validation fixture in {args.out}")
    print("NO REAL RECORDS; NOT A RESEARCH EXPERIMENT RESULT")


if __name__ == "__main__":
    main()
