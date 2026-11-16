#!/usr/bin/env python3
"""Validate the saved 2,000-update GPU samples and write one diagnostic manifest.

This reads existing arrays and metadata only. It never trains, samples a model,
measures a target, or deserializes checkpoints or target values. The shared
fit_id identifies a matched experiment block, not a common checkpoint.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from check_prototype2_rivanna_inputs import EXPECTED_CATEGORIES, check_inputs, sha256_file
from prototype2_evaluation import FEATURES, evaluation_codes


ARMS = ("A_diag", "clip_only", "B_private", "E_matched", "C")
REQUIRED_METADATA = ("predeclared_plan.json", "COMPLETED.json", "metrics.json",
                     "training_metadata.json", "dp_targets_01.json")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _verify_plan(plan, preflight):
    _require(isinstance(plan, dict), "The predeclared plan must be a JSON object")
    expected = {"training_replicates": 1, "target_repeats": 1,
                "generation_repeats": 3, "samples_per_arm": 512}
    for key, value in expected.items():
        _require(type(plan.get(key)) is int and plan[key] == value,
                 f"Expected predeclared {key}={value}")
    config, dataset = plan.get("config"), plan.get("dataset")
    _require(isinstance(config, dict) and config.get("features") == FEATURES,
             "Plan config.features must match the seven modeled features")
    _require(type(config.get("steps")) is int and config["steps"] == 2000,
             "Expected exactly 2,000 training updates")
    _require(isinstance(dataset, dict), "Plan dataset metadata is required")
    for key, value in (("train_rows", 2048), ("holdout_rows", 512)):
        _require(type(dataset.get(key)) is int and dataset[key] == value,
                 f"Expected dataset.{key}={value}")
    _require(dataset.get("categorical_domains") == EXPECTED_CATEGORIES,
             "Plan categorical domains differ from the historical input")
    job = plan.get("job_id")
    _require(type(job) in (str, int) and re.fullmatch(r"[0-9]+", str(job)),
             "Plan job_id must be a nonempty numeric identifier")
    seeds = plan.get("generation_seeds")
    _require(isinstance(seeds, list) and len(seeds) == 3
             and all(type(seed) is int for seed in seeds) and len(set(seeds)) == 3,
             "Expected three distinct integer public generation seeds")
    hashes = plan.get("input_and_original_artifact_sha256")
    _require(isinstance(hashes, dict), "Plan input hash map is required")
    for filename, key in (("pilot_data.npz", "data_npz_sha256"),
                          ("schema.json", "schema_sha256")):
        matches = [value for path, value in hashes.items()
                   if path.endswith("/ipums_2048/" + filename)]
        _require(len(matches) == 1 and matches[0] == preflight[key],
                 f"Plan must record the verified historical {filename} hash unambiguously")
    return str(job), seeds


def _verify_metrics(metrics, job, seeds):
    _require(isinstance(metrics, dict), "The metrics report must be a JSON object")
    report_job = metrics.get("job_id")
    _require(type(report_job) in (str, int) and str(report_job) == job,
             "Metrics job_id differs from the predeclared plan")
    expected = {"independent_training_replicates": 1, "marginal_releases": 1,
                "generation_repeats": 3, "samples_per_arm_per_draw": 512}
    for key, value in expected.items():
        _require(type(metrics.get(key)) is int and metrics[key] == value,
                 f"Expected metrics {key}={value}")
    evaluations = metrics.get("evaluations")
    _require(isinstance(evaluations, dict), "Metrics evaluations mapping is required")
    for arm in ARMS:
        draws = evaluations.get(arm)
        _require(isinstance(draws, list) and len(draws) == 3,
                 f"Expected three recorded evaluations for {arm}")
        for index, draw in enumerate(draws):
            _require(isinstance(draw, dict) and type(draw.get("generation_seed")) is int
                     and draw["generation_seed"] == seeds[index],
                     f"Recorded generation-seed order differs for {arm}")
            if arm == "C":
                _require(type(draw.get("target_index")) is int and draw["target_index"] == 1,
                         "Every C draw must record the same cached target_index=1")


def _recorded_sample_hashes(node, names, location="metrics"):
    """Recognize only explicit saved filenames mapped to SHA256 strings."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{location}/{key}"
            if key in names and isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                found.append((key, value.lower(), child))
            found.extend(_recorded_sample_hashes(value, names, child))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_recorded_sample_hashes(value, names, f"{location}/{index}"))
    return found


def prepare_manifest(run_dir, data_npz, schema_path, out):
    """Validate all inputs, then exclusively create out and return its payload."""
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite manifest: {out}")
    _require(run_dir.is_dir(), "--run-dir must be an existing directory")
    for name in REQUIRED_METADATA:
        _require((run_dir / name).is_file(), f"Missing required run file: {name}")
    preflight = check_inputs(data_npz, schema_path)
    plan = json.loads((run_dir / "predeclared_plan.json").read_text(encoding="utf-8"))
    job, seeds = _verify_plan(plan, preflight)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    _verify_metrics(metrics, job, seeds)
    samples, sample_hashes = [], {}
    for arm in ARMS:
        for draw, seed in enumerate(seeds, 1):
            path = run_dir / f"{arm}_generation_{draw:02d}.npz"
            _require(path.is_file(), f"Missing saved sample: {path.name}")
            with np.load(path, allow_pickle=False) as archive:
                _require({"X_synthetic", "categories", "features"} <= set(archive.files),
                         f"Missing required arrays in {path.name}")
                values, categories, features = (archive[key] for key in
                                                ("X_synthetic", "categories", "features"))
                _require(features.shape == (7,) and np.array_equal(features, FEATURES),
                         f"Feature values/order differ in {path.name}")
                _require(categories.shape == (6,) and np.array_equal(categories, EXPECTED_CATEGORIES),
                         f"Categorical cardinalities differ in {path.name}")
                _require(values.shape == (512, 7) and values.dtype.kind in "iuf"
                         and np.isfinite(values).all(),
                         f"Expected a finite numeric 512-by-7 array in {path.name}")
                evaluation_codes(values, EXPECTED_CATEGORIES)  # Preserve invalid AGE tails.
            digest = sha256_file(path)
            sample_hashes[path.name] = digest
            samples.append({"arm": arm, "fit_id": f"gpu{job}_trainingrep01",
                            "target_id": "target01" if arm == "C" else None,
                            "draw_id": draw, "generation_seed": seed,
                            "model_arm": "B_private" if arm == "C" else arm,
                            "path": str(path.resolve(strict=True)), "key": "X_synthetic", "n_rows": 512,
                            "sha256": digest})
    recorded = _recorded_sample_hashes(metrics, set(sample_hashes))
    for name, digest, location in recorded:
        _require(digest == sample_hashes[name], f"Recorded sample hash mismatch at {location}")
    files = list(REQUIRED_METADATA)
    files += [f"{arm}_final.pt" for arm in ARMS if arm != "C"
              and (run_dir / f"{arm}_final.pt").is_file()]
    payload = {
        "format_version": 1, "status": "validated_saved_gpu_sample_manifest",
        "nonprivate_research": True, "training_performed": False, "new_marginal_releases": 0,
        "scope": {"independent_training_replicates": 1, "cached_target_releases": 1,
                  "generation_draws_per_arm": 3, "rows_per_draw": 512,
                  "fit_id_meaning": "Matched experiment block; trained arms have separate checkpoints. C reuses frozen B.",
                  "generation_seed_source": "Recorded metrics evaluation order agrees with the predeclared seeds and original writer convention; no model replay performed.",
                  "limitations": "Checks establish metadata and array consistency, not independent reconstruction of checkpoint lineage. Completion-marker contents and target values are not interpreted."},
        "provenance": {"run_dir": str(run_dir), "input_preflight": preflight,
                       "files": {name: {"path": str(run_dir / name), "sha256": sha256_file(run_dir / name)}
                                 for name in files},
                       "recorded_sample_hash_verification": {
                           "verified_entries": [{"filename": name, "json_location": loc} for name, _, loc in recorded],
                           "filenames_without_recognized_digest": sorted(set(sample_hashes) - {x[0] for x in recorded})}},
        "samples": samples,
    }
    with out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New manifest JSON; no overwrites")
    args = parser.parse_args(argv)
    try:
        payload = prepare_manifest(args.run_dir, args.data_npz, args.schema, args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f"Saved GPU manifest preparation failed: {error}", file=sys.stderr)
        return 2
    print(f"Validated {len(payload['samples'])} saved arrays; wrote {args.out}. No training or new target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
