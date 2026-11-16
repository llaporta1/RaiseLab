#!/usr/bin/env python3
"""Audit one existing categorical DP target against its exact TRAIN marginals.

This is a CPU-only, NONPRIVATE LOCAL RESEARCH diagnostic. It reads existing
data and a cached target; it never fits, draws noise, or releases a new target.
Only the historical format defined in run_ipums_cd_pilot.py is accepted.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

import check_prototype2_rivanna_inputs as input_checks
import prototype2_evaluation as evaluation
from check_prototype2_rivanna_inputs import check_inputs, sha256_file
from prototype2_evaluation import FEATURES, load_pilot


TARGETS = FEATURES[1:]
TARGET_SOURCE_SHA256 = "fa67be1dfba13a200d96cc9ef2257a0ef053f63f564ecfe5c0855de706280e90"
MECHANISM = "six_laplace_histogram_vectors_add_remove_research"
PSEUDOCOUNT = 1e-3
POSTPROCESSING = "max(noisy_counts,0)+fixed_pseudocount; normalize noisy vector"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_cached_targets(payload, schema, epsilon):
    """NumPy-only validation of the verified historical writer's one format.

    The source is run_ipums_cd_pilot.py:create_dp_targets/validate_dp_targets,
    whose exact hash is recorded in the original GPU plan. No schema guessing
    and no target generation are supported here.
    """
    require(type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon > 0,
            "Expected finite positive cached-target epsilon")
    require(isinstance(payload, dict), "Cached target must be a JSON object")
    require(type(payload.get("format_version")) is int and payload["format_version"] == 1,
            "Unknown cached target format")
    require(payload.get("mechanism") == MECHANISM, "Unknown cached target mechanism")
    require(payload.get("features") == TARGETS, "Cached target features/order differ")
    require(payload.get("category_domains") == schema["category_domains"],
            "Cached target category domains/order differ from the verified schema")
    require(payload.get("postprocessing") == POSTPROCESSING,
            "Cached target postprocessing differs from the verified writer")
    expected = {"epsilon_marginals": epsilon, "delta_marginals": 0.0,
                "epsilon_per_histogram": epsilon / len(TARGETS),
                "l1_sensitivity_per_histogram": 1.0,
                "laplace_scale_per_cell": len(TARGETS) / epsilon,
                "count_pseudocount": PSEUDOCOUNT}
    for name, value in expected.items():
        actual = payload.get(name)
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12),
                f"Cached target setting does not match: {name}")
    probabilities = payload.get("probabilities")
    require(isinstance(probabilities, dict) and set(probabilities) == set(TARGETS),
            "Expected exactly six named cached probability vectors")
    result = {}
    for name in TARGETS:
        raw = probabilities[name]
        require(isinstance(raw, list) and all(type(x) in (int, float) for x in raw),
                f"Expected numeric probability vector for {name}")
        vector = np.asarray(raw, dtype=np.float64)
        require(vector.shape == (len(schema["category_domains"][name]),),
                f"Wrong cached target length for {name}")
        require(np.isfinite(vector).all() and np.all(vector > 0)
                and np.isclose(vector.sum(), 1.0, rtol=0, atol=1e-10),
                f"Invalid cached probability vector for {name}")
        result[name] = vector
    return result


def _verify_plan(plan, preflight):
    require(isinstance(plan, dict), "The predeclared plan must be a JSON object")
    require(type(plan.get("target_repeats")) is int and plan["target_repeats"] == 1,
            "Expected exactly one cached target release in the plan")
    source_hashes = plan.get("original_source_sha256")
    require(isinstance(source_hashes, dict)
            and source_hashes.get("scripts/run_ipums_cd_pilot.py") == TARGET_SOURCE_SHA256,
            "Plan does not identify the verified cached-target writer")
    privacy = plan.get("privacy_accounting")
    require(isinstance(privacy, dict), "Plan privacy_accounting is required")
    epsilon = privacy.get("epsilon_marginals_one_cached_release")
    require(type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon > 0,
            "Plan cached-target epsilon must be finite and positive")
    hashes = plan.get("input_and_original_artifact_sha256")
    require(isinstance(hashes, dict), "Plan historical-input hash map is required")
    for filename, key in (("pilot_data.npz", "data_npz_sha256"), ("schema.json", "schema_sha256")):
        matches = [value for path, value in hashes.items()
                   if path.endswith("/ipums_2048/" + filename)]
        require(len(matches) == 1 and matches[0] == preflight[key],
                f"Plan does not unambiguously match the verified {filename}")
    return float(epsilon)


def _text_report(report):
    lines = ["CACHED TARGET AUDIT — NONPRIVATE LOCAL RESEARCH", "",
             "No training, new noise draw, or new DP target release was performed.",
             "Primary reference: exact empirical TRAINING marginals.",
             "Target-minus-training includes noise postprocessing and normalization; it is not the raw Laplace noise.",
             "Holdout comparisons are separate sampling context, not noise attribution.", "",
             "Feature | target vs train L1 | target vs smoothed train L1 | train vs holdout L1 | target vs holdout L1"]
    for name, values in report["features"].items():
        lines.append(f"{name} | {values['target_vs_training_l1']:.9f} | "
                     f"{values['target_vs_smoothed_training_oracle_l1']:.9f} | "
                     f"{values['training_vs_holdout_l1']:.9f} | {values['target_vs_holdout_l1']:.9f}")
    for name, entries in report["targeted_entries"].items():
        lines += ["", name + " public-code entries", "Code | train count | train probability | cached target probability | target minus train | holdout probability"]
        for row in entries:
            lines.append(f"{row['public_code']} | {row['training_count']} | {row['training_probability']:.9f} | "
                         f"{row['cached_target_probability']:.9f} | {row['target_minus_training_probability']:+.9f} | "
                         f"{row['holdout_probability']:.9f}")
    lines += ["", "Cached target SHA256: " + report["provenance"]["files"]["cached_target"]["sha256"],
              "", "This audit characterizes one realized target. It does not show how a frozen model responds to it or establish target-noise variability across releases."]
    return "\n".join(lines) + "\n"


def audit_cached_targets(run_dir, data_npz, schema_path, out):
    """Validate existing inputs and exclusively write two reports to a new dir."""
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output directory: {out}")
    preflight = check_inputs(data_npz, schema_path)
    plan_path, target_path = run_dir / "predeclared_plan.json", run_dir / "dp_targets_01.json"
    plan_bytes, target_bytes = plan_path.read_bytes(), target_path.read_bytes()
    plan, payload = json.loads(plan_bytes), json.loads(target_bytes)
    epsilon = _verify_plan(plan, preflight)
    source = Path(__file__).with_name("run_ipums_cd_pilot.py")
    require(sha256_file(source) == TARGET_SOURCE_SHA256,
            "Local historical target writer differs from its verified source hash")
    train, holdout, categories, schema = load_pilot(data_npz, schema_path)
    targets = validate_cached_targets(payload, schema, epsilon)
    features, entries = {}, {}
    for column, (name, size) in enumerate(zip(TARGETS, categories), 1):
        tc = np.bincount(train[:, column].astype(np.int64), minlength=int(size))
        hc = np.bincount(holdout[:, column].astype(np.int64), minlength=int(size))
        truth, held = tc / len(train), hc / len(holdout)
        smoothed = (tc + PSEUDOCOUNT) / (len(train) + int(size) * PSEUDOCOUNT)
        target = targets[name]
        difference = target - truth
        features[name] = {
            "domain_size": int(size), "training_observed_categories": int(np.count_nonzero(tc)),
            "holdout_observed_categories": int(np.count_nonzero(hc)),
            "target_vs_training_l1": float(abs(difference).sum()),
            "target_vs_smoothed_training_oracle_l1": float(abs(target - smoothed).sum()),
            "training_vs_holdout_l1": float(abs(truth - held).sum()),
            "target_vs_holdout_l1": float(abs(target - held).sum()),
            "max_absolute_target_minus_training_probability": float(abs(difference).max()),
            "target_mass_in_training_unobserved_categories": float(target[tc == 0].sum()),
            "target_probability_sum": float(target.sum()),
        }
        if name in ("RACE", "HISPAN"):
            entries[name] = [{"encoded_index": i, "public_code": code,
                              "training_count": int(tc[i]), "holdout_count": int(hc[i]),
                              "training_probability": float(truth[i]),
                              "smoothed_training_oracle_probability": float(smoothed[i]),
                              "holdout_probability": float(held[i]),
                              "cached_target_probability": float(target[i]),
                              "target_minus_training_probability": float(difference[i]),
                              "absolute_target_minus_training_probability": float(abs(difference[i]))}
                             for i, code in enumerate(schema["category_domains"][name])]
    target_hash = hashlib.sha256(target_bytes).hexdigest()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    require(sha256_file(target_path) == target_hash and sha256_file(plan_path) == plan_hash,
            "Target or plan changed during the audit")
    report = {
        "format_version": 1, "status": "complete", "NONPRIVATE_LOCAL_RESEARCH": True,
        "nonprivate_research": True, "formal_end_to_end_dp_claimed": False,
        "training_performed": False, "new_marginal_releases": 0, "fresh_target_draws": 0,
        "cached_target_reused": True, "training_rows": len(train), "holdout_rows": len(holdout),
        "epsilon_of_existing_cached_target": epsilon,
        "provenance": {"job_id_from_plan": plan.get("job_id"), "input_preflight": preflight,
                       "files": {"cached_target": {"path": str(target_path.resolve()), "sha256": target_hash},
                                 "predeclared_plan": {"path": str(plan_path.resolve()), "sha256": plan_hash},
                                 "historical_target_writer": {"path": str(source.resolve()), "sha256": TARGET_SOURCE_SHA256},
                                 "target_auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
                                 "common_evaluator": {"path": str(Path(evaluation.__file__).resolve()), "sha256": sha256_file(evaluation.__file__)},
                                 "check_inputs": {"path": str(Path(input_checks.__file__).resolve()), "sha256": sha256_file(input_checks.__file__)},
                                 "data_npz": {"path": str(Path(data_npz).expanduser().resolve()), "sha256": preflight["data_npz_sha256"]},
                                 "schema": {"path": str(Path(schema_path).expanduser().resolve()), "sha256": preflight["schema_sha256"]}}},
        "target_validation": {"format_version": 1, "mechanism": MECHANISM,
                              "features": TARGETS, "category_domains_match": True,
                              "all_probability_vectors_finite_positive_and_normalized": True,
                              "postprocessing": POSTPROCESSING},
        "features": features, "targeted_entries": entries,
        "interpretation": [
            "Target-versus-training is the realized postprocessed target discrepancy, including Laplace perturbation, rectification, pseudocount, and normalization. The cached probabilities do not recover the original additive noise draws.",
            "Exact empirical training marginals are the primary reference. A smoothed training oracle matching the historical pseudocount is reported separately.",
            "Training-versus-holdout and target-versus-holdout are sampling context. Holdout discrepancies alone do not identify target noise.",
            "The exact training/holdout summaries make this report NONPRIVATE internal research, not a privacy-accounted release.",
            "One cached target does not measure target-realization variability. A separate fixed-model guidance ablation is needed to measure response to this target."
        ],
    }
    text = _text_report(report)
    out.mkdir(parents=False, exist_ok=False)
    with (out / "target_audit.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (out / "target_audit.txt").open("x", encoding="utf-8") as handle:
        handle.write(text)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-npz", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New output directory; no overwrites")
    args = parser.parse_args(argv)
    try:
        audit_cached_targets(args.run_dir, args.data_npz, args.schema, args.out)
    except (OSError, ValueError, KeyError) as error:
        print(f"Cached target audit failed: {error}", file=sys.stderr)
        return 2
    print(f"Cached target audit complete: {args.out}. No training or new target release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
