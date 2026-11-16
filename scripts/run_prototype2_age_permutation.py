#!/usr/bin/env python3
"""Evaluate saved samples after permuting AGE alone, using CPU/NumPy only.

The six categorical columns remain unchanged in every row. No model is
loaded, sampled, trained, or changed, and no private target is measured.
Real-data runs require the exact historical pilot hashes. Generated fixtures
require --public-fixture and the explicit fixture declarations in the schema.
Repeat --samples-manifest to evaluate multiple saved-output manifests with one
shared series of dependent, disjoint-within-repeat real-real controls.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys

import numpy as np

import check_prototype2_rivanna_inputs as input_gate
import prototype2_evaluation as evaluation
import run_prototype2_diagnostics as diagnostics


FEATURES = evaluation.FEATURES
PAIR_QUERIES = tuple(("AGE", name) for name in FEATURES[1:])
TRIPLE_QUERIES = evaluation.FIXED_THREE_WAY_QUERIES
QUERY_COLUMNS = tuple(
    (("two_way/" if len(query) == 2 else "three_way/") + " x ".join(query),
     tuple(FEATURES.index(name) for name in query))
    for query in PAIR_QUERIES + TRIPLE_QUERIES)
METRIC_FIELDS = ("original", "age_permuted_mean", "age_permuted_minus_original")


def normalized_draw_id(draw_id):
    """Normalize only an explicit integer or recognizable numeric draw label.

    This affects permutation streams only; the original manifest identifier
    is retained unchanged and does not establish statistical pairing.
    """
    if isinstance(draw_id, bool) or not isinstance(draw_id, (str, int)):
        raise ValueError("draw_id must be a string or integer")
    if isinstance(draw_id, int):
        return {"type": "index", "value": draw_id}
    match = re.fullmatch(r"(?:(?:draw|generation)[_-]?)?(\d+)", draw_id, re.IGNORECASE)
    return ({"type": "index", "value": int(match.group(1))} if match else
            {"type": "literal", "value": draw_id})


def permutation_seed(seed, draw_id, rows, *, control=False):
    payload = ["real-real AGE-only" if control else "saved-sample AGE-only",
               int(seed), int(rows), normalized_draw_id(draw_id)]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def permute_age(sample, rng):
    """Return a copy with AGE permuted and categorical rows exactly unchanged."""
    values, _ = diagnostics.finite_matrix(sample, "AGE-permutation input")
    shuffled = values.copy()
    shuffled[:, 0] = values[rng.permutation(len(values)), 0]
    if not np.array_equal(shuffled[:, 1:], values[:, 1:]):
        raise RuntimeError("AGE permutation altered categorical rows")
    if not np.array_equal(np.sort(shuffled[:, 0]), np.sort(values[:, 0])):
        raise RuntimeError("AGE permutation altered the AGE values or histogram")
    return shuffled


def query_metrics(codes, reference_codes, domains):
    metrics = {name: evaluation.marginal_l1(codes, reference_codes, columns, domains)
               for name, columns in QUERY_COLUMNS}
    metrics["marginal/age_two_way"] = float(np.mean(
        [metrics[name] for name, _ in QUERY_COLUMNS if name.startswith("two_way/")]))
    metrics["marginal/fixed_three_way"] = float(np.mean(
        [metrics[name] for name, _ in QUERY_COLUMNS if name.startswith("three_way/")]))
    return metrics


def evaluate_age_permutations(sample, reference, categories, *, repeats=20,
                              seed=0, sample_real_records=False):
    """Use the common code construction and L1 evaluator for nine fixed queries."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 50:
        raise ValueError("Permutation repeats must be an integer from 1 through 50")
    codes, age_info = evaluation.evaluation_codes(sample, categories,
                                                 real_records=sample_real_records)
    reference_codes, _ = evaluation.evaluation_codes(reference, categories, real_records=True)
    domains = np.r_[len(evaluation.AGE_EDGES) + 1, categories]
    original = query_metrics(codes, reference_codes, domains)
    original_histogram = np.bincount(codes[:, 0], minlength=int(domains[0]))
    rng = np.random.default_rng(seed)
    permutations = {key: [] for key in original}
    for _ in range(repeats):
        shuffled = permute_age(sample, rng)
        shuffled_codes, current_age = evaluation.evaluation_codes(
            shuffled, categories, real_records=sample_real_records)
        if not np.array_equal(np.bincount(shuffled_codes[:, 0], minlength=int(domains[0])),
                              original_histogram) or current_age != age_info:
            raise RuntimeError("AGE-only permutation changed an AGE histogram or support metric")
        current = query_metrics(shuffled_codes, reference_codes, domains)
        for key, value in current.items():
            permutations[key].append(value)
    metrics = {}
    for key, values in permutations.items():
        mean = statistics.mean(values)
        metrics[key] = {
            "original": original[key], "age_permuted_mean": mean,
            "age_permuted_minus_original": mean - original[key],
            "age_permuted_sample_sd": statistics.stdev(values) if repeats > 1 else None,
            "age_permuted_min": min(values), "age_permuted_max": max(values),
        }
    return {
        "metrics": metrics, "age": age_info,
        "age_one_way_l1_unchanged": evaluation.marginal_l1(codes, reference_codes, (0,), domains),
        "checks": {"categorical_rows_exactly_unchanged": True,
                   "AGE_values_histogram_and_support_unchanged": True,
                   "sample_input_not_modified": True,
                   "completed_permutations": repeats},
    }


def load_manifests(paths, categories, *, public_fixture=False):
    """Combine manifests, reject identifier collisions, and verify declared hashes."""
    records, manifest_sources, identifiers, seen_manifests = [], [], set(), set()
    for supplied in paths:
        path = Path(supplied).expanduser().resolve(strict=True)
        if path in seen_manifests:
            raise ValueError("The same sample manifest was supplied more than once")
        seen_manifests.add(path)
        payload = json.loads(path.read_text())
        entries = payload.get("samples") if isinstance(payload, dict) else payload
        current = diagnostics.load_manifest(path, categories)
        declared_fixture = isinstance(payload, dict) and payload.get("public_fixture") is True
        if not public_fixture and (declared_fixture or any(entry.get("public_fixture") is True for entry in entries)):
            raise ValueError("Public generated sample fixtures require --public-fixture")
        manifest_sources.append({"path": str(path), "sha256": diagnostics.sha256(path),
                                 "sample_count": len(current)})
        for entry, record in zip(entries, current):
            expected = entry.get("sha256")
            if expected is not None and expected != record["sha256"]:
                raise ValueError(f"Saved sample hash differs from manifest: {record['path']}")
            identity = (diagnostics.canonical_arm(record["arm"]),
                        *diagnostics.pairing_key(record))
            if identity in identifiers:
                raise ValueError("Duplicate arm/fit/target/draw identifiers across manifests")
            identifiers.add(identity)
            record.update(record_id=f"sample_{len(records):04d}",
                          source_manifest=str(path),
                          manifest_declared_sample_hash_verified=expected is not None)
            records.append(record)
    if not records:
        raise ValueError("At least one saved sample is required")
    return records, manifest_sources


def aggregate_fit_groups(samples):
    """Nest generation draws within targets within fits, without new fit units."""
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample in samples:
        groups[sample["arm"]][diagnostics.identifier_token(sample["fit_id"])][
            diagnostics.identifier_token(sample["target_id"])].append(sample)
    aggregate = {}
    for arm, fits in sorted(groups.items()):
        per_fit = []
        for fit_id, targets in sorted(fits.items()):
            target_means = []
            for target_id, draws in sorted(targets.items()):
                target_means.append({
                    "target_id": json.loads(target_id), "generation_draws": len(draws),
                    "metrics": {metric: {field: statistics.mean(s["metrics"][metric][field] for s in draws)
                                         for field in METRIC_FIELDS}
                                for metric in draws[0]["metrics"]},
                })
            per_fit.append({
                "fit_id": json.loads(fit_id), "target_groups": target_means,
                "metrics": {metric: {field: statistics.mean(t["metrics"][metric][field] for t in target_means)
                                     for field in METRIC_FIELDS}
                            for metric in target_means[0]["metrics"]},
            })
        summary = {}
        for metric in per_fit[0]["metrics"]:
            summary[metric] = {}
            for field in METRIC_FIELDS:
                values = [f["metrics"][metric][field] for f in per_fit]
                summary[metric][field] = {
                    "mean": statistics.mean(values),
                    "between_fit_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                    "n_fit_groups": len(values),
                }
        aggregate[arm] = {"n_fit_groups": len(per_fit), "metrics": summary, "per_fit": per_fit}
    return aggregate


def source_provenance(args, manifests, gate):
    files = {
        "data_npz": args.data_npz, "schema": args.schema,
        "runner": Path(__file__), "common_evaluator": Path(evaluation.__file__),
        "diagnostic_helpers": Path(diagnostics.__file__), "input_gate": Path(input_gate.__file__),
    }
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).resolve().parent,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "files": {name: {"path": str(Path(path).resolve()), "sha256": diagnostics.sha256(path)}
                  for name, path in files.items()},
        "sample_manifests": manifests, "input_preflight": gate, "git_head": revision,
        "python": platform.python_version(), "numpy": np.__version__, "seed": args.seed,
        "utc_started": datetime.now(timezone.utc).isoformat(),
    }


def write_metric_csv(path, samples, *, controls=False):
    identifiers = (["repeat_id", "rows_per_side"] if controls else
                   ["record_id", "arm", "fit_id", "target_id", "draw_id", "rows", "reference_rows"])
    fields = identifiers + ["permutation_seed", "metric", *METRIC_FIELDS,
                            "age_permuted_sample_sd", "age_permuted_min", "age_permuted_max"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {key: sample[key] for key in identifiers}
            if not controls:
                for key in ("fit_id", "target_id", "draw_id"):
                    row[key] = diagnostics.identifier_token(row[key])
            row["permutation_seed"] = sample["permutation_seed"]
            for metric, values in sample["metrics"].items():
                writer.writerow({**row, "metric": metric, **values})


def run_into(args):
    if args.public_fixture:
        fixture_schema = json.loads(args.schema.read_text())
        if (fixture_schema.get("public_fixture") is not True or
                fixture_schema.get("data_origin") != "programmatically_generated_no_real_records"):
            raise ValueError("--public-fixture requires an explicitly declared generated fixture schema")
        gate = {"status": "public_generated_fixture_only", "historical_input_hash_gate_bypassed": True}
    else:
        gate = input_gate.check_inputs(args.data_npz, args.schema)
    train, holdout, categories, schema = evaluation.load_pilot(args.data_npz, args.schema)
    if args.real_real_repeats and len(train) < 2 * args.sample_rows:
        raise ValueError("Training pool is too small for disjoint real-real samples; no bootstrap fallback")
    records, manifests = load_manifests(args.samples_manifest, categories,
                                       public_fixture=args.public_fixture)
    report = {
        "format_version": 1, "status": "complete", "public_fixture": args.public_fixture,
        "label": ("PUBLIC GENERATED FIXTURE: SOFTWARE VALIDATION ONLY" if args.public_fixture else
                  "NONPRIVATE RESEARCH DIAGNOSTICS ON EXISTING OUTPUTS"),
        "nonprivate_research": not args.public_fixture,
        "training_performed": False, "new_marginal_releases": 0,
        "frozen_model_intervention_performed": False,
        "permutation_type": "AGE_only", "permutation_repeats": args.permutation_repeats,
        "sample_count": len(records), "features": FEATURES,
        "age_edges": evaluation.AGE_EDGES.tolist(),
        "category_cardinalities": categories.tolist(),
        "provenance": source_provenance(args, manifests, gate),
        "permutation_stream_policy": {
            "common_across_arms_and_fits": True,
            "key": "base seed + normalized draw identifier + row count",
            "numeric_draw_normalization": "1, '1', 'draw001', 'generation_01' share a stream for the same row count",
            "manifest_order_affects_streams": False,
            "controls_use_separate_namespace": True,
            "note": "Shared permutations reduce Monte Carlo differences; they do not make separate fitted models paired training replicates.",
        },
        "samples": [],
        "real_real": {"pool": "X_train only", "repeats": args.real_real_repeats,
                      "rows_per_side": args.sample_rows, "within_repeat_disjoint": True,
                      "across_repeat_independent": False, "results": []},
        "interpretation": [
            "Positive age_permuted_minus_original means the original has lower empirical joint L1 than its AGE-only permutation control.",
            "Only AGE is reassigned. All six categorical values remain aligned in their original rows; categorical one-way and joint distributions are exactly unchanged.",
            "The full empirical AGE distribution, tail fractions and histogram are preserved. Generated AGE is never rounded, clipped or filtered.",
            "The control probes AGE alignment with the categorical row; it is not a significance test, causal model intervention, or certification of higher-order learning.",
            "Scores use full denominators and the existing public AGE bins. Near-ceiling L1 can limit the observable shuffle advantage.",
            "Generation draws and target variants are nested within declared fit groups. One historical GPU fit provides no between-fit SD; C shares B and one cached target.",
            "Equal fit labels across arms do not establish independent paired training trials. No p-values are computed.",
            "The 20 default real-real repetitions sample disjoint sides within X_train, can overlap across repeats, and are dependent reference diagnostics rather than an attainable floor.",
            "--sample-rows controls real-real sizes only. Saved arrays are evaluated in full against the full supplied holdout; unequal sizes are recorded explicitly.",
            "Input and output hashes record provenance but supplied fit/target identifiers do not independently authenticate checkpoint lineage.",
            "Reports contain aggregate statistics and identifiers, not raw records. Exact reference metrics are not an end-to-end private release.",
        ],
    }
    for record in records:
        values = record["values"]
        stream_seed = permutation_seed(args.seed, record["draw_id"], len(values))
        evaluated = evaluate_age_permutations(values, holdout, categories,
                                             repeats=args.permutation_repeats, seed=stream_seed)
        report["samples"].append({
            **{key: value for key, value in record.items() if key != "values"},
            "reference_rows": len(holdout), "permutation_seed": stream_seed,
            "normalized_draw_id": normalized_draw_id(record["draw_id"]),
            "rows_equal_real_real_side": len(values) == args.sample_rows,
            **evaluated,
        })
    report["aggregate"] = aggregate_fit_groups(report["samples"])
    rng = np.random.default_rng(args.seed)
    for repeat in range(args.real_real_repeats):
        left, right = diagnostics.draw_disjoint_indices(len(train), args.sample_rows, rng)
        if np.intersect1d(left, right).size:
            raise RuntimeError("Real-real control sides overlap")
        stream_seed = permutation_seed(args.seed, repeat, args.sample_rows, control=True)
        evaluated = evaluate_age_permutations(train[left], train[right], categories,
                                             repeats=args.permutation_repeats, seed=stream_seed,
                                             sample_real_records=True)
        report["real_real"]["results"].append({
            "repeat_id": repeat, "rows_per_side": args.sample_rows,
            "within_repeat_index_overlap_count": 0, "permutation_seed": stream_seed, **evaluated,
        })
    if report["real_real"]["results"]:
        report["real_real"]["summary"] = {
            metric: {field: {
                "mean": statistics.mean(s["metrics"][metric][field] for s in report["real_real"]["results"]),
                "min": min(s["metrics"][metric][field] for s in report["real_real"]["results"]),
                "max": max(s["metrics"][metric][field] for s in report["real_real"]["results"]),
            } for field in METRIC_FIELDS}
            for metric in report["real_real"]["results"][0]["metrics"]
        }
    (args.out / "age_permutation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_metric_csv(args.out / "metrics_per_draw.csv", report["samples"])
    write_metric_csv(args.out / "real_real_controls.csv", report["real_real"]["results"], controls=True)
    (args.out / "README.txt").write_text(
        report["label"] + "\n\n"
        "AGE-ONLY PERMUTATION: preserved categorical rows, AGE values, and all one-way marginals.\n"
        "No training, target release, model loading, resampling from models, or model intervention occurred.\n"
        "age_permutation.json contains provenance, sample hashes, per-draw scores, nested fit summaries and controls.\n"
        "metrics_per_draw.csv contains the six AGE pairs, three fixed triples and their two mean scores.\n"
        "real_real_controls.csv contains the dependent disjoint-within-repeat training-pool controls.\n"
        "Positive age_permuted_minus_original favors the original empirical joint score. This is descriptive, not a p-value.\n"
        "Generation and permutation repeats are not independent fits. Fit SD is null for a single declared fit group.\n"
        "Target-specific draw means are averaged within fit before any between-fit summary.\n"
        "Common permutation streams depend on normalized draw ID and row count, not manifest order or arm.\n"
        "No raw records are copied to these reports; exact real-reference metrics remain nonprivate.\n",
        encoding="utf-8")
    return report


def run(args):
    if (args.seed < 0 or not 1 <= args.permutation_repeats <= 50 or
            not 0 <= args.real_real_repeats <= 100 or args.sample_rows < 1):
        raise ValueError("Require seed>=0, 1–50 permutations, 0–100 real-real repeats, and sample_rows>=1")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        return run_into(args)
    except Exception as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                   "training_performed": False, "new_marginal_releases": 0,
                   "no_rows_filtered": True,
                   "partial_reports_are_not_completed_results": True}
        (args.out / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--samples-manifest", type=Path, action="append", required=True,
                        help="Repeat to combine saved-output manifests in one diagnostic run")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--permutation-repeats", type=int, default=20, help="1–50, fixed before inspection")
    parser.add_argument("--real-real-repeats", type=int, default=20)
    parser.add_argument("--sample-rows", type=int, default=512, help="Rows on each side of a real-real control")
    parser.add_argument("--public-fixture", action="store_true",
                        help="Software validation only; schema must explicitly declare generated public records")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        report = run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"AGE-only diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(f"Completed AGE-only permutation for {report['sample_count']} saved samples, "
          f"{args.permutation_repeats} permutations each and {args.real_real_repeats} dependent controls: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
