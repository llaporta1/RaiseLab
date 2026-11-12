#!/usr/bin/env python3
"""Audit existing Prototype 2 outputs without training or releasing new targets.

Inputs are the legacy encoded N-by-7 matrices: AGE=(years-50)/50, followed
by categorical indices in FEATURES order. Reports contain aggregate counts
and metrics, never copies of input rows. Real-data diagnostics are NONPRIVATE.

A samples manifest is a JSON list, or an object containing a samples list,
of explicit records with arm, fit_id, target_id, draw_id, path, and key.
Paths are relative to the manifest. Use null or omit key for .npy, and use
an explicit array key for .npz. C is paired with a
unique B/E control having identical fit_id and draw_id. B/E target_id should
be null because those controls have no measured target. C target variants
can share that control, with the reuse recorded explicitly in the report.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prototype2_evaluation as evaluation


FEATURES = list(evaluation.FEATURES)
AGE_EDGES = np.asarray(evaluation.AGE_EDGES, dtype=float)
PAIR_QUERIES = (("RACE", "HISPAN"), ("OCC", "IND"))
AGE_QUERIES = [("AGE",)] + [("AGE", x) for x in FEATURES[1:]] + [
    ("AGE", "SEX", "EDUCD"), ("AGE", "EDUCD", "OCC"),
    ("AGE", "OCC", "IND")]
MAX_PAIR_CELLS = 1_000_000
ID_COLUMNS = ["record_id", "arm", "fit_id", "target_id", "draw_id"]


class NonfiniteDataError(ValueError):
    def __init__(self, label, counts):
        self.counts = counts
        super().__init__(f"{label}: nonfinite values found; no rows were filtered: "
                         f"{json.dumps(counts, allow_nan=False)}")


def finite_matrix(values, label):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) == 0:
        raise ValueError(f"{label}: expected a nonempty encoded N-by-7 matrix")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise ValueError(f"{label}: expected real numeric values")
    bad = ~np.isfinite(values)
    counts = {"values": int(bad.sum()), "rows": int(bad.any(axis=1).sum()),
              "by_feature": {name: int(bad[:, i].sum())
                             for i, name in enumerate(FEATURES)}}
    if counts["values"]:
        raise NonfiniteDataError(label, counts)
    return values, counts


def decoded_ages(values, real_records=False):
    with np.errstate(over="ignore", invalid="ignore"):
        ages = 50.0 + 50.0 * np.asarray(values[:, 0], dtype=np.float64)
    if not np.isfinite(ages).all():
        raise NonfiniteDataError("Decoded AGE", {
            "values": int((~np.isfinite(ages)).sum()),
            "rows": int((~np.isfinite(ages)).sum()),
            "by_feature": {"AGE": int((~np.isfinite(ages)).sum())}})
    return np.rint(ages) if real_records else ages


def age_profile(values, *, real_records=False):
    values, nonfinite = finite_matrix(values, "AGE diagnostic input")
    raw = decoded_ages(values)
    ages = np.rint(raw) if real_records else raw
    levels = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    result = {
        "rows": len(values), "nonfinite_counts": nonfinite,
        "real_record_rounding_for_evaluation": bool(real_records),
        "quantiles_years": {str(q): float(v) for q, v in
                            zip(levels, np.quantile(ages, levels))},
        "outside_0_100_count": int(np.sum((ages < 0) | (ages > 100))),
        "outside_0_100_fraction": float(np.mean((ages < 0) | (ages > 100))),
        "outside_18_65_count": int(np.sum((ages < 18) | (ages > 65))),
        "outside_18_65_fraction": float(np.mean((ages < 18) | (ages > 65))),
        "below_0_fraction": float(np.mean(ages < 0)),
        "above_100_fraction": float(np.mean(ages > 100)),
        "at_0_fraction": float(np.mean(ages == 0)),
        "at_100_fraction": float(np.mean(ages == 100)),
        "raw_inverse_transform_min": float(raw.min()),
        "raw_inverse_transform_max": float(raw.max()),
    }
    if real_records:
        raw_bins = np.searchsorted(AGE_EDGES, raw, side="right")
        real_bins = np.searchsorted(AGE_EDGES, ages, side="right")
        result["roundtrip"] = {
            "max_absolute_inverse_roundoff_years": float(np.max(np.abs(raw - ages))),
            "max_absolute_reencoding_difference": float(np.max(
                np.abs((ages - 50.0) / 50.0 - values[:, 0]))),
            "records_whose_bin_changes_when_integer_roundoff_is_corrected":
                int(np.sum(raw_bins != real_bins)),
            "observed_integer_age_count": int(len(np.unique(ages))),
            "note": "This checks the encoded input, not pre-clipping original ages.",
        }
    return result


def clamp_age(values):
    bounded = np.array(values, copy=True)
    bounded[:, 0] = np.clip(bounded[:, 0], -1.0, 1.0)
    return bounded


def age_bin_label(index):
    if index == 0:
        return f"AGE < {AGE_EDGES[0]:g}"
    if index == len(AGE_EDGES):
        return f"AGE >= {AGE_EDGES[-1]:g}"
    return f"[{AGE_EDGES[index-1]:g}, {AGE_EDGES[index]:g})"


def sparse_counts(codes, columns, domains):
    if math.prod(int(domains[c]) for c in columns) > np.iinfo(np.int64).max:
        raise ValueError("Diagnostic query exceeds supported int64 cell-key range")
    keys = np.zeros(len(codes), dtype=np.int64)
    for col in columns:
        keys = keys * int(domains[col]) + codes[:, col]
    return np.unique(keys, return_counts=True)


def align_counts(keys, counts, union):
    aligned = np.zeros(len(union), dtype=np.int64)
    aligned[np.searchsorted(union, keys)] = counts
    return aligned


def age_bin_audit(sample, reference, categories):
    """All AGE bins contribute, including empty bins and out-of-range bins.

    For joint queries, each bin contribution sums absolute joint-cell errors
    over every categorical combination. It is not the error of bin totals.
    """
    raw_codes, _ = evaluation.evaluation_codes(sample, categories)
    bounded_codes, _ = evaluation.evaluation_codes(clamp_age(sample), categories)
    ref_codes, _ = evaluation.evaluation_codes(reference, categories, real_records=True)
    domains = np.r_[len(AGE_EDGES) + 1, categories]
    rows, summaries = [], []
    for query in AGE_QUERIES:
        columns = tuple(FEATURES.index(x) for x in query)
        histograms = [sparse_counts(x, columns, domains)
                      for x in (raw_codes, bounded_codes, ref_codes)]
        union = np.unique(np.concatenate([x[0] for x in histograms]))
        raw_counts, bounded_counts, ref_counts = [
            align_counts(k, c, union) for k, c in histograms]
        raw_p = raw_counts / len(sample)
        bounded_p = bounded_counts / len(sample)
        ref_p = ref_counts / len(reference)
        raw_error, bounded_error = np.abs(raw_p - ref_p), np.abs(bounded_p - ref_p)
        stride = math.prod(int(domains[c]) for c in columns[1:])
        bin_ids = union // stride
        raw_l1, bounded_l1 = float(raw_error.sum()), float(bounded_error.sum())
        expected = evaluation.marginal_l1(raw_codes, ref_codes, columns, domains)
        if not np.isclose(raw_l1, expected, rtol=0, atol=1e-12):
            raise RuntimeError("AGE contributions do not reproduce the common evaluator")
        label = " x ".join(query)
        summaries.append({"query": label, "raw_l1": raw_l1,
                          "bounded_l1": bounded_l1,
                          "bounded_minus_raw_l1": bounded_l1 - raw_l1})
        for bin_id in range(len(AGE_EDGES) + 1):
            mask = bin_ids == bin_id
            rows.append({"query": label, "age_bin": bin_id,
                         "age_bin_label": age_bin_label(bin_id),
                         "reference_mass": float(ref_p[mask].sum()),
                         "raw_mass": float(raw_p[mask].sum()),
                         "bounded_mass": float(bounded_p[mask].sum()),
                         "raw_l1_contribution": float(raw_error[mask].sum()),
                         "bounded_l1_contribution": float(bounded_error[mask].sum()),
                         "bounded_minus_raw_contribution": float(
                             bounded_error[mask].sum() - raw_error[mask].sum())})
    return summaries, rows


def pair_tables(sample_codes, reference_codes, query, domains):
    x, y = (FEATURES.index(n) for n in query)
    nx, ny = int(domains[x]), int(domains[y])
    if nx * ny > MAX_PAIR_CELLS:
        raise ValueError(f"{' x '.join(query)} exceeds {MAX_PAIR_CELLS} public cells")
    def counts(codes):
        return np.bincount(codes[:, x] * ny + codes[:, y],
                           minlength=nx * ny).reshape(nx, ny)
    sc, rc = counts(sample_codes), counts(reference_codes)
    sp, rp = sc / len(sample_codes), rc / len(reference_codes)
    sm, rm = np.outer(sp.sum(1), sp.sum(0)), np.outer(rp.sum(1), rp.sum(0))
    sr, rr = sp - sm, rp - rm
    joint_difference, product_difference = sp - rp, sm - rm
    residual_difference = sr - rr
    identity = joint_difference - product_difference - residual_difference
    if not np.allclose(identity, 0, rtol=0, atol=1e-12):
        raise RuntimeError("Joint/product/residual cell identity failed")
    return {"query": " x ".join(query), "names": query, "shape": (nx, ny),
            "sample_counts": sc, "reference_counts": rc,
            "sample_probability": sp, "reference_probability": rp,
            "joint_difference": joint_difference,
            "product_difference": product_difference,
            "residual_difference": residual_difference,
            "sample_residual": sr, "reference_residual": rr,
            "identity_max_error": float(np.abs(identity).max())}


def cell_selection(tables, additional=None):
    # The small RACE/HISPAN table is complete. Large tables emit their occupied
    # union; all aggregate/decomposition totals still use ALL public cells.
    if tables["sample_counts"].size <= 1000:
        return np.ones(tables["shape"], dtype=bool)
    mask = (tables["sample_counts"] > 0) | (tables["reference_counts"] > 0)
    if additional is not None:
        mask |= additional["sample_counts"] > 0
    return mask


def pair_summary(tables, selection=None):
    sc, rc = tables["sample_counts"], tables["reference_counts"]
    selected = cell_selection(tables) if selection is None else selection
    total = sc.size
    return {
        "query": tables["query"], "public_cells": int(total),
        "sample_rows": int(sc.sum()), "reference_rows": int(rc.sum()),
        "sample_occupied_cells": int(np.count_nonzero(sc)),
        "reference_occupied_cells": int(np.count_nonzero(rc)),
        "shared_occupied_cells": int(np.sum((sc > 0) & (rc > 0))),
        "sample_occupancy_fraction": float(np.count_nonzero(sc) / total),
        "reference_occupancy_fraction": float(np.count_nonzero(rc) / total),
        "overlap_probability_mass": float(np.minimum(
            tables["sample_probability"], tables["reference_probability"]).sum()),
        "joint_l1": float(np.abs(tables["joint_difference"]).sum()),
        "product_marginal_difference_l1": float(np.abs(tables["product_difference"]).sum()),
        "interaction_residual_l1": float(np.abs(tables["residual_difference"]).sum()),
        "sample_residual_magnitude_l1": float(np.abs(tables["sample_residual"]).sum()),
        "reference_residual_magnitude_l1": float(np.abs(tables["reference_residual"]).sum()),
        "cell_identity_max_error": tables["identity_max_error"],
        "csv_cells_emitted": int(selected.sum()),
        "csv_cells_omitted": int((~selected).sum()),
        "omitted_joint_l1_contribution": float(np.abs(tables["joint_difference"])[~selected].sum()),
        "omitted_product_l1_contribution": float(np.abs(tables["product_difference"])[~selected].sum()),
        "omitted_interaction_l1_contribution": float(np.abs(tables["residual_difference"])[~selected].sum()),
    }


def pair_cell_rows(tables, schema):
    selected = cell_selection(tables)
    domain = schema["category_domains"]
    name_x, name_y = tables["names"]
    for x, y in zip(*np.nonzero(selected)):
        yield {"query": tables["query"], "x_index": int(x), "y_index": int(y),
               "x_public_code": domain[name_x][int(x)], "y_public_code": domain[name_y][int(y)],
               "reference_count": int(tables["reference_counts"][x, y]),
               "sample_count": int(tables["sample_counts"][x, y]),
               "reference_probability": float(tables["reference_probability"][x, y]),
               "sample_probability": float(tables["sample_probability"][x, y]),
               "signed_joint_difference": float(tables["joint_difference"][x, y]),
               "absolute_joint_error": float(abs(tables["joint_difference"][x, y])),
               "product_marginal_difference": float(tables["product_difference"][x, y]),
               "interaction_residual_difference": float(tables["residual_difference"][x, y]),
               "absolute_interaction_error": float(abs(tables["residual_difference"][x, y]))}


def compare_pair_tables(baseline, candidate):
    if not np.array_equal(baseline["reference_counts"], candidate["reference_counts"]):
        raise ValueError("Matched cell comparisons require the same reference")
    selected = cell_selection(candidate, baseline)
    joint_change = (np.abs(candidate["joint_difference"]) -
                    np.abs(baseline["joint_difference"]))
    residual_change = (np.abs(candidate["residual_difference"]) -
                       np.abs(baseline["residual_difference"]))
    summary = {
        "query": candidate["query"],
        "C_minus_baseline_joint_l1": float(joint_change.sum()),
        "C_minus_baseline_interaction_l1": float(residual_change.sum()),
        "emitted_cell_joint_error_change_sum": float(joint_change[selected].sum()),
        "omitted_cell_joint_error_change_sum": float(joint_change[~selected].sum()),
        "omitted_cell_interaction_error_change_sum": float(residual_change[~selected].sum()),
        "csv_cells_emitted": int(selected.sum()),
        "sign_convention": "Positive error change means C is worse.",
    }
    rows = []
    for x, y in zip(*np.nonzero(selected)):
        rows.append({"query": candidate["query"], "x_index": int(x), "y_index": int(y),
                     "reference_count": int(candidate["reference_counts"][x, y]),
                     "baseline_count": int(baseline["sample_counts"][x, y]),
                     "C_count": int(candidate["sample_counts"][x, y]),
                     "C_minus_baseline_probability": float(candidate["sample_probability"][x, y] - baseline["sample_probability"][x, y]),
                     "baseline_absolute_joint_error": float(abs(baseline["joint_difference"][x, y])),
                     "C_absolute_joint_error": float(abs(candidate["joint_difference"][x, y])),
                     "C_minus_baseline_absolute_joint_error": float(joint_change[x, y]),
                     "C_minus_baseline_absolute_interaction_error": float(residual_change[x, y])})
    return summary, rows


def draw_disjoint_indices(pool_rows, sample_rows, rng):
    if sample_rows < 1 or pool_rows < 2 * sample_rows:
        raise ValueError("Real-real controls require at least 2 * sample_rows training rows; "
                         "no bootstrap or replacement fallback is allowed")
    chosen = rng.choice(pool_rows, size=2 * sample_rows, replace=False)
    return chosen[:sample_rows], chosen[sample_rows:]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_arm(arm):
    return {"B_private": "B", "B": "B", "E_matched": "E", "E": "E", "C": "C"}.get(arm, arm)


def identifier_token(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def pairing_key(record):
    return tuple(identifier_token(record[x]) for x in ("fit_id", "target_id", "draw_id"))


def csv_ids(record):
    return {"record_id": record["record_id"], "arm": record["arm"],
            **{x: identifier_token(record[x]) for x in ("fit_id", "target_id", "draw_id")}}


def load_manifest(path, categories):
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        payload = payload["samples"]
    if not isinstance(payload, list):
        raise ValueError("Samples manifest must be an explicit JSON list or an object with a samples list")
    result, seen = [], set()
    required = {"arm", "fit_id", "target_id", "draw_id", "path"}
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict) or not required <= entry.keys():
            raise ValueError(f"Manifest entry {index} requires {sorted(required)}")
        if not isinstance(entry["arm"], str) or not entry["arm"].strip():
            raise ValueError(f"Manifest entry {index} needs a nonempty arm string")
        for field in ("fit_id", "target_id", "draw_id"):
            value = entry[field]
            if isinstance(value, bool) or (value is not None and not isinstance(value, (str, int))):
                raise ValueError(f"Manifest {field} must be a string, integer, or explicit null target")
            if field != "target_id" and value is None:
                raise ValueError(f"Manifest {field} may not be null")
            if isinstance(value, str) and not value:
                raise ValueError(f"Manifest {field} may not be empty")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise ValueError(f"Manifest entry {index} needs an explicit path")
        entry = {**entry, "key": entry.get("key")}
        source = (path.parent / entry["path"]).resolve()
        if source.suffix.lower() == ".npy":
            if entry["key"] is not None:
                raise ValueError("For a .npy sample, key must explicitly be null")
            values = np.load(source, allow_pickle=False)
        elif source.suffix.lower() == ".npz":
            if not isinstance(entry["key"], str) or not entry["key"]:
                raise ValueError("For a .npz sample, provide an explicit array key")
            with np.load(source, allow_pickle=False) as archive:
                values = np.array(archive[entry["key"]], copy=True)
        else:
            raise ValueError("Manifest sample files must be .npy or .npz")
        values, nonfinite = finite_matrix(values, f"Manifest entry {index}")
        decoded_ages(values)  # Detect inverse-transform overflow before binning.
        evaluation.evaluation_codes(values, categories)
        record = {**{x: entry[x] for x in required if x != "path"}, "key": entry["key"],
                  "record_id": f"sample_{index:04d}", "path": str(source),
                  "sha256": sha256(source), "rows": len(values),
                  "nonfinite_counts": nonfinite, "values": values}
        unique = (canonical_arm(record["arm"]), *pairing_key(record))
        if unique in seen:
            raise ValueError("Duplicate canonical arm/fit/target/draw identifiers in manifest")
        seen.add(unique)
        result.append(record)
    return result


def match_records(records):
    lookup = {}
    for record in records:
        key = (canonical_arm(record["arm"]), identifier_token(record["fit_id"]),
               identifier_token(record["draw_id"]))
        lookup.setdefault(key, []).append(record)
    matched, unmatched = [], []
    for candidate in records:
        if canonical_arm(candidate["arm"]) != "C":
            continue
        for arm in ("B", "E"):
            choices = lookup.get((arm, identifier_token(candidate["fit_id"]),
                                  identifier_token(candidate["draw_id"])), [])
            baseline = choices[0] if len(choices) == 1 else None
            if baseline is None or baseline["rows"] != candidate["rows"]:
                reason = ("no exact fit/draw control" if not choices else
                          "ambiguous controls for fit/draw" if len(choices) > 1 else
                          "row counts differ")
                unmatched.append({"C_record_id": candidate["record_id"],
                                  "baseline_arm": arm,
                                  "reason": reason,
                                  "candidate_control_record_ids": [r["record_id"] for r in choices],
                                  "C_identifiers": {x: candidate[x] for x in ("fit_id", "target_id", "draw_id")}})
            else:
                matched.append((baseline, candidate))
    return matched, unmatched


def new_csv(stack, path, fields):
    handle = stack.enter_context(path.open("x", newline="", encoding="utf-8"))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return writer


def source_provenance(args):
    paths = {"data_npz": args.data_npz, "schema": args.schema,
             "diagnostic_runner": Path(__file__),
             "common_evaluator": Path(evaluation.__file__),
             "relationship_evaluator": Path(__file__).resolve().parent / "prototype2_relationship_checks.py"}
    if args.samples_manifest is not None:
        paths["samples_manifest"] = args.samples_manifest
    files = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
             for name, path in paths.items()}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).resolve().parent,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"files": files, "git_head": revision, "python": platform.python_version(),
            "numpy": np.__version__, "utc_started": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed}


def run_into(args):
    provenance = source_provenance(args)
    # Count failures before load_pilot validates the remaining input contract.
    with np.load(args.data_npz, allow_pickle=False) as archive:
        for key in ("X_train", "X_test"):
            finite_matrix(archive[key], key)
    train, holdout, categories, schema = evaluation.load_pilot(args.data_npz, args.schema)
    if args.real_real_repeats and len(train) < 2 * args.sample_rows:
        raise ValueError("Too few training rows for disjoint real-real samples; "
                         "reduce --sample-rows or explicitly set --real-real-repeats 0")
    records = load_manifest(args.samples_manifest, categories)
    domains = np.r_[len(AGE_EDGES) + 1, categories]
    train_codes, _ = evaluation.evaluation_codes(train, categories, real_records=True)
    holdout_codes, _ = evaluation.evaluation_codes(holdout, categories, real_records=True)
    report = {
        "format_version": 1, "status": "complete", "nonprivate_research": True,
        "declared_public_fixture": schema.get("public_fixture") is True,
        "declared_data_origin": schema.get("data_origin"),
        "training_performed": False, "new_marginal_releases": 0,
        "frozen_model_intervention_performed": False,
        "provenance": provenance, "features": FEATURES,
        "encoding": "AGE=(years-50)/50; remaining columns are categorical indices",
        "age_edges": AGE_EDGES.tolist(),
        "age_bin_note": "Legacy bin [85,101) includes invalid generated AGE in (100,101). Strict tails are reported separately.",
        "category_cardinalities": np.asarray(categories).tolist(),
        "reference_age": {"train": age_profile(train, real_records=True),
                          "holdout": age_profile(holdout, real_records=True)},
        "reference_category_support": {
            name: {"declared": int(categories[i-1]),
                   "observed_train": int(len(np.unique(train_codes[:, i]))),
                   "observed_holdout": int(len(np.unique(holdout_codes[:, i])))}
            for i, name in enumerate(FEATURES) if i},
        "reference_pair_support": [pair_summary(pair_tables(train_codes, holdout_codes, q, domains))
                                   for q in PAIR_QUERIES],
        "samples": [], "matched_comparisons": [], "unmatched_comparisons": [],
        "real_real": {"pool": "X_train only", "repeats": args.real_real_repeats,
                      "rows_per_side": args.sample_rows, "within_repeat_disjoint": True,
                      "across_repeat_independent": False, "results": []},
        "interpretation": [
            "No raw records are copied to these reports. Counts and exact reference metrics remain nonprivate.",
            "Real-real repetitions are dependent internal-pool diagnostics, not independent training fits or an attainable error floor.",
            "Generation/target identifiers are supplied metadata, not independently verified checkpoint provenance.",
            "No rows are filtered, no generated ages are rounded, and no model is trained or re-sampled here.",
            "The clamp modifies AGE only. Ages18–65 and declared bounds0–100 are separate support questions.",
            "All cell totals use full public domains. Large-table CSVs omit jointly unoccupied cells and report omitted decomposition contributions.",
            "Marginal and interaction errors are distinct; cell decompositions describe distributions and do not establish causality.",
        ],
    }
    code_cache = {}
    with ExitStack() as stack:
        metrics_writer = new_csv(stack, args.out / "metrics_per_draw.csv", ID_COLUMNS + ["rows", "reference_rows", "metric", "value", "shuffled_mean", "shuffle_minus_original"])
        age_writer = new_csv(stack, args.out / "age_bin_contributions.csv", ID_COLUMNS + ["query", "age_bin", "age_bin_label", "reference_mass", "raw_mass", "bounded_mass", "raw_l1_contribution", "bounded_l1_contribution", "bounded_minus_raw_contribution"])
        cells_writer = new_csv(stack, args.out / "pair_cell_errors.csv", ID_COLUMNS + ["query", "x_index", "y_index", "x_public_code", "y_public_code", "reference_count", "sample_count", "reference_probability", "sample_probability", "signed_joint_difference", "absolute_joint_error", "product_marginal_difference", "interaction_residual_difference", "absolute_interaction_error"])
        comparison_writer = new_csv(stack, args.out / "matched_cell_changes.csv", ["baseline_record_id", "C_record_id", "baseline_arm", "fit_id", "target_id", "draw_id", "query", "x_index", "y_index", "reference_count", "baseline_count", "C_count", "C_minus_baseline_probability", "baseline_absolute_joint_error", "C_absolute_joint_error", "C_minus_baseline_absolute_joint_error", "C_minus_baseline_absolute_interaction_error"])
        real_writer = new_csv(stack, args.out / "real_real_controls.csv", ["repeat_id", "rows_per_side", "metric", "original", "shuffled_mean", "shuffle_minus_original"])
        for index, record in enumerate(records):
            values = record["values"]
            common = evaluation.evaluate_sample(values, holdout, categories,
                                                shuffle_repeats=args.shuffle_repeats,
                                                seed=args.seed + 1000 + index)
            for metric, value in evaluation.flatten_metrics(common).items():
                shuffle_mean = common["shuffle"]["mean"]
                shuffle_delta = common["shuffle"]["shuffle_minus_original"]
                metrics_writer.writerow({**csv_ids(record), "rows": len(values),
                                         "reference_rows": len(holdout), "metric": metric, "value": value,
                                         "shuffled_mean": None if shuffle_mean is None else shuffle_mean[metric],
                                         "shuffle_minus_original": None if shuffle_delta is None else shuffle_delta[metric]})
            age_summaries, age_rows = age_bin_audit(values, holdout, categories)
            for row in age_rows:
                age_writer.writerow({**csv_ids(record), **row})
            codes, _ = evaluation.evaluation_codes(values, categories)
            pair_data = [pair_tables(codes, holdout_codes, q, domains) for q in PAIR_QUERIES]
            code_cache[record["record_id"]] = codes
            for table in pair_data:
                for row in pair_cell_rows(table, schema):
                    cells_writer.writerow({**csv_ids(record), **row})
            report["samples"].append({
                **{k: v for k, v in record.items() if k != "values"},
                "age_raw": age_profile(values), "age_bounded": age_profile(clamp_age(values)),
                "age_queries": age_summaries, "pair_statistics": [pair_summary(t) for t in pair_data],
                "metrics": common})
        matched, unmatched = match_records(records)
        report["unmatched_comparisons"] = unmatched
        control_use_counts = {}
        for baseline, _ in matched:
            key = baseline["record_id"]
            control_use_counts[key] = control_use_counts.get(key, 0) + 1
        for baseline, candidate in matched:
            comparisons = []
            # Keep only small row-code matrices between draws. Rebuilding two
            # bounded tables avoids retaining ~10 dense arrays per draw.
            for query in PAIR_QUERIES:
                bt = pair_tables(code_cache[baseline["record_id"]], holdout_codes, query, domains)
                ct = pair_tables(code_cache[candidate["record_id"]], holdout_codes, query, domains)
                summary, rows = compare_pair_tables(bt, ct)
                comparisons.append(summary)
                for row in rows:
                    comparison_writer.writerow({"baseline_record_id": baseline["record_id"],
                                                "C_record_id": candidate["record_id"],
                                                "baseline_arm": baseline["arm"],
                                                **{x: identifier_token(candidate[x]) for x in ("fit_id", "target_id", "draw_id")}, **row})
            report["matched_comparisons"].append({"baseline_record_id": baseline["record_id"],
                                                  "C_record_id": candidate["record_id"],
                                                  "C_target_id": candidate["target_id"],
                                                  "baseline_declared_target_id": baseline["target_id"],
                                                  "pairing": "unique B/E control with exact manifest fit_id and draw_id",
                                                  "control_comparison_count": control_use_counts[baseline["record_id"]],
                                                  "shared_control": control_use_counts[baseline["record_id"]] > 1,
                                                  "independence_note": "Target variants sharing a control are dependent comparisons, not additional training fits.",
                                                  "queries": comparisons})
        rng = np.random.default_rng(args.seed)
        for repeat in range(args.real_real_repeats):
            left, right = draw_disjoint_indices(len(train), args.sample_rows, rng)
            controls = evaluation.evaluate_sample(train[left], train[right], categories,
                                                  shuffle_repeats=args.shuffle_repeats,
                                                  seed=args.seed + 100000 + repeat,
                                                  sample_real_records=True)
            flat = evaluation.flatten_metrics(controls)
            shuffled = controls["shuffle"]["mean"]
            delta = controls["shuffle"]["shuffle_minus_original"]
            for metric, value in flat.items():
                real_writer.writerow({"repeat_id": repeat, "rows_per_side": args.sample_rows,
                                      "metric": metric, "original": value,
                                      "shuffled_mean": None if shuffled is None else shuffled[metric],
                                      "shuffle_minus_original": None if delta is None else delta[metric]})
            report["real_real"]["results"].append({"repeat_id": repeat, "metrics": controls})
    reused = {}
    for record in records:
        token = (record["path"], record["key"])
        reused.setdefault(token, []).append(record["record_id"])
    report["explicit_source_reuse"] = [v for v in reused.values() if len(v) > 1]
    (args.out / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "README.txt").write_text(
        ("GENERATED PUBLIC FIXTURE: SOFTWARE VALIDATION ONLY, NOT RESEARCH RESULTS\n"
         if schema.get("public_fixture") is True else "NONPRIVATE DIAGNOSTIC REPORT\n") +
        "No training, new target release, or frozen-model intervention occurred.\n"
        "diagnostics.json contains provenance, AGE/support audits, original/clamped query errors, and exact paired-cell totals.\n"
        "metrics_per_draw.csv retains explicit manifest fit/target/draw identifiers. No generation-as-training inference is performed.\n"
        "age_bin_contributions.csv includes every legacy AGE bin for each fixed AGE query.\n"
        "pair_cell_errors.csv is complete for small domains and uses occupied unions for large domains; JSON totals include all cells.\n"
        "matched_cell_changes.csv pairs C with a unique B/E control for identical explicit fit/draw IDs and row counts. C target IDs remain explicit; shared controls create dependent comparisons. Positive error change means C is worse.\n"
        "real_real_controls.csv uses disjoint samples within each repetition from X_train. Repetitions can overlap and are not independent experiments or an attainable floor.\n"
        "No raw records were copied, but aggregate counts and exact real-reference metrics are not a private release.\n", encoding="utf-8")
    return report


def run(args):
    if args.seed < 0 or args.real_real_repeats < 0 or args.shuffle_repeats < 0 or args.sample_rows < 1:
        raise ValueError("Seed/repeat counts must be nonnegative and sample rows positive")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        return run_into(args)
    except Exception as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                   "no_rows_filtered": True, "training_performed": False,
                   "partial_files_must_not_be_interpreted_as_a_complete_report": True}
        if isinstance(exc, NonfiniteDataError):
            failure["nonfinite_counts"] = exc.counts
        (args.out / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--samples-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; existing directories are rejected")
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--real-real-repeats", type=int, default=20)
    parser.add_argument("--sample-rows", type=int, default=512)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Diagnostics failed: {exc}", file=sys.stderr)
        return 2
    print(f"Completed diagnostics for {len(report['samples'])} supplied samples and "
          f"{args.real_real_repeats} dependent real-real controls: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
