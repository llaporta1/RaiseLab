#!/usr/bin/env python3
"""Replay the historical frozen B/C sampler, then vary categorical guidance.

This is a local, nonprivate research diagnostic, not a new training run or an
end-to-end private release. The historical CUDA backend is imported only after
source and artifact verification. --check-files performs no numerical imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from prototype2_gpu_replay_backend import (HistoricalGPUBackend,
                                          verify_context, verify_source_group)

FEATURES = ("AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND")
CATEGORIES = (3, 9, 6, 44, 531, 266)
GENERATION_SEEDS = (12026, 12126, 12226)
REFERENCE_SEED = 13026
REFERENCE_ROWS = 4096
SAMPLE_ROWS = 512
SAMPLE_BATCH = 64
SAMPLING_STEPS = 50
FIT_ID = "gpu22692665_B_private"
MASKS = {"all": FEATURES[1:], "race_hispan": ("RACE", "HISPAN"),
         "occ_ind": ("OCC", "IND")}


class ReplayMismatch(ValueError):
    """A historical or zero-guidance replay gate failed; do not run the grid."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, payload, *, replace=False):
    path = Path(path)
    if replace:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    import numpy as np
    value = np.asarray(array)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode("ascii"))
    h.update(json.dumps(list(value.shape)).encode("ascii"))
    h.update(value.tobytes(order="C"))
    return h.hexdigest()


def vector_hash(vectors):
    payload = [array_hash(value) for value in vectors]
    return hashlib.sha256(json.dumps(payload).encode("ascii")).hexdigest()


def build_conditions(include_train_oracle=False):
    """Ten unique cached-target conditions; nineteen with the optional oracle."""
    conditions = [{"id": "zero", "target_kind": None, "strength": 0.0,
                   "mask": "none", "active_features": []}]
    kinds = ["cached", "train_oracle"] if include_train_oracle else ["cached"]
    for kind in kinds:
        for mask, active in MASKS.items():
            for strength, suffix in ((0.25, "025"), (0.5, "050"), (1.0, "100")):
                conditions.append({"id": f"{kind}_{mask}_s{suffix}",
                                   "target_kind": kind, "strength": strength,
                                   "mask": mask, "active_features": list(active)})
    return conditions


def _probabilities(vectors, label):
    import numpy as np
    _require(len(vectors) == len(CATEGORIES), f"{label}: expected six vectors")
    result = []
    for feature, count, vector in zip(FEATURES[1:], CATEGORIES, vectors):
        value = np.asarray(vector, dtype=np.float64)
        _require(value.shape == (count,) and np.isfinite(value).all()
                 and np.all(value > 0)
                 and np.isclose(value.sum(), 1.0, rtol=0, atol=1e-10),
                 f"{label}: invalid normalized vector for {feature}")
        result.append(value.copy())
    return result


def make_condition_offsets(condition, targets, reference, make_offsets_fn):
    """Mask the historical strength-times-clipped-log-ratio offset vectors."""
    import numpy as np
    targets = _probabilities(targets, "target")
    reference = _probabilities(reference, "reference")
    active = condition["active_features"]
    strength = condition["strength"]
    _require(isinstance(active, (list, tuple)) and len(set(active)) == len(active)
             and set(active) <= set(FEATURES[1:]), "Invalid active-feature mask")
    _require(type(strength) in (float, int) and math.isfinite(strength)
             and 0 <= strength <= 1, "Strength must be finite and in [0, 1]")
    raw = make_offsets_fn(targets, reference, strength=strength,
                          max_log_ratio=2.0, prob_floor=1e-8)
    _require(len(raw) == 6, "Historical offset constructor did not return six vectors")
    result = []
    for feature, count, value in zip(FEATURES[1:], CATEGORIES, raw):
        value = np.asarray(value, dtype=np.float64)
        _require(value.shape == (count,) and np.isfinite(value).all(),
                 f"Invalid constructed offset for {feature}")
        result.append(value.copy() if feature in active and strength != 0
                      else np.zeros(count, dtype=np.float64))
    return result


def validate_sample(sample, rows, label):
    import numpy as np
    sample = np.asarray(sample)
    _require(sample.shape == (rows, 7) and sample.dtype.kind in "iuf"
             and np.isfinite(sample).all(), f"{label}: expected a finite {rows}-by-7 array")
    for index, count in enumerate(CATEGORIES, 1):
        values = sample[:, index]
        _require(np.all(values == np.floor(values)) and np.all(values >= 0)
                 and np.all(values < count), f"{label}: invalid categorical codes")
    # AGE remains in its original encoded form, including invalid-age tails.
    return sample


def _exact(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C"))


def _comparison(actual, expected):
    import numpy as np
    same_shape = actual.shape == expected.shape
    return {"exact_array_bytes": _exact(actual, expected),
            "actual_array_sha256": array_hash(actual),
            "expected_array_sha256": array_hash(expected),
            "dtype_equal": actual.dtype == expected.dtype,
            "shape_equal": same_shape,
            "numerically_different_values": int(np.count_nonzero(actual != expected))
                if same_shape else None,
            "max_absolute_difference": float(np.max(np.abs(actual.astype(np.float64)
                - expected.astype(np.float64)))) if same_shape else None}


def _hooks(hooks, offsets):
    import numpy as np
    _require(isinstance(hooks, dict) and hooks.get("hook_calls") == 400,
             "Expected 400 guidance hook calls for 512 rows / batch 64 / 50 steps")
    adjusted = hooks.get("masked_cells_adjusted")
    _require(type(adjusted) is int and adjusted >= 0, "Invalid guidance hook counter")
    active = any(np.any(value != 0) for value in offsets)
    _require((adjusted > 0) if active else (adjusted == 0),
             "Guidance hook activity differs from the declared offsets")


def _target_distances(sample, target):
    import numpy as np
    return {name: float(np.abs(np.bincount(sample[:, index].astype(np.int64),
                          minlength=count).astype(np.float64) / len(sample)
                          - target[index - 1]).sum())
            for index, (name, count) in enumerate(zip(FEATURES[1:], CATEGORIES), 1)}


def run_sampling(backend, historical_samples, cached_targets, out, *, mode="sweep",
                 oracle_targets=None):
    """Gate all historical draws before saving any condition or running the grid.

    The backend owns exact historical CUDA generation. This orchestration can
    also be tested with public deterministic fixtures without asserting CUDA
    compatibility or historical checkpoint replay.
    """
    import numpy as np
    out = Path(out)
    _require(mode in ("replay", "sweep"), "Unknown mode")
    _require(mode == "sweep" or oracle_targets is None, "Oracle requires sweep mode")
    cached_targets = _probabilities(cached_targets, "cached target")
    if oracle_targets is not None:
        oracle_targets = _probabilities(oracle_targets, "TRAIN oracle")
    expected_keys = {(arm, draw) for arm in ("B_private", "C") for draw in (1, 2, 3)}
    _require(set(historical_samples) == expected_keys, "Need all three historical B and C draws")
    historical_samples = {key: validate_sample(value, SAMPLE_ROWS, str(key))
                          for key, value in historical_samples.items()}
    frozen_model = backend.fingerprint()
    conditions = build_conditions(oracle_targets is not None)
    if mode == "replay":
        conditions = [item for item in conditions if item["id"] in ("zero", "cached_all_s100")]
    by_id = {item["id"]: item for item in conditions}
    replay = {"status": "running", "reference_rows": REFERENCE_ROWS,
              "reference_seed": REFERENCE_SEED, "sample_batch": SAMPLE_BATCH,
              "comparison": "Exact dtype, shape and array bytes; not fresh NPZ bytes",
              "draws": []}
    _json(out / "replay.json", replay)

    def sample(seed, count, offsets=None):
        value, hooks, consumed = backend.sample(seed, count, offsets)
        value = validate_sample(value, count, "Generated sample")
        _require(backend.fingerprint() == frozen_model, "Sampling changed frozen model weights")
        return value, hooks, consumed

    try:
        print("Reconstructing original B reference: 4096 rows, seed 13026, batch 64", flush=True)
        reference_array, _, _ = sample(REFERENCE_SEED, REFERENCE_ROWS)
        reference = _probabilities(backend.normalize_reference(reference_array), "B calibration")
        reference_digest = vector_hash(reference)
        replay["reference_array_sha256"] = array_hash(reference_array)
        replay["reference_sha256"] = reference_digest
        _json(out / "reference.json", {"rows": REFERENCE_ROWS, "seed": REFERENCE_SEED,
              "sample_batch": SAMPLE_BATCH, "count_pseudocount": 0.001,
              "reference_sha256": reference_digest,
              "probabilities": {name: value.tolist()
                  for name, value in zip(FEATURES[1:], reference)}})
        del reference_array
        zero_offsets = make_condition_offsets(by_id["zero"], cached_targets,
                                               reference, backend.make_offsets)
        full_offsets = make_condition_offsets(by_id["cached_all_s100"], cached_targets,
                                               reference, backend.make_offsets)
        # Historical C had nonzero offsets. Never redraw a target to obtain them.
        _require(any(np.any(value != 0) for value in full_offsets),
                 "Historical all-feature target has no active offset")
        retained = {}
        for draw, seed in enumerate(GENERATION_SEEDS, 1):
            print(f"Historical replay draw {draw}/3: B, zero guidance, and C", flush=True)
            ordinary, _, ordinary_rng = sample(seed, SAMPLE_ROWS)
            zero, zero_hooks, zero_rng = sample(seed, SAMPLE_ROWS, zero_offsets)
            c_sample, c_hooks, _ = sample(seed, SAMPLE_ROWS, full_offsets)
            _hooks(zero_hooks, zero_offsets)
            _hooks(c_hooks, full_offsets)
            checks = {"draw_id": draw, "generation_seed": seed,
                      "b_matches_historical": _exact(ordinary, historical_samples[("B_private", draw)]),
                      "zero_matches_b": _exact(zero, ordinary),
                      "zero_rng_matches_b": bool(backend.same_rng(ordinary_rng, zero_rng)),
                      "c_matches_historical": _exact(c_sample, historical_samples[("C", draw)]),
                      "B": _comparison(ordinary, historical_samples[("B_private", draw)]),
                      "C": _comparison(c_sample, historical_samples[("C", draw)]),
                      "zero_hooks": zero_hooks, "c_hooks": c_hooks}
            replay["draws"].append(checks)
            _json(out / "replay.json", replay, replace=True)
            if not all(checks[key] for key in ("b_matches_historical", "zero_matches_b",
                                               "zero_rng_matches_b", "c_matches_historical")):
                raise ReplayMismatch(f"Historical replay failed for draw {draw}; no guidance grid executed")
            retained[("zero", draw)] = (zero, zero_hooks)
            retained[("cached_all_s100", draw)] = (c_sample, c_hooks)
        replay["status"] = "passed"
        _json(out / "replay.json", replay, replace=True)
    except Exception as error:
        replay["status"] = "failed"
        replay["error_type"] = type(error).__name__
        replay["error"] = str(error)
        _json(out / "replay.json", replay, replace=True)
        raise

    # No grid draws or condition files are allowed before every replay passes.
    sample_dir = out / "samples"
    sample_dir.mkdir()
    records = []
    for condition in conditions:
        cid = condition["id"]
        target = oracle_targets if condition["target_kind"] == "train_oracle" else cached_targets
        offsets = make_condition_offsets(condition, target, reference, backend.make_offsets)
        condition["offsets_sha256"] = vector_hash(offsets)
        for draw, seed in enumerate(GENERATION_SEEDS, 1):
            start = time.monotonic()
            reused = (cid, draw) in retained
            if reused:
                value, hooks = retained.pop((cid, draw))
            else:
                print(f"Sampling {cid} draw {draw}/3, public seed {seed}", flush=True)
                value, hooks, _ = sample(seed, SAMPLE_ROWS, offsets)
            _hooks(hooks, offsets)
            path = sample_dir / f"{cid}_generation_{draw:02d}.npz"
            backend.save_sample(path, value)
            target_id = ("train_oracle_NONPRIVATE" if condition["target_kind"] == "train_oracle"
                         else "cached_target01" if condition["target_kind"] == "cached" else None)
            record = {"arm": cid, "condition_id": cid, "fit_id": FIT_ID,
                      "target_id": target_id, "draw_id": draw, "generation_seed": seed,
                      "model_arm": "B_private", "path": str(path.resolve()),
                      "key": "X_synthetic", "n_rows": SAMPLE_ROWS,
                      "sha256": file_hash(path), "array_sha256": array_hash(value),
                      "target_kind": condition["target_kind"], "strength": condition["strength"],
                      "active_features": condition["active_features"], "hooks": hooks,
                      "reused_successful_replay": reused,
                      "sampling_and_save_seconds": None if reused else time.monotonic() - start,
                      "cached_target_l1": _target_distances(value, cached_targets)}
            if oracle_targets is not None:
                record["train_oracle_l1"] = _target_distances(value, oracle_targets)
            records.append(record)
            print(f"Retained {cid} draw {draw}/3", flush=True)
    _require(not retained, "Unexpected unused replay sample")
    _require(backend.fingerprint() == frozen_model, "Frozen model changed during the sweep")
    for record in records:
        _require(file_hash(record["path"]) == record["sha256"], "Saved sample changed during the sweep")
    scope = {"independent_training_replicates": 1, "new_training_fits": 0,
             "new_marginal_releases": 0, "cached_target_releases_reused": 1,
             "oracle_target_constructed": oracle_targets is not None,
             "generation_draws_per_condition": 3, "rows_per_draw": SAMPLE_ROWS,
             "fit_id_meaning": "All conditions reuse the same frozen B checkpoint; no independent training replication",
             "privacy": "NONPRIVATE local research; exact TRAIN oracle, if enabled, is not a DP condition",
             "interpretation": "Fixed-target sweep tests guidance response. Matched TRAIN-oracle contrasts test the conditional realized target perturbation including postprocessing, not raw noise or target-release variability."}
    _json(out / "sample_manifest.json", {"format_version": 1, "samples": records, "scope": scope})
    return {"status": "complete", "mode": mode, "conditions": conditions, "samples": records,
            "condition_count": len(conditions), "sample_count": len(records),
            "replay": replay, "reference_sha256": reference_digest,
            "model_sha256": frozen_model, "model_unchanged": True,
            "training_performed": False, "new_training_fits": 0, "new_marginal_releases": 0,
            "scope": scope, "generation_seeds": list(GENERATION_SEEDS)}


def _summary(report):
    lines = ["FROZEN-B GUIDANCE REPLAY / SWEEP", "NONPRIVATE LOCAL RESEARCH",
             f"Mode: {report['mode']}; historical B and C replay gates passed for all three draws.",
             f"Conditions: {report['condition_count']}; saved generation outputs: {report['sample_count']}.",
             "One historical B fit; 3 matched generation draws per condition; no between-fit SD.",
             "No new training or private target release. Original reference recipe: 4096/13026/64.",
             "Strengths and masks were fixed before sampling; every planned condition was retained.",
             "Use the separate common diagnostics for holdout fidelity and relationship scores.",
             "The optional TRAIN oracle is nonprivate and must not be pooled with DP-mechanism comparisons.", ""]
    for condition in report["conditions"]:
        values = [item for item in report["samples"] if item["condition_id"] == condition["id"]]
        lines.append(condition["id"] + ": mean per-feature L1 to retained cached target")
        lines.append("  " + "; ".join(f"{name}={sum(v['cached_target_l1'][name] for v in values)/len(values):.6f}"
                                     for name in FEATURES[1:]))
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("replay", "sweep"), default="replay")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--acknowledge-trusted-checkpoint", action="store_true")
    parser.add_argument("--include-train-oracle", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    created = False
    try:
        _require(not args.include_train_oracle or args.mode == "sweep", "Oracle is only allowed in sweep mode")
        context = verify_context(args.project_root, args.run_dir, args.data_npz, args.schema)
        if args.check_files:
            print("FILES VERIFIED: exact historical plan, source kits, saved artifacts and inputs. No numerical imports or checkpoint loading.")
            return 0
        _require(args.out is not None, "--out is required for compute modes")
        _require(args.acknowledge_trusted_checkpoint,
                 "Use --acknowledge-trusted-checkpoint only for the original owned checkpoint")
        args.out = args.out.expanduser().absolute()
        os.umask(0o077)
        args.out.mkdir(parents=True, exist_ok=False)
        created = True
        runner_sources = {str(path): file_hash(path) for path in
                          (Path(__file__).resolve(), Path(__file__).resolve().with_name(
                              "prototype2_gpu_replay_backend.py"))}
        _json(args.out / "preflight.json", context)
        conditions = build_conditions(args.include_train_oracle)
        if args.mode == "replay":
            conditions = [item for item in conditions if item["id"] in ("zero", "cached_all_s100")]
        _json(args.out / "plan.json", {"mode": args.mode, "conditions": conditions,
              "fixed_before_sampling": True, "generation_seeds": list(GENERATION_SEEDS),
              "reference_seed": REFERENCE_SEED, "reference_rows": REFERENCE_ROWS,
              "sample_rows": SAMPLE_ROWS, "sample_batch": SAMPLE_BATCH,
              "sampling_steps": SAMPLING_STEPS, "max_log_ratio": 2.0, "probability_floor": 1e-8,
              "include_train_oracle": args.include_train_oracle,
              "runner_source_sha256": runner_sources,
              "new_training_fits": 0, "new_marginal_releases": 0,
              "failure_policy": "Stop, retain partial artifacts, no grid before all historical B/C gates pass; no retries or redraws",
              "privacy": "NONPRIVATE local research. Optional exact TRAIN oracle is a diagnostic only."})
        backend = HistoricalGPUBackend(context, acknowledge_trusted_checkpoint=True,
                                       allow_oracle=args.include_train_oracle)
        historical = backend.load_historical_samples()
        oracle = backend.oracle_targets() if args.include_train_oracle else None
        report = run_sampling(backend, historical, backend.cached_targets, args.out,
                              mode=args.mode, oracle_targets=oracle)
        unchanged = backend.verify_unchanged()
        _require(all(file_hash(path) == digest for path, digest in runner_sources.items()),
                 "Replay/sweep implementation changed during this job")
        report.update({"inputs_unchanged": True, "input_checks": unchanged,
                       "runtime": backend.provenance, "runner_source_sha256": runner_sources})
        _json(args.out / "results.json", report)
        with (args.out / "summary.txt").open("x", encoding="utf-8") as handle:
            handle.write(_summary(report))
        artifact_names = ("preflight.json", "plan.json", "replay.json", "reference.json", "results.json",
                          "summary.txt", "sample_manifest.json")
        completion = {"status": "complete", "mode": args.mode, "replay_passed": True,
                      "condition_count": report["condition_count"], "sample_count": report["sample_count"],
                      "training_performed": False, "new_training_fits": 0, "new_marginal_releases": 0,
                      "inputs_unchanged": True, "model_unchanged": True,
                      "artifact_sha256": {name: file_hash(args.out / name) for name in artifact_names}}
        _json(args.out / "COMPLETED.json", completion)
        print(f"Completed {args.mode}: {report['sample_count']} retained outputs in {args.out}", flush=True)
        return 0
    except Exception as error:
        if created:
            _json(args.out / "FAILED.json", {"status": "failed", "error_type": type(error).__name__,
                  "error": str(error), "no_automatic_retry": True, "partial_outputs_retained": True})
        print(f"Guidance replay/sweep stopped: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
