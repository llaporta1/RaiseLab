#!/usr/bin/env python3
"""Run pinned official AIM + Private-PGM and MST + Private-PGM baselines.

Research evaluation, not an audited end-to-end DP release. Each fit receives
the complete supplied mechanism budget, with no additional C marginal target.
The fit interface never receives holdout data. Generation repeats reuse one
fitted model; inference SDs use only independent fit-level means.

Example after installing the pinned official checkout in an isolated venv:
  python scripts/run_ipums_pgm_baselines.py --data-npz DATA --schema SCHEMA \
    --mbi-repo REPO --expected-mbi-commit 5f97065a17b3e53cfefc7a8ce3e293fd153dd064

--public-fixture --smoke is software validation only. Reproducibility also
requires a fixed PYTHONHASHSEED (for example, PYTHONHASHSEED=0). It requires
an explicitly generated-fixture schema and never represents an IPUMS result.
"""

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototype2_evaluation import load_pilot, evaluate_sample, flatten_metrics
from prototype2_pgm_backend import (
    AGE_EDGES, ARM_LABELS, FEATURES, PINNED_MBI_COMMIT, OfficialPGMBackend,
    aim_workload, decode_synthetic_data, encode_training_data, public_seed, sha256_file,
)


DEFAULT_EPSILON = 2.5638977279554265
DEFAULT_DELTA = 1e-5


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--mbi-repo", type=Path, required=True)
    parser.add_argument("--expected-mbi-commit", required=True,
                        help=f"Exact expected checkout SHA (tested: {PINNED_MBI_COMMIT})")
    parser.add_argument("--arms", nargs="+", choices=list(ARM_LABELS), default=["aim", "mst"])
    parser.add_argument("--fits", type=int, default=3)
    parser.add_argument("--fit-ids", nargs="+", help="Public independent-fit IDs; count must equal --fits")
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--age-representation", choices=["integer", "legacy-bins"], default="integer")
    parser.add_argument("--seed", type=int, default=20261112,
                        help="Public generation/evaluation/fixture seed only; never seeds real-data private fitting")
    parser.add_argument("--aim-max-model-size", type=float,
                        help="MB cap (default80; --smoke default1)")
    parser.add_argument("--aim-max-iters", type=int, help="Inference iterations (default1000; --smoke default25)")
    parser.add_argument("--aim-max-cells", type=int,
                        help="Optional explicit pair resource exclusion, logged per query; default includes all21 pairs")
    parser.add_argument("--mst-selection-iters", type=int, help="Preselection inference iterations (default2500; smoke25)")
    parser.add_argument("--mst-final-iters", type=int, help="Final inference iterations (default10000; smoke50)")
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--public-fixture", action="store_true",
                        help="Require generated fixture schema and use public fit randomness; fix PYTHONHASHSEED for reproducibility")
    parser.add_argument("--smoke", action="store_true",
                        help="Small nonbenchmark inference settings; requires --public-fixture and fixture schema")
    parser.add_argument("--out", type=Path, help="NEW output directory; existing directories are never overwritten")
    return parser


def validate_args(args, schema):
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Each baseline arm must appear only once")
    if min(args.fits, args.draws, args.samples, args.shuffle_repeats) < 1:
        raise ValueError("Fits, draws, sample count, and shuffle repeats must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0 or not 0 < args.delta < 1:
        raise ValueError("Need finite epsilon>0 and 0<delta<1")
    if args.smoke and not args.public_fixture:
        raise ValueError("--smoke requires --public-fixture and cannot be used for a real-data comparison")
    if args.public_fixture and not (
        schema.get("public_fixture") is True
        and schema.get("data_origin") == "programmatically_generated_no_real_records"
    ):
        raise ValueError("--public-fixture requires a schema explicitly declaring programmatically generated data without real records")
    if schema.get("public_fixture") is True and not args.public_fixture:
        raise ValueError("This schema is a public fixture; pass --public-fixture so results cannot be mislabeled")
    defaults = (1.0, 25, 25, 50) if args.smoke else (80.0, 1000, 2500, 10000)
    for name, default in zip(("aim_max_model_size", "aim_max_iters", "mst_selection_iters", "mst_final_iters"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not math.isfinite(args.aim_max_model_size) or args.aim_max_model_size <= 0:
        raise ValueError("AIM maximum model size must be finite and positive")
    if min(args.aim_max_iters, args.mst_selection_iters, args.mst_final_iters) < 1:
        raise ValueError("Inference iteration counts must be positive")
    if args.aim_max_cells is not None and args.aim_max_cells < 1:
        raise ValueError("The explicit AIM cell cap must be positive")
    if args.fit_ids is None:
        args.fit_ids = [f"fit{i+1:03d}" for i in range(args.fits)]
    if len(args.fit_ids) != args.fits or len(set(args.fit_ids)) != args.fits:
        raise ValueError("Public fit IDs must be unique and their count must equal --fits")
    if any(not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value)
           for value in args.fit_ids):
        raise ValueError("Fit IDs may contain only letters, digits, underscores, and hyphens")


def aggregate_fit_metrics(fit_results):
    """Average generation draws inside fits; SD only across independent fit means."""
    grouped = {}
    for fit in fit_results:
        grouped.setdefault(fit["arm"], []).append(fit["generation_mean_metrics"])
    summary = {}
    for arm, rows in grouped.items():
        if not rows:
            continue
        names = set(rows[0])
        if any(set(row) != names for row in rows):
            raise ValueError("Metric keys differ across fits")
        summary[arm] = {}
        for name in sorted(names):
            values = np.array([row[name] for row in rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite fit-level metric {name}")
            summary[arm][name] = {
                "mean": float(np.mean(values)),
                "between_fit_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "independent_fits": int(len(values)),
            }
    return summary


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _draw_mean(draws):
    if not draws:
        raise ValueError("No generation draws")
    names = set(draws[0]["flat_metrics"])
    if any(set(draw["flat_metrics"]) != names for draw in draws):
        raise ValueError("Metric keys differ across generation draws")
    return {name: float(np.mean([draw["flat_metrics"][name] for draw in draws])) for name in sorted(names)}


def run(args):
    # Import the evaluator's actual relationship helper before recording source
    # provenance, so hashes describe the files this process will execute.
    import prototype2_relationship_checks as relationships
    local_files = {
        "runner": Path(__file__).resolve(),
        "backend_adapter": Path(sys.modules[OfficialPGMBackend.__module__].__file__).resolve(),
        "common_evaluator": Path(sys.modules[evaluate_sample.__module__].__file__).resolve(),
        "relationship_helper": Path(relationships.__file__).resolve(),
    }
    local_provenance = {role: {"path": str(path), "sha256": sha256_file(path)}
                        for role, path in local_files.items()}
    train, holdout, categories, schema = load_pilot(args.data_npz, args.schema)
    validate_args(args, schema)
    codes, domain_sizes = encode_training_data(train, categories, args.age_representation)
    workload, exclusions = aim_workload(domain_sizes, args.aim_max_cells)
    if args.jax_platform != "auto":
        os.environ["JAX_PLATFORMS"] = args.jax_platform
    init_start = time.perf_counter()
    backend = OfficialPGMBackend(args.mbi_repo, args.expected_mbi_commit)
    initialization_seconds = time.perf_counter() - init_start
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (args.out or Path("pipeline_runs/prototype2_pgm") / f"baseline_{stamp}_{uuid.uuid4().hex[:8]}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / ".gitignore").write_text("*\n", encoding="utf-8")
    sample_dir = out / "samples"
    sample_dir.mkdir()
    standard = (args.aim_max_model_size == 80 and args.aim_max_iters == 1000 and
                args.mst_selection_iters == 2500 and args.mst_final_iters == 10000 and args.aim_max_cells is None)
    label = ("PUBLIC GENERATED FIXTURE: SOFTWARE VALIDATION ONLY" if args.public_fixture
             else "LOCAL RESEARCH: NOT AN END-TO-END DP RELEASE")
    configuration = ("fixture_smoke" if args.smoke else "fixture" if args.public_fixture
                     else "standard_inference_defaults" if standard else "custom_inference_settings")
    plan = {
        "status": "planned", "label": label, "configuration": configuration,
        "public_fixture": args.public_fixture, "smoke": args.smoke,
        "data": {"npz_path": str(args.data_npz.resolve()), "npz_sha256": sha256_file(args.data_npz),
                 "schema_path": str(args.schema.resolve()), "schema_sha256": sha256_file(args.schema),
                 "n_train": int(len(train)), "n_holdout": int(len(holdout)),
                 "features": list(FEATURES), "categorical_cardinalities": np.asarray(categories).tolist(),
                 "age_representation": args.age_representation, "pgm_domain": domain_sizes,
                 "age_support": [0, 100], "evaluation_age_edges": AGE_EDGES.tolist(),
                 "representation_note": "Integer AGE0..100 preserves integer input AGE. Legacy bins emit public midpoints and do not learn within-bin ages.",
                 "holdout_used_for_fit": False},
        "privacy": {"epsilon_per_fit": args.epsilon, "delta_per_fit": args.delta,
                    "scope": "each standalone mechanism, including selection and measurements",
                    "additional_C_target_release": False,
                    "composition_note": "Distinct private fits compose if released. Averages and research diagnostics do not establish an end-to-end DP release.",
                    "private_noise_seeds_recorded": False,
                    "rng_note": "Real fits use secret-seeded ordinary NumPy research RNG. Public fixture fitting uses known public seeds and carries no DP guarantee; reproducibility also requires a fixed PYTHONHASHSEED."},
        "arms": args.arms, "fit_ids": args.fit_ids, "draws_per_fit": args.draws,
        "synthetic_rows_per_draw": args.samples, "public_generation_seed_base": args.seed,
        "shuffle_repeats": args.shuffle_repeats,
        "sampling_method": "i.i.d. from fixed fitted model, method=sample",
        "metric_convention": "common holdout L1 in [0,2]; not official CLI training TV",
        "independent_unit": "private fit; generation draws average inside each fit",
        "aim": {"max_model_size_mb": args.aim_max_model_size, "max_iters": args.aim_max_iters,
                "explicit_max_cells": args.aim_max_cells,
                "pair_workload": [list(q) for q, _ in workload], "excluded_pairs": exclusions},
        "mst": {"selection_inference_iters": args.mst_selection_iters, "final_inference_iters": args.mst_final_iters},
        "backend": backend.provenance,
        "local_source": local_provenance,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "initialization_seconds": initialization_seconds,
                    "requested_jax_platform": args.jax_platform,
                    "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                    "timing_note": "Whole fit includes first-use compilation and opaque upstream phases. Generation timing is separate. No FLOPs estimate."},
    }
    _write_json(out / "plan.json", plan)
    print(label, flush=True)
    print(f"Configuration={configuration}; independent fits per arm={args.fits}; draws per fit={args.draws}; rows per draw={args.samples}", flush=True)
    print(f"AIM declared pair workload={len(workload)}/21; explicit exclusions={len(exclusions)}", flush=True)
    fits, manifest_entries = [], []
    active = {}
    try:
        for fit_id in args.fit_ids:
            for arm in args.arms:
                active = {"arm": arm, "fit_id": fit_id}
                print(f"Fitting {ARM_LABELS[arm]} {fit_id}", flush=True)
                fixture_seed = public_seed(args.seed, arm, fit_id, "fixture-fit") if args.public_fixture else None
                log_path = out / f"{arm}_{fit_id}_fit.log"
                started = time.perf_counter()
                with log_path.open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log):
                    fitted = backend.fit(
                        arm, codes, domain_sizes, epsilon=args.epsilon, delta=args.delta,
                        public_fixture_seed=fixture_seed, aim_max_model_size=args.aim_max_model_size,
                        aim_max_iters=args.aim_max_iters, aim_max_cells=args.aim_max_cells,
                        mst_selection_iters=args.mst_selection_iters, mst_final_iters=args.mst_final_iters)
                fit_seconds = time.perf_counter() - started
                draws = []
                for index in range(args.draws):
                    draw_id = f"draw{index+1:03d}"
                    generation_seed = public_seed(args.seed, arm, fit_id, draw_id, "generation")
                    generation_start = time.perf_counter()
                    discrete = fitted.sample_codes(args.samples, generation_seed)
                    sample = decode_synthetic_data(discrete, domain_sizes, args.age_representation)
                    generation_seconds = time.perf_counter() - generation_start
                    sample_path = sample_dir / f"{arm}_{fit_id}_{draw_id}.npy"
                    with sample_path.open("xb") as handle:
                        np.save(handle, sample, allow_pickle=False)
                    entry = {
                        "path": str(sample_path.relative_to(out)), "sha256": sha256_file(sample_path),
                        "arm": arm, "baseline": ARM_LABELS[arm], "fit_id": fit_id,
                        "training_id": fit_id, "target_id": None, "draw_id": draw_id,
                        "generation_seed": generation_seed, "n_rows": args.samples,
                        "age_representation": args.age_representation, "public_fixture": args.public_fixture,
                        "label": label,
                    }
                    manifest_entries.append(entry)
                    evaluation_start = time.perf_counter()
                    evaluation = evaluate_sample(sample, holdout, categories,
                                                 shuffle_repeats=args.shuffle_repeats,
                                                 seed=public_seed(args.seed, arm, fit_id, draw_id, "shuffle"))
                    flat = flatten_metrics(evaluation)
                    # Keep shuffle contrasts in the same fit-level aggregation.
                    for key, value in evaluation.get("shuffle", {}).get("shuffle_minus_original", {}).items():
                        flat[f"shuffle_minus_original/{key}"] = float(value)
                    draws.append({
                        "draw_id": draw_id, "sample": entry,
                        "generation_seconds": generation_seconds,
                        "evaluation_seconds": time.perf_counter() - evaluation_start,
                        "evaluation": evaluation, "flat_metrics": flat,
                    })
                fit_result = {
                    "arm": arm, "baseline": ARM_LABELS[arm], "fit_id": fit_id,
                    "label": label, "public_fixture": args.public_fixture,
                    "configuration": configuration, "fit_seconds": fit_seconds,
                    "fit_metadata": fitted.metadata, "draws": draws,
                    "generation_mean_metrics": _draw_mean(draws),
                    "fit_log": log_path.name,
                }
                if args.public_fixture:
                    fit_result["public_fixture_fit_seed"] = fixture_seed
                fits.append(fit_result)
                _write_json(out / f"{arm}_{fit_id}_results.json", fit_result)
                print(f"Completed {arm} {fit_id}: fit={fit_seconds:.2f}s, {len(draws)} generation draws", flush=True)
                del fitted
    except BaseException as error:
        _write_json(out / "failure.json", {
            "status": "FAILED_PARTIAL_RUN", "label": label, "active": active,
            "completed_fits": [{"arm": x["arm"], "fit_id": x["fit_id"]} for x in fits],
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
            "interpretation": "Incomplete run. Do not report partial favorable fits as a complete comparison.",
        })
        raise
    aggregate = aggregate_fit_metrics(fits)
    _write_json(out / "sample_manifest.json", {
        "version": 1, "status": "complete", "label": label, "public_fixture": args.public_fixture,
        "sample_format": "encoded AGE=(years-50)/50, followed by six contiguous public categorical indices",
        "features": list(FEATURES), "schema_path": str(args.schema.resolve()),
        "data_npz_path": str(args.data_npz.resolve()), "samples": manifest_entries,
    })
    _write_json(out / "results.json", {
        "status": "complete", "label": label, "configuration": configuration,
        "public_fixture": args.public_fixture, "backend": backend.provenance,
        "independent_unit": "fit-level means, after averaging generation draws",
        "n_fits_per_arm": args.fits, "draws_per_fit": args.draws,
        "aggregate": aggregate, "fits": fits,
    })
    with (out / "metric_summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "metric", "mean", "between_fit_sample_sd", "independent_fits", "public_fixture"])
        for arm, metrics in aggregate.items():
            for metric, values in metrics.items():
                writer.writerow([arm, metric, values["mean"], values["between_fit_sample_sd"], values["independent_fits"], args.public_fixture])
    lines = [label, f"Configuration: {configuration}",
             f"Nominal per-fit mechanism budget: epsilon={args.epsilon}, delta={args.delta}",
             f"Independent fits per arm: {args.fits}; generation draws within each fit: {args.draws}",
             "SD is across independent fit-level means only. With one fit, SD is not reported.",
             "No holdout enters fitting. Scores use the common holdout L1/relationship evaluator.",
             f"AIM candidate pairs: {len(workload)}/21; explicit resource exclusions: {len(exclusions)}",
             f"AGE representation: {args.age_representation}; numerical evaluation uses the fixed public bins.",
             "Repeated generation reuses each fitted private model. New fits consume separate mechanisms.",
             f"Official MBI source: {backend.provenance['commit']}", ""]
    important = ["marginal/categorical_one_way", "marginal/categorical_two_way", "marginal/age_one_way",
                 "marginal/age_two_way", "marginal/fixed_three_way", "marginal/age_outside_fraction",
                 "marginal/age_outside_cohort_fraction"]
    for arm, metrics in aggregate.items():
        lines.append(ARM_LABELS[arm])
        for name in important:
            if name in metrics:
                value = metrics[name]
                sd = "not estimable with one fit" if value["between_fit_sample_sd"] is None else f"{value['between_fit_sample_sd']:.6f}"
                lines.append(f"  {name}: mean={value['mean']:.6f}; between-fit sample SD={sd}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved complete {configuration} results: {out}", flush=True)
    return out


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"PGM baseline run stopped: {error}\n")


if __name__ == "__main__":
    main()
