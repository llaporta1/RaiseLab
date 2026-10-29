#!/usr/bin/env python3
"""Frozen-model categorical marginal-guidance C/D CPU pilot.

This is a RESEARCH pilot, not an end-to-end DP release. C uses one cached
Laplace histogram release; D and all reference-data metrics are NONPRIVATE.
Only load B_final.pt from YOUR OWN locally generated A/B run: the old
checkpoint contains optimizer/accountant objects and needs trusted pickle.
No original model source is edited, no training occurs, and AGE is not clipped.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import secrets
import sys
import time

import numpy as np
import opacus
import torch
from opacus.accountants import RDPAccountant

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_tabdiff_smoke as smoke
import run_ipums_ab_pilot as ab
from prototype2_guidance_core import guided_sampling, make_offsets


FEATURES = ab.FEATURES
TARGETS = FEATURES[1:]
PSEUDOCOUNT = 1e-3
PROB_FLOOR = 1e-8
EXPECTED_ARCHITECTURE = {"num_layers": 1, "d_token": 4, "n_head": 1,
                         "factor": 2, "dim_t": 64, "numeric_columns": 1}


def write_json(path, content):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def histograms(table, categories):
    """One fixed-public-domain count vector for each categorical attribute."""
    ab.evaluation_codes(table, categories)
    return [np.bincount(table[:, j + 1].astype(np.int64), minlength=int(k))
            .astype(np.float64) for j, k in enumerate(categories)]


def normalize_counts(counts):
    """Postprocess only the supplied vector; no true-count fallback."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not len(counts) or not np.isfinite(counts).all():
        raise ValueError("Expected a nonempty finite count vector")
    positive = np.maximum(counts, 0) + PSEUDOCOUNT
    return positive / positive.sum()


def create_dp_targets(counts, schema, epsilon, rng=None):
    """Six add/remove-sensitive histogram vectors; sequential composition.

    Each histogram has L1 sensitivity 1, so each CELL gets Laplace scale
    1/(epsilon/6), NOT an additional division of epsilon by its cardinality.
    An optional rng is solely for mechanism unit tests. Production research
    calls use independent, undisclosed system-entropy seeds.
    """
    if not math.isfinite(epsilon) or epsilon <= 0 or len(counts) != len(TARGETS):
        raise ValueError("Need positive epsilon and all six histogram vectors")
    rng = rng if rng is not None else np.random.default_rng(secrets.randbits(128))
    scale = len(TARGETS) / epsilon
    probabilities = {}
    for name, vector in zip(TARGETS, counts):
        noisy = vector + rng.laplace(0, scale, size=len(vector))
        probabilities[name] = normalize_counts(noisy).tolist()
    return {
        "format_version": 1,
        "mechanism": "six_laplace_histogram_vectors_add_remove_research",
        "features": TARGETS,
        "category_domains": schema["category_domains"],
        "epsilon_marginals": epsilon, "delta_marginals": 0.0,
        "epsilon_per_histogram": epsilon / len(TARGETS),
        "l1_sensitivity_per_histogram": 1.0, "laplace_scale_per_cell": scale,
        "count_pseudocount": PSEUDOCOUNT,
        "postprocessing": "max(noisy_counts,0)+fixed_pseudocount; normalize noisy vector",
        "probabilities": probabilities,
        "warning": "Research floating-point RNG, not production privacy certification. "
                   "No true counts, exact record count, raw-data hashes, or noise seeds in this file.",
    }


def validate_dp_targets(payload, schema, epsilon):
    if payload.get("format_version") != 1:
        raise ValueError("Unknown DP target format")
    if payload.get("mechanism") != "six_laplace_histogram_vectors_add_remove_research":
        raise ValueError("Unknown target mechanism")
    if payload.get("features") != TARGETS or payload.get("category_domains") != schema["category_domains"]:
        raise ValueError("Cached targets use different features or category-code ordering")
    expected = {"epsilon_marginals": epsilon, "delta_marginals": 0.0,
                "epsilon_per_histogram": epsilon / len(TARGETS),
                "l1_sensitivity_per_histogram": 1.0,
                "laplace_scale_per_cell": len(TARGETS) / epsilon,
                "count_pseudocount": PSEUDOCOUNT}
    for key, value in expected.items():
        if not math.isclose(float(payload.get(key, float("nan"))), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Cached target setting does not match: {key}")
    result = []
    for name in TARGETS:
        vector = np.asarray(payload["probabilities"][name], dtype=np.float64)
        if vector.shape != (len(schema["category_domains"][name]),):
            raise ValueError(f"Wrong target length for {name}")
        if not np.isfinite(vector).all() or np.any(vector <= 0) or not np.isclose(vector.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError(f"Invalid probability vector for {name}")
        result.append(vector)
    return result


def compare_sample(sample, train_codes, validation_codes, categories):
    codes, age_info = ab.evaluation_codes(sample, categories)
    sizes = np.r_[len(ab.AGE_EDGES) + 1, categories]
    return {"train": ab.compare_marginals(codes, train_codes, sizes),
            "validation": ab.compare_marginals(codes, validation_codes, sizes),
            "age": age_info}


def grouped_errors(comparison):
    one, two = comparison["one_way"], comparison["two_way"]
    age_pairs = [value for key, value in two.items() if "AGE" in key.split(" x ")]
    cat_pairs = [value for key, value in two.items() if "AGE" not in key.split(" x ")]
    return {"all_one_way": comparison["mean_one_way_l1"],
            "categorical_one_way": float(np.mean([one[name] for name in TARGETS])),
            "age_one_way": one["AGE"],
            "all_two_way": comparison["mean_two_way_l1"],
            "categorical_two_way": float(np.mean(cat_pairs)),
            "age_two_way": float(np.mean(age_pairs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True,
                        help="Your own original A/B run directory, containing trusted B_final.pt")
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW directory; no overwrites")
    parser.add_argument("--epsilon-marginals", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-log-ratio", type=float, default=2.0)
    parser.add_argument("--reference-samples", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reuse-dp-targets", type=Path,
                        help="Reuse an existing release; requires its companion private_provenance.json")
    args = parser.parse_args()
    if min(args.reference_samples, args.samples, args.sample_batch, args.threads) < 1:
        parser.error("Sample counts, chunk size and threads must be positive")
    if not math.isfinite(args.epsilon_marginals) or args.epsilon_marginals <= 0:
        parser.error("epsilon-marginals must be finite and positive")
    if not math.isfinite(args.strength) or args.strength < 0:
        parser.error("strength must be finite and nonnegative")
    if not math.isfinite(args.max_log_ratio) or args.max_log_ratio <= 0:
        parser.error("max-log-ratio must be finite and positive")
    if args.out.exists():
        parser.error("Output directory already exists. Preserve it and choose a NEW path.")
    torch.set_num_threads(args.threads)
    base = json.loads((args.base_run / "metrics.json").read_text())
    config = base["config"]
    if config.get("architecture") != EXPECTED_ARCHITECTURE or config.get("fixed_schedules") is not True:
        raise ValueError("This script supports only the unchanged tiny fixed-schedule A/B pilot")
    if config.get("features") != FEATURES or args.samples != base["n_synthetic"]:
        raise ValueError("Features and sample count must match the original A/B pilot")
    train, validation, categories, schema = ab.load_inputs(argparse.Namespace(
        data_npz=args.data_npz, schema=args.schema, batch_size=config["batch_size"]))
    if categories.tolist() != config["categories"] or len(train) != base["n_train"] or len(validation) != base["n_holdout"]:
        raise ValueError("Input cardinalities/sizes differ from original run")
    public_schema = json.loads((args.base_run / "public_schema.json").read_text())
    for key in ("features", "category_domains", "age_transform"):
        if schema[key] != public_schema[key]:
            raise ValueError(f"Schema differs from original run: {key}")
    checkpoint_path = args.base_run / "B_final.pt"
    print("PRELIMINARY C/D LOGIT-GUIDANCE PILOT. No model training. D/metrics are NONPRIVATE.", flush=True)
    print("Loading your own trusted local B_final.pt (never use an untrusted checkpoint).", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError("Checkpoint and original metrics configuration disagree")
    accountant = RDPAccountant()
    accountant.load_state_dict(checkpoint["accountant"])
    epsilon_train = float(accountant.get_epsilon(delta=config["delta"]))
    if sum(item[2] for item in accountant.history) != config["steps"]:
        raise ValueError("Accountant step count differs from trained checkpoint")
    if not math.isclose(epsilon_train, base["training_privacy_accounting"]["epsilon"], rel_tol=1e-8):
        raise ValueError("Checkpoint accountant does not reproduce original epsilon")
    model = smoke.build_diffusion(categories, config["sampling_steps"])
    model.load_state_dict(checkpoint["model"], strict=True)
    # Preserve original requires_grad flags to preserve identical CPU sampling.
    # Frozen here means no optimizer/backward/weight changes, checked by hash.
    model.eval()
    original_weights = model_hash(model)
    generation_seed = int(config["model_seed"]) + 1000
    reference_seed = int(config["model_seed"]) + 2000
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    with np.load(args.base_run / "B_synthetic.npz", allow_pickle=False) as saved:
        old_b = saved["X_synthetic"].copy()
    print("Checking zero-guidance sampling exactly reproduces saved B...", flush=True)
    torch.manual_seed(generation_seed)
    zero_offsets = [np.zeros(int(k), dtype=np.float64) for k in categories]
    with guided_sampling(model, zero_offsets) as zero_stats:
        b_sample = ab.generate(model, args.samples, args.sample_batch, categories)
    if not np.array_equal(b_sample, old_b):
        raise RuntimeError("B reproduction failed. Do not interpret C/D yet. Check unchanged source, "
                           "PyTorch version, CPU threads, chunk size and original sampling configuration.")
    if model_hash(model) != original_weights:
        raise RuntimeError("Model weights changed during baseline generation")
    print("PASS: zero guidance reproduces B exactly; weights unchanged.", flush=True)

    # This provenance is explicitly NONPRIVATE and is not part of DP targets.
    provenance = {"privacy_status": "NONPRIVATE LOCAL RESEARCH METADATA; do not publish",
                  "input_npz_sha256": file_hash(args.data_npz),
                  "schema_sha256": file_hash(args.schema),
                  "checkpoint_sha256": file_hash(checkpoint_path),
                  "original_input_hash_available": False,
                  "note": "Hashes identify current files, not proof of original training provenance."}
    write_json(args.out / "private_provenance.json", provenance)
    train_counts = histograms(train.numpy(), categories)
    oracle_targets = [normalize_counts(vector) for vector in train_counts]
    if args.reuse_dp_targets:
        previous_provenance = json.loads((args.reuse_dp_targets.parent / "private_provenance.json").read_text())
        for key in ("input_npz_sha256", "schema_sha256"):
            if previous_provenance[key] != provenance[key]:
                raise ValueError("Cached target provenance does not match current inputs")
        dp_payload = json.loads(args.reuse_dp_targets.read_text())
        dp_targets = validate_dp_targets(dp_payload, schema, args.epsilon_marginals)
        target_origin = "reused_existing_release_no_fresh_histogram_noise"
        print("Reusing the same cached DP marginal release.", flush=True)
    else:
        dp_payload = create_dp_targets(train_counts, schema, args.epsilon_marginals)
        dp_targets = validate_dp_targets(dp_payload, schema, args.epsilon_marginals)
        target_origin = "one_new_histogram_release"
        print(f"One DP marginal release: total epsilon={args.epsilon_marginals:g}; "
              f"Laplace scale={len(TARGETS)/args.epsilon_marginals:g} counts per cell.", flush=True)
    write_json(args.out / "dp_targets.json", dp_payload)
    print("Targets cached. Future repeats should use --reuse-dp-targets with this file.", flush=True)

    print(f"Generating {args.reference_samples} B-only calibration samples (no raw-data queries)...", flush=True)
    torch.manual_seed(reference_seed)
    reference_sample = ab.generate(model, args.reference_samples, args.sample_batch, categories)
    reference_targets = [normalize_counts(vector) for vector in histograms(reference_sample, categories)]
    offsets = {
        "C": make_offsets(dp_targets, reference_targets, args.strength, args.max_log_ratio, PROB_FLOOR),
        "D": make_offsets(oracle_targets, reference_targets, args.strength, args.max_log_ratio, PROB_FLOOR),
    }
    samples, hook_stats = {"B": b_sample}, {}
    for arm in ("C", "D"):
        print(f"Generating {arm}: same frozen B, same evaluation sampling seed, categorical logit guidance...", flush=True)
        torch.manual_seed(generation_seed)
        with guided_sampling(model, offsets[arm]) as stats:
            samples[arm] = ab.generate(model, args.samples, args.sample_batch, categories)
        hook_stats[arm] = dict(stats)
        if model_hash(model) != original_weights:
            raise RuntimeError(f"Model weights changed in arm {arm}")
        expected_calls = math.ceil(args.samples / args.sample_batch) * config["sampling_steps"]
        if stats["hook_calls"] != expected_calls:
            raise RuntimeError("Guidance hook did not execute at each expected MDLM sampling step")
        if args.strength > 0 and any(np.any(vector) for vector in offsets[arm]) and stats["masked_cells_adjusted"] == 0:
            raise RuntimeError("Nonzero guidance failed to reach masked categorical states")
        np.savez_compressed(args.out / f"{arm}_synthetic.npz", X_synthetic=samples[arm],
                            categories=categories, features=np.asarray(FEATURES))

    # Evaluation is intentionally downstream of target construction/generation.
    # Neither validation records nor their metrics select target features/strength.
    train_codes, _ = ab.evaluation_codes(train.numpy(), categories, real_records=True)
    validation_codes, _ = ab.evaluation_codes(validation, categories, real_records=True)
    with np.load(args.base_run / "A_synthetic.npz", allow_pickle=False) as saved:
        samples["A"] = saved["X_synthetic"].copy()
    results = {arm: compare_sample(samples[arm], train_codes, validation_codes, categories)
               for arm in ("A", "B", "C", "D")}
    for arm in ("A", "B"):
        for new_ref, old_ref in (("train", "train"), ("validation", "holdout")):
            for order in ("one_way", "two_way"):
                current = results[arm][new_ref][order]
                previous = base["results"][arm][old_ref][order]
                if current.keys() != previous.keys() or any(not math.isclose(current[k], previous[k], abs_tol=1e-10) for k in current):
                    raise RuntimeError("A/B reference metrics changed. Check that the original input NPZ/schema were used.")
    target_errors = {name: float(np.abs(dp - truth).sum())
                     for name, dp, truth in zip(TARGETS, dp_targets, oracle_targets)}
    grouped = {arm: grouped_errors(results[arm]["validation"]) for arm in results}
    report = {
        "purpose": "Frozen-model categorical marginal logit-guidance C/D pilot",
        "formal_end_to_end_dp_claimed": False, "metrics_are_dp": False,
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "opacus": opacus.__version__},
        "method": "Fixed capped log(target/reference) offsets before MDLM categorical sampling. "
                  "Heuristic, not exact conditional sampling or guaranteed marginal matching.",
        "base_config": config, "target_features": TARGETS,
        "untargeted": "AGE one-way and ALL two-way marginals (no direct joint constraints)",
        "strength": args.strength, "max_log_ratio": args.max_log_ratio,
        "probability_floor": PROB_FLOOR, "count_pseudocount": PSEUDOCOUNT,
        "oracle_definition": "Unnoised training histogram plus fixed 0.001 pseudocount per public cell, normalized",
        "reference_samples": args.reference_samples, "reference_seed": reference_seed,
        "evaluation_seed": generation_seed, "samples_per_arm": args.samples,
        "sample_batch": args.sample_batch, "cpu_threads": args.threads,
        "n_train": len(train), "n_validation": len(validation),
        "privacy": {"B_training_epsilon": epsilon_train,
                    "C_marginal_epsilon": args.epsilon_marginals,
                    "C_nominal_composed_epsilon": epsilon_train + args.epsilon_marginals,
                    "delta": config["delta"], "D_is_dp": False,
                    "adjacency_assumption": "add/remove one record; fixed public training sampling parameters",
                    "composition_scope": "Conditional nominal mechanism composition only. "
                        "Input selection, preprocessing, tuning, floating-point RNG, and releases of A/D/evaluation are not audited.",
                    "fixed_total_budget_comparison": False, "target_origin": target_origin,
                    "reusing_C_targets": "No new histogram release when exactly the same dp_targets.json is reused."},
        "baseline_reproduction_bitwise": True, "model_weights_unchanged": True,
        "zero_guidance_hook_stats": zero_stats, "guidance_hook_stats": hook_stats,
        "dp_target_l1_vs_smoothed_training_oracle": target_errors,
        "results": results, "validation_grouped_errors": grouped,
        "source_sha256": {name: file_hash(Path(__file__).parent / name) for name in
            ("local_tabdiff_smoke.py", "run_ipums_ab_pilot.py", "prototype2_guidance_core.py", "run_ipums_cd_pilot.py")},
        "tabdiff_source_sha256": {name: file_hash(Path(__file__).resolve().parents[1] / name) for name in
            ("tabdiff/models/unified_ctime_diffusion.py", "tabdiff/models/noise_schedule.py",
             "tabdiff/modules/main_modules.py", "tabdiff/modules/transformer.py")},
        "base_artifact_sha256": {name: file_hash(args.base_run / name) for name in
            ("B_final.pt", "B_synthetic.npz", "A_synthetic.npz", "metrics.json", "public_schema.json")},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Tiny model, one training/noise draw; no convergence/generalization claim.",
            "No direct AGE correction. AGE can change indirectly through categorical context.",
            "Only categorical ONE-WAY targets; two-way fidelity and associations may worsen.",
            "C has an additional marginal budget; B/C is not privacy-budget matched.",
            "Sparse high-cardinality Laplace targets may be poor after rectifying negative counts.",
            "D and all reference-data evaluation are NONPRIVATE; do not publish this folder.",
            "Validation has informed development; final research needs a separate untouched test set.",
            "C2ST, downstream ML utility, and full mixed-type marginal guidance are NOT computed."]}
    write_json(args.out / "metrics.json", report)
    with (args.out / "marginal_errors.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "reference", "order", "query", "l1"])
        for arm in results:
            for reference in ("train", "validation"):
                for order, key in ((1, "one_way"), (2, "two_way")):
                    for query, error in results[arm][reference][key].items():
                        writer.writerow([arm, reference, order, query, error])
    lines = ["PRELIMINARY A/B/C/D: CATEGORICAL ONE-WAY LOGIT GUIDANCE ONLY",
             "No new training. Same frozen B model and evaluation sampling draws for B/C/D.",
             "A/B are reproduced from the original run; guidance is inside diffusion, NOT post-hoc column replacement.",
             "Targets: SEX, RACE, HISPAN, EDUCD, OCC, IND. AGE and all two-way marginals are untargeted.",
             "Heuristic capped log-marginal-ratio guidance; no exact-matching or dependence-preservation guarantee.",
             f"Guidance strength={args.strength:g}; log-ratio cap={args.max_log_ratio:g}; B reference samples={args.reference_samples}.",
             f"B epsilon_train={epsilon_train:.6f}; C adds epsilon_marginals={args.epsilon_marginals:g}.",
             f"C NOMINAL composed epsilon={epsilon_train + args.epsilon_marginals:.6f}, delta={config['delta']:g}; NOT a fixed-budget B/C comparison.",
             "Nominal composition assumes consistent add/remove-record adjacency and fixed public sampling parameters.",
             "NOT AN END-TO-END DP RELEASE. A, D, reference metrics and local provenance are NONPRIVATE.",
             "", "VALIDATION L1 ERRORS (lower is better; range 0 to 2)",
             "arm\tall_1way\tcat_1way\tAGE_1way\tall_2way\tcat_pairs\tAGE_pairs\tAGE_outside_[0,100]"]
    for arm in results:
        row = grouped[arm]
        values = [row[key] for key in ("all_one_way", "categorical_one_way", "age_one_way", "all_two_way", "categorical_two_way", "age_two_way")]
        age_fraction = results[arm]["age"]["decoded_outside_0_100_fraction"]
        lines.append(arm + "\t" + "\t".join(f"{value:.6f}" for value in values) + f"\t{age_fraction:.4%}")
    lines += ["", "PER-VARIABLE VALIDATION ONE-WAY L1", "variable\tA\tB\tC\tD"]
    for name in FEATURES:
        lines.append(name + "\t" + "\t".join(f"{results[arm]['validation']['one_way'][name]:.6f}" for arm in results))
    lines += ["", "DP TARGET L1 VS SMOOTHED TRUE TRAINING MARGINAL (before generation)"]
    lines += [f"{name}\t{value:.6f}" for name, value in target_errors.items()]
    lines += ["", "Checks: zero-guidance B exact reproduction PASS; model weights unchanged PASS; native sampling hook PASS.",
              "D is a diagnostic oracle, NOT a private arm or guaranteed upper bound on performance.",
              "Do not clip ages, remove invalid samples, or cherry-pick guidance using validation scores.",
              "Do not commit raw NPZs, models, this result folder or oracle/validation outputs.",
              "Future sampling repeats: reuse this folder's dp_targets.json; drawing fresh target noise adds a new release.",
              "C2ST, downstream ML utility and direct numerical guidance: NOT COMPUTED."]
    with (args.out / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Complete. Read {args.out / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
