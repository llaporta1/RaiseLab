#!/usr/bin/env python3
"""Fixed-model sampling stability and categorical-dependency controls.

Reuses an existing A/B run and C/D run, including the EXACT cached DP targets.
No training and no new marginal-noise release. A, D, and reference metrics are
NONPRIVATE. This measures generation variability conditional on fixed models
and one DP marginal release, not independent training/noise replications.
Only load YOUR OWN trusted local checkpoints: torch.load uses trusted pickle.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import opacus
import torch
from opacus.accountants import RDPAccountant

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_tabdiff_smoke as smoke
import run_ipums_ab_pilot as ab
import run_ipums_cd_pilot as cd
from prototype2_guidance_core import guided_sampling, make_offsets


ARMS = ("A", "B", "C", "D")
GROUPS = ("all_one_way", "categorical_one_way", "age_one_way",
          "all_two_way", "categorical_two_way", "age_two_way",
          "age_outside_fraction")
PAIR_GROUPS = ("all_two_way", "categorical_two_way", "age_two_way")
SCRIPT_SOURCES = ("local_tabdiff_smoke.py", "run_ipums_ab_pilot.py",
                  "prototype2_guidance_core.py", "run_ipums_cd_pilot.py")
TABDIFF_SOURCES = ("tabdiff/models/unified_ctime_diffusion.py",
                   "tabdiff/models/noise_schedule.py",
                   "tabdiff/modules/main_modules.py", "tabdiff/modules/transformer.py")
BASE_ARTIFACTS = ("B_final.pt", "B_synthetic.npz", "A_synthetic.npz",
                  "metrics.json", "public_schema.json")
SHUFFLE_NAMESPACE = 982451653


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hashes(recorded, names, root, label):
    """Only use this fixed allowlist of paths, not arbitrary metadata paths."""
    if not isinstance(recorded, dict):
        raise ValueError(f"Original C/D metadata lacks {label} hashes")
    for name in names:
        if name not in recorded or cd.file_hash(root / name) != recorded[name]:
            raise ValueError(f"{label} changed since the C/D run: {name}. "
                             "Preserve prior results and inspect the mismatch before continuing.")


def mean_sd(values):
    """Sample SD across GENERATION seeds; do not pool nested shuffle draws."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Need at least two finite generation-seed values")
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1)),
            "n_generation_seeds": len(array)}


def grouped_sample(sample, train_codes, validation_codes, categories):
    comparison = cd.compare_sample(sample, train_codes, validation_codes, categories)
    grouped = cd.grouped_errors(comparison["validation"])
    grouped["age_outside_fraction"] = comparison["age"]["decoded_outside_0_100_fraction"]
    return {"grouped_validation": grouped,
            "one_way_validation": comparison["validation"]["one_way"],
            "two_way_validation": comparison["validation"]["two_way"]}


def shuffle_categorical_columns(sample, rng):
    """Independently permute six categorical columns; leave AGE in place.

    This preserves each empirical one-way distribution exactly while breaking
    row alignment. Finite shuffles can retain chance associations; no exact
    independence or population null hypothesis is asserted.
    """
    values = np.asarray(sample)
    if values.ndim != 2 or values.shape[1] != len(ab.FEATURES) or not len(values):
        raise ValueError("Expected a nonempty seven-column synthetic sample")
    if not np.isfinite(values).all():
        raise ValueError("Cannot shuffle nonfinite synthetic values")
    shuffled = values.copy()
    for column in range(1, values.shape[1]):
        shuffled[:, column] = values[rng.permutation(len(values)), column]
        if not np.array_equal(np.sort(shuffled[:, column]), np.sort(values[:, column])):
            raise RuntimeError("Categorical shuffle changed an empirical one-way histogram")
    if not np.array_equal(shuffled[:, 0], values[:, 0]):
        raise RuntimeError("Categorical shuffle changed AGE")
    return shuffled


def check_model_unchanged(model, weights_hash, flags, arm):
    if cd.model_hash(model) != weights_hash:
        raise RuntimeError(f"Frozen model weights changed in arm {arm}")
    if [p.requires_grad for p in model.parameters()] != flags:
        raise RuntimeError(f"Model gradient flags changed in arm {arm}")


def check_original_metrics(current, original, arm):
    for current_key, original_key in (("one_way_validation", "one_way"),
                                       ("two_way_validation", "two_way")):
        new = current[current_key]
        old = original["results"][arm]["validation"][original_key]
        if new.keys() != old.keys() or any(
                not math.isclose(new[key], old[key], rel_tol=0, abs_tol=1e-10)
                for key in new):
            raise RuntimeError(f"Original-seed {arm} validation metrics changed")
    expected_age = original["results"][arm]["age"]["decoded_outside_0_100_fraction"]
    if current["grouped_validation"]["age_outside_fraction"] != expected_age:
        raise RuntimeError(f"Original-seed {arm} AGE support metric changed")


def aggregate_results(per_seed):
    arms = {arm: {key: mean_sd([row["arms"][arm]["grouped_validation"][key]
                               for row in per_seed]) for key in GROUPS} for arm in ARMS}
    per_feature = {arm: {name: mean_sd([row["arms"][arm]["one_way_validation"][name]
                                      for row in per_seed]) for name in ab.FEATURES}
                   for arm in ARMS}
    paired = {key: mean_sd([row["paired_B_minus_C"][key] for row in per_seed])
              for key in GROUPS}
    shuffled = {key: mean_sd([row["shuffle_control"]["mean_grouped_validation"][key]
                             for row in per_seed]) for key in GROUPS}
    benefit = {key: mean_sd([row["shuffle_control"]["paired_shuffle_minus_C"][key]
                            for row in per_seed]) for key in PAIR_GROUPS}
    return {"arms": arms, "one_way_by_feature": per_feature,
            "paired_B_minus_C": paired, "C_shuffle_means": shuffled,
            "paired_shuffle_minus_C": benefit}


def write_outputs(out, report):
    cd.write_json(out / "metrics.json", report)
    with (out / "per_seed.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["repeat_index", "generation_seed", "kind", "label", "metric", "value"])
        for row in report["per_seed"]:
            prefix = [row["repeat_index"], row["generation_seed"]]
            for arm, result in row["arms"].items():
                for key, value in result["grouped_validation"].items():
                    writer.writerow(prefix + ["grouped_validation", arm, key, value])
                for key, value in result["one_way_validation"].items():
                    writer.writerow(prefix + ["one_way_validation", arm, key, value])
                for key, value in result["two_way_validation"].items():
                    writer.writerow(prefix + ["two_way_validation", arm, key, value])
            for key, value in row["paired_B_minus_C"].items():
                writer.writerow(prefix + ["paired_reduction", "B_minus_C", key, value])
            for key, value in row["shuffle_control"]["mean_grouped_validation"].items():
                writer.writerow(prefix + ["within_seed_shuffle_mean", "C_shuffled", key, value])
            for key, value in row["shuffle_control"]["paired_shuffle_minus_C"].items():
                writer.writerow(prefix + ["paired_dependency_control", "shuffle_minus_C", key, value])

    def display(summary, percent=False):
        scale = 100 if percent else 1
        suffix = "%" if percent else ""
        return f"{summary['mean']*scale:.6f} +/- {summary['sample_sd']*scale:.6f}{suffix}"

    aggregates = report["aggregate"]
    lines = ["FROZEN-MODEL SAMPLING STABILITY AND COLUMN-SHUFFLE CONTROL",
             "NO TRAINING; NO NEW DP MARGINAL RELEASE; EXISTING DP TARGET CACHE UNCHANGED.",
             "A, D and reference-data metrics remain NONPRIVATE. Keep this result folder local.",
             f"Generation seeds: {report['generation_seeds']}",
             f"Samples/arm/seed={report['samples_per_arm']}; shuffles/C sample={report['shuffle_repeats']}.",
             "All values are MEAN +/- SAMPLE SD across GENERATION seeds, not training/noise replications.",
             "Models, training randomness, DP target noise, and B reference pool are fixed.",
             "Original seed retained as a reproduction check; additional seeds fixed in advance; no seeds removed by outcome.",
             "", "VALIDATION ERRORS (L1 range [0,2], lower is better)",
             "arm\tall_1way\tcat_1way\tAGE_1way\tall_2way\tcat_pairs\tAGE_pairs\tAGE_outside_[0,100]"]
    for arm in ARMS:
        values = [display(aggregates["arms"][arm][key], percent=(key == "age_outside_fraction"))
                  for key in GROUPS]
        lines.append(arm + "\t" + "\t".join(values))
    lines += ["", "PAIRED B MINUS C (positive means C has lower error; AGE fraction is in percentage points)"]
    for key in GROUPS:
        lines.append(f"{key}\t{display(aggregates['paired_B_minus_C'][key], key == 'age_outside_fraction')}")
    lines += ["", "C VERSUS INDEPENDENT-COLUMN-SHUFFLE CONTROL",
              "Each shuffle preserves ALL empirical one-way marginals exactly and leaves the AGE vector unchanged.",
              "Average shuffles WITHIN each generation seed first, then summarize across generation seeds.",
              "metric\tC_error\tshuffled_C_error\tpaired_shuffle_minus_C"]
    for key in PAIR_GROUPS:
        lines.append(key + "\t" + "\t".join((
            display(aggregates["arms"]["C"][key]),
            display(aggregates["C_shuffle_means"][key]),
            display(aggregates["paired_shuffle_minus_C"][key]))))
    lines += ["Positive shuffle-minus-C means C's row associations help that error metric relative to the shuffle control.",
              "Negative means shuffled columns score better. Neither sign is proof about higher-order relationships.",
              "This is NOT a significance test, confidence interval, independent-training replication or global fidelity claim.",
              "", "PER-FEATURE VALIDATION ONE-WAY L1", "variable\tA\tB\tC\tD"]
    for name in ab.FEATURES:
        lines.append(name + "\t" + "\t".join(display(aggregates["one_way_by_feature"][arm][name]) for arm in ARMS))
    lines += ["", "Checks PASS: original A/B/C/D samples bitwise reproduced; fixed weights/flags; exact one-way shuffle preservation; cache unchanged.",
              "Privacy accounting inherited from the previous C/D pilot; repeated sampling adds no fresh training or marginal release.",
              "Oracle use, repeated validation, source selection and RNG/security caveats still prevent an end-to-end DP claim.",
              "Validation has informed development. A final study needs untouched test data and independent training/DP-noise runs.",
              "C2ST, downstream ML utility and higher-order tests: NOT COMPUTED."]
    with (out / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cd-run", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW directory; no overwrites")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 20 or not 1 <= args.shuffle_repeats <= 100 or args.threads < 1:
        parser.error("Use 2..20 generation repeats, 1..100 shuffles per repeat, and positive threads")
    if args.out.exists():
        parser.error("Output directory already exists. Preserve it and choose a NEW path.")

    scripts = Path(__file__).resolve().parent
    repository = scripts.parent
    base = read_json(args.base_run / "metrics.json")
    original = read_json(args.cd_run / "metrics.json")
    provenance = read_json(args.cd_run / "private_provenance.json")
    config = base["config"]
    if original["base_config"] != config or config.get("architecture") != cd.EXPECTED_ARCHITECTURE:
        raise ValueError("A/B and C/D model configurations differ or are unsupported")
    if config.get("fixed_schedules") is not True or config.get("features") != ab.FEATURES:
        raise ValueError("Expected the original seven-feature fixed-schedule pilot")
    if original["target_features"] != cd.TARGETS:
        raise ValueError("C/D target features differ from the original six categorical columns")
    if args.threads != original["cpu_threads"]:
        raise ValueError("Use the same --threads value as the original C/D run for exact reproduction")
    if original["count_pseudocount"] != cd.PSEUDOCOUNT or original["probability_floor"] != cd.PROB_FLOOR:
        raise ValueError("Guidance smoothing/floor changed since the C/D run")
    versions = {"torch": torch.__version__, "numpy": np.__version__, "opacus": opacus.__version__}
    if original.get("versions") != versions:
        raise ValueError("Package versions differ from the recorded C/D run. Do not upgrade this environment.")
    check_hashes(original.get("source_sha256"), SCRIPT_SOURCES, scripts, "pilot source")
    check_hashes(original.get("tabdiff_source_sha256"), TABDIFF_SOURCES, repository, "TabDiff source")
    check_hashes(original.get("base_artifact_sha256"), BASE_ARTIFACTS, args.base_run, "A/B artifact")
    for key, path in (("input_npz_sha256", args.data_npz), ("schema_sha256", args.schema),
                      ("checkpoint_sha256", args.base_run / "B_final.pt")):
        if provenance.get(key) != cd.file_hash(path):
            raise ValueError(f"Current input does not match C/D private provenance: {key}")
    train, validation, categories, schema = ab.load_inputs(argparse.Namespace(
        data_npz=args.data_npz, schema=args.schema, batch_size=config["batch_size"]))
    if categories.tolist() != config["categories"] or len(train) != base["n_train"] or len(validation) != base["n_holdout"]:
        raise ValueError("Data dimensions differ from the original A/B run")
    if len(train) != original["n_train"] or len(validation) != original["n_validation"]:
        raise ValueError("Data dimensions differ from the original C/D run")
    public_schema = read_json(args.base_run / "public_schema.json")
    for key in ("features", "category_domains", "age_transform"):
        if schema[key] != public_schema[key]:
            raise ValueError(f"Public schema differs from the original run: {key}")
    count = int(original["samples_per_arm"])
    batch = int(original["sample_batch"])
    reference_count = int(original["reference_samples"])
    if min(count, batch, reference_count) < 1 or count != base["n_synthetic"]:
        raise ValueError("Invalid or mismatched original sampling settings")
    target_path = args.cd_run / "dp_targets.json"
    target_hash = cd.file_hash(target_path)
    dp_targets = cd.validate_dp_targets(read_json(target_path), schema,
                                       original["privacy"]["C_marginal_epsilon"])
    torch.set_num_threads(args.threads)
    print("SAMPLING CHECKS ONLY: fixed A/B models; reused C targets; D and metrics NONPRIVATE.", flush=True)
    print("Only your own trusted local checkpoints may be loaded. No training or fresh target noise.", flush=True)
    models, hashes, flags = {}, {}, {}
    for arm in ("A", "B"):
        checkpoint = torch.load(args.base_run / f"{arm}_final.pt", map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError(f"{arm} checkpoint configuration differs from original A/B configuration")
        model = smoke.build_diffusion(categories, config["sampling_steps"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models[arm], hashes[arm] = model, cd.model_hash(model)
        flags[arm] = [parameter.requires_grad for parameter in model.parameters()]
        if arm == "B":
            accountant = RDPAccountant()
            accountant.load_state_dict(checkpoint["accountant"])
            epsilon = float(accountant.get_epsilon(delta=config["delta"]))
            if sum(item[2] for item in accountant.history) != config["steps"] or not math.isclose(
                    epsilon, original["privacy"]["B_training_epsilon"], rel_tol=0, abs_tol=1e-10):
                raise ValueError("B checkpoint privacy accountant differs from original C/D run")

    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print(f"Reconstructing the SAME fixed B reference pool ({reference_count} samples)...", flush=True)
    torch.manual_seed(int(original["reference_seed"]))
    reference_sample = ab.generate(models["B"], reference_count, batch, categories)
    reference = [cd.normalize_counts(vector) for vector in cd.histograms(reference_sample, categories)]
    # Oracle counts are deliberately NONPRIVATE and are used only for D.
    oracle = [cd.normalize_counts(vector) for vector in cd.histograms(train.numpy(), categories)]
    offsets = {"C": make_offsets(dp_targets, reference, original["strength"], original["max_log_ratio"], original["probability_floor"]),
               "D": make_offsets(oracle, reference, original["strength"], original["max_log_ratio"], original["probability_floor"])}
    check_model_unchanged(models["B"], hashes["B"], flags["B"], "B reference")
    train_codes, _ = ab.evaluation_codes(train.numpy(), categories, real_records=True)
    validation_codes, _ = ab.evaluation_codes(validation, categories, real_records=True)
    first_seed = int(original["evaluation_seed"])
    seeds = [first_seed + 10000 * i for i in range(args.repeats)]
    per_seed = []
    for index, seed in enumerate(seeds):
        samples, metrics, hook_counts = {}, {}, {}
        print(f"Generation seed {index + 1}/{args.repeats}: {seed}", flush=True)
        for arm in ARMS:
            model = models["A" if arm == "A" else "B"]
            torch.manual_seed(seed)
            if arm in offsets:
                with guided_sampling(model, offsets[arm]) as stats:
                    samples[arm] = ab.generate(model, count, batch, categories)
                if stats["hook_calls"] != math.ceil(count / batch) * config["sampling_steps"]:
                    raise RuntimeError(f"Wrong guidance hook count in arm {arm}")
                hook_counts[arm] = dict(stats)
            else:
                samples[arm] = ab.generate(model, count, batch, categories)
            key = "A" if arm == "A" else "B"
            check_model_unchanged(model, hashes[key], flags[key], arm)
            metrics[arm] = grouped_sample(samples[arm], train_codes, validation_codes, categories)
            if index == 0:
                location = args.base_run if arm in ("A", "B") else args.cd_run
                with np.load(location / f"{arm}_synthetic.npz", allow_pickle=False) as saved:
                    if not np.array_equal(samples[arm], saved["X_synthetic"]):
                        raise RuntimeError(f"Original-seed {arm} samples are not bitwise identical. "
                                           "Do not interpret further results; inspect checkpoint/source/environment lineage.")
                check_original_metrics(metrics[arm], original, arm)
        if index == 0:
            print("PASS: original-seed A/B/C/D samples and metrics reproduced exactly.", flush=True)
        shuffle_rows = []
        for shuffle_index in range(args.shuffle_repeats):
            rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_NAMESPACE, seed, shuffle_index]))
            shuffled = shuffle_categorical_columns(samples["C"], rng)
            shuffled_metrics = grouped_sample(shuffled, train_codes, validation_codes, categories)
            if shuffled_metrics["one_way_validation"] != metrics["C"]["one_way_validation"]:
                raise RuntimeError("Shuffle changed validation one-way errors")
            if shuffled_metrics["grouped_validation"]["age_outside_fraction"] != metrics["C"]["grouped_validation"]["age_outside_fraction"]:
                raise RuntimeError("Shuffle changed AGE support")
            shuffle_rows.append(shuffled_metrics["grouped_validation"])
        shuffle_mean = {key: float(np.mean([row[key] for row in shuffle_rows])) for key in GROUPS}
        c_grouped, b_grouped = metrics["C"]["grouped_validation"], metrics["B"]["grouped_validation"]
        per_seed.append({"repeat_index": index, "generation_seed": seed, "arms": metrics,
                         "guidance_hook_counts": hook_counts,
                         "paired_B_minus_C": {key: b_grouped[key] - c_grouped[key] for key in GROUPS},
                         "shuffle_control": {"repeats": args.shuffle_repeats,
                             "per_shuffle_grouped_validation": shuffle_rows,
                             "mean_grouped_validation": shuffle_mean,
                             "paired_shuffle_minus_C": {key: shuffle_mean[key] - c_grouped[key] for key in PAIR_GROUPS}}})
        print(f"Seed {index + 1} complete: C cat-pair L1={c_grouped['categorical_two_way']:.6f}; "
              f"shuffled C mean={shuffle_mean['categorical_two_way']:.6f}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    if cd.file_hash(target_path) != target_hash:
        raise RuntimeError("Original DP target cache changed during the experiment")
    for arm in models:
        check_model_unchanged(models[arm], hashes[arm], flags[arm], arm)
    report = {"purpose": "Conditional fixed-model sampling stability and categorical column-shuffle control",
        "formal_end_to_end_dp_claimed": False, "metrics_are_dp": False,
        "new_training_runs": 0, "new_dp_marginal_releases": 0,
        "generation_seeds": seeds, "seed_policy": "original evaluation_seed + 10000 * repeat_index; original observed seed retained for reproduction, additional seeds fixed before this run",
        "shuffle_repeats": args.shuffle_repeats,
        "shuffle_seed_policy": f"SeedSequence([{SHUFFLE_NAMESPACE}, generation_seed, shuffle_index])",
        "sampling_config": {"sampling_steps": config["sampling_steps"], "sample_batch": batch,
                            "reference_samples": reference_count, "reference_seed": original["reference_seed"],
                            "strength": original["strength"], "max_log_ratio": original["max_log_ratio"],
                            "probability_floor": original["probability_floor"], "count_pseudocount": original["count_pseudocount"]},
        "samples_per_arm": count, "threads": args.threads, "versions": versions,
        "privacy_inherited_from_cd": original["privacy"],
        "dp_target_cache_sha256": target_hash, "dp_target_cache_unchanged": True,
        "frozen_model_tensor_sha256": hashes, "original_seed_bitwise_reproduction": {arm: True for arm in ARMS},
        "input_schema_checkpoint_and_source_lineage_validated": True,
        "source_sha256": {"run_ipums_sampling_checks.py": cd.file_hash(Path(__file__)), **original["source_sha256"]},
        "input_run_metadata_sha256": {"base_metrics": cd.file_hash(args.base_run / "metrics.json"),
                                     "cd_metrics": cd.file_hash(args.cd_run / "metrics.json")},
        "aggregate": aggregate_results(per_seed), "per_seed": per_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": {"uncertainty": "Sample SD across generation seeds conditional on fixed A/B models, fixed DP target noise and fixed B reference pool. Not training/noise replication, SE, CI, or a significance test.",
            "paired_B_minus_C": "Positive means C has lower error. C includes an additional marginal privacy budget, so this is not fixed-total-budget comparison.",
            "paired_shuffle_minus_C": "For each generation seed, average its shuffles first; positive means C row associations help relative to independently shuffled categorical columns.",
            "shuffle_scope": "Exact empirical one-way preservation; AGE values/row order unchanged. Finite permutations retain chance associations. No proof of population or higher-order dependence preservation."},
        "limitations": ["One fixed training draw and one cached marginal-noise release.",
            "A/D/reference metrics remain NONPRIVATE; no end-to-end privacy claim.",
            "Validation has informed development; use a separate untouched final test set.",
            "Two-way diagnostics do not establish useful higher-order relationships.",
            "C2ST and downstream ML utility were NOT computed.",
            "No new raw/synthetic record dumps or model files are saved by this runner."]}
    write_outputs(args.out, report)
    print(f"Complete. Read {args.out / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
