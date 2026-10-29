#!/usr/bin/env python3
"""Bounded, nested Prototype-2 replication and relationship-diagnostic suite.

All outputs are LOCAL NONPRIVATE RESEARCH. This is not an end-to-end DP release.
Each training replicate independently fits matched A/clip-only/B controls and a
budget-matched E. Two independent marginal releases per B and repeated sampling
are NESTED within training replicate, never counted as independent model fits.
Old source/data/checkpoint lineage must validate before any new training.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_tabdiff_smoke as smoke
import run_ipums_ab_pilot as ab
import run_ipums_cd_pilot as cd
import run_ipums_training_checks as tc
import prototype2_replicated_training as training
import prototype2_relationship_checks as relationships
from prototype2_guidance_core import guided_sampling, make_offsets


DISPLAY_ARMS = ("A_diag", "clip_only", "B_private", "E_matched", "C")
MARGINAL_KEYS = ("all_one_way", "categorical_one_way", "age_one_way",
                 "all_two_way", "categorical_two_way", "age_two_way",
                 "age_outside_fraction")
RELATION_KEYS = ("mean_interaction_residual_l1", "mean_conditional_tv",
                 "mean_context_coverage", "mean_missing_context_mass",
                 "mean_context_marginal_tv")
CONTRASTS = {
    "B_private_minus_C": ("B_private", "C"),
    "E_matched_minus_C": ("E_matched", "C"),
    "clip_only_minus_A_diag": ("clip_only", "A_diag"),
    "B_private_minus_clip_only": ("B_private", "clip_only"),
}
SOURCE_NAMES = tc.HELPERS + ("run_ipums_training_checks.py",
    "prototype2_replicated_training.py", "prototype2_relationship_checks.py",
    "run_ipums_replication_suite.py")


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Statistics require a nonempty finite scalar sequence")
    return {"mean": float(array.mean()),
            "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "n": int(len(array)), "min": float(array.min()), "max": float(array.max())}


def scalar_statistics(rows):
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError("Every nested scalar row must have the same metric keys")
    return {key: stats([row[key] for row in rows]) for key in rows[0]}


def scalar_mean(rows):
    return {key: value["mean"] for key, value in scalar_statistics(rows).items()}


def difference(left, right):
    if set(left) != set(right):
        raise ValueError("Paired comparison keys differ")
    return {key: float(left[key] - right[key]) for key in left}


def scalar_metrics(sample, validation, validation_codes, categories):
    codes, age = ab.evaluation_codes(sample, categories)
    sizes = np.r_[len(ab.AGE_EDGES) + 1, categories]
    marginals = ab.compare_marginals(codes, validation_codes, sizes)
    grouped = cd.grouped_errors(marginals)
    grouped["age_outside_fraction"] = age["decoded_outside_0_100_fraction"]
    relation = relationships.evaluate_relationships(sample, validation, categories)
    flat = {"marginal/" + key: float(value) for key, value in grouped.items()}
    flat.update({"one_way/" + key: float(value) for key, value in marginals["one_way"].items()})
    flat.update({"two_way/" + key: float(value) for key, value in marginals["two_way"].items()})
    flat.update({"relationship/" + key: float(value) for key, value in relation["aggregate"].items()})
    for query, result in relation["interaction"].items():
        for key in ("residual_l1", "reference_residual_l1", "synthetic_residual_l1",
                    "independence_baseline_residual_l1"):
            flat[f"interaction/{query}/{key}"] = float(result[key])
    for query, result in relation["conditional"].items():
        for key in ("weighted_tv", "reference_weighted_context_coverage",
                    "missing_reference_context_mass", "context_marginal_tv",
                    "synthetic_mass_in_reference_absent_contexts"):
            flat[f"conditional/{query}/{key}"] = float(result[key])
    if not all(math.isfinite(value) for value in flat.values()):
        raise RuntimeError("Nonfinite evaluation scalar")
    return flat, {"marginals": marginals, "age": age, "relationships": relation}


def evaluate_with_shuffles(sample, validation, validation_codes, categories,
                           shuffle_repeats, shuffle_seed):
    original, details = scalar_metrics(sample, validation, validation_codes, categories)
    rng = np.random.default_rng(shuffle_seed)
    shuffled_rows = []
    for _ in range(shuffle_repeats):
        shuffled = relationships.shuffle_categorical_columns(sample, rng)
        shuffled_flat, _ = scalar_metrics(shuffled, validation, validation_codes, categories)
        for key in original:
            if key.startswith("one_way/") or key in (
                    "marginal/all_one_way", "marginal/categorical_one_way",
                    "marginal/age_one_way", "marginal/age_outside_fraction"):
                if shuffled_flat[key] != original[key]:
                    raise RuntimeError(f"Shuffle changed exact one-way/AGE metric: {key}")
        shuffled_rows.append(shuffled_flat)
    shuffled_mean = scalar_mean(shuffled_rows)
    return {"original": original, "shuffle_mean": shuffled_mean,
            "shuffle_minus_original": difference(shuffled_mean, original),
            "within_generation_shuffle_statistics": scalar_statistics(shuffled_rows),
            "details": details,
            "shuffle_repeats": shuffle_repeats, "shuffle_seed": shuffle_seed,
            "exact_empirical_one_way_and_AGE_preservation": True}


def collapse_generations(rows):
    """First collapse shuffles, then generation; neither creates another fit."""
    return {key: scalar_mean([row[key] for row in rows]) for key in
            ("original", "shuffle_mean", "shuffle_minus_original")}


def collapse_targets(rows):
    """Each DP-target draw has already averaged its own generation samples."""
    return {key: scalar_mean([row[key] for row in rows]) for key in
            ("original", "shuffle_mean", "shuffle_minus_original")}


def aggregate_replicates(replicates):
    """Main independent unit: one fresh training replicate, NOT every sample."""
    result = {"training_replicates": len(replicates), "arms": {}, "paired": {}}
    for arm in DISPLAY_ARMS:
        result["arms"][arm] = {
            key: scalar_statistics([rep["training_level"][arm][key] for rep in replicates])
            for key in ("original", "shuffle_mean", "shuffle_minus_original")}
    for name, (left, right) in CONTRASTS.items():
        paired = [difference(rep["training_level"][left]["original"],
                             rep["training_level"][right]["original"])
                  for rep in replicates]
        result["paired"][name] = scalar_statistics(paired)
    result["positive_E_minus_C_training_replicates"] = {
        key: sum(rep["training_level"]["E_matched"]["original"][key]
                 > rep["training_level"]["C"]["original"][key] for rep in replicates)
        for key in ("marginal/categorical_one_way", "marginal/categorical_two_way",
                    "relationship/mean_interaction_residual_l1",
                    "relationship/mean_conditional_tv")}
    return result


def format_stat(statistic, percent=False):
    scale = 100.0 if percent else 1.0
    sd = "NA" if statistic["sample_sd"] is None else f"{scale * statistic['sample_sd']:.6f}"
    return f"{scale * statistic['mean']:.6f} +/- {sd}"


def write_summary(path, report):
    plan, aggregate, replicates = report["plan"], report["aggregate"], report["replicates"]
    privacy = plan["privacy"]
    lines = ["NESTED INDEPENDENT-TRAINING / MARGINAL-NOISE / RELATIONSHIP PILOT",
        "LOCAL NONPRIVATE RESEARCH; NOT AN END-TO-END DP RELEASE.",
        f"Completed training replicates: {len(replicates)}/{plan['training_repeats']}; initialization seeds={plan['training_seeds']}.",
        f"Each: {plan['target_repeats']} independent DP-target releases; {plan['generation_repeats']} generations/arm/target; {plan['shuffle_repeats']} shuffles/sample.",
        "ALL headline SDs below are SAMPLE SD across TRAINING-LEVEL means (n=completed training replicates).",
        "First average shuffles within sample, generations within target, and targets within training replicate.",
        "C target draws reuse that replicate's SAME B; they are NOT independent trained models.",
        "Each C target is used separately: only evaluation scores are averaged, never the target probabilities or synthetic datasets.",
        "Repeated sampling adds no new DP mechanism. New training and new targets DO require composition.",
        f"Each B training epsilon={privacy['B_training_epsilon']:.9f}; each C variant nominal epsilon={privacy['C_variant_nominal_epsilon']:.9f}; E={privacy['E_achieved_epsilon']:.9f}; delta={plan['base_config']['delta']:g}.",
        f"Releasing all {plan['target_repeats']} target variants with one B instead has nominal epsilon={privacy['B_plus_all_targets_per_training_epsilon']:.9f} (not each-variant budget).",
        f"Planned fresh training mechanisms={privacy['planned_private_training_mechanisms']}; fresh target releases={privacy['planned_marginal_releases']}.",
        "No overall DP guarantee for this suite: nonprivate controls, validation, probes and preprocessing are excluded from these conditional mechanism budgets.",
        "", "VALIDATION ERRORS: MEAN +/- BETWEEN-TRAINING SAMPLE SD",
        "L1 range [0,2]; lower is better. AGE outside is PERCENT; its SD is percentage points.",
        "arm\tall_1way\tcat_1way\tAGE_1way\tall_2way\tcat_pairs\tAGE_pairs\tAGE_outside_%"]
    for arm in DISPLAY_ARMS:
        flat = aggregate["arms"][arm]["original"]
        lines.append(arm + "\t" + "\t".join(format_stat(flat["marginal/" + key],
            key == "age_outside_fraction") for key in MARGINAL_KEYS))
    lines += ["", "PAIRED CONTRASTS WITHIN TRAINING REPLICATE",
              "B-C and E-C: positive means conditioning has lower error.",
              "clip-A: positive means clipping hurts. B-clip: positive means added gradient noise hurts in matched trio.",
              "contrast\tcat_1way\tAGE_1way\tcat_pairs\tAGE_outside_pp"]
    for name, flat in aggregate["paired"].items():
        keys = ("categorical_one_way", "age_one_way", "categorical_two_way", "age_outside_fraction")
        lines.append(name + "\t" + "\t".join(format_stat(flat["marginal/" + key],
            key == "age_outside_fraction") for key in keys))
    lines += ["", "E-C POSITIVE COUNTS ACROSS TRAINING REPLICATES (descriptive, not a hypothesis test)"]
    lines += [f"{key}\t{count}/{len(replicates)}" for key, count in
              aggregate["positive_E_minus_C_training_replicates"].items()]
    lines += ["", "RELATIONSHIP METRICS AND THEIR OWN EXACT-MARGINAL SHUFFLE CONTROLS",
        "Interaction residual L1 compares p(x,y)-p(x)p(y); range [0,4]. Conditional TV range [0,1].",
        "Lower is better. Positive shuffle-minus-original means original associations help this metric.",
        "Conditional TV weights contexts by validation frequency; synthetic-missing positive contexts get penalty1, not exclusion.",
        "Neither metric is marginal-invariant or a standalone population/higher-order dependence certificate.",
        "arm\tmetric\toriginal\tshuffled\tpaired_shuffle_minus_original"]
    for arm in DISPLAY_ARMS:
        for key in ("mean_interaction_residual_l1", "mean_conditional_tv"):
            metric = "relationship/" + key
            cells = [format_stat(aggregate["arms"][arm][part][metric]) for part in
                     ("original", "shuffle_mean", "shuffle_minus_original")]
            lines.append(f"{arm}\t{key}\t" + "\t".join(cells))
    lines += ["", "FIXED RELATIONSHIP QUERIES: C VS E AND C'S SHUFFLE CONTROL",
              "query\tC_error\tE_error\tE_minus_C\tC_shuffle_minus_C"]
    for left, right in relationships.INTERACTION_PAIRS:
        query = left + " x " + right
        key = f"interaction/{query}/residual_l1"
        cells = [aggregate["arms"]["C"]["original"][key],
                 aggregate["arms"]["E_matched"]["original"][key],
                 aggregate["paired"]["E_matched_minus_C"][key],
                 aggregate["arms"]["C"]["shuffle_minus_original"][key]]
        lines.append(query + " (residual L1)\t" + "\t".join(format_stat(value) for value in cells))
    for left, right in relationships.CONDITIONAL_QUERIES:
        query = left + " -> " + right
        key = f"conditional/{query}/weighted_tv"
        cells = [aggregate["arms"]["C"]["original"][key],
                 aggregate["arms"]["E_matched"]["original"][key],
                 aggregate["paired"]["E_matched_minus_C"][key],
                 aggregate["arms"]["C"]["shuffle_minus_original"][key]]
        lines.append(query + " (conditional TV)\t" + "\t".join(format_stat(value) for value in cells))
    lines += ["", "CONDITIONAL CONTEXT COVERAGE (reference-weighted; higher is better)",
              "arm\tmean_context_coverage\tmean_context_marginal_TV"]
    for arm in DISPLAY_ARMS:
        flat = aggregate["arms"][arm]["original"]
        lines.append(arm + "\t" + format_stat(flat["relationship/mean_context_coverage"])
                     + "\t" + format_stat(flat["relationship/mean_context_marginal_tv"]))
    lines += ["", "WITHIN-TRAINING TARGET-NOISE VARIABILITY (SD over target-level generation means; not training n)",
              "training_seed\tC_cat1_mean\tC_cat1_target_SD\tC_catpair_mean\tC_catpair_target_SD"]
    for rep in replicates:
        variation = rep["C_between_target_statistics"]
        one, two = variation["marginal/categorical_one_way"], variation["marginal/categorical_two_way"]
        lines.append(f"{rep['training_seed']}\t{one['mean']:.6f}\t{one['sample_sd']}\t{two['mean']:.6f}\t{two['sample_sd']}")
    if plan["gradient_probe_rows"]:
        lines += ["", "GRADIENT SNAPSHOTS: means across training replicates of each 32-row (or requested count) probe summary.",
            "Snapshot fraction is NOT the fraction clipped throughout training. A_diag values are hypothetical threshold exceedances.",
            "model\tcat_median_norm\tnum_median_norm\tcombined_median_norm\tfraction_over_clip"]
        for name in ("initial",) + training.ARMS:
            values = [[rep["gradient_probes"][name][component][field] for rep in replicates]
                      for component, field in (("categorical", "median_norm"),
                          ("numeric", "median_norm"), ("combined", "median_norm"),
                          ("combined", "fraction_norm_exceeds_clip"))]
            lines.append(name + "\t" + "\t".join(f"{np.mean(value):.6f}" for value in values))
    lines += ["", "LIMITATIONS AND SCOPE:",
        "This small number of training replicates is preliminary; SD is descriptive, not a confidence interval or formal significance test.",
        "Training-level variation combines fresh initialization, Poisson, diffusion and Gaussian draws; C also averages the chosen number of target releases.",
        "A/clip/B isolate clipping and incremental noise within each matched trio. E uses independent Poisson/diffusion/noise, matched initialization and fixed settings.",
        "The B reference pool is fixed within each training replicate, and varies across replicates; reference-pool uncertainty is not separately isolated.",
        f"All generation seeds use {plan['samples_per_arm']} rows; sparse cells/rare contexts remain noisy. Consult per-query metrics, not only averages.",
        "The validation split has informed development. No new untouched test split or population-level estimate is claimed.",
        "No post-hoc AGE clipping, invalid-row filtering, best-seed selection, guidance tuning or loss reweighting is performed.",
        "Relationship queries were fixed before this suite, not held out from all prior exploratory work; do not claim confirmatory testing.",
        "C2ST, income-prediction utility, higher-order tests and full DP preprocessing/security audit are NOT performed.",
        "All checkpoints, diagnostics, exact reference counts, input hashes and outputs are LOCAL NONPRIVATE research. Do not commit/publish this folder."]
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def generate_checked(model, categories, previous_cd, seed, offsets=None):
    before = cd.model_hash(model)
    flags = [p.requires_grad for p in model.parameters()]
    model.eval()
    torch.manual_seed(seed)
    hook_stats = None
    if offsets is None:
        sample = ab.generate(model, previous_cd["samples_per_arm"],
                             previous_cd["sample_batch"], categories)
    else:
        with guided_sampling(model, offsets) as hooks:
            sample = ab.generate(model, previous_cd["samples_per_arm"],
                                 previous_cd["sample_batch"], categories)
        hook_stats = dict(hooks)
        expected = math.ceil(previous_cd["samples_per_arm"] / previous_cd["sample_batch"]) * model.num_timesteps
        if hook_stats["hook_calls"] != expected:
            raise RuntimeError("Guidance was not applied at every expected sampling step")
    if cd.model_hash(model) != before or flags != [p.requires_grad for p in model.parameters()]:
        raise RuntimeError("Sampling changed weights or trainable flags")
    return sample, hook_stats


def run_replicate(index, seed, plan, previous_cd, train, validation, validation_codes,
                   categories, schema, train_counts, oracle_targets, out):
    directory = out / f"replicate_{index + 1:02d}"
    directory.mkdir(exist_ok=False)
    config = copy.deepcopy(plan["base_config"])
    config["model_seed"] = seed
    print(f"\nTraining replicate {index + 1}/{plan['training_repeats']}; seed={seed}; four small models.", flush=True)
    torch.manual_seed(seed)
    initial = smoke.build_diffusion(categories, config["sampling_steps"])
    models, accountants, metadata = training.train_replicate(initial, train, config,
        plan["privacy"]["E_noise_multiplier"], directory, seed)
    epsilon_b = float(accountants["B_private"].get_epsilon(delta=config["delta"]))
    epsilon_e = float(accountants["E_matched"].get_epsilon(delta=config["delta"]))
    if (not math.isclose(epsilon_b, plan["privacy"]["B_training_epsilon"], abs_tol=1e-10)
            or not math.isclose(epsilon_e, plan["privacy"]["E_achieved_epsilon"], abs_tol=1e-10)):
        raise RuntimeError("Actual accountant does not match planned per-mechanism budget")
    cd.write_json(directory / "training_metadata.json", metadata)

    # Every final model is already saved. No evaluation chooses a checkpoint.
    final_hashes = {arm: cd.model_hash(model) for arm, model in models.items()}
    final_flags = {arm: [p.requires_grad for p in model.parameters()] for arm, model in models.items()}
    for model in models.values():
        model.eval()
    reference_seed = seed + 2000
    torch.manual_seed(reference_seed)
    print("Building this B model's fixed synthetic-only reference pool...", flush=True)
    reference_sample = ab.generate(models["B_private"], previous_cd["reference_samples"],
                                   previous_cd["sample_batch"], categories)
    reference_targets = [cd.normalize_counts(vector) for vector in cd.histograms(reference_sample, categories)]
    del reference_sample

    # Each release is independent and cached once. Its entire file is only DP
    # targets/public metadata; input hashes and oracle diagnostics live elsewhere.
    target_payloads, target_files, offsets, target_diagnostics = [], [], [], []
    for target_index in range(plan["target_repeats"]):
        payload = cd.create_dp_targets(train_counts, schema, plan["privacy"]["epsilon_marginals_per_release"])
        target = cd.validate_dp_targets(payload, schema, plan["privacy"]["epsilon_marginals_per_release"])
        target_file = directory / f"dp_targets_{target_index + 1:02d}.json"
        cd.write_json(target_file, payload)
        target_payloads.append(payload)
        target_files.append((target_file, cd.file_hash(target_file)))
        offsets.append(make_offsets(target, reference_targets, previous_cd["strength"],
                                    previous_cd["max_log_ratio"], cd.PROB_FLOOR))
        target_diagnostics.append({name: float(np.abs(estimated - truth).sum())
            for name, estimated, truth in zip(cd.TARGETS, target, oracle_targets)})
    cd.write_json(directory / "private_target_provenance.json", {
        "NONPRIVATE_LOCAL_ONLY": True, "input_npz_sha256": plan["input_npz_sha256"],
        "schema_sha256": plan["schema_sha256"], "B_checkpoint_sha256": cd.file_hash(directory / "B_private_final.pt"),
        "target_file_sha256": {path.name: digest for path, digest in target_files},
        "target_l1_vs_smoothed_training_oracle": target_diagnostics,
        "reference_seed": reference_seed, "reference_samples": previous_cd["reference_samples"]})
    print(f"Cached {len(target_files)} NEW independent marginal releases; no target draws will be selected or discarded.", flush=True)

    generation_seeds = [seed + 1000 + 100 * repeat for repeat in range(plan["generation_repeats"])]
    records = {arm: [] for arm in training.ARMS}
    c_records = [[] for _ in range(plan["target_repeats"])]
    for generation_index, generation_seed in enumerate(generation_seeds):
        print(f"Replicate {index + 1}: generation {generation_index + 1}/{len(generation_seeds)}, all arms and target variants...", flush=True)
        for arm_index, arm in enumerate(training.ARMS):
            sample, _ = generate_checked(models[arm], categories, previous_cd, generation_seed)
            if arm == "B_private" and generation_index == 0:
                zeros = [np.zeros(int(k), dtype=np.float64) for k in categories]
                replay, _ = generate_checked(models[arm], categories, previous_cd, generation_seed, zeros)
                if not np.array_equal(sample, replay):
                    raise RuntimeError("Zero guidance did not reproduce B exactly")
            result = evaluate_with_shuffles(sample, validation, validation_codes, categories,
                plan["shuffle_repeats"], generation_seed + 1_000_000 + 10000 * arm_index)
            result["generation_seed"] = generation_seed
            records[arm].append(result)
        for target_index, target_offsets in enumerate(offsets):
            sample, hook_stats = generate_checked(models["B_private"], categories,
                previous_cd, generation_seed, target_offsets)
            if any(np.any(value) for value in target_offsets) and hook_stats["masked_cells_adjusted"] == 0:
                raise RuntimeError("Nonzero C guidance never reached a masked categorical cell")
            result = evaluate_with_shuffles(sample, validation, validation_codes, categories,
                plan["shuffle_repeats"], generation_seed + 2_000_000 + 10000 * target_index)
            result.update({"generation_seed": generation_seed,
                           "target_index": target_index + 1, "guidance_hook_stats": hook_stats})
            c_records[target_index].append(result)
    per_target = [collapse_generations(rows) for rows in c_records]
    training_level = {arm: collapse_generations(rows) for arm, rows in records.items()}
    training_level["C"] = collapse_targets(per_target)
    gradient_probes = {}
    if plan["gradient_probe_rows"]:
        probe_rows = train[:plan["gradient_probe_rows"]]
        for name, model in {"initial": initial, **models}.items():
            gradient_probes[name] = tc.gradient_probe(model, probe_rows, seed + 3000, config["clip"])
    for arm, model in models.items():
        if cd.model_hash(model) != final_hashes[arm] or [p.requires_grad for p in model.parameters()] != final_flags[arm]:
            raise RuntimeError("Evaluation/probes changed final model state")
    if any(cd.file_hash(path) != digest for path, digest in target_files):
        raise RuntimeError("A cached DP target changed during evaluation")
    report = {"training_seed": seed, "training_metadata": metadata,
        "generation_seeds": generation_seeds, "training_level": training_level,
        "C_between_target_statistics": scalar_statistics([item["original"] for item in per_target]),
        "C_target_level": per_target, "gradient_probes": gradient_probes,
        "generation_variability": {arm: scalar_statistics([row["original"] for row in rows])
                                  for arm, rows in records.items()},
        "C_generation_variability_per_target": [scalar_statistics([row["original"] for row in rows]) for rows in c_records],
        "evaluations": records, "C_evaluations_per_target": c_records,
        "target_l1_vs_smoothed_training_oracle": target_diagnostics,
        "checks": {"zero_guidance_B_exact_reproduction": True, "final_models_unchanged": True,
                   "target_caches_unchanged": True, "matched_A_clip_B_training": True,
                   "exact_shuffle_one_way_preservation": True},
        "privacy_warning": "NONPRIVATE research. Every independently drawn target set is a new marginal release; variants are not independent trained models."}
    cd.write_json(directory / "metrics.json", report)
    return report


def self_test_aggregation():
    # Deliberately unbalanced inner counts: equal weight per nested target,
    # NOT per generated row. Production uses balanced counts, but this catches
    # future flattening and accidental pseudoreplication.
    row = lambda value: {part: {"x": float(value)} for part in
                         ("original", "shuffle_mean", "shuffle_minus_original")}
    target_a = collapse_generations([row(0), row(0), row(0)])
    target_b = collapse_generations([row(10)])
    combined = collapse_targets([target_a, target_b])
    if combined["original"]["x"] != 5.0:
        raise AssertionError("Nested target weighting was flattened")
    check = stats([1, 2, 3])
    if check != {"mean": 2.0, "sample_sd": 1.0, "n": 3, "min": 1.0, "max": 3.0}:
        raise AssertionError("Training sample SD is not ddof=1")
    if stats([1])["sample_sd"] is not None:
        raise AssertionError("A single training fit cannot supply between-fit SD")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cd-run", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW directory; never overwrite or automatically rerun a partial experiment")
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--target-repeats", type=int, default=2)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--training-seed-base", type=int, default=11026)
    parser.add_argument("--gradient-probe-rows", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true", help="Read-only input/source/accounting/self-test preflight; no output, training or marginal release")
    args = parser.parse_args()
    for name, upper in (("training_repeats", 5), ("target_repeats", 3),
                        ("generation_repeats", 5), ("shuffle_repeats", 20), ("threads", 4)):
        if not 1 <= getattr(args, name) <= upper:
            parser.error(f"{name.replace('_', '-')} must be between 1 and {upper}")
    if not 0 <= args.gradient_probe_rows <= 128 or not 0 <= args.training_seed_base < 2**31:
        parser.error("Use 0..128 probe rows and a nonnegative initialization seed below 2**31")
    if args.out.exists():
        parser.error("Output already exists. Preserve it; do not delete/retry a partial private experiment automatically.")
    torch.set_num_threads(args.threads)
    self_test_aggregation()
    relationships.self_test()
    base, previous_cd, config, _, train, validation, categories, schema = tc.load_validated_context(args)
    if config["steps"] > 2000 or config["batch_size"] > 32 or len(train) > 16384:
        parser.error("This CPU pilot is capped at 2000 updates, expected batch32, and 16384 training rows")
    if (previous_cd["samples_per_arm"] > 1024 or previous_cd["reference_samples"] > 4096
            or previous_cd["sample_batch"] > 64 or args.gradient_probe_rows > len(train)):
        parser.error("CPU bounds require samples<=1024, reference<=4096, chunk<=64, probes<=train rows")
    epsilon_b = float(previous_cd["privacy"]["B_training_epsilon"])
    marginal_epsilon = float(previous_cd["privacy"]["C_marginal_epsilon"])
    total_target = float(previous_cd["privacy"]["C_nominal_composed_epsilon"])
    sigma_e, predicted_e = tc.calibrate_training_noise(total_target, config["delta"], config["q"], config["steps"])
    if previous_cd["strength"] != 1.0 or previous_cd["max_log_ratio"] != 2.0:
        parser.error("This predeclared suite requires original guidance strength1 and log-ratio cap2")
    seeds = [args.training_seed_base + 10000 * index for index in range(args.training_repeats)]
    scripts = Path(__file__).resolve().parent
    plan = {"format_version": 1, "purpose": "Predeclared nested replication plus matched clipping/noise and relationship diagnostics",
        "NONPRIVATE_LOCAL_RESEARCH": True, "formal_end_to_end_dp_claimed": False,
        "base_config": config, "training_seeds": seeds,
        "training_repeats": args.training_repeats, "target_repeats": args.target_repeats,
        "generation_repeats": args.generation_repeats, "shuffle_repeats": args.shuffle_repeats,
        "gradient_probe_rows": args.gradient_probe_rows, "threads": args.threads,
        "samples_per_arm": previous_cd["samples_per_arm"], "reference_samples": previous_cd["reference_samples"],
        "guidance": {"strength": 1.0, "max_log_ratio": 2.0, "features": cd.TARGETS,
                     "target_mechanism": "independent six-histogram Laplace release per target replicate"},
        "relationship_queries": {"interaction": relationships.INTERACTION_PAIRS,
                                 "conditional": relationships.CONDITIONAL_QUERIES},
        "privacy": {"B_training_epsilon": epsilon_b,
            "epsilon_marginals_per_release": marginal_epsilon,
            "C_variant_nominal_epsilon": epsilon_b + marginal_epsilon,
            "E_target_epsilon": total_target, "E_achieved_epsilon": predicted_e,
            "E_noise_multiplier": sigma_e,
            "B_plus_all_targets_per_training_epsilon": epsilon_b + args.target_repeats * marginal_epsilon,
            "planned_private_training_mechanisms": 2 * args.training_repeats,
            "planned_marginal_releases": args.target_repeats * args.training_repeats,
            "delta_per_private_training": config["delta"],
            "scope": "Conditional single-mechanism/add-remove-record accounting only. Multiple training and target releases require composition; the whole research output is NONPRIVATE."},
        "new_initializations_not_original_model_replays": True,
        "aggregation": "shuffle -> generation -> target -> training; main SD over training means only; E is not duplicated as independent evidence for each C target",
        "selection_policy": "All requested seeds/target draws kept; final update checkpoints only; no metric-based stopping or best-of selection",
        "input_npz_sha256": cd.file_hash(args.data_npz), "schema_sha256": cd.file_hash(args.schema),
        "source_sha256": {name: cd.file_hash(scripts / name) for name in SOURCE_NAMES},
        "tabdiff_source_sha256": previous_cd["tabdiff_source_sha256"],
        "versions": previous_cd["versions"],
        "original_AB_metrics_sha256": cd.file_hash(args.base_run / "metrics.json"),
        "original_CD_metrics_sha256": cd.file_hash(args.cd_run / "metrics.json")}
    print(f"Preflight PASS: {len(seeds)} training replicates, {args.target_repeats} target draws each, {args.generation_repeats} generation draws.", flush=True)
    print(f"Per-variant C nominal epsilon={total_target:.6f}; E achieved={predicted_e:.6f}; E sigma={sigma_e:.9f}.", flush=True)
    if args.validate_only:
        print("VALIDATION ONLY: no training, marginal noise, output files, or privacy mechanism was run.", flush=True)
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    cd.write_json(args.out / "predeclared_plan.json", plan)
    train_counts = cd.histograms(train.numpy(), categories)
    oracle_targets = [cd.normalize_counts(vector) for vector in train_counts]
    validation_codes, _ = ab.evaluation_codes(validation, categories, real_records=True)
    started = time.perf_counter()
    report = {"plan": plan, "replicates": [], "aggregate": None}
    for index, seed in enumerate(seeds):
        replicate = run_replicate(index, seed, plan, previous_cd, train, validation,
            validation_codes, categories, schema, train_counts, oracle_targets, args.out)
        report["replicates"].append(replicate)
        report["aggregate"] = aggregate_replicates(report["replicates"])
        report["elapsed_seconds"] = time.perf_counter() - started
        write_summary(args.out / f"summary_after_{index + 1:02d}.txt", report)
        print(f"Completed replicate {index + 1}/{len(seeds)}. Partial summary saved; all completed fits retained.", flush=True)
    initial_hashes = [rep["training_metadata"]["initial_state_sha256"] for rep in report["replicates"]]
    if len(set(initial_hashes)) != len(initial_hashes):
        raise RuntimeError("Independent initialization seeds unexpectedly produced duplicate model hashes")
    if plan["source_sha256"] != {name: cd.file_hash(scripts / name) for name in SOURCE_NAMES}:
        raise RuntimeError("Source changed during experiment; keep partial artifacts but do not interpret pooled results")
    if any(cd.file_hash(scripts.parent / name) != digest
           for name, digest in plan["tabdiff_source_sha256"].items()):
        raise RuntimeError("TabDiff source changed during experiment")
    if plan["input_npz_sha256"] != cd.file_hash(args.data_npz) or plan["schema_sha256"] != cd.file_hash(args.schema):
        raise RuntimeError("Input changed during experiment")
    cd.write_json(args.out / "metrics.json", report)
    write_summary(args.out / "summary.txt", report)
    print(f"Complete. Read {args.out / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
