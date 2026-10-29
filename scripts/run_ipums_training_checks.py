#!/usr/bin/env python3
"""Bounded, local training diagnostics for the existing IPUMS A/B/C/D pilot.

E_matched spends C's nominal model-plus-marginal budget on training alone.
A_diag and clip_only form a NEW matched, NONPRIVATE clipping ablation.
This script never queries or releases another set of DP marginal targets.

Every reference-data metric and every gradient diagnostic is NONPRIVATE.
One training run per arm plus repeated sampling is NOT independent training
replication. Only load B_final.pt from your OWN locally generated run: the old
accountant checkpoint requires trusted pickle, not weights-only loading.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import secrets
import sys
import time
import warnings

import numpy as np
import opacus
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_tabdiff_smoke as smoke
import run_ipums_ab_pilot as ab
import run_ipums_cd_pilot as cd


EPSILON_TOLERANCE = 1e-4
ARMS = ("E_matched", "A_diag", "clip_only")
HELPERS = ("local_tabdiff_smoke.py", "run_ipums_ab_pilot.py",
           "prototype2_guidance_core.py", "run_ipums_cd_pilot.py")
TABDIFF_SOURCES = ("tabdiff/models/unified_ctime_diffusion.py",
                   "tabdiff/models/noise_schedule.py",
                   "tabdiff/modules/main_modules.py", "tabdiff/modules/transformer.py")
BASE_ARTIFACTS = ("B_final.pt", "B_synthetic.npz", "A_synthetic.npz",
                  "metrics.json", "public_schema.json")
GROUPED_KEYS = ("all_one_way", "categorical_one_way", "age_one_way",
                "all_two_way", "categorical_two_way", "age_two_way",
                "age_outside_0_100_fraction")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_hashes(recorded, directory, names, label):
    if set(recorded) != set(names):
        raise ValueError(f"Missing or unexpected {label} provenance entries")
    for name in names:
        if cd.file_hash(directory / name) != recorded[name]:
            raise ValueError(f"{label} changed since C/D: {name}. Do not bypass lineage checks.")


def load_validated_context(args):
    """Validate this exact local lineage before any new private mechanism runs."""
    base = read_json(args.base_run / "metrics.json")
    previous_cd = read_json(args.cd_run / "metrics.json")
    provenance = read_json(args.cd_run / "private_provenance.json")
    config = base["config"]
    if previous_cd.get("base_config") != config:
        raise ValueError("C/D and A/B configurations differ")
    if config.get("architecture") != cd.EXPECTED_ARCHITECTURE:
        raise ValueError("Only the unchanged tiny A/B architecture is supported")
    if config.get("fixed_schedules") is not True or config.get("features") != ab.FEATURES:
        raise ValueError("Unexpected schedules or feature order")
    if not previous_cd.get("baseline_reproduction_bitwise") or not previous_cd.get("model_weights_unchanged"):
        raise ValueError("C/D must have passed its baseline and weight-preservation checks")
    scripts = Path(__file__).resolve().parent
    require_hashes(previous_cd["source_sha256"], scripts, HELPERS, "Helper source")
    require_hashes(previous_cd["tabdiff_source_sha256"], scripts.parent,
                   TABDIFF_SOURCES, "TabDiff source")
    require_hashes(previous_cd["base_artifact_sha256"], args.base_run,
                   BASE_ARTIFACTS, "Original A/B artifact")
    current_versions = {"torch": str(torch.__version__), "numpy": np.__version__,
                        "opacus": opacus.__version__}
    if previous_cd.get("versions") != current_versions:
        raise ValueError("Package versions differ from the C/D run; use its original environment")
    for key, path in (("input_npz_sha256", args.data_npz), ("schema_sha256", args.schema),
                      ("checkpoint_sha256", args.base_run / "B_final.pt")):
        if provenance.get(key) != cd.file_hash(path):
            raise ValueError(f"Current input/checkpoint does not match C/D provenance: {key}")
    train, validation, categories, schema = ab.load_inputs(argparse.Namespace(
        data_npz=args.data_npz, schema=args.schema, batch_size=config["batch_size"]))
    if (categories.tolist() != config["categories"] or len(train) != base["n_train"]
            or len(validation) != base["n_holdout"]
            or len(train) != previous_cd["n_train"]
            or len(validation) != previous_cd["n_validation"]):
        raise ValueError("Input row counts or category cardinalities changed")
    public_schema = read_json(args.base_run / "public_schema.json")
    if any(schema[key] != public_schema[key]
           for key in ("features", "category_domains", "age_transform")):
        raise ValueError("Public schema/code ordering differs from the A/B run")
    if not math.isclose(config["q"], config["batch_size"] / len(train), abs_tol=1e-15):
        raise ValueError("Original sampling rate does not match fixed batch/data sizes")
    if (int(config["steps"]) != config["steps"] or config["steps"] < 1
            or not 0 < config["q"] <= 1 or not 0 < config["delta"] < 1):
        raise ValueError("Invalid original step count/sampling rate/delta")
    if int(previous_cd["evaluation_seed"]) != int(config["model_seed"]) + 1000:
        raise ValueError("Unexpected original evaluation sampling seed")
    if (previous_cd["samples_per_arm"] != base["n_synthetic"]
            or previous_cd["sample_batch"] < 1):
        raise ValueError("Original synthetic sample sizes disagree")
    print("Loading your OWN trusted B_final.pt (never use an untrusted checkpoint).", flush=True)
    checkpoint = torch.load(args.base_run / "B_final.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError("B checkpoint configuration differs from A/B metadata")
    accountant = RDPAccountant()
    accountant.load_state_dict(checkpoint["accountant"])
    if sum(item[2] for item in accountant.history) != config["steps"]:
        raise ValueError("Original B accountant step count mismatch")
    if any(not math.isclose(float(sigma), config["sigma"], rel_tol=1e-12)
           or not math.isclose(float(q), config["q"], rel_tol=1e-12)
           for sigma, q, _ in accountant.history):
        raise ValueError("Original B accountant settings mismatch")
    epsilon_b = float(accountant.get_epsilon(delta=config["delta"]))
    privacy = previous_cd["privacy"]
    if (not math.isclose(epsilon_b, base["training_privacy_accounting"]["epsilon"], rel_tol=1e-10)
            or not math.isclose(epsilon_b, privacy["B_training_epsilon"], rel_tol=1e-10)
            or privacy["delta"] != config["delta"]):
        raise ValueError("Original B privacy accounting disagrees across artifacts")
    target = float(privacy["C_nominal_composed_epsilon"])
    if (not math.isfinite(target) or target <= epsilon_b
            or not math.isclose(target, epsilon_b + privacy["C_marginal_epsilon"], rel_tol=1e-12)):
        raise ValueError("Invalid original C nominal composed budget")
    return base, previous_cd, config, checkpoint, train, validation, categories, schema


def calibrate_training_noise(target, delta, q, steps):
    # Very large trial sigma values used only to bracket the binary search can
    # trigger this warning. Keep warnings for the achieved/final bounds visible.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Optimal order is the largest alpha\..*",
                                category=UserWarning, module=r"opacus\.accountants\.analysis\.rdp")
        sigma = float(get_noise_multiplier(target_epsilon=target, target_delta=delta,
                        sample_rate=q, steps=steps, accountant="rdp",
                        epsilon_tolerance=EPSILON_TOLERANCE))
    predicted = RDPAccountant()
    predicted.history = [(sigma, q, steps)]
    achieved = float(predicted.get_epsilon(delta=delta))
    if (not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(achieved)
            or achieved > target or target - achieved > EPSILON_TOLERANCE):
        raise RuntimeError("Noise calibration did not meet the requested <=target, 1e-4 tolerance")
    return sigma, achieved


def build_model(categories, config, state=None):
    model = smoke.build_diffusion(categories, config["sampling_steps"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Model checkpoint contains nonfinite tensors")
    return model


def train_arms(initial, train, config, sigma, out):
    """A_diag/clip_only share all draws; E has its own Poisson/noise generators."""
    smoke.CLIP = float(config["clip"])
    models = {arm: copy.deepcopy(initial) for arm in ARMS}
    initial_hash = cd.model_hash(initial)
    if any(cd.model_hash(model) != initial_hash for model in models.values()):
        raise RuntimeError("Initial weights do not match across diagnostic arms")
    optimizers = {}
    for arm, model in models.items():
        base_optimizer = torch.optim.AdamW([p for _, p in smoke.trainable_parameters(model)],
                         lr=config["lr"], weight_decay=0)
        if arm == "A_diag":
            optimizers[arm] = base_optimizer
        else:
            optimizers[arm] = DPOptimizer(base_optimizer,
                noise_multiplier=sigma if arm == "E_matched" else 0.0,
                max_grad_norm=config["clip"], expected_batch_size=config["batch_size"],
                loss_reduction="mean", secure_mode=False,
                generator=torch.Generator().manual_seed(secrets.randbits(63)))
    accountant = RDPAccountant()
    optimizers["E_matched"].attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=config["q"]))
    # Never attach a privacy accountant to the zero-noise clipping control.
    e_poisson = torch.Generator().manual_seed(secrets.randbits(63))
    pair_poisson = torch.Generator().manual_seed(secrets.randbits(63))
    empty = {"E_matched": 0, "A_diag_and_clip_only": 0}
    history = []
    started = time.perf_counter()
    for step in range(1, int(config["steps"]) + 1):
        e_batch = train[torch.rand(len(train), generator=e_poisson) < config["q"]]
        empty["E_matched"] += int(len(e_batch) == 0)
        smoke.private_update(models["E_matched"], optimizers["E_matched"], e_batch)
        pair_batch = train[torch.rand(len(train), generator=pair_poisson) < config["q"]]
        empty["A_diag_and_clip_only"] += int(len(pair_batch) == 0)
        before_a = torch.get_rng_state()
        smoke.nonprivate_update(models["A_diag"], optimizers["A_diag"], pair_batch,
                                config["batch_size"])
        after_a = torch.get_rng_state()
        torch.set_rng_state(before_a)
        smoke.private_update(models["clip_only"], optimizers["clip_only"], pair_batch)
        if not torch.equal(torch.get_rng_state(), after_a):
            raise RuntimeError("A_diag and clip_only did not consume identical diffusion RNG draws")
        if any(not torch.isfinite(p).all() for p in models["A_diag"].parameters()):
            raise RuntimeError("A_diag developed nonfinite parameters")
        if step == 1 or step % 50 == 0 or step == config["steps"]:
            epsilon = float(accountant.get_epsilon(delta=config["delta"]))
            if not math.isfinite(epsilon):
                raise RuntimeError("Nonfinite E training accountant")
            history.append({"step": step, "epsilon": epsilon})
            print(f"Update {step}/{config['steps']}; E epsilon={epsilon:.6f}; "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    if sum(item[2] for item in accountant.history) != config["steps"]:
        raise RuntimeError("E privacy accountant step count mismatch")
    # Predetermined final models are saved before any evaluation or probes.
    for arm, model in models.items():
        arm_config = copy.deepcopy(config)
        arm_config.update({"sigma": sigma if arm == "E_matched" else 0.0,
                           "clip": config["clip"] if arm != "A_diag" else None,
                           "arm": arm})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizers[arm].state_dict(),
                      "config": arm_config,
                      "privacy_status": "training accountant only; NOT end-to-end DP" if arm == "E_matched"
                                        else "NONPRIVATE; no finite DP claim"}
        if arm == "E_matched":
            checkpoint["accountant"] = accountant.state_dict()
        torch.save(checkpoint, out / f"{arm}_final.pt")
    return models, accountant, {"seconds": elapsed, "history": history,
                               "empty_poisson_updates": empty,
                               "initial_state_sha256": initial_hash,
                               "A_diag_clip_only_diffusion_draws_matched": True}


def gradient_probe(model, records, seed, clip):
    """NONPRIVATE, per-record component gradients; never retain a gradient stack.

    mixed_loss's first output is CATEGORICAL d_loss, second NUMERIC c_loss.
    Global clipping acts on the sum of their gradients, not each separately.
    """
    weights = cd.model_hash(model)
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    params = [p for _, p in smoke.trainable_parameters(model)]
    values = {"categorical": [], "numeric": [], "combined": []}
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.train()
            for row in records:
                d_loss, c_loss = model.mixed_loss(row[None, :].float())
                if not torch.isfinite(d_loss).all() or not torch.isfinite(c_loss).all():
                    raise RuntimeError("Nonfinite gradient-probe loss")
                cat = torch.autograd.grad(d_loss, params, allow_unused=True,
                                         retain_graph=True) if d_loss.requires_grad else [None] * len(params)
                num = torch.autograd.grad(c_loss, params, allow_unused=True) if c_loss.requires_grad else [None] * len(params)
                sums = {key: 0.0 for key in values}
                for categorical, numeric in zip(cat, num):
                    if categorical is None and numeric is None:
                        raise RuntimeError("A trainable parameter has no gradient from either loss component")
                    if categorical is not None:
                        if not torch.isfinite(categorical).all():
                            raise RuntimeError("Nonfinite categorical diagnostic gradient")
                        sums["categorical"] += float(categorical.double().square().sum())
                    if numeric is not None:
                        if not torch.isfinite(numeric).all():
                            raise RuntimeError("Nonfinite numerical diagnostic gradient")
                        sums["numeric"] += float(numeric.double().square().sum())
                    if categorical is not None and numeric is not None:
                        sums["combined"] += float((categorical.double() + numeric.double()).square().sum())
                    elif categorical is not None:
                        sums["combined"] += float(categorical.double().square().sum())
                    elif numeric is not None:
                        sums["combined"] += float(numeric.double().square().sum())
                for name in values:
                    values[name].append(math.sqrt(sums[name]))
    finally:
        model.train(was_training)
    if cd.model_hash(model) != weights or flags != [p.requires_grad for p in model.parameters()]:
        raise RuntimeError("Diagnostic probe changed model tensors or requires_grad flags")
    summary = {name: {"median_norm": float(np.median(norms)),
                       "p95_norm": float(np.percentile(norms, 95))}
               for name, norms in values.items()}
    summary["combined"]["fraction_norm_exceeds_clip"] = float(np.mean(np.asarray(values["combined"]) > clip))
    return summary


def repeated_evaluation(models, train, validation, categories, previous_cd, repeats):
    train_codes, _ = ab.evaluation_codes(train.numpy(), categories, real_records=True)
    validation_codes, _ = ab.evaluation_codes(validation, categories, real_records=True)
    seeds = [int(previous_cd["evaluation_seed"]) + 10000 * i for i in range(repeats)]
    per_repeat = {arm: [] for arm in ARMS}
    hashes = {arm: cd.model_hash(model) for arm, model in models.items()}
    started = time.perf_counter()
    for index, seed in enumerate(seeds):
        for arm, model in models.items():
            print(f"Sampling {arm}: generation repeat {index+1}/{repeats}, seed={seed}", flush=True)
            torch.manual_seed(seed)
            sample = ab.generate(model, previous_cd["samples_per_arm"],
                                 previous_cd["sample_batch"], categories)
            result = cd.compare_sample(sample, train_codes, validation_codes, categories)
            grouped = cd.grouped_errors(result["validation"])
            grouped["age_outside_0_100_fraction"] = result["age"]["decoded_outside_0_100_fraction"]
            per_repeat[arm].append({"seed": seed, "validation_grouped": grouped,
                                    "results": result})
            if cd.model_hash(model) != hashes[arm]:
                raise RuntimeError(f"Sampling changed {arm} weights")
    aggregate = {}
    for arm, rows in per_repeat.items():
        aggregate[arm] = {}
        for key in GROUPED_KEYS:
            numbers = np.array([row["validation_grouped"][key] for row in rows])
            aggregate[arm][key] = {"mean": float(numbers.mean()),
                                   "sampling_sd": float(numbers.std(ddof=1)) if repeats > 1 else None}
    return {"generation_seeds": seeds, "generation_repeats": repeats,
            "independent_training_runs_per_arm": 1,
            "interpretation": "Mean/sample-SD describe generation randomness conditional on ONE fitted model per arm; not training-replicate uncertainty.",
            "aggregate_validation": aggregate, "per_repeat": per_repeat,
            "model_hashes_after_sampling": hashes, "seconds": time.perf_counter() - started}


def write_summary(out, report):
    privacy = report["privacy"]
    config = report["original_training_config"]
    evaluation = report["evaluation"]
    lines = ["PRELIMINARY TRAINING CHECKS — NOT AN END-TO-END DP RELEASE",
        "E_matched: NEW private training-only model at original C's NOMINAL total budget.",
        "A_diag vs clip_only: NEW matched NONPRIVATE controls; only gradient clipping differs.",
        "Each arm trained once. Repeated generation does NOT supply independent training replicates.",
        f"Fixed updates={config['steps']}; expected batch={config['batch_size']}; q={config['q']:.8f}; C={config['clip']}",
        f"E target epsilon={privacy['E_target_nominal_epsilon']:.9f}; achieved epsilon={privacy['E_achieved_training_epsilon']:.9f}; delta={config['delta']:g}",
        f"E calibrated sigma={privacy['E_noise_multiplier']:.12f}; target minus achieved={privacy['E_budget_shortfall']:.9f} (<=1e-4)",
        "These are PER-MECHANISM budgets, not a reset or total for all previous research runs.",
        "clip_only has noise multiplier ZERO and is NONPRIVATE, NOT epsilon=0.",
        "No new DP marginal target release was made.",
        "All reference metrics and gradient probes below are NONPRIVATE; keep this folder local.",
        "", "VALIDATION MEAN +/- SAMPLING SD (L1 range [0,2]; AGE outside is PERCENT, its SD is percentage points)",
        "arm\tall_1way\tcat_1way\tAGE_1way\tall_2way\tcat_pairs\tAGE_pairs\tAGE_outside_[0,100]_%"]
    for arm in ARMS:
        cells = []
        for key in GROUPED_KEYS:
            statistic = evaluation["aggregate_validation"][arm][key]
            scale = 100.0 if key == "age_outside_0_100_fraction" else 1.0
            sd = "NA" if statistic["sampling_sd"] is None else f"{scale * statistic['sampling_sd']:.6f}"
            cells.append(f"{scale * statistic['mean']:.6f} +/- {sd}")
        lines.append(arm + "\t" + "\t".join(cells))
    lines += ["", "ORIGINAL C SINGLE-SEED REFERENCE (not a multi-seed mean)",
              "Compare its sampling-checks summary with the E multi-seed mean above for equal seed lists.",
              "For a same-generation-seed descriptive comparison only:",
              "arm\tseed\tall_1way\tcat_1way\tAGE_1way\tall_2way\tcat_pairs\tAGE_pairs\tAGE_outside_[0,100]_%"]
    for arm, grouped in report["same_generation_seed_reference"].items():
        lines.append(arm + f"\t{evaluation['generation_seeds'][0]}\t" +
                     "\t".join(f"{grouped[key] * (100 if key == 'age_outside_0_100_fraction' else 1):.6f}"
                               for key in GROUPED_KEYS))
    probes = report["gradient_probes"]
    lines += ["", "GRADIENT PROBES: NONPRIVATE, one matched diffusion draw per predetermined training row.",
              "d_loss is categorical; c_loss is numeric. Combined norm is norm(g_cat+g_num), NOT their norm sum.",
              "Clipping fraction is at each inspected model snapshot, not a logged whole-training clipping rate."]
    if probes["rows"]:
        lines.append("model\tcat_median\tcat_p95\tnum_median\tnum_p95\tcombined_median\tcombined_p95\tcombined_fraction_over_C")
        for name, result in probes["models"].items():
            numbers = [result[component][stat] for component in ("categorical", "numeric", "combined")
                       for stat in ("median_norm", "p95_norm")]
            numbers.append(result["combined"]["fraction_norm_exceeds_clip"])
            lines.append(name + "\t" + "\t".join(f"{number:.6f}" for number in numbers))
    else:
        lines.append("Skipped (--gradient-probe-rows 0).")
    lines += ["", "LIMITATIONS:",
        "A_diag/clip_only isolate clipping within their matched pair. Comparing clip_only with OLD B does not isolate noise exactly: their batch/training randomness differs.",
        "E matches C's nominal budget, not its trained weights or training random draw; conclusions need independent training replicates.",
        "No loss/metric-based stopping, checkpoint selection, age clipping, or invalid-sample filtering was used.",
        "Do not infer that C is significantly better/worse from one model per arm or compare a single C draw as if it were a multi-seed mean.",
        "The existing validation split has informed development; final research needs a fresh untouched test set.",
        "Do not commit this folder, input data, model checkpoints, or nonprivate diagnostics."]
    with (out / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--cd-run", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="NEW output directory; no overwrites")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--generation-repeats", type=int, default=5)
    parser.add_argument("--gradient-probe-rows", type=int, default=32,
                        help="Predetermined training rows per model; 0 skips NONPRIVATE gradient diagnostics")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.generation_repeats <= 20 or not 0 <= args.gradient_probe_rows <= 256:
        parser.error("threads must be positive; generation-repeats must be 1..20; gradient-probe-rows must be 0..256")
    if args.out.exists():
        parser.error("Output already exists. Preserve it and choose a NEW directory.")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    base, previous_cd, config, old_b_checkpoint, train, validation, categories, schema = load_validated_context(args)
    target = float(previous_cd["privacy"]["C_nominal_composed_epsilon"])
    sigma, predicted = calibrate_training_noise(target, config["delta"], config["q"], int(config["steps"]))
    print(f"E calibrated sigma={sigma:.12f}; target epsilon={target:.9f}; predicted={predicted:.9f}", flush=True)
    print("Training three NEW models: E_matched private; A_diag and clip_only NONPRIVATE.", flush=True)
    print(f"Fixed {config['steps']} updates per arm. CPU threads={args.threads}. No new marginal release.", flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    cd.write_json(args.out / "private_provenance.json", {
        "privacy_status": "NONPRIVATE local research metadata; do not publish",
        "input_npz_sha256": cd.file_hash(args.data_npz), "schema_sha256": cd.file_hash(args.schema),
        "base_metrics_sha256": cd.file_hash(args.base_run / "metrics.json"),
        "old_B_checkpoint_sha256": cd.file_hash(args.base_run / "B_final.pt"),
        "cd_metrics_sha256": cd.file_hash(args.cd_run / "metrics.json"),
        "note": "Current inputs match prior C/D provenance, which did not prove original A/B input provenance."})
    torch.manual_seed(int(config["model_seed"]))
    initial = build_model(categories, config)
    models, accountant, training = train_arms(initial, train, config, sigma, args.out)
    achieved = float(accountant.get_epsilon(delta=config["delta"]))
    if (achieved > target or target - achieved > EPSILON_TOLERANCE
            or not math.isclose(achieved, predicted, rel_tol=1e-12, abs_tol=1e-12)):
        raise RuntimeError("Actual E accounting missed the predeclared target/tolerance")
    print("Final checkpoints saved. Performing fixed gradient diagnostics and repeated generation.", flush=True)
    probe_count = min(args.gradient_probe_rows, len(train))
    probe_seed = int(config["model_seed"]) + 90000
    probes = {"privacy_status": "NONPRIVATE", "rows": probe_count, "diffusion_seed": probe_seed,
              "selection": "Fixed seeded permutation of training rows; identical rows/diffusion draws for all models",
              "one_diffusion_draw_per_row": True, "models": {}}
    if probe_count:
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(probe_seed))[:probe_count]
        with torch.random.fork_rng(devices=[]):
            old_b = build_model(categories, config, old_b_checkpoint["model"])
        snapshots = {"initial": initial, "old_B": old_b, **models}
        for name, model in snapshots.items():
            print(f"NONPRIVATE gradient probe: {name}, {probe_count} rows", flush=True)
            probes["models"][name] = gradient_probe(model, train[indices], probe_seed, config["clip"])
    evaluation = repeated_evaluation(models, train, validation, categories, previous_cd, args.generation_repeats)
    original_c_grouped = cd.grouped_errors(previous_cd["results"]["C"]["validation"])
    original_c_grouped["age_outside_0_100_fraction"] = previous_cd["results"]["C"]["age"]["decoded_outside_0_100_fraction"]
    scripts = Path(__file__).resolve().parent
    report = {
        "purpose": "Budget-matched training-only E plus matched nonprivate clipping controls and gradient diagnostics",
        "formal_end_to_end_dp_claimed": False, "metrics_are_dp": False,
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "opacus": opacus.__version__},
        "original_training_config": config, "n_train": len(train), "n_validation": len(validation),
        "samples_per_generation": previous_cd["samples_per_arm"], "sample_batch": previous_cd["sample_batch"],
        "cpu_threads": args.threads, "training": training,
        "privacy": {"E_target_nominal_epsilon": target, "E_achieved_training_epsilon": achieved,
                    "E_budget_shortfall": target - achieved, "calibration_tolerance": EPSILON_TOLERANCE,
                    "E_noise_multiplier": sigma, "delta": config["delta"],
                    "old_B_epsilon": previous_cd["privacy"]["B_training_epsilon"],
                    "C_nominal_epsilon": target, "E_accountant": "RDP",
                    "A_diag_is_dp": False, "clip_only_is_dp": False,
                    "clip_only_noise_multiplier": 0.0, "new_marginal_releases": 0,
                    "scope": "Per-mechanism nominal training accounting only, not complete pipeline/research-run composition. Releasing another trained E on protected data requires additional privacy composition.",
                    "randomness": "Independent undisclosed entropy seeds for E noise/Poisson; shared separate Poisson draws for NONPRIVATE A_diag/clip_only. secure_mode=False."},
        "gradient_probes": probes, "evaluation": evaluation,
        "same_generation_seed_reference": {"C_original_single_seed": original_c_grouped,
                        "E_matched_seed0": evaluation["per_repeat"]["E_matched"][0]["validation_grouped"]},
        "source_sha256": {name: cd.file_hash(scripts / name) for name in (*HELPERS, Path(__file__).name)},
        "tabdiff_source_sha256": {name: cd.file_hash(scripts.parent / name) for name in TABDIFF_SOURCES},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["One trained model per new arm; repeated generation is not independent training replication.",
                        "A_diag/clip_only are matched to each other, not original B's training draws.",
                        "E/C budgets match nominally, but independent training randomness remains a confound in one-run conclusions.",
                        "Gradient diagnostics and all reference-data evaluations are NONPRIVATE.",
                        "Validation has informed research development; no untouched final test set is evaluated.",
                        "No new DP marginal release. This does not erase privacy cost of earlier or current training releases."]}
    cd.write_json(args.out / "metrics.json", report)
    write_summary(args.out, report)
    print(f"Complete. Read {args.out / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
