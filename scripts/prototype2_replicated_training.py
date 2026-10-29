#!/usr/bin/env python3
"""Training kernel for predeclared, independent Prototype-2 research replicates.

A_diag, clip_only and B_private share initialization, Poisson batches and
per-record diffusion draws within a replicate. E_matched shares initialization
but uses independent sampling/noise generators. Clipping-only and A_diag are
NONPRIVATE; coupling/releasing these controls is NOT an end-to-end DP release.

This helper neither estimates marginals nor evaluates/selects checkpoints. Its
caller owns dataset/source provenance, budget allocation, evaluation, and a
fresh existing output directory. Only numeric_weight=1.0 is currently supported.
"""

import copy
import math
from pathlib import Path
import secrets
import time

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

import local_tabdiff_smoke as smoke
import run_ipums_cd_pilot as cd


ARMS = ("A_diag", "clip_only", "B_private", "E_matched")
PRIVATE_ARMS = ("B_private", "E_matched")
MATCHED_ARMS = ("A_diag", "clip_only", "B_private")


def _validate(initial, train, config, sigma_e, out, replicate_seed, numeric_weight):
    if numeric_weight != 1.0:
        raise ValueError("This kernel currently supports numeric_weight=1.0 only; no loss-reweighting claim is made")
    if not 1 <= torch.get_num_threads() <= 4:
        raise ValueError("Caller must set CPU threads between 1 and 4 before training")
    if (isinstance(replicate_seed, bool) or int(replicate_seed) != replicate_seed
            or not 0 <= replicate_seed < 2**63 - 1):
        raise ValueError("replicate_seed must be a nonnegative integer below 2**63-1")
    if not out.is_dir():
        raise ValueError("Caller must supply a fresh, existing output directory")
    for arm in ARMS:
        if (out / f"{arm}_final.pt").exists():
            raise FileExistsError(f"Refusing to overwrite {arm}_final.pt")
    if config.get("architecture") != cd.EXPECTED_ARCHITECTURE or config.get("fixed_schedules") is not True:
        raise ValueError("Only the original tiny, fixed-schedule architecture is supported")
    for name in ("steps", "batch_size", "sampling_steps"):
        if isinstance(config[name], bool) or int(config[name]) != config[name] or config[name] < 1:
            raise ValueError(f"config.{name} must be a positive integer")
    if config["sampling_steps"] < 2:
        raise ValueError("At least two reverse sampling steps are required")
    for name, value in (("sigma_B", config["sigma"]), ("sigma_E", sigma_e),
                        ("clip", config["clip"]), ("lr", config["lr"])):
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 < config["delta"] < 1 or not 0 < config["q"] <= 1:
        raise ValueError("Invalid delta or Poisson sampling rate")
    categories = np.asarray(config["categories"], dtype=np.int64)
    if (categories.ndim != 1 or not len(categories) or np.any(categories < 1)
            or categories.tolist() != config["categories"]):
        raise ValueError("Invalid fixed category cardinalities")
    if (not isinstance(train, torch.Tensor) or train.device.type != "cpu"
            or train.dtype != torch.float32 or train.ndim != 2
            or train.shape[1] != len(categories) + 1 or len(train) < config["batch_size"]):
        raise ValueError("Training data must be float32 CPU rows with one numeric and the fixed categorical columns")
    if not torch.isfinite(train).all():
        raise ValueError("Nonfinite training input")
    if not math.isclose(config["q"], config["batch_size"] / len(train), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Sampling rate differs from the fixed expected batch/record count")
    if np.asarray(initial.num_classes).tolist() != categories.tolist():
        raise ValueError("Initial model and public categorical domains disagree")
    if any(p.device.type != "cpu" for p in initial.parameters()):
        raise ValueError("This bounded kernel supports CPU models only")
    if any(not torch.isfinite(v).all() for v in initial.state_dict().values()):
        raise ValueError("Initial model contains nonfinite state")
    # Verify, not merely label, the caller's predeclared initialization seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(replicate_seed))
        reference = smoke.build_diffusion(categories, config["sampling_steps"])
    if cd.model_hash(initial) != cd.model_hash(reference):
        raise ValueError("Initial weights do not match the predeclared replicate_seed")
    flags = [(name, p.requires_grad) for name, p in initial.named_parameters()]
    if flags != [(name, p.requires_grad) for name, p in reference.named_parameters()]:
        raise ValueError("Initial requires_grad flags differ from the unchanged model builder")
    return flags


def _optimizer(model, config, sigma):
    base = torch.optim.AdamW([p for _, p in smoke.trainable_parameters(model)],
                             lr=config["lr"], weight_decay=0)
    if sigma is None:
        return base
    return DPOptimizer(base, noise_multiplier=sigma, max_grad_norm=config["clip"],
                       expected_batch_size=config["batch_size"], loss_reduction="mean",
                       secure_mode=False,
                       generator=torch.Generator().manual_seed(secrets.randbits(63)))


def train_replicate(initial, train, config, sigma_e, out, replicate_seed,
                    numeric_weight=1.0):
    """Train one predeclared four-arm replicate without any evaluation/tuning.

    Returns ``(models, accountants, metadata)``. ``models`` keys are A_diag,
    clip_only, B_private, E_matched. ``accountants`` contains only B_private and
    E_matched. A_diag/clip_only have no finite DP guarantee, never epsilon=0.

    Only the initialization seed, replicate_seed, is public. Shared A/F/B and
    independent E diffusion streams, Poisson sampling and Gaussian generators
    all use fresh undisclosed entropy seeds. No training-augmentation, Poisson,
    or Gaussian seeds/states are returned/saved. Global caller RNG state and
    smoke.CLIP are restored; checkpoint-based sampling remains available.
    """
    out = Path(out)
    original_flags = _validate(initial, train, config, sigma_e, out,
                                replicate_seed, numeric_weight)
    replicate_seed = int(replicate_seed)
    initial_hash = cd.model_hash(initial)
    models = {arm: copy.deepcopy(initial) for arm in ARMS}
    if any(cd.model_hash(model) != initial_hash for model in models.values()):
        raise RuntimeError("Replicate arms do not have identical initial weights")
    sigma_by_arm = {"A_diag": None, "clip_only": 0.0,
                    "B_private": float(config["sigma"]), "E_matched": float(sigma_e)}
    optimizers = {arm: _optimizer(model, config, sigma_by_arm[arm])
                  for arm, model in models.items()}
    accountants = {arm: RDPAccountant() for arm in PRIVATE_ARMS}
    for arm, accountant in accountants.items():
        optimizers[arm].attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=config["q"]))
    shared_poisson = torch.Generator().manual_seed(secrets.randbits(63))
    e_poisson = torch.Generator().manual_seed(secrets.randbits(63))
    shared_diffusion = torch.Generator().manual_seed(secrets.randbits(63)).get_state()
    e_diffusion = torch.Generator().manual_seed(secrets.randbits(63)).get_state()
    empty = {"A_diag_clip_only_B_private": 0, "E_matched": 0}
    history = {arm: [] for arm in PRIVATE_ARMS}
    original_clip = smoke.CLIP
    smoke.CLIP = float(config["clip"])
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            for step in range(1, int(config["steps"]) + 1):
                shared_batch = train[torch.rand(len(train), generator=shared_poisson) < config["q"]]
                empty["A_diag_clip_only_B_private"] += int(len(shared_batch) == 0)
                torch.set_rng_state(shared_diffusion)
                before = torch.get_rng_state()
                smoke.nonprivate_update(models["A_diag"], optimizers["A_diag"], shared_batch,
                                        config["batch_size"])
                shared_diffusion = torch.get_rng_state()
                for arm in ("clip_only", "B_private"):
                    torch.set_rng_state(before)
                    smoke.private_update(models[arm], optimizers[arm], shared_batch)
                    if not torch.equal(torch.get_rng_state(), shared_diffusion):
                        raise RuntimeError(f"{arm} did not consume A_diag's identical per-record diffusion draws")
                e_batch = train[torch.rand(len(train), generator=e_poisson) < config["q"]]
                empty["E_matched"] += int(len(e_batch) == 0)
                torch.set_rng_state(e_diffusion)
                smoke.private_update(models["E_matched"], optimizers["E_matched"], e_batch)
                e_diffusion = torch.get_rng_state()
                if any(not torch.isfinite(p).all() for p in models["A_diag"].parameters()):
                    raise RuntimeError("A_diag developed nonfinite parameters")
                if step == 1 or step % 50 == 0 or step == config["steps"]:
                    current = {}
                    for arm, accountant in accountants.items():
                        epsilon = float(accountant.get_epsilon(delta=config["delta"]))
                        if not math.isfinite(epsilon):
                            raise RuntimeError(f"Nonfinite {arm} accountant result")
                        history[arm].append({"step": step, "epsilon": epsilon})
                        current[arm] = epsilon
                    print(f"Replicate seed {replicate_seed}; update {step}/{config['steps']}; "
                          f"B epsilon={current['B_private']:.6f}; E epsilon={current['E_matched']:.6f}; "
                          f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    finally:
        smoke.CLIP = original_clip
    seconds = time.perf_counter() - started
    for arm, accountant in accountants.items():
        if sum(record[2] for record in accountant.history) != config["steps"]:
            raise RuntimeError(f"{arm} accountant did not count every update, including empty batches")
    for arm, model in models.items():
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise RuntimeError(f"{arm} final state is nonfinite")
        if original_flags != [(name, p.requires_grad) for name, p in model.named_parameters()]:
            raise RuntimeError(f"{arm} requires_grad flags changed")
    if cd.model_hash(initial) != initial_hash:
        raise RuntimeError("Caller-provided initial model was changed")
    configs = {}
    # Fixed final checkpoints only, before downstream metrics/probes/selection.
    for arm, model in models.items():
        arm_config = copy.deepcopy(config)
        # Never retain an obsolete public augmentation seed from older configs.
        for key in ("diffusion_seed", "matched_diffusion_seed", "E_diffusion_seed"):
            arm_config.pop(key, None)
        arm_config.update({"arm": arm, "model_seed": replicate_seed,
            "sigma": sigma_by_arm[arm] if sigma_by_arm[arm] is not None else 0.0,
            "clip": config["clip"] if arm != "A_diag" else None,
            "numeric_loss_weight": 1.0})
        checkpoint = {"model": model.state_dict(), "optimizer": optimizers[arm].state_dict(),
            "config": arm_config,
            "privacy_status": "Single training mechanism accounting only; coupled NONPRIVATE controls prevent any whole-release DP claim"
                              if arm in PRIVATE_ARMS else "NONPRIVATE; no finite DP claim"}
        if arm in PRIVATE_ARMS:
            checkpoint["accountant"] = accountants[arm].state_dict()
        destination = out / f"{arm}_final.pt"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination.name}")
        torch.save(checkpoint, destination)
        configs[arm] = arm_config
    training_epsilons = {arm: float(accountant.get_epsilon(delta=config["delta"]))
                         for arm, accountant in accountants.items()}
    metadata = {
        "initialization_seed": replicate_seed,
        "initialization_seed_verified": True, "initial_state_sha256": initial_hash,
        "final_state_sha256": {arm: cd.model_hash(model) for arm, model in models.items()},
        "arm_configs": configs, "steps_per_arm": int(config["steps"]),
        "q": config["q"], "delta": config["delta"], "numeric_loss_weight": 1.0,
        "sigma_B": sigma_by_arm["B_private"], "sigma_E": sigma_by_arm["E_matched"],
        "training_epsilons": training_epsilons, "epsilon": training_epsilons.copy(),
        "accountant_history": history, "empty_poisson_updates": empty,
        "shared_rng_alignment_passed": True, "initial_weights_matched": True,
        "requires_grad_flags_preserved": True, "caller_initial_model_unchanged": True,
        "shared_training_arms": list(MATCHED_ARMS),
        "E_independent_Poisson_and_noise": True, "private_generators_saved": False,
        "E_independent_diffusion_stream": True, "training_augmentation_seeds_saved": False,
        "randomness": "Fresh undisclosed entropy seeds for shared A/F/B Poisson and diffusion streams, independent E Poisson and diffusion streams, and separate B/E Gaussian generators. Only the public initialization seed is saved.",
        "private_accounting_scope": "Individual B/E training mechanisms only, not the coupled experiment or cumulative project. A_diag/clip_only and downstream reference diagnostics are NONPRIVATE. Fresh private training releases require composition.",
        "secure_mode": False, "checkpoint_selection": "Predetermined final update; no evaluation/selection in kernel",
        "kernel_sha256": cd.file_hash(Path(__file__)), "seconds": seconds,
    }
    return models, accountants, metadata
