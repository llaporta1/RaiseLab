#!/usr/bin/env python3
"""Seeded CPU mechanism smoke test of this repository's actual TabDiff.

This is NOT an IPUMS fidelity experiment, a privacy audit, or a production DP
release. The default data are generated toy records. --data-npz must contain
X_train, X_test, and categories (the public cardinalities of categorical columns).
Each X array has one numerical column followed by categorical integer indices.
Input preprocessing and category-domain provenance are NOT certified here.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import secrets
import sys
import time
import uuid

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import Model, UniModMLP


SIGMA = 1.2
CLIP = 1.0
DELTA = 1e-5
TOY_CATEGORIES = [2, 9, 5, 24, 531, 266]


def load_inputs(args):
    if args.data_npz:
        with np.load(args.data_npz, allow_pickle=False) as data:
            required = {"X_train", "X_test", "categories"}
            if not required.issubset(data.files):
                raise ValueError("NPZ must contain X_train, X_test, categories")
            train = np.asarray(data["X_train"], dtype=np.float32)
            test = np.asarray(data["X_test"], dtype=np.float32)
            raw_categories = np.asarray(data["categories"])
        source = "user_npz_not_audited"
    else:
        raw_categories = np.asarray(TOY_CATEGORIES, dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        values = [rng.uniform(-1, 1, size=args.n)]
        values += [rng.integers(int(k), size=args.n) for k in raw_categories]
        train = np.column_stack(values).astype(np.float32)
        test = np.empty((0, train.shape[1]), dtype=np.float32)
        source = "generated_toy_pipeline_validation"
    if raw_categories.ndim != 1 or not len(raw_categories):
        raise ValueError("categories must be a nonempty vector of cardinalities")
    if not np.isfinite(raw_categories).all():
        raise ValueError("Nonfinite category cardinality")
    categories = raw_categories.astype(np.int64)
    if not np.array_equal(categories, raw_categories) or np.any(categories < 1):
        raise ValueError("Category cardinalities must be positive integers")
    for label, values in [("X_train", train), ("X_test", test)]:
        if values.ndim != 2 or values.shape[1] != 1 + len(categories):
            raise ValueError(f"{label} must have one numeric + len(categories) columns")
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contains nonfinite values")
        cats = values[:, 1:]
        if np.any(cats != np.floor(cats)) or np.any(cats < 0):
            raise ValueError(f"{label} categorical codes must be nonnegative integers")
        if np.any(cats >= categories[None, :]):
            raise ValueError(f"{label} categorical code outside declared domain")
    if len(train) < args.batch_size:
        raise ValueError("batch-size cannot exceed the number of training records")
    return torch.from_numpy(train), categories, source


def build_diffusion(categories, sampling_steps):
    backbone = UniModMLP(
        d_numerical=1, categories=(categories + 1).tolist(), num_layers=1,
        d_token=4, n_head=1, factor=2, bias=True, dim_t=64, use_mlp=True,
    )
    # These modules are allocated, but never used in Transformer.forward.
    # Freeze only these known unused parameters; an unexpected unused parameter
    # anywhere else is a hard error in manual_grad_samples below.
    for transformer in (backbone.encoder, backbone.decoder):
        for module in (transformer.head, transformer.last_normalization):
            if module is not None:
                module.requires_grad_(False)
    denoiser = Model(backbone, precond=True, sigma_data=1.0,
                     net_conditioning="sigma")
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories, num_numerical_features=1,
        denoise_fn=denoiser, y_only_model=None, num_timesteps=sampling_steps,
        scheduler="power_mean", cat_scheduler="log_linear",
        noise_dist="uniform_t", edm_params={"sigma_data": 1.0},
        noise_schedule_params={"sigma_min": 0.002, "sigma_max": 80,
                               "rho": 7, "eps_max": 1e-3, "eps_min": 1e-5},
        sampler_params={"stochastic_sampler": True,
                        "second_order_correction": True},
        device=torch.device("cpu"),
    ).float()
    if any(True for _ in diffusion.num_schedule.parameters()):
        raise RuntimeError("Numeric schedule unexpectedly has parameters")
    if any(True for _ in diffusion.cat_schedule.parameters()):
        raise RuntimeError("Categorical schedule unexpectedly has parameters")
    return diffusion


def trainable_parameters(model):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def manual_grad_samples(model, batch):
    """Exact one-record autograd; no hooks, vmap, or omitted custom parameters."""
    named = trainable_parameters(model)
    params = [p for _, p in named]
    collected = [[] for _ in params]
    model.train()
    for row in batch:
        dloss, closs = model.mixed_loss(row[None, :].float())
        loss = dloss + closs  # fixed d_lambda=c_lambda=1; matches original loss
        if not torch.isfinite(loss).all():
            raise RuntimeError("Nonfinite per-record loss; no update performed")
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for (name, _), grad, bucket in zip(named, grads, collected):
            if grad is None:
                raise RuntimeError(f"Unexpected missing per-record gradient: {name}")
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"Nonfinite per-record gradient: {name}")
            bucket.append(grad.detach())
    return [torch.stack(bucket) if bucket else p.new_empty((0, *p.shape))
            for p, bucket in zip(params, collected)]


def private_update(model, optimizer, batch):
    """Populate grad_sample ourselves; Opacus owns flat clipping and noise."""
    optimizer.zero_grad(set_to_none=True)
    params = [p for _, p in trainable_parameters(model)]
    grads = manual_grad_samples(model, batch)
    if len(batch):
        squared = sum(g.flatten(start_dim=1).square().sum(1) for g in grads)
        norms = squared.sqrt()
        factors = (CLIP / (norms + 1e-6)).clamp(max=1.0)
        if not torch.all(norms * factors <= CLIP + 1e-5):
            raise RuntimeError("Per-record global clipping check failed")
    else:
        factors = batch.new_empty((0,))
    expected = []
    for p, grad in zip(params, grads):
        p.grad_sample = grad
        expected.append(torch.einsum("i,i...->...", factors, grad))
    optimizer.step()  # an empty Poisson batch still receives noise and is counted
    for (name, p), clipped_sum in zip(trainable_parameters(model), expected):
        if not torch.allclose(p.summed_grad, clipped_sum, atol=1e-5, rtol=1e-4):
            raise RuntimeError(f"Opacus clipped-sum reference check failed: {name}")
        if not torch.isfinite(p).all():
            raise RuntimeError(f"Nonfinite model parameter after private update: {name}")


def nonprivate_update(model, optimizer, batch, expected_batch_size):
    optimizer.zero_grad(set_to_none=True)
    grads = manual_grad_samples(model, batch)
    for (_, p), grad in zip(trainable_parameters(model), grads):
        # Same Poisson normalization as B, but neither clipping nor noise.
        p.grad = grad.sum(0) / expected_batch_size
    optimizer.step()


def sample_diagnostics(model, count, categories, destination, save_samples):
    if count == 0:
        return {"status": "skipped"}
    model.eval()
    try:
        # One bounded call: do not silently resample/filter away invalid outputs.
        values = model.sample(count).numpy()
        finite = np.isfinite(values).all(axis=1)
        cats = values[:, 1:]
        valid = finite & (cats == np.floor(cats)).all(axis=1)
        valid &= ((cats >= 0) & (cats < categories[None, :])).all(axis=1)
        result = {"status": "ok" if valid.all() else "invalid_samples",
                  "requested": count, "returned": int(len(values)),
                  "finite_rows": int(finite.sum()), "valid_rows": int(valid.sum())}
        if save_samples:
            np.savez_compressed(destination, X_synthetic=values, categories=categories)
        return result
    except (RuntimeError, AssertionError, ValueError) as error:
        return {"status": "sampling_failed", "error_type": type(error).__name__,
                "message": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n", type=int, default=256, help="Generated toy training rows")
    parser.add_argument("--data-npz", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.threads, args.batch_size, args.n) < 1:
        parser.error("steps, threads, batch-size, and n must be positive")
    if args.samples < 0 or args.sampling_steps < 2:
        parser.error("samples must be nonnegative; sampling-steps must be at least 2")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data, categories, source = load_inputs(args)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = args.output_dir or Path("pipeline_runs/prototype2_cpu") / f"smoke_{stamp}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print("CPU MECHANISM VALIDATION ONLY: no fidelity or end-to-end DP claim.", flush=True)
    print(f"Source: {source}. CPU threads: {args.threads}. Output: {out}", flush=True)
    started = time.perf_counter()
    a = build_diffusion(categories, args.sampling_steps)
    b = copy.deepcopy(a)
    initial_match = all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    if not initial_match:
        raise RuntimeError("A/B initial parameters do not match")
    a_optimizer = torch.optim.AdamW([p for _, p in trainable_parameters(a)],
                                    lr=0.001, weight_decay=0.0)
    base_optimizer = torch.optim.AdamW([p for _, p in trainable_parameters(b)],
                                       lr=0.001, weight_decay=0.0)
    # Reproducible DP noise is suitable ONLY for the generated-toy mechanism test.
    # For supplied data, draw independent seeds from system entropy and do not
    # save/disclose either seed or generator state. This still is not an audit of
    # preprocessing or of a production cryptographically-secure DP mechanism.
    noise_seed = secrets.randbits(63) if args.data_npz else args.seed + 1
    poisson_seed = secrets.randbits(63) if args.data_npz else args.seed + 2
    noise_rng = torch.Generator().manual_seed(noise_seed)
    poisson_rng = torch.Generator().manual_seed(poisson_seed)
    dp_optimizer = DPOptimizer(base_optimizer, noise_multiplier=SIGMA,
                               max_grad_norm=CLIP, expected_batch_size=args.batch_size,
                               loss_reduction="mean", generator=noise_rng,
                               secure_mode=False)
    accountant = RDPAccountant()
    q = args.batch_size / len(data)
    dp_optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=q))
    epsilon_history = []
    empty_steps = 0
    for step in range(args.steps):
        indices = torch.rand(len(data), generator=poisson_rng) < q
        batch = data[indices]
        empty_steps += int(len(batch) == 0)
        state = torch.get_rng_state()
        nonprivate_update(a, a_optimizer, batch, args.batch_size)
        torch.set_rng_state(state)  # matched diffusion draws, separate DP-noise RNG
        private_update(b, dp_optimizer, batch)
        epsilon = float(accountant.get_epsilon(delta=DELTA))
        if not math.isfinite(epsilon) or (epsilon_history and epsilon < epsilon_history[-1]):
            raise RuntimeError("Accountant epsilon is not finite/nondecreasing")
        epsilon_history.append(epsilon)
        print(f"Completed A/B update {step + 1}/{args.steps}; B accountant epsilon={epsilon:.6f}",
              flush=True)
    accounted_steps = sum(item[2] for item in accountant.history)
    if accounted_steps != args.steps:
        raise RuntimeError("Accountant logical update count does not match training")
    training_seconds = time.perf_counter() - started
    if args.save_models:
        torch.save({"model": a.state_dict(), "steps": args.steps, "source": source},
                   out / "A_nonprivate.pt")
        torch.save({"model": b.state_dict(), "optimizer": dp_optimizer.state_dict(),
                    "accountant": accountant.state_dict(), "steps": args.steps,
                    "source": source, "categories": categories.tolist()},
                   out / "B_mechanism_validation.pt")
    state = torch.get_rng_state()
    a_sample = sample_diagnostics(a, args.samples, categories, out / "A_samples.npz",
                                  args.save_samples)
    torch.set_rng_state(state)
    b_sample = sample_diagnostics(b, args.samples, categories, out / "B_samples.npz",
                                  args.save_samples)
    diagnostics = {
        "purpose": "CPU pipeline/mechanism validation; not fidelity research",
        "source": source, "formal_dp_guarantee_claimed": False,
        "caveats": ["Toy DP-noise seeds are fixed; supplied-data noise/sampling seeds are not disclosed.",
                    "secure_mode=False; non-cryptographic RNG; not a production privacy audit.",
                    "Preprocessing, public-domain provenance and input selection not audited.",
                    "No C/D conditioning, held-out utility, privacy audit, or convergence claim.",
                    "A is nonprivate: do not publish its artifacts for protected input data.",
                    "X_test is shape-checked but not used for training or evaluation."],
        "torch_version": torch.__version__, "n_train": len(data),
        "categories": categories.tolist(), "initial_weights_match": initial_match,
        "architecture": {"dim_t": 64, "num_layers": 1, "d_token": 4, "factor": 2,
                         "numeric_columns": 1, "fixed_noise_schedules": True},
        "steps": args.steps, "threads": args.threads, "expected_batch_size": args.batch_size,
        "accounted_steps": accounted_steps, "empty_poisson_steps": empty_steps,
        "sample_rate": q, "noise_multiplier": SIGMA, "max_grad_norm": CLIP,
        "delta": DELTA, "epsilon_history": epsilon_history,
        "finite_gradients_checked": True, "global_clipped_sums_checked": True,
        "training_seconds": training_seconds, "sampling_steps": args.sampling_steps,
        "A_samples": a_sample, "B_samples": b_sample,
    }
    with (out / "diagnostics.json").open("x", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, allow_nan=False)
    print(f"Diagnostics saved: {out / 'diagnostics.json'}", flush=True)
    if a_sample["status"] not in {"ok", "skipped"} or b_sample["status"] not in {"ok", "skipped"}:
        print("Sampling validation failed; inspect diagnostics before any longer run.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
