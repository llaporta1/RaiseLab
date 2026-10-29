#!/usr/bin/env python3
"""Bounded CPU A/B TabDiff pilot on a local, prepared public-use IPUMS sample.

This is preliminary research, NOT an audited end-to-end DP release. A and all
reference-data metrics are nonprivate. No C/D guidance, C2ST, or ML utility is
computed. Input NPZ files contain REAL records and must not be published.
"""

import argparse
import copy
import csv
import itertools
import json
import math
from pathlib import Path
import secrets
import sys
import time
import uuid

import numpy as np
import opacus
import torch
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_tabdiff_smoke as smoke


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
AGE_EDGES = np.array([0, 18, 25, 35, 45, 55, 65, 75, 85, 101], dtype=float)


def load_inputs(args):
    train, categories, _ = smoke.load_inputs(args)
    with np.load(args.data_npz, allow_pickle=False) as archive:
        holdout = np.asarray(archive["X_test"], dtype=np.float32)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    if schema.get("features") != FEATURES:
        raise ValueError(f"Schema features must be exactly {FEATURES}, in that order")
    domains = schema.get("category_domains", {})
    for name, size in zip(FEATURES[1:], categories):
        values = domains.get(name)
        if not isinstance(values, list) or len(values) != int(size):
            raise ValueError(f"Public category domain/cardinality mismatch for {name}")
        if len(set(values)) != len(values):
            raise ValueError(f"Repeated public category code for {name}")
    if len(holdout) == 0:
        raise ValueError("A nonempty held-out sample is required")
    transform = schema.get("age_transform", "").replace(" ", "")
    if "(AGE-50)/50" not in transform:
        raise ValueError("Schema must declare the fixed AGE encoding (AGE-50)/50")
    for name, table in [("training", train.numpy()), ("held-out", holdout)]:
        if np.any((table[:, 0] < -1) | (table[:, 0] > 1)):
            raise ValueError(f"{name} AGE is outside the declared clipped input range")
    return train, holdout, categories, schema


def evaluation_codes(table, categories, real_records=False):
    """Fixed, public numeric bins; categorical codes remain exact public IDs."""
    values = np.asarray(table)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or not len(values):
        raise ValueError("Evaluation table must be nonempty and have seven columns")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values: no rows will be silently dropped")
    cats = values[:, 1:]
    if np.any(cats != np.floor(cats)) or np.any(cats < 0):
        raise ValueError("Invalid categorical index in evaluation table")
    if np.any(cats >= categories[None, :]):
        raise ValueError("Categorical index outside the public domain")
    ages = 50.0 + 50.0 * values[:, 0].astype(np.float64)
    if real_records:
        # AGE is integer-valued in the extract. Undo float32 encoding error
        # (e.g. the inverse of encoded age 35 can be 34.9999994) before binning.
        # Generated numerical values are NOT rounded this way.
        ages = np.rint(ages)
    codes = np.empty(values.shape, dtype=np.int64)
    # Codes: 0 = AGE<0; 1 = [0,18); ...; 9 = [85,101); 10 = AGE>=101.
    # Do not clip/round generated AGE; the extra support check also detects
    # generated values in (100,101), which are inside the final histogram bin.
    codes[:, 0] = np.searchsorted(AGE_EDGES, ages, side="right")
    codes[:, 1:] = cats.astype(np.int64)
    age_info = {
        "decoded_outside_0_100_fraction": float(np.mean((ages < 0) | (ages > 100))),
        "histogram_counts": np.bincount(codes[:, 0], minlength=len(AGE_EDGES) + 1).tolist(),
    }
    return codes, age_info


def marginal_l1(left, right, columns, domain_sizes):
    """Normalized histogram L1 in [0,2], including zero-probability cells.

    Sparse union counting is exactly equivalent to a full Cartesian histogram;
    it avoids allocating high-cardinality tables with mostly zero cells.
    """
    if not len(left) or not len(right) or len(columns) not in (1, 2):
        raise ValueError("Need nonempty tables and one or two column indices")

    def keys(table):
        result = table[:, columns[0]].astype(np.int64)
        if len(columns) == 2:
            result = result * int(domain_sizes[columns[1]]) + table[:, columns[1]]
        return result

    left_keys, left_counts = np.unique(keys(left), return_counts=True)
    right_keys, right_counts = np.unique(keys(right), return_counts=True)
    union = np.union1d(left_keys, right_keys)
    difference = np.zeros(len(union), dtype=np.float64)
    difference[np.searchsorted(union, left_keys)] += left_counts / len(left)
    difference[np.searchsorted(union, right_keys)] -= right_counts / len(right)
    result = float(np.abs(difference).sum())
    if not -1e-12 <= result <= 2 + 1e-12:
        raise RuntimeError("L1 calculation outside its expected [0,2] range")
    return result


def compare_marginals(left, right, domain_sizes):
    one_way = {name: marginal_l1(left, right, (i,), domain_sizes)
               for i, name in enumerate(FEATURES)}
    two_way = {f"{FEATURES[i]} x {FEATURES[j]}":
               marginal_l1(left, right, (i, j), domain_sizes)
               for i, j in itertools.combinations(range(len(FEATURES)), 2)}
    return {"one_way": one_way, "two_way": two_way,
            "mean_one_way_l1": float(np.mean(list(one_way.values()))),
            "mean_two_way_l1": float(np.mean(list(two_way.values())))}


def generate(model, count, batch_size, categories):
    model.eval()
    chunks = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        chunk = model.sample(size).numpy()
        if chunk.shape != (size, len(FEATURES)):
            raise RuntimeError("Sampler returned an unexpected shape")
        evaluation_codes(chunk, categories)  # fail instead of filtering/retrying
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def write_results(out, report, schema):
    with (out / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with (out / "marginal_errors.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "reference", "order", "query", "l1"])
        comparisons = [(arm, ref, report["results"][arm][ref])
                       for arm in ("A", "B") for ref in ("train", "holdout")]
        comparisons.append(("real_reference", "train_vs_holdout", report["real_reference"]))
        for arm, reference, metrics in comparisons:
            for order, key in [(1, "one_way"), (2, "two_way")]:
                for query, error in metrics[key].items():
                    writer.writerow([arm, reference, order, query, error])
    privacy = report["training_privacy_accounting"]
    lines = [
        "PRELIMINARY IPUMS A/B CPU PILOT — NOT AN END-TO-END DP RELEASE",
        "A: no clipping/no noise. B: per-record clipping + Gaussian gradient noise.",
        "Both: matched initialization, fixed schedules, same Poisson batches and diffusion draws.",
        "Tiny architecture and short training: no convergence or general-performance claim.",
        f"Train rows: {report['n_train']}; held-out rows: {report['n_holdout']}; synthetic rows per arm: {report['n_synthetic']}",
        f"Updates: {privacy['steps']}; q={privacy['q']:.8f}; sigma={privacy['sigma']}; C={privacy['clip']}",
        f"B training accountant: epsilon={privacy['epsilon']:.6f}, delta={privacy['delta']}",
        "This accounting excludes preprocessing/selection/evaluation; A and metrics are NONPRIVATE.",
        "L1 is sum(abs(probability differences)), range [0,2], not percent accuracy.",
        "AGE uses fixed public bins; synthetic AGE is neither clipped nor rounded for metrics.",
        "", "arm\treference\tmean_1way_L1\tmean_2way_L1",
    ]
    for arm in ("A", "B"):
        for reference in ("train", "holdout"):
            m = report["results"][arm][reference]
            lines.append(f"{arm}\t{reference}\t{m['mean_one_way_l1']:.6f}\t{m['mean_two_way_l1']:.6f}")
    real = report["real_reference"]
    lines += ["", f"Real train-vs-holdout sample variation: 1-way={real['mean_one_way_l1']:.6f}; 2-way={real['mean_two_way_l1']:.6f}",
              "This is a sample-variation reference, NOT an attainable error floor.",
              f"A AGE outside [0,100]: {report['results']['A']['age']['decoded_outside_0_100_fraction']:.4%}",
              f"B AGE outside [0,100]: {report['results']['B']['age']['decoded_outside_0_100_fraction']:.4%}",
              "C/D marginal guidance, C2ST and downstream ML utility: NOT COMPUTED.",
              "Rare categories can be absent from these small samples; inspect per-query errors.",
              "Keep real input NPZs and all nonprivate outputs local; do not commit them.",
              "Rerunning on protected data is another mechanism and requires composition.",
              f"Input domain note: {schema.get('domain_warning', 'See public schema provenance.')}"]
    with (out / "summary.txt").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-npz", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--sample-batch", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--noise-multiplier", type=float, default=1.2)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, help="NEW output directory; no overwrites")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.threads, args.samples, args.sample_batch) < 1:
        parser.error("Steps, batch sizes, threads, and samples must be positive")
    if args.sampling_steps < 2 or not 0 < args.delta < 1:
        parser.error("Need sampling-steps>=2 and 0<delta<1")
    if any(not math.isfinite(x) or x <= 0 for x in (args.noise_multiplier, args.clip, args.lr)):
        parser.error("Noise multiplier, clip, and learning rate must be finite and positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    smoke.CLIP = args.clip  # runtime setting for its independent clipping reference
    train, holdout, categories, schema = load_inputs(args)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out = args.out or Path("pipeline_runs/prototype2_cpu") / f"ab_{stamp}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    print("LOCAL PRELIMINARY A/B PILOT. A and metrics are nonprivate; no end-to-end DP claim.", flush=True)
    print(f"Fixed updates: {args.steps}; train={len(train)}; holdout={len(holdout)}; CPU threads={args.threads}", flush=True)
    a = smoke.build_diffusion(categories, args.sampling_steps)
    b = copy.deepcopy(a)
    if not all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())):
        raise RuntimeError("A/B initial weights differ")
    a_opt = torch.optim.AdamW([p for _, p in smoke.trainable_parameters(a)],
                              lr=args.lr, weight_decay=0)
    b_base = torch.optim.AdamW([p for _, p in smoke.trainable_parameters(b)],
                               lr=args.lr, weight_decay=0)
    b_opt = DPOptimizer(b_base, noise_multiplier=args.noise_multiplier,
                        max_grad_norm=args.clip, expected_batch_size=args.batch_size,
                        loss_reduction="mean", secure_mode=False,
                        generator=torch.Generator().manual_seed(secrets.randbits(63)))
    sample_rng = torch.Generator().manual_seed(secrets.randbits(63))
    q = args.batch_size / len(train)
    accountant = RDPAccountant()
    b_opt.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=q))
    history, empty_steps = [], 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = train[torch.rand(len(train), generator=sample_rng) < q]
        empty_steps += int(len(batch) == 0)
        state = torch.get_rng_state()
        smoke.nonprivate_update(a, a_opt, batch, args.batch_size)
        torch.set_rng_state(state)
        smoke.private_update(b, b_opt, batch)
        if step == 1 or step % 50 == 0 or step == args.steps:
            epsilon = float(accountant.get_epsilon(delta=args.delta))
            if not math.isfinite(epsilon):
                raise RuntimeError("Nonfinite accountant epsilon")
            history.append({"step": step, "epsilon": epsilon})
            print(f"Update {step}/{args.steps}; B training epsilon={epsilon:.6f}; elapsed={time.perf_counter()-started:.1f}s", flush=True)
    training_seconds = time.perf_counter() - started
    if sum(item[2] for item in accountant.history) != args.steps:
        raise RuntimeError("Privacy accountant step count mismatch")
    config = {"architecture": {"num_layers": 1, "d_token": 4, "n_head": 1,
                               "factor": 2, "dim_t": 64, "numeric_columns": 1},
              "fixed_schedules": True, "sampling_steps": args.sampling_steps,
              "categories": categories.tolist(), "features": FEATURES,
              "lr": args.lr, "steps": args.steps, "batch_size": args.batch_size,
              "q": q, "sigma": args.noise_multiplier, "clip": args.clip,
              "delta": args.delta, "model_seed": args.seed}
    # Predetermined final checkpoints, never chosen with a private loss/metric.
    torch.save({"model": a.state_dict(), "optimizer": a_opt.state_dict(),
                "config": config, "privacy_status": "NONPRIVATE"}, out / "A_final.pt")
    torch.save({"model": b.state_dict(), "optimizer": b_opt.state_dict(),
                "accountant": accountant.state_dict(), "config": config,
                "privacy_status": "training accounting only; not audited end-to-end"},
               out / "B_final.pt")
    with (out / "public_schema.json").open("x", encoding="utf-8") as handle:
        json.dump({"features": FEATURES, "category_domains": schema["category_domains"],
                   "age_transform": schema["age_transform"]}, handle, indent=2)
    print("Final checkpoints saved. Generating bounded A/B samples...", flush=True)
    sampling_started = time.perf_counter()
    torch.manual_seed(args.seed + 1000)
    a_sample = generate(a, args.samples, args.sample_batch, categories)
    torch.manual_seed(args.seed + 1000)
    b_sample = generate(b, args.samples, args.sample_batch, categories)
    sampling_seconds = time.perf_counter() - sampling_started
    for arm, values in [("A", a_sample), ("B", b_sample)]:
        np.savez_compressed(out / f"{arm}_synthetic.npz", X_synthetic=values,
                            categories=categories, features=np.asarray(FEATURES))
    domain_sizes = np.r_[len(AGE_EDGES) + 1, categories]
    train_codes, train_age = evaluation_codes(train.numpy(), categories, real_records=True)
    holdout_codes, holdout_age = evaluation_codes(holdout, categories, real_records=True)
    results = {}
    for arm, values in [("A", a_sample), ("B", b_sample)]:
        codes, age_info = evaluation_codes(values, categories)
        results[arm] = {"train": compare_marginals(codes, train_codes, domain_sizes),
                        "holdout": compare_marginals(codes, holdout_codes, domain_sizes),
                        "age": age_info}
    report = {
        "purpose": "preliminary small fixed-schedule TabDiff A/B pilot, not converged research",
        "formal_end_to_end_dp_claimed": False, "metrics_are_dp": False,
        "torch_version": torch.__version__, "opacus_version": opacus.__version__,
        "n_train": len(train), "n_holdout": len(holdout), "n_synthetic": args.samples,
        "initial_weights_matched": True, "config": config,
        "training_privacy_accounting": {"epsilon": history[-1]["epsilon"],
             "delta": args.delta, "q": q, "sigma": args.noise_multiplier,
             "clip": args.clip, "steps": args.steps, "accountant": "RDP",
             "empty_poisson_steps": empty_steps, "history": history,
             "scope": "DP-SGD mechanism only; excludes preprocessing/selection/evaluation",
             "randomness": "Independent undisclosed entropy-seeded noise and Poisson RNGs; secure_mode=False"},
        "metrics_protocol": {"definition": "L1=sum absolute probability differences, range [0,2]",
             "one_way_queries": 7, "two_way_queries": 21, "aggregation": "unweighted mean across queries",
             "age_decode": "AGE=50+50*x; round reconstructed integer real AGE only; generated ages not clipped/rounded",
             "age_edges": AGE_EDGES.tolist(), "age_underflow": "AGE<0",
             "age_overflow": "AGE>=101", "bins_chosen_using_holdout": False,
             "weighting": "unweighted record sample, not national population estimates"},
        "results": results, "train_age": train_age, "holdout_age": holdout_age,
        "real_reference": compare_marginals(train_codes, holdout_codes, domain_sizes),
        "real_reference_interpretation": "finite-sample variation, NOT an error floor",
        "not_computed": ["C/D conditioning", "C2ST", "downstream ML utility", "privacy audit"],
        "training_seconds": training_seconds, "sampling_seconds": sampling_seconds,
        "input_domain_warning": schema.get("domain_warning"),
        "limitations": ["Small model and short run; no convergence claim.",
             "Rare categories may be missing from small real/synthetic samples.",
             "No private hyperparameter selection; final checkpoint determined in advance.",
             "A and reference-data metrics are nonprivate; keep protected-data outputs local.",
             "Reruns on protected data require privacy composition."]}
    write_results(out, report, schema)
    print(f"Complete. Read {out / 'summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
