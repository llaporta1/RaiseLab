"""
Prototype orchestration for MWEM → TabDiff synthetic-data pipeline.

Responsibilities (no heavy logic yet):
1) Load a YAML config (see configs/mwem_tabdiff.yaml).
2) Run MWEM to produce DP marginals + a DP synthetic seed table.
3) Stage the seed table and dataset info for TabDiff training.
4) Launch TabDiff training via its existing CLI (tabdiff/main.py).
5) Launch TabDiff sampling with an optional marginal-correction callback.
6) Write final SDNIST-compatible synthetic data to disk.

Note: This file intentionally contains scaffolding only; the MWEM/TabDiff
execution steps are placeholders to be filled with project-specific
commands and light glue code. Do not reimplement MWEM or TabDiff here.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class PipelinePaths:
    """Container for key filesystem locations used across steps."""

    config_path: Path
    workdir: Path
    mwem_repo: Path
    tabdiff_repo: Path
    sdnist_raw: Path
    sdnist_info: Path
    mwem_seed: Path
    mwem_marginals: Path
    tabdiff_data_root: Path
    tabdiff_ckpt: Path
    final_output: Path


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        text = f.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ModuleNotFoundError:
        print("[Warn] PyYAML not installed; using minimal YAML parser for config.")
        return parse_simple_yaml(text)


def resolve_paths(cfg: Dict[str, Any], config_path: Path) -> PipelinePaths:
    root = config_path.parent
    pipeline_cfg = cfg.get("pipeline", {})
    workdir = Path(pipeline_cfg.get("workdir", root / "pipeline_runs")).resolve()

    mwem_cfg = cfg["mwem"]
    tabdiff_cfg = cfg["tabdiff"]
    sdnist_cfg = cfg["sdnist"]

    mwem_repo = Path(mwem_cfg.get("repo_path", "MWEM")).resolve()
    tabdiff_repo = Path(tabdiff_cfg.get("repo_path", ".")).resolve()
    sdnist_raw = Path(sdnist_cfg["raw_data_path"]).resolve()
    sdnist_info = Path(sdnist_cfg["info_json"]).resolve()

    outputs = mwem_cfg.get("output", {})
    mwem_seed = workdir / outputs.get("synthetic_path", "mwem_seed.csv")
    mwem_marginals = workdir / outputs.get("marginals_path", "mwem_marginals.json")

    tabdiff_data_root = Path(tabdiff_cfg.get("data_root", "data")).resolve()
    ckpt_rel = tabdiff_cfg.get("ckpt_relpath", f"tabdiff/ckpt/{tabdiff_cfg['dataname']}/{tabdiff_cfg.get('exp_name','mwem_seed_run')}/ema_model_final.pt")
    tabdiff_ckpt = tabdiff_repo / ckpt_rel

    final_output = workdir / pipeline_cfg.get("final_output", "synthetic_final.csv")

    return PipelinePaths(
        config_path=config_path,
        workdir=workdir,
        mwem_repo=mwem_repo,
        tabdiff_repo=tabdiff_repo,
        sdnist_raw=sdnist_raw,
        sdnist_info=sdnist_info,
        mwem_seed=mwem_seed,
        mwem_marginals=mwem_marginals,
        tabdiff_data_root=tabdiff_data_root,
        tabdiff_ckpt=tabdiff_ckpt,
        final_output=final_output,
    )

def cfg_tabdiff_dataname_fallback(paths: PipelinePaths) -> str:
    """Fallback dataname if info.json lacks it; derive from mwem seed filename."""
    return paths.mwem_seed.stem.replace("_seed", "") or "sdnist_sample"


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Very small YAML subset parser (mappings only, no lists)."""

    def parse_scalar(raw: str):
        raw = raw.strip()
        if raw == "":
            return {}
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        try:
            if "." in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            pass
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            return raw[1:-1]
        return raw

    lines = []
    for ln in text.splitlines():
        stripped = ln.split("#", 1)[0].rstrip()
        if stripped:
            lines.append(stripped)

    root: Dict[str, Any] = {}
    stack = [(0, root)]
    for ln in lines:
        indent = len(ln) - len(ln.lstrip(" "))
        key, _, val = ln.strip().partition(":")
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = parse_scalar(val)
        if isinstance(value, dict):
            current[key] = {}
            stack.append((indent + 2, current[key]))
        else:
            current[key] = value
    return root


def run_mwem(cfg: Dict[str, Any], paths: PipelinePaths) -> None:
    """Run MWEM using the existing private_pgm implementation.

    This is a minimal, self-contained runner that:
    - loads the SDNIST-prepared CSV,
    - builds a simple 1-way workload (fast),
    - fits MWEM,
    - writes DP seed + empirical marginals JSON.
    """

    mwem_cfg = cfg["mwem"]
    paths.workdir.mkdir(parents=True, exist_ok=True)
    if not paths.sdnist_raw.exists():
        raise FileNotFoundError(f"Raw data not found at {paths.sdnist_raw}")
    if not paths.sdnist_info.exists():
        raise FileNotFoundError(f"info.json not found at {paths.sdnist_info}")
    data = pd.read_csv(paths.sdnist_raw)

    # Fast path: keep a tiny subset if requested to make prototype quick.
    max_rows = mwem_cfg.get("max_rows")
    if max_rows is not None:
        data = data.sample(n=min(max_rows, len(data)), random_state=0).reset_index(drop=True)

    # Domain per column (stringified for consistency with private_pgm expectations).
    domain = {c: sorted(data[c].dropna().astype(str).unique().tolist()) for c in data.columns}

    # Simple 1-way workload across all columns; cheap and adequate for prototype.
    query_sets = [(c,) for c in data.columns]

    # Add MWEM repo to path and import.
    sys.path.insert(0, str(paths.mwem_repo))
    try:
        from private_pgm.mwem import MWEM  # type: ignore
        from private_pgm.workload import Marginals  # type: ignore
    except ModuleNotFoundError:
        # Fallback to stubbed implementation bundled in the repo
        sys.path.insert(0, str(paths.mwem_repo / "private_pgm_stub"))
        from mwem import MWEM  # type: ignore
        from workload import Marginals  # type: ignore

    workload = Marginals(domain, query_sets)
    epsilon = float(mwem_cfg.get("epsilon", 0.5))
    rounds = int(mwem_cfg.get("iterations", len(query_sets)))

    mwem = MWEM(domain=domain, epsilon=epsilon, rounds=rounds)
    model = mwem.fit(data.to_records(index=False), workload)

    rows = int(mwem_cfg.get("synthetic_rows", len(data)))
    seed = model.synthetic_data(rows=rows)
    paths.mwem_seed.parent.mkdir(parents=True, exist_ok=True)
    seed.to_csv(paths.mwem_seed, index=False)

    # Empirical marginals from the DP seed (lightweight placeholder).
    marginals = []
    for cols in query_sets:
        key = seed[list(cols)].astype(str).agg("|".join, axis=1)
        probs = (key.value_counts(normalize=True)).to_dict()
        marginals.append({"columns": list(cols), "probs": probs})

    with open(paths.mwem_marginals, "w") as f:
        json.dump({"marginals": marginals}, f, indent=2)

    print(f"[MWEM] wrote seed to {paths.mwem_seed}")
    print(f"[MWEM] wrote marginals to {paths.mwem_marginals}")


def stage_tabdiff_seed(cfg: Dict[str, Any], paths: PipelinePaths) -> None:
    """Ensure the MWEM synthetic seed sits where TabDiff expects training data.

    Typical TabDiff layout is data/{dataname}/{train,val,test}.csv plus info.json.
    For quick prototyping we can symlink/copy the MWEM seed to train.csv and reuse
    the existing val/test splits shipped with SDNIST if present.
    """

    with open(paths.sdnist_info, "r") as f:
        info = json.load(f)
    column_order = info["column_names"]

    dataname = cfg["tabdiff"].get("dataname", info.get("dataname", cfg_tabdiff_dataname_fallback(paths)))
    data_dir = paths.tabdiff_data_root / dataname
    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy info.json to TabDiff data dir for downstream loaders.
    info_dest = data_dir / "info.json"
    if paths.sdnist_info.resolve() != info_dest.resolve():
        shutil.copy(paths.sdnist_info, info_dest)

    seed_df = pd.read_csv(paths.mwem_seed)
    missing = [c for c in column_order if c not in seed_df.columns]
    if missing:
        raise ValueError(f"MWEM seed missing columns: {missing}")
    seed_df = seed_df[column_order]

    # Simple split: 80/10/10 train/val/test (deterministic).
    rng = np.random.default_rng(0)
    idx = np.arange(len(seed_df))
    rng.shuffle(idx)
    train_cut = int(0.8 * len(seed_df))
    val_cut = int(0.9 * len(seed_df))
    train_df = seed_df.iloc[idx[:train_cut]]
    val_df = seed_df.iloc[idx[train_cut:val_cut]]
    test_df = seed_df.iloc[idx[val_cut:]] if val_cut < len(seed_df) else seed_df.iloc[idx[train_cut:]]

    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    # Metrics expect synthetic/{dataname}/{real,test,val}.csv
    synthetic_dir = paths.tabdiff_repo / "synthetic" / data_dir.name
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    real_df = pd.read_csv(paths.sdnist_raw)
    missing_real = [c for c in column_order if c not in real_df.columns]
    if missing_real:
        raise ValueError(f"Real data missing columns: {missing_real}")
    real_df = real_df[column_order]
    real_df.to_csv(synthetic_dir / "real.csv", index=False)
    test_df.to_csv(synthetic_dir / "test.csv", index=False)
    val_df.to_csv(synthetic_dir / "val.csv", index=False)

    print(f"[Stage] Seed staged at {data_dir} with train/val/test splits.")


def train_tabdiff(cfg: Dict[str, Any], paths: PipelinePaths) -> Path:
    """Launch TabDiff training on the DP seed using its existing CLI."""

    tabdiff_cfg = cfg["tabdiff"]
    python_bin = tabdiff_cfg.get("python", sys.executable)
    cmd = [
        str(python_bin),
        "main.py",  # root-level TabDiff entrypoint
        "--mode",
        "train",
        "--dataname",
        tabdiff_cfg["dataname"],
        "--gpu",
        str(tabdiff_cfg.get("gpu", -1)),
        "--exp_name",
        tabdiff_cfg.get("exp_name", "mwem_seed_run"),
    ]
    if tabdiff_cfg.get("deterministic", True):
        cmd.append("--deterministic")
    cmd.append("--no_wandb")

    print("[Train] Running TabDiff:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    subprocess.run(cmd, cwd=paths.tabdiff_repo, check=True, env=env)

    return paths.tabdiff_ckpt


def sample_with_correction(cfg: Dict[str, Any], paths: PipelinePaths, ckpt_path: Path) -> None:
    """Invoke TabDiff sampling and (optionally) plug in marginal correction.

    Integration points:
    - Pass MWEM marginals to a correction callback (see pipelines.marginal_correction).
    - Hook into UnifiedCtimeDiffusion.sample/edm_update to apply correction every k steps.
    - Keep everything post-processing (no extra access to raw data).

    TODO: wire up correction flag once the sampler exposes a callback hook.
    """

    sample_cfg = cfg["tabdiff"].get("sample", {})
    python_bin = cfg["tabdiff"].get("python", sys.executable)
    cmd = [
        str(python_bin),
        "main.py",
        "--mode",
        "test",
        "--dataname",
        cfg["tabdiff"]["dataname"],
        "--ckpt_path",
        str(ckpt_path),
        "--num_samples_to_generate",
        str(sample_cfg.get("num_samples", 10000)),
        "--gpu",
        str(cfg["tabdiff"].get("gpu", -1)),
    ]
    cmd.append("--no_wandb")
    cmd.append("--skip_metrics")

    if sample_cfg.get("keep_nan_samples", False):
        cmd.append("--keep_nan_samples")

    if sample_cfg.get("apply_marginal_correction", False):
        cmd.extend(
            [
                "--marginal_correction_config",
                str(paths.mwem_marginals),
                "--marginal_correction_cadence",
                str(sample_cfg.get("correction_cadence", 2)),
                "--marginal_correction_iters",
                str(sample_cfg.get("correction_iters", 1)),
                "--marginal_correction_step_size",
                str(sample_cfg.get("correction_step_size", 0.1)),
                "--marginal_correction_strength",
                str(sample_cfg.get("correction_strength", 1.0)),
            ]
        )
        if sample_cfg.get("enable_marginal_correction", True):
            cmd.append("--enable_marginal_correction")

    print("[Sample] Running TabDiff:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    subprocess.run(cmd, cwd=paths.tabdiff_repo, check=True, env=env)

    # Expected sample path under TabDiff result directory.
    ckpt_dir = ckpt_path.parent
    result_dir = Path(str(ckpt_dir).replace("ckpt", "result"))
    sample_path = result_dir / "0" / "samples.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"Expected TabDiff samples at {sample_path}")

    paths.final_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_path, paths.final_output)

    # Validate schema on final output.
    with open(paths.sdnist_info, "r") as f:
        info = json.load(f)
    column_order = info["column_names"]
    final_df = pd.read_csv(paths.final_output)
    missing_cols = [c for c in column_order if c not in final_df.columns]
    if missing_cols:
        raise ValueError(f"Final synthetic missing columns: {missing_cols}")
    final_df = final_df[column_order]
    final_df.to_csv(paths.final_output, index=False)

    print(f"[Sample] Final synthetic saved to {paths.final_output}")

    compute_diagnostics(paths, final_df, column_order)


def compute_diagnostics(paths: PipelinePaths, final_df: pd.DataFrame, column_order) -> None:
    """Lightweight post-run diagnostics."""

    diagnostics = {}
    info = json.load(open(paths.sdnist_info))
    expected_cols = column_order

    diagnostics["schema_match"] = list(final_df.columns) == expected_cols
    diagnostics["missing_columns"] = [c for c in expected_cols if c not in final_df.columns]
    diagnostics["extra_columns"] = [c for c in final_df.columns if c not in expected_cols]
    diagnostics["row_count"] = len(final_df)
    diagnostics["nan_cells"] = int(final_df.isna().sum().sum())
    diagnostics["correction_cadence"] = cfg_sample_cadence(paths)

    # Marginal errors (1-way only)
    try:
        mwem_marg = json.load(open(paths.mwem_marginals))
        marginals = []
        entropies = []
        improvements = []  # defined as negative L1 error (higher is better)
        for m in mwem_marg.get("marginals", []):
            if len(m.get("columns", [])) != 1:
                continue
            col = m["columns"][0]
            target = m.get("probs", {})
            synth_counts = final_df[col].astype(str).value_counts(normalize=True)
            synth_probs = synth_counts.to_dict()
            keys = set(target.keys()) | set(synth_probs.keys())
            l1 = 0.0
            for k in keys:
                t = float(target.get(k, 0.0))
                s = float(synth_probs.get(k, 0.0))
                l1 += abs(t - s)
            ent = entropy_from_probs(target)
            entropies.append(ent)
            improvements.append(-l1)
            marginals.append(
                {
                    "column": col,
                    "target": target,
                    "synthetic": synth_probs,
                    "l1_error": l1,
                    "target_entropy": ent,
                }
            )
        diagnostics["marginals"] = marginals
        if len(entropies) >= 2:
            corr = pearson_corr(entropies, improvements)
            diagnostics["entropy_error_correlation"] = corr
    except Exception as e:
        diagnostics["marginals_error"] = str(e)

    diag_path = paths.workdir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"[Diagnostics] schema_match={diagnostics['schema_match']} rows={diagnostics['row_count']} nan_cells={diagnostics['nan_cells']}")
    if "marginals" in diagnostics:
        for m in diagnostics["marginals"]:
            print(f"[Diagnostics] col={m['column']} l1_error={m['l1_error']:.4f}")


def cfg_sample_cadence(paths: PipelinePaths) -> int:
    # Helper to read cadence from the saved YAML config if present.
    try:
        cfg = load_pipeline_config(paths.config_path)
        return int(cfg.get("tabdiff", {}).get("sample", {}).get("correction_cadence", -1))
    except Exception:
        return -1


def entropy_from_probs(prob_dict: Dict[str, float]) -> float:
    """Shannon entropy (natural log) from a dict of probabilities."""
    vals = [max(float(v), 1e-12) for v in prob_dict.values()]
    s = sum(vals)
    if s <= 0:
        return 0.0
    vals = [v / s for v in vals]
    import math
    return -sum(v * math.log(v) for v in vals)


def pearson_corr(x, y) -> float:
    import numpy as np
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    return float(num / den) if den != 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MWEM→TabDiff pipeline (scaffold)")
    parser.add_argument("--config", type=Path, default=Path("configs/mwem_tabdiff.yaml"))
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    paths = resolve_paths(cfg, args.config)
    paths.workdir.mkdir(parents=True, exist_ok=True)

    run_mwem(cfg, paths)
    stage_tabdiff_seed(cfg, paths)
    ckpt_path = train_tabdiff(cfg, paths)
    sample_with_correction(cfg, paths, ckpt_path)


if __name__ == "__main__":
    main()
