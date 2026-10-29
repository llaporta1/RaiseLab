"""
Lightweight marginal-correction scaffold for TabDiff sampling.

Goal: Nudge sampled batches toward MWEM-produced target marginals without
changing the diffusion model weights. The actual projection math is left as
TODOs; this module only defines interfaces and expected data shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import json
import numpy as np
import torch


@dataclass
class MarginalSpec:
    """Single marginal target over a column subset.

    Attributes
    ----------
    columns: list of column names in the SDNIST/TabDiff schema order.
    probabilities: mapping from joined-key ("a|b") to target probability.
    """

    columns: List[str]
    probabilities: Dict[str, float]


@dataclass
class CorrectionConfig:
    """Runtime knobs for correction strength and cadence."""

    cadence: int = 2          # apply every k reverse steps
    max_iters: int = 1        # projection sweeps per application
    step_size: float = 0.1    # interpolation weight toward target
    strength: float = 1.0     # overall scaling of step_size
    clip_to_valid: bool = True


class MarginalCorrector:
    """Callable correction helper to plug into the diffusion sampling loop.

    Expected usage inside UnifiedCtimeDiffusion.sample/edm_update:
        corrector = MarginalCorrector.from_json(mwem_marginals_path, schema)
        for t in reverse_steps:
            x_num, x_cat = model_step(...)
            x_num, x_cat = corrector.project_batch(x_num, x_cat, step=t)
    """

    def __init__(self, marginals: List[MarginalSpec], schema: List[str], cat_col_idx: List[int], cfg: CorrectionConfig, enabled: bool = True):
        self.marginals = marginals
        self.schema = schema
        self.cat_col_idx = cat_col_idx
        self.cfg = cfg
        self.enabled = enabled

    @classmethod
    def from_json(cls, path: Path, schema: List[str], cat_col_idx: List[int], cfg: CorrectionConfig | None = None, enabled: bool = True) -> "MarginalCorrector":
        """Load MWEM marginals saved as JSON.

        Expected JSON structure:
        {
            "marginals": [
                {"columns": ["col_a"], "probs": {"0": 0.4, "1": 0.6}},
                {"columns": ["col_a", "col_b"], "probs": {"0|0": 0.2, ...}}
            ]
        }
        """

        cfg = cfg or CorrectionConfig()
        with open(path, "r") as f:
            raw = json.load(f)
        marginals = [MarginalSpec(m["columns"], m["probs"]) for m in raw.get("marginals", [])]
        return cls(marginals, schema, cat_col_idx, cfg, enabled=enabled)

    def project_batch(self, x_num: torch.Tensor, x_cat: torch.Tensor, step: int, total_steps: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adjust a batch toward target 1-way categorical marginals (prototype).

        - Works purely on the categorical part (x_cat) since logits are unavailable here.
        - Uses convex interpolation between current batch frequencies and MWEM targets,
          then resamples the column from the blended distribution.
        - Leaves numerical features unchanged.
        - Keeps corrections lightweight to avoid destabilizing sampling.
        """

        if (not self.enabled) or self.cfg.cadence <= 0:
            return x_num, x_cat

        if step % self.cfg.cadence != 0:
            return x_num, x_cat

        device = x_cat.device
        x_cat_out = x_cat.clone()

        alpha = min(1.0, max(0.0, self.cfg.step_size * self.cfg.strength))

        for m in self.marginals:
            # Only 1-way for now; skip anything longer.
            if len(m.columns) != 1:
                continue
            col_name = m.columns[0]
            if col_name not in self.schema:
                continue

            schema_idx = self.schema.index(col_name)
            try:
                cat_pos = self.cat_col_idx.index(schema_idx)
            except ValueError:
                # Not a categorical column in this batch representation.
                continue

            col_vals = x_cat_out[:, cat_pos]
            # Compute domain size from target keys and current batch.
            target_probs = self._normalize_target_probs(m.probabilities)
            if not target_probs:
                continue
            target_ids = torch.tensor(sorted(target_probs.keys()), device=device)
            max_allowed = int(col_vals.max().item())
            # Restrict targets to observed domain to avoid producing invalid category ids.
            target_probs = {k: v for k, v in target_probs.items() if k <= max_allowed}
            if not target_probs:
                continue
            target_ids = torch.tensor(sorted(target_probs.keys()), device=device)
            max_id = max_allowed
            counts = torch.bincount(col_vals.clamp_min(0), minlength=max_id + 1).float()
            total = counts.sum().clamp(min=1.0)
            current_probs = counts / total

            # Align target vector to length max_id+1.
            target_vec = torch.zeros_like(current_probs)
            for k, v in target_probs.items():
                if k < len(target_vec):
                    target_vec[k] = v
            # Normalize target_vec in case of slight drift.
            target_vec = target_vec / target_vec.sum().clamp(min=1e-12)

            blended = (1 - alpha) * current_probs + alpha * target_vec
            blended = blended / blended.sum().clamp(min=1e-12)

            # Resample column categories from blended distribution.
            cat_dist = torch.distributions.Categorical(probs=blended)
            new_vals = cat_dist.sample((col_vals.shape[0],)).to(device)
            x_cat_out[:, cat_pos] = new_vals

            # Log simple diagnostics
            self._log_marginal(col_name, target_vec, current_probs, blended, step)

        return x_num, x_cat_out

    def _normalize_target_probs(self, probs: Dict[str, float]) -> Dict[int, float]:
        out = {}
        for k, v in probs.items():
            try:
                out[int(k)] = float(v)
            except ValueError:
                # If key was pipe-joined, skip for 1-way handling.
                continue
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    def _log_marginal(self, name: str, target_vec: torch.Tensor, current_probs: torch.Tensor, blended: torch.Tensor, step: int) -> None:
        # Lightweight logging; prints top categories by probability.
        def topn(t):
            vals, idx = torch.topk(t, k=min(5, len(t)))
            return {int(i): float(v) for v, i in zip(vals, idx)}

        print(f"[Correction][step={step}] col={name} target_top={topn(target_vec)} current_top={topn(current_probs)} blended_top={topn(blended)}")


def load_schema_from_info(info_path: Path) -> Tuple[List[str], List[int]]:
    """Extract column order and categorical column indices from info.json."""

    with open(info_path, "r") as f:
        info = json.load(f)
    return info.get("column_names", []), info.get("cat_col_idx", [])


def attach_corrector(diffusion_model: Any, corrector: MarginalCorrector) -> None:
    """Placeholder hook for wiring the corrector into the sampler.

    When implementing, consider:
    - Adding an optional callback argument to UnifiedCtimeDiffusion.sample/edm_update.
    - Applying correction after each denoise step, before unmasking categorical tokens.
    - Ensuring gradients are not tracked (pure inference-time projection).
    """

    # TODO: monkey-patch or extend the sampler to call corrector.project_batch(...).
    pass
