#!/usr/bin/env python3
"""Sampling-only categorical marginal logit guidance for the small TabDiff pilot.

This is a heuristic, not an exact conditional sampler or a guarantee of marginal
matching. It changes only clean categorical predictions for currently masked
entries, before TabDiff's native MDLM transition. It does not edit AGE, reorder
rows, train a model, or independently redraw completed categorical columns.

The caller must supply targets and reference probabilities. This module never
loads real data or estimates private statistics. Its correctness alone does not
establish differential privacy of a caller's complete pipeline.
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from types import MethodType

import numpy as np
import torch


def _probability_vector(value, label):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional vector")
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{label} must contain finite, strictly positive probabilities")
    if not np.isclose(result.sum(), 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{label} must already be normalized to sum to one")
    return result.copy()


def make_offsets(targets, reference, strength=1.0, max_log_ratio=2.0,
                 prob_floor=1e-8):
    """Return fixed, bounded logit offsets in each feature's public code order.

    Inputs must already be smoothed to positive normalized probabilities. The
    fixed floor only stabilizes the logarithm; no categories are inferred from
    a batch. Zero strength returns exact zero arrays after validating inputs.
    """
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")
    if not math.isfinite(max_log_ratio) or max_log_ratio <= 0:
        raise ValueError("max_log_ratio must be finite and positive")
    if not math.isfinite(prob_floor) or not 0 < prob_floor < 1:
        raise ValueError("prob_floor must be finite and strictly between zero and one")
    if len(targets) != len(reference):
        raise ValueError("targets and reference must have the same feature count")
    offsets = []
    for j, (target, baseline) in enumerate(zip(targets, reference)):
        p = _probability_vector(target, f"targets[{j}]")
        r = _probability_vector(baseline, f"reference[{j}]")
        if p.shape != r.shape:
            raise ValueError(f"Target/reference category counts differ at feature {j}")
        if strength == 0:
            offsets.append(np.zeros_like(p))
            continue
        ratio = np.log(np.maximum(p, prob_floor)) - np.log(np.maximum(r, prob_floor))
        offset = strength * np.clip(ratio, -max_log_ratio, max_log_ratio)
        if not np.isfinite(offset).all():
            raise ValueError("strength produced a nonfinite offset")
        offsets.append(offset)
    return offsets


@contextmanager
def guided_sampling(model, offsets):
    """Temporarily install a model-instance-local pre-MDLM guidance hook.

    Pass ``None``, ``[]``, or all-zero offsets for an exact no-op path. The
    original instance attribute, or its absence, is restored even on exceptions.
    This context does not change weights, requires_grad flags, mode, RNG state,
    numerical predictions, or the native categorical mask/unmask schedule.

    Stochastic TabDiff may itself remask an earlier categorical value. The hook
    preserves entries unmasked *at this native transition*, not all values ever
    sampled in the chain. Subsequent numerical predictions may indirectly change
    in response to guided categorical states.
    """
    categories = np.asarray(model.num_classes, dtype=np.int64)
    if categories.ndim != 1 or np.any(categories < 1):
        raise ValueError("Model must declare its positive categorical cardinalities")
    if not torch.equal(model.mask_index.detach().cpu(), torch.as_tensor(categories)):
        raise ValueError("Model mask indices do not match its category cardinalities")
    if offsets is None or len(offsets) == 0:
        vectors = [np.zeros(int(k), dtype=np.float64) for k in categories]
    else:
        if len(offsets) != len(categories):
            raise ValueError("Need one offset vector for each model categorical feature")
        vectors = []
        for j, (offset, count) in enumerate(zip(offsets, categories)):
            vector = np.asarray(offset, dtype=np.float64)
            if vector.shape != (int(count),) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid offset shape or value at feature {j}")
            vectors.append(vector.copy())
    active = [bool(np.any(v != 0)) for v in vectors]
    all_zero = not any(active)
    original_bound = model._subs_parameterization
    name = "_subs_parameterization"
    had_instance_attribute = name in vars(model)
    original_instance_attribute = vars(model).get(name)
    stats = {"hook_calls": 0, "masked_cells_adjusted": 0}

    def parameterize(instance, raw_logits, xt):
        stats["hook_calls"] += 1
        if all_zero:
            # No clone, addition, normalization, or random draw before the base
            # call: this is intentionally bit-for-bit the original code path.
            return original_bound(raw_logits, xt)
        width = int(np.sum(categories + 1))
        if raw_logits.ndim != 2 or raw_logits.shape[1] != width:
            raise ValueError("Unexpected concatenated categorical-logit shape")
        if xt.shape != (raw_logits.shape[0], len(categories)):
            raise ValueError("Unexpected current categorical-state shape")
        if not torch.isfinite(raw_logits).all():
            raise ValueError("Nonfinite model logits; no guided sampling performed")
        adjusted = raw_logits.clone()
        start = 0
        for j, (count, vector, enabled) in enumerate(zip(categories, vectors, active)):
            count = int(count)
            if enabled:
                masked = xt[:, j] == instance.mask_index[j]
                n_masked = int(masked.sum().item())
                if n_masked:
                    delta = torch.as_tensor(vector, dtype=raw_logits.dtype,
                                            device=raw_logits.device)
                    if not torch.isfinite(delta).all():
                        raise ValueError("Offset cannot be represented in the model dtype")
                    block = adjusted[:, start:start + count]
                    block[masked] = block[masked] + delta
                    stats["masked_cells_adjusted"] += n_masked
            # Skip the native mask logit as well as the real-category block.
            start += count + 1
        if not torch.isfinite(adjusted).all():
            raise ValueError("Guided logits are nonfinite")
        return original_bound(adjusted, xt)

    setattr(model, name, MethodType(parameterize, model))
    try:
        yield stats
    finally:
        if had_instance_attribute:
            setattr(model, name, original_instance_attribute)
        else:
            delattr(model, name)


def self_test():
    """Exercise the actual TabDiff hook on toy public domains, never real data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import local_tabdiff_smoke as smoke

    rng_state = torch.get_rng_state()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(4102)
        categories = np.array([2, 3, 4], dtype=np.int64)
        model = smoke.build_diffusion(categories, sampling_steps=10).eval()
        flags_before = [p.requires_grad for p in model.parameters()]
        weights_before = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
        targets = [np.full(k, 1 / k, dtype=np.float64) for k in categories]
        reference = [np.arange(1, k + 1, dtype=np.float64) for k in categories]
        reference = [v / v.sum() for v in reference]
        offsets = make_offsets(targets, reference)
        zero = make_offsets(targets, reference, strength=0)
        if not all(np.count_nonzero(v) == 0 for v in zero):
            raise AssertionError("Zero strength did not return exact zeros")

        torch.manual_seed(5721)
        original_samples = model.sample(16)
        torch.manual_seed(5721)
        with guided_sampling(model, zero) as zero_stats:
            zero_samples = model.sample(16)
        if not torch.equal(original_samples, zero_samples):
            raise AssertionError("Zero-strength full chain differs from original")
        if zero_stats["hook_calls"] != model.num_timesteps or zero_stats["masked_cells_adjusted"]:
            raise AssertionError("Zero-strength hook accounting is incorrect")

        raw = torch.randn(5, int(np.sum(categories + 1)))
        raw_before = raw.clone()
        xt = torch.tensor([[2, 3, 4], [0, 1, 2], [2, 0, 4],
                           [1, 3, 3], [0, 2, 4]], dtype=torch.long)
        xt_before = xt.clone()
        base_logits = model._subs_parameterization(raw, xt)
        with guided_sampling(model, offsets) as direct_stats:
            changed_logits = model._subs_parameterization(raw, xt)
        if not torch.equal(raw, raw_before) or not torch.equal(xt, xt_before):
            raise AssertionError("Guidance changed its input tensors")
        unmasked = xt != model.mask_index
        if not torch.equal(changed_logits[unmasked], base_logits[unmasked]):
            raise AssertionError("Guidance changed currently unmasked predictions")
        if torch.equal(changed_logits[~unmasked], base_logits[~unmasked]):
            raise AssertionError("Nonzero guidance had no effect on masked predictions")
        probabilities = changed_logits.exp()
        for j, count in enumerate(categories):
            if not torch.allclose(probabilities[:, j, :count].sum(-1),
                                  torch.ones(len(xt)), atol=1e-6, rtol=1e-6):
                raise AssertionError("Real-category probabilities do not normalize")
            if torch.count_nonzero(probabilities[:, j, count:]).item() != 0:
                raise AssertionError("Mask/dummy clean-prediction probability is nonzero")
        if direct_stats != {"hook_calls": 1, "masked_cells_adjusted": 7}:
            raise AssertionError("Masked-cell hook accounting is incorrect")

        with guided_sampling(model, None) as none_stats:
            no_offset_logits = model._subs_parameterization(raw, xt)
        if not torch.equal(no_offset_logits, base_logits) or none_stats["masked_cells_adjusted"]:
            raise AssertionError("No-offset path is not identical to baseline")

        if "_subs_parameterization" in vars(model):
            raise AssertionError("Hook left a new instance attribute behind")
        try:
            with guided_sampling(model, offsets):
                raise RuntimeError("intentional restoration test")
        except RuntimeError as error:
            if str(error) != "intentional restoration test":
                raise
        if "_subs_parameterization" in vars(model):
            raise AssertionError("Exception did not restore class-method lookup")

        # An existing instance override must be restored as the same object.
        original = model._subs_parameterization
        existing = MethodType(lambda instance, logits, states: original(logits, states), model)
        model._subs_parameterization = existing
        try:
            with guided_sampling(model, offsets):
                raise RuntimeError("intentional existing-attribute restoration test")
        except RuntimeError as error:
            if str(error) != "intentional existing-attribute restoration test":
                raise
        if vars(model)["_subs_parameterization"] is not existing:
            raise AssertionError("Existing instance override was not restored exactly")
        delattr(model, "_subs_parameterization")

        torch.manual_seed(5721)
        with guided_sampling(model, offsets) as active_stats:
            active_samples = model.sample(16)
        if active_stats["hook_calls"] != model.num_timesteps or active_stats["masked_cells_adjusted"] <= 0:
            raise AssertionError("Full-chain sampling did not use the active hook")
        if not torch.isfinite(active_samples).all():
            raise AssertionError("Guided toy sample contains nonfinite values")
        cats = active_samples[:, 1:]
        if torch.any(cats < 0) or torch.any(cats >= torch.as_tensor(categories)):
            raise AssertionError("Guided toy sample contains invalid categorical codes")
        if flags_before != [p.requires_grad for p in model.parameters()]:
            raise AssertionError("Guidance changed requires_grad flags")
        if any(not torch.equal(value, weights_before[name])
               for name, value in model.state_dict().items()):
            raise AssertionError("Guidance changed model weights/state")

        invalid_cases = [
            ([np.array([0.0, 1.0])], [np.array([0.5, 0.5])]),
            ([np.array([0.1, 0.1])], [np.array([0.5, 0.5])]),
            ([np.array([float("nan"), 0.5])], [np.array([0.5, 0.5])]),
            ([np.array([0.5, 0.5])], [np.array([1.0])]),
        ]
        for target, baseline in invalid_cases:
            try:
                make_offsets(target, baseline)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid probability input was accepted")
        return {"passed": True, "source": "generated toy validation only",
                "zero_strength_full_chain_bitwise_equal": True,
                "unmasked_predictions_preserved": True,
                "mask_and_dummy_probabilities_zero": True,
                "raw_inputs_unchanged": True, "probabilities_normalized": True,
                "exception_restoration": True, "weights_and_grad_flags_unchanged": True,
                "active_hook_calls": active_stats["hook_calls"],
                "active_masked_cells_adjusted": active_stats["masked_cells_adjusted"]}
    finally:
        torch.set_rng_state(rng_state)
        torch.set_num_threads(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="Run toy-only checks against this repository's actual TabDiff")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("This helper is imported by the runner; use --self-test for standalone checks")
    print(json.dumps(self_test(), indent=2))
