"""Public fake-backend contracts for the saved-model guidance intervention.

These tests validate orchestration, masks, replay gates and failure handling.
They do not load private records/checkpoints or establish actual CUDA replay.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_prototype2_guidance_sweep as sweep


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
CATEGORIES = np.asarray([3, 9, 6, 44, 531, 266], dtype=np.int64)
SEEDS = (12026, 12126, 12226)


def historical_offsets(targets, reference, strength=1.0, max_log_ratio=2.0,
                       prob_floor=1e-8):
    """Public numerical oracle for the pinned strength-times-clipped-log rule."""
    return [strength * np.clip(np.log(np.maximum(t, prob_floor)) -
                               np.log(np.maximum(r, prob_floor)),
                               -max_log_ratio, max_log_ratio)
            for t, r in zip(targets, reference)]


def fixture_reference(array):
    vectors = [np.bincount(array[:, j + 1].astype(np.int64), minlength=int(k)).astype(np.float64)
               for j, k in enumerate(CATEGORIES)]
    return [(np.maximum(v, 0.0) + 0.001) / (np.maximum(v, 0.0) + 0.001).sum()
            for v in vectors]


def base_array(seed, count):
    rows = np.arange(count)
    table = np.empty((count, 7), dtype=np.float32)
    table[:, 0] = np.linspace(-0.63, 0.37, count, dtype=np.float32)
    # Deliberate finite tails: replay/retention must not clamp or filter AGE.
    table[0, 0], table[-1, 0] = -1.5, 2.5
    table[17, 0] = 0.0  # permits a signed-zero bitwise replay regression
    for j, k in enumerate(CATEGORIES):
        table[:, j + 1] = (rows * (j + 1) + seed + 11 * j) % k
    return table


class FakeBackend:
    """Deterministic source-preserving backend with observable call semantics."""

    def __init__(self):
        self.calls = []
        self.normalized = []
        self.saved = []
        self.fingerprint_value = "public-frozen-model-fingerprint"
        self.mismatch_zero_rng = False
        self.mutate_on_active = False
        self.fail_on_active = False
        self.before_first_save = None

    make_offsets = staticmethod(historical_offsets)

    @staticmethod
    def render(seed, count, offsets=None):
        result = base_array(seed, count)
        if offsets is not None:
            for j, vector in enumerate(offsets):
                if np.any(vector != 0):
                    # Explicit fixture response, unrelated to TabDiff dynamics.
                    shift = 1 + int(round(float(np.sum(np.abs(vector))) * 19))
                    result[:, j + 1] = (result[:, j + 1] + shift) % CATEGORIES[j]
        return result

    def sample(self, seed, count, offsets=None):
        frozen = None if offsets is None else [np.asarray(v).copy() for v in offsets]
        active = offsets is not None and any(np.any(v != 0) for v in offsets)
        zero = offsets is not None and not active
        self.calls.append({"seed": seed, "count": count, "offsets": frozen,
                           "active": active, "zero": zero})
        if active and self.fail_on_active:
            raise RuntimeError("public fixture active-sampler failure")
        if active and self.mutate_on_active:
            self.fingerprint_value = "changed-public-model-fingerprint"
        hooks = None if offsets is None else {
            "hook_calls": (count + 63) // 64 * 50,
            "masked_cells_adjusted": count * sum(bool(np.any(v != 0)) for v in offsets),
        }
        rng = ("public-consumed-CPU-and-selected-CUDA", seed, count)
        if zero and self.mismatch_zero_rng:
            rng += ("unexpected-consumption",)
        return self.render(seed, count, offsets), hooks, rng

    def normalize_reference(self, array):
        self.normalized.append(np.array(array, copy=True))
        return fixture_reference(array)

    def same_rng(self, left, right):
        return left == right

    def fingerprint(self):
        return self.fingerprint_value

    def save_sample(self, path, array):
        if self.before_first_save is None:
            self.before_first_save = copy.deepcopy(self.calls)
        path = Path(path)
        self.saved.append((path, np.array(array, copy=True)))
        with path.open("xb") as stream:
            np.savez_compressed(stream, X_synthetic=array,
                                categories=CATEGORIES, features=FEATURES)


@pytest.fixture
def case(tmp_path):
    backend = FakeBackend()
    targets = [np.arange(1, int(k) + 1, dtype=np.float64) for k in CATEGORIES]
    targets = [v / v.sum() for v in targets]
    reference = fixture_reference(base_array(13026, 4096))
    active = historical_offsets(targets, reference)
    historical = {}
    for draw, seed in enumerate(SEEDS, 1):
        historical[("B_private", draw)] = backend.render(seed, 512)
        historical[("C", draw)] = backend.render(seed, 512, active)
    out = tmp_path / "new-public-output"
    out.mkdir()
    return backend, historical, targets, reference, out


def assert_failed_before_grid(case):
    backend, historical, targets, reference, out = case
    assert backend.saved == []
    assert not list(out.rglob("*.npz"))
    assert not list(out.rglob("*manifest.json"))
    assert not list(out.glob("COMPLETED*"))
    replay = out / "replay.json"
    assert replay.is_file()
    assert isinstance(json.loads(replay.read_text()), dict)
    historical_active = historical_offsets(targets, reference)
    for call in backend.calls:
        if call["count"] != 512 or not call["active"]:
            continue
        assert all(np.array_equal(a, b) for a, b in zip(call["offsets"], historical_active))


def test_constants_and_prespecified_grid_have_one_zero_and_nine_unique_cached_conditions():
    assert list(sweep.FEATURES) == FEATURES
    assert np.array_equal(sweep.CATEGORIES, CATEGORIES)
    assert tuple(sweep.GENERATION_SEEDS) == SEEDS
    conditions = sweep.build_conditions()
    assert len(conditions) == 10
    assert len({c["id"] for c in conditions}) == 10
    zero, = [c for c in conditions if c["strength"] == 0]
    assert {key: zero[key] for key in ("id", "target_kind", "strength", "active_features")} == {
        "id": "zero", "target_kind": None, "strength": 0, "active_features": []}
    masks = {"all": FEATURES[1:], "race_hispan": ["RACE", "HISPAN"], "occ_ind": ["OCC", "IND"]}
    indexed = {c["id"]: c for c in conditions}
    for mask, names in masks.items():
        for code, strength in (("025", 0.25), ("050", 0.5), ("100", 1.0)):
            c = indexed[f"cached_{mask}_s{code}"]
            assert c["target_kind"] == "cached" and c["strength"] == strength
            assert c["active_features"] == names
    extended = sweep.build_conditions(include_train_oracle=True)
    assert len(extended) == len({c["id"] for c in extended}) == 19
    assert sum(c["strength"] == 0 for c in extended) == 1
    assert sum(c["target_kind"] == "train_oracle" for c in extended) == 9


def test_masks_keep_all_six_vectors_in_order_and_do_not_mutate_inputs(case):
    _, _, targets, reference, _ = case
    targets_before, reference_before = copy.deepcopy(targets), copy.deepcopy(reference)
    calls = []

    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return historical_offsets(*args, **kwargs)

    for condition in sweep.build_conditions():
        values = sweep.make_condition_offsets(condition, targets, reference, observed)
        assert len(values) == 6
        full = historical_offsets(targets, reference, condition["strength"])
        for i, (name, k) in enumerate(zip(FEATURES[1:], CATEGORIES)):
            assert values[i].shape == (k,)
            if name in condition["active_features"]:
                np.testing.assert_array_equal(values[i], full[i])
            else:
                np.testing.assert_array_equal(values[i], np.zeros(int(k)))
    for args, kwargs in calls:
        assert kwargs.get("max_log_ratio") == 2.0
        assert kwargs.get("prob_floor") == 1e-8
    for before, after in zip(targets_before + reference_before, targets + reference):
        np.testing.assert_array_equal(before, after)


def test_strength_multiplies_after_log_ratio_cap(case):
    _, _, targets, reference, _ = case
    condition = {"id": "public-cap-probe", "target_kind": "cached", "strength": 0.25,
                 "active_features": FEATURES[1:]}
    values = sweep.make_condition_offsets(condition, targets, reference, historical_offsets)
    # The synthetic reference omits some public codes; at least one log-ratio
    # exceeds 2 and therefore has exactly 0.5 magnitude at strength 0.25.
    assert any(np.any(np.abs(v) == 0.5) for v in values)
    assert max(float(np.max(np.abs(v))) for v in values) <= 0.5


@pytest.mark.parametrize("arm,draw", [("B_private", 1), ("C", 1), ("C", 3)])
def test_any_historical_array_mismatch_stops_before_grid_and_preserves_failure(case, arm, draw):
    backend, historical, targets, _, out = case
    historical[(arm, draw)][17, 0] += np.float32(0.125)
    with pytest.raises(sweep.ReplayMismatch):
        sweep.run_sampling(backend, historical, targets, out)
    assert_failed_before_grid(case)


def test_equal_zero_guidance_arrays_with_different_consumed_RNG_fail(case):
    backend, historical, targets, _, out = case
    backend.mismatch_zero_rng = True
    with pytest.raises(sweep.ReplayMismatch):
        sweep.run_sampling(backend, historical, targets, out)
    assert_failed_before_grid(case)


@pytest.mark.parametrize("change", ["dtype", "signed_zero"])
def test_numerically_equal_but_different_historical_array_bytes_fail(case, change):
    backend, historical, targets, _, out = case
    key = ("B_private", 1)
    original = historical[key].copy()
    if change == "dtype":
        historical[key] = historical[key].astype(np.float64)
    else:
        historical[key][17, 0] = np.float32(-0.0)
    assert np.array_equal(original, historical[key])
    with pytest.raises(sweep.ReplayMismatch):
        sweep.run_sampling(backend, historical, targets, out)
    assert_failed_before_grid(case)


def test_sampler_exception_is_not_retried_or_reported_as_success(case):
    backend, historical, targets, _, out = case
    backend.fail_on_active = True
    with pytest.raises(RuntimeError, match="public fixture active-sampler failure"):
        sweep.run_sampling(backend, historical, targets, out)
    assert sum(call["active"] for call in backend.calls) == 1
    assert backend.saved == []
    assert not list(out.glob("COMPLETED*"))


def test_public_sweep_gates_all_draws_uses_one_reference_and_retains_all_conditions(case):
    backend, historical, targets, reference, out = case
    before = {key: value.copy() for key, value in historical.items()}
    target_before = copy.deepcopy(targets)
    result = sweep.run_sampling(backend, historical, targets, out)
    references = [c for c in backend.calls if c["count"] == 4096]
    assert len(references) == 1
    assert references[0]["seed"] == 13026 and references[0]["offsets"] is None
    assert len(backend.normalized) == 1
    np.testing.assert_array_equal(backend.normalized[0], base_array(13026, 4096))
    assert all(v.shape == (k,) and np.all(v > 0) and np.isclose(v.sum(), 1)
               for v, k in zip(reference, CATEGORIES))
    before_save = backend.before_first_save
    for seed in SEEDS:
        assert any(c["seed"] == seed and c["count"] == 512 and c["offsets"] is None for c in before_save)
        assert any(c["seed"] == seed and c["count"] == 512 and c["zero"] for c in before_save)
        assert any(c["seed"] == seed and c["count"] == 512 and c["active"] for c in before_save)
    assert all(c["seed"] in SEEDS and c["count"] == 512 for c in backend.calls if c["count"] != 4096)
    assert len(result["conditions"]) == 10
    assert len(result["samples"]) == len(backend.saved) == 30
    assert result["model_unchanged"] is True
    assert result["new_training_fits"] == result["new_marginal_releases"] == 0
    assert len(result["reference_sha256"]) == 64
    int(result["reference_sha256"], 16)
    assert result["replay"]
    assert len(list(out.rglob("*manifest.json"))) == 1
    assert not list(out.glob("COMPLETED*"))  # main finalizes only after post-hash checks.
    for path, array in backend.saved:
        assert path.is_file() and array.dtype == np.float32 and array.shape == (512, 7)
        assert array[0, 0] == -1.5 and array[-1, 0] == 2.5
    for key, value in before.items():
        np.testing.assert_array_equal(value, historical[key])
    for before_values, values in zip(target_before, targets):
        np.testing.assert_array_equal(before_values, values)


def test_condition_order_does_not_change_saved_array_multiset(case, monkeypatch, tmp_path):
    backend, historical, targets, _, out = case
    sweep.run_sampling(backend, historical, targets, out)
    expected = sorted(hashlib.sha256(array.tobytes()).hexdigest() for _, array in backend.saved)
    original_builder = sweep.build_conditions

    def reversed_builder(include_train_oracle=False):
        return list(reversed(original_builder(include_train_oracle=include_train_oracle)))

    monkeypatch.setattr(sweep, "build_conditions", reversed_builder)
    reordered_out = tmp_path / "reordered-public-output"
    reordered_out.mkdir()
    reordered = FakeBackend()
    sweep.run_sampling(reordered, historical, targets, reordered_out)
    actual = sorted(hashlib.sha256(array.tobytes()).hexdigest() for _, array in reordered.saved)
    assert actual == expected


def test_replay_mode_retains_only_six_successful_B_and_C_replays(case):
    backend, historical, targets, _, out = case
    result = sweep.run_sampling(backend, historical, targets, out, mode="replay")
    assert {c["id"] for c in result["conditions"]} == {"zero", "cached_all_s100"}
    assert len(result["samples"]) == len(backend.saved) == 6
    assert all(record["reused_successful_replay"] for record in result["samples"])
    assert len(backend.calls) == 10  # one reference plus B/zero/C for three draws
    assert not list(out.glob("COMPLETED*"))


def test_model_mutation_cannot_produce_a_successful_result(case):
    backend, historical, targets, _, out = case
    backend.mutate_on_active = True
    with pytest.raises((ValueError, RuntimeError)):
        sweep.run_sampling(backend, historical, targets, out)
    assert not list(out.glob("COMPLETED*"))


def test_optional_public_oracle_adds_nine_conditions_without_new_target_release(case):
    backend, historical, targets, _, out = case
    oracle = [np.full(int(k), 1 / k, dtype=np.float64) for k in CATEGORIES]
    result = sweep.run_sampling(backend, historical, targets, out, oracle_targets=oracle)
    assert len(result["conditions"]) == 19
    assert len(result["samples"]) == len(backend.saved) == 57
    assert result["new_training_fits"] == result["new_marginal_releases"] == 0
    assert len([c for c in backend.calls if c["count"] == 4096]) == 1


@pytest.mark.parametrize("postcheck_fails", [False, True])
def test_main_completes_only_after_input_postcheck_and_hashes_reference(case, monkeypatch, tmp_path,
                                                                     postcheck_fails):
    backend, historical, targets, _, _ = case
    out = tmp_path / "public-cli-output"
    context = {"status": "public-generated-fixture", "no_private_inputs": True}
    monkeypatch.setattr(sweep, "verify_context", lambda *args: context)
    backend.cached_targets = targets
    backend.load_historical_samples = lambda: historical
    backend.provenance = {"public_fixture": True, "backend": "deterministic test fake"}
    postchecks = []

    def verify_unchanged():
        postchecks.append(True)
        assert not (out / "COMPLETED.json").exists()
        assert not (out / "results.json").exists()
        if postcheck_fails:
            raise ValueError("public fixture target/source postcheck changed")
        return {"all_verified_files_unchanged": True}

    def backend_factory(received, acknowledge_trusted_checkpoint, allow_oracle):
        assert received == context and acknowledge_trusted_checkpoint is True
        assert allow_oracle is False
        return backend

    backend.verify_unchanged = verify_unchanged
    monkeypatch.setattr(sweep, "HistoricalGPUBackend", backend_factory)
    # The fake context gate deliberately stands in for real inputs. No fixture
    # file is presented as a historical checkpoint or passed to torch.load.
    previous_umask = os.umask(0o077)
    try:
        code = sweep.main([
            "--project-root", str(tmp_path / "unused-project"),
            "--run-dir", str(tmp_path / "unused-history"),
            "--data-npz", str(tmp_path / "unused-data.npz"),
            "--schema", str(tmp_path / "unused-schema.json"),
            "--out", str(out), "--mode", "replay", "--acknowledge-trusted-checkpoint",
        ])
    finally:
        os.umask(previous_umask)
    assert postchecks == [True]
    assert len(list((out / "samples").glob("*.npz"))) == 6
    assert (out / "sample_manifest.json").is_file()
    if postcheck_fails:
        assert code == 2
        assert not (out / "COMPLETED.json").exists()
        failed = json.loads((out / "FAILED.json").read_text())
        assert failed["status"] == "failed" and failed["partial_outputs_retained"] is True
        assert "postcheck changed" in failed["error"]
    else:
        assert code == 0 and not (out / "FAILED.json").exists()
        complete = json.loads((out / "COMPLETED.json").read_text())
        assert complete["status"] == "complete" and complete["inputs_unchanged"] is True
        assert "reference.json" in complete["artifact_sha256"]
        for name, digest in complete["artifact_sha256"].items():
            assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_source_verification_hashes_actual_bytes_and_rejects_missing_changed_or_unsafe_paths(tmp_path):
    root = tmp_path / "public-source"
    root.mkdir()
    member = root / "module.py"
    member.write_text("# public source fixture\n")
    digest = hashlib.sha256(member.read_bytes()).hexdigest()
    verified = sweep.verify_source_group(root, {"module.py": digest})
    assert len(verified) == 1 and digest in verified.values()
    before = member.read_bytes()
    with pytest.raises(ValueError):
        sweep.verify_source_group(root, {"missing.py": digest})
    with pytest.raises(ValueError):
        sweep.verify_source_group(root, {"module.py": "0" * 64})
    outside = tmp_path / "outside.py"
    outside.write_bytes(before)
    for unsafe in ("../outside.py", str(outside.resolve())):
        with pytest.raises(ValueError):
            sweep.verify_source_group(root, {unsafe: digest})
    (root / "escape.py").symlink_to(outside)
    with pytest.raises(ValueError):
        sweep.verify_source_group(root, {"escape.py": digest})
    assert member.read_bytes() == before
