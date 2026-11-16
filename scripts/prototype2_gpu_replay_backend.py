#!/usr/bin/env python3
"""Verified, sampling-only access to the retained Prototype-2 GPU B model.

The file gate uses only the standard library. Numerical dependencies and the
trusted historical checkpoint are loaded only by HistoricalGPUBackend, inside
the original Slurm/CUDA environment. No training or new private-release API is
exposed; an exact TRAIN oracle requires separate nonprivate authorization.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.machinery
import io
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType


JOB_ID = "22692665"
PLAN_SHA256 = "0cd082d709c2934254e21c55e29344d7016125e12651b212a4c0790dd53ec461"
DATA_SHA256 = "677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906"
SCHEMA_SHA256 = "3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73"
COMPLETION_SHA256 = "b5fd75d4e7f7923c56753660ee0f8b29bd8b02b10dd1e9c76bd20f4f3fe54ac4"
CHECKPOINT_SHA256 = "cd1e26d25caf0157b4b8ec12a2382e22938bae458d94949ad56c991961cfca46"
TARGET_SHA256 = "38003fd8c187493f83e3210fcd541f193d04af6eaaaf2e40f0f0a6ff5914bd05"
FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
CATEGORIES = [3, 9, 6, 44, 531, 266]
GENERATION_SEEDS = [12026, 12126, 12226]
MAX_FILE_BYTES = 64 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _root(path):
    path = Path(path).expanduser()
    _require(not path.is_symlink(), f"Symlink directory is not allowed: {path}")
    result = path.resolve(strict=True)
    _require(result.is_dir(), f"Expected directory: {result}")
    return result


def _member(root, relative):
    _require(isinstance(relative, str) and relative and "\\" not in relative,
             "Expected a safe relative file name")
    rel = Path(relative)
    _require(not rel.is_absolute() and all(p not in ("", ".", "..") for p in relative.split("/")),
             f"Unsafe relative path: {relative}")
    current = root
    for part in rel.parts:
        current = current / part
        _require(not current.is_symlink(), f"Symlink source/artifact is not allowed: {current}")
    _require(current.is_file() and current.resolve(strict=True).is_relative_to(root),
             f"Missing or escaping file: {relative}")
    return current


def _bytes(path, expected=None, limit=MAX_FILE_BYTES):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"Missing regular file: {path}")
    _require(path.resolve(strict=True) == path.absolute(),
             f"File path contains a symlink or changed directory: {path}")
    before = path.stat()
    _require(0 < before.st_size <= limit, f"Unexpected file size: {path}")
    data = path.read_bytes()
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _require(signature(before) == signature(after) and len(data) == before.st_size,
             f"File changed during verification: {path}")
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                 f"Invalid digest for {path}")
        _require(digest == expected, f"Historical file hash mismatch: {path}")
    return data, digest


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path, expected=None):
    data, digest = _bytes(path, expected, 32 * 1024 * 1024)
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(data, object_pairs_hook=_pairs, parse_constant=reject)
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value, digest


def verify_source_group(root, mapping):
    """Hash a declared relative-path group, rejecting symlinks and traversal."""
    root = _root(root)
    _require(isinstance(mapping, dict) and mapping, "Expected a nonempty source hash map")
    return {str(path): _bytes(path, digest, 2 * 1024 * 1024)[1]
            for name, digest in mapping.items() for path in [_member(root, name)]}


def verify_context(project_root, run_dir, data_npz, schema):
    """Read-only, stdlib-only historical file gate; suitable for a login node."""
    project, run = _root(project_root), _root(run_dir)
    plan_path = _member(run, "predeclared_plan.json")
    plan, _ = _json(plan_path, PLAN_SHA256)
    _require(plan.get("job_id") == JOB_ID, "Expected historical GPU job 22692665")
    config = plan["config"]
    _require(plan["training_replicates"] == 1 and plan["target_repeats"] == 1
             and plan["generation_repeats"] == 3 and plan["samples_per_arm"] == 512
             and plan["sample_batch"] == 64 and plan["reference_samples"] == 4096
             and plan["reference_seed"] == 13026
             and plan["generation_seeds"] == GENERATION_SEEDS,
             "Historical fit/reference/generation protocol differs")
    _require(config["features"] == FEATURES and config["categories"] == CATEGORIES
             and config["steps"] == 2000 and config["sampling_steps"] == 50
             and config["model_seed"] == 11026 and config["batch_size"] == 16
             and config["fixed_schedules"] is True,
             "Historical model configuration differs")
    _require(plan["dataset"] == {"train_rows": 2048, "holdout_rows": 512,
                                 "categorical_domains": CATEGORIES}
             and plan["guidance"] == {"strength": 1.0, "max_log_ratio": 2.0,
                                      "features": FEATURES[1:]},
             "Historical domain/guidance contract differs")
    sources = verify_source_group(project, plan["original_source_sha256"])
    sources.update(verify_source_group(project / "gpu-pilot-v1", plan["parent_pilot_kit_sha256"]))
    sources.update(verify_source_group(project / "gpu-longrun-v1p1", plan["new_runner_kit_sha256"]))
    for name in ("tabdiff/__init__.py", "tabdiff/models/__init__.py", "tabdiff/modules/__init__.py"):
        _require(not (project / name).exists() and not (project / name).is_symlink(),
                 f"Unexpected initializer in historical namespace package: {name}")
    marker_path = _member(run, "COMPLETED.json")
    marker, marker_hash = _json(marker_path, COMPLETION_SHA256)
    _require(not (run / "FAILED.json").exists() and not (run / "FAILED.json").is_symlink(),
             "Historical run has a failure marker")
    _require(marker.get("result") == "LONG GPU EXPERIMENT PASSED"
             and marker.get("job_id") == JOB_ID and marker.get("originals_unchanged") is True,
             "Historical completion metadata differs")
    mapping = marker.get("artifact_sha256")
    _require(isinstance(mapping, dict), "Missing completion artifact map")
    required = {"predeclared_plan.json", "metrics.json", "training_metadata.json",
                "B_private_final.pt", "dp_targets_01.json"} | {
                    f"{arm}_generation_{draw:02d}.npz" for arm in ("B_private", "C")
                    for draw in (1, 2, 3)}
    _require(required <= set(mapping), "Completion map omits required B/C replay artifacts")
    artifacts = {str(path): _bytes(path, digest)[1]
                 for name, digest in mapping.items() for path in [_member(run, name)]}
    artifacts[str(marker_path)] = marker_hash
    _require(artifacts[str(plan_path)] == PLAN_SHA256
             and artifacts[str(run / "B_private_final.pt")] == CHECKPOINT_SHA256
             and artifacts[str(run / "dp_targets_01.json")] == TARGET_SHA256,
             "Historical B/target identity differs")
    inputs = {}
    paths = []
    for raw, expected in ((data_npz, DATA_SHA256), (schema, SCHEMA_SHA256)):
        path = Path(raw).expanduser()
        _require(not path.is_symlink(), f"Symlink input is not allowed: {path}")
        path = path.resolve(strict=True)
        inputs[str(path)] = _bytes(path, expected)[1]
        paths.append(path)
    for suffix, digest in (("pilot_data.npz", DATA_SHA256), ("schema.json", SCHEMA_SHA256)):
        matches = [v for k, v in plan["input_and_original_artifact_sha256"].items()
                   if k.endswith("/ipums_2048/" + suffix)]
        _require(matches == [digest], f"Plan does not identify the exact input {suffix}")
    metrics, _ = _json(run / "metrics.json", artifacts[str(run / "metrics.json")])
    _require(metrics.get("job_id") == JOB_ID and metrics.get("plan") == plan
             and metrics.get("target_file_sha256") == TARGET_SHA256
             and metrics.get("checkpoint_file_sha256", {}).get("B_private") == CHECKPOINT_SHA256,
             "Historical metrics disagree with the plan/checkpoint/target")
    for arm in ("B_private", "C"):
        for draw, seed in enumerate(GENERATION_SEEDS, 1):
            name = f"{arm}_generation_{draw:02d}.npz"
            _require(metrics["sample_file_sha256"].get(name) == mapping[name],
                     f"Recorded sample digest differs: {name}")
            entry = metrics["evaluations"][arm][draw - 1]
            _require(entry.get("generation_seed") == seed
                     and (arm != "C" or entry.get("target_index") == 1),
                     f"Historical draw/target mapping differs: {name}")
    return {"status": "verified_exact_historical_gpu_replay_context", "plan": plan,
            "project_root": str(project), "run_dir": str(run),
            "data_npz": str(paths[0]), "schema": str(paths[1]),
            "source_file_sha256": sources, "artifact_file_sha256": artifacts,
            "input_file_sha256": inputs,
            "verified_file_sha256": {**sources, **artifacts, **inputs}}


def _namespace(name, path):
    loaded = sys.modules.get(name)
    if loaded is not None:
        _require(getattr(loaded, "__file__", None) is None
                 and list(getattr(loaded, "__path__", [])) == [str(path)],
                 f"An incompatible namespace package is already loaded: {name}")
        return
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)


def _load_source(name, path, hashes):
    """Execute the verified source bytes, never an unverified bytecode cache."""
    path = Path(path)
    data, digest = _bytes(path, hashes[str(path)], 2 * 1024 * 1024)
    loaded = sys.modules.get(name)
    if loaded is not None:
        _require(getattr(loaded, "__file__", None) == str(path)
                 and getattr(loaded, "_replay_verified_source_sha256", None) == digest,
                 f"Historical module was already imported outside this verified loader: {name}")
        return loaded
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, origin=str(path))
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
        module._replay_verified_source_sha256 = digest
    except BaseException:
        sys.modules.pop(name, None)
        raise
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


class HistoricalGPUBackend:
    """One verified frozen B; all generation delegates to historical code."""

    def __init__(self, ctx, acknowledge_trusted_checkpoint: bool, *, allow_oracle=False):
        _require(acknowledge_trusted_checkpoint is True,
                 "The verified historical checkpoint requires an explicit trusted-file acknowledgement")
        _require(type(allow_oracle) is bool, "Oracle authorization must be an explicit boolean")
        fresh = verify_context(*(ctx[k] for k in ("project_root", "run_dir", "data_npz", "schema")))
        _require(fresh == ctx, "Verified context changed before backend initialization")
        self.ctx, self.allow_oracle = fresh, bool(allow_oracle)
        self.project, self.run = Path(ctx["project_root"]), Path(ctx["run_dir"])
        hashes = ctx["source_file_sha256"]
        sys.dont_write_bytecode = True
        pilot_root = self.project / "gpu-pilot-v1"
        _load_source("prototype2_gpu_migration_v1", pilot_root / "prototype2_gpu_migration_v1.py", hashes)
        self.pilot = _load_source("prototype2_gpu_pilot_v1", pilot_root / "prototype2_gpu_pilot_v1.py", hashes)
        job_id = self.pilot.require_allocation()
        self.device, versions = self.pilot.configure_runtime()
        import numpy as np
        import torch
        from opacus.accountants import RDPAccountant
        self.np, self.torch = np, torch
        for name, directory in (("tabdiff", self.project / "tabdiff"),
                                ("tabdiff.models", self.project / "tabdiff/models"),
                                ("tabdiff.modules", self.project / "tabdiff/modules")):
            _namespace(name, directory)
        for name, relative in (
            ("tabdiff.models.noise_schedule", "tabdiff/models/noise_schedule.py"),
            ("tabdiff.modules.transformer", "tabdiff/modules/transformer.py"),
            ("tabdiff.modules.main_modules", "tabdiff/modules/main_modules.py"),
            ("tabdiff.models.unified_ctime_diffusion", "tabdiff/models/unified_ctime_diffusion.py"),
            ("local_tabdiff_smoke", "scripts/local_tabdiff_smoke.py"),
            ("run_ipums_ab_pilot", "scripts/run_ipums_ab_pilot.py"),
            ("prototype2_guidance_core", "scripts/prototype2_guidance_core.py"),
            ("run_ipums_cd_pilot", "scripts/run_ipums_cd_pilot.py"),
            ("prototype2_gpu_preflight_v2", "scripts/prototype2_gpu_preflight_v2.py"),
            ("prototype2_gpu_kernel_v1", "gpu-pilot-v1/prototype2_gpu_kernel_v1.py"),
            ("prototype2_gpu_sampling_v1", "gpu-pilot-v1/prototype2_gpu_sampling_v1.py"),
            ("prototype2_gpu_long_kernel_v1", "gpu-longrun-v1p1/prototype2_gpu_long_kernel_v1.py"),
        ):
            _load_source(name, self.project / relative, hashes)
        self.ab = sys.modules["run_ipums_ab_pilot"]
        self.cd = sys.modules["run_ipums_cd_pilot"]
        self.guidance = sys.modules["prototype2_guidance_core"]
        self.kernel = sys.modules["prototype2_gpu_long_kernel_v1"]
        self.categories = np.asarray(CATEGORIES, dtype=np.int64)
        self.schema = _json(Path(ctx["schema"]), SCHEMA_SHA256)[0]
        self.metrics = _json(self.run / "metrics.json", ctx["artifact_file_sha256"][str(self.run / "metrics.json")])[0]
        metadata = _json(self.run / "training_metadata.json", ctx["artifact_file_sha256"][str(self.run / "training_metadata.json")])[0]
        data, _ = _bytes(self.run / "B_private_final.pt", CHECKPOINT_SHA256)
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        _require(isinstance(checkpoint, dict) and set(checkpoint) == {
            "model", "config", "privacy_status", "optimizer_state_saved", "private_generators_saved", "accountant"},
            "Unexpected GPU checkpoint structure (the CPU checkpoint format is different)")
        expected_config = dict(ctx["plan"]["config"])
        expected_config.update(arm="B_private", numeric_loss_weight=1.0,
                               execution_device="cuda:0", training_kernel_version=1)
        _require(checkpoint["config"] == expected_config == metadata["arm_configs"]["B_private"]
                 and checkpoint["optimizer_state_saved"] is False
                 and checkpoint["private_generators_saved"] is False
                 and checkpoint["privacy_status"] ==
                 "Individual training mechanism accounting only; coupled experiment is NONPRIVATE",
                 "GPU B checkpoint configuration or saved-state flags differ")
        self.model = self.kernel.build_model(self.categories, 50, self.device, 11026)
        expected_state, state = self.model.state_dict(), checkpoint["model"]
        _require(isinstance(state, Mapping) and set(state) == set(expected_state),
                 "GPU B checkpoint model keys differ")
        for name, value in state.items():
            expected = expected_state[name]
            _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
                     and value.shape == expected.shape and value.dtype == expected.dtype
                     and bool(torch.isfinite(value).all()), f"Invalid GPU B model tensor: {name}")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.base_hash = self.cd.model_hash(self.model)
        _require(self.base_hash == metadata["final_state_sha256"]["B_private"]
                 == self.metrics["final_model_state_sha256"]["B_private"],
                 "Loaded B state differs from the recorded final model")
        saved = checkpoint["accountant"]
        _require(isinstance(saved, Mapping) and set(saved) == {"history", "mechanism"},
                 "Unexpected B accountant structure")
        history = saved["history"]
        _require(isinstance(history, (list, tuple)) and 1 <= len(history) <= 2000,
                 "Invalid B accountant history")
        steps = 0
        for entry in history:
            _require(isinstance(entry, (list, tuple)) and len(entry) == 3, "Invalid accountant entry")
            sigma, q, count = entry
            _require(type(count) is int and count > 0
                     and math.isclose(float(sigma), 1.2, rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(q), 16 / 2048, rel_tol=0, abs_tol=1e-15),
                     "B accountant noise/sampling schedule differs")
            steps += count
        _require(steps == 2000, "B accountant does not contain exactly 2000 updates")
        accountant = RDPAccountant()
        accountant.load_state_dict(saved)
        epsilon = float(accountant.get_epsilon(delta=1e-5))
        planned = ctx["plan"]["privacy_accounting"]["B_training_epsilon"]
        _require(math.isfinite(epsilon) and math.isclose(epsilon, planned, rel_tol=1e-10, abs_tol=1e-10)
                 and math.isclose(epsilon, metadata["training_epsilons"]["B_private"], rel_tol=1e-10, abs_tol=1e-10),
                 "Loaded B accounting differs from the original plan")
        payload = _json(self.run / "dp_targets_01.json", TARGET_SHA256)[0]
        self.cached_targets = self.cd.validate_dp_targets(payload, self.schema,
            ctx["plan"]["privacy_accounting"]["epsilon_marginals_one_cached_release"])
        self._cached_targets_before = [value.copy() for value in self.cached_targets]
        self._plain = {name: getattr(self.model, name).detach().clone()
                       for name in self.kernel.PLAIN_DEVICE_TENSORS}
        self._flags = [(name, p.requires_grad) for name, p in self.model.named_parameters()]
        self._backend_path = Path(__file__).resolve(strict=True)
        self._backend_hash = _bytes(self._backend_path)[1]
        self.provenance = {"historical_job_id": JOB_ID, "current_slurm_job_id": job_id,
            "runtime_versions": versions, "device": str(self.device),
            "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(self.device),
            "backend_source": {"path": str(self._backend_path), "sha256": self._backend_hash},
            "checkpoint_sha256": CHECKPOINT_SHA256, "model_state_sha256": self.base_hash,
            "cached_target_sha256": TARGET_SHA256, "training_epsilon": epsilon,
            "training_delta": 1e-5, "accountant_updates": steps,
            "training_performed": False, "new_marginal_releases": 0,
            "oracle_authorized": self.allow_oracle, "verified_file_sha256": dict(ctx["verified_file_sha256"])}
        del checkpoint, data, state, expected_state, accountant
        self.verify_unchanged()

    def _validate_array(self, array, count):
        np = self.np
        _require(isinstance(array, np.ndarray) and array.shape == (count, 7)
                 and array.dtype == np.dtype("float32") and np.isfinite(array).all(),
                 "Expected finite historical-format float32 encoded rows")
        self.ab.evaluation_codes(array, self.categories)
        return array

    def load_historical_samples(self):
        result = {}
        for arm in ("B_private", "C"):
            for draw in (1, 2, 3):
                path = self.run / f"{arm}_generation_{draw:02d}.npz"
                data, _ = _bytes(path, self.ctx["artifact_file_sha256"][str(path)], 4 * 1024 * 1024)
                with self.np.load(io.BytesIO(data), allow_pickle=False) as archive:
                    _require(set(archive.files) == {"X_synthetic", "categories", "features"}
                             and self.np.array_equal(archive["categories"], self.categories)
                             and archive["features"].tolist() == FEATURES,
                             f"Historical sample schema differs: {path.name}")
                    result[arm, draw] = self._validate_array(archive["X_synthetic"].copy(), 512)
        return result

    def sample(self, seed, count, offsets=None):
        _require(type(seed) is int and 0 <= seed < 2**63 - 1, "Invalid public generation seed")
        _require(type(count) is int and count in (512, 4096), "Replay sampling supports 512 or 4096 rows")
        array, hooks, rng = self.pilot.generate_checked(self.model, self.categories,
                                                       self.device, seed, count, offsets)
        return self._validate_array(array, count), hooks, rng

    def normalize_reference(self, array):
        self._validate_array(array, 4096)
        return [self.cd.normalize_counts(v) for v in self.cd.histograms(array, self.categories)]

    def same_rng(self, left, right):
        return self.kernel.same_rng(left, right)

    def fingerprint(self):
        return self.cd.model_hash(self.model)

    def save_sample(self, path, array):
        self._validate_array(array, 512)
        path = Path(path)
        destination = path.parent.resolve(strict=True) / path.name
        _require(not destination.is_relative_to(self.project) and not destination.is_relative_to(self.run),
                 "New samples must remain outside immutable historical trees")
        self.pilot.save_sample(destination, array, self.categories, FEATURES)

    def make_offsets(self, targets, reference, strength, max_log_ratio, prob_floor):
        return self.guidance.make_offsets(targets, reference, strength=strength,
                                         max_log_ratio=max_log_ratio, prob_floor=prob_floor)

    def oracle_targets(self):
        _require(self.allow_oracle, "Exact TRAIN oracle access requires explicit nonprivate authorization")
        data, _ = _bytes(Path(self.ctx["data_npz"]), DATA_SHA256, 4 * 1024 * 1024)
        with self.np.load(io.BytesIO(data), allow_pickle=False) as archive:
            _require(self.np.array_equal(archive["categories"], self.categories), "TRAIN domains differ")
            train = self._validate_array(archive["X_train"].copy(), 2048)
        return [self.cd.normalize_counts(v) for v in self.cd.histograms(train, self.categories)]

    def verify_unchanged(self):
        for path, digest in self.ctx["verified_file_sha256"].items():
            _bytes(Path(path), digest)
        _bytes(self._backend_path, self._backend_hash)
        self.kernel.assert_model_device(self.model, self.device)
        _require(self.fingerprint() == self.base_hash, "Frozen B model state changed")
        _require(self._flags == [(name, p.requires_grad) for name, p in self.model.named_parameters()]
                 and all(self.torch.equal(getattr(self.model, name), before)
                         for name, before in self._plain.items()),
                 "Frozen B gradient flags or plain device tensors changed")
        _require(self.model.num_timesteps == 50
                 and self.np.array_equal(self.model.num_classes, self.np.asarray(CATEGORIES))
                 and self.categories.tolist() == CATEGORIES
                 and self.model.sampler_params == {"stochastic_sampler": True, "second_order_correction": True},
                 "Historical sampler settings changed")
        _require(len(self.cached_targets) == 6
                 and all(self.np.array_equal(value, before)
                         for value, before in zip(self.cached_targets, self._cached_targets_before)),
                 "Cached target vectors changed in memory")
        return {"verified_files_unchanged": True, "frozen_model_unchanged": True,
                "plain_model_tensors_unchanged": True, "training_performed": False,
                "new_marginal_releases": 0}
