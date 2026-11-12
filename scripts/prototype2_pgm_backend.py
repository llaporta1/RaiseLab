"""Adapters for official AIM and MST using modern Private-PGM/MBI.

The official source checkout stays unchanged. The MST adapter follows its
measure/compress/select/measure/estimate sequence, retaining the fitted model
so repeated generation does not repeat a private mechanism. The tree-selection
and domain-compression routines below are adapted from the Apache-2.0 licensed
official mst.py. They expose inference iterations for labeled smoke tests and
replace removed Domain.attrs accesses with the modern domain iteration API.
The adapted routines are distributed under Apache-2.0; see
licenses/MBI-Apache-2.0.txt and licenses/MBI-NOTICE.md.

Official source: https://github.com/ryan112358/mbi
Tested source commit: 5f97065a17b3e53cfefc7a8ce3e293fd153dd064
These research implementations are not a release-grade DP certification.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import itertools
import math
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tomllib

import numpy as np


PINNED_MBI_COMMIT = "5f97065a17b3e53cfefc7a8ce3e293fd153dd064"
FEATURES = ("AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND")
AGE_EDGES = np.array([0, 18, 25, 35, 45, 55, 65, 75, 85, 101], dtype=np.float64)
ARM_LABELS = {"aim": "AIM + Private-PGM", "mst": "MST + Private-PGM"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_seed(base, *parts):
    """Derive distinct, reproducible generation/fixture seeds, never DP noise seeds."""
    value = "\0".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


@contextmanager
def numpy_random_scope(public_fixture_seed=None):
    """Isolate upstream NumPy RNG use and forget real-data mechanism seed material.

    A real-data call uses 256 bits from secrets. A public software fixture may
    supply an explicit deterministic seed. Neither this context nor the real
    fit metadata returns seed material or the private NumPy state.
    """
    previous = np.random.get_state()
    if public_fixture_seed is None:
        seed_material = np.frombuffer(secrets.token_bytes(32), dtype=np.uint32).copy()
        np.random.seed(seed_material)
        seed_material.fill(0)
        del seed_material
    else:
        if not isinstance(public_fixture_seed, (int, np.integer)) or not 0 <= public_fixture_seed < 2**32:
            raise ValueError("Public fixture/generation seed must be a uint32 integer")
        np.random.seed(int(public_fixture_seed))
    try:
        yield
    finally:
        np.random.set_state(previous)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def verify_official_checkout(repo, expected_commit):
    """Fail before importing code if provenance or the pinned checkout differs."""
    repo = Path(repo).expanduser().resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("--expected-mbi-commit must be a full lowercase 40-character SHA")
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--mbi-repo must name the root of the official checkout")
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise ValueError(f"Official MBI commit mismatch: expected {expected_commit}, found {actual}")
    origin = _git(repo, "remote", "get-url", "origin")
    if origin not in {
        "https://github.com/ryan112358/mbi", "https://github.com/ryan112358/mbi.git",
        "git@github.com:ryan112358/mbi.git", "ssh://git@github.com/ryan112358/mbi.git",
    }:
        raise ValueError("MBI origin is not the official ryan112358/mbi repository")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Official MBI checkout has tracked modifications; use a clean pinned checkout")
    required = ["src/mbi/__init__.py", "mechanisms/aim.py", "mechanisms/mst.py",
                "mechanisms/mechanism.py", "mechanisms/cdp2adp.py", "pyproject.toml"]
    for relative in required:
        if not (repo / relative).is_file():
            raise ValueError(f"Official checkout is missing {relative}")
    project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
    if project.get("name") != "mbi":
        raise ValueError("Expected official modern mbi package metadata")
    return repo, {
        "repository": str(repo), "origin": origin, "commit": actual,
        "source_version": project["version"],
        "source_sha256": {name: sha256_file(repo / name) for name in required},
    }


def _within(path, parent):
    try:
        Path(path).resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_file_module(name, file):
    if name in sys.modules:
        previous = getattr(sys.modules[name], "__file__", None)
        if previous is None or Path(previous).resolve() != file.resolve():
            raise RuntimeError(f"Refusing conflicting previously imported module {name}: {previous}")
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load verified official module {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def encode_training_data(train, categories, age_representation="integer"):
    """Encode training data without consulting holdout data or observed domains."""
    values = np.asarray(train, dtype=np.float64)
    categories = np.asarray(categories)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or len(values) == 0:
        raise ValueError("PGM training requires a nonempty seven-column array")
    if (categories.shape != (6,) or categories.dtype.kind not in "iuf"
            or not np.isfinite(categories).all() or np.any(categories < 1)
            or np.any(categories != np.floor(categories))):
        raise ValueError("Six positive public categorical cardinalities are required")
    categories = categories.astype(np.int64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite training values are not accepted")
    categorical = values[:, 1:]
    if np.any(categorical != np.floor(categorical)) or np.any(categorical < 0) or np.any(categorical >= categories):
        raise ValueError("Categorical training codes violate the fixed public domain")
    ages = 50.0 + 50.0 * values[:, 0]
    integer_ages = np.rint(ages)
    if np.any(np.abs(ages - integer_ages) > 1e-4):
        raise ValueError("Real AGE must be integer-valued apart from float encoding error")
    if np.any(integer_ages < 0) or np.any(integer_ages > 100):
        raise ValueError("Real AGE is outside the declared public [0,100] support")
    codes = np.empty(values.shape, dtype=np.int64)
    codes[:, 1:] = categorical.astype(np.int64)
    if age_representation == "integer":
        codes[:, 0] = integer_ages.astype(np.int64)
        age_size = 101
    elif age_representation == "legacy-bins":
        codes[:, 0] = np.searchsorted(AGE_EDGES, integer_ages, side="right") - 1
        age_size = len(AGE_EDGES) - 1
    else:
        raise ValueError("AGE representation must be integer or legacy-bins")
    return codes, dict(zip(FEATURES, [age_size, *(int(k) for k in categories)]))


def decode_synthetic_data(codes, domain_sizes, age_representation="integer"):
    """Convert a valid discrete draw to the common encoded array without filtering."""
    values = np.asarray(codes)
    if values.ndim != 2 or values.shape[1] != len(FEATURES) or not len(values):
        raise ValueError("PGM sample must have seven columns and at least one row")
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ValueError("PGM returned noninteger/nonfinite domain codes")
    sizes = np.asarray([domain_sizes[name] for name in FEATURES])
    if np.any(values < 0) or np.any(values >= sizes):
        raise ValueError("PGM returned an out-of-domain code; no rows will be dropped")
    result = values.astype(np.float64)
    if age_representation == "integer":
        ages = result[:, 0].copy()
    elif age_representation == "legacy-bins":
        # Public bin midpoints. No claim to have learned within-bin numerical AGE.
        midpoints = (AGE_EDGES[:-1] + AGE_EDGES[1:] - 1) / 2
        ages = midpoints[values[:, 0].astype(np.int64)]
    else:
        raise ValueError("Unknown AGE representation")
    result[:, 0] = (ages - 50.0) / 50.0
    return result


def aim_workload(domain_sizes, max_cells=None):
    """All 21 pairs by default, with explicit and fully logged optional exclusions."""
    if tuple(domain_sizes) != FEATURES:
        raise ValueError("Public domain order must match the seven modeled features")
    if max_cells is not None and max_cells < 1:
        raise ValueError("Explicit maximum cell count must be positive")
    workload, excluded = [], []
    for pair in itertools.combinations(FEATURES, 2):
        cells = math.prod(domain_sizes[name] for name in pair)
        if max_cells is not None and cells > max_cells:
            excluded.append({"query": list(pair), "cells": cells,
                             "reason": "explicit --aim-max-cells resource exclusion"})
        else:
            workload.append((pair, 1.0))
    covered = {name for pair, _ in workload for name in pair}
    if covered != set(FEATURES):
        raise ValueError("Explicit AIM cell cap excludes every pair for an attribute; raise the cap")
    return workload, excluded


@dataclass
class FittedPGM:
    model: object
    undo_compression: object
    domain_sizes: dict
    metadata: dict

    def sample_codes(self, rows, generation_seed):
        if rows < 1:
            raise ValueError("Sample row count must be positive")
        with numpy_random_scope(generation_seed):
            draw = self.model.synthetic_data(rows=int(rows), method="sample")
            if self.undo_compression is not None:
                draw = self.undo_compression(draw)
        values = draw.to_dict()
        result = np.column_stack([values[name] for name in FEATURES])
        if result.shape != (rows, len(FEATURES)):
            raise RuntimeError("Official sampler returned the wrong sample shape")
        sizes = np.array([self.domain_sizes[name] for name in FEATURES])
        if not np.isfinite(result).all() or np.any(result != np.floor(result)) or np.any(result < 0) or np.any(result >= sizes):
            raise RuntimeError("Official sampler violated the declared domain")
        return result.astype(np.int64)


class OfficialPGMBackend:
    """Checked source-bound backend. Its fit API accepts training data only."""

    def __init__(self, repo, expected_commit):
        self.repo, self.provenance = verify_official_checkout(repo, expected_commit)
        source = self.repo / "src" / "mbi"
        for name, module in list(sys.modules.items()):
            if name == "mbi" or name.startswith("mbi."):
                path = getattr(module, "__file__", None)
                if path is None or not _within(path, source):
                    raise RuntimeError(f"Refusing imported mbi outside the pinned source: {path}")
        # Require the environment's editable install to resolve the source. Do not
        # hide an accidental stub by silently inserting another mbi on sys.path.
        found = importlib.util.find_spec("mbi")
        if found is None or found.origin is None or not _within(found.origin, source):
            location = None if found is None else found.origin
            raise RuntimeError(f"mbi resolves to {location}; install the pinned checkout editable")
        import jax
        self.jax = jax
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        try:
            self.mbi = importlib.import_module("mbi")
        except (AttributeError, ImportError) as error:
            raise RuntimeError(f"Official MBI dependency/API mismatch: {error}. Use the tested isolated dependency pins.") from error
        for attr in ("Dataset", "Domain", "LinearMeasurement", "MarkovRandomField", "estimation"):
            if not hasattr(self.mbi, attr):
                raise RuntimeError(f"Modern official MBI API missing {attr}; legacy/stub adapter rejected")
        if not hasattr(self.mbi.estimation, "MirrorDescent"):
            raise RuntimeError("Modern estimator API absent; use the pinned checkout")
        mechanisms = self.repo / "mechanisms"
        _load_file_module("cdp2adp", mechanisms / "cdp2adp.py")
        _load_file_module("mechanism", mechanisms / "mechanism.py")
        self.aim = _load_file_module("_raise_official_aim", mechanisms / "aim.py")
        self.mst = _load_file_module("_raise_official_mst", mechanisms / "mst.py")
        if not {"data", "workload", "num_synth_rows"}.issubset(inspect.signature(self.aim.AIM.run).parameters):
            raise RuntimeError("Official AIM API differs from this verified adapter")
        if not {"rows", "method"}.issubset(inspect.signature(self.mbi.MarkovRandomField.synthetic_data).parameters):
            raise RuntimeError("The backend must expose explicit i.i.d. synthetic sampling")
        versions = {}
        for name in ("mbi", "attrs", "numpy", "scipy", "networkx", "jax", "jaxlib", "pandas", "chex", "optax"):
            versions[name] = importlib.metadata.version(name)
        self.provenance.update({
            "mbi_import_path": str(Path(self.mbi.__file__).resolve()),
            "aim_import_path": str(Path(self.aim.__file__).resolve()),
            "mst_import_path": str(Path(self.mst.__file__).resolve()),
            "api_family": "modern MBI Estimator / LinearMeasurement",
            "versions": versions, "jax_x64": bool(jax.config.jax_enable_x64),
            "jax_persistent_compilation_cache": bool(jax.config.jax_enable_compilation_cache),
            "jax_devices": [str(device) for device in jax.devices()],
        })

    def fit(self, arm, train_codes, domain_sizes, *, epsilon, delta,
            public_fixture_seed=None, aim_max_model_size=80.0, aim_max_iters=1000,
            aim_max_cells=None, mst_selection_iters=2500, mst_final_iters=10000):
        """Run one private fit. No holdout, evaluation score, or true target enters."""
        if arm not in ARM_LABELS:
            raise ValueError("Arm must be aim or mst")
        if not math.isfinite(epsilon) or epsilon <= 0 or not 0 < delta < 1:
            raise ValueError("Need epsilon>0 and 0<delta<1")
        if min(aim_max_iters, mst_selection_iters, mst_final_iters) < 1 or aim_max_model_size <= 0:
            raise ValueError("Iteration counts and model size must be positive")
        values = np.asarray(train_codes)
        sizes = np.asarray([domain_sizes[name] for name in FEATURES])
        if values.ndim != 2 or values.shape[1] != 7 or not len(values) or not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values >= sizes):
            raise ValueError("Invalid discrete training table")
        domain = self.mbi.Domain.fromdict(domain_sizes)
        data = self.mbi.Dataset({name: values[:, i].astype(np.int64) for i, name in enumerate(FEATURES)}, domain)
        metadata = {
            "arm": arm, "baseline": ARM_LABELS[arm], "epsilon": float(epsilon),
            "delta": float(delta), "adjacency": "record add/remove (unbounded)",
            "rng": "known public fixture seed; process reproducibility additionally requires fixed PYTHONHASHSEED" if public_fixture_seed is not None else "secret-seeded NumPy research RNG; seed/state not recorded",
            "no_additional_C_target": True, "sampling_method": "iid: method=sample",
            "private_fit_count": 1,
        }
        with numpy_random_scope(public_fixture_seed):
            if arm == "aim":
                workload, excluded = aim_workload(domain_sizes, aim_max_cells)
                mechanism = self.aim.AIM(epsilon, delta, prng=np.random,
                                         max_model_size=aim_max_model_size, max_iters=aim_max_iters)
                # The upstream API always generates a final dataset. Use one
                # discarded row, then make separately seeded evaluation draws.
                model, _ = mechanism.run(data, workload, num_synth_rows=1)
                undo = None
                metadata.update({
                    "rho": float(mechanism.rho),
                    "candidate_pair_workload": [{"query": list(q), "weight": w,
                                                 "cells": math.prod(domain_sizes[x] for x in q)} for q, w in workload],
                    "excluded_pairs": excluded,
                    "candidate_policy": "downward closure of the listed equal-weight pairs; upstream adaptive model-size eligibility remains active",
                    "aim_max_model_size_mb": float(aim_max_model_size), "aim_max_iters": int(aim_max_iters),
                    "upstream_discarded_postprocess_rows": 1,
                })
            else:
                rho = self.mst.cdp_rho(epsilon, delta)
                sigma = np.sqrt(3 / (2 * rho))
                singleton_log = self.mst.measure(data, [(name,) for name in domain], sigma)
                compressed, singleton_log, undo = self._compress_mst_data(data, singleton_log)
                pairs = self._select_mst_pairs(compressed, rho / 3, singleton_log, mst_selection_iters)
                pair_log = self.mst.measure(compressed, pairs, sigma)
                model = self.mbi.estimation.MirrorDescent().estimate(
                    compressed.domain, singleton_log + pair_log, iters=mst_final_iters)
                metadata.update({
                    "rho": float(rho), "selected_pairs": [list(q) for q in pairs],
                    "compression": "official MST noisy-singleton compression/decompression, adapted only to modern Domain/Dataset data access",
                    "compressed_domain": {name: int(compressed.domain[name]) for name in FEATURES},
                    "mst_selection_iters": int(mst_selection_iters), "mst_final_iters": int(mst_final_iters),
                    "candidate_policy": "all pairwise tree-edge candidates on the compressed public attribute schema",
                })
        # Include asynchronous accelerator work in the caller's whole-fit wall time.
        self.jax.block_until_ready((model.potentials, model.marginals, model.total))
        metadata["model_cliques"] = [list(q) for q in model.cliques]
        return FittedPGM(model, undo, domain_sizes, metadata)

    def _compress_mst_data(self, data, measurements):
        """Official noisy compression, with modern Domain/Dataset data access.

        Upstream mst.py still uses the removed Domain.attrs in transform_data
        and reverse_data. This adapter preserves the same threshold, query,
        mappings, and random decompression while leaving that source unchanged.
        """
        supports, new_measurements = {}, []
        for measurement in measurements:
            col = measurement.clique[0]
            y = measurement.noisy_measurement
            support = np.asarray(y >= 3 * measurement.stddev)
            supports[col] = support
            if support.all():
                new_measurements.append(measurement)
            else:
                y2 = np.append(y[support], y[~support].sum())
                scale = np.ones(y2.size)
                scale[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
                y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
                query = (lambda scale: lambda mu: mu.datavector() * scale)(scale)
                new_measurements.append(self.mbi.LinearMeasurement(
                    y2, measurement.clique, measurement.stddev, query=query))
        return (self._transform_mst_data(data, supports), new_measurements,
                lambda draw: self._reverse_mst_data(draw, supports))

    def _transform_mst_data(self, data, supports):
        columns, sizes = {}, {}
        source = data.to_dict()
        for col in data.domain:
            values = source[col]
            support = supports[col]
            size = int(support.sum())
            sizes[col] = size + int(size < support.size)
            mapping = np.full(support.size, size, dtype=np.int64)
            mapping[support] = np.arange(size)
            columns[col] = mapping[np.asarray(values)]
        return self.mbi.Dataset(columns, self.mbi.Domain.fromdict(sizes))

    def _reverse_mst_data(self, data, supports):
        columns, sizes = {}, {}
        source = data.to_dict()
        for col in data.domain:
            support = supports[col]
            size = int(support.sum())
            sizes[col] = int(support.size)
            supported, extra = np.where(support)[0], np.where(~support)[0]
            values = np.asarray(source[col]).copy()
            mask = values == size
            if extra.size:
                values[mask] = np.random.choice(extra, int(mask.sum()))
            values[~mask] = supported[values[~mask]]
            columns[col] = values.astype(np.int64)
        return self.mbi.Dataset(columns, self.mbi.Domain.fromdict(sizes))

    def _select_mst_pairs(self, data, rho, measurements, iters):
        """Official mst.select logic, exposing only its postprocessing iteration cap."""
        est = self.mbi.estimation.MirrorDescent().estimate(data.domain, measurements, iters=iters)
        candidates = list(itertools.combinations(FEATURES, 2))
        weights = {pair: np.linalg.norm(data.project(pair).datavector() - est.project(pair).datavector(), 1)
                   for pair in candidates}
        tree = self.mst.nx.Graph()
        tree.add_nodes_from(FEATURES)
        connected = self.mst.DisjointSet(FEATURES)
        epsilon = np.sqrt(8 * rho / (len(FEATURES) - 1))
        for _ in range(len(FEATURES) - 1):
            eligible = [pair for pair in candidates if not connected.connected(*pair)]
            scores = np.asarray([weights[pair] for pair in eligible])
            chosen = self.mst.exponential_mechanism(scores, epsilon, sensitivity=1.0)
            edge = eligible[chosen]
            tree.add_edge(*edge)
            connected.merge(*edge)
        return list(tree.edges)
