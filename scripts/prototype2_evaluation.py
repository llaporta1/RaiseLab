"""Shared NumPy-only evaluation for the seven-column Prototype-2 pilot.

Inputs use the existing encoded representation: AGE=(years-50)/50 in column
zero, followed by six contiguous categorical indices in FEATURES order. These
are internal NONPRIVATE research metrics. This module does not train, measure
a private target, choose a workload, filter records, or claim a DP release.

The historical AGE bins, real-record-only rounding, complete denominators,
and fixed categorical relationship queries remain unchanged. Extra three-way
queries are reported separately from the historical one/two-way averages.
"""

from __future__ import annotations

import itertools
import json
from numbers import Integral
from pathlib import Path

import numpy as np


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
AGE_EDGES = np.array([0, 18, 25, 35, 45, 55, 65, 75, 85, 101], dtype=float)
FIXED_THREE_WAY_QUERIES = (
    ("AGE", "SEX", "EDUCD"),
    ("AGE", "EDUCD", "OCC"),
    ("AGE", "OCC", "IND"),
)


def _cardinalities(values, expected_length=None):
    array = np.asarray(values)
    if (array.ndim != 1 or not len(array)
            or (expected_length is not None and len(array) != expected_length)
            or array.dtype.kind not in "iuf"
            or not np.isfinite(array).all()
            or np.any(array < 1) or np.any(array != np.floor(array))):
        raise ValueError("Cardinalities must be finite positive integers of the expected length")
    # Check Python integers before casting, including the uint64/int64 boundary.
    if any(int(value) > np.iinfo(np.int64).max for value in array):
        raise ValueError("Cardinality exceeds the supported int64 code range")
    return array.astype(np.int64)


def _encoded_table(table, categories, label="Evaluation"):
    values = np.asarray(table)
    if (values.ndim != 2 or values.shape[1] != len(FEATURES)
            or len(values) == 0 or values.dtype.kind not in "iuf"):
        raise ValueError(f"{label} table must be a nonempty numeric array with seven columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains nonfinite values; no rows will be silently dropped")
    cats = values[:, 1:]
    if (np.any(cats != np.floor(cats)) or np.any(cats < 0)
            or np.any(cats >= categories[None, :])):
        raise ValueError(f"{label} contains an invalid categorical index for the public domain")
    return values


def load_pilot(data_npz, schema_path):
    """Load and validate the complete legacy pilot schema without PyTorch.

    Returns (train, holdout, categories, schema), with NumPy arrays in their
    stored encoded representation. Real AGE must satisfy the declared [0,100]
    input domain. No implicit 18-65 cohort filter or row dropping occurs.
    Optional row counts in the schema must agree with the arrays.
    """
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    if not isinstance(schema, dict) or schema.get("features") != FEATURES:
        raise ValueError(f"Schema features must be exactly {FEATURES}, in that order")
    transform = schema.get("age_transform")
    if (not isinstance(transform, str)
            or "(AGE-50)/50" not in "".join(transform.split())):
        raise ValueError("Schema must declare the fixed AGE encoding (AGE-50)/50")
    with np.load(data_npz, allow_pickle=False) as archive:
        missing = {"X_train", "X_test", "categories"} - set(archive.files)
        if missing:
            raise ValueError(f"Pilot archive is missing required arrays: {sorted(missing)}")
        categories = _cardinalities(archive["categories"], 6)
        train = np.array(archive["X_train"], copy=True)
        holdout = np.array(archive["X_test"], copy=True)

    domains = schema.get("category_domains")
    if not isinstance(domains, dict) or set(domains) != set(FEATURES[1:]):
        raise ValueError("Schema must contain exactly the six modeled categorical domains")
    for name, size in zip(FEATURES[1:], categories):
        codes = domains[name]
        if not isinstance(codes, list) or len(codes) != int(size):
            raise ValueError(f"Public category domain/cardinality mismatch for {name}")
        if (any(isinstance(code, bool) or not isinstance(code, int) or code < 0
                for code in codes) or len(set(codes)) != len(codes)):
            raise ValueError(f"Public domain {name} requires distinct nonnegative integer source codes")

    for label, table in (("Training", train), ("Held-out", holdout)):
        _encoded_table(table, categories, label)
        if np.any((table[:, 0] < -1) | (table[:, 0] > 1)):
            raise ValueError(f"{label} AGE is outside the declared clipped input range")
    for key, count in (("train_rows", len(train)), ("test_rows", len(holdout)),
                       ("holdout_rows", len(holdout))):
        if key in schema and (isinstance(schema[key], bool)
                              or not isinstance(schema[key], int)
                              or schema[key] != count):
            raise ValueError(f"Schema {key} does not match the stored row count")
    return train, holdout, categories, schema


def evaluation_codes(table, categories, real_records=False):
    """Return historical evaluation codes and AGE metadata, without filtering.

    Real AGE alone is rounded after decoding to repair float32 integer-boundary
    error. Generated values are not rounded or clipped. AGE in (100,101) is in
    the last regular histogram bin, but the separate support fraction flags it.
    The remembered study cohort [18,65] is reported separately: this does not
    filter rows or change the public [0,100] input domain or legacy bins.
    """
    cardinalities = _cardinalities(categories, 6)
    values = _encoded_table(table, cardinalities)
    ages = 50.0 + 50.0 * values[:, 0].astype(np.float64)
    if not np.isfinite(ages).all():
        raise ValueError("Decoded AGE is nonfinite; no rows will be silently dropped")
    if real_records:
        ages = np.rint(ages)
    codes = np.empty(values.shape, dtype=np.int64)
    codes[:, 0] = np.searchsorted(AGE_EDGES, ages, side="right")
    codes[:, 1:] = values[:, 1:].astype(np.int64)
    age_info = {
        "decoded_outside_0_100_fraction": float(np.mean((ages < 0) | (ages > 100))),
        "decoded_outside_18_65_fraction": float(np.mean((ages < 18) | (ages > 65))),
        "histogram_counts": np.bincount(codes[:, 0], minlength=len(AGE_EDGES) + 1).tolist(),
    }
    return codes, age_info


def _code_tables(left, right, domain_sizes):
    domains = _cardinalities(domain_sizes)
    checked = []
    for label, table in (("Left", left), ("Right", right)):
        values = np.asarray(table)
        if (values.ndim != 2 or values.shape[1] != len(domains)
                or not len(values) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError(f"{label} codes must be a nonempty finite table matching the public domains")
        if (np.any(values < 0) or np.any(values != np.floor(values))
                or np.any(values >= domains[None, :])):
            raise ValueError(f"{label} has an evaluation code outside its public domain")
        checked.append(values.astype(np.int64, copy=False))
    return checked[0], checked[1], domains


def _columns(columns, count):
    try:
        columns = tuple(columns)
    except TypeError as exc:
        raise ValueError("Supply one to three distinct column indices") from exc
    if (len(columns) not in (1, 2, 3)
            or any(isinstance(column, bool) or not isinstance(column, Integral)
                   or not 0 <= column < count for column in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("Supply one to three distinct valid column indices")
    return columns


def _sparse_l1(left, right, columns, domain_sizes):
    public_cells = 1
    for column in columns:
        public_cells *= int(domain_sizes[column])
    if public_cells <= np.iinfo(np.int64).max:
        # Mixed-radix keys allocate O(rows), never the Cartesian domain.
        def keys(table):
            result = table[:, columns[0]].copy()
            for column in columns[1:]:
                result = result * int(domain_sizes[column]) + table[:, column]
            return result
        both = np.concatenate((keys(left), keys(right)))
        _, inverse = np.unique(both, return_inverse=True)
    else:
        # Exact tuple counting avoids key overflow for unusually large domains.
        both = np.concatenate((left[:, columns], right[:, columns]), axis=0)
        _, inverse = np.unique(both, axis=0, return_inverse=True)
    cells = int(inverse.max()) + 1
    left_counts = np.bincount(inverse[:len(left)], minlength=cells)
    right_counts = np.bincount(inverse[len(left):], minlength=cells)
    result = float(np.abs(left_counts / len(left) - right_counts / len(right)).sum())
    if not np.isfinite(result) or not -1e-12 <= result <= 2 + 1e-12:
        raise RuntimeError("L1 calculation outside its expected [0,2] range")
    return min(2.0, max(0.0, result))


def marginal_l1(left, right, columns, domain_sizes):
    """Exact sparse normalized histogram L1 for one to three columns.

    All rows contribute to the denominators, including AGE under/overflow
    codes. Cells absent from both arrays have zero contribution. Unequal row
    counts are supported and no dense Cartesian table is allocated.
    """
    left, right, domains = _code_tables(left, right, domain_sizes)
    columns = _columns(columns, len(domains))
    return _sparse_l1(left, right, columns, domains)


def compare_marginals(left, right, domain_sizes):
    """Legacy seven-feature metrics plus separate AGE/category/three-way rows."""
    left, right, domains = _code_tables(left, right, domain_sizes)
    if len(domains) != len(FEATURES) or domains[0] != len(AGE_EDGES) + 1:
        raise ValueError("Expected seven public domains with the historical 11-bin AGE domain")
    one_way = {name: _sparse_l1(left, right, (i,), domains)
               for i, name in enumerate(FEATURES)}
    pair_indices = list(itertools.combinations(range(len(FEATURES)), 2))
    two_way = {f"{FEATURES[i]} x {FEATURES[j]}":
               _sparse_l1(left, right, (i, j), domains) for i, j in pair_indices}
    three_way = {" x ".join(query): _sparse_l1(
        left, right, tuple(FEATURES.index(name) for name in query), domains)
        for query in FIXED_THREE_WAY_QUERIES}
    return {
        "one_way": one_way,
        "two_way": two_way,
        "mean_one_way_l1": float(np.mean(list(one_way.values()))),
        "mean_two_way_l1": float(np.mean(list(two_way.values()))),
        "categorical_one_way": float(np.mean([one_way[name] for name in FEATURES[1:]])),
        "age_one_way": one_way["AGE"],
        "categorical_two_way": float(np.mean([
            two_way[f"{FEATURES[i]} x {FEATURES[j]}"] for i, j in pair_indices if i != 0])),
        "age_two_way": float(np.mean([two_way[f"AGE x {name}"] for name in FEATURES[1:]])),
        "three_way": three_way,
        "mean_three_way_l1": float(np.mean(list(three_way.values()))),
    }


def flatten_metrics(evaluation):
    """Return named scalar errors from evaluate_sample for tables/comparisons.

    The result excludes public-cell/row counts and per-context metadata. It
    never treats target, generation, or shuffle repeats as independent fits.
    """
    marginals = evaluation["marginals"]
    relationships = evaluation["relationships"]
    names = {
        "all_one_way": "mean_one_way_l1", "all_two_way": "mean_two_way_l1",
        "categorical_one_way": "categorical_one_way", "age_one_way": "age_one_way",
        "categorical_two_way": "categorical_two_way", "age_two_way": "age_two_way",
        "fixed_three_way": "mean_three_way_l1",
    }
    flat = {f"marginal/{name}": float(marginals[key]) for name, key in names.items()}
    flat["marginal/age_outside_fraction"] = float(
        evaluation["age"]["decoded_outside_0_100_fraction"])
    flat["marginal/age_outside_cohort_fraction"] = float(
        evaluation["age"]["decoded_outside_18_65_fraction"])
    for order in ("one_way", "two_way", "three_way"):
        flat.update({f"{order}/{query}": float(value)
                     for query, value in marginals[order].items()})
    flat.update({f"relationship/{key}": float(value)
                 for key, value in relationships["aggregate"].items()})
    for query, metrics in relationships["interaction"].items():
        for name in ("residual_l1", "reference_residual_l1", "synthetic_residual_l1",
                     "independence_baseline_residual_l1", "joint_l1"):
            flat[f"interaction/{query}/{name}"] = float(metrics[name])
    for query, metrics in relationships["conditional"].items():
        for name in ("weighted_tv", "reference_weighted_context_coverage",
                     "missing_reference_context_mass", "context_marginal_tv",
                     "synthetic_mass_in_reference_absent_contexts"):
            flat[f"conditional/{query}/{name}"] = float(metrics[name])
    return flat


def evaluate_sample(sample, reference, categories, shuffle_repeats=10, seed=0,
                    *, sample_real_records=False):
    """Evaluate encoded rows against real reference rows, with nested shuffles.

    ``sample_real_records=True`` is for real-versus-real controls: it applies
    the same integer-AGE reconstruction repair to both arrays. Ordinary
    generated numerical outputs must keep the default False.

    Returns marginals, relationships, age, and shuffle. Shuffle ``mean``,
    ``shuffle_minus_original``, and each ``per_shuffle`` row use the exact flat
    key schema of flatten_metrics(). For zero repeats these are None/None/[].
    The shuffle observations are diagnostics within this single sample.
    """
    if (isinstance(shuffle_repeats, bool) or not isinstance(shuffle_repeats, Integral)
            or shuffle_repeats < 0):
        raise ValueError("shuffle_repeats must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(sample_real_records, (bool, np.bool_)):
        raise ValueError("sample_real_records must be a boolean")
    # Lazy import breaks the evaluator/relationship dependency cycle. The
    # helper's fixed relationship queries involve categorical columns only.
    import prototype2_relationship_checks as relationships

    cardinalities = _cardinalities(categories, 6)
    sample_codes, age = evaluation_codes(sample, cardinalities, sample_real_records)
    reference_codes, _ = evaluation_codes(reference, cardinalities, real_records=True)
    domain_sizes = np.r_[len(AGE_EDGES) + 1, cardinalities]
    original = {
        "marginals": compare_marginals(sample_codes, reference_codes, domain_sizes),
        "relationships": relationships.evaluate_relationships(sample, reference, cardinalities),
        "age": age,
    }
    original_flat = flatten_metrics(original)
    rng = np.random.default_rng(int(seed))
    shuffled_rows = []
    for _ in range(int(shuffle_repeats)):
        shuffled = relationships.shuffle_categorical_columns(sample, rng)
        shuffled_codes, shuffled_age = evaluation_codes(
            shuffled, cardinalities, real_records=sample_real_records)
        current = {
            "marginals": compare_marginals(shuffled_codes, reference_codes, domain_sizes),
            "relationships": relationships.evaluate_relationships(shuffled, reference, cardinalities),
            "age": shuffled_age,
        }
        current_flat = flatten_metrics(current)
        for key in original_flat:
            if key.startswith("one_way/") or key in {
                    "marginal/age_outside_fraction", "marginal/age_outside_cohort_fraction"}:
                if current_flat[key] != original_flat[key]:
                    raise RuntimeError("Categorical shuffle changed a one-way or AGE-support metric")
        shuffled_rows.append(current_flat)
    mean = ({key: float(np.mean([row[key] for row in shuffled_rows]))
             for key in original_flat} if shuffled_rows else None)
    original["shuffle"] = {
        "repeats": int(shuffle_repeats),
        "seed": int(seed),
        "mean": mean,
        "shuffle_minus_original": ({key: mean[key] - original_flat[key]
                                      for key in original_flat} if mean is not None else None),
        "per_shuffle": shuffled_rows,
        "scope": "Nested within this sample; positive shuffled-minus-original favors original associations",
    }
    return original
