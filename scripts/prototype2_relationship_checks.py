"""Descriptive relationship checks for the existing seven-column IPUMS pilot.

All reference-data metrics in this module are NONPRIVATE. This helper neither
trains a model nor obtains another DP marginal release. It compares fixed,
predeclared queries over the supplied public categorical domains; it never
selects queries or drops cells based on the reference data.

``sample`` and ``reference`` are raw encoded N-by-7 arrays in ``ab.FEATURES``
order, NOT already-binned evaluation-code arrays. ``categories`` is the six
public categorical cardinalities in the existing schema order. Reference
weights are empirical and nonprivate, not public population weights.

Interaction-residual L1 is in [0, 4]. It compares p(x,y)-p(x)p(y) between
samples and the reference. It is not a marginal-invariant association measure
or a certificate of population dependence. Conditional TV is in [0, 1] and
also responds to outcome-marginal changes. Interpret both with the exact
one-way-preserving shuffle control, coverage, and finite-sample limitations.
"""

import json

import numpy as np

import run_ipums_ab_pilot as ab


# Fixed BEFORE evaluation: no pair selection from observed metric values.
INTERACTION_PAIRS = (
    ("SEX", "EDUCD"),
    ("SEX", "OCC"),
    ("EDUCD", "OCC"),
    ("OCC", "IND"),
    ("RACE", "HISPAN"),
)
CONDITIONAL_QUERIES = (
    ("SEX", "EDUCD"),
    ("SEX", "OCC"),
    ("RACE", "EDUCD"),
)
MAX_PUBLIC_PAIR_CELLS = 1_000_000


def _public_cardinalities(categories):
    values = np.asarray(categories, dtype=np.float64)
    if (values.shape != (6,) or not np.isfinite(values).all()
            or np.any(values < 1) or np.any(values != np.floor(values))
            or np.any(values > MAX_PUBLIC_PAIR_CELLS)):
        raise ValueError("Expected six finite positive integer public cardinalities")
    return values.astype(np.int64)


def _checked_range(value, lower, upper, label):
    result = float(value)
    if not np.isfinite(result) or not lower - 1e-12 <= result <= upper + 1e-12:
        raise RuntimeError(f"{label} outside [{lower}, {upper}]")
    # Remove floating-point summation overshoot only, not an estimator change.
    return min(upper, max(lower, result))


def _joint_counts(codes, x, y, domain_sizes):
    nx, ny = int(domain_sizes[x]), int(domain_sizes[y])
    if nx * ny > MAX_PUBLIC_PAIR_CELLS:
        raise ValueError("A fixed query exceeds the bounded public-cell allocation")
    keys = codes[:, x] * ny + codes[:, y]
    return np.bincount(keys, minlength=nx * ny).reshape(nx, ny)


def _interaction_metrics(synthetic_counts, reference_counts):
    synthetic = synthetic_counts / float(synthetic_counts.sum())
    reference = reference_counts / float(reference_counts.sum())
    synthetic_residual = synthetic - np.outer(synthetic.sum(axis=1),
                                              synthetic.sum(axis=0))
    reference_residual = reference - np.outer(reference.sum(axis=1),
                                              reference.sum(axis=0))
    reference_l1 = _checked_range(np.abs(reference_residual).sum(), 0, 2,
                                  "Reference interaction-residual magnitude")
    return {
        "residual_l1": _checked_range(
            np.abs(synthetic_residual - reference_residual).sum(), 0, 4,
            "Interaction-residual mismatch"),
        "reference_residual_l1": reference_l1,
        "synthetic_residual_l1": _checked_range(
            np.abs(synthetic_residual).sum(), 0, 2,
            "Synthetic interaction-residual magnitude"),
        # An exactly independent empirical probability table has residual 0.
        # A FINITE shuffled sample generally does not have residual exactly 0.
        "independence_baseline_residual_l1": reference_l1,
        "joint_l1": _checked_range(np.abs(synthetic - reference).sum(), 0, 2,
                                    "Joint histogram L1"),
        "public_cell_count": int(synthetic_counts.size),
        "reference_nonzero_cell_count": int(np.count_nonzero(reference_counts)),
        "synthetic_nonzero_cell_count": int(np.count_nonzero(synthetic_counts)),
    }


def _conditional_metrics(synthetic_counts, reference_counts):
    """Reference-weighted conditional TV, with explicit missing-context cost.

    Every public context is retained. For reference-positive context x, use
    TV(P_synthetic(Y|x), P_reference(Y|x)) when both counts are positive. If
    synthetic count is zero, the conditional is undefined and is assigned a
    declared maximal-error penalty of 1. No missing contexts are dropped and
    the original reference weights are never renormalized over shared support.

    A reference-absent context has weight zero and no estimable conditional
    target. Its conditional TV is reported as null, not zero. Synthetic mass
    in such contexts and context-marginal TV are separately reported.
    """
    synthetic_context_counts = synthetic_counts.sum(axis=1)
    reference_context_counts = reference_counts.sum(axis=1)
    synthetic_weights = synthetic_context_counts / float(synthetic_counts.sum())
    reference_weights = reference_context_counts / float(reference_counts.sum())
    reference_positive = reference_context_counts > 0
    synthetic_positive = synthetic_context_counts > 0
    shared = reference_positive & synthetic_positive
    missing = reference_positive & ~synthetic_positive

    # Zero is only an internal placeholder for zero-weight reference contexts.
    # Undefined conditionals are exposed as null in per-context reports below.
    costs = np.zeros(len(reference_weights), dtype=np.float64)
    if np.any(shared):
        synthetic_conditional = (
            synthetic_counts[shared] / synthetic_context_counts[shared, None])
        reference_conditional = (
            reference_counts[shared] / reference_context_counts[shared, None])
        costs[shared] = 0.5 * np.abs(
            synthetic_conditional - reference_conditional).sum(axis=1)
    costs[missing] = 1.0
    weighted_costs = reference_weights * costs
    missing_mass = _checked_range(reference_weights[missing].sum(), 0, 1,
                                  "Missing reference-context mass")
    coverage = _checked_range(reference_weights[shared].sum(), 0, 1,
                              "Reference-weighted context coverage")
    score = _checked_range(weighted_costs.sum(), 0, 1,
                           "Reference-weighted conditional TV")
    if not np.isclose(coverage + missing_mass, 1.0, rtol=0, atol=1e-12):
        raise RuntimeError("Reference context weights do not sum to one")
    if score + 1e-12 < missing_mass:
        raise RuntimeError("A missing context escaped its declared penalty")

    contexts = []
    for index in range(len(reference_weights)):
        status = ("both_present" if shared[index] else
                  "missing_synthetic" if missing[index] else "reference_absent")
        contexts.append({
            "encoded_context_index": index,
            "reference_count": int(reference_context_counts[index]),
            "synthetic_count": int(synthetic_context_counts[index]),
            "reference_weight": float(reference_weights[index]),
            "conditional_tv": float(costs[index]) if shared[index] else None,
            "penalized_tv": float(costs[index]) if reference_positive[index] else None,
            "weighted_contribution": float(weighted_costs[index]),
            "status": status,
        })

    return {
        "weighted_tv": score,
        "reference_weighted_context_coverage": coverage,
        "missing_reference_context_mass": missing_mass,
        "context_marginal_tv": _checked_range(
            0.5 * np.abs(synthetic_weights - reference_weights).sum(), 0, 1,
            "Context marginal TV"),
        "synthetic_mass_in_reference_absent_contexts": _checked_range(
            synthetic_weights[~reference_positive].sum(), 0, 1,
            "Synthetic mass in reference-absent contexts"),
        "public_context_count": len(reference_weights),
        "reference_positive_context_count": int(reference_positive.sum()),
        "synthetic_missing_reference_context_count": int(missing.sum()),
        "missing_context_penalty": 1.0,
        "per_context": contexts,
    }


def evaluate_relationships(sample, reference, categories):
    """Return JSON-safe fixed-query empirical relationship metrics.

    Numeric aggregates are unweighted means over the predeclared queries,
    not means over observed cells and not population-weighted estimates.
    The caller must average nested shuffles within generation seed first and
    must keep generation variation distinct from training/target variation.
    This function does not aggregate repetitions or test significance.
    """
    cardinalities = _public_cardinalities(categories)
    synthetic_codes, _ = ab.evaluation_codes(sample, cardinalities)
    reference_codes, _ = ab.evaluation_codes(reference, cardinalities,
                                            real_records=True)
    domain_sizes = np.r_[len(ab.AGE_EDGES) + 1, cardinalities]
    counts = {}

    def query_counts(pair):
        if pair not in counts:
            x, y = (ab.FEATURES.index(name) for name in pair)
            counts[pair] = (
                _joint_counts(synthetic_codes, x, y, domain_sizes),
                _joint_counts(reference_codes, x, y, domain_sizes))
        return counts[pair]

    interaction = {" x ".join(pair): _interaction_metrics(*query_counts(pair))
                   for pair in INTERACTION_PAIRS}
    conditional = {" -> ".join(pair): _conditional_metrics(*query_counts(pair))
                   for pair in CONDITIONAL_QUERIES}
    aggregate = {
        "mean_interaction_residual_l1": float(np.mean(
            [value["residual_l1"] for value in interaction.values()])),
        "mean_conditional_tv": float(np.mean(
            [value["weighted_tv"] for value in conditional.values()])),
        "mean_context_coverage": float(np.mean(
            [value["reference_weighted_context_coverage"]
             for value in conditional.values()])),
        "mean_missing_context_mass": float(np.mean(
            [value["missing_reference_context_mass"] for value in conditional.values()])),
        "mean_context_marginal_tv": float(np.mean(
            [value["context_marginal_tv"] for value in conditional.values()])),
    }
    return {
        "schema_version": 1,
        "nonprivate": True,
        "n_synthetic": int(len(synthetic_codes)),
        "n_reference": int(len(reference_codes)),
        "interaction": interaction,
        "conditional": conditional,
        "aggregate": aggregate,
        "notes": [
            "All reference-data metrics, context weights and counts are NONPRIVATE.",
            "Queries and domains are fixed; every public cell/context is included, including zero counts.",
            "Interaction-residual mismatch is L1 in [0,4]; lower is closer to the empirical reference.",
            "Residuals depend on marginals and sampling noise; this is not a marginal-invariant dependence certificate.",
            "Conditional TV is in [0,1], weighted by original reference context frequencies; lower is closer.",
            "A reference-positive but synthetic-missing context receives penalty 1; weights are never renormalized.",
            "Reference-absent conditionals are undefined, have zero weight, and retain explicit synthetic-mass diagnostics.",
            "Context indices follow the public schema ordering, not guessed IPUMS semantic labels.",
            "Aggregate metrics are unweighted means over the fixed queries.",
            "Finite samples and sparse high-cardinality cells can have large empirical relationship-estimation error.",
            "Interpret with an exact one-way-preserving shuffle control; finite shuffles can retain chance associations.",
            "No significance test, convergence claim, higher-order preservation claim or end-to-end DP guarantee is made.",
        ],
    }


def shuffle_categorical_columns(sample, rng):
    """Independently permute each categorical column, preserving AGE exactly.

    Each synthetic empirical one-way distribution is preserved exactly.
    The input is not mutated. Finite shuffled rows are not asserted to form
    an exactly independent empirical joint table or a population null sample.
    """
    values = np.asarray(sample)
    if values.ndim != 2 or values.shape[1] != len(ab.FEATURES) or not len(values):
        raise ValueError("Expected a nonempty seven-column encoded sample")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite sample values must not be silently dropped")
    if np.any(values[:, 1:] < 0) or np.any(values[:, 1:] != np.floor(values[:, 1:])):
        raise ValueError("Categorical values must be nonnegative integer indices")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Supply an explicit numpy.random.Generator")
    shuffled = values.copy()
    for column in range(1, values.shape[1]):
        shuffled[:, column] = values[rng.permutation(len(values)), column]
        if not np.array_equal(np.sort(shuffled[:, column]), np.sort(values[:, column])):
            raise RuntimeError("Shuffle changed a categorical empirical one-way distribution")
    if not np.array_equal(shuffled[:, 0], values[:, 0]):
        raise RuntimeError("Shuffle changed AGE")
    return shuffled


def self_test():
    """Small deterministic, generated-toy-only checks; no training or file I/O."""
    cardinalities = np.array([3, 2, 2, 2, 2, 2], dtype=np.int64)
    reference = np.zeros((8, 7), dtype=np.float32)
    reference[:, 1] = np.repeat([0, 1], 4)
    reference[:, 4] = reference[:, 1]
    independent = reference.copy()
    independent[:, 4] = np.tile([0, 0, 1, 1], 2)
    original = evaluate_relationships(reference, reference, cardinalities)
    changed = evaluate_relationships(independent, reference, cardinalities)
    key = "SEX x EDUCD"
    conditional_key = "SEX -> EDUCD"
    for column in range(7):
        if not np.array_equal(np.sort(reference[:, column]), np.sort(independent[:, column])):
            raise AssertionError("Toy relationship comparison must keep the same one-ways")
    if not np.isclose(original["interaction"][key]["residual_l1"], 0):
        raise AssertionError("Identical dependent tables must have zero residual mismatch")
    if not np.isclose(changed["interaction"][key]["residual_l1"], 1):
        raise AssertionError("Residual metric failed to distinguish independence from perfect association")
    if not np.isclose(changed["interaction"][key]["synthetic_residual_l1"], 0):
        raise AssertionError("Exactly independent toy must have zero interaction residual")
    if not np.isclose(changed["conditional"][conditional_key]["weighted_tv"], 0.5):
        raise AssertionError("Conditional TV did not detect the equal-marginal relationship change")

    imperfect = reference.copy()
    imperfect[4:6, 4] = 0
    with_context = evaluate_relationships(imperfect, reference, cardinalities)
    missing_context = evaluate_relationships(imperfect[:4], reference, cardinalities)
    full_tv = with_context["conditional"][conditional_key]["weighted_tv"]
    missing_result = missing_context["conditional"][conditional_key]
    if not (np.isclose(full_tv, 0.25) and np.isclose(missing_result["weighted_tv"], 0.5)
            and missing_result["weighted_tv"] >= full_tv):
        raise AssertionError("Dropping a reference-positive context improved the conditional score")
    if not (np.isclose(missing_result["reference_weighted_context_coverage"], 0.5)
            and np.isclose(missing_result["missing_reference_context_mass"], 0.5)):
        raise AssertionError("Missing-context coverage/penalty was not explicitly retained")
    absent = missing_result["per_context"][2]
    if absent["conditional_tv"] is not None or absent["reference_weight"] != 0:
        raise AssertionError("Zero-count reference context must have undefined TV and zero weight")

    extra = reference.copy()
    extra[0, 1] = 2
    extra_result = evaluate_relationships(extra, reference, cardinalities)
    if not np.isclose(extra_result["conditional"][conditional_key]
                      ["synthetic_mass_in_reference_absent_contexts"], 1 / 8):
        raise AssertionError("Synthetic mass in reference-absent context was not exposed")
    all_missing = reference.copy()
    all_missing[:, 1] = 2
    all_missing_result = evaluate_relationships(all_missing, reference, cardinalities)
    absent_result = all_missing_result["conditional"][conditional_key]
    if not (absent_result["weighted_tv"] == 1
            and absent_result["reference_weighted_context_coverage"] == 0
            and absent_result["missing_reference_context_mass"] == 1
            and absent_result["synthetic_mass_in_reference_absent_contexts"] == 1):
        raise AssertionError("All reference contexts missing must receive full penalty without division by zero")

    shuffle_input = np.tile(reference, (8, 1))
    shuffle_input[:, 0] = np.linspace(-1, 1, len(shuffle_input), dtype=np.float32)
    before = shuffle_input.copy()
    shuffled = shuffle_categorical_columns(shuffle_input, np.random.default_rng(314159))
    if not np.array_equal(shuffle_input, before):
        raise AssertionError("Shuffle mutated its input")
    for column in range(7):
        if not np.array_equal(np.sort(shuffled[:, column]), np.sort(before[:, column])):
            raise AssertionError("Shuffle failed exact one-way invariance")
    if not np.array_equal(shuffled[:, 0], before[:, 0]):
        raise AssertionError("Shuffle moved AGE between rows")
    shuffled_result = evaluate_relationships(shuffled, shuffle_input, cardinalities)
    if shuffled_result["interaction"][key]["residual_l1"] <= 0:
        raise AssertionError("Fixed toy shuffle unexpectedly retained perfect association")

    # Unequal sample sizes and all unused public cells are legitimate. A
    # one-row self-comparison also checks all-zero residual/conditional paths.
    singleton = evaluate_relationships(reference[:1], reference[:1], cardinalities)
    if singleton["aggregate"]["mean_interaction_residual_l1"] != 0:
        raise AssertionError("Singleton self-comparison must have zero residual mismatch")
    json.dumps([original, changed, missing_context, extra_result, all_missing_result,
                shuffled_result, singleton], allow_nan=False)
    return {"status": "passed", "checks": [
        "equal-marginal perfect-dependence versus independence",
        "reference-weighted conditional TV and maximal missing-context penalty",
        "reference-absent context diagnostics and zero-count divisions",
        "exact one-way and AGE-preserving shuffle without input mutation",
        "unequal sample sizes, singleton input and strict JSON finiteness",
    ]}
