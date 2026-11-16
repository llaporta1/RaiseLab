"""Generated public fixtures for the shared evaluator, with no real records."""

import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prototype2_evaluation as ev
import prototype2_relationship_checks as relationships


def encoded(ages, dtype=np.float64):
    ages = np.asarray(ages, dtype=dtype)
    table = np.zeros((len(ages), 7), dtype=dtype)
    table[:, 0] = (ages - 50) / 50
    return table


class EvaluationTests(unittest.TestCase):
    categories = np.array([3, 2, 2, 2, 2, 2], dtype=np.int64)

    def test_tail_codes_and_separate_support_fraction(self):
        ages = [-4, 0, 17, 18, 24, 25, 34, 35, 44, 45, 54, 55,
                64, 65, 74, 75, 84, 85, 100, 100.5, 101, 130]
        expected = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                    6, 7, 7, 8, 8, 9, 9, 9, 10, 10]
        codes, info = ev.evaluation_codes(encoded(ages), self.categories)
        np.testing.assert_array_equal(codes[:, 0], expected)
        self.assertEqual(sum(info["histogram_counts"]), len(ages))
        self.assertAlmostEqual(info["decoded_outside_0_100_fraction"], 4 / len(ages))

    def test_real_rounding_repairs_float32_boundary_only(self):
        table = encoded([35], np.float32)
        synthetic, _ = ev.evaluation_codes(table, self.categories)
        real, _ = ev.evaluation_codes(table, self.categories, real_records=True)
        self.assertEqual(synthetic[0, 0], 3)
        self.assertEqual(real[0, 0], 4)
        self.assertEqual(table.dtype, np.float32)

    def test_cohort_support_distinguishes_65_from_66_and_74_in_same_bin(self):
        sample = encoded([65, 66, 74])
        reference = encoded([65, 65, 65])
        codes, info = ev.evaluation_codes(sample, self.categories)
        np.testing.assert_array_equal(codes[:, 0], [7, 7, 7])
        self.assertEqual(info["decoded_outside_0_100_fraction"], 0)
        self.assertAlmostEqual(info["decoded_outside_18_65_fraction"], 2 / 3)
        result = ev.evaluate_sample(sample, reference, self.categories, shuffle_repeats=1)
        self.assertEqual(result["marginals"]["age_one_way"], 0)
        self.assertAlmostEqual(ev.flatten_metrics(result)["marginal/age_outside_cohort_fraction"], 2 / 3)
        self.assertEqual(result["shuffle"]["shuffle_minus_original"]["marginal/age_outside_cohort_fraction"], 0)
        _, lower_boundary = ev.evaluation_codes(encoded([17, 18]), self.categories)
        self.assertEqual(lower_boundary["decoded_outside_18_65_fraction"], .5)

    def test_generated_integer_float64_adapter_preserves_public_boundaries(self):
        ages = np.arange(101, dtype=np.float64)
        codes, _ = ev.evaluation_codes(encoded(ages), self.categories)
        # This comparison is against the literal integer age workload, not a
        # second round trip through the implementation under test.
        expected = np.searchsorted(ev.AGE_EDGES, ages, side="right")
        np.testing.assert_array_equal(codes[:, 0], expected)

    def test_invalid_rows_are_rejected_instead_of_filtered(self):
        for value, column in [(np.nan, 0), (np.inf, 3), (-1, 1), (1.5, 2), (2, 3)]:
            with self.subTest(value=value, column=column):
                table = encoded([20, 40])
                table[1, column] = value
                with self.assertRaises(ValueError):
                    ev.evaluation_codes(table, self.categories)
        with self.assertRaises(ValueError):
            ev.evaluation_codes(np.empty((0, 7)), self.categories)
        with self.assertRaises(ValueError):
            ev.evaluation_codes(encoded([20]), [3, 2, 2, 2, 2, 0])

    def test_sparse_matches_independent_dense_histograms_orders_one_to_three(self):
        rng = np.random.default_rng(502)
        sizes = np.array([3, 4, 2])
        left = np.column_stack([rng.integers(n, size=13) for n in sizes])
        right = np.column_stack([rng.integers(n, size=19) for n in sizes])
        for order in (1, 2, 3):
            for columns in itertools.combinations(range(3), order):
                shape = tuple(int(sizes[i]) for i in columns)
                a, b = np.zeros(shape), np.zeros(shape)
                for row in left:
                    a[tuple(row[i] for i in columns)] += 1
                for row in right:
                    b[tuple(row[i] for i in columns)] += 1
                expected = np.abs(a / len(left) - b / len(right)).sum()
                self.assertAlmostEqual(ev.marginal_l1(left, right, columns, sizes), expected)

    def test_three_way_keys_do_not_overflow_or_allocate_cartesian_domain(self):
        sizes = [2**31, 2**31, 2**31]
        left = np.array([[0, 0, 0], [2**31 - 1, 2**31 - 1, 2**31 - 1]])
        right = np.array([[0, 0, 0]])
        self.assertEqual(ev.marginal_l1(left, right, (0, 1, 2), sizes), 1)

    def test_tail_rows_remain_in_full_denominator(self):
        sample, _ = ev.evaluation_codes(encoded([-20, 35]), self.categories)
        reference, _ = ev.evaluation_codes(encoded([35, 35]), self.categories, True)
        self.assertEqual(ev.marginal_l1(sample, reference, (0,), np.r_[11, self.categories]), 1)

    def test_clamp_moves_mass_between_empty_reference_bins_without_l1_change(self):
        raw = encoded([-100, 150, 35, 45])
        clamped = encoded([0, 100, 35, 45])
        reference = encoded([35, 45, 35, 45])
        raw_codes, raw_age = ev.evaluation_codes(raw, self.categories)
        clamp_codes, clamp_age = ev.evaluation_codes(clamped, self.categories)
        real_codes, _ = ev.evaluation_codes(reference, self.categories, True)
        self.assertEqual(raw_age["decoded_outside_0_100_fraction"], .5)
        self.assertEqual(clamp_age["decoded_outside_0_100_fraction"], 0)
        a = ev.compare_marginals(raw_codes, real_codes, np.r_[11, self.categories])
        b = ev.compare_marginals(clamp_codes, real_codes, np.r_[11, self.categories])
        self.assertEqual(a, b)

    def test_legacy_numeric_results_are_preserved(self):
        # Compile only the historical NumPy metric functions, avoiding the
        # legacy runner's training/Torch/Opacus imports and main routine.
        source = ast.parse((SCRIPTS / "run_ipums_ab_pilot.py").read_text())
        wanted = {"evaluation_codes", "marginal_l1", "compare_marginals"}
        tree = ast.Module(body=[node for node in source.body
                               if isinstance(node, ast.FunctionDef) and node.name in wanted],
                          type_ignores=[])
        legacy = {"np": np, "itertools": itertools, "FEATURES": ev.FEATURES,
                  "AGE_EDGES": ev.AGE_EDGES}
        exec(compile(tree, "historical_metric_functions", "exec"), legacy)
        rng = np.random.default_rng(81)
        sample = encoded(rng.uniform(-25, 130, 23))
        reference = encoded(rng.integers(18, 66, 31), np.float32)
        for j, count in enumerate(self.categories, 1):
            sample[:, j] = rng.integers(count, size=len(sample))
            reference[:, j] = rng.integers(count, size=len(reference))
        codes, age = ev.evaluation_codes(sample, self.categories)
        old_codes, old_age = legacy["evaluation_codes"](sample, self.categories)
        np.testing.assert_array_equal(codes, old_codes)
        for key in old_age:
            self.assertEqual(age[key], old_age[key])
        real_codes, _ = ev.evaluation_codes(reference, self.categories, True)
        expected = legacy["compare_marginals"](codes, real_codes, np.r_[11, self.categories])
        actual = ev.compare_marginals(codes, real_codes, np.r_[11, self.categories])
        for key in expected:
            if isinstance(expected[key], dict):
                for query in expected[key]:
                    self.assertAlmostEqual(actual[key][query], expected[key][query])
            else:
                self.assertAlmostEqual(actual[key], expected[key])
        self.assertEqual(len(actual["two_way"]), 21)
        self.assertEqual(len(actual["three_way"]), 3)

    def test_missing_context_penalty_keeps_original_reference_weights(self):
        reference = encoded([40] * 8)
        reference[:, 1] = [0] * 4 + [1] * 4
        reference[:, 4] = reference[:, 1]
        sample = reference[:4].copy()
        result = ev.evaluate_sample(sample, reference, self.categories, shuffle_repeats=0)
        query = result["relationships"]["conditional"]["SEX -> EDUCD"]
        self.assertEqual(query["weighted_tv"], .5)
        self.assertEqual(query["missing_reference_context_mass"], .5)
        self.assertEqual(query["reference_weighted_context_coverage"], .5)
        self.assertIsNone(query["per_context"][2]["conditional_tv"])
        self.assertIsNone(result["shuffle"]["mean"])
        self.assertEqual(result["shuffle"]["per_shuffle"], [])

    def test_shuffle_preserves_all_one_ways_and_age_without_mutating_input(self):
        sample = encoded(np.arange(64) + 10)
        for j, count in enumerate(self.categories, 1):
            sample[:, j] = np.arange(64) % count
        before = sample.copy()
        shuffled = relationships.shuffle_categorical_columns(sample, np.random.default_rng(4))
        np.testing.assert_array_equal(sample, before)
        np.testing.assert_array_equal(shuffled[:, 0], sample[:, 0])
        for column in range(7):
            np.testing.assert_array_equal(np.sort(sample[:, column]), np.sort(shuffled[:, column]))
        a = ev.evaluate_sample(sample, sample, self.categories, shuffle_repeats=3, seed=91)
        b = ev.evaluate_sample(sample, sample, self.categories, shuffle_repeats=3, seed=91)
        self.assertEqual(a, b)
        original = ev.flatten_metrics(a)
        for row in a["shuffle"]["per_shuffle"]:
            for feature in ev.FEATURES:
                self.assertEqual(row["one_way/" + feature], original["one_way/" + feature])
        json.dumps(a, allow_nan=False)

    def test_real_real_controls_round_both_arrays_including_shuffles(self):
        real = encoded([35, 35], np.float32)
        result = ev.evaluate_sample(real, real, self.categories, shuffle_repeats=2,
                                    sample_real_records=True)
        self.assertEqual(result["marginals"]["age_one_way"], 0)
        self.assertEqual(result["shuffle"]["mean"]["one_way/AGE"], 0)
        numerical_sample = ev.evaluate_sample(real, real, self.categories, shuffle_repeats=0)
        self.assertEqual(numerical_sample["marginals"]["age_one_way"], 2)

    def test_complete_evaluation_imports_no_training_framework(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "import numpy as np; import prototype2_evaluation as e; "
                "e.evaluate_sample(np.zeros((2,7)),np.zeros((2,7)),[2]*6,shuffle_repeats=1); "
                "assert 'torch' not in sys.modules; assert 'opacus' not in sys.modules")
        subprocess.run([sys.executable, "-c", code, str(SCRIPTS)], check=True)


class PilotLoadingTests(unittest.TestCase):
    def write_fixture(self, root, *, train=None, holdout=None, mutate_schema=None):
        categories = np.array([3, 2, 2, 2, 2, 2])
        train = encoded([0, 35, 100], np.float32) if train is None else train
        holdout = encoded([18, 65], np.float32) if holdout is None else holdout
        schema = {"features": ev.FEATURES, "age_transform": "clip AGE to [0,100], then (AGE-50)/50",
                  "category_domains": {name: list(range(int(n)))
                                       for name, n in zip(ev.FEATURES[1:], categories)},
                  "train_rows": len(train), "test_rows": len(holdout)}
        if mutate_schema:
            mutate_schema(schema)
        data = Path(root) / "public_fixture.npz"
        schema_path = Path(root) / "public_schema.json"
        np.savez(data, X_train=train, X_test=holdout, categories=categories)
        schema_path.write_text(json.dumps(schema))
        return data, schema_path

    def test_load_preserves_rows_and_does_not_infer_cohort_filter(self):
        with tempfile.TemporaryDirectory() as root:
            data, schema_path = self.write_fixture(root)
            train, holdout, categories, schema = ev.load_pilot(data, schema_path)
            self.assertEqual((train.shape, holdout.shape), ((3, 7), (2, 7)))
            self.assertEqual(train.dtype, np.float32)
            np.testing.assert_array_equal(train[[0, 2], 0], [-1, 1])
            self.assertEqual(categories.dtype, np.int64)
            self.assertEqual(schema["features"], ev.FEATURES)

    def test_rejects_stale_or_incomplete_schema(self):
        mutations = [
            lambda s: s.update(features=list(reversed(ev.FEATURES))),
            lambda s: s.update(age_transform="standardize using fitted mean"),
            lambda s: s["category_domains"].pop("OCC"),
            lambda s: s["category_domains"].update(SEX=[0, 1]),
            lambda s: s["category_domains"].update(SEX=[0, 1, 1]),
            lambda s: s.update(train_rows=999),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                paths = self.write_fixture(root, mutate_schema=mutation)
                with self.assertRaises(ValueError):
                    ev.load_pilot(*paths)

    def test_validates_every_holdout_value_before_evaluation(self):
        for value, column in [(np.nan, 0), (1.1, 0), (3, 1), (.5, 4)]:
            with self.subTest(value=value, column=column), tempfile.TemporaryDirectory() as root:
                holdout = encoded([18, 65])
                holdout[1, column] = value
                paths = self.write_fixture(root, holdout=holdout)
                with self.assertRaises(ValueError):
                    ev.load_pilot(*paths)


if __name__ == "__main__":
    unittest.main()
