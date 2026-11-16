"""Focused generated-fixture tests for the AGE-only evaluation control."""

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prototype2_evaluation as evaluation
import run_prototype2_age_permutation as age


CATEGORIES = np.array([2, 2, 2, 2, 2, 2], dtype=np.int64)


def dependent_table(rows):
    latent = np.arange(rows) % 2
    result = np.zeros((rows, 7), dtype=np.float64)
    result[:, 0] = (np.where(latent == 0, 20.0, 60.0) - 50.0) / 50.0
    # Strong AGE alignment plus strong categorical relationships, which the
    # AGE-only control must leave intact in each individual categorical row.
    result[:, 1:] = latent[:, None]
    return result


def make_fixture(directory):
    data = directory / "pilot.npz"
    np.savez(data, X_train=dependent_table(128), X_test=dependent_table(32),
             categories=CATEGORIES)
    schema = directory / "schema.json"
    schema.write_text(json.dumps({
        "features": evaluation.FEATURES, "age_transform": "(AGE-50)/50",
        "category_domains": {name: [0, 1] for name in evaluation.FEATURES[1:]},
        "public_fixture": True, "data_origin": "programmatically_generated_no_real_records",
    }))
    gpu, pgm = [], []
    for draw in (1, 2):
        for arm in ("B_private", "C"):
            path = directory / f"{arm}_{draw}.npy"
            np.save(path, dependent_table(32), allow_pickle=False)
            gpu.append({"arm": arm, "fit_id": "gpu_fit1",
                        "target_id": "target01" if arm == "C" else None,
                        "draw_id": draw, "path": path.name,
                        "sha256": age.diagnostics.sha256(path)})
        for fit in (1, 2, 3):
            path = directory / f"aim_{fit}_{draw}.npy"
            np.save(path, dependent_table(32), allow_pickle=False)
            pgm.append({"arm": "aim", "fit_id": f"fit{fit:03d}", "target_id": None,
                        "draw_id": f"draw{draw:03d}", "path": path.name,
                        "sha256": age.diagnostics.sha256(path)})
    manifests = []
    for filename, entries in (("gpu_manifest.json", gpu), ("pgm_manifest.json", pgm)):
        path = directory / filename
        path.write_text(json.dumps({"public_fixture": True, "samples": entries}))
        manifests.append(path)
    return data, schema, manifests


class AgePermutationTests(unittest.TestCase):
    def test_preserves_categorical_rows_AGE_values_and_input(self):
        sample = dependent_table(128)
        # Retain out-of-domain AGE rather than filtering or clipping it.
        sample[0, 0], sample[1, 0] = -3.0, 3.0
        before = sample.copy()
        shuffled = age.permute_age(sample, np.random.default_rng(4))
        np.testing.assert_array_equal(sample, before)
        np.testing.assert_array_equal(shuffled[:, 1:], sample[:, 1:])
        np.testing.assert_array_equal(np.sort(shuffled[:, 0]), np.sort(sample[:, 0]))
        self.assertFalse(np.array_equal(shuffled[:, 0], sample[:, 0]))
        original_codes, original_age = evaluation.evaluation_codes(sample, CATEGORIES)
        shuffled_codes, shuffled_age = evaluation.evaluation_codes(shuffled, CATEGORIES)
        self.assertEqual(original_age, shuffled_age)
        self.assertEqual(evaluation.marginal_l1(original_codes, shuffled_codes, (5, 6),
                                              np.r_[11, CATEGORIES]), 0)

    def test_AGE_dependent_toy_detects_alignment_without_breaking_categorical_joint(self):
        sample = dependent_table(256)
        result = age.evaluate_age_permutations(sample, sample.copy(), CATEGORIES,
                                              repeats=20, seed=123)
        self.assertEqual(result["metrics"]["two_way/AGE x SEX"]["original"], 0)
        self.assertGreater(result["metrics"]["two_way/AGE x SEX"]["age_permuted_mean"], .8)
        self.assertGreater(result["metrics"]["three_way/AGE x OCC x IND"]["age_permuted_minus_original"], .8)
        self.assertTrue(result["checks"]["categorical_rows_exactly_unchanged"])
        self.assertEqual(result["age_one_way_l1_unchanged"], 0)

    def test_constant_AGE_control_has_zero_difference(self):
        sample = dependent_table(64)
        sample[:, 0] = 0.0
        result = age.evaluate_age_permutations(sample, sample.copy(), CATEGORIES,
                                              repeats=3, seed=18)
        self.assertTrue(all(m["age_permuted_minus_original"] == 0
                            for m in result["metrics"].values()))

    def test_common_stream_normalizes_draw_ids_and_is_manifest_order_independent(self):
        seeds = [age.permutation_seed(20261112, draw, 512)
                 for draw in (1, "1", "draw001", "generation_01", "Generation01")]
        self.assertEqual(len(set(seeds)), 1)
        self.assertNotEqual(seeds[0], age.permutation_seed(20261112, 2, 512))
        self.assertNotEqual(seeds[0], age.permutation_seed(20261112, 1, 64))
        self.assertNotEqual(seeds[0], age.permutation_seed(20261112, 1, 512, control=True))
        self.assertNotEqual(age.permutation_seed(1, "custom_a", 32),
                            age.permutation_seed(1, "custom_b", 32))

    def test_fit_summary_does_not_count_draws_or_targets_as_fits(self):
        def row(arm, fit, target, draw, value):
            return {"arm": arm, "fit_id": fit, "target_id": target, "draw_id": draw,
                    "metrics": {"test_metric": {field: value for field in age.METRIC_FIELDS}}}
        records = [row("aim", "f1", None, 1, 0), row("aim", "f1", None, 2, 2),
                   row("aim", "f2", None, 1, 9),
                   row("C", "f1", "t1", 1, 0), row("C", "f1", "t1", 2, 2),
                   row("C", "f1", "t2", 1, 9)]
        result = age.aggregate_fit_groups(records)
        self.assertEqual(result["aim"]["n_fit_groups"], 2)
        self.assertEqual(result["aim"]["metrics"]["test_metric"]["original"]["mean"], 5)
        self.assertEqual(result["C"]["n_fit_groups"], 1)
        self.assertEqual(result["C"]["metrics"]["test_metric"]["original"]["mean"], 5)
        self.assertIsNone(result["C"]["metrics"]["test_metric"]["original"]["between_fit_sample_sd"])

    def test_combined_fixture_run_uses_real_outputs_and_disjoint_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data, schema, manifests = make_fixture(directory)
            args = argparse.Namespace(data_npz=data, schema=schema, samples_manifest=manifests,
                                      out=directory / "out", seed=20261112,
                                      permutation_repeats=3, real_real_repeats=2,
                                      sample_rows=32, public_fixture=True)
            result = age.run(args)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["sample_count"], 10)
            self.assertEqual(result["permutation_type"], "AGE_only")
            self.assertFalse(result["training_performed"])
            self.assertEqual(result["new_marginal_releases"], 0)
            self.assertEqual(result["aggregate"]["aim"]["n_fit_groups"], 3)
            self.assertEqual(result["aggregate"]["B_private"]["n_fit_groups"], 1)
            self.assertTrue(all(r["within_repeat_index_overlap_count"] == 0
                                for r in result["real_real"]["results"]))
            self.assertFalse(result["real_real"]["across_repeat_independent"])
            self.assertEqual(len({s["permutation_seed"] for s in result["samples"]}), 2)
            for sample in result["samples"]:
                self.assertEqual(sample["sha256"], age.diagnostics.sha256(sample["path"]))
                self.assertTrue(sample["manifest_declared_sample_hash_verified"])
                self.assertNotIn("values", sample)
            expected = {"age_permutation.json", "metrics_per_draw.csv", "real_real_controls.csv", "README.txt"}
            self.assertEqual({p.name for p in args.out.iterdir()}, expected)
            text = (args.out / "age_permutation.json").read_text()
            reread = json.loads(text, parse_constant=lambda x: self.fail("Nonfinite JSON value"))
            self.assertEqual(reread["sample_count"], 10)
            with (args.out / "metrics_per_draw.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 110)
            with (args.out / "real_real_controls.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 22)
            with self.assertRaises(FileExistsError):
                age.run(args)

    def test_real_input_gate_and_fixture_opt_in_are_not_silently_bypassed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data, schema, manifests = make_fixture(directory)
            args = argparse.Namespace(data_npz=data, schema=schema, samples_manifest=manifests,
                                      out=directory / "real_attempt", seed=1,
                                      permutation_repeats=2, real_real_repeats=0,
                                      sample_rows=32, public_fixture=False)
            with self.assertRaisesRegex(ValueError, "Historical input hash mismatch"):
                age.run(args)
            declaration = json.loads(schema.read_text())
            declaration["public_fixture"] = False
            schema.write_text(json.dumps(declaration))
            args.public_fixture = True
            args.out = directory / "wrong_fixture"
            with self.assertRaisesRegex(ValueError, "explicitly declared generated fixture"):
                age.run(args)

    def test_manifest_integrity_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, _, manifests = make_fixture(directory)
            with self.assertRaisesRegex(ValueError, "more than once"):
                age.load_manifests([manifests[0], manifests[0]], CATEGORIES, public_fixture=True)
            payload = json.loads(manifests[0].read_text())
            payload["samples"][0]["sha256"] = "0" * 64
            manifests[0].write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "hash differs"):
                age.load_manifests([manifests[0]], CATEGORIES, public_fixture=True)

    def test_real_record_rounding_matches_common_evaluator(self):
        values = np.zeros((8, 7), dtype=np.float32)
        values[:, 0] = (35.0 - 50.0) / 50.0
        repaired = age.evaluate_age_permutations(values, values, CATEGORIES,
                                                repeats=2, seed=1, sample_real_records=True)
        self.assertEqual(repaired["age_one_way_l1_unchanged"], 0)
        generated = age.evaluate_age_permutations(values, values, CATEGORIES,
                                                 repeats=2, seed=1, sample_real_records=False)
        self.assertEqual(generated["age_one_way_l1_unchanged"], 2)


if __name__ == "__main__":
    unittest.main()
