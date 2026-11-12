"""Public fixture checks for diagnostic invariants, without research records."""

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
import prototype2_evaluation as ev
import run_prototype2_diagnostics as diag


def encoded(ages, pairs=None, dtype=np.float64):
    values = np.zeros((len(ages), 7), dtype=dtype)
    values[:, 0] = (np.asarray(ages) - 50) / 50
    if pairs is not None:
        values[:, [2, 3]] = np.asarray(pairs)
    return values


class DiagnosticTests(unittest.TestCase):
    categories = np.array([2, 2, 2, 2, 3, 2], dtype=np.int64)

    def test_clamp_preserves_denominator_and_all_bins(self):
        raw = encoded([-25, 150, 35, 45])
        reference = encoded([35, 45, 35, 45])
        summaries, rows = diag.age_bin_audit(raw, reference, self.categories)
        age_rows = [r for r in rows if r["query"] == "AGE"]
        self.assertEqual(len(age_rows), 11)
        self.assertAlmostEqual(sum(r["raw_mass"] for r in age_rows), 1)
        self.assertAlmostEqual(sum(r["bounded_mass"] for r in age_rows), 1)
        self.assertEqual(age_rows[0]["raw_mass"], .25)
        self.assertEqual(age_rows[0]["bounded_mass"], 0)
        self.assertEqual(age_rows[1]["bounded_mass"], .25)
        self.assertEqual(age_rows[9]["bounded_mass"], .25)
        self.assertEqual(age_rows[10]["raw_mass"], .25)
        for summary in summaries:
            self.assertEqual(summary["raw_l1"], 1)
            self.assertEqual(summary["bounded_minus_raw_l1"], 0)
        np.testing.assert_array_equal(raw[:, 1:], diag.clamp_age(raw)[:, 1:])
        self.assertEqual(diag.age_profile(raw)["outside_0_100_fraction"], .5)
        self.assertEqual(diag.age_profile(diag.clamp_age(raw))["outside_0_100_fraction"], 0)

    def test_clamp_improves_when_endpoint_is_real_support(self):
        summaries, _ = diag.age_bin_audit(encoded([150]), encoded([100]), self.categories)
        age = summaries[0]
        self.assertEqual(age["raw_l1"], 2)
        self.assertEqual(age["bounded_l1"], 0)

    def test_real_roundtrip_is_distinct_from_generated_rounding(self):
        values = encoded([35], dtype=np.float32)
        profile = diag.age_profile(values, real_records=True)
        self.assertEqual(profile["quantiles_years"]["0.5"], 35)
        self.assertEqual(profile["roundtrip"]["records_whose_bin_changes_when_integer_roundoff_is_corrected"], 1)
        self.assertLess(diag.age_profile(values)["quantiles_years"]["0.5"], 35)

    def test_cell_changes_sum_to_joint_regression(self):
        reference = encoded([35]*4, [(0, 0), (0, 0), (1, 1), (1, 1)])
        baseline = encoded([35]*4, [(0, 0), (0, 0), (0, 0), (1, 1)])
        candidate = encoded([35]*4, [(0, 0), (0, 1), (1, 0), (1, 1)])
        codes = [ev.evaluation_codes(v, self.categories)[0]
                 for v in (reference, baseline, candidate)]
        domains = np.r_[11, self.categories]
        bt = diag.pair_tables(codes[1], codes[0], ("RACE", "HISPAN"), domains)
        ct = diag.pair_tables(codes[2], codes[0], ("RACE", "HISPAN"), domains)
        summary, cells = diag.compare_pair_tables(bt, ct)
        self.assertEqual(diag.pair_summary(bt)["joint_l1"], .5)
        self.assertEqual(diag.pair_summary(ct)["joint_l1"], 1)
        self.assertEqual(summary["C_minus_baseline_joint_l1"], .5)
        self.assertEqual(sum(c["C_minus_baseline_absolute_joint_error"] for c in cells), .5)
        self.assertAlmostEqual(sum(c["C_minus_baseline_probability"] for c in cells), 0)
        self.assertEqual(summary["omitted_cell_joint_error_change_sum"], 0)
        self.assertEqual(ev.marginal_l1(codes[2], codes[0], (2,), domains), 0)
        self.assertEqual(ev.marginal_l1(codes[1], codes[0], (2,), domains), .5)

    def test_large_domain_omission_retains_full_decomposition(self):
        domains = np.array([11, 2, 2, 2, 2, 40, 40])
        sample = np.zeros((2, 7), dtype=np.int64)
        reference = np.zeros((2, 7), dtype=np.int64)
        sample[:, 5:7] = [[0, 0], [1, 1]]
        reference[:, 5:7] = [[2, 2], [3, 3]]
        table = diag.pair_tables(sample, reference, ("OCC", "IND"), domains)
        summary = diag.pair_summary(table)
        self.assertEqual(summary["public_cells"], 1600)
        self.assertEqual(summary["csv_cells_emitted"], 4)
        self.assertEqual(summary["omitted_joint_l1_contribution"], 0)
        self.assertGreater(summary["omitted_interaction_l1_contribution"], 0)
        self.assertEqual(summary["joint_l1"], 2)
        self.assertEqual(summary["overlap_probability_mass"], 0)

    def test_real_real_is_disjoint_and_rejects_small_pool(self):
        left, right = diag.draw_disjoint_indices(12, 6, np.random.default_rng(7))
        self.assertEqual(len(set(left) & set(right)), 0)
        self.assertEqual(len(set(np.r_[left, right])), 12)
        with self.assertRaises(ValueError):
            diag.draw_disjoint_indices(11, 6, np.random.default_rng(7))

    def test_unique_control_can_be_shared_but_ambiguity_is_not_guessed(self):
        def record(arm, target, identifier):
            return dict(arm=arm, target_id=target, record_id=identifier,
                        fit_id="fit1", draw_id=1, rows=8)
        baseline = record("B_private", None, "b")
        c1, c2 = record("C", "t1", "c1"), record("C", "t2", "c2")
        matches, failures = diag.match_records([baseline, c1, c2])
        self.assertEqual([(b["record_id"], c["record_id"]) for b, c in matches],
                         [("b", "c1"), ("b", "c2")])
        self.assertTrue(all(r["baseline_arm"] == "E" for r in failures))
        other = record("B", "ambiguous_metadata", "b2")
        matches, failures = diag.match_records([baseline, other, c1])
        self.assertEqual(matches, [])
        self.assertIn("ambiguous controls for fit/draw", [r["reason"] for r in failures])
        # A string and an integer identifier do not silently become equal.
        c1["draw_id"] = "1"
        matches, _ = diag.match_records([baseline, c1])
        self.assertEqual(matches, [])

    def fixture(self, directory):
        train = encoded([18, 25, 35, 45, 55, 65, 35, 45], dtype=np.float32)
        holdout = encoded([18, 35, 45, 65], dtype=np.float32)
        data = directory / "pilot.npz"
        np.savez(data, X_train=train, X_test=holdout, categories=self.categories)
        schema = directory / "schema.json"
        schema.write_text(json.dumps({"features": ev.FEATURES,
                                      "age_transform": "(AGE-50)/50",
                                      "category_domains": {name: list(range(int(n))) for name, n in zip(ev.FEATURES[1:], self.categories)}}))
        return data, schema

    def test_complete_fixture_has_strict_json_and_shared_control_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data, schema = self.fixture(directory)
            np.save(directory / "baseline.npy", encoded([18, 35, 45, 65]))
            np.savez(directory / "candidate.npz", X=encoded([-20, 35, 45, 120]))
            records = [
                dict(arm="B_private", fit_id="f1", target_id=None, draw_id=1, path="baseline.npy", key=None),
                dict(arm="C", fit_id="f1", target_id="t1", draw_id=1, path="candidate.npz", key="X"),
                dict(arm="C", fit_id="f1", target_id="t2", draw_id=1, path="candidate.npz", key="X")]
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps(records))
            args = argparse.Namespace(data_npz=data, schema=schema, samples_manifest=manifest,
                                      out=directory / "report", seed=4, real_real_repeats=2,
                                      sample_rows=4, shuffle_repeats=2)
            report = diag.run(args)
            parsed = json.loads((args.out / "diagnostics.json").read_text(),
                                parse_constant=lambda s: self.fail(f"Nonstandard JSON constant: {s}"))
            self.assertEqual(parsed["status"], "complete")
            self.assertEqual(len(parsed["samples"]), 3)
            self.assertEqual(len(parsed["real_real"]["results"]), 2)
            self.assertEqual(len(parsed["matched_comparisons"]), 2)
            self.assertTrue(all(x["shared_control"] for x in parsed["matched_comparisons"]))
            self.assertEqual(parsed["explicit_source_reuse"], [["sample_0001", "sample_0002"]])
            self.assertEqual(report["real_real"]["rows_per_side"], 4)
            with (args.out / "real_real_controls.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(int(r["rows_per_side"]) == 4 for r in rows))
            self.assertFalse(list(args.out.glob("*.npy")))
            self.assertFalse(list(args.out.glob("*.npz")))
            with self.assertRaises(FileExistsError):
                diag.run(args)

    def test_nonfinite_failure_reports_counts_without_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data, schema = self.fixture(directory)
            bad = encoded([18, 35, 45, 65])
            bad[1, 0], bad[2, 3] = np.nan, np.inf
            np.save(directory / "bad.npy", bad)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps([dict(arm="C", fit_id=1, target_id=1, draw_id=1,
                                                path="bad.npy", key=None)]))
            args = argparse.Namespace(data_npz=data, schema=schema, samples_manifest=manifest,
                                      out=directory / "report", seed=4, real_real_repeats=0,
                                      sample_rows=512, shuffle_repeats=0)
            with self.assertRaises(diag.NonfiniteDataError):
                diag.run(args)
            failure = json.loads((args.out / "failure.json").read_text())
            self.assertEqual(failure["nonfinite_counts"]["values"], 2)
            self.assertEqual(failure["nonfinite_counts"]["rows"], 2)
            self.assertFalse((args.out / "diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
