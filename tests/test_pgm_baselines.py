"""Contract checks for baseline comparability and provenance, without private data."""

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prototype2_pgm_backend as pgm
from prototype2_evaluation import evaluation_codes
from run_ipums_pgm_baselines import aggregate_fit_metrics, make_parser, validate_args


CATEGORIES = np.array([3, 9, 6, 44, 531, 266])


def table_with_ages(ages, dtype=np.float64):
    result = np.zeros((len(ages), 7), dtype=dtype)
    result[:, 0] = (np.asarray(ages, dtype=float) - 50) / 50
    return result


def parsed(*flags):
    return make_parser().parse_args([
        "--data-npz", "fixture.npz", "--schema", "schema.json",
        "--mbi-repo", "/unused", "--expected-mbi-commit", pgm.PINNED_MBI_COMMIT,
        *flags,
    ])


def fixture_schema():
    return {"public_fixture": True, "data_origin": "programmatically_generated_no_real_records"}


def test_integer_age_encoding_preserves_all_public_bin_boundaries():
    real = table_with_ages(np.arange(101), np.float32)
    discrete, domains = pgm.encode_training_data(real, CATEGORIES)
    assert domains["AGE"] == 101
    np.testing.assert_array_equal(discrete[:, 0], np.arange(101))
    generated = pgm.decode_synthetic_data(discrete, domains)
    assert generated.dtype == np.float64
    real_codes, _ = evaluation_codes(real, CATEGORIES, real_records=True)
    generated_codes, age = evaluation_codes(generated, CATEGORIES)
    np.testing.assert_array_equal(real_codes, generated_codes)
    assert age["decoded_outside_0_100_fraction"] == 0


def test_legacy_bin_mapping_is_public_and_faithful_on_evaluation_queries():
    real = table_with_ages([0, 17, 18, 24, 25, 34, 35, 64, 65, 84, 85, 100], np.float32)
    discrete, domains = pgm.encode_training_data(real, CATEGORIES, "legacy-bins")
    assert domains["AGE"] == 9
    generated = pgm.decode_synthetic_data(discrete, domains, "legacy-bins")
    before, _ = evaluation_codes(real, CATEGORIES, real_records=True)
    after, _ = evaluation_codes(generated, CATEGORIES)
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("mutation", ["nonfinite", "fractional_cat", "negative_cat", "beyond_cat", "fractional_age", "age_below", "age_above"])
def test_training_domain_violations_raise_without_filtering(mutation):
    real = table_with_ages([18, 35])
    if mutation == "nonfinite":
        real[0, 3] = np.nan
    elif mutation == "fractional_cat":
        real[0, 1] = 0.5
    elif mutation == "negative_cat":
        real[0, 1] = -1
    elif mutation == "beyond_cat":
        real[0, 1] = 3
    elif mutation == "fractional_age":
        real[0, 0] = (18.2 - 50) / 50
    elif mutation == "age_below":
        real[0, 0] = -1.02
    else:
        real[0, 0] = 1.02
    with pytest.raises(ValueError):
        pgm.encode_training_data(real, CATEGORIES)


def test_full_workload_keeps_high_cardinality_occupation_industry_pair():
    _, domains = pgm.encode_training_data(table_with_ages([18]), CATEGORIES)
    workload, excluded = pgm.aim_workload(domains)
    assert len(workload) == 21
    assert (("OCC", "IND"), 1.0) in workload
    assert excluded == []
    restricted, exclusions = pgm.aim_workload(domains, max_cells=10000)
    assert len(restricted) < 21
    item = next(x for x in exclusions if x["query"] == ["OCC", "IND"])
    assert item["cells"] == 141246
    assert "explicit" in item["reason"]


def test_exclusions_cannot_silently_remove_an_attribute():
    _, domains = pgm.encode_training_data(table_with_ages([18]), CATEGORIES)
    with pytest.raises(ValueError, match="every pair"):
        pgm.aim_workload(domains, max_cells=20)


def test_generation_repeats_do_not_inflate_inference_sample_size():
    # Per-fit metrics are averages of arbitrary numbers of generation draws.
    summary = aggregate_fit_metrics([
        {"arm": "aim", "generation_mean_metrics": {"error": 0.3}},
        {"arm": "aim", "generation_mean_metrics": {"error": 0.7}},
        {"arm": "mst", "generation_mean_metrics": {"error": 0.5}},
    ])
    assert summary["aim"]["error"]["independent_fits"] == 2
    assert summary["aim"]["error"]["mean"] == pytest.approx(0.5)
    assert summary["aim"]["error"]["between_fit_sample_sd"] == pytest.approx(np.std([0.3, 0.7], ddof=1))
    assert summary["mst"]["error"]["between_fit_sample_sd"] is None


def test_private_random_scope_uses_secret_seed_and_restores_global_state(monkeypatch):
    calls = []
    monkeypatch.setattr(pgm.secrets, "token_bytes", lambda n: calls.append(n) or bytes(range(n)))
    np.random.seed(901)
    before = np.random.get_state()
    with pgm.numpy_random_scope():
        np.random.normal(size=20)
    after = np.random.get_state()
    assert calls == [32]
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_public_fixture_random_scope_is_explicitly_reproducible():
    with pgm.numpy_random_scope(810):
        a = np.random.normal(size=10)
    with pgm.numpy_random_scope(810):
        b = np.random.normal(size=10)
    np.testing.assert_array_equal(a, b)
    assert pgm.public_seed(1, "aim", "fit1", "draw1") != pgm.public_seed(1, "aim", "fit1", "draw2")


def test_smoke_cannot_be_mislabeled_as_real_or_public_data():
    with pytest.raises(ValueError, match="requires --public-fixture"):
        validate_args(parsed("--smoke"), {})
    with pytest.raises(ValueError, match="generated data"):
        validate_args(parsed("--public-fixture", "--smoke"), {})
    with pytest.raises(ValueError, match="pass --public-fixture"):
        validate_args(parsed(), fixture_schema())
    args = parsed("--public-fixture", "--smoke")
    validate_args(args, fixture_schema())
    assert (args.aim_max_model_size, args.aim_max_iters, args.mst_selection_iters, args.mst_final_iters) == (1, 25, 25, 50)


def test_standard_inference_defaults_are_not_the_smoke_settings():
    args = parsed()
    validate_args(args, {})
    assert (args.aim_max_model_size, args.aim_max_iters, args.mst_selection_iters, args.mst_final_iters) == (80, 1000, 2500, 10000)
    assert args.aim_max_cells is None
    assert args.epsilon == 2.5638977279554265
    assert args.delta == 1e-5


def test_fit_api_has_no_evaluation_or_true_target_inputs():
    params = set(inspect.signature(pgm.OfficialPGMBackend.fit).parameters)
    assert not {"holdout", "validation", "reference", "targets", "evaluation"} & params


def test_wrong_official_commit_is_rejected_before_any_import(tmp_path, monkeypatch):
    answers = {("rev-parse", "--show-toplevel"): str(tmp_path),
               ("rev-parse", "HEAD"): "a" * 40}
    monkeypatch.setattr(pgm, "_git", lambda _repo, *args: answers[args])
    with pytest.raises(ValueError, match="commit mismatch"):
        pgm.verify_official_checkout(tmp_path, pgm.PINNED_MBI_COMMIT)


def test_lookalike_repository_origin_is_rejected(tmp_path, monkeypatch):
    answers = {("rev-parse", "--show-toplevel"): str(tmp_path),
               ("rev-parse", "HEAD"): pgm.PINNED_MBI_COMMIT,
               ("remote", "get-url", "origin"): "https://github.com/example/bootstrap-stub.git"}
    monkeypatch.setattr(pgm, "_git", lambda _repo, *args: answers[args])
    with pytest.raises(ValueError, match="not the official"):
        pgm.verify_official_checkout(tmp_path, pgm.PINNED_MBI_COMMIT)


def test_generation_uses_iid_mode_and_reuses_same_model():
    class Draw:
        def __init__(self, n):
            self.n = n
        def to_dict(self):
            return {name: np.zeros(self.n, dtype=int) for name in pgm.FEATURES}
    class Model:
        calls = []
        def synthetic_data(self, *, rows, method):
            self.calls.append((rows, method))
            return Draw(rows)
    model = Model()
    fitted = pgm.FittedPGM(model, None, dict(zip(pgm.FEATURES, [101, 3, 9, 6, 44, 531, 266])), {})
    fitted.sample_codes(8, 1)
    fitted.sample_codes(8, 2)
    assert fitted.model is model
    assert model.calls == [(8, "sample"), (8, "sample")]


def test_mst_compression_supports_modern_domain_and_preserves_weighted_measurement():
    # Deliberately has no legacy .attrs property: this was an actual upstream
    # incompatibility found by the generated-data integration run.
    class Domain(dict):
        @staticmethod
        def fromdict(sizes):
            return Domain(sizes)
    class Dataset:
        def __init__(self, columns, domain):
            self.columns, self.domain = columns, domain
        def to_dict(self):
            return self.columns
    def Measurement(y, clique, stddev, query=None):
        return SimpleNamespace(noisy_measurement=np.asarray(y), clique=clique,
                               stddev=stddev, query=query)
    backend = pgm.OfficialPGMBackend.__new__(pgm.OfficialPGMBackend)
    backend.mbi = SimpleNamespace(Dataset=Dataset, Domain=Domain, LinearMeasurement=Measurement)
    original = Dataset({"mixed": np.array([0, 1, 2]), "kept": np.array([0, 1, 0]),
                        "collapsed": np.array([0, 1, 2])},
                       Domain({"mixed": 3, "kept": 2, "collapsed": 3}))
    measurements = [Measurement([12., 2., -1.], ("mixed",), 2.),
                    Measurement([8., 10.], ("kept",), 2.),
                    Measurement([0., -1., 1.], ("collapsed",), 2.)]
    compressed, log, undo = backend._compress_mst_data(original, measurements)
    assert compressed.domain == {"mixed": 2, "kept": 2, "collapsed": 1}
    np.testing.assert_array_equal(compressed.to_dict()["mixed"], [0, 1, 1])
    np.testing.assert_array_equal(compressed.to_dict()["collapsed"], [0, 0, 0])
    np.testing.assert_allclose(log[0].noisy_measurement, [12., 1 / np.sqrt(2)])
    assert log[0].stddev == 2
    np.testing.assert_allclose(log[0].query(SimpleNamespace(datavector=lambda: np.array([4., 6.]))),
                               [4., 6 / np.sqrt(2)])
    assert log[1] is measurements[1]
    with pgm.numpy_random_scope(54):
        expected_mixed = np.r_[0, np.random.choice([1, 2], 2)]
        expected_collapsed = np.random.choice([0, 1, 2], 3)
        expected_next = np.random.random()
    with pgm.numpy_random_scope(54):
        reconstructed = undo(compressed)
        actual_next = np.random.random()
    assert reconstructed.domain == original.domain
    np.testing.assert_array_equal(reconstructed.to_dict()["mixed"], expected_mixed)
    np.testing.assert_array_equal(reconstructed.to_dict()["kept"], [0, 1, 0])
    np.testing.assert_array_equal(reconstructed.to_dict()["collapsed"], expected_collapsed)
    assert actual_next == expected_next
