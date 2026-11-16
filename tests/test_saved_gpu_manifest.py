"""Saved-GPU manifest contracts exercised only with generated public arrays.

The real original-input gate is mocked at its call boundary for these unit
fixtures. Production code has no hash override or fixture bypass. No historical
record, checkpoint, or private target is included in this test module.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_prototype2_saved_gpu_manifest as prepare
import run_prototype2_diagnostics as diagnostics


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]
CATEGORIES = np.array([3, 9, 6, 44, 531, 266], dtype=np.int64)
ARMS = ("A_diag", "clip_only", "B_private", "E_matched", "C")
SEEDS = [12026, 12126, 12226]
DATA_HASH = "677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906"
SCHEMA_HASH = "3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73"
DATA_PLAN_KEY = "pipeline_runs/prototype2_cpu/ipums_2048/pilot_data.npz"
SCHEMA_PLAN_KEY = "pipeline_runs/prototype2_cpu/ipums_2048/schema.json"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generated_array(seed):
    """Return a reproducible 512-by-7 software fixture, never real records."""
    rng = np.random.default_rng(seed)
    values = np.empty((512, 7), dtype=np.float64)
    values[:, 0] = (rng.integers(18, 66, size=512) - 50) / 50
    for column, size in enumerate(CATEGORIES, start=1):
        values[:, column] = rng.integers(0, size, size=512)
    return values


def write_sample(path, *, values=None, features=FEATURES, categories=CATEGORIES):
    np.savez_compressed(
        path,
        X_synthetic=generated_array(712) if values is None else values,
        features=np.asarray(features),
        categories=np.asarray(categories),
    )


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    run = tmp_path / "generated_gpu_run"
    run.mkdir()
    data = tmp_path / "public_fixture_pilot.npz"
    # All records are generated here. The training pool concatenates four
    # generated 512-row blocks so disjoint 512-vs-512 controls are possible.
    np.savez_compressed(data,
                        X_train=np.vstack([generated_array(seed)
                                           for seed in (101, 102, 103, 104)]),
                        X_test=generated_array(105), categories=CATEGORIES)
    schema = tmp_path / "public_fixture_schema.json"
    schema.write_text(json.dumps({
        "features": FEATURES,
        "age_transform": "(AGE-50)/50",
        "public_fixture": True,
        "data_origin": "programmatically_generated_no_real_records",
        "category_domains": {
            name: list(range(int(size)))
            for name, size in zip(FEATURES[1:], CATEGORIES)
        },
    }))
    plan = {
        "format_version": 1,
        "job_id": "22692665",
        "training_replicates": 1,
        "target_repeats": 1,
        "generation_repeats": 3,
        "samples_per_arm": 512,
        "generation_seeds": SEEDS,
        "config": {"features": FEATURES, "steps": 2000},
        "dataset": {"train_rows": 2048, "holdout_rows": 512,
                    "categorical_domains": CATEGORIES.tolist()},
        "guidance": {"strength": 1.0, "max_log_ratio": 2.0,
                     "features": FEATURES[1:]},
        "input_and_original_artifact_sha256": {
            DATA_PLAN_KEY: DATA_HASH, SCHEMA_PLAN_KEY: SCHEMA_HASH,
        },
    }
    plan_path = run / "predeclared_plan.json"
    plan_path.write_text(json.dumps(plan))
    metrics = {
        "job_id": "22692665",
        "independent_training_replicates": 1,
        "marginal_releases": 1,
        "generation_repeats": 3,
        "samples_per_arm_per_draw": 512,
        "evaluations": {
            arm: [dict(generation_seed=seed,
                       **({"target_index": 1} if arm == "C" else {}))
                  for seed in SEEDS]
            for arm in ARMS
        },
    }
    (run / "metrics.json").write_text(json.dumps(metrics))
    # The helper checks the presence of these provenance artifacts. Their
    # contents are opaque here: no private target or checkpoint is fabricated.
    for name in ("COMPLETED.json", "training_metadata.json",
                 "dp_targets_01.json"):
        (run / name).write_text(json.dumps({"generated_unit_fixture": True}))
    sample_paths = {}
    for arm_index, arm in enumerate(ARMS):
        for draw in range(1, 4):
            path = run / f"{arm}_generation_{draw:02d}.npz"
            write_sample(path, values=generated_array(5000 + arm_index * 10 + draw))
            sample_paths[arm, draw] = path
    # Only the immutable historical-input gate is mocked. This is deliberately
    # not an option, environment variable, or production code path.
    gate = Mock(return_value={
        "status": "verified_exact_historical_inputs",
        "data_npz_sha256": DATA_HASH,
        "schema_sha256": SCHEMA_HASH,
        "training_shape": [2048, 7],
        "holdout_shape": [512, 7],
        "feature_count": 7,
        "categorical_cardinalities": CATEGORIES.tolist(),
        "unit_test_standin_only": True,
    })
    monkeypatch.setattr(prepare, "check_inputs", gate)
    return SimpleNamespace(run=run, data=data, schema=schema, plan=plan, metrics=metrics,
                           plan_path=plan_path, paths=sample_paths,
                           out=tmp_path / "manifest.json", gate=gate)


def make_manifest(fixture):
    return prepare.prepare_manifest(fixture.run, fixture.data, fixture.schema,
                                    fixture.out)


def test_exact_15_entries_form_six_real_diagnostic_matches(saved_run):
    before = {key: file_hash(path) for key, path in saved_run.paths.items()}
    payload = make_manifest(saved_run)
    saved = json.loads(saved_run.out.read_text(),
                       parse_constant=lambda value: pytest.fail(value))
    assert payload == saved
    entries = payload["samples"]
    assert len(entries) == 15
    assert Counter(entry["arm"] for entry in entries) == {arm: 3 for arm in ARMS}
    assert {entry["fit_id"] for entry in entries} == {"gpu22692665_trainingrep01"}
    assert len({entry["path"] for entry in entries}) == 15
    assert len({entry["sha256"] for entry in entries}) == 15
    for entry in entries:
        arm, draw = entry["arm"], entry["draw_id"]
        assert type(draw) is int and draw in (1, 2, 3)
        assert entry["generation_seed"] == SEEDS[draw - 1]
        assert entry["target_id"] == ("target01" if arm == "C" else None)
        assert entry["model_arm"] == ("B_private" if arm == "C" else arm)
        assert entry["key"] == "X_synthetic"
        assert entry["n_rows"] == 512
        assert Path(entry["path"]) == saved_run.paths[arm, draw].resolve()
        assert entry["sha256"] == before[arm, draw]
    saved_run.gate.assert_called_once()
    called_args = saved_run.gate.call_args.args
    assert Path(called_args[0]).resolve() == saved_run.data.resolve()
    assert Path(called_args[1]).resolve() == saved_run.schema.resolve()
    # Exercise the downstream consumer rather than merely reimplementing its
    # equality rules: shared block IDs must yield B/C and E/C for all draws.
    loaded = diagnostics.load_manifest(saved_run.out, CATEGORIES)
    matches, failures = diagnostics.match_records(loaded)
    assert failures == []
    assert len(matches) == 6
    assert Counter((baseline["arm"], candidate["draw_id"])
                   for baseline, candidate in matches) == Counter(
        (arm, draw) for arm in ("B_private", "E_matched") for draw in (1, 2, 3))
    assert all(candidate["arm"] == "C" and candidate["target_id"] == "target01"
               for _, candidate in matches)
    assert {key: file_hash(path) for key, path in saved_run.paths.items()} == before


def test_finite_out_of_domain_age_is_preserved_for_diagnosis(saved_run):
    values = generated_array(981)
    values[:4, 0] = [-4.25, 3.125, 0.123456789, -0.30000000000000004]
    path = saved_run.paths["C", 2]
    write_sample(path, values=values)
    before = path.read_bytes()
    make_manifest(saved_run)
    loaded = diagnostics.load_manifest(saved_run.out, CATEGORIES)
    record = next(x for x in loaded if x["arm"] == "C" and x["draw_id"] == 2)
    np.testing.assert_array_equal(record["values"], values)
    assert record["rows"] == 512
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", [
    "feature_order", "cardinality", "rows", "nonfinite",
])
def test_sample_contract_violations_fail_without_creating_manifest(saved_run, mutation):
    values = generated_array(732)
    features, categories = list(FEATURES), CATEGORIES.copy()
    if mutation == "feature_order":
        features[1:3] = features[2:0:-1]
    elif mutation == "cardinality":
        categories[0] += 1
    elif mutation == "rows":
        values = values[:-1]
    else:
        values[1, 0] = np.nan
        values[2, 3] = np.inf
    # The last expected sample is malformed, so validation must finish before
    # any manifest is published, even after fourteen valid inputs were read.
    write_sample(saved_run.paths["C", 3], values=values,
                 features=features, categories=categories)
    with pytest.raises(ValueError):
        make_manifest(saved_run)
    assert not saved_run.out.exists()


def test_incompatible_plan_or_matching_metrics_fail(saved_run):
    for field in ("training_replicates", "target_repeats", "generation_repeats"):
        altered = {**saved_run.plan, field: saved_run.plan[field] + 1}
        saved_run.plan_path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            make_manifest(saved_run)
        assert not saved_run.out.exists(), field
    saved_run.plan_path.write_text(json.dumps(saved_run.plan))
    # Authentic pairing also requires agreement with the writer's evaluation
    # metadata, rather than assigning identities from filenames alone.
    for mutation in ("job", "release_count", "seed_order", "target_index"):
        altered = json.loads(json.dumps(saved_run.metrics))
        if mutation == "job":
            altered["job_id"] = "99999999"
        elif mutation == "release_count":
            altered["marginal_releases"] = 2
        elif mutation == "seed_order":
            altered["evaluations"]["E_matched"][0]["generation_seed"] = SEEDS[1]
        else:
            altered["evaluations"]["C"][1]["target_index"] = 2
        (saved_run.run / "metrics.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            make_manifest(saved_run)
        assert not saved_run.out.exists(), mutation


def test_recorded_reference_and_sample_hash_mismatches_fail(saved_run):
    for key in (DATA_PLAN_KEY, SCHEMA_PLAN_KEY):
        altered = json.loads(json.dumps(saved_run.plan))
        altered["input_and_original_artifact_sha256"][key] = "0" * 64
        saved_run.plan_path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            make_manifest(saved_run)
        assert not saved_run.out.exists(), key
    saved_run.plan_path.write_text(json.dumps(saved_run.plan))
    (saved_run.run / "metrics.json").write_text(json.dumps({**saved_run.metrics,
        "sample_sha256": {"C_generation_03.npz": "0" * 64},
    }))
    with pytest.raises(ValueError, match="Recorded sample hash mismatch"):
        make_manifest(saved_run)
    assert not saved_run.out.exists()


def test_original_input_gate_failure_prevents_manifest(saved_run):
    saved_run.gate.side_effect = ValueError("Historical input hash mismatch")
    with pytest.raises(ValueError, match="Historical input hash mismatch"):
        make_manifest(saved_run)
    saved_run.gate.assert_called_once()
    assert not saved_run.out.exists()


def test_missing_expected_sample_is_not_substituted_by_a_glob(saved_run):
    missing = saved_run.paths["E_matched", 2]
    missing.rename(saved_run.run / "E_matched_generation_02_backup.npz")
    with pytest.raises((FileNotFoundError, ValueError)):
        make_manifest(saved_run)
    assert not saved_run.out.exists()


def test_existing_manifest_is_never_overwritten(saved_run):
    sentinel = b"Existing manifest belongs to another completed run.\n"
    saved_run.out.write_bytes(sentinel)
    with pytest.raises(FileExistsError):
        make_manifest(saved_run)
    assert saved_run.out.read_bytes() == sentinel


def test_generated_manifest_runs_existing_diagnostics_end_to_end(saved_run):
    make_manifest(saved_run)
    args = SimpleNamespace(
        data_npz=saved_run.data, schema=saved_run.schema,
        samples_manifest=saved_run.out, out=saved_run.out.parent / "diagnostics",
        seed=20261112, real_real_repeats=1, sample_rows=512, shuffle_repeats=1,
    )
    # Reduced controls are specific to this software test. The production
    # launcher retains twenty real-real repeats and ten shuffles per sample.
    report = diagnostics.run(args)
    assert report["status"] == "complete"
    assert len(report["samples"]) == 15
    assert len(report["matched_comparisons"]) == 6
    assert report["unmatched_comparisons"] == []
    assert report["training_performed"] is False
    assert report["new_marginal_releases"] == 0
    assert report["frozen_model_intervention_performed"] is False
    manifest = json.loads(saved_run.out.read_text())
    assert {x["path"]: x["sha256"] for x in report["samples"]} == {
        x["path"]: x["sha256"] for x in manifest["samples"]}
    assert all(x["rows"] == 512 for x in report["samples"])
    assert report["declared_public_fixture"] is True
    assert not list(args.out.glob("*.npy"))
    assert not list(args.out.glob("*.npz"))
