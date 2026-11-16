"""Cached-target audits use generated public arrays and an explicitly mocked gate.

The production command has no fixture mode or historical-hash override. These
tests never import training code, draw DP noise, or use actual IPUMS records.
"""

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
import audit_prototype2_cached_targets as audit
from check_prototype2_rivanna_inputs import EXPECTED_DATA_SHA256, EXPECTED_SCHEMA_SHA256


CATEGORIES = np.array([3, 9, 6, 44, 531, 266], dtype=np.int64)


@pytest.fixture
def generated_case(tmp_path, monkeypatch):
    run = tmp_path / "generated_run"
    run.mkdir()
    data, schema_path = tmp_path / "fixture.npz", tmp_path / "schema.json"
    train, holdout = np.zeros((2048, 7)), np.zeros((512, 7))
    for col, size in enumerate(CATEGORIES, 1):
        train[:, col] = np.arange(len(train)) % size
        holdout[:, col] = (np.arange(len(holdout)) + 2) % size
    np.savez(data, X_train=train, X_test=holdout, categories=CATEGORIES)
    domains = {name: list(range(int(size))) for name, size in zip(audit.TARGETS, CATEGORIES)}
    domains["RACE"], domains["HISPAN"] = list(range(1, 10)), [0, 1, 2, 3, 4, 9]
    schema = {"features": audit.FEATURES, "age_transform": "(AGE-50)/50",
              "category_domains": domains, "public_fixture": True,
              "data_origin": "programmatically_generated_no_real_records"}
    schema_path.write_text(json.dumps(schema))
    probabilities = {}
    for col, (name, size) in enumerate(zip(audit.TARGETS, CATEGORIES), 1):
        counts = np.bincount(train[:, col].astype(int), minlength=int(size))
        vector = (counts + 0.001) / (len(train) + size * 0.001)
        if name == "RACE":
            vector[0] -= 0.01
            vector[6] += 0.01
        probabilities[name] = vector.tolist()
    target = {"format_version": 1, "mechanism": audit.MECHANISM,
              "features": audit.TARGETS, "category_domains": domains,
              "epsilon_marginals": 1.0, "delta_marginals": 0.0,
              "epsilon_per_histogram": 1 / 6, "l1_sensitivity_per_histogram": 1.0,
              "laplace_scale_per_cell": 6.0, "count_pseudocount": 0.001,
              "postprocessing": audit.POSTPROCESSING, "probabilities": probabilities}
    target_path = run / "dp_targets_01.json"
    target_path.write_text(json.dumps(target))
    plan = {"job_id": "generated_unit_test", "target_repeats": 1,
            "privacy_accounting": {"epsilon_marginals_one_cached_release": 1.0},
            "original_source_sha256": {"scripts/run_ipums_cd_pilot.py": audit.TARGET_SOURCE_SHA256},
            "input_and_original_artifact_sha256": {
                "pipeline_runs/prototype2_cpu/ipums_2048/pilot_data.npz": EXPECTED_DATA_SHA256,
                "pipeline_runs/prototype2_cpu/ipums_2048/schema.json": EXPECTED_SCHEMA_SHA256}}
    plan_path = run / "predeclared_plan.json"
    plan_path.write_text(json.dumps(plan))
    gate = Mock(return_value={"status": "verified_exact_historical_inputs",
                              "data_npz_sha256": EXPECTED_DATA_SHA256,
                              "schema_sha256": EXPECTED_SCHEMA_SHA256,
                              "unit_test_standin_only": True})
    monkeypatch.setattr(audit, "check_inputs", gate)
    return SimpleNamespace(run=run, data=data, schema_path=schema_path, schema=schema,
                           train=train, holdout=holdout, target=target, target_path=target_path,
                           plan=plan, plan_path=plan_path, gate=gate, out=tmp_path / "audit")


def run_case(case, out=None):
    return audit.audit_cached_targets(case.run, case.data, case.schema_path, out or case.out)


def test_exact_training_reference_and_public_code_mapping(generated_case):
    case = generated_case
    paths = [case.data, case.schema_path, case.target_path, case.plan_path]
    before = {p: p.read_bytes() for p in paths}
    report = run_case(case)
    assert report["status"] == "complete"
    assert report["training_performed"] is False
    assert report["new_marginal_releases"] == report["fresh_target_draws"] == 0
    assert report["formal_end_to_end_dp_claimed"] is False
    assert set(p.name for p in case.out.iterdir()) == {"target_audit.json", "target_audit.txt"}
    assert json.loads((case.out / "target_audit.json").read_text()) == report
    assert report["provenance"]["files"]["cached_target"]["sha256"] == hashlib.sha256(before[case.target_path]).hexdigest()
    assert report["training_rows"] == 2048 and report["holdout_rows"] == 512
    race = report["targeted_entries"]["RACE"]
    assert [entry["public_code"] for entry in race] == list(range(1, 10))
    assert race[6]["public_code"] == 7 and race[6]["encoded_index"] == 6
    assert race[6]["training_count"] == np.count_nonzero(case.train[:, 2] == 6)
    truth = np.bincount(case.train[:, 2].astype(int), minlength=9) / 2048
    expected = np.abs(np.asarray(case.target["probabilities"]["RACE"]) - truth).sum()
    assert report["features"]["RACE"]["target_vs_training_l1"] == expected
    assert report["features"]["RACE"]["target_vs_smoothed_training_oracle_l1"] == pytest.approx(0.02)
    assert report["features"]["HISPAN"]["target_vs_smoothed_training_oracle_l1"] == 0
    assert report["targeted_entries"]["HISPAN"][-1]["public_code"] == 9
    assert {p: p.read_bytes() for p in paths} == before
    case.gate.assert_called_once_with(case.data, case.schema_path)


def test_holdout_changes_do_not_change_target_vs_training_error(generated_case):
    case = generated_case
    first = run_case(case)
    altered = case.holdout.copy()
    altered[:, 1:] = 0
    np.savez(case.data, X_train=case.train, X_test=altered, categories=CATEGORIES)
    second = run_case(case, case.out.with_name("audit_other_holdout"))
    for name in audit.TARGETS:
        assert first["features"][name]["target_vs_training_l1"] == second["features"][name]["target_vs_training_l1"]
        assert first["features"][name]["target_vs_smoothed_training_oracle_l1"] == second["features"][name]["target_vs_smoothed_training_oracle_l1"]
    assert first["features"]["RACE"]["training_vs_holdout_l1"] != second["features"]["RACE"]["training_vs_holdout_l1"]


@pytest.mark.parametrize("mutation", ["domain_order", "vector_length", "nonfinite", "zero", "not_normalized"])
def test_invalid_target_contract_fails_before_creating_outputs(generated_case, mutation):
    case = generated_case
    target = json.loads(json.dumps(case.target))
    if mutation == "domain_order":
        target["category_domains"]["RACE"][0:2] = [2, 1]
    elif mutation == "vector_length":
        target["probabilities"]["RACE"].pop()
    elif mutation == "nonfinite":
        target["probabilities"]["RACE"][0] = float("nan")
    elif mutation == "zero":
        target["probabilities"]["RACE"][0] = 0
    else:
        target["probabilities"]["RACE"][0] += 0.1
    case.target_path.write_text(json.dumps(target))
    with pytest.raises(ValueError):
        run_case(case)
    assert not case.out.exists()


def test_wrong_source_contract_and_input_gate_are_rejected(generated_case):
    case = generated_case
    case.gate.side_effect = ValueError("Historical input hash mismatch")
    with pytest.raises(ValueError, match="Historical input hash mismatch"):
        run_case(case)
    assert not case.out.exists()
    case.gate.side_effect = None
    case.plan["original_source_sha256"]["scripts/run_ipums_cd_pilot.py"] = "0" * 64
    case.plan_path.write_text(json.dumps(case.plan))
    with pytest.raises(ValueError, match="verified cached-target writer"):
        run_case(case)
    assert not case.out.exists()


def test_existing_output_is_not_overwritten(generated_case):
    case = generated_case
    case.out.mkdir()
    sentinel = case.out / "sentinel.txt"
    sentinel.write_text("preserve existing run")
    with pytest.raises(FileExistsError):
        run_case(case)
    assert sentinel.read_text() == "preserve existing run"
    case.gate.assert_not_called()
