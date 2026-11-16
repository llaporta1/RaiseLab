"""Exercise the real launcher package gate with generated local software fixtures.

The sampling orchestrator uses the public FakeBackend; diagnostic reports are
explicit contract stubs. No Slurm submission, CUDA, original checkpoint, private
record, production preflight bypass, or actual guidance result is involved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from test_guidance_sweep import (
    CATEGORIES, SEEDS, FakeBackend, base_array, fixture_reference,
    historical_offsets, sweep,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/prototype2_guidance_sweep.sbatch"
INPUT_HASHES = {
    "data_npz": "677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906",
    "schema": "3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73",
}
GENERATION_HASHED = (
    "preflight.json", "plan.json", "replay.json", "reference.json",
    "results.json", "summary.txt", "sample_manifest.json",
)
EXPECTED_MEMBERS = {
    "completion_summary.json",
    "generation/COMPLETED.json", "generation/preflight.json", "generation/plan.json",
    "generation/replay.json", "generation/results.json", "generation/summary.txt",
    "generation/sample_manifest.json",
    "diagnostics/diagnostics.json", "diagnostics/metrics_per_draw.csv",
    "diagnostics/age_bin_contributions.csv", "diagnostics/pair_cell_errors.csv",
    "diagnostics/matched_cell_changes.csv", "diagnostics/real_real_controls.csv",
    "diagnostics/README.txt",
    "age_permutation/age_permutation.json", "age_permutation/metrics_per_draw.csv",
    "age_permutation/real_real_controls.csv", "age_permutation/README.txt",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generated_reports(out, mode="sweep", include_oracle=False):
    """Build valid reports/arrays without invoking the historical file gate."""
    generation = out / "generation"
    generation.mkdir(parents=True)
    backend = FakeBackend()
    targets = [np.arange(1, int(k) + 1, dtype=np.float64) for k in CATEGORIES]
    targets = [values / values.sum() for values in targets]
    reference = fixture_reference(base_array(13026, 4096))
    offsets = historical_offsets(targets, reference)
    historical = {(arm, draw): backend.render(seed, 512, None if arm == "B_private" else offsets)
                  for arm in ("B_private", "C") for draw, seed in enumerate(SEEDS, 1)}
    report = sweep.run_sampling(
        backend, historical, targets, generation, mode=mode,
        oracle_targets=[values[::-1].copy() for values in targets] if include_oracle else None,
    )
    for name in ("preflight.json", "plan.json"):
        write_json(generation / name, {"generated_packaging_fixture": True})
    write_json(generation / "results.json", report)
    (generation / "summary.txt").write_text(sweep._summary(report))
    complete = {
        "status": "complete", "mode": mode, "replay_passed": True,
        "condition_count": report["condition_count"], "sample_count": report["sample_count"],
        "training_performed": False, "new_training_fits": 0, "new_marginal_releases": 0,
        "inputs_unchanged": True, "model_unchanged": True,
        "artifact_sha256": {name: sweep.file_hash(generation / name) for name in GENERATION_HASHED},
    }
    write_json(generation / "COMPLETED.json", complete)
    manifest = generation / "sample_manifest.json"
    digest = sweep.file_hash(manifest)
    samples = [{**entry, "rows": 512, "metrics": {"shuffle": {"repeats": 20}}}
               for entry in report["samples"]]
    common = {
        "generated_packaging_fixture": True, "status": "complete",
        "training_performed": False, "new_marginal_releases": 0, "samples": samples,
        "provenance": {"seed": 20261112, "files": {
            name: {"sha256": value} for name, value in INPUT_HASHES.items()}},
        "real_real": {"repeats": 20, "rows_per_side": 512,
                      "within_repeat_disjoint": True, "across_repeat_independent": False,
                      "results": [{"generated_packaging_fixture": True}] * 20},
    }
    diagnostics = json.loads(json.dumps(common))
    diagnostics["provenance"]["files"]["samples_manifest"] = {"sha256": digest}
    age = json.loads(json.dumps(common))
    age.update(sample_count=len(samples), permutation_type="AGE_only", permutation_repeats=20)
    age["provenance"]["sample_manifests"] = [{"path": str(manifest), "sha256": digest}]
    for directory, name, value in (("diagnostics", "diagnostics.json", diagnostics),
                                   ("age_permutation", "age_permutation.json", age)):
        (out / directory).mkdir()
        write_json(out / directory / name, value)
    for member in EXPECTED_MEMBERS - {"completion_summary.json"}:
        path = out / member
        if not path.exists():
            path.write_text("generated_packaging_fixture_only\n")
    # Deliberate excluded content beside approved reports catches glob/tar drift.
    (out / "private_checkpoint.pt").write_bytes(b"PUBLIC TEST SENTINEL")
    write_json(generation / "dp_targets_01.json", {"probabilities": [0.2, 0.8]})
    return out


def execute_actual_package_block(out, mode="sweep", include_oracle=False):
    text = LAUNCHER.read_text()
    begin, end = "# BEGIN GUIDANCE POSTCHECKS AND PACKAGE", "# END GUIDANCE POSTCHECKS AND PACKAGE"
    assert text.count(begin) == text.count(end) == 1
    block = text.split(begin, 1)[1].split(end, 1)[0]
    prelude = (
        "set -euo pipefail\nset -o noclobber\numask 077\n"
        "die() { printf 'ERROR: %s\\n' \"$*\" >&2; exit 2; }\n"
        + ("ORACLE_FLAGS=(--include-train-oracle)\n" if include_oracle else "ORACLE_FLAGS=()\n")
    )
    environment = {**os.environ, "GUIDANCE_PYTHON": sys.executable, "RUN_OUT": str(out),
                   "GUIDANCE_MODE": mode, "SLURM_JOB_ID": "999001"}
    return subprocess.run(["bash", "-c", prelude + block], env=environment,
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("mode,oracle,conditions,samples", [
    ("replay", False, 2, 6), ("sweep", False, 10, 30), ("sweep", True, 19, 57),
])
def test_actual_package_accepts_each_contract_and_excludes_local_vectors(tmp_path, mode, oracle, conditions, samples):
    out = generated_reports(tmp_path / "output", mode, oracle)
    completed = execute_actual_package_block(out, mode, oracle)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((out / "completion_summary.json").read_text())
    assert all(summary["checks"].values())
    assert (summary["condition_count"], summary["sample_count"]) == (conditions, samples)
    assert summary["include_train_oracle"] is oracle
    assert (out / "COMPLETED.txt").is_file()
    assert (out / "generation/reference.json").is_file()
    with tarfile.open(out / "aggregate_reports.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == EXPECTED_MEMBERS
        assert len(archive.getmembers()) == 19
        for member in archive.getmembers():
            assert member.isfile()
            assert archive.extractfile(member).read() == (out / member.name).read_bytes()
            assert not member.name.endswith((".npy", ".npz", ".pt"))
            assert Path(member.name).name not in {"reference.json", "dp_targets_01.json"}


@pytest.mark.parametrize("missing", ["generation/reference.json", "age_permutation/README.txt"])
def test_missing_local_or_packaged_artifact_prevents_completion(tmp_path, missing):
    out = generated_reports(tmp_path / "output", "replay")
    (out / missing).unlink()
    completed = execute_actual_package_block(out, "replay")
    assert completed.returncode != 0
    assert Path(missing).name in completed.stderr
    assert not (out / "COMPLETED.txt").exists()
    assert not (out / "aggregate_reports.tar.gz").exists()


@pytest.mark.parametrize("changed", ["reference", "sample"])
def test_artifact_or_sample_hash_mismatch_prevents_packaging(tmp_path, changed):
    out = generated_reports(tmp_path / "output", "replay")
    path = out / "generation/reference.json" if changed == "reference" else next((out / "generation/samples").glob("*.npz"))
    with path.open("ab") as handle:
        handle.write(b"changed-generated-fixture")
    completed = execute_actual_package_block(out, "replay")
    assert completed.returncode != 0
    assert "hash" in completed.stderr
    assert not (out / "COMPLETED.txt").exists()
    assert not (out / "aggregate_reports.tar.gz").exists()


def test_failed_replay_is_rejected_even_with_refreshed_artifact_hash(tmp_path):
    out = generated_reports(tmp_path / "output", "replay")
    path = out / "generation/replay.json"
    replay = json.loads(path.read_text())
    replay["draws"][2]["c_matches_historical"] = False
    write_json(path, replay)
    marker_path = out / "generation/COMPLETED.json"
    marker = json.loads(marker_path.read_text())
    marker["artifact_sha256"]["replay.json"] = sweep.file_hash(path)
    write_json(marker_path, marker)
    completed = execute_actual_package_block(out, "replay")
    assert completed.returncode != 0
    assert "all_three_B_C_and_zero_guidance_replays_pass" in completed.stderr
    assert not (out / "COMPLETED.txt").exists()
    assert not (out / "aggregate_reports.tar.gz").exists()


def test_successful_package_cannot_be_overwritten(tmp_path):
    out = generated_reports(tmp_path / "output", "replay")
    first = execute_actual_package_block(out, "replay")
    assert first.returncode == 0, first.stderr
    before = {name: (out / name).read_bytes() for name in (
        "completion_summary.json", "aggregate_reports.tar.gz", "COMPLETED.txt")}
    second = execute_actual_package_block(out, "replay")
    assert second.returncode != 0
    assert all((out / name).read_bytes() == value for name, value in before.items())


def test_full_launcher_rejects_login_execution_without_creating_outputs(tmp_path):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    completed = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=environment,
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "Slurm allocation" in completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_full_launcher_stops_after_generator_failure_without_running_analyses(tmp_path):
    # Local shell control-flow test only: these synthetic variables never reach
    # Slurm or the real GPU runner. The fake interpreter exits on its first call.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_prototype2_guidance_sweep.py", "prototype2_gpu_replay_backend.py",
                 "run_prototype2_diagnostics.py", "run_prototype2_age_permutation.py",
                 "prototype2_evaluation.py", "prototype2_relationship_checks.py",
                 "check_prototype2_rivanna_inputs.py"):
        (scripts / name).write_text("# public launcher control-flow fixture\n")
    for name in ("original_project", "original_run"):
        (tmp_path / name).mkdir()
    for name in ("input.npz", "schema.json"):
        (tmp_path / name).write_text("public fixture; never read by production code")
    interpreter = tmp_path / "fake-interpreter"
    interpreter.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$FIXTURE_CALL_LOG\"\nexit 23\n")
    interpreter.chmod(0o700)
    call_log = tmp_path / "calls.txt"
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    environment.update(SLURM_JOB_ID="999001", SLURM_CPUS_PER_TASK="1",
                       SLURM_SUBMIT_DIR=str(tmp_path), RAISE_GUIDANCE_PYTHON=str(interpreter),
                       FIXTURE_CALL_LOG=str(call_log))
    completed = subprocess.run([
        "bash", str(LAUNCHER), "sweep", str(tmp_path / "original_project"),
        str(tmp_path / "original_run"), str(tmp_path / "input.npz"), str(tmp_path / "schema.json"),
    ], env=environment, text=True, capture_output=True, check=False)
    assert completed.returncode == 23
    calls = call_log.read_text().splitlines()
    assert len(calls) == 1 and "run_prototype2_guidance_sweep.py" in calls[0]
    out = tmp_path / "pipeline_runs/guidance_sweep_999001"
    assert (out / "FAILED.txt").is_file()
    assert not (out / "COMPLETED.txt").exists()
    assert not (out / "diagnostics").exists()
    assert not (out / "age_permutation").exists()
