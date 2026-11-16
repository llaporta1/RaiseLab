"""Exercise the actual Slurm postcheck/tar block on real generated-fixture reports.

This runs no Slurm command and fits no model. Only the original-input gates are
mocked for the declared public fixtures; report producers and packaging are real.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from test_cached_target_audit import CATEGORIES, generated_case

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import audit_prototype2_cached_targets as target_audit
import prepare_prototype2_saved_gpu_manifest as gpu_manifest
import run_prototype2_age_permutation as age_permutation


EXPECTED_MEMBERS = {
    "completion_summary.json", "gpu_samples_manifest.json",
    "target_audit/target_audit.json", "target_audit/target_audit.txt",
    "age_permutation/age_permutation.json", "age_permutation/metrics_per_draw.csv",
    "age_permutation/real_real_controls.csv", "age_permutation/README.txt",
}


@pytest.fixture
def real_reports(generated_case, monkeypatch):
    case = generated_case
    out = case.out.with_name("saved_followup_999001")
    out.mkdir()
    # Nonconstant AGE makes the actual permutation pipeline do useful work.
    case.train[:, 0] = ((np.arange(2048) % 48) + 18 - 50) / 50
    case.holdout[:, 0] = ((np.arange(512) % 48) + 18 - 50) / 50
    np.savez(case.data, X_train=case.train, X_test=case.holdout, categories=CATEGORIES)
    seeds = [12026, 12126, 12226]
    case.plan.update(job_id="22692665", training_replicates=1,
                     generation_repeats=3, samples_per_arm=512,
                     generation_seeds=seeds,
                     config={"features": target_audit.FEATURES, "steps": 2000},
                     dataset={"train_rows": 2048, "holdout_rows": 512,
                              "categorical_domains": CATEGORIES.tolist()})
    case.plan_path.write_text(json.dumps(case.plan))
    metrics = {"job_id": "22692665", "independent_training_replicates": 1,
               "marginal_releases": 1, "generation_repeats": 3,
               "samples_per_arm_per_draw": 512,
               "evaluations": {arm: [{"generation_seed": seed,
                                      **({"target_index": 1} if arm == "C" else {})}
                                     for seed in seeds] for arm in gpu_manifest.ARMS}}
    (case.run / "metrics.json").write_text(json.dumps(metrics))
    for name in ("COMPLETED.json", "training_metadata.json"):
        (case.run / name).write_text('{"generated_unit_fixture":true}')
    for arm_index, arm in enumerate(gpu_manifest.ARMS):
        for draw in (1, 2, 3):
            values = case.train[(np.arange(512) + 73 * arm_index + 13 * draw) % 2048].copy()
            values[:, 0] += arm_index * 0.0123
            np.savez(case.run / f"{arm}_generation_{draw:02d}.npz",
                     X_synthetic=values, categories=CATEGORIES, features=target_audit.FEATURES)
    monkeypatch.setattr(gpu_manifest, "check_inputs", case.gate)
    gpu_manifest.prepare_manifest(case.run, case.data, case.schema_path,
                                  out / "gpu_samples_manifest.json")
    target_audit.audit_cached_targets(case.run, case.data, case.schema_path, out / "target_audit")
    pgm_dir = out.parent / "generated_pgm_saved_arrays"
    pgm_dir.mkdir()
    entries = []
    for fit in (1, 2, 3):
        for arm_index, arm in enumerate(("aim", "mst")):
            for draw in (1, 2, 3):
                path = pgm_dir / f"{arm}_fit{fit:03d}_draw{draw:03d}.npy"
                values = case.train[(np.arange(512) + 83 * fit + 19 * arm_index + draw) % 2048]
                np.save(path, values)
                entries.append({"arm": arm, "fit_id": f"fit{fit:03d}", "target_id": None,
                                "draw_id": f"draw{draw:03d}", "path": str(path.resolve()),
                                "sha256": target_audit.sha256_file(path), "public_fixture": True})
    pgm_path = pgm_dir / "sample_manifest.json"
    pgm_path.write_text(json.dumps({"public_fixture": True, "samples": entries}))
    report = age_permutation.run(SimpleNamespace(
        data_npz=case.data, schema=case.schema_path,
        samples_manifest=[out / "gpu_samples_manifest.json", pgm_path],
        out=out / "age_permutation", seed=20261112, sample_rows=512,
        permutation_repeats=20, real_real_repeats=20, public_fixture=True))
    assert report["sample_count"] == 33
    # Extra files deliberately sit beside reports. The whitelist must exclude
    # them even when they have tempting names or live in the output directory.
    np.save(out / "should_not_be_archived.npy", np.array([1]))
    (out / "dp_targets_01.json").write_bytes(case.target_path.read_bytes())
    (out / "should_not_be_archived.pt").write_bytes(b"generated fixture sentinel")
    return out


def execute_actual_packaging_block(out):
    launcher = SCRIPTS.parent / "slurm/prototype2_saved_followup.sbatch"
    text = launcher.read_text()
    begin = "# BEGIN FOLLOWUP POSTCHECKS AND PACKAGE"
    end = "# END FOLLOWUP POSTCHECKS AND PACKAGE"
    assert text.count(begin) == text.count(end) == 1
    block = text.split(begin, 1)[1].split(end, 1)[0]
    environment = {**os.environ, "FOLLOWUP_PYTHON": sys.executable,
                   "RUN_OUT": str(out), "SLURM_JOB_ID": "999001"}
    return subprocess.run(["bash", "-c", "set -euo pipefail\nset -o noclobber\n" + block],
                          env=environment, text=True, capture_output=True, check=False)


def test_actual_postchecks_package_exact_eight_real_report_files(real_reports):
    completed = execute_actual_packaging_block(real_reports)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((real_reports / "completion_summary.json").read_text())
    assert summary["status"] == "complete" and all(summary["checks"].values())
    assert (real_reports / "COMPLETED.txt").is_file()
    with tarfile.open(real_reports / "aggregate_reports.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == EXPECTED_MEMBERS
        assert len(archive.getmembers()) == 8
        for member in archive.getmembers():
            assert member.isfile()
            assert archive.extractfile(member).read() == (real_reports / member.name).read_bytes()
            assert not member.name.endswith((".npy", ".npz", ".pt"))
            assert Path(member.name).name != "dp_targets_01.json"


def test_missing_actual_report_filename_prevents_completion(real_reports):
    # This specifically catches the prior class of bug where tar expected a
    # filename that the real report producer never emitted.
    (real_reports / "age_permutation/README.txt").rename(real_reports / "age_permutation/README.renamed")
    completed = execute_actual_packaging_block(real_reports)
    assert completed.returncode != 0
    assert "README.txt" in completed.stderr
    assert not (real_reports / "COMPLETED.txt").exists()


def test_changed_target_hash_prevents_packaging(real_reports):
    path = real_reports / "target_audit/target_audit.json"
    report = json.loads(path.read_text())
    report["provenance"]["files"]["cached_target"]["sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    completed = execute_actual_packaging_block(real_reports)
    assert completed.returncode != 0
    assert "cached_target_hash_matches_preflight" in completed.stderr
    assert not (real_reports / "aggregate_reports.tar.gz").exists()
    assert not (real_reports / "COMPLETED.txt").exists()
