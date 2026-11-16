#!/bin/bash
#SBATCH --job-name=prototype2-guidance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=pipeline_runs/slurm/%x-%j.out

# Submit from this repository root after creating pipeline_runs/slurm.
# Usage: MODE ORIGINAL_PROJECT ORIGINAL_GPU_RUN DATA_NPZ SCHEMA_JSON
#        [--include-train-oracle]
# MODE is replay or sweep. The documented default study is sweep, which first
# passes all historical replay gates. This launcher uses the original GPU
# environment and samples the frozen B model. It does not train a new model or
# draw a new private target. All outputs remain nonprivate local research.

set -euo pipefail
set -o noclobber
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die "Submit with sbatch from a login shell; this script requires a Slurm allocation."
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "Submit one job, not a job array."
[[ "${SLURM_CPUS_PER_TASK:-}" == 1 ]] || die "The historical GPU runtime requires exactly one requested CPU."
[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || die "SLURM_SUBMIT_DIR is missing. Submit from the repository root."
[[ $# -eq 5 || $# -eq 6 ]] || die "Usage: replay|sweep ORIGINAL_PROJECT ORIGINAL_GPU_RUN DATA_NPZ SCHEMA_JSON [--include-train-oracle]"

GUIDANCE_MODE="$1"
[[ "$GUIDANCE_MODE" == replay || "$GUIDANCE_MODE" == sweep ]] || die "MODE must be replay or sweep."
ORACLE_FLAGS=()
if [[ $# -eq 6 ]]; then
    [[ "$6" == --include-train-oracle ]] || die "The only optional argument is --include-train-oracle."
    [[ "$GUIDANCE_MODE" == sweep ]] || die "The TRAIN-oracle comparison is available in sweep mode."
    ORACLE_FLAGS=(--include-train-oracle)
fi

cd -- "$SLURM_SUBMIT_DIR"
REPO_ROOT="$(pwd -P)"
for REQUIRED_SOURCE in scripts/run_prototype2_guidance_sweep.py \
    scripts/prototype2_gpu_replay_backend.py \
    scripts/prototype2_gpu_replay_backend.py \
    scripts/run_prototype2_diagnostics.py scripts/run_prototype2_age_permutation.py \
    scripts/prototype2_evaluation.py scripts/prototype2_relationship_checks.py \
    scripts/check_prototype2_rivanna_inputs.py; do
    [[ -r "$REQUIRED_SOURCE" ]] || die "Missing $REQUIRED_SOURCE; submit from the prepared repository root."
done
[[ -d "$2" && -d "$3" ]] || die "The original project and saved GPU run must be existing directories."
[[ -r "$4" && -r "$5" ]] || die "The original data NPZ and schema must be readable."

# The verified historical submitter invoked this interpreter directly. No
# Miniforge module or .venv-pgm substitution belongs in the GPU replay path.
GUIDANCE_PYTHON="${RAISE_GUIDANCE_PYTHON:-$HOME/.conda/envs/tabdiff-p2-gpu/bin/python}"
[[ -x "$GUIDANCE_PYTHON" ]] || die "The original GPU Python is unavailable: $GUIDANCE_PYTHON"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
unset PYTHONPATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN_OUT="$REPO_ROOT/pipeline_runs/guidance_${GUIDANCE_MODE}_${SLURM_JOB_ID}"
mkdir -p -- "$REPO_ROOT/pipeline_runs"
mkdir -- "$RUN_OUT" || die "Output already exists; preserve and inspect it: $RUN_OUT"
printf '*\n' > "$RUN_OUT/.gitignore"
on_exit() {
    EXIT_STATUS=$?
    if [[ "$EXIT_STATUS" -ne 0 ]]; then
        printf 'FAILED: frozen-B %s job=%s exit=%s\nPreserve all outputs. No automatic retry or resume.\n' \
            "$GUIDANCE_MODE" "$SLURM_JOB_ID" "$EXIT_STATUS" > "$RUN_OUT/FAILED.txt"
    fi
}
trap on_exit EXIT

printf 'Frozen-B %s job=%s; output=%s\n' "$GUIDANCE_MODE" "$SLURM_JOB_ID" "$RUN_OUT"
printf 'GPU interpreter: %s\n' "$GUIDANCE_PYTHON"
"$GUIDANCE_PYTHON" -u -B scripts/run_prototype2_guidance_sweep.py \
    --project-root "$2" --run-dir "$3" --data-npz "$4" --schema "$5" \
    --out "$RUN_OUT/generation" --mode "$GUIDANCE_MODE" \
    --acknowledge-trusted-checkpoint "${ORACLE_FLAGS[@]}"
[[ -r "$RUN_OUT/generation/COMPLETED.json" ]] || die "Generation returned without its completion marker."

"$GUIDANCE_PYTHON" -u -B scripts/run_prototype2_diagnostics.py \
    --data-npz "$4" --schema "$5" \
    --samples-manifest "$RUN_OUT/generation/sample_manifest.json" \
    --sample-rows 512 --shuffle-repeats 20 --real-real-repeats 20 \
    --seed 20261112 --out "$RUN_OUT/diagnostics"
"$GUIDANCE_PYTHON" -u -B scripts/run_prototype2_age_permutation.py \
    --data-npz "$4" --schema "$5" \
    --samples-manifest "$RUN_OUT/generation/sample_manifest.json" \
    --sample-rows 512 --permutation-repeats 20 --real-real-repeats 20 \
    --seed 20261112 --out "$RUN_OUT/age_permutation"

# BEGIN GUIDANCE POSTCHECKS AND PACKAGE
"$GUIDANCE_PYTHON" -B - "$RUN_OUT" "$GUIDANCE_MODE" "${#ORACLE_FLAGS[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

out = Path(sys.argv[1]).resolve(strict=True)
mode = sys.argv[2]
include_oracle = sys.argv[3] == "1"
generation = out / "generation"
expected_conditions = 2 if mode == "replay" else (19 if include_oracle else 10)
expected_samples = 3 * expected_conditions
expected_input_hashes = {
    "data_npz": "677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906",
    "schema": "3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73",
}

def read_json(path):
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path.name}")
    return value

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def integer_equals(value, expected):
    return type(value) is int and value == expected

complete = read_json(generation / "COMPLETED.json")
replay = read_json(generation / "replay.json")
results = read_json(generation / "results.json")
manifest_path = generation / "sample_manifest.json"
manifest = read_json(manifest_path)
diagnostics = read_json(out / "diagnostics/diagnostics.json")
age = read_json(out / "age_permutation/age_permutation.json")
samples = manifest.get("samples", [])
checks = {
    "generation_complete": complete.get("status") == "complete",
    "mode_matches": complete.get("mode") == results.get("mode") == mode,
    "condition_count_matches": integer_equals(complete.get("condition_count"), expected_conditions)
        and integer_equals(results.get("condition_count"), expected_conditions)
        and len(results.get("conditions", [])) == expected_conditions,
    "sample_count_matches": integer_equals(complete.get("sample_count"), expected_samples)
        and integer_equals(results.get("sample_count"), expected_samples)
        and isinstance(samples, list) and len(samples) == expected_samples,
    "generation_results_complete": results.get("status") == "complete",
    "manifest_matches_results": results.get("samples") == samples,
    "replay_passed": complete.get("replay_passed") is True and replay.get("status") == "passed",
    "reference_recipe_matches": integer_equals(replay.get("reference_rows"), 4096)
        and integer_equals(replay.get("reference_seed"), 13026),
    "inputs_and_model_unchanged": complete.get("inputs_unchanged") is True
        and complete.get("model_unchanged") is True and results.get("model_unchanged") is True,
    "no_new_training_or_private_release": all(
        report.get("training_performed") is False
        and integer_equals(report.get("new_training_fits"), 0)
        and integer_equals(report.get("new_marginal_releases"), 0)
        for report in (complete, results)),
    "diagnostics_complete": diagnostics.get("status") == "complete",
    "age_analysis_complete": age.get("status") == "complete"
        and integer_equals(age.get("sample_count"), expected_samples),
    "age_only_twenty_permutations": age.get("permutation_type") == "AGE_only"
        and integer_equals(age.get("permutation_repeats"), 20),
    "analysis_seed_matches": all(report.get("provenance", {}).get("seed") == 20261112
        for report in (diagnostics, age)),
    "downstream_analysis_no_training_or_release": all(
        report.get("training_performed") is False
        and integer_equals(report.get("new_marginal_releases"), 0)
        for report in (diagnostics, age)),
}

draws = replay.get("draws", [])
expected_seeds = {1: 12026, 2: 12126, 3: 12226}
checks["all_three_B_C_and_zero_guidance_replays_pass"] = (
    isinstance(draws, list) and len(draws) == 3
    and {draw.get("draw_id") for draw in draws} == set(expected_seeds)
    and all(type(draw.get("draw_id")) is int
            and draw.get("generation_seed") == expected_seeds[draw["draw_id"]]
            and all(draw.get(key) is True for key in (
                "b_matches_historical", "zero_matches_b", "zero_rng_matches_b", "c_matches_historical"))
            for draw in draws))

artifact_names = ("preflight.json", "plan.json", "replay.json", "reference.json", "results.json", "summary.txt", "sample_manifest.json")
declared_artifacts = complete.get("artifact_sha256", {})
checks["generation_artifact_hashes_match"] = isinstance(declared_artifacts, dict) and all(
    declared_artifacts.get(name) == sha256(generation / name) for name in artifact_names)

expected = {}
identities = set()
condition_draws = {}
allowed_targets = {None, "cached_target01"}
if include_oracle:
    allowed_targets.add("train_oracle_NONPRIVATE")
for entry in samples:
    if not isinstance(entry, dict):
        raise RuntimeError("Invalid generated sample manifest entry")
    source = Path(entry["path"])
    if not source.is_absolute() or source.is_symlink():
        raise RuntimeError("Generated sample paths must be absolute regular files")
    source = source.resolve(strict=True)
    source.relative_to(generation)
    if not source.is_file() or source.suffix != ".npz" or str(source) in expected:
        raise RuntimeError("Generated sample path is duplicated or has the wrong format")
    if entry.get("key") != "X_synthetic" or not integer_equals(entry.get("n_rows"), 512):
        raise RuntimeError("Generated manifest array contract is not 512 rows of X_synthetic")
    if entry.get("fit_id") != "gpu22692665_B_private" or entry.get("target_id") not in allowed_targets:
        raise RuntimeError("Generated sample fit or target identity differs from the planned fixed objects")
    if type(entry.get("draw_id")) is not int or entry["draw_id"] not in expected_seeds:
        raise RuntimeError("Generated sample draw ID differs from 1, 2, 3")
    if not isinstance(entry.get("arm"), str) or not entry["arm"]:
        raise RuntimeError("Generated sample condition name is missing")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != sha256(source):
        raise RuntimeError("Generated sample byte hash differs from its manifest")
    identity = (entry["arm"], entry["fit_id"], entry["target_id"], entry["draw_id"])
    if identity in identities:
        raise RuntimeError("Generated sample identifiers are duplicated")
    identities.add(identity)
    condition_draws.setdefault(entry["arm"], []).append(entry["draw_id"])
    expected[str(source)] = digest
checks["three_draws_per_condition"] = len(condition_draws) == expected_conditions and all(
    sorted(values) == [1, 2, 3] for values in condition_draws.values())
checks["oracle_scope_matches_request"] = (
    any(entry.get("target_id") == "train_oracle_NONPRIVATE" for entry in samples) == include_oracle)

manifest_digest = sha256(manifest_path)
checks["diagnostic_manifest_hash_matches"] = diagnostics.get("provenance", {}).get("files", {}).get(
    "samples_manifest", {}).get("sha256") == manifest_digest
age_manifests = age.get("provenance", {}).get("sample_manifests", [])
checks["age_manifest_hash_matches"] = len(age_manifests) == 1 and age_manifests[0].get("sha256") == manifest_digest
for label, report in (("diagnostics", diagnostics), ("age", age)):
    observed = report.get("samples", [])
    checks[f"{label}_sample_hashes_and_rows_match"] = (
        isinstance(observed, list) and len(observed) == expected_samples
        and {entry.get("path"): entry.get("sha256") for entry in observed} == expected
        and all(integer_equals(entry.get("rows"), 512) for entry in observed))
    checks[f"{label}_sample_identities_match"] = {
        (entry.get("arm"), entry.get("fit_id"), entry.get("target_id"), entry.get("draw_id"))
        for entry in observed} == identities
    controls = report.get("real_real", {})
    checks[f"{label}_twenty_dependent_controls"] = (
        integer_equals(controls.get("repeats"), 20) and len(controls.get("results", [])) == 20
        and integer_equals(controls.get("rows_per_side"), 512)
        and controls.get("within_repeat_disjoint") is True
        and controls.get("across_repeat_independent") is False)
    files = report.get("provenance", {}).get("files", {})
    checks[f"{label}_exact_input_hashes_match"] = all(
        files.get(name, {}).get("sha256") == digest for name, digest in expected_input_hashes.items())
checks["twenty_column_shuffles_per_sample"] = all(
    integer_equals(entry.get("metrics", {}).get("shuffle", {}).get("repeats"), 20)
    for entry in diagnostics.get("samples", []))

failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError("Guidance workflow completion checks failed: " + ", ".join(failed))
summary = {
    "status": "complete", "mode": mode, "include_train_oracle": include_oracle,
    "condition_count": expected_conditions, "sample_count": expected_samples,
    "training_performed": False, "new_training_fits": 0, "new_marginal_releases": 0,
    "checks": checks, "generated_manifest_sha256": manifest_digest,
    "scope": "Frozen-B generation and descriptive diagnostics. One existing private fit and cached target. Optional TRAIN oracle is explicitly nonprivate. No end-to-end DP release is claimed.",
}
with (out / "completion_summary.json").open("x") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(f"Verified {expected_conditions} conditions, {expected_samples} samples, all replay gates, and both analyses.")
PY

REPORT_FILES=(
    completion_summary.json
    generation/COMPLETED.json generation/preflight.json generation/plan.json
    generation/replay.json generation/results.json generation/summary.txt
    generation/sample_manifest.json
    diagnostics/diagnostics.json diagnostics/metrics_per_draw.csv
    diagnostics/age_bin_contributions.csv diagnostics/pair_cell_errors.csv
    diagnostics/matched_cell_changes.csv diagnostics/real_real_controls.csv diagnostics/README.txt
    age_permutation/age_permutation.json age_permutation/metrics_per_draw.csv
    age_permutation/real_real_controls.csv age_permutation/README.txt
)
for REPORT_FILE in "${REPORT_FILES[@]}"; do
    [[ -f "$RUN_OUT/$REPORT_FILE" && ! -L "$RUN_OUT/$REPORT_FILE" ]] || die "Missing regular aggregate report: $REPORT_FILE"
done
[[ ! -e "$RUN_OUT/aggregate_reports.tar.gz" && ! -L "$RUN_OUT/aggregate_reports.tar.gz" ]] || die "Aggregate archive already exists."
tar -czf "$RUN_OUT/aggregate_reports.tar.gz" -C "$RUN_OUT" "${REPORT_FILES[@]}"
printf 'COMPLETED: frozen-B %s job=%s\n' "$GUIDANCE_MODE" "$SLURM_JOB_ID" > "$RUN_OUT/COMPLETED.txt"
printf 'Completed frozen-B workflow. Aggregate report archive: %s/aggregate_reports.tar.gz\n' "$RUN_OUT"
# END GUIDANCE POSTCHECKS AND PACKAGE
