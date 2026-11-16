#!/bin/bash
#SBATCH --job-name=prototype2-followup
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=pipeline_runs/slurm/%x-%j.out

# CPU analysis of saved samples and one already-cached target. No model fits,
# model sampling, new target draws, or changes to the original run directories.
# Submit from the prepared repository root after mkdir -p pipeline_runs/slurm:
# sbatch --account=YOUR_ACCOUNT slurm/prototype2_saved_followup.sbatch \
#   SAVED_GPU_RUN SAVED_PGM_RUN DATA_NPZ SCHEMA_JSON

set -euo pipefail
set -o noclobber
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die "Submit with sbatch from a login node; do not run this script with bash there."
[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || die "SLURM_SUBMIT_DIR is missing. Submit from the repository root."
[[ $# -eq 4 ]] || die "Usage: SAVED_GPU_RUN SAVED_PGM_RUN DATA_NPZ SCHEMA_JSON"
cd -- "$SLURM_SUBMIT_DIR"
REPO_ROOT="$(pwd -P)"
for REQUIRED_SOURCE in scripts/prepare_prototype2_saved_gpu_manifest.py \
    scripts/audit_prototype2_cached_targets.py scripts/run_prototype2_age_permutation.py; do
    [[ -f "$REQUIRED_SOURCE" ]] || die "Missing $REQUIRED_SOURCE; submit from the prepared repository root."
done
[[ -f "$2/COMPLETED.txt" ]] || die "The supplied PGM comparison has no completion marker."
[[ -r "$2/baselines/sample_manifest.json" ]] || die "The supplied PGM sample manifest is missing."
[[ -r "$1/dp_targets_01.json" ]] || die "The saved GPU target file is missing."

type module >/dev/null 2>&1 || die "The module command is unavailable."
module purge
module load "${RAISE_PGM_MODULE:-miniforge}"
module list
FOLLOWUP_PYTHON="$REPO_ROOT/.venv-pgm/bin/python"
[[ -x "$FOLLOWUP_PYTHON" ]] || die "The existing .venv-pgm environment is required."
[[ "${SLURM_CPUS_PER_TASK:-4}" =~ ^[1-9][0-9]*$ ]] || die "Invalid CPU count."
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

RUN_OUT="$REPO_ROOT/pipeline_runs/saved_followup_${SLURM_JOB_ID}"
mkdir -p -- "$REPO_ROOT/pipeline_runs"
mkdir -- "$RUN_OUT" || die "Output already exists; inspect it before submitting another job: $RUN_OUT"
printf '*\n' > "$RUN_OUT/.gitignore"
on_exit() {
    EXIT_STATUS=$?
    if [[ "$EXIT_STATUS" -ne 0 ]]; then
        printf 'FAILED: saved-sample follow-up job=%s exit=%s\nInspect partial outputs before retrying.\n' \
            "$SLURM_JOB_ID" "$EXIT_STATUS" > "$RUN_OUT/FAILED.txt"
    fi
}
trap on_exit EXIT

printf 'Saved-sample follow-up job=%s; output=%s\n' "$SLURM_JOB_ID" "$RUN_OUT"
"$FOLLOWUP_PYTHON" scripts/prepare_prototype2_saved_gpu_manifest.py \
    --run-dir "$1" --data-npz "$3" --schema "$4" \
    --out "$RUN_OUT/gpu_samples_manifest.json"
"$FOLLOWUP_PYTHON" scripts/audit_prototype2_cached_targets.py \
    --run-dir "$1" --data-npz "$3" --schema "$4" \
    --out "$RUN_OUT/target_audit"
"$FOLLOWUP_PYTHON" scripts/run_prototype2_age_permutation.py \
    --data-npz "$3" --schema "$4" \
    --samples-manifest "$RUN_OUT/gpu_samples_manifest.json" \
    --samples-manifest "$2/baselines/sample_manifest.json" \
    --sample-rows 512 --permutation-repeats 20 --real-real-repeats 20 \
    --seed 20261112 --out "$RUN_OUT/age_permutation"

# BEGIN FOLLOWUP POSTCHECKS AND PACKAGE
"$FOLLOWUP_PYTHON" - "$RUN_OUT" <<'PY'
from collections import Counter
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
age = json.loads((out / "age_permutation/age_permutation.json").read_text())
target = json.loads((out / "target_audit/target_audit.json").read_text())
manifest = json.loads((out / "gpu_samples_manifest.json").read_text())
expected_counts = {"A_diag": 3, "clip_only": 3, "B_private": 3,
                   "E_matched": 3, "C": 3, "aim": 9, "mst": 9}
samples = age.get("samples", [])
expected_hashes = {entry["path"]: entry["sha256"] for entry in manifest["samples"]}
observed_gpu_hashes = {entry["path"]: entry["sha256"] for entry in samples
                       if entry["arm"] in {"A_diag", "clip_only", "B_private", "E_matched", "C"}}
checks = {
    "age_complete": age.get("status") == "complete",
    "target_complete": target.get("status") == "complete",
    "all_33_saved_samples": age.get("sample_count") == 33 and len(samples) == 33,
    "expected_arm_counts": dict(Counter(entry["arm"] for entry in samples)) == expected_counts,
    "age_only_permutations": age.get("permutation_type") == "AGE_only",
    "twenty_permutations": age.get("permutation_repeats") == 20,
    "twenty_dependent_real_real_controls": age.get("real_real", {}).get("repeats") == 20
        and age["real_real"].get("within_repeat_disjoint") is True
        and age["real_real"].get("across_repeat_independent") is False,
    "no_training": age.get("training_performed") is False and target.get("training_performed") is False,
    "no_new_target": age.get("new_marginal_releases") == 0 and target.get("new_marginal_releases") == 0
        and target.get("fresh_target_draws") == 0,
    "no_model_intervention": age.get("frozen_model_intervention_performed") is False,
    "gpu_sample_hashes_match_preflight": expected_hashes == observed_gpu_hashes,
    "cached_target_hash_matches_preflight": target.get("provenance", {}).get("files", {}).get("cached_target", {}).get("sha256")
        == manifest["provenance"]["files"]["dp_targets_01.json"]["sha256"],
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError("Follow-up completion checks failed: " + ", ".join(failed))
summary = {"status": "complete", "checks": checks,
           "scope": "Local research using the original saved outputs. No new fits, model draws or private targets. Target comparison with exact training frequencies is nonprivate."}
with (out / "completion_summary.json").open("x") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write("\n")
print("Verified AGE-only analysis of 33 saved samples and one cached-target audit.")
PY

tar -czf "$RUN_OUT/aggregate_reports.tar.gz" -C "$RUN_OUT" \
    completion_summary.json gpu_samples_manifest.json \
    target_audit/target_audit.json target_audit/target_audit.txt \
    age_permutation/age_permutation.json age_permutation/metrics_per_draw.csv \
    age_permutation/real_real_controls.csv age_permutation/README.txt
printf 'COMPLETED: saved-sample follow-up job=%s\n' "$SLURM_JOB_ID" > "$RUN_OUT/COMPLETED.txt"
printf 'Completed follow-up. Aggregate report archive: %s/aggregate_reports.tar.gz\n' "$RUN_OUT"
# END FOLLOWUP POSTCHECKS AND PACKAGE
