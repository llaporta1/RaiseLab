#!/bin/bash
#SBATCH --job-name=prototype2-diagnostics
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=pipeline_runs/slurm/%x-%j.out

# Submit from the prepared repository root, after mkdir -p pipeline_runs/slurm:
# sbatch --account=YOUR_ACCOUNT slurm/prototype2_saved_gpu_diagnostics.sbatch \
#   SAVED_GPU_RUN_DIR DATA_NPZ SCHEMA_JSON
# This evaluates saved samples only. It never loads a model, creates a private
# target, samples a model, or calls the AIM/MST fitting runner.

set -euo pipefail
set -o noclobber
umask 077

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die "Submit with sbatch from a login node; do not run this script with bash there."
[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || die "SLURM_SUBMIT_DIR is missing. Submit from the repository root."
[[ $# -eq 3 ]] || die "Usage: SAVED_GPU_RUN_DIR DATA_NPZ SCHEMA_JSON"
cd -- "$SLURM_SUBMIT_DIR"
REPO_ROOT="$(pwd -P)"
for REQUIRED_SOURCE in scripts/prepare_prototype2_saved_gpu_manifest.py \
    scripts/check_prototype2_rivanna_inputs.py scripts/run_prototype2_diagnostics.py; do
    [[ -f "$REQUIRED_SOURCE" ]] || die "Missing $REQUIRED_SOURCE; submit from the prepared repository root."
done

type module >/dev/null 2>&1 || die "The module command is unavailable."
module purge
module load "${RAISE_PGM_MODULE:-miniforge}"
module list
DIAG_PYTHON="$REPO_ROOT/.venv-pgm/bin/python"
[[ -x "$DIAG_PYTHON" ]] || die "The existing .venv-pgm environment is required."
[[ "${SLURM_CPUS_PER_TASK:-4}" =~ ^[1-9][0-9]*$ ]] || die "Invalid CPU count."
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
unset PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

RUN_OUT="$REPO_ROOT/pipeline_runs/diffusion_diagnostics_${SLURM_JOB_ID}"
mkdir -p -- "$REPO_ROOT/pipeline_runs"
mkdir -- "$RUN_OUT" || die "Output already exists; no overwrite or automatic retry: $RUN_OUT"
printf '*\n' > "$RUN_OUT/.gitignore"
on_exit() {
    EXIT_STATUS=$?
    if [[ "$EXIT_STATUS" -ne 0 ]]; then
        printf 'FAILED: diagnostic-only job=%s exit=%s\nPartial reports are not completed results.\n' \
            "$SLURM_JOB_ID" "$EXIT_STATUS" > "$RUN_OUT/FAILED.txt"
    fi
}
trap on_exit EXIT

printf 'Saved-GPU diagnostics job=%s; output=%s\n' "$SLURM_JOB_ID" "$RUN_OUT"
"$DIAG_PYTHON" scripts/prepare_prototype2_saved_gpu_manifest.py \
    --run-dir "$1" --data-npz "$2" --schema "$3" \
    --out "$RUN_OUT/samples_manifest.json"
"$DIAG_PYTHON" scripts/run_prototype2_diagnostics.py \
    --data-npz "$2" --schema "$3" \
    --samples-manifest "$RUN_OUT/samples_manifest.json" \
    --sample-rows 512 --real-real-repeats 20 --shuffle-repeats 10 \
    --seed 20261112 --out "$RUN_OUT/diagnostics"

"$DIAG_PYTHON" - "$RUN_OUT" <<'PY'
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
manifest = json.loads((out / "samples_manifest.json").read_text())
report = json.loads((out / "diagnostics/diagnostics.json").read_text())
checks = {
    "status_complete": report.get("status") == "complete",
    "fifteen_samples": len(report.get("samples", [])) == 15,
    "six_matched_comparisons": len(report.get("matched_comparisons", [])) == 6,
    "no_unmatched_comparisons": report.get("unmatched_comparisons") == [],
    "no_training": report.get("training_performed") is False,
    "no_new_target": report.get("new_marginal_releases") == 0,
    "no_frozen_model_intervention": report.get("frozen_model_intervention_performed") is False,
}
expected = {entry["path"]: entry["sha256"] for entry in manifest["samples"]}
observed = {entry["path"]: entry["sha256"] for entry in report["samples"]}
checks["sample_hashes_match_preflight"] = expected == observed
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError("Diagnostic completion checks failed: " + ", ".join(failed))
summary = {"status": "complete", "checks": checks,
           "note": "One historical training replicate; C shares B and one cached target. No new fits or targets."}
with (out / "completion_summary.json").open("x") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write("\n")
print("Verified 15 saved samples, 6 matched comparisons, and unchanged sample hashes.")
PY

# Bundle only the manifest and aggregate diagnostic reports for review.
tar -czf "$RUN_OUT/aggregate_reports.tar.gz" -C "$RUN_OUT" \
    samples_manifest.json completion_summary.json \
    diagnostics/diagnostics.json diagnostics/metrics_per_draw.csv \
    diagnostics/age_bin_contributions.csv diagnostics/pair_cell_errors.csv \
    diagnostics/matched_cell_changes.csv diagnostics/real_real_controls.csv \
    diagnostics/summary.txt
printf 'COMPLETED: diagnostic-only job=%s\n' "$SLURM_JOB_ID" > "$RUN_OUT/COMPLETED.txt"
printf 'Completed diagnostics. Aggregate report archive: %s/aggregate_reports.tar.gz\n' "$RUN_OUT"
