#!/bin/bash
#SBATCH --job-name=prototype2-pgm
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=pipeline_runs/slurm/%x-%j.out

# Submit from the repository root, after creating pipeline_runs/slurm:
#   sbatch --account=raiselab slurm/prototype2_pgm.sbatch validate
#   sbatch --account=raiselab slurm/prototype2_pgm.sbatch compare DATA_NPZ SCHEMA [DIFFUSION_MANIFEST]
# In an existing interactive allocation started from the repository root:
#   bash slurm/prototype2_pgm.sbatch validate
# Submit sbatch jobs from the login node, not from an interactive allocation.
# Local syntax/stub checks validate launcher control flow only; they do not
# constitute a Rivanna job or evidence about the historical research data.

set -euo pipefail
set -o noclobber
umask 077

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || die "Submit with sbatch from a login node, or use bash inside an existing Slurm allocation; never run bash on a login node."
[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || die "SLURM_SUBMIT_DIR is missing; submit from the repository root."
[[ $# -ge 1 ]] || die "Usage: validate OR compare DATA_NPZ SCHEMA [DIFFUSION_MANIFEST]"
MODE="$1"
shift
case "$MODE" in
    validate)
        [[ $# -eq 0 ]] || die "validate takes no input-data arguments."
        ;;
    compare)
        [[ $# -eq 2 || $# -eq 3 ]] || die "compare requires DATA_NPZ SCHEMA and optionally DIFFUSION_MANIFEST."
        ;;
    *) die "Mode must be validate or compare." ;;
esac

cd -- "$SLURM_SUBMIT_DIR"
REPO_ROOT="$(pwd -P)"
for REQUIRED_SOURCE in \
    scripts/check_prototype2_rivanna_inputs.py \
    scripts/make_prototype2_public_fixture.py \
    scripts/run_ipums_pgm_baselines.py \
    scripts/prototype2_pgm_backend.py \
    scripts/prototype2_evaluation.py \
    scripts/prototype2_relationship_checks.py \
    scripts/run_prototype2_diagnostics.py; do
    [[ -f "$REPO_ROOT/$REQUIRED_SOURCE" ]] || die "Missing $REQUIRED_SOURCE. Submit from the prepared repository root."
done

type module >/dev/null 2>&1 || die "The Rivanna module command is unavailable in this job shell."
module purge
module load "${RAISE_PGM_MODULE:-miniforge/24.11.3-py3.12}"
module list

PGM_PYTHON="$REPO_ROOT/.venv-pgm/bin/python"
PGM_REPO="${RAISE_PGM_REPO:-$REPO_ROOT/../mbi}"
MBI_COMMIT="5f97065a17b3e53cfefc7a8ce3e293fd153dd064"
[[ -x "$PGM_PYTHON" ]] || die "Prepare the isolated environment at .venv-pgm before submission."
[[ -d "$PGM_REPO" ]] || die "The pinned MBI checkout is missing: $PGM_REPO"
[[ "${SLURM_CPUS_PER_TASK:-4}" =~ ^[1-9][0-9]*$ ]] || die "SLURM_CPUS_PER_TASK must be a positive integer."

export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
unset PYTHONPATH
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"

RUN_OUT="$REPO_ROOT/pipeline_runs/pgm_${MODE}_${SLURM_JOB_ID}"
mkdir -p -- "$REPO_ROOT/pipeline_runs"
mkdir -- "$RUN_OUT" || die "Output directory already exists; this launcher never retries or overwrites: $RUN_OUT"
printf '*\n' > "$RUN_OUT/.gitignore"
on_exit() {
    EXIT_STATUS=$?
    if [[ "$EXIT_STATUS" -ne 0 ]]; then
        printf 'FAILED: mode=%s job=%s exit=%s\nNo completed comparison may be inferred from a partial run.\n' \
            "$MODE" "$SLURM_JOB_ID" "$EXIT_STATUS" > "$RUN_OUT/FAILED.txt"
    fi
}
trap on_exit EXIT

printf 'Prototype-2 mode=%s Slurm job=%s CPUs=%s\nOutput=%s\n' \
    "$MODE" "$SLURM_JOB_ID" "$OMP_NUM_THREADS" "$RUN_OUT"

if [[ "$MODE" == validate ]]; then
    printf 'PUBLIC GENERATED FIXTURE: SOFTWARE VALIDATION ONLY; NO IPUMS FINDINGS.\n'
    "$PGM_PYTHON" scripts/make_prototype2_public_fixture.py \
        --out "$RUN_OUT/fixture" --train-rows 256 --holdout-rows 128 --sample-rows 64
    "$PGM_PYTHON" scripts/run_prototype2_diagnostics.py \
        --data-npz "$RUN_OUT/fixture/pilot_data.npz" --schema "$RUN_OUT/fixture/schema.json" \
        --samples-manifest "$RUN_OUT/fixture/samples_manifest.json" \
        --sample-rows 64 --real-real-repeats 3 --shuffle-repeats 10 \
        --out "$RUN_OUT/fixture_diagnostics"
    "$PGM_PYTHON" scripts/run_ipums_pgm_baselines.py \
        --data-npz "$RUN_OUT/fixture/pilot_data.npz" --schema "$RUN_OUT/fixture/schema.json" \
        --mbi-repo "$PGM_REPO" --expected-mbi-commit "$MBI_COMMIT" \
        --arms aim mst --fits 1 --draws 2 --samples 64 --shuffle-repeats 10 \
        --epsilon 2.5638977279554265 --delta 1e-5 --jax-platform cpu \
        --public-fixture --smoke --out "$RUN_OUT/baselines"
    "$PGM_PYTHON" scripts/run_prototype2_diagnostics.py \
        --data-npz "$RUN_OUT/fixture/pilot_data.npz" --schema "$RUN_OUT/fixture/schema.json" \
        --samples-manifest "$RUN_OUT/baselines/sample_manifest.json" \
        --sample-rows 64 --real-real-repeats 3 --shuffle-repeats 10 \
        --out "$RUN_OUT/baseline_diagnostics"
else
    DATA_NPZ="$1"
    SCHEMA_PATH="$2"
    DIFFUSION_MANIFEST="${3:-}"
    if [[ -n "$DIFFUSION_MANIFEST" ]]; then
        [[ -f "$DIFFUSION_MANIFEST" ]] || die "The supplied diffusion sample manifest does not exist."
    fi
    # Exact original inputs are required BEFORE any baseline fit. A mismatch
    # exits immediately, and no fixture or alternate input is substituted.
    "$PGM_PYTHON" scripts/check_prototype2_rivanna_inputs.py \
        --data-npz "$DATA_NPZ" --schema "$SCHEMA_PATH" > "$RUN_OUT/input_preflight.json"
    cat "$RUN_OUT/input_preflight.json"
    # Validate supplied existing samples before consuming any new private fits.
    if [[ -n "$DIFFUSION_MANIFEST" ]]; then
        "$PGM_PYTHON" scripts/run_prototype2_diagnostics.py \
            --data-npz "$DATA_NPZ" --schema "$SCHEMA_PATH" \
            --samples-manifest "$DIFFUSION_MANIFEST" \
            --sample-rows 512 --real-real-repeats 20 --shuffle-repeats 10 \
            --out "$RUN_OUT/diffusion_diagnostics"
    fi
    printf 'LOCAL RESEARCH COMPARISON: standard inference defaults; diagnostics use private holdout data.\n'
    "$PGM_PYTHON" scripts/run_ipums_pgm_baselines.py \
        --data-npz "$DATA_NPZ" --schema "$SCHEMA_PATH" \
        --mbi-repo "$PGM_REPO" --expected-mbi-commit "$MBI_COMMIT" \
        --arms aim mst --fits 3 --draws 3 --samples 512 --shuffle-repeats 10 \
        --epsilon 2.5638977279554265 --delta 1e-5 --jax-platform cpu \
        --out "$RUN_OUT/baselines"
    "$PGM_PYTHON" scripts/run_prototype2_diagnostics.py \
        --data-npz "$DATA_NPZ" --schema "$SCHEMA_PATH" \
        --samples-manifest "$RUN_OUT/baselines/sample_manifest.json" \
        --sample-rows 512 --real-real-repeats 20 --shuffle-repeats 10 \
        --out "$RUN_OUT/baseline_diagnostics"
fi

printf 'COMPLETED: mode=%s job=%s\n' "$MODE" "$SLURM_JOB_ID" > "$RUN_OUT/COMPLETED.txt"
printf 'Completed %s; inspect %s\n' "$MODE" "$RUN_OUT"
