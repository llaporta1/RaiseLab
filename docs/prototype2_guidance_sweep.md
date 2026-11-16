# Replay B and C, then test frozen-model guidance

This workflow uses the original B checkpoint, the retained categorical target,
and the historical sampling implementation. It regenerates the 4,096-row B
calibration reference once, then checks all three saved B and C draws before any
new guidance condition. It creates new samples and research diagnostics, with
no new training fit or private marginal release.

## Choose the mode

| Mode | Generation conditions | Saved 512-row outputs |
| --- | --- | --- |
| `replay` | Original B and C replay gates | 6 |
| `sweep` | Zero guidance, then strengths 0.25, 0.5, 1 across three active-feature masks | 30 |
| `sweep --include-train-oracle` | The same cached-target grid plus nine matched TRAIN-oracle conditions | 57 |

The three masks are all six categorical features, RACE/HISPAN only, and OCC/IND
only. Zero guidance is one shared condition. The fixed cap is 2, and the three
generation seeds are 12026, 12126, and 12226. The calibration recipe is 4,096 rows,
seed 13026, batch 64; the three saved B draws do not replace this reference.

Both modes require all historical B/C array comparisons and zero-guidance
sample/RNG equivalence to pass. A replay mismatch stops the job before the grid.
The optional smoothed-TRAIN oracle uses the matching 0.001 pseudocount and is an
explicitly nonprivate diagnostic. The default study uses the retained target.

## Use the original GPU environment

Run the commands below in a Rivanna login terminal. The GPU interpreter defaults
to `$HOME/.conda/envs/tabdiff-p2-gpu/bin/python`; `RAISE_GUIDANCE_PYTHON` can name
the same verified environment at another path. The historical submitters invoked
it directly, without a module load or Conda activation. Keep the prepared
`.venv-pgm` separate from this GPU replay environment.

The runner checks the historical environment: Python 3.11.18, NumPy 1.26.4,
Torch 2.2.2+cu121 with CUDA 12.1, Opacus 1.5.2, SciPy 1.16.3, and one NVIDIA RTX
A6000. The job requests one CPU, 8 GB RAM, and a six-hour limit with no requeue.
Six hours is a resource limit, not a runtime prediction. The launcher performs
the NumPy-only diagnostic work on the same allocation after GPU generation.

## Check the original files, then submit one job

The source kit must already be present under the original project. Use the
completed run `gpu_longrun_v1p1_22692665`, its saved B/C arrays and target, and the
exact historical input files. The read-only `--check-files` command below uses
standard-library hash checks and does not initialize a GPU.

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
git pull --ff-only
umask 077
mkdir -p pipeline_runs/slurm

RAISE_ORIGINAL_PROJECT="$HOME/projects/TabDiff-prototype2"
RAISE_ORIGINAL_RUN="/scratch/asj7gw/tabdiff-p2/runs/gpu_longrun_v1p1_22692665"
RAISE_PILOT_DIR="$RAISE_ORIGINAL_PROJECT/pipeline_runs/prototype2_cpu/ipums_2048"
RAISE_GUIDANCE_PYTHON="${RAISE_GUIDANCE_PYTHON:-$HOME/.conda/envs/tabdiff-p2-gpu/bin/python}"
export RAISE_GUIDANCE_PYTHON
RAISE_MODE=sweep
RAISE_ORACLE_FLAGS=()
# Optional nonprivate comparison, selected before submission:
# RAISE_ORACLE_FLAGS=(--include-train-oracle)

(
  set -euo pipefail
  "$RAISE_GUIDANCE_PYTHON" -B scripts/run_prototype2_guidance_sweep.py \
    --check-files --mode "$RAISE_MODE" \
    --project-root "$RAISE_ORIGINAL_PROJECT" --run-dir "$RAISE_ORIGINAL_RUN" \
    --data-npz "$RAISE_PILOT_DIR/pilot_data.npz" --schema "$RAISE_PILOT_DIR/schema.json"

  RAISE_RECEIPT="$PWD/pipeline_runs/guidance_${RAISE_MODE}_submission"
  mkdir -m 700 "$RAISE_RECEIPT"
  printf 'One submission attempt. Preserve this receipt if the terminal disconnects.\n' \
    > "$RAISE_RECEIPT/attempt.txt"
  printf '%s\n' "$RAISE_MODE" > "$RAISE_RECEIPT/mode.txt"
  sbatch --parsable --account=raiselab \
    slurm/prototype2_guidance_sweep.sbatch \
    "$RAISE_MODE" "$RAISE_ORIGINAL_PROJECT" "$RAISE_ORIGINAL_RUN" \
    "$RAISE_PILOT_DIR/pilot_data.npz" "$RAISE_PILOT_DIR/schema.json" \
    "${RAISE_ORACLE_FLAGS[@]}" > "$RAISE_RECEIPT/sbatch_response.txt"
  RAISE_JOB_RECEIPT=$(cat "$RAISE_RECEIPT/sbatch_response.txt")
  RAISE_GUIDANCE_JOB="${RAISE_JOB_RECEIPT%%;*}"
  [[ "$RAISE_GUIDANCE_JOB" =~ ^[0-9]+$ ]]
  printf '%s\n' "$RAISE_GUIDANCE_JOB" > "$RAISE_RECEIPT/job_id.txt"
  printf 'Submitted job %s. Receipt: %s\n' "$RAISE_GUIDANCE_JOB" "$RAISE_RECEIPT"
)
```

Set `RAISE_MODE=replay` for replay alone. The optional oracle flag is supported
only with `sweep`. The receipt directory is created exclusively; if it already
exists, inspect it rather than submitting a duplicate. A reviewed additional
attempt should use a new receipt directory while retaining the previous one.

The launcher rejects direct login-node execution and creates a fresh output
directory for the Slurm job. It passes the explicit trusted-checkpoint flag to
the runner for the user's own saved B checkpoint. It preserves the original
project, source kits, target, checkpoints, and saved samples.

## Reconnect and inspect completion

A successfully submitted Slurm job continues after the browser disconnects.
From a new login terminal, restore the mode and read its saved receipt:

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
RAISE_MODE=sweep
RAISE_RECEIPT="$PWD/pipeline_runs/guidance_${RAISE_MODE}_submission"
RAISE_GUIDANCE_JOB=$(cat "$RAISE_RECEIPT/job_id.txt")
sacct -j "$RAISE_GUIDANCE_JOB" --format=JobID,State%20,ExitCode,Elapsed
tail -n 100 "pipeline_runs/slurm/prototype2-guidance-${RAISE_GUIDANCE_JOB}.out"
```

If the connection was lost before `job_id.txt` was written, inspect
`sbatch_response.txt` and `squeue --user="$USER" --name=prototype2-guidance`.
An empty response is ambiguous; do not submit again until the queue/accounting
state is known. Queue disappearance alone is not a failure diagnosis: use
`sacct` for completed jobs.

After Slurm reports `COMPLETED` with exit `0:0`:

```bash
RAISE_GUIDANCE_OUT="pipeline_runs/guidance_${RAISE_MODE}_${RAISE_GUIDANCE_JOB}"
cat "$RAISE_GUIDANCE_OUT/COMPLETED.txt"
cat "$RAISE_GUIDANCE_OUT/completion_summary.json"
cat "$RAISE_GUIDANCE_OUT/generation/summary.txt"
ls -lh "$RAISE_GUIDANCE_OUT/aggregate_reports.tar.gz"
```

The final marker is written only after generation checks, both diagnostic
analyses, postchecks, and archive creation. `generation/COMPLETED.json` alone
establishes generation completion, not completion of the entire workflow.
Retain logs and partial outputs after failure. A packaging or diagnostic failure
can be investigated without automatically regenerating the model samples.

## Read and share the aggregate results

The workflow runs twenty marginal-preserving column shuffles, twenty AGE-only
permutations, and twenty dependent train-pool real-real controls. Positive
permuted-minus-original gaps favor the original alignment for that query. The
AGE-only control preserves complete categorical rows and every one-way
marginal. The full-column shuffle also breaks categorical relationships.

Download `aggregate_reports.tar.gz` through Rivanna's file browser. Its explicit
whitelist contains the generation plan, replay/provenance summaries and sample
manifest, plus aggregate diagnostics and AGE reports. The local reference
probability vectors, generated NPZ arrays, checkpoint, and original cached
target are excluded. Exact training/holdout summaries remain nonprivate research
information, even though the archive contains no record arrays.

The sweep measures guidance response conditional on one B fit, one calibration
reference, and the retained target. Three generation draws are not three training
fits. A matched TRAIN-oracle contrast addresses the realized postprocessed
target perturbation conditional on those same objects; it is not a private,
budget-matched result. Keep all prespecified conditions, including regressions.
Independent training or fresh targets are separate future experiments.

## Validation boundary

The launcher and walkthrough are prepared for the verified historical source and
output contracts. `bash -n` and eleven focused launcher/package tests passed.
The tests exercise the actual postcheck and archive block for all three output
counts, check the local reference hash without archiving its vectors, reject
missing/changed files and failed replay, prevent overwrite, and check early
failure handling. They use generated arrays and stubbed analysis reports only.
Successful historical CUDA replay, runtime, resource use, and new research
results must be established by the submitted Rivanna job. No new Rivanna replay
or sweep is claimed by this handoff.
