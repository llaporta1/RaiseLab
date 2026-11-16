# Analyze AGE dependence and the cached categorical target

This CPU follow-up reads the original saved diffusion samples, the completed
AIM/MST samples, and the already-cached categorical marginal target. It performs
two analyses without fitting or sampling a model and without drawing a new
private target. Original run directories remain unchanged.

## What the two analyses answer

**AGE-only permutation:** retain each row's six categorical values together and
permute only AGE. Evaluate the six AGE-by-category pairs and three previously
fixed AGE triples against the same holdout. The categorical joint distribution
and every one-way marginal are preserved exactly. A positive
`age_permuted_minus_original` means the original AGE assignment had lower error
than the permuted assignments for that query. A near-zero contrast supplies
little evidence that the original AGE assignment helps that query. This is a
descriptive diagnostic, not a significance test or an independent model fit.

The same permutation schedules are shared across equal row counts and normalized
draw identifiers. Twenty train-only real-real comparisons provide context. Each
control uses disjoint halves within that repeat, but repeated controls reuse the
same training pool. They are dependent and do not define a population error floor.

**Cached-target audit:** compare the single saved categorical target with the
exact training frequencies from the verified historical input.
The audit checks the exact historical serialization contract and source hash.
The target hash is checked against this job's manifest preflight. These checks
bind the report to the current cached bytes; they do not independently prove
that those bytes are unchanged from the target used for the original C draws.
Differences from the holdout are separately labeled as sampling context. A
target-versus-holdout difference alone cannot identify privacy noise, because
training and holdout frequencies can differ before noise is added.

The target-minus-training comparison describes the realized output of the whole
cached-target construction, including noise, clipping, pseudocounts and
normalization. It does not recover the original Laplace noise draws, estimate
their distribution from one target, or prove that target perturbation caused a
particular synthetic joint error.

These are local research diagnostics. Exact training frequencies and private
holdout scores do not constitute an end-to-end differentially private release.

## Submit one saved-output follow-up

Use the existing prepared checkout and `.venv-pgm`. Set the two saved-run paths
to the **already completed** original GPU run and AIM/MST comparison. Do not
resubmit either training workflow to run this analysis.

From a login shell:

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
git pull --ff-only
umask 077
mkdir -p pipeline_runs/slurm

RAISE_GPU_RUN="/absolute/path/to/the/saved/gpu_run"
RAISE_PGM_RUN="$PWD/pipeline_runs/pgm_compare_JOBID"
RAISE_PILOT_DIR="$HOME/projects/TabDiff-prototype2/pipeline_runs/prototype2_cpu/ipums_2048"

if FOLLOWUP_JOB=$(sbatch --parsable --account=raiselab \
  slurm/prototype2_saved_followup.sbatch \
  "$RAISE_GPU_RUN" "$RAISE_PGM_RUN" \
  "$RAISE_PILOT_DIR/pilot_data.npz" "$RAISE_PILOT_DIR/schema.json"); then
  FOLLOWUP_JOB="${FOLLOWUP_JOB%%;*}"
  printf '%s\n' "$FOLLOWUP_JOB" > pipeline_runs/saved_followup_jobid.txt
  printf 'Saved-output follow-up job: %s\n' "$FOLLOWUP_JOB"
fi
```

The resource request is four CPU cores, 16 GB memory, and a 30-minute limit. It
requires no GPU and uses the installed NumPy environment. This is a requested
limit, not a measured runtime promise. The exact original data and schema hashes
are checked before research analysis. The launcher also requires the PGM run's
completion marker and its original sample manifest.

The launcher prepares a fresh manifest for the fifteen original GPU samples,
audits the cached target, then evaluates those samples together with eighteen
existing PGM samples. It checks all thirty-three samples and the two completed
reports before creating a whitelist-only aggregate archive. No record arrays,
checkpoints, or source target file are added to that archive.

## Resume monitoring after a disconnected terminal

The Slurm job persists independently of the login shell. Read the saved job ID:

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
FOLLOWUP_JOB=$(cat pipeline_runs/saved_followup_jobid.txt)
sacct -j "$FOLLOWUP_JOB" --format=JobID,State%20,ExitCode,Elapsed
tail -n 80 "pipeline_runs/slurm/prototype2-followup-${FOLLOWUP_JOB}.out"
```

An empty log immediately after submission can mean startup is still in progress.
Use accounting and the completion marker before interpreting the result. If the
job no longer appears in `squeue`, use `sacct`; absence from the queue is not a
failure diagnosis.

When accounting shows completion, verify:

```bash
FOLLOWUP_OUT="pipeline_runs/saved_followup_${FOLLOWUP_JOB}"
cat "$FOLLOWUP_OUT/COMPLETED.txt"
cat "$FOLLOWUP_OUT/completion_summary.json"
cat "$FOLLOWUP_OUT/target_audit/target_audit.txt"
cat "$FOLLOWUP_OUT/age_permutation/README.txt"
ls -lh "$FOLLOWUP_OUT/aggregate_reports.tar.gz"
```

Download `aggregate_reports.tar.gz` through the existing Rivanna file browser if
the separate SCP connection is unreliable. Keep the unchanged source samples
and target in their original locations. If packaging fails, inspect the reports
and recover only the archive; do not automatically rerun computations.

## Use the findings to choose a controlled next experiment

1. Compare the AGE-only contrasts with the real-real context before attributing
   a three-way gain to AGE dependence. Full-column shuffles destroy the
   categorical relationships too and cannot answer that narrower question.
2. Inspect target-versus-training errors before assigning the RACE/HISPAN
   tradeoff to a noisy target. One cached target gives realized perturbation,
   not a replicated privacy-noise study.
3. A subsequent generation-only experiment can hold the B checkpoint and cached
   target fixed and vary prespecified guidance strengths using common generation
   seeds. This identifies sensitivity to guidance for those fixed objects.
   A training-frequency oracle control would be an explicitly nonprivate
   diagnostic. It should be kept separate from privacy-budget-matched results.
4. Prespecify AGE support/encoding and any public occupation/industry grouping
   before a balanced training comparison. Report detailed and grouped queries,
   zero/nonzero patterns, and conditional queries with their denominators.
   Holdout inspection in this pilot is exploratory; a confirmatory comparison
   should use an untouched evaluation split or an explicitly frozen protocol.

All new model fits or fresh targets require their own recorded mechanisms and
accounting. Repeated draws from a fixed model do not create independent training
replicates. The saved-output job here creates neither fits nor targets.
