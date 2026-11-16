# Run the Prototype 2 follow-up on UVA Rivanna/Afton

This walkthrough runs the new AIM + Private-PGM and MST + Private-PGM baselines
on the original 2,048-training / 512-holdout IPUMS pilot. It also produces AGE,
pairwise, shuffle, and disjoint real-real diagnostics for the generated baseline
samples. Existing diffusion samples can be supplied separately.

The launcher has been checked locally with shell/control-flow fixtures. It has
**not yet been submitted to Rivanna**. The earlier local execution of the real
AIM/MST implementations used invented records and smoke inference settings;
it established software execution, not IPUMS utility findings.

## 1. Log in and inspect the environment

Use your UVA computing ID with `login.hpc.virginia.edu`, or open a login shell
through [Open OnDemand](https://ood.hpc.virginia.edu). The following commands
run on the login node:

```bash
allocations -a raiselab
module spider miniforge
```

`raiselab` is the allocation used for the earlier experiment. The allocation
command checks current access/balance. Load the currently available default
with `module load miniforge`; the launcher records the actual loaded module.
UVA's October 27, 2026 update removed older module versions, so an archived
version label may no longer load. The setup below creates a separate Conda
prefix with Python 3.12, independently of the site's base Python version.
An available full module label can still be selected with `RAISE_PGM_MODULE`.
Clear or update an override that refers to an unavailable version.

## 2. Get the follow-up code in a separate checkout

Run these on the login node. This location is separate from the earlier GPU
project and its Python environment. Use the same path in the later commands.

```bash
umask 077
mkdir -p "$HOME/projects/raise-pgm-followup"
git clone --single-branch --branch codex/prototype2-diagnostics-baselines \
  https://github.com/llaporta1/RaiseLab.git \
  "$HOME/projects/raise-pgm-followup/RaiseLab"
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
mkdir -p pipeline_runs/slurm
```

If this checkout already exists, enter it and use `git status` and
`git pull --ff-only` to update it. Do not delete an existing checkout or its
outputs to repeat the instructions.

## 3. Install on an allocated compute node

UVA restricts workflows and large environment installations on login nodes.
From the new repository root, request a one-hour setup allocation:

```bash
ijob -A raiselab -p interactive -c 4 --mem=32G -t 01:00:00
```

Wait until the allocation is granted and the shell moves to a compute node.
Then run:

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
module purge
module load miniforge
module list
conda --version
unset PYTHONPATH
export PYTHONNOUSERSITE=1

conda create --prefix "$PWD/.venv-pgm" python=3.12 pip
.venv-pgm/bin/python --version
git clone https://github.com/ryan112358/mbi.git ../mbi
git -C ../mbi checkout --detach 5f97065a17b3e53cfefc7a8ce3e293fd153dd064
.venv-pgm/bin/python -m pip install -e ../mbi -r requirements-pgm.txt
.venv-pgm/bin/python -m pip check
```

Enter `y` when Conda asks to create the new environment. Its interpreter
should report Python 3.12; the module's base Python version can differ.
Use `.venv-pgm/bin/python` directly throughout; this prefix is a Conda
environment, so do not use the old Python-venv activation command. If the MBI
clone and pinned checkout already succeeded, resume with the pip installation
instead of cloning it again.

The new environment is separate from the older PyTorch/Opacus GPU environment.
The exact upstream source and dependencies are pinned; the adapter checks the
import location and source commit before fitting. Do not substitute an
unrelated `private_pgm` package or the old baseline stubs.

The pinned `mechanisms/cdp2adp.py` imports Matplotlib even though the baseline
does not draw a plot. `requirements-pgm.txt` therefore includes Matplotlib
explicitly, and the launcher selects its noninteractive `Agg` backend. A
successful `pip check` verifies declared dependencies; it cannot catch an
undeclared import in an upstream mechanism script.

## 4. Verify execution on Rivanna using invented records

Still inside the allocated compute shell, run the validation mode directly:

```bash
bash slurm/prototype2_pgm.sbatch validate
cat "pipeline_runs/pgm_validate_${SLURM_JOB_ID}/COMPLETED.txt"
```

This is a direct command inside the existing allocation, not a nested job
submission. The launcher reads Slurm's existing environment variables; do not
assign them manually. Start the `ijob` from the repository root so that
`SLURM_SUBMIT_DIR` identifies the correct checkout.

The check creates invented records, runs diagnostic fixtures, fits actual AIM
and MST with small smoke settings, generates samples, and checks those samples
through the same diagnostics. A successful run ends with `COMPLETED.txt`.
`FAILED.txt` means the workflow did not finish. The launcher refuses to
overwrite a directory from a previous attempt; inspect the failure before
starting a different job.

### Retry a failed public validation after a dependency correction

Pull the correction and install the updated requirements in the prepared
environment. If retrying within the same interactive allocation, preserve the
failed public-fixture directory under a different name first. The following
only moves an output with a failure marker and no completion marker, and
refuses to overwrite the archive:

```bash
git pull --ff-only
.venv-pgm/bin/python -m pip install -r requirements-pgm.txt
.venv-pgm/bin/python -m pip check

FAILED_VALIDATION="pipeline_runs/pgm_validate_${SLURM_JOB_ID}"
if [ -f "$FAILED_VALIDATION/FAILED.txt" ] &&
   [ ! -e "$FAILED_VALIDATION/COMPLETED.txt" ]; then
  mv -T -n "$FAILED_VALIDATION" "${FAILED_VALIDATION}_failed_dependency"
fi
bash slurm/prototype2_pgm.sbatch validate
```

This starts a fresh generated-fixture validation; it does not resume a research
fit. Do not change `SLURM_JOB_ID` or remove failure markers. Research comparison
failures require inspecting which fits completed before deciding on any rerun.

## 5. Verify the original research inputs

Still inside the compute allocation, point to the existing `ipums_2048` input
folder in the earlier GPU project. Replace the one path below with that actual
folder; do not create a new train/holdout split.

```bash
PILOT_DIR="/absolute/path/to/the/original/pipeline_runs/prototype2_cpu/ipums_2048"
.venv-pgm/bin/python scripts/check_prototype2_rivanna_inputs.py \
  --data-npz "$PILOT_DIR/pilot_data.npz" \
  --schema "$PILOT_DIR/schema.json"
```

Expected output is a JSON object with
`status: verified_exact_historical_inputs`. The preflight checks
the exact historical file hashes, row counts, feature order, categorical
domains, encoding, and absence of public-fixture markers:

| File | Expected SHA256 |
| --- | --- |
| `pilot_data.npz` | `677c9ace8e4226ef50253e56d173a1052fb5e009f1f246780954384dbfebd906` |
| `schema.json` | `3132f1eaca5eb94448b586a8a583ecad9108f7fa62bcd18325786a057dc60e73` |

A missing file or mismatch is an input-location/provenance issue to resolve,
not a reason to disable the check. A rewritten NPZ or JSON file can contain
equivalent values but different bytes; find the original archived pair before
claiming an exact match to the recorded run.

When the validation and input checks pass, leave the interactive allocation:

```bash
exit
```

This returns to the login node. UVA directs users to submit `sbatch` jobs from
login nodes, never from an interactive or Open OnDemand compute job.

## 6. Submit the real baseline comparison

On the login node, re-enter the repository and set the input path again,
because variables created inside the interactive shell do not persist after
it exits:

```bash
cd "$HOME/projects/raise-pgm-followup/RaiseLab"
PILOT_DIR="/absolute/path/to/the/original/pipeline_runs/prototype2_cpu/ipums_2048"
umask 077
mkdir -p pipeline_runs/slurm
sbatch --account=raiselab slurm/prototype2_pgm.sbatch compare \
  "$PILOT_DIR/pilot_data.npz" "$PILOT_DIR/schema.json"
```

The resource request is one CPU node, four cores, 32 GB RAM, and a six-hour
limit in `standard`. This is an initial request, not a measured requirement
or runtime promise. No GPU is requested. If resource limits are reached,
inspect `sacct`/`seff` and the logs before changing resources or running another
mechanism; do not silently change scientific settings to obtain a completion.

The job executes:

1. Exact-input preflight before any private fit.
2. Optional diagnostics on supplied existing diffusion samples, before fitting.
3. Three independent AIM fits and three independent MST fits.
4. Three generation draws of 512 rows per fit, using the retained fitted model.
5. Shared holdout evaluation, 10 column shuffles per draw, and diagnostics
   including 20 disjoint 512-versus-512 real-real controls.

| Setting | Real comparison |
| --- | --- |
| Training / held-out records | Original 2,048 / 512 |
| Nominal privacy budget | epsilon `2.5638977279554265`, delta `1e-5`, per fit |
| AIM settings | 80 MB model cap; 1,000 inference iterations; all 21 requested pairs |
| MST settings | 2,500 selection-inference iterations; 10,000 final-inference iterations |
| AGE representation | Public integer domain 0–100; shared historical bins for evaluation |
| Aggregation | Average generation draws within fit; sample SD across independent fit means |

These are the standard settings, with no `--smoke` or `--public-fixture` flag.
AIM chooses measurements adaptively: requesting a pair does not force it to
measure that pair. The budget includes each mechanism's selection and
measurements; no extra C target is created. Separate private fits compose if
their outputs are released. This research workflow is not an audited
end-to-end private release.

## 7. Monitor and inspect outputs

Record the job number printed by `sbatch`. Replace `JOBID` below with it:

```bash
squeue -u "$USER"
tail -n 60 pipeline_runs/slurm/prototype2-pgm-JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS
seff JOBID
```

The expected output root is `pipeline_runs/pgm_compare_JOBID/`. Confirm both
Slurm success (`COMPLETED`, exit code `0:0`) and the workflow's
`COMPLETED.txt`. The baseline `results.json` should say `status: complete`,
`public_fixture: false`, and `configuration: standard_inference_defaults`.
Each arm should have three fits and each fit three draws, for 18 saved arrays.

Start with these files:

| File under the output root | Purpose |
| --- | --- |
| `input_preflight.json` | Exact original-input verification |
| `baselines/summary.txt` | Main means, between-fit SDs, and run configuration |
| `baselines/metric_summary.csv` | All aggregate scores for AIM and MST |
| `baselines/results.json` | Per-fit/per-draw detail and selected model structures |
| `baselines/sample_manifest.json` | Saved sample paths, hashes, and identities |
| `baseline_diagnostics/diagnostics.json` | AGE and pairwise diagnostic totals |
| `baseline_diagnostics/real_real_controls.csv` | Sampling-resolution controls |
| `baseline_diagnostics/pair_cell_errors.csv` | RACE × HISPAN and OCC × IND cell contributions |

The main L1 metrics are lower-is-better. Keep AGE domain validity and cohort
support separate, and inspect query-specific joint errors alongside the
categorical average. A large near-ceiling OCC × IND L1 needs the real-real and
shuffle context before a causal interpretation.

## 8. Add the existing diffusion outputs when available

The baseline job does not need old diffusion arrays to fit AIM/MST. However,
the original AGE and pairwise outliers require the actual B/E/C generated
arrays for cell-level diagnosis. An aggregate metric CSV cannot reconstruct
those arrays.

If those arrays already exist, prepare the explicit manifest described in
[the experiment guide](prototype2_diagnostics_baselines.md). It must contain
encoded ages and categorical indices, plus authentic fit/draw identifiers;
C also has a target identifier, while B/E use `target_id: null`.

To include the existing arrays in the initial real job, add the manifest as
the final argument to the command in step 6. If the baseline job has already
completed, run the diagnostics CLI separately inside a compute allocation;
do not refit the baselines merely to add a diagnostic report.

```bash
.venv-pgm/bin/python scripts/run_prototype2_diagnostics.py \
  --data-npz "$PILOT_DIR/pilot_data.npz" \
  --schema "$PILOT_DIR/schema.json" \
  --samples-manifest /absolute/path/to/diffusion_samples_manifest.json \
  --sample-rows 512 --real-real-repeats 20 --shuffle-repeats 10 \
  --out pipeline_runs/diffusion_diagnostics_01
```

If only checkpoints and cached targets exist, recover the original validated
GPU sampling kit and export samples from the frozen models. The current GitHub
snapshot does not contain that GPU kit or an automatic export/replay driver.
The new diagnostic script reads samples; it does not resample checkpoints or
implement feature-specific guidance interventions.

### Diagnose all saved samples from the completed GPU long run

For the verified v1p1 GPU output format, the dedicated launcher prepares the
manifest and evaluates the 15 existing arrays (five arms, three draws each).
It uses the existing `.venv-pgm` environment on a CPU node. Submit it from the
repository root after pulling this branch:

```bash
mkdir -p pipeline_runs/slurm
sbatch --parsable --account=YOUR_ACCOUNT \
  slurm/prototype2_saved_gpu_diagnostics.sbatch \
  /absolute/path/to/completed_gpu_run \
  /absolute/path/to/pilot_data.npz \
  /absolute/path/to/schema.json
```

The helper requires the original input hashes, a compatible one-replicate GPU
plan, and all 15 exact `ARM_generation_NN.npz` filenames. Each file must have
`X_synthetic` with 512 rows and seven columns, the exact `features` order, and
the original `categories`. Nonfinite values and invalid categorical indices
fail validation; finite generated AGE tails are retained for diagnosis.

The manifest's shared `fit_id` identifies the historical matched experiment
block, not a common checkpoint for all arms. C reuses B; E has its own fitted
model. C alone receives the cached-target identifier, and draw IDs follow the
original plan's generation-seed order. Generations remain nested observations
from one training replicate per trained arm.

Outputs are created under `pipeline_runs/diffusion_diagnostics_JOBID/`. A
successful job verifies 15 samples, six matched C-versus-B/E comparisons, no
unmatched comparisons, and unchanged sample hashes between preparation and
evaluation. It writes `COMPLETED.txt` only after these checks and creation of
`aggregate_reports.tar.gz`. Failures produce `FAILED.txt` when the shell can
run its exit trap; never interpret partial outputs as a completed run.

The archive includes the sample manifest, completion-check summary, and the
aggregate JSON/CSV/text reports. It contains no sample arrays, checkpoints,
or target vectors. These exact reference counts remain local nonprivate
research outputs. Download the archive through the same approved file-transfer
route used for the baseline reports.

This job does not train, sample a model, create a target, or refit AIM/MST.
AGE-only permutations and frozen-model guidance interventions are separate
proposed follow-ups; this launcher runs the existing diagnostic evaluator.

## Interpretation and the next replication stage

The new baseline fits provide real-data evidence, but the recorded GPU study
has one trained B/E model per arm and one cached C target. Comparing those
existing means with three baseline fits is an exploratory comparison with
unequal replication. The three old generation draws are not three trained
models. A stronger follow-up requires independently replicated B/E/C fits
and a predeclared treatment of new versus shared target releases, using the
recovered GPU harness.

The same holdout has already informed follow-up hypotheses. Preserve it for
historical comparability, then reserve an untouched evaluation set for a
later confirmatory assessment after tuning. Keep research outputs in the
ignored local output directories and preserve important scratch artifacts
in durable research storage.

## Official UVA references

- [SSH and Open OnDemand access](https://rc.virginia.edu/request-manage/ssz-login)
- [Current login-node and scratch policies](https://rc.virginia.edu/services/our-policies)
- [Current partitions](https://rc.virginia.edu/services/compute-and-storage/aftonrivanna)
- [Published Miniforge module and environment instructions](https://archive.rc.virginia.edu/userinfo/hpc/software/miniforge/)
- [UVA notice about updates and removal of older modules](https://rc.virginia.edu/news/security-updates-following-emudhra-certificate-incident)
- [Interactive jobs, allocation commands, and submission rules](https://learning.rc.virginia.edu/notes/slurm-from-cli/section4/)
- [Job monitoring and completion checks](https://learning.rc.virginia.edu/notes/slurm-from-cli/section3/)
