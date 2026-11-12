# Prototype 2: diagnostics and AIM / Private-PGM baselines

This adds two executable research workflows:

1. Diagnose AGE support and histogram errors, RACE × HISPAN cell errors, and
   OCC × IND sparsity using prepared data and saved generated arrays.
2. Fit **AIM + Private-PGM** and **MST + Private-PGM** on the same training
   records, then evaluate their samples through the common pilot evaluator.

Private-PGM is the inference component of both methods. “PGM” alone is not a
fully specified private synthesizer; this comparison uses the MST measurement
and private tree-selection mechanism as the second baseline.

## What is needed to run on the research data

GitHub contains source code, not the local research inputs. The baseline run
requires these two existing files, with their original pairing preserved:

| Input | Required contents |
| --- | --- |
| `pilot_data.npz` | `X_train`, `X_test`, and the six public `categories` cardinalities |
| `schema.json` | Feature order, AGE transform, and all categorical index-to-code domains |

Arrays use the established order `AGE, SEX, RACE, HISPAN, EDUCD, OCC, IND`.
AGE is encoded as `(AGE - 50) / 50`; categorical entries are contiguous
indices into the schema, not the original IPUMS numeric codes. The scripts
validate the complete input and never infer a smaller domain from observed
unique values.

For the primary comparison with the recorded Rivanna run, confirm the original
2,048 training and 512 held-out records, their file hashes, and the matching
schema before interpreting results. A smaller CPU pilot or a newly prepared
split is a separate experiment even if its feature names match.

To diagnose the existing diffusion outputs, also supply their raw encoded
`.npy` arrays or `.npz` arrays such as `X_synthetic`, with explicit fit, target,
and generation identifiers. If raw arrays were not retained, first replay the
saved checkpoints and cached targets in the original environment. A replay is
not claimed to be bitwise identical merely because its histograms match.
The diagnostic script does not create a new target or train a model.

Keep real records, checkpoints, targets, per-cell reports, and generated output
under ignored local directories such as `pipeline_runs/`. Do not commit them.
These reference-data analyses and research implementations are not an audited
end-to-end private release.

## Isolated baseline environment

The baseline environment does not need PyTorch, Opacus, CUDA, or a GPU. Use
Python 3.11 or 3.12 and an isolated environment. The official source revision
and the compatible JAX/NetworkX versions are pinned; the experiment records
the imported source path and actual package versions.

```bash
python3 -m venv .venv-pgm
. .venv-pgm/bin/activate
git clone https://github.com/ryan112358/mbi.git ../raise-mbi
PGM_REPO="$(cd ../raise-mbi && pwd)"
git -C "$PGM_REPO" checkout --detach 5f97065a17b3e53cfefc7a8ce3e293fd153dd064
python -m pip install -e "$PGM_REPO" -r requirements-pgm.txt
```

The runner verifies the official repository origin, exact commit, clean
tracked source, expected API, and editable import location. It will reject an
accidental local `private_pgm` resampling stub or an incompatible installation.

The pinned official MST script still uses the older `Domain.attrs` API. The
local adapter preserves its compression/decompression and private selection
logic while converting data through the current MBI API and retaining the
fitted model. The official checkout is not patched. Adapted routines and their
source are documented in `licenses/MBI-NOTICE.md`, with the upstream Apache
license included alongside it.

## 1. Run the diagnostics

Create a local sample manifest. Paths are relative to the manifest file.
Control arms have no private target, so their `target_id` is `null`.
A top-level list or an object containing `samples` is accepted. `.npz` files
require an explicit array `key`; `.npy` files permit an omitted or null key.

```json
{
  "samples": [
    {"arm": "B_private", "fit_id": "gpu_fit_1", "target_id": null,
     "draw_id": 1, "path": "B_draw_1.npz", "key": "X_synthetic"},
    {"arm": "E_matched", "fit_id": "gpu_fit_1", "target_id": null,
     "draw_id": 1, "path": "E_draw_1.npz", "key": "X_synthetic"},
    {"arm": "C", "fit_id": "gpu_fit_1", "target_id": "target_1",
     "draw_id": 1, "path": "C_draw_1.npz", "key": "X_synthetic"}
  ]
}
```

```bash
python scripts/run_prototype2_diagnostics.py \
  --data-npz /path/to/pilot_data.npz \
  --schema /path/to/schema.json \
  --samples-manifest /path/to/samples_manifest.json \
  --sample-rows 512 \
  --real-real-repeats 20 \
  --shuffle-repeats 10 --seed 20261112 \
  --out pipeline_runs/diagnostics_01
```

The output directory must be new. Omitting the sample manifest runs the
prepared-data and real-versus-real checks only; it cannot diagnose absent
generated outputs. `--sample-rows` sets the size of each side of the real-real
controls. It does not truncate or resample supplied generated arrays, which
are evaluated in full against the complete supplied holdout.

The output files are:

| File | Contents |
| --- | --- |
| `diagnostics.json` | Input/source hashes, AGE/support audits, per-draw results, pairing decisions, and all-domain cell totals |
| `metrics_per_draw.csv` | Original and shuffled metrics with explicit fit, target, and draw identifiers |
| `age_bin_contributions.csv` | Every historical AGE bin's contribution to each fixed AGE one-, two-, and three-way query, before and after clamping |
| `pair_cell_errors.csv` | Signed joint differences and interaction decomposition for RACE × HISPAN and OCC × IND |
| `matched_cell_changes.csv` | Cell-level C-minus-control probability and error changes for unambiguous matches |
| `real_real_controls.csv` | Per-repetition real-real original and shuffled errors |

Large pair tables emit the occupied union of cells to CSV. Their JSON totals
still include every declared cell, with omitted product/interaction
contributions reported explicitly. No raw rows are copied to these reports.
A failed run writes `failure.json`; any partial files in that directory must
not be treated as a completed analysis.

### AGE interpretation

- Check actual cohort support and encoding round-trip before changing a model.
- Preserve the historical fixed histogram bins for comparable scores, and
  report outside-`[18,65]` and outside-`[0,100]` separately.
- The existing `[65,75)` bin mixes valid age 65 with ages 66–74 outside the
  selected cohort. Histogram fidelity alone cannot certify cohort validity.
- Clamping can move mass between bins with zero reference mass, leaving L1
  unchanged. Per-bin contributions expose this directly.
- The runner reports quantiles and extreme values alongside histogram scores.
  A continuous distance measured in years would be an additional diagnostic;
  the current runner does not calculate one. A boundary clamp is a validity
  control, not evidence that AGE was learned.

Nonfinite values and invalid categorical codes cause a clear failure; they
are not filtered out or retried. The original row denominator is retained.

### Pairwise interpretation

RACE × HISPAN reports both full joint errors and interaction residuals, plus
cell contributions to C-minus-B and C-minus-E error changes. These are
different quantities: improved one-way errors and a lower interaction-error
norm do not guarantee a lower full joint L1. Comparisons require an explicit,
unambiguous fit/draw match and equal row counts. C retains its target identifier,
while B/E controls need no target identifier. A unique control can serve several
C target variants, with that reuse recorded as dependent comparisons.
Ambiguous or missing controls remain unpaired with a stated reason. Positive
C-minus-control error changes mean C is worse; shared controls are not
independent replications.

OCC × IND reports support/overlap and compares original with shuffled data.
Same-size real-versus-real controls use disjoint subsamples of the training
pool. Repeated subsamples are dependent diagnostics of sampling variation,
not new model fits or an attainable lower bound on error. If the pool is too
small for two disjoint samples, reduce the declared sample size explicitly;
the script does not silently switch to bootstrap sampling.

The runner examines supplied outputs. It does not establish a causal model
explanation from summary statistics alone. A later frozen-model ablation can
toggle RACE and HISPAN guidance while reusing the same cached targets and
holding the other guidance settings fixed.

## 2. Fit the AIM and MST baselines

```bash
python scripts/run_ipums_pgm_baselines.py \
  --data-npz /path/to/pilot_data.npz \
  --schema /path/to/schema.json \
  --mbi-repo "$PGM_REPO" \
  --expected-mbi-commit 5f97065a17b3e53cfefc7a8ce3e293fd153dd064 \
  --arms aim mst \
  --fits 3 --draws 3 --samples 512 \
  --epsilon 2.5638977279554265 --delta 1e-5 \
  --out pipeline_runs/pgm_baselines_01
```

The baseline privacy budget is the total for one fitted mechanism, including
its private measurements and selection. It is not a training budget plus a
new extra epsilon-one target. New independent fits are new mechanisms;
generating multiple samples from a fixed fit does not refit the mechanism.
Private random seeds are not written to metadata. Public generation seeds
are recorded separately. The fit API receives training records only.

The default numerical representation uses the public integer AGE domain
`0,...,100`. This deliberately preserves exact integer ages in the input;
diffusion remains continuous. All methods are evaluated using the same
historical AGE bins. Validity guaranteed by this discrete domain is not
reported as learned continuous-age fidelity. The optional `legacy-bins`
representation uses fixed public bin midpoints on decoding and cannot support
claims about learned within-bin age variation.

The coarse representation also changes the meaning of cohort support: age 65
falls in the historical `[65,75)` bin, whose decoded midpoint is 69.5. It can
therefore create an outside-18–65 value from an in-cohort record. Use the
default integer representation for the primary cohort-validity comparison.

### Workload and evaluation contract

- AIM receives all 21 equally weighted pairs among the seven attributes, with
  singleton candidates supplied through its downward closure. The existing
  three AGE triples remain evaluation queries outside the pairwise workload.
- The adapter does not silently use the official CLI's 10,000-cell workload
  filter. OCC × IND has 141,246 declared cells and remains in the default
  requested workload. Any explicit resource exclusions are recorded. AIM's
  model-size feasibility restrictions and selected cliques are also recorded;
  being requested does not mean every pair will be measured.
- MST selects its own private tree and records it. Its noisy singleton-based
  domain compression is retained; the same fitted model and decompression map
  are reused for every generation draw.
- Both use explicit `method="sample"` generation. The official randomized
  rounding default is not silently mixed with diffusion's sampled records.
- Scores come from the shared holdout evaluator, not the official AIM CLI's
  training-data half-L1 output.
- Reports preserve fit/draw identities and first average within fit. An SD
  across generation draws is not labeled between-fit uncertainty. Comparing
  new repeated baselines to a single existing GPU fit remains exploratory.

Per-fit fitting and per-draw generation times are recorded. The first fit
includes cold JAX compilation; later fits can benefit from compilation already
performed in the same process. These timings are not claimed to be controlled
CPU-versus-GPU speed comparisons.

## Software validation using invented data

The following runs the real algorithms on generated test records. The public
fixture has the pilot's categorical cardinalities and exercises the large
OCC × IND workload, but its values and relationships are artificial.

```bash
python scripts/make_prototype2_public_fixture.py \
  --out pipeline_runs/public_fixture_01
python scripts/run_prototype2_diagnostics.py \
  --data-npz pipeline_runs/public_fixture_01/pilot_data.npz \
  --schema pipeline_runs/public_fixture_01/schema.json \
  --samples-manifest pipeline_runs/public_fixture_01/samples_manifest.json \
  --sample-rows 64 --real-real-repeats 3 \
  --out pipeline_runs/public_diagnostic_check_01
PYTHONHASHSEED=0 python scripts/run_ipums_pgm_baselines.py \
  --data-npz pipeline_runs/public_fixture_01/pilot_data.npz \
  --schema pipeline_runs/public_fixture_01/schema.json \
  --mbi-repo "$PGM_REPO" \
  --expected-mbi-commit 5f97065a17b3e53cfefc7a8ce3e293fd153dd064 \
  --arms aim mst --fits 1 --draws 2 --samples 64 \
  --public-fixture --smoke \
  --out pipeline_runs/public_baseline_check_01
```

`--smoke` changes inference iteration and model-size settings and is restricted
to explicitly marked public fixtures. Its results must never enter the IPUMS
research comparison. The fixed `PYTHONHASHSEED` is needed for reproducible
fixture runs because the upstream AIM candidate ordering depends on Python
sets; a fixed NumPy seed alone is insufficient across processes. Real-data
fits still use secret randomness, independently of this public hash seed.
Run the focused unit tests with:

```bash
python -m pip install pytest
python -m pytest tests -q
```

### Validation completed on 2026-11-12 UTC

The focused suite passed **47 tests and 15 subtests** in Python 3.12. The
generated-fixture diagnostics completed for eight supplied arrays plus three
64-versus-64 disjoint real-real controls. All eight pairwise cell-error change
totals matched their aggregate joint-L1 changes exactly. All 80 AGE-query
groups retained the 11 historical bins and unit probability mass.

A complete run of the actual AIM and MST mechanisms also passed, using the
fixture's 256 training records, 128 held-out records, and the full declared
categorical domains. Each arm fitted once and generated two separate draws
of 64 records. All 21 AIM pairs were requested, including OCC × IND, with no
explicit cell-cap exclusions. The smoke settings were AIM 1 MB / 25 inference
iterations and MST 25 selection / 50 final inference iterations; these are
**software checks, not the standard research configuration**. Fitting took
43.85 seconds for AIM and 15.99 seconds for MST on the local CPU, including
first-use compilation. These timings are not controlled performance results.

Independent compatibility checks also matched the official MST selection
logic and RNG consumption over 50 public seeds, and matched compression and
decompression over 21 support patterns and 210 generation seeds. The official
MST reference for the latter check changed only `.attrs` to `.attributes` to
accommodate modern MBI. Earlier partial compatibility attempts were excluded
from the completed comparison.

No real IPUMS records, original diffusion sample arrays, checkpoints, or
cached targets were available for these checks. They therefore establish
software execution and metric accounting only; they do not answer which
method performs best on the research data or establish the causes of the
original AGE/RACE × HISPAN/OCC × IND observations.

## Sources and provenance

- [AIM paper](https://arxiv.org/abs/2201.12677)
- [MST / Private-PGM synthesis paper](https://arxiv.org/abs/2108.04978)
- [Private-PGM inference paper](https://proceedings.mlr.press/v97/mckenna19a.html)
- [Pinned official AIM implementation](https://github.com/ryan112358/mbi/blob/5f97065a17b3e53cfefc7a8ce3e293fd153dd064/mechanisms/aim.py)
- [Pinned official MST implementation](https://github.com/ryan112358/mbi/blob/5f97065a17b3e53cfefc7a8ce3e293fd153dd064/mechanisms/mst.py)

The existing CPU/core source files at the base research commit
`1ead9d5f82a7359092e56b4dca8d9c0f6d9e6315` match the corresponding original-source
hashes recorded in the November 9 Rivanna plan. The GPU preflight/launch helper,
real input arrays, and original trained artifacts are not included in that
GitHub snapshot. This source match does not establish byte identity of any
missing runtime input or output.
