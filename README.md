# Thermal Comfort Benchmark

Code, data and fixed splits behind the paper *A Multi-Dataset Benchmark for Indoor Thermal Comfort
Prediction* (Building and Environment, submitted). It predicts three occupant votes from indoor, personal
and outdoor variables:

- **TSV-7**: thermal sensation vote, 7 classes (−3 … +3), column `thermal_sensation`;
- **TSV-3**: the same vote on 3 classes (−3…−1 → −1, 0 → 0, +1…+3 → +1), column `TSV_3p`;
- **TPV**: thermal preference vote, 3 classes (want cooler −1, no change 0, want warmer +1), column
  `thermal_preference`.

Every number, table and figure of the paper is recomputed from the outputs of the recorded run. Each
experiment writes a `run_manifest.json` with its command, git commit, environment (`pip freeze`, lock-file
hash), configuration hash, input hashes (data, splits, folds), seeds, per-job time and device, and the
SHA-256 of every output; the run index `manifest.json` of a run directory lists its completed experiments.
Two variables are used below:

- `REF`: the recorded run (directory `rerun_2026-10-XX_v2/` for E1, E2, E4, E7, E7b and E10, and
  `rerun_2026-10-XX_v2_release/` for E6 and E9b, rerun with the code of this release), distributed
  separately as an archive whose link and DOI are given on the Releases page of this repository. Nothing below requires it: the SHA-256 of every recorded output
  are tracked in `reference/expected_outputs_sha256.csv`;
- `REPRO`: your own run directory, for example `REPRO=rerun_reproduction` (`rerun_*/` is git-ignored).

Contents:

1. [What reproduces what](#what-reproduces-what)
2. [Data](#data)
3. [Environment](#environment)
4. [Running the experiments](#running-the-experiments)
5. [Figures, tables and cited numbers](#figures-tables-and-cited-numbers)
6. [Quick identity check on a CPU](#quick-identity-check-on-a-cpu)
7. [Tests](#tests)
8. [Repository layout](#repository-layout)
9. [Citation and licence](#citation-and-licence)

## What reproduces what

| Paper part | Experiment (`run_revision.py` subcommand) | Output folder of a run | Paper floats (labels) |
|---|---|---|---|
| Phase 1: reproduction of Haghirad et al. (2024) on ASHRAE-2018 (2018 v2 release), Random Forest, 12 features | pinned grid, not rerun (see [Phase 1](#phase-1-reproduction-grid-pinned)) | `results/phase1/` (in this repository) | Figure_Table3 + `tab:phase1_haghirad`, Figure_Table4 + `tab:phase1_highcap` |
| Phase 2: in-domain models on ASHRAE-2022 (RF, XGBoost, SVM, ANN, FT-Transformer; 17 features) | `e1` | `E1_phase2/` | Figure_Table5 + `tab:phase2_indomain`, `fig:{tsv7,tsv3,tpv}_confusion_matrices`, `fig:feature_importance` |
| Phase 2: hybrids FTT+RF and FTT+XGBoost (heads on the FT-Transformer embeddings) | `e10` | `E10_hybrids/` | rows of `tab:phase2_indomain`, Figure_Table5 and `fig:nn_agreement_vs_accuracy` |
| Phase 2: heat-balance models PMV and aPMV | `e7` | `E7_pmv/` | Figure_Table6 + `tab:heat_balance_indices` |
| Phase 2: SET, PET (as PTS_SET, PTS_PET) and ePMV | `e7b` | `E7b_indices/` | Figure_Table6 + `tab:heat_balance_indices` |
| Phase 2: leave-one-publication-out evaluation | `e2` | `E2_lopo/` | numbers in the text (`aggregates/E2_aggregates.csv`) |
| Phase 2: contribution of the five outdoor running-mean (EMA) features | `e4` | `E4_ema/` | numbers in the text (`aggregates/E4_ema_gains.csv`) |
| Phase 3: transfer to the French cohorts (zero-shot, local retraining of RF / XGBoost, FTT head-only and full fine-tuning; 20/80 and 80/20 regimes) | `e6` | `E6_phase3/` | Figure_Table7 + `tab:phase3_direct`, `tab:phase3_adaptive` |
| Nearest-neighbour analysis (primary 1-NN, Case B, Case C, each beside its chance reference) | `e9b` | `E9b_nn_cases/` | `tab:nn_disagreement`, `fig:nn_agreement_vs_accuracy` |
| Class distributions of the cohorts | (inputs only) | — | `fig:distributions` |

`docs/PAPER_FIGURES.md` maps every float to its output files and source columns. Only the experiments of
this table are part of the paper; `reference/expected_outputs_sha256.csv` lists their recorded outputs.

## Data

All inputs are in `Data/` and `splits/`. Every experiment checks them against the SHA-256 recorded in the
run manifests, so a modified file is refused.

| File | Rows | Content and source |
|---|---|---|
| `Data/ASHRAE_09_06_2022.csv`, `Data/ASHRAE_09_06_2022_metadata.xlsx` | 109 033 | ASHRAE Global Thermal Comfort Database II, release v2.1.0 (Borealis Dataverse, doi:[10.5683/SP2/GNVEM8](https://doi.org/10.5683/SP2/GNVEM8), version 2), as downloaded |
| `Data/ASHRAE_2022_Clean_api.csv` | 15 224 | ASHRAE-2022 cohort of Phases 2 and 3: the database above after `data_preparation.py` (cleaning, target encoding, Köppen climate) and `feature_extraction_api.py` (outdoor EMA features from Open-Meteo) |
| `Data/ASHRAE_2018_v2.csv` | 17 801 | ASHRAE-2018 cohort of Phase 1: the 2018 release of the database (Dryad, doi:[10.6078/D1F671](https://doi.org/10.6078/D1F671)) in the version used by Haghirad et al. (2024), "2018 v2". The Dryad record version it derives from is not identified (see `DATA_LICENSE.md`) |
| `Data/Moujalled.csv`, `Data/Moujalled_api.csv` | 251 | French cohort (Moujalled); `_api` adds the outdoor EMA features |
| `Data/Hostein.csv`, `Data/Hostein_api.csv` | 163 | French cohort (Hostein); `_api` adds the outdoor EMA features |

The two French cohorts are published with the authorisation of their authors under CC BY 4.0. The ASHRAE
Global Thermal Comfort Database II records are under CC0 1.0 (Borealis and Dryad records, read on 2026-10-11);
the Open-Meteo weather data are under CC BY 4.0. Details and URLs: `DATA_LICENSE.md`.

**Open-Meteo.** `feature_extraction_api.py` queries the Open-Meteo historical weather API. A new query can
return slightly different values, so the regeneration of the `*_api.csv` files is not exactly reproducible.
The `*_api.csv` files used by the run are therefore shipped, and they are the inputs of every experiment.

SHA-256 of the inputs (`sha256sum -c` accepts this block saved as a file):

```text
1bb6b1d84434f0b0604d61ede8304f4b586f5ab649c6fe1b991b0effd6b227f3  Data/ASHRAE_09_06_2022.csv
d16324e2f26d1a8b0e567914acd016a79c4d4433b4f05d9593111e42ffe0eb32  Data/ASHRAE_09_06_2022_metadata.xlsx
ce71980083ce5b2c9e8374d070a5daef25e7805153bfc569691c5cb20f184bc9  Data/ASHRAE_2022_Clean_api.csv
e98712c381dc227a482a19dd1ac1417793fcaf7e817aa466f17c97bf1f31e338  Data/ASHRAE_2018_v2.csv
4117a85c39a0e24c91eebf44140860bdf095452cdf128eb029c94ef93bc96cfb  Data/Moujalled.csv
e13b6be45d225124b9c5fc0d6f3058e9e4a3bebfd24ebec55dcc1662743cb7fd  Data/Moujalled_api.csv
3215760139f0d4069045761d0e70eda2c310fb6662c8cf5ae18e32e19f1f2b10  Data/Hostein.csv
791819bfcd671662bb285b93950fe610fb22c21604dfd738f6931c0d38b64e9f  Data/Hostein_api.csv
8d520cf35363f885d16fcee0f6c588604c12cb30701a3c6f80ce1f19b551bede  results/phase1/experiment_grid_full.csv
eb692cdd9091adb9a4a41d325966ba0513d479fcda1e58f4a0b98ec8e472b859  splits/thermal_sensation_folds.json
49da3c866710d34f62ab611b80ae3ac34936e9a0fdb8c5c92924ff5c4ed17003  splits/TSV_3p_folds.json
10b72f57e58baf7b597a6f7858c2d55211471542d5241279312bf3f293301576  splits/thermal_preference_folds.json
8dc56f0dfd0a479e67451c84e63a4163175912b8f8e1f08629a9bd1976722741  splits/ASHRAE_2022/thermal_sensation_split_indices.json
1c695614a35a4bf11abdfa891dc4d6ed8ac942bd8e5c63f00074cd3430eaf151  splits/ASHRAE_2022/TSV_3p_split_indices.json
8e9634feca57ad140abc839ef06720f17f90f8cf7d5f55581c2945c6d9e8adc8  splits/ASHRAE_2022/thermal_preference_split_indices.json
```

Splits:

- `splits/ASHRAE_2022/*_split_indices.json`: the archived stratified 80/20 train/test split of each target
  (Phase 2); `splits/{target}_folds.json`: the 10 stratified folds of the training rows, shared by every model;
- `splits/phase3/{cohort}_{target}_{regime}.json`: the 20 development/test splits of each French cohort,
  target and regime (Phase 3);
- `splits/ASHRAE_2018/*_split_indices.json`: the 80/20 split of Phase 1.

The categorical values of the cohorts are normalised to one vocabulary when the data are loaded
(`category_normalisation.py`); each experiment writes the mapping it applied
(`category_coverage_*.json`).

## Environment

Python 3.12.3 and the packages of `requirements-revision-2026-10.lock`; details, GPU notes and the unused
configuration keys in `docs/ENVIRONMENT_revision_2026-10.md`.

```bash
git clone <repository URL> thermal-preferences-benchmark && cd thermal-preferences-benchmark
python3.12 -m venv .venv-revision
.venv-revision/bin/python -m pip install --no-deps -r requirements-revision-2026-10.lock \
    --extra-index-url https://download.pytorch.org/whl/cu128
PY=.venv-revision/bin/python
```

- **GPU.** The recorded run used one NVIDIA GeForce RTX 3090 (driver 580.119.02, CUDA 12.8). ANN and
  FT-Transformer (E1) and the FTT fine-tuning (E6) ran on the GPU; RF, SVM and XGBoost ran on the CPU.
- **XGBoost.** The locked xgboost 3.2.0 wheel is built with CUDA. With an NVIDIA driver installed and the GPU
  hidden (`CUDA_VISIBLE_DEVICES=`), every multi-class XGBoost fit fails with `cudaErrorNoDevice`, even on the
  CPU. Run E1, E2, E4, E6 and E10 with the GPU visible. A machine without an NVIDIA driver was not tested.
- **CPU only.** E7, E7b, E9b and the figure regeneration need no GPU.
- **Configuration and lock file.** Every manifest records the SHA-256 of `config.yaml` (file and parsed
  content) and of the lock file. For the release, only comment lines of the two files changed:

  | File | SHA-256 recorded by the run manifests | SHA-256 in this release |
  |---|---|---|
  | `config.yaml` | `771fef24e01ef5fa49dc5d43cb7677bb91c242625c9a12d699db1b465f987f4a` | `49053732523b5105b57ef4951e3bfe91ccadeee846458462852f7d11dd27b30c` |
  | `requirements-revision-2026-10.lock` | `53cbbdfeaea1e91ee42063addefbd6b37da07e56b37557e7a73264ac226943d0` | `6e369df15041ca94cf3ab78946aa1fc29ca28933c4021c4ada8ac1e9ee1776f1` |

  The parsed configuration (`config_used_sha256` `e859f3e226de006d2609a08540d91e65e98e33993c404992d64728242ba6fda1`)
  and the 86 package pins are unchanged. The code maps each recorded hash to the release hash
  (`revision_runners.RELEASE_FILE_SHA256`, checked by `tests/test_folds_seeds.py`); change neither file.

## Running the experiments

Run every command from the repository root of a clean checkout (no modified tracked file): a real run
refuses another working directory, a modified checkout, another configuration or an environment that differs
from the lock file. Use a new run directory `REPRO`. E1 must run first: E4, E6, E7, E7b and E10 read the E1 outputs of the same run directory and
check them against the E1 manifest; E4 also requires the same commit as E1. E2 and E9b are independent.

`--dry_run` runs any subcommand with tiny hyperparameters into `*_dry_run` directories (a smoke check, never
results). The runtimes below are those of the recorded run (`started_utc` to `ended_utc` of each manifest).

### Phase 1: reproduction grid (pinned)

Phase 1 is not rerun. `results/phase1/experiment_grid_full.csv` is the 216-cell Random Forest grid
(3 hyperparameter families × 3 class weightings × 2 oversampling settings × 2 encodings × 2 datasets ×
3 targets) of the earlier run; its SHA-256 is pinned in the code, and every real run and the figure
regeneration refuse another file. It was produced by:

```bash
$PY analysis/haghirad_reproduction.py --datasets all --encodings all --output_dir <new directory>
```

This command has not been rerun in the locked environment; its output is not guaranteed to equal the pinned
grid. Compare: `sha256sum results/phase1/experiment_grid_full.csv` (hash above).

### Phase 2: E1, in-domain models

```bash
$PY run_revision.py e1 --ann_seeds 42 43 44 --ftt_seeds 42 43 44 --run_dir $REPRO
```

- RF, XGBoost, ANN and FT-Transformer with seeds 42, 43 and 44, SVM with seed 42; hyperparameters selected by
  10-fold cross-validation on the training rows, test on the archived split. The ANN and FTT seed defaults are
  42 only: pass the seed options as above to reproduce the paper.
- Recorded: 1 h 54 min, GPU (ANN, FTT) and CPU (RF, SVM, XGBoost).
- Output: `$REPRO/E1_phase2/` (`metrics.csv`, per-model folders with the fitted models, `predictions/`).
- Compare: `aggregates/E1_metrics.csv` of the figure regeneration with the recorded one; the per-vote
  prediction files are listed with their SHA-256 in `E1_phase2/run_manifest.json`. Bit-identity of a full
  rerun was not checked (GPU training of ANN and FTT is not guaranteed to be deterministic across machines).

### Phase 2: E10, hybrids

```bash
$PY run_revision.py e10 --run_dir $REPRO
```

- RF and XGBoost heads on the embeddings of the nine E1 FT-Transformers (3 targets × seeds 42, 43, 44),
  extracted on the CPU with 8 torch threads; no model is refitted. Launch with the GPU visible (XGBoost note
  above); nothing runs on it.
- Recorded: 16 min 21 s, CPU.
- Output: `$REPRO/E10_hybrids/` (`metrics.csv`, `predictions/`, one embedding record per FTT with the SHA-256 of
  the embeddings).

### Phase 2: E7 and E7b, heat-balance models

```bash
$PY run_revision.py e7  --run_dir $REPRO      # PMV, aPMV
$PY run_revision.py e7b --run_dir $REPRO      # SET (PTS_SET), PET (PTS_PET), ePMV; TSV-7 and TSV-3
```

- pythermalcomfort 4.0.2 (the code refuses another version or another `pmv_ppd_iso.py`), on the archived
  test rows of each target, each model on its validity mask, with the E1 predictions on the same rows.
- Recorded: E7 4 s, E7b 52 s, CPU.
- Output: `$REPRO/E7_pmv/` (`metrics.csv`, `coverage.csv`, `apmv_calibration.json`, `provenance.json`,
  `predictions/`) and `$REPRO/E7b_indices/` (`metrics.csv`, `coverage.csv`, `calibration.json`,
  `provenance.json`, `predictions/`).

### Phase 2: E2, leave-one-publication-out

```bash
$PY run_revision.py e2 --run_dir $REPRO
```

- RF and XGBoost (seed 42, the single `config.yaml` hyperparameter point); one fold per publication of the
  database, every vote predicted once.
- Recorded: 49 s, CPU (GPU visible for XGBoost).
- Output: `$REPRO/E2_lopo/` (`lopo_folds.json`, `metrics.csv`, `aggregates.csv`, `predictions/`).

### Phase 2: E4, outdoor EMA features

```bash
$PY run_revision.py e4 --run_dir $REPRO
```

- RF and XGBoost refitted without the five EMA features (F12), paired with the E1 fits on the 17 features
  (F17), seeds 42–44.
- Recorded: 15 s, CPU (GPU visible for XGBoost).
- Output: `$REPRO/E4_ema/` (`pairs.csv`, `metrics.csv`, `predictions/`).

### Phase 3: E6, transfer to the French cohorts

```bash
$PY run_revision.py e6 --splits splits/phase3 --regimes 20-80 80-20 --run_dir $REPRO
```

- The E1 models applied to Moujalled and Hostein: zero-shot on every vote, local retraining of RF and XGBoost
  on the development part of each split, FTT head-only and full fine-tuning (fixed budget: 40 epochs, last
  epoch); 20 splits per cohort, target and regime. The default regime is 20-80 only: pass `--regimes` as above.
- Recorded: 19 min 21 s, CPU (RF, XGBoost) and GPU (FTT fine-tuning).
- Output: `$REPRO/E6_phase3/` (`split_summary.csv`, `e6_run_record.json`, `ftt_adaptation/`, `predictions/`).

### Nearest-neighbour analysis: E9b

```bash
$PY run_revision.py e9b --run_dir $REPRO
```

- Primary 1-NN disagreement, Case B (identical feature vectors) and Case C (near-identical pairs) on the 12
  features of Phase 1 (`features_12`) over the full ASHRAE-2022 cohort, each beside its chance reference.
  Descriptive rates; no model, no split.
- Recorded: 19 s, CPU.
- Output: `$REPRO/E9b_nn_cases/`. Compare with the hashes in [Quick identity check](#quick-identity-check-on-a-cpu).

### A failed real experiment

A failed real experiment (exception, interruption) leaves its directory without `run_manifest.json` and
without an entry in `$REPRO/manifest.json`; nothing reads it, and the same subcommand refuses a non-empty
directory. To retry in the same run directory, move the failed directory aside (for example
`mv $REPRO/E6_phase3 $REPRO/failed/E6_phase3_<UTC time>`), then rerun the subcommand. A completed experiment is
never replaced: use a new run directory.

## Figures, tables and cited numbers

```bash
CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg $PY -m analysis.paper_figures --run_dir $REF/rerun_2026-10-XX_v2 \
    --extra_run_dir $REF/rerun_2026-10-XX_v2_release --data_root . --out <new directory>
```

- Reads only files whose SHA-256 matches the manifest that recorded them, trains nothing and uses no GPU;
  every experiment an item needs must be complete in a run index. Runs in about a minute on a CPU.
- `--extra_run_dir`: E6 and E9b come from the release run, which read the E1 of the first run directory (its
  E1 manifest SHA-256 is checked). For your own complete run: `--run_dir $REPRO` alone.
- Writes the figures (PDF and 300 dpi PNG), the LaTeX tables and their CSV, the captions, and `aggregates/`:
  the metric tables of E1, E6 and E7 and the numbers cited in the text (E2, E4), all recomputed from the
  per-vote prediction files. `aggregates/E1_model_differences.csv` gives the accuracy difference of each
  Phase 2 model from XGBoost with its paired building-cluster bootstrap 95 % interval. `paper_figures_manifest.json` records every input and output with its SHA-256.
- Compare: `python reference/compare_outputs.py --figures <new directory>`. The reference file gives, for
  each file, its relation to the figure package of the recorded run (`PAPER_FIGURES_v2/`): identical; or
  different in the `%` comment lines only (four tables: `table_phase1_haghirad.tex`,
  `table_phase1_highcap.tex`, `table_phase2_indomain.tex`, `table_nn_disagreement.tex`, source notes
  reworded); or new (`aggregates/`, added in this release; in the two E6 aggregate tables the note columns
  were reworded, numbers identical). Two regenerations from the same run are byte-identical (no date or
  software metadata in the PDF and PNG files).

Options and the float-by-float mapping: `docs/PAPER_FIGURES.md`.

## Quick identity check on a CPU

These checks need no GPU and no recorded run: the reference SHA-256 are tracked in
`reference/expected_outputs_sha256.csv` (`python reference/compare_outputs.py --help`). Run them in a fresh
clone, with a new run directory.

1. **E9b** (about 20 s):

   ```bash
   CUDA_VISIBLE_DEVICES= $PY run_revision.py e9b --run_dir $REPRO
   $PY reference/compare_outputs.py --run $REPRO --experiments E9b
   ```

2. **E2, Random Forest only** (under a minute; XGBoost is left out because of the CUDA build, see
   [Environment](#environment)); the 30 RF prediction files must equal the recorded ones:

   ```bash
   CUDA_VISIBLE_DEVICES= $PY run_revision.py e2 --families RF --run_dir $REPRO
   $PY reference/compare_outputs.py --run $REPRO --experiments E2 --match __RF__
   ```

   `--match __RF__` keeps the RF prediction files only (the metric and aggregate files of E2 mix both models).

3. **Figures and tables** (needs `REF`): the command of the previous section, then
   `$PY reference/compare_outputs.py --figures <new directory>`.

How the reference file is read (`reference/compare_outputs.py`):

- Each row is `expected` or `superseded`. An `expected` file must exist and match; a `superseded` row is a
  recorded output that the release does not write (its note says why: the E6 strategies of the recorded
  run that are not in the paper, their FTT adaptation records, and the E6 run record). Superseded rows are
  listed, never verified, and never make a check pass.
- Most rows compare the file bytes. Two declared exceptions compare the content: the E6
  `split_summary.csv` on the columns that the release writes (`csv_columns:`), and the three E9b detail
  files without the fields `analysis` and `source` (`json_without:`), which the release words differently.
- The last line is `RESULT: COMPLETE` (exit status 0), `RESULT: PARTIAL` (exit status 2, only with
  `--partial`: every file present matches but some are absent; the coverage is printed) or `RESULT: FAIL`
  (exit status 1: a mismatch, an absent file, an absent directory, an empty selection or nothing compared).
- The E6 retained predictions (zero-shot, local retraining, FTT head-only, FTT full) are compared with the
  hashes of the recorded run; the release run of E6 must reproduce them.

## Tests

```bash
CUDA_VISIBLE_DEVICES= $PY -m pytest -q -rs
```

The tests use small synthetic data and the shipped inputs; they never read a recorded run. With the GPU hidden
the tests that need XGBoost multi-class fits or a CUDA device are skipped (`-rs` prints each reason).
`tests/test_environment.py` fails on purpose outside the locked environment.

## Repository layout

| Path | Role |
|---|---|
| `run_revision.py` | entry point: one subcommand per experiment |
| `revision_runners.py` | experiments E1, E2, E4, E7, E7b, E9b, E10 and the manifests |
| `phase3_runner.py`, `transfert.py`, `ftt_adaptation.py` | Phase 3 (E6) |
| `models.py`, `models_classic.py`, `models_deep.py`, `hybrid.py` | models, cross-validation, hybrids |
| `indicateurs_classiques.py` | PMV, aPMV, SET, PET, ePMV (E7, E7b) |
| `revision_folds.py`, `revision_stats.py`, `stats_utils.py` | folds, seeds, metrics, bootstrap intervals |
| `compute.py`, `compute_scheduler.py` | CPU / GPU scheduling of the fits |
| `config.yaml`, `config_loader.py` | hyperparameters, feature sets, datasets |
| `data_preparation.py`, `feature_extraction_api.py`, `category_normalisation.py`, `data_groups.py`, `unified_data_loader.py` | data preparation and loading |
| `analysis/nn_cases.py` | nearest-neighbour analysis (E9b) |
| `analysis/haghirad_reproduction.py` | Phase 1 grid |
| `analysis/paper_figures/`, `analysis/feature_importance_heatmap.py` | figures and tables of the paper |
| `Data/`, `splits/`, `results/phase1/` | inputs (see [Data](#data)) |
| `tests/` | tests |
| `docs/` | `PAPER_FIGURES.md` (float mapping), `ENVIRONMENT_revision_2026-10.md` (environment) |
| `reference/` | SHA-256 of the recorded outputs and the comparison script |

## Citation and licence

Citation: the article is under review at *Building and Environment*; its reference will be added here on publication. Until then, cite this repository (authors: Molka Grayaa, Damien Bouchabou, Bassam Moujalled).

Licence: the code is under the MIT licence (`LICENSE`). The French cohorts and the recorded results are under
CC BY 4.0; the ASHRAE database files keep the CC0 1.0 licence of their source records (`DATA_LICENSE.md`).
