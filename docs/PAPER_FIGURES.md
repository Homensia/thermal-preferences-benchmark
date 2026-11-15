# Paper figures and tables: regeneration from the run outputs

Package: `analysis/paper_figures/` (tests: `tests/test_paper_figures.py`). It regenerates every figure and
table of the paper, and the numbers cited in the text, from the outputs of the recorded run (`RUN`). It reads
only files whose SHA-256 matches the manifest that recorded them:

- the run index `RUN/manifest.json` and each `RUN/<experiment>/run_manifest.json`, which cover outputs and inputs;
- the per-vote prediction files of E1, E2, E4, E6, E7 and E10, from which the aggregate tables are recomputed
  (`analysis/paper_figures/aggregate.py`, item `aggregates`);
- the cohort CSV `Data/ASHRAE_2022_Clean_api.csv`, checked against its pinned SHA-256
  (`data_groups.PINNED_SHA256`);
- the pinned Phase 1 grid `results/phase1/experiment_grid_full.csv`.

It never trains a model and never uses the GPU. It writes only into a new or empty `--out` directory, creates
each file exclusively, and checks every label, caption and file name against a list of banned wordings
(`wording.py`).

## Command

From the repository root (the inputs `Data/` and `splits/` are in the repository; `--data_root` defaults to it):

```bash
CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg $PY -m analysis.paper_figures --run_dir $RUN --data_root . --out <new directory>
```

Every experiment an item needs must be complete in the run index, otherwise nothing is written. To build a
single item, add `--only <item>` (with a new `--out`); `--list` prints the items and their output files.

`--extra_run_dir DIR` (repeatable) adds a later run directory whose completed experiments replace those of
`--run_dir`. The paper uses it for E6 and E9b, rerun with the release code against the recorded E1:
`--run_dir <recorded run> --extra_run_dir <release run>`. E1 always comes from `--run_dir`; an extra
experiment that records `e1_manifest_sha256` must have read that E1, and the Phase 1 pin of both run
indices must agree. The manifest lists each extra run directory, its index SHA-256 and the experiments it
replaces.

Every output directory receives `paper_figures_manifest.json`: the mode, the command, the code commit, every
input with its SHA-256, every output with its SHA-256. Two regenerations from the same run give byte-identical
outputs (the PDF and PNG files carry no date or software metadata). If a library compiles a numba JIT cache,
its directory must be writable: the package defaults `NUMBA_CACHE_DIR` to `$TMPDIR/tpb_numba_cache`.

`--preview` builds the items while some experiments are still missing from a run: their rows are left out,
each LaTeX table that lost rows carries a `% PENDING` comment, the manifest records `"mode": "preview"`, and the
output directory must lie outside the run directory. It is not needed with a complete run.

Tests (synthetic run directory, no recorded output read):

```bash
CUDA_VISIBLE_DEVICES= $PY -m pytest tests/test_paper_figures.py -q
```

## Mapping

| Paper float (label) | Item | Output files | Source |
|---|---|---|---|
| `fig:distributions` (Method) | `distributions` | `distributions/{tsv7,tsv3,tpv}_{ashrae,moujalled,hostein}.{pdf,png}`, `class_counts.csv`, `captions/distributions.tex` | cohort CSV (pinned SHA-256, the input of E1 and E9b) and the French `_api.csv` files (inputs of E6); French counts checked against the E6 support |
| Figure_Table3 + `tab:phase1_haghirad` | `phase1` | `phase1_haghirad_bars.{pdf,png}`, `table_phase1_haghirad.{tex,csv}`, `captions/phase1_haghirad_bars.tex` | `results/phase1/experiment_grid_full.csv` (pinned SHA-256; not rerun) |
| Figure_Table4 + `tab:phase1_highcap` | `phase1` | `phase1_highcap_bars.{pdf,png}`, `table_phase1_highcap.{tex,csv}`, `captions/phase1_highcap_bars.tex` | same |
| Figure_Table5 + `tab:phase2_indomain` | `phase2` | `phase2_indomain_bars.{pdf,png}`, `table_phase2_indomain.{tex,csv}`, `table_phase2_indomain_per_seed.csv`, `captions/phase2_indomain_bars.tex` | `aggregates/E1_metrics.csv` (one row per seed, from the E1 prediction files); E10 `metrics.csv` |
| `fig:{tsv7,tsv3,tpv}_confusion_matrices` | `confusion_matrices` | `confusion_matrices/{target}__{RandomForest,XGBoost,SVM,ANN,FTTransformer}_confusion_matrix.{pdf,png}`, `confusion_matrices.csv`, `captions/{tsv7,tsv3,tpv}_confusion_matrices.tex` | `aggregates/E1_confusion_matrices.csv` (checked against `E1_metrics.csv`); seed 42 by default |
| Figure_Table6 + `tab:heat_balance_indices` | `indices` | `heat_balance_indices_bars.{pdf,png}`, `table_heat_balance_indices.{tex,csv}`, `captions/heat_balance_indices_bars.tex` | `aggregates/E7_metrics.csv` (PMV, aPMV, primary variant, on the validity mask); E7b `metrics.csv` (ePMV, PTS_SET, PTS_PET); TSV-7 and TSV-3 |
| `fig:feature_importance` and the per-model importance bars | `feature_importance` | `feature_importance/feature_importance_heatmap.{pdf,png,csv}`, `feature_importance/{target}_{RandomForest,XGBoost}_feature_importance.{pdf,png}`, `feature_importance_per_seed.csv`, `captions/feature_importance_{heatmap,bars}.tex` | E1 model files `{target}/{RF,XGB}/seed_{s}/*.joblib`; importances normalised per seed and averaged over seeds 42–44 |
| Figure_Table7 + `tab:phase3_direct` | `phase3` | `phase3_zero_shot_bars.{pdf,png}`, `table_phase3_zero_shot.{tex,csv}`, `captions/phase3_zero_shot_bars.tex` | `aggregates/E6_full_cohort_zero_shot.csv`; row **Majority class** = the most frequent vote of the cohort predicted for all its votes (descriptive); the CSV also holds the majority of the ASHRAE training votes |
| `tab:phase3_adaptive` | `phase3` | `table_phase3_adaptive.{tex,csv}` | `aggregates/E6_strategy_metrics.csv`: local retraining of RF and XGBoost (refitted on the development part of each split), FTT head-only and full fine-tuning, pooled over the 20 splits of each regime; row **Majority class (DEV)** = the most frequent vote of each development part predicted for its test votes |
| `fig:nn_agreement_vs_accuracy` | `nn` | `nn_agreement_vs_accuracy.{pdf,png}`, `nn_agreement_vs_accuracy_data.csv`, `captions/nn_agreement_vs_accuracy.tex` | E1 and E10 seed means (whiskers: range over seeds); red line = 1 − primary 1-NN disagreement of E9b; grey line = chance agreement over the cohort votes |
| `tab:nn_disagreement` | `nn` | `table_nn_disagreement.{tex,csv}` | E9b summary and detail files, recomputed from the E9b cohort input: primary 1-NN disagreement, Case B group agreement, Case C pair agreement, each beside its chance reference |
| numbers in the text | `aggregates` | `aggregates/E2_aggregates.csv` (leave-one-publication-out, micro and macro, with both majority baselines), `aggregates/E4_ema_gains.csv` (paired F17 − F12 accuracy gain, mean over seeds 42–44, building-cluster bootstrap 95 % interval), `aggregates/E1_model_differences.csv` (paired accuracy difference of each Phase 2 model from XGBoost, same bootstrap) | E2, E4, E1 and E10 prediction files |
| `fig:pipeline`, `fig:transfer_strategy`, `tab:datasets`, `tab:hyperparameters`, `tab:residual_votes` | — | — | drawn or written by hand; no data to regenerate |

## Aggregate tables

Item `aggregates` recomputes, with `revision_stats`, from the hash-verified prediction files:

- `E1_metrics.csv`, `E1_confusion_matrices.csv`: one row per target × model × seed on the archived test votes
  (seeds 42, 43, 44; SVM 42 only; a run with other seeds is refused);
- `E6_full_cohort_zero_shot.csv`: zero-shot models on every vote of each French cohort, with the ASHRAE
  training majority and the cohort majority;
- `E6_strategy_metrics.csv`: each Phase 3 strategy pooled over the 20 splits of each regime, with the majority
  baselines;
- `E7_metrics.csv`: PMV and aPMV on their validity mask;
- `E2_aggregates.csv`, `E4_ema_gains.csv`: see the mapping;
- `E1_model_differences.csv`: accuracy of each Phase 2 model (RF, SVM, ANN, FTT; FTT+RF and FTT+XGB from the
  E10 prediction files) minus that of XGBoost on the archived test votes, each model averaged over its seeds
  (SVM: seed 42 only), with the paired building-cluster bootstrap 95 % interval of the E4 gain (B = 10 000,
  seed 20261008).

Each prediction file is checked before use: columns, `row_idx` equal to the split rows, `y_true` equal to the
cohort votes.

## Figure conventions

- **Print size.** Every figure is drawn at the width at which the manuscript includes it, so that no text
  prints below 7 pt: bar charts and the NN figure at the text width (510 pt in the 5p layout), confusion
  matrices and per-model importance bars at `0.49\linewidth`, distributions at `0.32\linewidth`. Every figure
  is a vector PDF and a 300 dpi PNG. The importance heatmap (`analysis/feature_importance_heatmap.py`) keeps
  its own size.
- **No ranking by typography.** No table sets a model in bold. Differences between models are assessed with
  paired bootstrap intervals, cited in the text.
- **One name per model and dataset** (`metrics.MODEL_NAME`, `metrics.DATASET_NAME`): RF, XGBoost, SVM, ANN,
  FTT, FTT+RF, FTT+XGBoost; ASHRAE-2018 (2018 v2 release), ASHRAE-2022, Moujalled, Hostein. File names keep the
  model spellings of the manuscript (`RandomForest`, `FTTransformer`).
- **Captions.** Each figure gets `captions/<figure stem>.tex` (one `\caption{...}\label{fig:...}` and a comment
  with the inclusion width). They state the vote codes, the coverage of each heat-balance model, the QWK = 0 of
  a constant predictor, the seed handling, the shared colour scale of the confusion matrices, impurity-based
  (RF) versus gain-based (XGBoost) importance and, for the NN figure, that the whiskers are the range over the
  model seeds (not a confidence interval; SVM has one seed).
- **Distributions.** Every class of the scale is drawn, an empty one as a zero bar, with one colour per vote
  value (blue = cold side or "want cooler", grey = neutral or "no change", red = warm side or "want warmer").
- **Feature names.** `Âge` and `Sexe` remain in `feature_importance_*.csv` and in the `variables` column of
  `table_nn_disagreement.csv` (they are the model feature names); the figures show `age` and `gender`.

Code origin: `nn_agreement_vs_accuracy.py` and `tables.py` derive from the plotting and table scripts of the
earlier version of this repository; `bar_charts.py` and `build_distributions` reproduce the layout and colours
of the paper's bar charts (Figure_Table3 to Figure_Table7) and class-count charts, which had no Python code;
the confusion-matrix and importance layouts follow `models.py`.

## The nearest-neighbour analysis (E9b) and its chance references

E9b (`analysis/nn_cases.py`, `run_revision.py e9b`) computes the nearest-neighbour analysis of the paper on the
12 features of Phase 1 (`features_12`) over the full ASHRAE-2022 cohort: every vote is a query and its
neighbour is searched among the other votes. The definitions are those of the earlier version of the code
(frozen copy in `tests/data/snapshot_nn_disagreement.py.txt`, checked by the tests), with two corrections: the
data go through the category normalisation of the other experiments (a missing value raises), and in the
primary analysis the query is excluded by its row position (the earlier code assumed that the first returned
row was the query, which fails for exact duplicates).

- **Primary**: nearest neighbour (k = 1), Euclidean distance after integer coding of the categorical variables
  and standardisation of every column; disagreement = share of the votes that differ from their neighbour's
  vote; chance = `1 − Σ_c p_c²`, `p_c` the class shares of the cohort votes.
- **Case B** (groups of identical feature vectors, numerical variables rounded to 3 decimals; group agreement
  = share of groups whose votes all agree): chance = mean over the groups of `Σ_c q_c^{n_g}`, `q_c` the class
  shares of the grouped votes, `n_g` the group size.
- **Case B pairs and Case C** (pairs within Tair ±1 °C, Tout ±1 °C, RH ±5 %, air speed ±0.3 m/s, clo ±0.3,
  met ±0.2, equal categorical variables and age): chance agreement = `Σ_c p_c²`, `p_c` the class shares of the
  paired votes, each vote counted once per pair it enters.

Every rate is descriptive; none is a limit on the accuracy of a model. In `nn_agreement_vs_accuracy` the
primary value is drawn as the agreement `1 − D_1` (red dashed line) beside the chance agreement (grey dotted
line).

## Options

- `--cm_seed 42|43|44|sum`: model seed of the confusion matrices (default 42; `sum` adds the counts of the
  three seeds).
- `--fi_seeds`: seeds averaged in the feature-importance figures (default 42 43 44).
- `--nn_cases_in_figure`: also draw Case B and Case C in `nn_agreement_vs_accuracy`, each with its own chance
  reference (default: table only, as in the paper).
- `--indices_tpv`: add a TPV panel with PMV and aPMV to the heat-balance items. Not the paper scope (the paper
  evaluates the indices on TSV-7 and TSV-3 only).
