#  🌡️Thermal Comfort Banchmark
This project focuses on data-driven modeling of human thermal comfort using the ASHRAE Global Thermal Comfort Database II (2022 release) and two independent French datasets (Moujalled and Hostein).

It evaluates multiple machine-learning and deep-learning models for predicting three comfort-related targets:

🎯Thermal Sensation Vote (TSV-7) — 7-point scale
🎯Thermal Sensation Vote (TSV-3) — 3-point scale
🎯Thermal Preference Vote (TPV) — 3-point scale

The pipeline includes:

✔ data cleaning & preprocessing
✔ model training (classical ML, deep learning, hybrid models)
✔ cross-dataset transfer learning
✔ fine-tuning experiments
✔ unified evaluation & visualization

Results are compared using accuracy, F1-scores, precision, recall, and confusion matrices, emphasizing generalization and robustness across climatic and building contexts.

## Implemented models

#### 🌳 **Classics (Scikit-learn)**
- Random Forest
- SVM 
- XGBoost

#### 🧠 **Deep Learning (PyTorch)**
- ANN 
- FTTransformer

#### 🔀 **Hybrids**
- FT embeddings → Random Forest
- FT embeddings → XGBoost




## 📁 Project Structure

thermal-preferences-benchmark/
├── models.py                    # Main Orchestrator
├── models_classic.py            # Classic models (RF, XGBoost, SVM)
├── models_deep.py               # deep learning models (ANN, FTTransformer)
├── hybrid.py                    # hybrids models
├── unified_data_loader.py       # 
├── transfert.py                 # External tests
├── data_preparation.py          # data preprocessing
├── feature_extraction_api.py    # Weather features extraction from api
├── indicateurs_classiques.py    # 
│
├── Data/
│   ├── ASHRAE_09_06_2022_metadata.xlsx
│   ├── ASHRAE_09_06_2022.csv
│   ├── Moujalled.csv
│   └── Hostein.csv
└── requirements.txt
└── README.md.txt 



📦 Configuration System

The project  includes a centralized configuration file:

```bash
python data_preparation.py 
```
This file defines:
✔ dataset paths
✔ model hyperparameters (classical, deep, hybrid)
✔weather API extraction parameters
✔ parameters for transfer learning
✔ general settings 

The configuration is loaded using:

```bash
from config_loader import load_config
CONFIG = load_config()
```

## Prerequisites

```bash
python --version

# Virtual environment (recommended)
python -m venv thermal_comfort_env
thermal_comfort_env\Scripts\activate     
```

## 💻 Installation

```bash
# Clone repository
git clone <repository-url>
cd thermal-preferences-benchmark

# Install dependencies
pip install -r requirements.txt
```

## 📊 Phase 1 - Data Preparation 
### 1️⃣ Data Cleaning & Normalization (data_preparation.py)

This step loads the raw thermal-comfort dataset (ASHRAE) and applies:

✔ Standardization of column names
✔ Target encoding (TSV-7 → TSV-3, TPV mapping)
✔ Numerical / categorical harmonization
✔ Removal of invalid or incomplete rows
✔ Köppen climate classification using kgcpy.lookupCZ(lat, lon)
✔ Save the cleaned dataset

```bash
python data_preparation.py 
```

**Outputs dataset :**

├── Data/ASHRAE_2022_Clean.csv


### 2️⃣ Meteorological Feature Extraction (feature_extraction_api.py)

This script enriches each dataset with outdoor weather indicators retrieved from the Open-Meteo API.
For every (lat, lon, date) pair, it fetches historical weather data and computes long-term exponential 
running averages (EMA):

✔Trm_ema_28  (running mean outdoor temperature)
✔RHout_ema_28 (outdoor relative humidity)
✔precip_ema_28 (precipitation)
✔sunshine_ema_h_28 (sunshine duration)
✔wind_ema_28 (wind speed)

```bash
python feature_extraction_api.py 
```
**Outputs datasets :**

├── Data/ASHRAE_2022_Clean_api.csv
├── Data/Moujalled_api.csv
├── Data/Hostein_api.csv

## 🚀 Phase 2 -  Model Training (models.py)

### All models on All targets

```bash
python models.py 
```

### 🎯 Target Selection
```bash
python models.py --targets [CHOICE]
```

**Choices :** `thermal_sensation`, `TSV_3p`, `thermal_preference`, `all` (défaut)

**Examples :**
```bash
python models.py --targets thermal_sensation
python models.py --targets all
```

---

### 🤖 Model family Selection

```bash
python models.py --models [TYPE]
```

**Choices :** `classical`, `deep`, `hybrid`, `all` (défaut)

**Examples :**
```bash
python models.py --models classical
python models.py --models all
```

---

### 🌲 Classics models

```bash
python models.py --classical_models [Models]
```

**Choices :** `RandomForest`, `SVM`, `XGBoost`, `all` (défaut)

**Examples :**
```bash
python models.py --classical_models RandomForest
python models.py --classical_models RandomForest XGBoost
```

---

### 🧠  deep learning models

```bash
python models.py --deep_models [MODEL]
```

**Choices :** `ANN`, `FTTransformer`, `all` (défaut)

**Examples :**
```bash
python models.py --deep_models FTTransformer
python models.py --deep_models all
```

---
### 🔀 Hybrid models 

```bash
python models.py --hybrid_heads [MODEL]
```

**Choices :** `RF`, `XGBoost`, `all` (défaut)

**Examples :**
```bash
# FTTransformer + RF head 
python models.py --models hybrid --deep_models FTTransformer --hybrid_heads RF
# FTTransformer + RF et XGBoost heads
python models.py --models hybrid --deep_models FTTransformer --hybrid_heads all

```

### ⏭️ Optional Skip Flags

```bash
--skip_feature_importance   # Disable feature-importance plots (RF / XGBoost)
--skip_embeddings           # Disable hybrid models (FT → RF/XGB)
--skip_visualizations       # Disable UMAP visualizations for FT embeddings


**Examples :**
```bash
python models.py --skip_feature_importance
python models.py --skip_embeddings 
```

---


### 🔄  Cross Validation

```bash
--cv_folds INT          # fold numbers (default 10) 
```

📁 Model Training Output 
├── kfold_results_unified/
│   ├── thermal_sensation/
|        ├── RandomForest
|        ├── XGBoost
|        ├── SVM
|        ├── ANN 
|        ├── FTTransormer
|        ├── FTTransormer_as_features
│   ├── TSV_3p/
|        ├── RandomForest
|        ├── XGBoost
|        ├── SVM
|        ├── ANN 
|        ├── FTTransormer
|        ├── FTTransormer_as_features
│   ├── thermal_preference/
|        ├── RandomForest
|        ├── XGBoost
|        ├── SVM
|        ├── ANN 
|        ├── FTTransormer
|        ├── FTTransormer_as_features
│   ├── comprehensive_experiment_comparison.csv



## 🔁 Phase 3 -  External Transfer Evaluation (transfert.py)

After training models on ASHRAE, this phase evaluates their transferability on two real-world French datasets (Moujalled and Hostein) with different climates, buildings, and collection protocols.

Two evaluation modes are implemented

### 1️⃣ Zero-Shot Transfer (Direct Application)

ASHRAE-trained models are directly applied to external datasets (no retraining).

Run only zero-shot:

```bash
python transfert.py --zero_shot_only 1       
```

### 2️⃣ Fine-Tuning Transfer

Models are adapted using a portion of the external dataset, with two fine-tuning ratios (20–80 and 80–20) used to evaluate sensitivity to training data volume.
Available adaptation types:

⏭️ RF_adapt / XGB_adapt
⏭️ FT_head (train last layer only)
⏭️ FT_full (full fine-tuning of the FTTransformer)

### 🧮 Fine-tuning ratios

```bash
python transfert.py --finetune_ratios [ratio]     
```

**Choices :** `0.2` → fine-tuning 20%, testing 80% , 
              `0.8` → fine-tuning 80%, testing 20% , 
              `0.2,0.8` (défault) → runs both 20-80 and 80-20

### 🎛️ Fine-tuning modes 

```bash
python transfert.py --finetune_ratios [mode]     
```

**Choices :** `FT_head` → Freeze FT backbone → train only final layer , 
              `FT_full` → Train full FTTransformer , 
              `RF_adapt`→ Refit RF prototype on external DEV
              `XGB_adapt`  → Refit XGB prototype on external DEV
              `all` (défault) → Run all four modes


📁 Transfer Learning Outputs

├── external_results/
│   ├── Moujalled/
|       ├── A_direct/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/
|       ├── B_finetune_20-80/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/
|       ├── B_finetune_80-20/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/
│   ├── Hostein/
|       ├── A_direct/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/
|       ├── B_finetune_20-80/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/
|       ├── B_finetune_80-20/
|               ├── thermal_sensation/
│               ├── TSV_3p/
│               ├── thermal_preference/

## 📘🌡️ Phase 3 - Heat balance baseline models (indicateurs_classiques.py)


This module implements a set of classical heat-balance and comfort indicator models that serve as baseline predictors for thermal sensation and preference.
These models rely on physical equations and long-established ergonomics standards rather than machine learning.

They are computed for the ASHRAE dataset and evaluated on the same test folds used in the ML experiments to ensure a fair comparison.

### Implemented indicators

The script computes a complete suite of comfort indices:

📐PMV & PPD (ISO 7730)
📐SET (Standard Effective Temperature)
📐PET (Physiological Equivalent Temperature)
📐PTS(PET) & PTS(SET) (linear regression of PET/SET vs TSV)
📐aPMV / ePMV (adaptive PMV coefficients)
📐aPTS / ePTS (adaptive SET-based corrections)

📁 Baseline models Outputs

├── ASHRAE_db2.01.1_clean_with_indicators.csv   # Original dataset enriched with all indicators.
├── resume_indicateurs_par_target_sur_test.csv  #Performance summary on all targets
├── results_indicateurs/                        #classification report per target × indicator.





## 📊 Phase 4 – Data Distribution & Vote Consistency Analyses

These two notebooks support the manuscript's data-quality and noise-floor
discussions (§3 / Discussion 4.4). They are **not** part of the five-command
pipeline described in *Reproducing the B&E paper* below; they read the
splits and predictions produced by `models.py` and add their own analyses.
Run them in a Jupyter environment, or non-interactively with
`jupyter nbconvert --to notebook --execute --inplace analysis/<notebook>.ipynb`,
*after* `models.py` has produced `kfold_results_unified/<target>/splits/`.

### 1 Train/Test Distribution Analysis

File: `analysis/train_test_comparaison.ipynb`
📁 Output folder: `analysis/analysis_results_distributions/`

This analysis checks whether the train and test splits used for ML training are statistically consistent.
It compares:

✔ categorical feature distributions (counts & percentages)
✔ numerical feature ranges (mean, std, min/max shifts)
✔ divergence patterns between train and test

Per-target CSV/PNG/JSON summaries land under
`analysis/analysis_results_distributions/<target>/`, with an aggregated
`global_summary.json` at the top of that folder.

### 2 Context-Based Votes Consistency Analysis

File: `analysis/context_votes.ipynb`

✔ Cas A — Exact Same Context

Strict identical conditions → consistency of TSV/TPV under perfect matching.
📁 Output folder: analysis/Votes_analysis/analysis_results_exact_context/

✔ Cas A.1 — Rounded Context (0.1 precision)

Small variations tolerated by rounding numeric features to one decimal.
📁 Output folder: analysis/Votes_analysis/analysis_results_rounded_context/

✔ Cas A.2 — Rounded-to-Int Context

Coarse grouping simulating by rounding numeric features to Int.
📁 Output folder: analysis/Votes_analysis/analysis_results_rounded_int_context/

✔ Cas B — Tolerance Windows (multi-dimensional ± ranges)

Similarity based on tolerance thresholds (±1°C, ±5% RH, ±0.3 m/s…).
📁 Output folder: analysis/Votes_analysis/analysis_results_tolerance_context/

✔ Cas C — KNN Nearest Neighbor in Normalized Feature Space

For each person, compare vote with the closest real-world neighbor in the full feature space.
This provides the upper bound of predictability (noise floor) for ML models
📁 Output folder: analysis/Votes_analysis/analysis_results_knn_neighbor/

Each case study writes `global_summary.json`, `groups_identical.json` and
`groups_different.json` to its output folder. Case C's primary agreement
rate (≈ 0.43 / 0.56 / 0.59 on TSV-7 / TSV-3 / TPV in the ASHRAE-2022 full
data) is the NN-agreement ceiling drawn as a dashed red line in
`regenerated_figures/performance_ceiling.pdf`.


## 📑 Reproducing the B&E paper

The current state of this repository — `config.yaml` together with `models_classic.py`, `models_deep.py`, `hybrid.py`, `transfert.py`, `indicateurs_classiques.py` and `analysis/haghirad_reproduction.py` — produces the figures published in Grayaa et al. 2026, *A Multi-Dataset Benchmark for Indoor Thermal Comfort Prediction* (Building and Environment), under the **no-reweighting regime** announced in §2.4.3: no `class_weight`, no `scale_pos_weight`, no `sample_weight`, no Focal Loss, no Random Over Sampling. Class imbalance is handled exclusively by stratified splits and stratified folds.

### Prerequisites
- Python 3.12, `torch==2.7.0+cu128`, a CUDA-compatible GPU recommended (tested on an NVIDIA RTX 3090 / driver 570.207 / CUDA 12.8).
- Without a GPU: classical models (RF, XGB, SVM) and ANN still complete on CPU; FT-Transformer is not recommended on CPU (>30 h on the full dataset).
- Datasets in `Data/`: `ASHRAE_2022_Clean_api.csv`, `ASHRAE_2018_v2.csv`, `Moujalled_api.csv`, `Hostein_api.csv`.

### Full pipeline (six commands, ~10–13 h on a single GPU)

```bash
# Phase 1 — Haghirad reproduction + 216-cell diagnostic factorial (RF, ASHRAE-2018, 12 features)
python3 analysis/haghirad_reproduction.py \
    --datasets all --encodings all \
    --output_dir rerun_2026-04-28_no_reweighting/phase1_haghirad_grid

# Phase 2 in-domain (ASHRAE-2022, 17 features) — all algorithms in one shot
python3 models.py \
    --dataset ASHRAE_2022 \
    --output_dir rerun_2026-04-28_no_reweighting/phase2_indomain

# Phase 2 empirical baselines (PMV, aPMV, SET, PET, PTS_*) on the same test split
python3 indicateurs_classiques.py \
    --data_csv Data/ASHRAE_2022_Clean_api.csv \
    --results_dir rerun_2026-04-28_no_reweighting/phase2_indomain \
    --output_dir rerun_2026-04-28_no_reweighting/phase2_baselines

# Phase 3 — direct (zero-shot) + adaptive (fine-tune 20-80 + 80-20) transfer
python3 transfert.py \
    --models_dir rerun_2026-04-28_no_reweighting/phase2_indomain \
    --output_dir rerun_2026-04-28_no_reweighting/phase3

# Phase 4 — nearest-neighbour disagreement ceiling (ASHRAE-2022, 12-feature Haghirad space)
python3 analysis/nn_disagreement.py \
    --scope full --features_key features_12 \
    --data_csv Data/ASHRAE_2022_Clean_api.csv \
    --output_dir rerun_2026-04-28_no_reweighting/diagnostics/nn_ashrae_2022

# Post-processing — SUMMARY.md, audit_log.json, regenerated LaTeX tables, performance ceiling figure
python3 analysis/finalize_paper_rerun.py \
    --rerun_dir rerun_2026-04-28_no_reweighting
```

### ✅ Verify the rerun reproduces the paper

After the pipeline above, one command checks every published table value against
the artefacts you just produced:

```bash
python3 analysis/verify_paper_tables.py
# Verifying 144 paper claims against rerun_2026-04-28_no_reweighting/ (tol = 0.010)
# ...
# SUMMARY: 144 pass, 0 fail, 0 missing
# ✅ All checked paper values reproduce within tolerance.
```

The expected values are frozen in [`analysis/paper_values.csv`](analysis/paper_values.csv)
(one row per claim across Phase 1 baseline, Phase 2 in-domain + hybrids + empirical
baselines, Phase 3 direct + adaptive transfer, and the Phase 4 NN ceiling). The
script re-reads each value from the rerun, compares within a tolerance (default
1 pp), prints a `PASS`/`FAIL`/`MISSING` table, and **exits non-zero on any
mismatch** — so it doubles as a CI gate. Point it at a different rerun with
`--rerun <dir>`, loosen the tolerance with `--tol`, or re-snapshot after an
intentional change with `--freeze` (then review the `paper_values.csv` diff).

### Reviewer flexibility — phase or algorithm at a time

Each script is self-contained and can be run on its own.

```bash
# Phase 1 only (ASHRAE-2018, no Phase 2/3 dependency)
python3 analysis/haghirad_reproduction.py --datasets all --encodings all --output_dir <out>

# Phase 2 — single model
python3 models.py --dataset ASHRAE_2022 --models classical --classical_models RandomForest --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models classical --classical_models XGBoost     --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models classical --classical_models SVM         --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models deep      --deep_models ANN              --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models deep      --deep_models FTTransformer    --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models hybrid    --hybrid_heads RF              --output_dir <out>
python3 models.py --dataset ASHRAE_2022 --models hybrid    --hybrid_heads XGBoost         --output_dir <out>

# Phase 2 — single target (e.g. TSV-7 only)
python3 models.py --dataset ASHRAE_2022 --targets thermal_sensation --output_dir <out>

# Phase 3 — single cohort, single regime (Moujalled, direct only) — controlled via transfert.py CLI flags
python3 transfert.py --models_dir <phase2_out> --output_dir <out>   # see --help for selective flags
```

### Sanity checks

| Check | Expected | Source artefact |
|---|---|---|
| RF Phase 2 hold-out test, TSV-7 | accuracy ∈ \[0.50, 0.53\] | `phase2_indomain/thermal_sensation/RandomForest/thermal_sensation__RandomForest_test_results.json` |
| Phase 1 grid `Haghirad_NotOpt_NoCW × label × ASHRAE_2022` | accuracy ≈ 0.516 ± 0.01 | `phase1_haghirad_grid/experiment_grid_full.csv` |
| Phase 1 grid `HighCap_BalSub × ROS=False × onehot × ASHRAE_2018` (Haghirad replica) | accuracy ≈ 0.521 ± 0.005 | same CSV |
| Phase 3 direct RF Moujalled TSV-7 | accuracy < 0.30 (domain shift) | `phase3/Moujalled/A_direct/thermal_sensation/RandomForest/...test_results.json` |
| Reproducibility | bit-exact across two independent runs (`seed=42` everywhere) | any `test_results.json` |

### Nomenclature

The configuration **HighCap** (`n_estimators=600`, `max_depth=25`, `min_samples_split=10`, `min_samples_leaf=2`) corresponds to the *high-capacity family* in the manuscript. The earlier archive `rerun_2026-04-23_haghirad_grid/` (weighted regime, kept for historical comparison only) refers to this same configuration by the internal name *TPB*.