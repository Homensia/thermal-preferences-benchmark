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
### 1 Train/Test Distribution Analysis 

File: analysis/train_test_compraison.ipynb 
📁 Output folder: analysis/analysis_results_distributions/

This analysis checks whether the train and test splits used for ML training are statistically consistent.
It compares:

✔ categorical feature distributions (counts & percentages)
✔ numerical feature ranges (mean, std, min/max shifts)
✔ divergence patterns between train and test



### 2 Context-Based Votes Consistency Analysis
File : analysis/context_votes.ipynb

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