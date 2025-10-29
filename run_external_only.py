import joblib
from pathlib import Path
import pandas as pd

import sys, importlib

# Remplace 'ton_script_entraînement' par le nom du fichier .py qui contient
# LabelEncodedClassifier, FTClassifier, TabPreprocessor, etc. (sans .py)
trained_mod = importlib.import_module('models')

# Le pickle attend les classes sur __main__ → on redirige
sys.modules['__main__'] = trained_mod

# === Charger les modèles déjà entraînés sur ASHRAE ===
base_out = Path("kfold_results_unified")
best_rf_by_target = joblib.load(base_out / "best_rf_by_target.joblib")
best_xgb_by_target = joblib.load(base_out / "best_xgb_by_target.joblib")
best_ft_by_target  = joblib.load(base_out / "best_ft_by_target.joblib")

# === Re-déclarer les CSV externes ===
CEREMA_CSV   = "Cerema.csv"     # <-- corrige ici le bon nom
MATHILDE_CSV = "Mathilde.csv"

# === Importer tes fonctions définies dans le gros script ===
from models import (
    direct_eval_one_base,
    finetune_eval_one_base,
    make_rf_xgb_protos,
    FEATURES_ALL, TARGETS_ALL, RANDOM_SEED
)

# === Colonnes num/cat d'après ASHRAE ===
df_ref = pd.read_csv("ASHRAE_db2.01.1_clean.csv")   # juste pour connaître types
num_cols_all = [c for c in FEATURES_ALL if pd.api.types.is_numeric_dtype(df_ref[c])]
cat_cols_all = [c for c in FEATURES_ALL if c not in num_cols_all]
rf_proto, xgb_proto = make_rf_xgb_protos(num_cols_all, cat_cols_all, seed=RANDOM_SEED)

# === Relancer uniquement les tests externes ===
direct_eval_one_base("CEREMA", CEREMA_CSV,
                     ft_by_target=best_ft_by_target,
                     rf_by_target=best_rf_by_target,
                     xgb_by_target=best_xgb_by_target)

direct_eval_one_base("MATHILDE", MATHILDE_CSV,
                     ft_by_target=best_ft_by_target,
                     rf_by_target=best_rf_by_target,
                     xgb_by_target=best_xgb_by_target)

for ratio in (0.2, 0.8):
    finetune_eval_one_base("CEREMA", CEREMA_CSV, dev_ratio=ratio,
                           ft_model_by_target=best_ft_by_target,
                           rf_proto=rf_proto, xgb_proto=xgb_proto,
                           ft_lrs=(1e-4, 5e-5), epochs=40, patience=6, seed=RANDOM_SEED)

    finetune_eval_one_base("MATHILDE", MATHILDE_CSV, dev_ratio=ratio,
                           ft_model_by_target=best_ft_by_target,
                           rf_proto=rf_proto, xgb_proto=xgb_proto,
                           ft_lrs=(1e-4, 5e-5), epochs=40, patience=6, seed=RANDOM_SEED)
