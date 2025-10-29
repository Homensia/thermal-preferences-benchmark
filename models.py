import os, random, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, platform
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.base import clone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import copy
import joblib
from unified_data_loader import UnifiedDataLoader
from unified_data_loader import create_label_mappings_from_models
from tab_transformer_pytorch import FTTransformer, TabTransformer
from stacking_implementation import train_and_evaluate_stacking
import torch.nn.functional as F







# ===== REPRODUCIBILITY SEEDS =====
RANDOM_SEED = 42

def set_all_seeds(seed=RANDOM_SEED):
    """
    Set all possible seeds for reproducibility
    """
    print(f"🔧 Setting all seeds to {seed} for reproducibility...")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("   ✅ All seeds set successfully")

# Set seeds at import time
set_all_seeds()

print("PyTorch:", torch.__version__)
print("Python:", platform.python_version())
print("CUDA dispo ?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))




SCORERS = {
    "f1_macro": make_scorer(lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
    "f1_micro": make_scorer(lambda yt, yp: f1_score(yt, yp, average="micro",  zero_division=0)),
    "f1_weighted": make_scorer(lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)),
    "precision_macro": make_scorer(lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0)),
    "recall_macro": make_scorer(lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0)),
    "accuracy": make_scorer(accuracy_score),
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p
def plot_confusion_matrix(cm, labels, title="Matrice de confusion",save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300) 
    plt.show()


def _get_transformed_feature_names_and_groups(pp: ColumnTransformer):
    """
    Retourne:
      - feat_out: liste des noms de features APRÈS transformation (num + OHE)
      - groups: liste de même longueur indiquant pour chaque feature transformée
                le nom de la variable d'origine (pour agréger les OHE)
    """
    feat_out, groups = [], []
    for name, trans, cols in pp.transformers_:
        if name == "remainder":
            continue

        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        else:
            names = np.array(cols, dtype=object)

        feat_out.extend(names.tolist())

        if name == "num":
            for c in names:
                groups.append(str(c).split("__", 1)[-1])
        elif name == "ohe" and hasattr(trans, "categories_"):
            for c in names:
                col_orig = str(c).split("__", 1)[-1]  
                groups.append(col_orig.split("_", 1)[0])
        else:
            for c in names:
                groups.append(str(c))

    return np.array(feat_out, dtype=object), np.array(groups, dtype=object)


def _extract_feature_importances_from_pipeline(pipe):
    """
    Récupère (importances, feat_out, groups) depuis un Pipeline(pp + clf ou clf.estimator_)
    Renvoie None si non disponible.
    """
    if not hasattr(pipe, "named_steps"):
        return None

    pp = pipe.named_steps.get("pp", None)
    clf = pipe.named_steps.get("clf", None)
    if pp is None or clf is None:
        return None

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "estimator_") and hasattr(clf.estimator_, "feature_importances_"):
        importances = clf.estimator_.feature_importances_

    if importances is None:
        return None

    try:
        feat_out, groups = _get_transformed_feature_names_and_groups(pp)
    except Exception:
        feat_out = getattr(pp, "get_feature_names_out", lambda: np.arange(len(importances)))()
        groups = np.array([str(f) for f in feat_out])

    if len(importances) != len(feat_out):
        print(f"[WARN] importances({len(importances)}) != features_out({len(feat_out)}). "
              f"Utilisation d'index générique.")
        feat_out = np.array([f"f{i}" for i in range(len(importances))])
        groups = feat_out.copy()

    return np.asarray(importances, dtype=float), feat_out, groups


def plot_feature_importance_pipeline(pipe, model_name, target_name, save_dir, top_n=None, save_csv=True):
    """
    - Récupère l'importance des features en sortie du ColumnTransformer
    - Agrège par variable d'origine (somme des dummies OHE)
    - Normalise et trace un barplot trié décroissant (comme Fig.12)
    """
    extr = _extract_feature_importances_from_pipeline(pipe)
    if extr is None:
        print(f"[WARN] Aucune importance dispo pour {model_name}.")
        return

    importances, feat_out, groups = extr

    df_imp = pd.DataFrame({"feat_out": feat_out, "group": groups, "importance": importances})
    grouped = df_imp.groupby("group", as_index=True)["importance"].sum()

    grouped = grouped / grouped.sum()

    grouped = grouped.sort_values(ascending=False)
    if top_n is not None:
        grouped = grouped.head(top_n)

    save_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        grouped.to_csv(save_dir / f"{target_name}_{model_name}_feature_importance_aggregated.csv")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=grouped.values, y=grouped.index, color="green")
    plt.xlabel("Feature Importances (normalized)")
    plt.ylabel("Features")
    plt.title(f"{model_name} – {target_name}")
    plt.tight_layout()
    plt.savefig(save_dir / f"{target_name}_{model_name}_feature_importance.png", dpi=300)
    plt.close()

    print(f"✅ Saved: {save_dir / f'{target_name}_{model_name}_feature_importance.png'}")


def split_train_test_stratified(X, y, test_size=0.20, seed=RANDOM_SEED):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(sss.split(X, y))
    return X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx]


def print_cv_fold_scores(grid: GridSearchCV, metric="f1_macro"):
    best = grid.best_index_
    trains = sorted(k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith(f"_train_{metric}"))
    vals   = sorted(k for k in grid.cv_results_.keys() if k.startswith("split") and k.endswith(f"_test_{metric}"))
    print("\n[Scores par fold] (train | val):")
    for i, (kt, kv) in enumerate(zip(trains, vals), 1):
        print(f"  fold {i}: {grid.cv_results_[kt][best]:.4f} | {grid.cv_results_[kv][best]:.4f}")



def eval_on_test(model, X_test, y_test, name="model", labels_order=None, target_names=None,save_dir: Path=None):
    y_pred = model.predict(X_test)
    if labels_order is None:
        labels_order = sorted(pd.Series(y_test).unique().tolist())
    if target_names is None:
        target_names = [str(l) for l in labels_order]

    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    f1mi = float(f1_score(y_test, y_pred, average="micro",   zero_division=0))
    f1w  = float(f1_score(y_test, y_pred, average="weighted",zero_division=0))
    rec  = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_test, y_pred,
        labels=labels_order,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    per_class_f1 = {lbl: report[str(lbl)]["f1-score"] for lbl in labels_order if str(lbl) in report}
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        cm_df.to_csv(save_dir / f"{name}_confusion_matrix.csv")
        plot_confusion_matrix(cm, labels_order, title=f"{name} — Confusion",
                              save_path=save_dir / f"{name}_confusion_matrix.png")
        with open(save_dir / f"{name}_test_report.json","w") as f:
            json.dump(report, f, indent=2)

    

    print(f"\n--- TEST FINAL — {name} ---")
    print(f"Acc: {acc:.4f} | F1_macro: {f1m:.4f}| F1_micro: {f1mi:.4f}| F1_weighted: {f1w:.4f}")
    print("F1 par classe   :", {str(k): f"{v:.3f}" for k, v in per_class_f1.items()})
    print("\n--- Rapport détaillé ---")
    print(classification_report(y_test, y_pred, zero_division=0))


    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1m,
        "f1_micro": f1mi,
        "f1_weighted": f1w,
        "labels": [str(l) for l in labels_order], 
        "confusion": cm.tolist(),                  
        "report": report,
        "y_pred": pd.Series(y_pred).tolist()
    }


def _scatter_2d(X2, y, title, save_path):
    plt.figure(figsize=(7,6))
    classes = pd.Series(y).astype(str)
    labels = classes.unique().tolist()
    for lab in labels:
        m = (classes == lab).values
        plt.scatter(X2[m,0], X2[m,1], s=12, alpha=0.75, label=str(lab))
    plt.title(title)
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.legend(markerscale=1.5, bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()





class TabPreprocessor:
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.encoders = {c: LabelEncoder() for c in self.cat_cols}
        self.scaler = SK_StandardScaler()

    def fit(self, X: pd.DataFrame):
        Xc = X[self.cat_cols].astype(str).copy()
        for c in self.cat_cols:
            Xc[c] = self.encoders[c].fit_transform(Xc[c].fillna("Unknown"))
        Xn = X[self.num_cols].copy()
        Xn = self.scaler.fit(Xn)
        self.n_categories = [len(self.encoders[c].classes_) for c in self.cat_cols]
        return self

    def transform(self, X: pd.DataFrame):
        Xc = X[self.cat_cols].astype(str).copy()
        for c in self.cat_cols:
            known = set(self.encoders[c].classes_)
            Xc[c] = Xc[c].apply(lambda v: v if v in known else self.encoders[c].classes_[0])
            Xc[c] = self.encoders[c].transform(Xc[c])
        Xn = X[self.num_cols].copy()
        Xn = self.scaler.transform(Xn)
        x_cat = torch.tensor(Xc.values, dtype=torch.long)
        x_num = torch.tensor(Xn, dtype=torch.float32)
        return x_cat, x_num


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=3, dropout=0.1):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class _TorchBase(BaseEstimator, ClassifierMixin):
    def __init__(self, num_cols, cat_cols=(), lr=3e-4, batch_size=1024, epochs=250, patience=10,
                  focal_gamma=None, class_weights=None, device=None, random_state=RANDOM_SEED,
                  curves_val_split=0.1, record_curves=True):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.curves_val_split = curves_val_split
        self.record_curves = record_curves  


    def _set_seeds(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)



    def _criterion(self, out_dim):
        w = None
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)

        if self.focal_gamma is not None:
            return FocalLoss(gamma=self.focal_gamma, weight=w)
        return nn.CrossEntropyLoss(weight=w)



class ANNClassifier(_TorchBase):
    def __init__(self, num_cols, cat_cols=(), hidden=(256,128), dropout=0.1, **kwargs):
        super().__init__(num_cols, cat_cols, **kwargs)
        self.hidden = hidden
        self.dropout = dropout

    def fit(self, X, y):
        self._set_seeds()
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols)).fit(X)
        x_cat_all, x_num_all = self.pp_.transform(X)

        self.le_y_ = LabelEncoder()
        y_idx_np = self.le_y_.fit_transform(pd.Series(y).values)
        self.classes_ = self.le_y_.classes_
        y_all  = torch.tensor(y_idx_np, dtype=torch.long)
        out_dim = len(self.le_y_.classes_)

        if self.record_curves and x_cat_all.size(0) > 100 and self.curves_val_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.curves_val_split, random_state=self.random_state)
            tr_idx, va_idx = next(sss.split(x_cat_all.numpy(), y_all.numpy()))
            tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)
        else:
            tr_idx = torch.arange(x_cat_all.size(0)); va_idx = None

        x_cat_tr, x_num_tr, y_tr = x_cat_all[tr_idx], x_num_all[tr_idx], y_all[tr_idx]
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_all[va_idx], x_num_all[va_idx], y_all[va_idx]

        Xtr = torch.cat([x_num_tr, x_cat_tr.float()], dim=1)
        if va_idx is not None:
            Xva = torch.cat([x_num_va, x_cat_va.float()], dim=1)

        self.model_ = MLP(in_dim=Xtr.shape[1], hidden=self.hidden, out_dim=out_dim, dropout=self.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(out_dim)

        
        self.history_ = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        best_loss, bad, best_state = float("inf"), 0, None
        Xtr, y_tr = Xtr.to(self.device), y_tr.to(self.device)
        if va_idx is not None:
            Xva, y_va = Xva.to(self.device), y_va.to(self.device)

        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(Xtr.size(0))
            losses = []
            for i in range(0, Xtr.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                logits = self.model_(Xtr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            val_loss = np.nan; val_f1 = np.nan
            if va_idx is not None:
                self.model_.eval()
                with torch.no_grad():
                    logits_va = self.model_(Xva)
                    val_loss = float(criterion(logits_va, y_va).item())
                    val_f1 = f1_score(y_va.cpu().numpy(),
                                      torch.argmax(logits_va, dim=1).cpu().numpy(),
                                      average="macro", zero_division=0)
            self.history_["epoch"].append(ep+1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_f1"].append(val_f1)

            monitor = val_loss if va_idx is not None else train_loss
            if monitor + 1e-6 < best_loss:
                best_loss, bad = monitor, 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X):
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        X_in = torch.cat([x_num, x_cat.float()], dim=1).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_in)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.le_y_.inverse_transform(pred_idx)


def _find_last_linear_for_out_dim(model: nn.Module, out_dim: int):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == out_dim:
            last = m
    return last

def ft_get_embeddings(ft_clf, X: pd.DataFrame, batch_size: int = 1024) -> np.ndarray:
    x_cat_all, x_num_all = ft_clf.pp_.transform(X)
    x_cat_all, x_num_all = x_cat_all.to(ft_clf.device), x_num_all.to(ft_clf.device)

    num_classes = len(ft_clf.le_y_.classes_)
    last_lin = _find_last_linear_for_out_dim(ft_clf.model_, num_classes)
    if last_lin is None:
        raise RuntimeError("Dernière couche Linear (out_dim=num_classes) introuvable dans le FT.")

    penult_batches = []
    def pre_hook(mod, inputs):
        penult_batches.append(inputs[0].detach().cpu())

    h = last_lin.register_forward_pre_hook(pre_hook)
    try:
        ft_clf.model_.eval()
        with torch.no_grad():
            N = x_cat_all.size(0)
            for i in range(0, N, batch_size):
                x_cat = x_cat_all[i:i+batch_size]
                x_num = x_num_all[i:i+batch_size]
                _ = ft_clf.model_(x_cat, x_num)
    finally:
        h.remove()

    if not penult_batches:
        raise RuntimeError("Aucun embedding capturé (hook non déclenché).")
    return torch.cat(penult_batches, dim=0).numpy()



class FTEmbeddingHead(BaseEstimator, ClassifierMixin):

    def __init__(self, ft_clf, head_estimator):
        self.ft_clf = ft_clf
        self.head_estimator = head_estimator

    def fit(self, X, y):
        Z = ft_get_embeddings(self.ft_clf, X)
        self.le_y_ = LabelEncoder().fit(y)
        y_enc = self.le_y_.transform(y)
        self.head_estimator_ = clone(self.head_estimator)
        self.head_estimator_.fit(Z, y_enc)
        return self

    def predict(self, X):
        Z = ft_get_embeddings(self.ft_clf, X)
        y_pred_enc = self.head_estimator_.predict(Z)
        return self.le_y_.inverse_transform(y_pred_enc)


def run_heads_on_ft_and_save(ft_clf, X_train, y_train, X_test, y_test, out_dir: Path,
                             target_name: str, tag_prefix="FTemb",n_umap_components=32):

    out_dir.mkdir(parents=True, exist_ok=True)

    Z_tr = ft_get_embeddings(ft_clf, X_train)
    Z_te = ft_get_embeddings(ft_clf, X_test)


    results_rows = []


    

    rf = RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced_subsample")
    rf_grid = {
        "n_estimators": [600],
        "max_depth": [25],
        "min_samples_split": [10],
        "min_samples_leaf": [2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rf_gs = GridSearchCV(rf, rf_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, return_train_score=True)
    rf_gs.fit(Z_tr, y_train.values)
    rf_best = rf_gs.best_estimator_
    y_pred_rf = rf_best.predict(Z_te)

    rf_dir = ensure_dir(out_dir / "RF")
    with open(rf_dir / f"{target_name}_{tag_prefix}_RF_cv_best.json", "w") as f:
        json.dump({"best_params": rf_gs.best_params_,
                   "cv_best_f1_macro": float(rf_gs.best_score_)}, f, indent=2)
    labels = sorted(pd.Series(y_test).unique())
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    pd.DataFrame(cm_rf, index=sorted(pd.Series(y_test).unique()), columns=sorted(pd.Series(y_test).unique())
                ).to_csv(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.csv", index=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
    plt.title(f"{target_name} — {tag_prefix}+RF — Confusion"); plt.tight_layout()
    plt.savefig(rf_dir / f"{target_name}_{tag_prefix}_RF_confusion_matrix.png", dpi=300); plt.close()

    report_rf = classification_report(y_test, y_pred_rf, zero_division=0, output_dict=True)
    with open(rf_dir / f"{target_name}_{tag_prefix}_RF_test_report.json", "w") as f:
        json.dump(report_rf, f, indent=2)

    results_rows.append({
        "experiment": target_name,
        "model": f"{tag_prefix}+RF",
        "test_accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "test_f1_macro": float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, y_pred_rf, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0))
    })

    print(f"[RF sur embeddings] {target_name} — "
      f"Acc: {accuracy_score(y_test, y_pred_rf):.4f}, "
      f"F1_macro: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
    




    le_y = LabelEncoder().fit(y_train.values)
    y_train_enc = le_y.transform(y_train.values)
    y_test_enc  = le_y.transform(y_test.values)

    xgb = XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss",
                        tree_method="hist", objective="multi:softprob")
    xgb_grid = {
        "n_estimators": [600],
        "max_depth": [6],
        "learning_rate": [0.05],
        "subsample": [0.7],
        "colsample_bytree": [0.7]
    }
    xgb_gs = GridSearchCV(xgb, xgb_grid, scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, return_train_score=True)
    xgb_gs.fit(Z_tr, y_train_enc)
    xgb_best = xgb_gs.best_estimator_
    y_pred_enc = xgb_best.predict(Z_te)
    y_pred_xgb = le_y.inverse_transform(y_pred_enc)

    xgb_dir = ensure_dir(out_dir / "XGBoost")
    with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_cv_best.json", "w") as f:
        json.dump({"best_params": xgb_gs.best_params_,
                   "cv_best_f1_macro": float(xgb_gs.best_score_)}, f, indent=2)
    labels = sorted(pd.Series(y_test).unique())
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    pd.DataFrame(cm_xgb, index=sorted(pd.Series(y_test).unique()), columns=sorted(pd.Series(y_test).unique())
                ).to_csv(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.csv", index=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues",xticklabels=labels, yticklabels=labels)
    plt.title(f"{target_name} — {tag_prefix}+XGB — Confusion"); plt.tight_layout()
    plt.savefig(xgb_dir / f"{target_name}_{tag_prefix}_XGB_confusion_matrix.png", dpi=300); plt.close()

    report_xgb = classification_report(y_test, y_pred_xgb, zero_division=0, output_dict=True)
    with open(xgb_dir / f"{target_name}_{tag_prefix}_XGB_test_report.json", "w") as f:
        json.dump(report_xgb, f, indent=2)

    results_rows.append({
        "experiment": target_name,
        "model": f"{tag_prefix}+XGB",
        "test_accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "test_f1_macro": float(f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, y_pred_xgb, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, y_pred_xgb, average="macro", zero_division=0))
    })

    print(f"[XGB sur embeddings] {target_name} — "
      f"Acc: {accuracy_score(y_test, y_pred_xgb):.4f}, "
      f"F1_macro: {f1_score(y_test, y_pred_xgb, average='macro'):.4f}")
    

    

    return results_rows
class FTClassifier(_TorchBase):
    def __init__(self, num_cols, cat_cols, dim=256, depth=4, heads=8, dim_head=32,
                 attn_dropout=0.1, ff_dropout=0.1, **kwargs):
        super().__init__(num_cols, cat_cols, **kwargs)
        self.dim, self.depth, self.heads, self.dim_head = dim, depth, heads, dim_head
        self.attn_dropout, self.ff_dropout = attn_dropout, ff_dropout

    def fit(self, X, y):
        self._set_seeds()
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols)).fit(X)
        x_cat_all, x_num_all = self.pp_.transform(X)

        self.le_y_ = LabelEncoder()
        y_idx_np = self.le_y_.fit_transform(pd.Series(y).values)
        self.classes_ = self.le_y_.classes_  
        y_all  = torch.tensor(y_idx_np, dtype=torch.long)
        num_classes = len(self.le_y_.classes_)
        
        if self.record_curves and x_cat_all.size(0) > 100 and self.curves_val_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.curves_val_split, random_state=self.random_state)
            tr_idx, va_idx = next(sss.split(x_cat_all.numpy(), y_all.numpy()))
            tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)
        else:
            tr_idx = torch.arange(x_cat_all.size(0)); va_idx = None

        x_cat_tr, x_num_tr, y_tr = x_cat_all[tr_idx], x_num_all[tr_idx], y_all[tr_idx]
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_all[va_idx], x_num_all[va_idx], y_all[va_idx]

        self.model_ = FTTransformer(
            categories=self.pp_.n_categories,
            num_continuous=x_num_tr.shape[1],
            dim=self.dim, depth=self.depth, heads=self.heads, dim_head=self.dim_head,
            dim_out=num_classes,
            attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout,
            num_residual_streams=4
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(num_classes)

        self.history_ = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        best_loss, bad, best_state = float("inf"), 0, None
        x_cat_tr, x_num_tr, y_tr = x_cat_tr.to(self.device), x_num_tr.to(self.device), y_tr.to(self.device)
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_va.to(self.device), x_num_va.to(self.device), y_va.to(self.device)

        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(x_cat_tr.size(0))
            losses = []
            for i in range(0, x_cat_tr.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                logits = self.model_(x_cat_tr[idx], x_num_tr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            val_loss = np.nan; val_f1 = np.nan
            if va_idx is not None:
                self.model_.eval()
                with torch.no_grad():
                    logits_va = self.model_(x_cat_va, x_num_va)
                    val_loss = float(criterion(logits_va, y_va).item())
                    val_f1 = f1_score(y_va.cpu().numpy(),
                                      torch.argmax(logits_va, dim=1).cpu().numpy(),
                                      average="macro", zero_division=0)
            self.history_["epoch"].append(ep+1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_f1"].append(val_f1)

            monitor = val_loss if va_idx is not None else train_loss
            if monitor + 1e-6 < best_loss:
                best_loss, bad = monitor, 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X):
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
        with torch.no_grad():
            logits = self.model_(x_cat, x_num)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        return self.le_y_.inverse_transform(pred_idx)
    
    def predict_proba(self, X):
        self.model_.eval()
        x_cat, x_num = self.pp_.transform(X)
        x_cat, x_num = x_cat.to(self.device), x_num.to(self.device)
        with torch.no_grad():
            logits = self.model_(x_cat, x_num)          
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

class TabTransformerClassifier(FTClassifier):
    def fit(self, X, y):
        self._set_seeds()
        self.pp_ = TabPreprocessor(list(self.cat_cols), list(self.num_cols)).fit(X)
        x_cat_all, x_num_all = self.pp_.transform(X)

        self.le_y_ = LabelEncoder()
        y_idx_np = self.le_y_.fit_transform(pd.Series(y).values)
        self.classes_ = self.le_y_.classes_ 
        y_all  = torch.tensor(y_idx_np, dtype=torch.long)
        num_classes = len(self.le_y_.classes_)

        if self.record_curves and x_cat_all.size(0) > 100 and self.curves_val_split > 0:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.curves_val_split, random_state=self.random_state)
            tr_idx, va_idx = next(sss.split(x_cat_all.numpy(), y_all.numpy()))
            tr_idx = torch.tensor(tr_idx); va_idx = torch.tensor(va_idx)
        else:
            tr_idx = torch.arange(x_cat_all.size(0)); va_idx = None

        x_cat_tr, x_num_tr, y_tr = x_cat_all[tr_idx], x_num_all[tr_idx], y_all[tr_idx]
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_all[va_idx], x_num_all[va_idx], y_all[va_idx]

        self.model_ = TabTransformer(
            categories=self.pp_.n_categories,
            num_continuous=x_num_tr.shape[1],
            dim=self.dim, depth=self.depth, heads=self.heads, dim_head=self.dim_head,
            dim_out=num_classes,
            attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = self._criterion(num_classes)

        self.history_ = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

        best_loss, bad, best_state = float("inf"), 0, None
        x_cat_tr, x_num_tr, y_tr = x_cat_tr.to(self.device), x_num_tr.to(self.device), y_tr.to(self.device)
        if va_idx is not None:
            x_cat_va, x_num_va, y_va = x_cat_va.to(self.device), x_num_va.to(self.device), y_va.to(self.device)

        for ep in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(x_cat_tr.size(0))
            losses = []
            for i in range(0, x_cat_tr.size(0), self.batch_size):
                idx = perm[i:i+self.batch_size]
                logits = self.model_(x_cat_tr[idx], x_num_tr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))

            val_loss = np.nan; val_f1 = np.nan
            if va_idx is not None:
                self.model_.eval()
                with torch.no_grad():
                    logits_va = self.model_(x_cat_va, x_num_va)
                    val_loss = float(criterion(logits_va, y_va).item())
                    val_f1 = f1_score(y_va.cpu().numpy(),
                                      torch.argmax(logits_va, dim=1).cpu().numpy(),
                                      average="macro", zero_division=0)
            self.history_["epoch"].append(ep+1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_f1"].append(val_f1)

            monitor = val_loss if va_idx is not None else train_loss
            if monitor + 1e-6 < best_loss:
                best_loss, bad = monitor, 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break

        if best_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self



class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrappe un estimateur clf pour encoder y en 0..K-1 à l'entraînement et décoder à la prédiction."""
    def __init__(self, estimator=None, **kwargs):
        self.estimator = estimator
        if kwargs:
            self.set_params(**kwargs)

    def get_params(self, deep=True):
        params = {"estimator": self.estimator}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=deep).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        est_params = {k.split("estimator__", 1)[1]: v
                      for k, v in params.items() if k.startswith("estimator__")}
        if est_params and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_params)
        return self

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.estimator_.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)
        raise AttributeError("Base estimator has no predict_proba")


def make_preprocess(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

def get_models(num_cols, cat_cols, seed=RANDOM_SEED):
    pp = make_preprocess(num_cols, cat_cols)
    models = {
        "RandomForest": (
            Pipeline([("pp", pp), ("clf", RandomForestClassifier(random_state=seed))]),
            {
                "clf__n_estimators": [600],
                "clf__max_depth": [25],
                "clf__min_samples_split": [10],
                "clf__min_samples_leaf": [2],
                "clf__class_weight": ["balanced_subsample"]
            }
        ),
        "SVM": (
            Pipeline([("pp", pp), ("clf", SVC(probability=False, random_state=seed))]),
            {
                "clf__kernel": ["rbf", "linear"],
                "clf__C": [0.5, 1, 5],
                "clf__gamma": ["scale", "auto"]
            }
        ),
        "XGBoost": (
    Pipeline([("pp", pp), 
              ("clf", LabelEncodedClassifier(
                  XGBClassifier(random_state=seed, eval_metric="mlogloss", tree_method="hist",
                                objective="multi:softprob")
              ))]),
            {
        "clf__estimator__n_estimators": [300],
        "clf__estimator__max_depth": [6],
        "clf__estimator__learning_rate": [0.05],
        "clf__estimator__subsample": [0.7],
        "clf__estimator__colsample_bytree": [0.7],
            }
        ),
    }
    return models

def get_torch_models(num_cols, cat_cols, seed=RANDOM_SEED):
    models = {
        "ANN": (
            ANNClassifier(num_cols=num_cols, cat_cols=cat_cols, epochs=250, patience=10, random_state=seed,focal_gamma=2.0,
                               batch_size=1024, lr=3e-4, record_curves=True),
            {
                "hidden": [(256,128), (512,256),(512,256,128), (1024,512,256)],
                "dropout": [0.1, 0.2],
                "batch_size": [1024]
            }
        ),
        "FTTransformer": (
            FTClassifier(num_cols=num_cols, cat_cols=cat_cols, epochs=250, patience=10, random_state=seed,focal_gamma=2.0,
                              batch_size=1024, lr=3e-4, record_curves=True),
            {
                "dim": [246],
                "depth": [4],
                "heads": [4],
                "batch_size": [1024]
            }
        )
        # ,
        # "TabTransformer": (
        #     TabTransformerClassifier(num_cols=num_cols, cat_cols=cat_cols, epochs=250, patience=10, random_state=seed,
        #                                   batch_size=1024, lr=3e-4, record_curves=True),
        #     {
        #         "dim": [128, 256],
        #         "depth": [3, 4],
        #         "heads": [4, 8],
        #         "batch_size": [1024]
        #     }
        # )
    }
    return models



def run_grid_search_on_train(target_name, model_name, pipe, param_grid, X_train, y_train,
                             refit_metric="f1_macro", k=5, n_jobs=-1, verbose=1):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=SCORERS,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"\n[{target_name} - {model_name}] Best params:", grid.best_params_)
    print_cv_fold_scores(grid, metric=refit_metric)

    best = grid.best_index_
    rows = []
    for split_id in range(k):
        rows.append({
            "fold": split_id + 1,
            "experiment": target_name,
            "model": model_name,
            "train_f1_macro": grid.cv_results_[f"split{split_id}_train_f1_macro"][best],
            "val_f1_macro":   grid.cv_results_[f"split{split_id}_test_f1_macro"][best],
            "train_accuracy": grid.cv_results_[f"split{split_id}_train_accuracy"][best],
            "val_accuracy":   grid.cv_results_[f"split{split_id}_test_accuracy"][best],
            "train_precision_macro": grid.cv_results_[f"split{split_id}_train_precision_macro"][best],
            "val_precision_macro":   grid.cv_results_[f"split{split_id}_test_precision_macro"][best],
            "train_recall_macro": grid.cv_results_[f"split{split_id}_train_recall_macro"][best],
            "val_recall_macro":   grid.cv_results_[f"split{split_id}_test_recall_macro"][best],
            "training_time": train_time
        })
    fold_summary_df = pd.DataFrame(rows)

    def stats_of(col):
        arr = fold_summary_df[col].values
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "median": float(np.median(arr))}
    stats_df = pd.DataFrame([
        {"experiment": target_name, "model": model_name, "metric": "val_accuracy", **stats_of("val_accuracy")},
        {"experiment": target_name, "model": model_name, "metric": "val_f1_macro", **stats_of("val_f1_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_precision_macro", **stats_of("val_precision_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_recall_macro", **stats_of("val_recall_macro")},
    ])
    return grid.best_estimator_, grid.best_params_, fold_summary_df, stats_df, train_time


def select_best(target_name, model_name, est, param_grid, X_tr, y_tr, k=5, refit_metric="f1_macro"):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    keys = list(param_grid.keys())

    def iter_param_sets(idx=0, current=None):
        if current is None: current = {}
        if idx == len(keys):
            yield current; return
        k_ = keys[idx]
        for v in param_grid[k_]:
            cc = dict(current); cc[k_] = v
            yield from iter_param_sets(idx+1, cc)

    best_score, best_params, best_model_proto = -np.inf, None, None
    per_param_val = []



    for params in iter_param_sets():
        base_params = {k: getattr(est, k) for k in est.__dict__.keys() if not k.endswith("_")}
        base_params.update(params)
        base_params.update({"record_curves": False, "curves_val_split": 0.0})
        cv_scores = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            m = est.__class__(**base_params)
            m.fit(X_tr_i, y_tr_i)
            y_pred = m.predict(X_va_i)
            cv_scores.append(f1_score(y_va_i, y_pred, average="macro", zero_division=0))
        mean_cv = float(np.mean(cv_scores))
        per_param_val.append({"params": params, "val_f1_macro_mean": mean_cv, "val_f1_macro_std": float(np.std(cv_scores))})
        print(f"[{target_name} - {model_name}] params={params} | {refit_metric}={mean_cv:.4f}")
        if mean_cv > best_score:
            best_score, best_params = mean_cv, params
            best_model_proto = est.__class__(**base_params)

    curve_params = {k: getattr(best_model_proto, k) for k in best_model_proto.__dict__.keys() if not k.endswith("_")}
    curve_params.update({"record_curves": True, "curves_val_split": 0.1})
    model_for_curves = est.__class__(**curve_params)
    model_for_curves.fit(X_tr, y_tr)

    final_params = {k: getattr(best_model_proto, k) for k in best_model_proto.__dict__.keys() if not k.endswith("_")}
    final_params.update({"record_curves": False, "curves_val_split": 0.0})
    best_model = est.__class__(**final_params)
    best_model.fit(X_tr, y_tr)

    skf2 = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED+1)
    rows = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf2.split(X_tr, y_tr), 1):
        X_tr_i, X_va_i = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_tr_i, y_va_i = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        mparams = {k: getattr(best_model, k) for k in best_model.__dict__.keys() if not k.endswith("_")}
        mparams.update({"record_curves": False, "curves_val_split": 0.0})
        m = est.__class__(**mparams)
        t0 = time.time()
        m.fit(X_tr_i, y_tr_i)
        train_time = time.time() - t0
        y_pred_tr = m.predict(X_tr_i)
        y_pred_va = m.predict(X_va_i)
        rows.append({
            "fold": fold_id,
            "experiment": target_name,
            "model": model_name,
            "train_f1_macro": f1_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_f1_macro":   f1_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "train_accuracy": accuracy_score(y_tr_i, y_pred_tr),
            "val_accuracy":   accuracy_score(y_va_i, y_pred_va),
            "train_precision_macro": precision_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_precision_macro":   precision_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "train_recall_macro": recall_score(y_tr_i, y_pred_tr, average="macro", zero_division=0),
            "val_recall_macro":   recall_score(y_va_i, y_pred_va, average="macro", zero_division=0),
            "training_time": train_time
        })
    fold_summary_df = pd.DataFrame(rows)

    def stats_of(col):
        arr = fold_summary_df[col].values
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "median": float(np.median(arr))}
    stats_df = pd.DataFrame([
        {"experiment": target_name, "model": model_name, "metric": "val_accuracy", **stats_of("val_accuracy")},
        {"experiment": target_name, "model": model_name, "metric": "val_f1_macro", **stats_of("val_f1_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_precision_macro", **stats_of("val_precision_macro")},
        {"experiment": target_name, "model": model_name, "metric": "val_recall_macro", **stats_of("val_recall_macro")},
    ])

    return best_model, best_params, fold_summary_df, stats_df, per_param_val, model_for_curves

def save_torch_curves(model, out_csv_path: Path):
    """Sauvegarde history_ (epoch, train_loss, val_loss, val_f1) si disponible."""
    if hasattr(model, "history_") and model.history_ and len(model.history_.get("epoch", [])) > 0:
        df = pd.DataFrame(model.history_)
        df.to_csv(out_csv_path, index=False)


def visualize_ft_embeddings(ft_clf, X, y, out_dir: Path, tag="train", max_points=10000, random_state=42):
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = ft_get_embeddings(ft_clf, X,)
    y_arr = pd.Series(y).values

    if Z.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(Z.shape[0], size=max_points, replace=False)
        Z_plot = Z[idx]; y_plot = y_arr[idx]
        X_plot = X.iloc[idx]
    else:
        Z_plot = Z; y_plot = y_arr; X_plot = X

    df_export = pd.DataFrame(Z, columns=[f"z{i+1}" for i in range(Z.shape[1])])
    df_export["target"] = y_arr
    df_export.to_csv(out_dir / f"FTemb_{tag}_full.csv", index=False)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=random_state, metric="cosine")
    Z_umap2 = reducer.fit_transform(Z_plot)

    _scatter_2d(
        Z_umap2, y_plot,
        title=f"UMAP 2D par target ({tag})",
        save_path=out_dir / f"FTemb_{tag}_UMAP2_by_target.png"
    )

    for col in X_plot.select_dtypes(include=["object", "category"]).columns:
        vals = X_plot[col].astype(str).values
        _scatter_2d(
            Z_umap2, vals,
            title=f"UMAP 2D par {col} ({tag})",
            save_path=out_dir / f"FTemb_{tag}_UMAP2_by_{col}.png"
        )



FEATURES_ALL = ['Tair','RH','clo','vel','Âge','Tout','Met','Sexe','Season','Climate','Building_type','cooling type','Trm_ema_7', 'RHout_ema_7','precip_ema_7','sunshine_ema_h_7','wind_ema_7']
TARGETS_ALL  = ["thermal_sensation","TSV_3p","thermal_preference"]

def stratified_dev_test(df: pd.DataFrame, target: str, dev_ratio: float, seed=42):
    """Split stratifié unique : DEV vs TEST sur la base externe."""
    y = df[target].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0-dev_ratio, random_state=seed)
    dev_idx, test_idx = next(sss.split(df, y))
    return df.iloc[dev_idx].copy(), df.iloc[test_idx].copy()

def make_rf_xgb_protos(num_cols, cat_cols, seed=RANDOM_SEED):
    """Pipelines réutilisables pour re-fit rapide sur DEV externe (fine-tune côté modèles sklearn)."""
    pp = make_preprocess(num_cols, cat_cols)
    rf_proto = Pipeline([
        ("pp", pp),
        ("clf", RandomForestClassifier(
            n_estimators=600, max_depth=25, min_samples_split=10,
            min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed
        ))
    ])
    xgb_proto = Pipeline([
        ("pp", pp),
        ("clf", LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=600, max_depth=10, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7,
                tree_method="hist", eval_metric="mlogloss",
                objective="multi:softprob", random_state=seed
            )
        ))
    ])
    return rf_proto, xgb_proto


def clone_ft_for_finetune(ft_model, lr=5e-5, epochs=50, patience=6, freeze_backbone=True):
    """
    Copie un FT entraîné sur ASHRAE.
    - freeze_backbone=True => FT_head (on entraîne seulement la dernière couche)
    - freeze_backbone=False => FT_full (on entraîne tout)
    """
    m = copy.deepcopy(ft_model)
    m.lr, m.epochs, m.patience = lr, epochs, patience
    m.record_curves, m.curves_val_split = True, 0.1
    if freeze_backbone:
        # geler tout le backbone
        for p in m.model_.parameters():
            p.requires_grad = False
        # réactiver juste la dernière couche 
        last = _find_last_linear_for_out_dim(m.model_, len(m.le_y_.classes_))
        if last is None:
            raise RuntimeError("Dernière couche Linear introuvable.")
        for p in last.parameters():
            p.requires_grad = True
    return m

def direct_eval_one_base(base_name: str, df: pd.DataFrame,
                         ft_by_target: dict, rf_by_target: dict=None, xgb_by_target: dict=None):
    """
    Test direct (zero-shot) sur une base externe : on n'apprend rien, on évalue tel quel.
    """
    out_root = ensure_dir(Path("external_results") / base_name / "A_direct")

    for target in [t for t in TARGETS_ALL if t in df.columns]:
        X, y = df[FEATURES_ALL], df[target]

        # Harmoniser les types → float
        try:
            y = y.astype(float)
        except:
            y = pd.to_numeric(y, errors="coerce")

        if target in ft_by_target:
            ft = ft_by_target[target]

            # Harmoniser les classes du FT
            allowed = set(map(float, ft.le_y_.classes_))
            mask = y.astype(float).isin(allowed)

            if mask.sum() > 0:
                _ = eval_on_test(ft, X[mask], y[mask],
                                 name=f"{base_name}__{target}__FTTransformer",
                                 save_dir=ensure_dir(out_root / target / "FTTransformer"))
            else:
                print(f"[{base_name}:{target}] Aucun label compatible avec le FT (skip).")

        # RF direct
        if rf_by_target and target in rf_by_target:
            rf = rf_by_target[target]
            _ = eval_on_test(rf, X, y,
                             name=f"{base_name}__{target}__RF",
                             save_dir=ensure_dir(out_root / target / "RandomForest"))

        # XGB direct
        if xgb_by_target and target in xgb_by_target:
            xgb = xgb_by_target[target]
            _ = eval_on_test(xgb, X, y,
                             name=f"{base_name}__{target}__XGB",
                             save_dir=ensure_dir(out_root / target / "XGBoost"))


def finetune_eval_one_base(base_name: str, df: pd.DataFrame, dev_ratio: float,
                           ft_model_by_target: dict,
                           rf_proto=None, xgb_proto=None,
                           ft_lrs=(1e-4, 5e-5), epochs=40, patience=6, seed=42):
    """
    Fine-tuning sur la base externe :
      - split stratifié DEV/TEST avec dev_ratio (0.2 => 20/80, 0.8 => 80/20)
      - FT_head  et FT_full 
      - RF/XGB ré-entraînés sur DEV (adapt)
    """
    out_root = ensure_dir(Path("external_results") / base_name / f"B_finetune_{int(dev_ratio*100)}-{int((1-dev_ratio)*100)}")

    for target in [t for t in TARGETS_ALL if t in df.columns]:
        ft_base = ft_model_by_target.get(target, None)
        if ft_base is None:
            print(f"[{base_name}:{target}] Pas de FT de référence — skip.")
            continue

        # --- Harmoniser types: y externe en float, classes FT en float ---
        allowed = set(map(float, ft_base.le_y_.classes_))  
        df["_y_float"] = pd.to_numeric(df[target], errors="coerce")  

        dff = df[df["_y_float"].isin(allowed)].copy()
        if dff["_y_float"].nunique() < 2:
            print(f"[{base_name}:{target}] Trop peu de classes après filtrage — skip.")
            df.drop(columns=["_y_float"], inplace=True, errors="ignore")
            continue

        # Remplacer la colonne cible par la version float harmonisée
        dff[target] = dff["_y_float"].astype(float)
        dff.drop(columns=["_y_float"], inplace=True, errors="ignore")

        df_dev, df_test = stratified_dev_test(dff, target, dev_ratio, seed=seed)
        Xd, yd = df_dev[FEATURES_ALL], df_dev[target].astype(float)
        Xt, yt = df_test[FEATURES_ALL], df_test[target].astype(float)

        # -------- FT_head  --------
        ft_head = clone_ft_for_finetune(ft_base, lr=ft_lrs[0], epochs=epochs, patience=patience, freeze_backbone=True)
        ft_head.fit(Xd, yd)
        _ = eval_on_test(ft_head, Xt, yt,
                         name=f"{base_name}__{target}__FT_head",
                         save_dir=ensure_dir(out_root / target / "FT_head"))

        # -------- FT_full --------
        ft_full = clone_ft_for_finetune(ft_base, lr=ft_lrs[1], epochs=epochs, patience=patience, freeze_backbone=False)
        ft_full.fit(Xd, yd)
        _ = eval_on_test(ft_full, Xt, yt,
                         name=f"{base_name}__{target}__FT_full",
                         save_dir=ensure_dir(out_root / target / "FT_full"))

        # -------- RF adapt --------
        if rf_proto is not None:
            rf_adapt = clone(rf_proto)
            rf_adapt.fit(Xd, yd)
            _ = eval_on_test(rf_adapt, Xt, yt,
                             name=f"{base_name}__{target}__RF_adapt",
                             save_dir=ensure_dir(out_root / target / "RF_adapt"))

        # -------- XGB adapt  --------
        if xgb_proto is not None:
            xgb_adapt = clone(xgb_proto)
            xgb_adapt.fit(Xd, yd)
            _ = eval_on_test(xgb_adapt, Xt, yt,
                             name=f"{base_name}__{target}__XGB_adapt",
                             save_dir=ensure_dir(out_root / target / "XGB_adapt"))




if __name__ == "__main__":

    Path("data_validation_reports").mkdir(parents=True, exist_ok=True)
    loader = UnifiedDataLoader(FEATURES_ALL, TARGETS_ALL)
    DATA = loader.load_and_validate("ASHRAE_2022_api.csv", "ASHRAE")
    loader.save_statistics(Path("data_validation_reports/ashrae_report.json"))


    FEATURES = FEATURES_ALL
    TARGETS  = TARGETS_ALL

    base_out = ensure_dir(Path("kfold_results_unified"))
    global_comparison_rows = []


    best_ft_by_target = {}     
    best_rf_by_target = {}     
    best_xgb_by_target = {}    


    best_stacking_by_target = {}

    for TARGET in TARGETS:
        print("\n" + "="*120)
        print(f"🎯 TARGET = {TARGET}")
        print("="*120)

        X_all = DATA[FEATURES].copy()
        y_all = DATA[TARGET].copy()

        X_train, X_test, y_train, y_test = split_train_test_stratified(X_all, y_all, test_size=0.20, seed=RANDOM_SEED)

        # indices de split pour indicateurs classiques
        split_dir = ensure_dir((Path("kfold_results_unified") / TARGET / "splits"))
        split_path = split_dir / f"{TARGET}_split_indices.json"
        with open(split_path, "w") as f:
            json.dump({
        "target": TARGET,
        "train_idx": X_train.index.astype(int).tolist(),
        "test_idx":  X_test.index.astype(int).tolist(),
        "features": FEATURES
    }, f, indent=2)
        print(f"[SPLIT] Indices train/test sauvegardés → {split_path}")

        num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        classes_sorted = np.array(sorted(pd.Series(y_train).unique().tolist(), key=lambda x: (str(type(x)), x)))
        MODELS_SK    = get_models(num_cols, cat_cols, seed=RANDOM_SEED)
        MODELS_TORCH = get_torch_models(num_cols, cat_cols, seed=RANDOM_SEED)
        exp_dir = ensure_dir(base_out / TARGET)


        per_model_fold_summaries, per_model_stats, per_model_test = [], [], []

        for model_name, (pipe, pgrid) in MODELS_SK.items():
            best_est, best_params, fold_summary_df, stats_df, _ = run_grid_search_on_train(
                TARGET, model_name, pipe, pgrid, X_train, y_train,
                refit_metric="f1_macro", k=5, verbose=2, n_jobs=-1
                        )

            mdir = ensure_dir(exp_dir / model_name)

            test_res = eval_on_test(
                best_est, X_test, y_test,
                name=f"{TARGET}__{model_name}",
                save_dir=mdir
                )
            if model_name in ["RandomForest", "XGBoost"]:
                plot_feature_importance_pipeline(
                pipe=best_est,
                model_name=model_name,
                target_name=TARGET,
                save_dir=mdir,
                top_n=None,          
                save_csv=True
                )




            if model_name == "RandomForest":
                best_rf_by_target[TARGET] = best_est
            elif model_name == "XGBoost":
                best_xgb_by_target[TARGET] = best_est

            fold_summary_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_summary.csv", index=False)
            stats_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_statistics.csv", index=False)

            pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])\
                .to_csv(mdir / f"{TARGET}_{model_name}_confusion_matrix.csv")

            with open(mdir / f"{TARGET}_{model_name}_test_results.json", "w") as f:
                json.dump(test_res, f, indent=2)

            per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
            per_model_stats.append(stats_df.assign(model=model_name))
            per_model_test.append({
             "experiment": TARGET, "model": model_name,
             "test_accuracy": test_res["accuracy"],
             "test_f1_macro": test_res["f1_macro"],
             "test_precision_macro": test_res["precision_macro"],
                "test_recall_macro": test_res["recall_macro"]
            })


        for model_name, (est, pgrid) in MODELS_TORCH.items():
            best_model, best_params, fold_summary_df, stats_df, per_param_val, model_for_curves = \
                select_best(TARGET, model_name, est, pgrid, X_train, y_train, k=5)


            mdir = ensure_dir(exp_dir / model_name)          
            save_torch_curves(model_for_curves, mdir / f"{TARGET}_{model_name}_training_curves.csv")
            test_res = eval_on_test(best_model, X_test, y_test, name=f"{TARGET}__{model_name}",save_dir=ensure_dir(exp_dir / model_name ))

            if model_name == "FTTransformer":
                best_ft_by_target[TARGET] = best_model

                # viz_dir = ensure_dir(exp_dir / f"{model_name}_viz")
                # visualize_ft_embeddings(best_model, X_train, y_train, viz_dir, tag="train", max_points=8000)
                # visualize_ft_embeddings(best_model, X_test,  y_test,  viz_dir, tag="test",  max_points=8000)

            if model_name == "FTTransformer":
                try:
                    ft_head_dir = ensure_dir(exp_dir / f"{model_name}_as_features")
                    head_rows = run_heads_on_ft_and_save(
                        ft_clf=best_model,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test,   y_test=y_test,
                        out_dir=ft_head_dir,
                        target_name=TARGET,
                        tag_prefix="FTemb"
                    )
                    for r in head_rows:
                        per_model_test.append(r)
                except Exception as e:
                    print(f"[WARN] FT→(RF/XGB) embedding heads skipped: {e}")


            fold_summary_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_summary.csv", index=False)
            stats_df.to_csv(mdir / f"{TARGET}_{model_name}_fold_statistics.csv", index=False)
            with open(mdir / f"{TARGET}_{model_name}_test_results.json", "w") as f:
                json.dump(test_res, f, indent=2)
            with open(mdir / f"{TARGET}_{model_name}_param_grid_cv.json", "w") as f:
                json.dump(per_param_val, f, indent=2)
            cm_df = pd.DataFrame(test_res["confusion"], index=test_res["labels"], columns=test_res["labels"])
            cm_df.to_csv(mdir / f"{TARGET}_{model_name}_confusion_matrix.csv")

            plot_confusion_matrix(test_res["confusion"], test_res["labels"], title=f"{TARGET} - {model_name}",save_path=mdir / f"{TARGET}_{model_name}_confusion_matrix.png")

            per_model_fold_summaries.append(fold_summary_df.assign(model=model_name))
            per_model_stats.append(stats_df.assign(model=model_name))
            per_model_test.append({
                "experiment": TARGET, "model": model_name,
                "test_accuracy": test_res["accuracy"],
                "test_f1_macro": test_res["f1_macro"],
                "test_precision_macro": test_res["precision_macro"],
                "test_recall_macro": test_res["recall_macro"]
            })


        # STACKING ENSEMBLE
        print("\n" + "="*120)
        print(f" STACKING ENSEMBLE - {TARGET}")
        print("="*120)

        if TARGET in best_rf_by_target and TARGET in best_xgb_by_target and TARGET in best_ft_by_target:
            try:
                stacking_model, stacking_results = train_and_evaluate_stacking(
                    rf_model=best_rf_by_target[TARGET],
                    xgb_model=best_xgb_by_target[TARGET],
                    ft_model=best_ft_by_target[TARGET],
                    X_train=X_train, y_train=y_train,
                    X_test=X_test,   y_test=y_test,
                    target_name=TARGET,
                    save_dir=exp_dir / "Stacking",
                    n_folds=5, random_state=RANDOM_SEED
                )
                best_stacking_by_target[TARGET] = stacking_model
                per_model_test.append({
                    "experiment": TARGET, "model": "Stacking",
                    "test_accuracy": stacking_results['stacking']['accuracy'],
                    "test_f1_macro": stacking_results['stacking']['f1_macro'],
                    "test_precision_macro": stacking_results['stacking']['precision_macro'],
                    "test_recall_macro": stacking_results['stacking']['recall_macro'],
                })
                
                print(f"\n Stacking pour {TARGET} terminé avec succès")
            except Exception as e:
                print(f"\n Erreur lors du stacking pour {TARGET} : {e}")
                import traceback; traceback.print_exc()
        else:
            print(f"\n  Stacking skippé : tous les base models ne sont pas dispos")
            print(f"   RF: {TARGET in best_rf_by_target} | XGB: {TARGET in best_xgb_by_target} | FT: {TARGET in best_ft_by_target}")

        if per_model_fold_summaries:
            pd.concat(per_model_fold_summaries, ignore_index=True).to_csv(
                exp_dir / f"{TARGET}_ALLMODELS_fold_summary.csv", index=False
            )
        if per_model_stats:
            pd.concat(per_model_stats, ignore_index=True).to_csv(
                exp_dir / f"{TARGET}_ALLMODELS_fold_statistics.csv", index=False
            )
        pd.DataFrame(per_model_test).to_csv(exp_dir / f"{TARGET}_ALLMODELS_test_scores.csv", index=False)

        for r in per_model_test:
            global_comparison_rows.append({
                "experiment": TARGET,
                "model": r["model"],
                "test_accuracy": r["test_accuracy"],
                "test_f1_macro": r["test_f1_macro"],
                "test_precision_macro": r["test_precision_macro"],
                "test_recall_macro": r["test_recall_macro"],
            })

        print(f" Résultats écrits dans: {exp_dir}")

    
    joblib.dump(best_rf_by_target, str(base_out / "best_rf_by_target.joblib"))
    joblib.dump(best_xgb_by_target, str(base_out / "best_xgb_by_target.joblib"))
    joblib.dump(best_ft_by_target,  str(base_out / "best_ft_by_target.joblib"))


    joblib.dump(best_stacking_by_target, str(base_out / "best_stacking_by_target.joblib"))
    print(f" Stacking models sauvegardés : {base_out / 'best_stacking_by_target.joblib'}")


    pd.DataFrame(global_comparison_rows).to_csv(base_out / "comprehensive_experiment_comparison.csv", index=False)
    print(f"\n Terminé. Récap global: {base_out / 'comprehensive_experiment_comparison.csv'}")





    # ===================== TESTS EXTERNES CEREMA / MATHILDE =====================
    label_mappings = create_label_mappings_from_models(best_ft_by_target)
    loader_external = UnifiedDataLoader(FEATURES_ALL, TARGETS_ALL, label_mappings)

    df_cerema = loader_external.load_and_validate("Cerema_api.csv", "CEREMA")
    cerema_loaded = True

    df_mathilde = loader_external.load_and_validate("Mathilde_api.csv", "MATHILDE")
    mathilde_loaded = True


    loader_external.save_statistics(Path("data_validation_reports/external_report.json"))







    num_cols_all = [c for c in FEATURES if pd.api.types.is_numeric_dtype(DATA[c])]
    cat_cols_all = [c for c in FEATURES if c not in num_cols_all]
    rf_proto, xgb_proto = make_rf_xgb_protos(num_cols_all, cat_cols_all, seed=RANDOM_SEED)

    # A) Tests DIRECTS
    direct_eval_one_base("CEREMA",   df_cerema,   ft_by_target=best_ft_by_target,
                         rf_by_target=best_rf_by_target, xgb_by_target=best_xgb_by_target)
    direct_eval_one_base("MATHILDE", df_mathilde, ft_by_target=best_ft_by_target,
                         rf_by_target=best_rf_by_target, xgb_by_target=best_xgb_by_target)

    # B) Fine-tuning — 20/80 et 80/20, avec FT_head et FT_full + RF/XGB adapt
    for ratio in (0.2, 0.8):
        finetune_eval_one_base("CEREMA",   df_cerema,   dev_ratio=ratio,
                               ft_model_by_target=best_ft_by_target,
                               rf_proto=rf_proto, xgb_proto=xgb_proto,
                               ft_lrs=(1e-4, 5e-5), epochs=50, patience=6, seed=RANDOM_SEED)
        finetune_eval_one_base("MATHILDE", df_mathilde, dev_ratio=ratio,
                               ft_model_by_target=best_ft_by_target,
                               rf_proto=rf_proto, xgb_proto=xgb_proto,
                               ft_lrs=(1e-4, 5e-5), epochs=50, patience=6, seed=RANDOM_SEED)