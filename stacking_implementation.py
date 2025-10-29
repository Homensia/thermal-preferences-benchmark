"""
Stacking Ensemble : RandomForest + XGBoost + FTTransformer + Meta-Learner

Architecture:
    RF    →  Probas RF   ┐
    XGB   →  Probas XGB  ├→ Logistic Regression → Prédiction finale
    FT    →  Probas FT   ┘
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json




def _proba_aligned(model, X, classes_ref):
    """Aligne les probabilités en gérant les classes manquantes."""
    proba = model.predict_proba(X)
    
    # Récupérer classes du modèle
    if hasattr(model, 'classes_'):
        classes_model = model.classes_
    elif hasattr(model, 'named_steps') and 'clf' in model.named_steps:
        classes_model = model.named_steps['clf'].classes_
    else:
        raise ValueError("Impossible de récupérer les classes du modèle")
    
    # Créer mapping
    ref_index = {c: i for i, c in enumerate(classes_ref)}
    aligned = np.zeros((proba.shape[0], len(classes_ref)), dtype=proba.dtype)
    
    for j, c in enumerate(classes_model):
        if c in ref_index:
            aligned[:, ref_index[c]] = proba[:, j]
        else:
            print(f"Warning: Classe {c} du modèle ignorée (pas dans classes_ref)")
    
    # Normaliser les probabilités après alignement
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  
    aligned = aligned / row_sums
    
    return aligned




#StackingEnsemble

class StackingEnsemble:
    """
    Ensemble de stacking avec validation croisée pour éviter data leakage.
    
    Fonctionnement:
    1. Génère prédictions out-of-fold pour chaque modèle base
    2. Entraîne meta-learner sur ces prédictions
    3. Combine intelligemment les prédictions sur test set
    """
    
    def __init__(self, base_models: dict, meta_learner=None, n_folds=5, 
                 random_state=42, verbose=True):
        """
        Args:
            base_models: Dict {"nom": model_sklearn_compatible}
                        Ex: {"RF": rf_model, "XGB": xgb_model, "FT": ft_model}
            meta_learner: Modèle pour combiner (défaut: LogisticRegression)
            n_folds: Nombre de folds pour out-of-fold predictions
            random_state: Seed pour reproductibilité
            verbose: Afficher progression
        """
        self.base_models = base_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # Meta-learner 
        self.meta_learner = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs',
                random_state=random_state
            )
        
        self.meta_learner_fitted_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.base_models_fitted_ = {}
        
    def fit(self, X_train, y_train):
        """
        Entraîne l'ensemble avec validation croisée.
        
        Étapes:
        1. Pour chaque fold, entraîne base models sur folds 1-4
        2. Prédit sur fold 5 (out-of-fold) pour éviter leakage
        3. Concatène toutes les prédictions out-of-fold
        4. Entraîne meta-learner sur ces prédictions
        5. Ré-entraîne base models sur train complet
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🎯 ENTRAÎNEMENT STACKING ENSEMBLE")
            print(f"{'='*70}")
            print(f"Base models : {list(self.base_models.keys())}")
            print(f"Meta-learner : {type(self.meta_learner).__name__}")
            print(f"CV folds : {self.n_folds}")
        
        # Récupérer classes
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Préparer arrays pour stocker prédictions out-of-fold
        n_samples = len(X_train)
        n_models = len(self.base_models)
        

        oof_predictions = np.zeros((n_samples, n_models * self.n_classes_))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                             random_state=self.random_state)
        
        if self.verbose:
            print(f"\n📊 Génération prédictions out-of-fold...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            if self.verbose:
                print(f"\n  Fold {fold_idx}/{self.n_folds}:")
            
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            
            for model_idx, (model_name, model) in enumerate(self.base_models.items()):
                if self.verbose:
                    print(f"    - {model_name}...", end=" ")
                
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_fold_train, y_fold_train)
                
                # Prédictions probabilistes sur validation fold
                if hasattr(model_clone, 'predict_proba'):
                    fold_proba = _proba_aligned(model_clone, X_fold_val, self.classes_)
                else:
                    raise ValueError(f"{model_name} n'a pas de méthode predict_proba")
                
                start_col = model_idx * self.n_classes_
                end_col = start_col + self.n_classes_
                oof_predictions[val_idx, start_col:end_col] = fold_proba
                
                if self.verbose:
                    print("✓")
        
        if self.verbose:
            print(f"\n✅ Prédictions out-of-fold générées : {oof_predictions.shape}")
        
        if self.verbose:
            print(f"\n🧠 Entraînement meta-learner...")
        
        self.meta_learner_fitted_ = clone(self.meta_learner)
        self.meta_learner_fitted_.fit(oof_predictions, y_train)
        
        if self.verbose:
            print(f"✅ Meta-learner entraîné")
        
        # Ré-entraîner base models sur train complet
        if self.verbose:
            print(f"\n🔄 Ré-entraînement base models sur train complet...")
        
        for model_name, model in self.base_models.items():
            if self.verbose:
                print(f"  - {model_name}...", end=" ")
            
            model_final = clone(model)
            model_final.fit(X_train, y_train)
            self.base_models_fitted_[model_name] = model_final
            
            if self.verbose:
                print("✓")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ STACKING ENSEMBLE ENTRAÎNÉ")
            print(f"{'='*70}")
        
        return self
    
    def predict_proba(self, X):
        if self.meta_learner_fitted_ is None:
            raise ValueError("Appeler fit() avant predict_proba().")
        n = len(X)
        n_models = len(self.base_models_fitted_)
        all_probas = np.zeros((n, n_models * self.n_classes_), dtype=float)

        for m_idx, (name, model) in enumerate(self.base_models_fitted_.items()):
            if not hasattr(model, "predict_proba"):
                raise ValueError(f"{name} doit implémenter predict_proba.")
            aligned = _proba_aligned(model, X, self.classes_)
            start = m_idx * self.n_classes_
            all_probas[:, start:start + self.n_classes_] = aligned

        final = self.meta_learner_fitted_.predict_proba(all_probas)
        return final
    
    def predict(self, X):
        """Prédictions de classes via stacking"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def get_model_contributions(self, X):
        """
        Analyse les contributions de chaque modèle.
        Utile pour comprendre quels modèles sont privilégiés.
        
        Returns:
            Dict avec probas de chaque modèle et poids du meta-learner
        """
        n_samples = len(X)
        contributions = {}
        
        # Probas de chaque base model
        for model_name, model in self.base_models_fitted_.items():
            contributions[model_name] = _proba_aligned(model, X, self.classes_)

        
        if hasattr(self.meta_learner_fitted_, 'coef_'):
            contributions['meta_weights'] = self.meta_learner_fitted_.coef_
        
        return contributions


# ÉVALUATION

def evaluate_stacking(stacking_model, X_test, y_test, model_name="Stacking",
                     save_dir=None, labels_order=None):
    """
    Évalue le modèle de stacking et compare avec base models.
    
    Args:
        stacking_model: StackingEnsemble entraîné
        X_test, y_test: Données de test
        model_name: Nom pour les sauvegardes
        save_dir: Dossier pour sauvegarder résultats
        labels_order: Ordre des classes pour confusion matrix
        
    Returns:
        Dict avec toutes les métriques
    """
    y_pred_stack = stacking_model.predict(X_test)
    y_proba_stack = stacking_model.predict_proba(X_test)
    
    base_predictions = {}
    for model_name_base, model in stacking_model.base_models_fitted_.items():
        base_predictions[model_name_base] = model.predict(X_test)
    
    if labels_order is None:
        labels_order = sorted(np.unique(y_test))
    
    # Métriques stacking
    results = {
        "stacking": {
            "accuracy": float(accuracy_score(y_test, y_pred_stack)),
            "f1_macro": float(f1_score(y_test, y_pred_stack, average='macro', zero_division=0)),
            "f1_micro": float(f1_score(y_test, y_pred_stack, average='micro', zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred_stack, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred_stack, average='macro', zero_division=0)),
        }
    }
    
    # Métriques base models
    for model_name_base, y_pred_base in base_predictions.items():
        results[model_name_base] = {
            "accuracy": float(accuracy_score(y_test, y_pred_base)),
            "f1_macro": float(f1_score(y_test, y_pred_base, average='macro', zero_division=0)),
        }
    
    # Affichage 
    print(f"\n{'='*70}")
    print(f"📊 RÉSULTATS : {model_name}")
    print(f"{'='*70}")
    print(f"\n🎯 STACKING ENSEMBLE:")
    print(f"   Accuracy       : {results['stacking']['accuracy']:.4f}")
    print(f"   F1 Macro       : {results['stacking']['f1_macro']:.4f}")
    print(f"   F1 Weighted    : {results['stacking']['f1_weighted']:.4f}")
    print(f"   Precision Macro: {results['stacking']['precision_macro']:.4f}")
    print(f"   Recall Macro   : {results['stacking']['recall_macro']:.4f}")
    
    print(f"\n📈 COMPARAISON AVEC BASE MODELS:")
    for model_name_base in base_predictions.keys():
        print(f"   {model_name_base:15s}: Acc={results[model_name_base]['accuracy']:.4f}, "
              f"F1={results[model_name_base]['f1_macro']:.4f}")
    
    # Calculer gains
    print(f"\n💰 GAINS DU STACKING:")
    for model_name_base in base_predictions.keys():
        gain_acc = results['stacking']['accuracy'] - results[model_name_base]['accuracy']
        gain_f1 = results['stacking']['f1_macro'] - results[model_name_base]['f1_macro']
        print(f"   vs {model_name_base:10s}: Acc {gain_acc:+.4f} ({gain_acc*100:+.1f}%), "
              f"F1 {gain_f1:+.4f} ({gain_f1*100:+.1f}%)")
    
    # Sauvegardes
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / f"{model_name}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        cm = confusion_matrix(y_test, y_pred_stack, labels=labels_order)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels_order, yticklabels=labels_order)
        plt.xlabel('Prédictions')
        plt.ylabel('Vérités')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_dir / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close()
        
        # Classification report
        report = classification_report(y_test, y_pred_stack, labels=labels_order,
                                      target_names=[str(l) for l in labels_order],
                                      zero_division=0, output_dict=True)
        with open(save_dir / f"{model_name}_classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Comparaison graphique
        plot_stacking_comparison(results, save_path=save_dir / f"{model_name}_comparison.png")
        
        print(f"\n💾 Résultats sauvegardés dans : {save_dir}")
    
    return results


def plot_stacking_comparison(results, save_path=None):
    """Graphique comparant stacking vs base models"""
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_macros = [results[m]['f1_macro'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    colors = ['green' if m == 'stacking' else 'steelblue' for m in models]
    bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Comparaison Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(accuracies) - 0.05, max(accuracies) + 0.02])
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1 Macro
    bars2 = axes[1].bar(models, f1_macros, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('F1 Macro', fontsize=12)
    axes[1].set_title('Comparaison F1 Macro', fontsize=14, fontweight='bold')
    axes[1].set_ylim([min(f1_macros) - 0.05, max(f1_macros) + 0.02])
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, f1_macros):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# INTÉGRATION

def train_and_evaluate_stacking(rf_model, xgb_model, ft_model,
                                X_train, y_train, X_test, y_test,
                                target_name, save_dir=None,
                                n_folds=5, random_state=42):
    """
    Pipeline complet : entraîne et évalue le stacking ensemble.
    
    Args:
        rf_model, xgb_model, ft_model: Modèles base (déjà entraînés ou pas)
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        target_name: Nom de la target (pour sauvegardes)
        save_dir: Dossier pour résultats
        n_folds: Nombre de folds CV
        random_state: Seed
        
    Returns:
        stacking_model: Modèle entraîné
        results: Dict avec métriques
    """
    print(f"\n{'='*70}")
    print(f"🚀 STACKING ENSEMBLE POUR : {target_name}")
    print(f"{'='*70}")
    
    base_models = {
        "RandomForest": rf_model,
        "XGBoost": xgb_model,
        "FTTransformer": ft_model
    }
    
    # Initialiser stacking
    stacking = StackingEnsemble(
        base_models=base_models,
        meta_learner=LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=random_state,
            C=1.0 ,
            class_weight= 'balanced' 
        ),
        n_folds=n_folds,
        random_state=random_state,
        verbose=True
    )
    
    stacking.fit(X_train, y_train)
    
    results = evaluate_stacking(
        stacking, X_test, y_test,
        model_name=f"Stacking_{target_name}",
        save_dir=save_dir
    )
    
    return stacking, results

