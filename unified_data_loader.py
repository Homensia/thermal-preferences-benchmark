import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import json


class UnifiedDataLoader:
    """
    Chargeur de données simplifié pour ASHRAE, CEREMA, MATHILDE.
    
    Fonctions :
    - Harmonisation des types float/string pour les labels
    - Harmonisation des types pour les features
    - Statistiques par target 
    - Rapports de validation
    """
    
    def __init__(self, features: List[str], targets: List[str], 
                 label_mappings: Optional[dict] = None):
        """
        Args:
            features: Liste des features à charger
            targets: Liste des targets possibles
            label_mappings: Mapping pour harmoniser labels entre datasets
                           Ex: {"thermal_sensation": {-3: -3.0, "-3": -3.0}}
        """
        self.features = features
        self.targets = targets
        self.label_mappings = label_mappings or {}
        self.statistics = {}
    
    def load_and_validate(self, csv_path: str, dataset_name: str = "unknown") -> pd.DataFrame:
        """
        Charge et harmonise un dataset.
        
        Args:
            csv_path: Chemin du CSV
            dataset_name: Nom pour le rapport
            
        Returns:
            DataFrame avec types harmonisés
        """
        print(f"\n📂 Chargement: {dataset_name} ({csv_path})")
        
        df = pd.read_csv(csv_path)
        original_rows = len(df)
        
        available_targets = [t for t in self.targets if t in df.columns]
        
        if not available_targets:
            print(f"⚠️  Aucune target trouvée dans {csv_path}")
            print(f"   Targets attendues : {self.targets}")
            print(f"   Colonnes disponibles : {df.columns.tolist()}")
        
        keep_cols = self.features + available_targets
        df = df[keep_cols].copy()
        
        # Harmoniser types FEATURES
        for col in self.features:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numérique → float
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # Catégoriel → string propre 
                df[col] = df[col].astype(str).str.strip()
        
        # Harmoniser types TARGETS (float vs string)
        stats_per_target = {}
        
        for target in available_targets:
            # Convertir en numérique
            df[target] = pd.to_numeric(df[target], errors='coerce')
            
            # harmonisation entre datasets
            if target in self.label_mappings:
                mapping = self.label_mappings[target]
                df[target] = df[target].map(
                    lambda x: mapping.get(x, x) if pd.notna(x) else np.nan
                )
            
            # Calculer statistiques
            unique_vals = sorted(df[target].dropna().unique())
            class_counts = df[target].value_counts().to_dict()
            
            stats_per_target[target] = {
                "n_samples": len(df),
                "n_classes": len(unique_vals),
                "classes": [float(v) for v in unique_vals],
                "class_distribution": {float(k): int(v) for k, v in class_counts.items()}
            }
            
            if class_counts:
                min_count = min(class_counts.values())
                max_count = max(class_counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                stats_per_target[target]["min_class_size"] = int(min_count)
                stats_per_target[target]["max_class_size"] = int(max_count)
                stats_per_target[target]["imbalance_ratio"] = float(imbalance_ratio)
        
        # Sauvegarder statistiques
        self.statistics[dataset_name] = {
            "original_shape": (original_rows, len(df.columns)),
            "final_shape": df.shape,
            "targets_available": available_targets,
            "targets_stats": stats_per_target
        }
        
        # Afficher rapport
        self._print_validation_report(dataset_name)
        
        return df
    
    def _print_validation_report(self, dataset_name: str):
        """Affiche rapport de validation console"""
        stats = self.statistics[dataset_name]
        
        print(f"\n{'='*70}")
        print(f"📊 RAPPORT : {dataset_name}")
        print(f"{'='*70}")
        print(f"Shape : {stats['original_shape']} → {stats['final_shape']}")
        print(f"Targets disponibles : {', '.join(stats['targets_available'])}")
        
        for target, tgt_stats in stats['targets_stats'].items():
            print(f"\n  🎯 {target} :")
            print(f"     Samples : {tgt_stats['n_samples']}")
            print(f"     Classes ({tgt_stats['n_classes']}) : {tgt_stats['classes']}")
            
            # Distribution
            print(f"     Distribution :")
            for cls, count in sorted(tgt_stats['class_distribution'].items()):
                pct = count / tgt_stats['n_samples'] * 100
                print(f"       Classe {cls:4.1f} : {count:5d} ({pct:5.1f}%)")
            
            # déséquilibre
            if 'imbalance_ratio' in tgt_stats:
                ratio = tgt_stats['imbalance_ratio']
                if ratio > 10:
                    print(f"     ⚠️  Déséquilibre sévère (ratio {ratio:.1f}:1)")
                elif ratio > 5:
                    print(f"     ⚠️  Déséquilibre modéré (ratio {ratio:.1f}:1)")
        
        print(f"{'='*70}\n")
    
    def save_statistics(self, output_path: Path):
        """Sauvegarde statistiques en JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        print(f"💾 Statistiques sauvegardées : {output_path}")



def create_label_mappings_from_models(ft_model_dict: dict) -> dict:
    """
    Crée mappings automatiquement depuis les modèles FT entraînés.
    
    Args:
        ft_model_dict: {target: trained_FT_model}
        
    Returns:
        label_mappings: {target: {label_variant: harmonized_label}}
        
    Exemple:
        ft_models = {"thermal_sensation": trained_ft_model}
        mappings = create_label_mappings_from_models(ft_models)
        
        # Résultat
        {
            "thermal_sensation": {
                -3: -3.0, "-3": -3.0, -3.0: -3.0,
                -2: -2.0, "-2": -2.0, -2.0: -2.0,
                ...
            }
        }
    """
    mappings = {}
    
    for target, model in ft_model_dict.items():
        ashrae_classes = set(model.le_y_.classes_)
        
        ashrae_classes_float = set(map(float, ashrae_classes))
        
        target_mapping = {}
        
        for cls in ashrae_classes_float:
            target_mapping[cls] = float(cls)          
            target_mapping[int(cls)] = float(cls)         
            target_mapping[str(int(cls))] = float(cls)    
            target_mapping[str(cls)] = float(cls)         
        
        mappings[target] = target_mapping
    
    return mappings

