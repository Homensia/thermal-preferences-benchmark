"""
Unified dataset loader for thermal comfort prediction.

This module centralizes data loading, cleaning, harmonization and validation for
multiple datasets used in the project:

    - ASHRAE global comfort dataset
    - CEREMA building-level comfort dataset
    - MATHILDE dataset (smart home comfort experiments)
    - External datasets (Moujalled, Hostein) when processed similarly

Its objectives are:
    • Harmonize feature types across datasets  
    • Normalize categorical + numeric formatting  
    • Convert labels to consistent float values  
    • Apply dataset-specific label mappings  
    • Generate dataset-level summary statistics  
    • Provide standardized JSON validation reports  

It ensures reproducibility and consistent preprocessing across all experiments
(classical ML, deep learning, hybrid models, transfer learning).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import json


class UnifiedDataLoader:
    """
    Generic loader for harmonizing feature and label types across datasets.

    This class provides:
        - Consistent casting of numerical/categorical feature types
        - Automatic numeric conversion of target variables
        - Optional label harmonization via a custom mapping dictionary
        - Computation of per-target descriptive statistics
        - Console reports + JSON export of dataset validation results

    Parameters
    ----------
    features : list of str
        Names of features expected in the loaded dataset.
    targets : list of str
        Names of possible target variables.
    label_mappings : dict, optional
        Dictionary of the form:
            { target_name : { raw_label : normalized_label } }
        Used to harmonize heterogeneous label formats across datasets
        (e.g., "-3", -3, -3.0 → all mapped to -3.0).
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
        Load, clean, and validate a comfort dataset.

        Steps
        -----
        1. Load CSV with pandas  
        2. Identify available target variables  
        3. Select features + targets  
        4. Normalize feature types:
             - numeric → coercion to float
             - categorical → clean string formatting  
        5. Convert targets to float + apply label mapping if provided  
        6. Compute statistics (class frequencies, imbalance ratios)  
        7. Print validation report  

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        dataset_name : str, default="unknown"
            Name used in printed / saved reports.

        Returns
        -------
        pandas.DataFrame
            Harmonized dataset.
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
        """
        Print a formatted validation report for a loaded dataset.

        Outputs:
            - Dataset shapes
            - Detected target variables
            - Per-target class distribution
            - Class imbalance warnings

        Parameters
        ----------
        dataset_name : str
            Name of dataset to display.
        """
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
        """
        Save accumulated dataset statistics to a JSON file.

        Parameters
        ----------
        output_path : pathlib.Path
            Path of the JSON file to create (directories auto-created).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        print(f"💾 Statistiques sauvegardées : {output_path}")



def create_label_mappings_from_models(ft_model_dict: dict) -> dict:
    """
    Build label harmonization mappings from trained FTTransformer models.

    Since FT models internally store consistent float-encoded target labels
    (model.le_y_.classes_), this function creates a mapping ensuring that:

        -3 / "-3" / -3.0 → -3.0  
        -2 / "-2" / -2.0 → -2.0  
        etc.

    This allows external datasets (Moujalled, Hostein) to be harmonized before
    evaluation or fine-tuning.

    Parameters
    ----------
    ft_model_dict : dict
        Mapping: { target_name : trained_FTClassifier }

    Returns
    -------
    dict
        Combined label mapping:
        {
            target_name: {
                raw_label_variant → normalized_float_label,
                ...
            },
            ...
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

