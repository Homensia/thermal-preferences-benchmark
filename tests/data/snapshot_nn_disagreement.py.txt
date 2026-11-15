"""Native near-neighbour disagreement analysis for tpb.

For each prediction target (TSV-7, TSV-3, TPV), computes three label-disagreement
statistics on the held-out test split used by `models.py`:

  Primary analysis
      For each test-set observation find its single nearest neighbour (K=1, L2
      distance in the standardised feature space); report the fraction of pairs
      whose labels disagree. This is the paper's information-ceiling estimator.

  Case B — strictly identical feature groups
      Group observations by rounded feature vectors (3 decimals for numerical,
      exact equality for categorical). Report, for groups of size >= 2, the
      fraction of within-group pairs whose labels disagree (macro-averaged over
      clusters, weighted by cluster size).

  Case C — pairs within tight physiological margins
      For every unordered pair of observations, keep only those that (i) share
      exactly the same categorical labels (Âge treated as strict categorical
      equality, matching the notebook) and (ii) differ by at most the following
      per-feature tolerances: Tair ±1 degC, Tout ±1 degC, RH ±5 %, vel
      ±0.3 m/s, clo ±0.3, met ±0.2. These tolerances match those used in
      `analysis/context_votes.ipynb` Cell 11 and were chosen to probe the lower
      bound of predictability under narrow physiological margins. Report the
      fraction of surviving pairs whose labels agree/disagree.

No retraining of any model is performed; the analysis operates purely on the
feature-space + target labels of the cohort already split by `models.py`. Test
indices are read from `{results_dir}/{target}/splits/{target}_split_indices.json`
produced at Phase 2 rerun time.

Outputs (one JSON per target + one summary CSV):

    {output_dir}/nn_disagreement_{target}.json    (detailed counters)
    {output_dir}/nn_disagreement_summary.csv       (3 rows, easy table)

Usage
-----
    python analysis/nn_disagreement.py \\
        --scope full --features_key features_12 \\
        --data_csv Data/ASHRAE_2022_Clean_api.csv \\
        --output_dir rerun_2026-04-28_no_reweighting/diagnostics/nn_ashrae_2022

Assumes the raw cohort CSV (Data/ASHRAE_2022_Clean_api.csv by default) is
reachable from the cwd in which the script runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# tpb package root is the parent of this file — add it to sys.path so
# `config_loader` resolves whether the script is invoked from tpb root or from
# within `analysis/`.
_TPB_ROOT = Path(__file__).resolve().parent.parent
if str(_TPB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TPB_ROOT))

from config_loader import load_config  # noqa: E402

CONFIG = load_config()
TARGETS = CONFIG["data"]["targets"]
FEATURES_ALL = CONFIG["data"]["features"]


# --- tight physiological margins for Case C (aligned with context_votes.ipynb) ---
NUM_TOLERANCES = {
    "Tair": 1.0,
    "Tout": 1.0,
    "RH": 5.0,
    "vel": 0.3,
    "clo": 0.3,
    "Met": 0.2,
}
CATEGORICAL_COLS = {"Sexe", "Season", "Climate", "Building_type", "cooling type", "Âge"}


def _load_cohort(data_csv: Path, target: str, features: list, scope: str,
                 results_dir: Path | None) -> tuple:
    """Load the cohort for NN analysis.

    scope='full'  — use every non-null row of data_csv (default; matches PFE
                    Molka methodology, which operated on the entire 15k-row
                    cohort to maximize the chance of finding exact-feature
                    duplicate clusters for Case B).
    scope='test'  — restrict to the test_idx saved in
                    results_dir/{target}/splits/{target}_split_indices.json.
                    Smaller sample (3 045 rows on the ASHRAE-2022 split);
                    useful when the NN analysis is meant to match the same
                    observations used at evaluation time.
    """
    df = pd.read_csv(data_csv)
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {data_csv}: {missing}")

    if scope == "test":
        if results_dir is None:
            raise ValueError("scope='test' requires --results_dir")
        split_path = Path(results_dir) / target / "splits" / f"{target}_split_indices.json"
        if not split_path.exists():
            raise FileNotFoundError(f"No split file found: {split_path}")
        splits = json.loads(split_path.read_text())
        idx = np.asarray(splits["test_idx"], dtype=int)
        df = df.loc[idx].copy()

    # Keep rows with non-null target + non-null features
    mask = df[target].notna()
    for c in features:
        mask &= df[c].notna() if pd.api.types.is_numeric_dtype(df[c]) else df[c].astype(str).str.strip().ne("")
    df = df[mask].reset_index(drop=True)

    X = df[features].copy()
    y = df[target].to_numpy()
    return X, y


def _encode_features(X: pd.DataFrame) -> np.ndarray:
    """Integer-encode categoricals + standardize numerics; return dense matrix."""
    X = X.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            # Integer codes for categorical (stable across rows)
            X[col] = pd.Categorical(X[col].astype(str).str.strip()).codes.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.to_numpy(dtype=float))
    return Xs


def primary_disagreement(X_std: np.ndarray, y: np.ndarray) -> dict:
    """K=1 nearest-neighbour disagreement rate in the standardized feature space."""
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X_std)
    _, idx = nn.kneighbors(X_std)
    neighbours = idx[:, 1]  # column 0 is the point itself
    n_pairs = len(y)
    disagree = int(np.sum(y != y[neighbours]))
    agree = n_pairs - disagree
    return {
        "n_obs": n_pairs,
        "n_pairs_considered": n_pairs,
        "disagreement_rate": disagree / n_pairs,
        "agreement_rate": agree / n_pairs,
        "n_disagree": disagree,
        "n_agree": agree,
    }


def case_b_identical_groups(X: pd.DataFrame, y: np.ndarray, round_decimals: int = 3) -> dict:
    """Cluster rows by rounded feature vectors; compute two disagreement metrics.

    Two complementary Case B metrics are reported:

    * cluster_agreement_rate: fraction of clusters where all members share the
      same label (*PFE Molka convention*). High value = most identical groups
      are label-consistent.
    * pair_agreement_rate: fraction of within-cluster unordered pairs whose
      labels agree (weights larger clusters more heavily). Complementary view.
    """
    X_round = X.copy()
    for col in X_round.columns:
        if pd.api.types.is_numeric_dtype(X_round[col]):
            X_round[col] = X_round[col].round(round_decimals)
        else:
            X_round[col] = X_round[col].astype(str).str.strip()
    key = X_round.apply(lambda r: tuple(r.values), axis=1)
    df = pd.DataFrame({"key": key, "y": y})
    total_pairs = 0
    total_disagree = 0
    clusters_all_agree = 0
    clusters_total = 0
    clusters = []
    for key_val, group in df.groupby("key"):
        n = len(group)
        if n < 2:
            continue
        clusters_total += 1
        ys = group["y"].to_numpy()
        if len(set(ys)) == 1:
            clusters_all_agree += 1
        disagree = 0
        for i, j in combinations(range(n), 2):
            if ys[i] != ys[j]:
                disagree += 1
        pairs = n * (n - 1) // 2
        total_pairs += pairs
        total_disagree += disagree
        clusters.append({"size": n, "pairs": pairs, "disagree": disagree})
    if total_pairs == 0:
        return {
            "n_clusters": 0,
            "n_pairs_considered": 0,
            "cluster_agreement_rate": float("nan"),
            "cluster_disagreement_rate": float("nan"),
            "pair_agreement_rate": float("nan"),
            "pair_disagreement_rate": float("nan"),
        }
    return {
        "n_clusters": clusters_total,
        "n_clusters_all_agree": clusters_all_agree,
        "cluster_agreement_rate": clusters_all_agree / clusters_total,
        "cluster_disagreement_rate": 1.0 - clusters_all_agree / clusters_total,
        "n_pairs_considered": total_pairs,
        "n_disagree_pairs": total_disagree,
        "pair_agreement_rate": 1.0 - total_disagree / total_pairs,
        "pair_disagreement_rate": total_disagree / total_pairs,
    }


def case_c_physio_margins(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Pairs within physiologically plausible per-feature tolerances."""
    n = len(X)
    # Extract per-column arrays to avoid repeated iloc overhead
    num_cols = [c for c in X.columns if c in NUM_TOLERANCES]
    cat_cols = [c for c in X.columns if c in CATEGORICAL_COLS]
    num_arrays = {c: X[c].to_numpy(dtype=float) for c in num_cols}
    cat_arrays = {c: X[c].astype(str).str.strip().to_numpy() for c in cat_cols}

    total_pairs = 0
    total_disagree = 0
    # Vectorized per-row comparison: for row i, compare with rows i+1..n-1 in one pass
    for i in range(n):
        mask = np.ones(n - i - 1, dtype=bool)
        # Categorical: require strict equality
        for c, arr in cat_arrays.items():
            mask &= (arr[i + 1 :] == arr[i])
        # Numerical: require |delta| <= tolerance
        for c, arr in num_arrays.items():
            mask &= (np.abs(arr[i + 1 :] - arr[i]) <= NUM_TOLERANCES[c])
        if not mask.any():
            continue
        js = np.where(mask)[0] + i + 1
        total_pairs += len(js)
        total_disagree += int(np.sum(y[js] != y[i]))
    if total_pairs == 0:
        return {
            "n_pairs_considered": 0,
            "disagreement_rate": float("nan"),
            "agreement_rate": float("nan"),
        }
    return {
        "n_pairs_considered": int(total_pairs),
        "disagreement_rate": total_disagree / total_pairs,
        "agreement_rate": 1.0 - total_disagree / total_pairs,
        "n_disagree": int(total_disagree),
        "n_agree": int(total_pairs - total_disagree),
    }


def main():
    p = argparse.ArgumentParser(
        description="Native NN-disagreement analysis (primary + Case B + Case C)."
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory containing {target}/splits/... (only required for --scope test).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write the JSONs and the summary CSV (default: results_dir or cwd).",
    )
    p.add_argument(
        "--data_csv",
        type=str,
        default="Data/ASHRAE_2022_Clean_api.csv",
        help="Raw cohort CSV (same as used by models.py).",
    )
    p.add_argument(
        "--scope",
        type=str,
        choices=["full", "test"],
        default="full",
        help=("full = all non-null rows (default, matches PFE Molka methodology); "
              "test = restrict to the held-out test split saved by models.py."),
    )
    p.add_argument(
        "--features_key",
        type=str,
        default="features_12",
        help=("Config key for the feature list (default: 'features_12' = 12-variable "
              "Haghirad baseline used in the paper's NN analysis). Use 'features' "
              "for the 17-variable Phase 2 extended space."),
    )
    args = p.parse_args()

    features = CONFIG["data"].get(args.features_key)
    if features is None:
        raise KeyError(f"Unknown features key '{args.features_key}' in config.yaml")

    results_dir = Path(args.results_dir) if args.results_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else (results_dir or Path("."))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_csv = Path(args.data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")

    print(f"Scope: {args.scope}  (features={args.features_key}, n_features={len(features)})")

    summary_rows = []
    for target in TARGETS:
        print(f"\n--- NN disagreement: {target} ---")
        try:
            X, y = _load_cohort(data_csv, target, features, args.scope, results_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[SKIP] {exc}")
            continue

        print(f"  n = {len(X)}, features = {len(features)}")
        X_std = _encode_features(X)

        primary = primary_disagreement(X_std, y)
        print(
            f"  Primary (1-NN): agreement {primary['agreement_rate']:.3f} / "
            f"disagreement {primary['disagreement_rate']:.3f}"
        )

        case_b = case_b_identical_groups(X, y)
        if case_b["n_clusters"]:
            print(
                f"  Case B ({case_b['n_clusters']} clusters, {case_b['n_pairs_considered']} pairs): "
                f"cluster agreement {case_b['cluster_agreement_rate']:.3f} "
                f"({case_b['n_clusters_all_agree']}/{case_b['n_clusters']} all-agree); "
                f"pair agreement {case_b['pair_agreement_rate']:.3f}"
            )
        else:
            print("  Case B: no strictly identical clusters found")

        case_c = case_c_physio_margins(X, y)
        if case_c["n_pairs_considered"]:
            print(
                f"  Case C ({case_c['n_pairs_considered']} pairs within margins): "
                f"agreement {case_c['agreement_rate']:.3f}"
            )
        else:
            print("  Case C: no pairs found within the specified margins")

        detail = {
            "target": target,
            "scope": args.scope,
            "n": int(len(X)),
            "primary": primary,
            "case_b_identical_groups": case_b,
            "case_c_physio_margins": case_c,
            "margins_case_c": NUM_TOLERANCES,
            "features": list(features),
        }
        out_json = output_dir / f"nn_disagreement_{target}.json"
        out_json.write_text(json.dumps(detail, indent=2))
        print(f"  wrote {out_json}")

        summary_rows.append(
            {
                "target": target,
                "scope": args.scope,
                "n": len(X),
                "primary_agreement": primary["agreement_rate"],
                "primary_disagreement": primary["disagreement_rate"],
                "case_b_clusters": case_b.get("n_clusters", 0),
                "case_b_pairs": case_b.get("n_pairs_considered", 0),
                "case_b_cluster_agreement": case_b.get("cluster_agreement_rate", float("nan")),
                "case_b_pair_agreement": case_b.get("pair_agreement_rate", float("nan")),
                "case_c_pairs": case_c.get("n_pairs_considered", 0),
                "case_c_agreement": case_c.get("agreement_rate", float("nan")),
                "case_c_disagreement": 1.0 - case_c.get("agreement_rate", float("nan")) if case_c.get("n_pairs_considered", 0) else float("nan"),
            }
        )

    if summary_rows:
        summary_csv = output_dir / "nn_disagreement_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"\n✅ Summary written to {summary_csv}")


if __name__ == "__main__":
    main()
