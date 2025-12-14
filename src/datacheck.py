#!/usr/bin/env python
"""
Analyse feature importances for all trained experiments.

- Reads data/models/baseline_results.csv
- Re-trains each experiment (same feature_set/model/params)
- Extracts feature importances where possible:
    * LogisticRegression: |coef_|
    * RF / XGB / HGB: feature_importances_ (if available)
- Saves:
    * data/models/feature_importances_long.csv
      (one row per experiment x feature)
    * data/models/feature_importances_aggregated.csv
      (features aggregated across experiments)

At the end it prints:
- Top 30 features overall
- All features whose name contains "top" (for top lane analysis)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Wir importieren die Helfer direkt aus train_model.py
from train_model import (
    DATA_PATH,
    load_data,
    prepare_splits,
    make_model,
)

RESULTS_PATH = Path("data/models/baseline_results.csv")


def get_feature_importances_generic(model, model_type: str, feature_names):
    """
    Gib ein numpy-Array von Importances in derselben Reihenfolge wie feature_names zurück,
    oder None falls das Modell keine Importances anbietet.
    """
    base_model = model

    # LogReg steckt bei dir in einer Pipeline(StandardScaler -> LogisticRegression)
    if model_type.lower() == "logreg" and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if isinstance(step, LogisticRegression):
                base_model = step
                break

    # 1) Logistische Regression: benutze abs(coef_)
    if isinstance(base_model, LogisticRegression) and hasattr(base_model, "coef_"):
        coefs = np.asarray(base_model.coef_, dtype=float)
        # Binary case: shape (1, n_features)
        if coefs.ndim == 2 and coefs.shape[0] == 1:
            coefs = coefs[0]
        return np.abs(coefs)

    # 2) Tree-Modelle: feature_importances_
    if hasattr(base_model, "feature_importances_"):
        fi = np.asarray(base_model.feature_importances_, dtype=float)
        return fi

    return None


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit(f"baseline_results.csv not found at {RESULTS_PATH}")

    results = pd.read_csv(RESULTS_PATH)
    results_sorted = results.sort_values("auc_test", ascending=False).reset_index(drop=True)

    print("Best 3 experiments by AUC_TEST:")
    print(
        results_sorted[["experiment", "feature_set", "model_type", "auc_test", "auc_val"]]
        .head(3)
        .to_string(index=False)
    )

    print("\nLade Dataset …")
    df = load_data(DATA_PATH)

    all_rows = []

    for _, row in results_sorted.iterrows():
        exp_name = row["experiment"]
        feature_set = row["feature_set"]
        model_type = row["model_type"]
        auc_test = float(row["auc_test"])
        params = json.loads(row["params"]) if isinstance(row["params"], str) else {}

        print("\n" + "=" * 80)
        print(f"Re-train experiment: {exp_name}")
        print(f"Feature set: {feature_set}, model: {model_type}, params={params}")

        # gleiche Splits wie im Trainingsskript
        X_train, X_test, X_val, y_train, y_test, y_val, feature_cols = prepare_splits(
            df,
            feature_set=feature_set,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
        )

        model = make_model(model_type, params)
        model.fit(X_train, y_train)

        importances = get_feature_importances_generic(model, model_type, feature_cols)
        if importances is None:
            print(f"{exp_name}: this model type does not expose feature importances.")
            continue

        importances = np.asarray(importances, dtype=float)
        if importances.shape[0] != len(feature_cols):
            print(f"{exp_name}: importance length mismatch (len={importances.shape[0]} vs {len(feature_cols)}).")
            continue

        abs_imp = np.abs(importances)
        total = abs_imp.sum()
        if total == 0:
            norm_imp = abs_imp
        else:
            norm_imp = abs_imp / total

        # Top 20 pro Experiment ausgeben
        order = np.argsort(-norm_imp)
        print(f"\nTop 20 Features für {exp_name}:")
        for idx in order[:20]:
            feat = feature_cols[idx]
            print(f"  {feat:35s} raw={importances[idx]: .4f}  norm={norm_imp[idx]:.4f}")

        # Long-Format für spätere Aggregation
        for feat, raw, norm in zip(feature_cols, importances, norm_imp):
            all_rows.append(
                {
                    "experiment": exp_name,
                    "feature_set": feature_set,
                    "model_type": model_type,
                    "auc_test": auc_test,
                    "feature": feat,
                    "raw_importance": float(raw),
                    "abs_importance": float(abs(raw)),
                    "norm_importance": float(norm),
                    # gewichtete Wichtigkeit: wichtiges Feature in Modell mit hohem AUC zählt mehr
                    "auc_weighted_importance": float(norm * auc_test),
                }
            )

    if not all_rows:
        print("\nNo feature importances collected (evtl. nur Modelle ohne Importances).")
        return

    out_dir = Path("data/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    fi_df = pd.DataFrame(all_rows)
    long_path = out_dir / "feature_importances_long.csv"
    fi_df.to_csv(long_path, index=False)
    print(f"\nPer-Experiment-Importances gespeichert unter: {long_path}")

    # Aggregation über alle Experimente
    agg = (
        fi_df
        .groupby("feature")
        .agg(
            n_experiments=("experiment", "nunique"),
            mean_norm_importance=("norm_importance", "mean"),
            total_auc_weighted=("auc_weighted_importance", "sum"),
        )
        .sort_values("total_auc_weighted", ascending=False)
        .reset_index()
    )

    agg_path = out_dir / "feature_importances_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Aggregierte Feature-Rangliste gespeichert unter: {agg_path}")

    print("\nTop 30 Features über alle Modelle (nach total_auc_weighted):")
    print(agg.head(30).to_string(index=False))

    # Beispiel: alle TOP-Lane-Features anzeigen
    mask_top = agg["feature"].str.contains("top", case=False, na=False)
    print("\nFeatures mit Bezug zu TOP-Lane (Name enthält 'top'):")
    print(agg[mask_top].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
