#!/usr/bin/env python3
"""
Erzeuge Abbildungen zur Modell-Performance für die Präsentation.

Voraussetzung:
- data/models/baseline_results.csv existiert
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")  # ggplot-ähnlicher Style

RESULTS_PATH = "data/models/baseline_results.csv"
OUT_DIR = "reports/figures"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    expected_cols = {
        "experiment",
        "feature_set",
        "model_type",
        "auc_train",
        "auc_test",
        "logloss_train",
        "logloss_test",
        "acc_train",
        "acc_test",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in baseline_results.csv: {missing}")
    return df


def plot_auc_bar(df: pd.DataFrame) -> None:
    """Balkendiagramm: Test-AUC pro Experiment (nach AUC sortiert)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    df_sorted = df.sort_values("auc_test", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_sorted["experiment"], df_sorted["auc_test"])
    ax.set_ylabel("ROC-AUC (Test)")
    ax.set_title("Vergleich der Modelle (Test-ROC-AUC)")
    ax.set_ylim(0.48, max(df_sorted["auc_test"].max() + 0.02, 0.55))
    ax.set_xticklabels(df_sorted["experiment"], rotation=45, ha="right")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "model_auc_bar.png")
    fig.savefig(out_path, dpi=300)
    print(f"Gespeichert: {out_path}")


def plot_train_vs_test_auc(df: pd.DataFrame) -> None:
    """
    Scatterplot: AUC_train vs. AUC_test pro Experiment.
    Diagonale Linie = perfekte Generalisierung.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(df["auc_train"], df["auc_test"])

    # Diagonale
    lo = min(df["auc_train"].min(), df["auc_test"].min())
    hi = max(df["auc_train"].max(), df["auc_test"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel("ROC-AUC (Train)")
    ax.set_ylabel("ROC-AUC (Test)")
    ax.set_title("Train vs. Test ROC-AUC (Overfitting sichtbar)")

    # ein paar wichtige Modelle beschriften
    highlight = [
        "A_Dummy_majority",
        "A_LogReg_L2_C1.0",
        "A_RF_100_depth10",
        "A_HGB_200iters",
        "A_XGB_300_depth8",
    ]
    for _, row in df[df["experiment"].isin(highlight)].iterrows():
        ax.annotate(
            row["experiment"],
            (row["auc_train"], row["auc_test"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "train_vs_test_auc.png")
    fig.savefig(out_path, dpi=300)
    print(f"Gespeichert: {out_path}")


def plot_logloss_bar(df: pd.DataFrame) -> None:
    """Balkendiagramm: Test-LogLoss pro Experiment (kleiner ist besser)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    df_sorted = df.sort_values("logloss_test", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_sorted["experiment"], df_sorted["logloss_test"])
    ax.set_ylabel("Log-Loss (Test)")
    ax.set_title("Vergleich der Modelle (Test-Log-Loss)")
    ax.set_xticklabels(df_sorted["experiment"], rotation=45, ha="right")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "model_logloss_bar.png")
    fig.savefig(out_path, dpi=300)
    print(f"Gespeichert: {out_path}")


def plot_auc_by_feature_set(df: pd.DataFrame) -> None:
    """
    Boxplot: Verteilung der Test-AUCs nach Feature-Set (A/B/C).
    Zeigt, ob komplexere Features wirklich helfen.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    groups = []
    labels = []
    for fs in sorted(df["feature_set"].unique()):
        groups.append(df.loc[df["feature_set"] == fs, "auc_test"].values)
        labels.append(fs)

    ax.boxplot(groups, labels=labels)
    ax.set_xlabel("Feature-Set")
    ax.set_ylabel("ROC-AUC (Test)")
    ax.set_title("Test-AUC nach Feature-Set (A/B/C)")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "auc_by_feature_set.png")
    fig.savefig(out_path, dpi=300)
    print(f"Gespeichert: {out_path}")


def main() -> None:
    print(f"Lese Ergebnisse aus: {RESULTS_PATH}")
    df = load_results()

    plot_auc_bar(df)
    plot_train_vs_test_auc(df)
    plot_logloss_bar(df)
    plot_auc_by_feature_set(df)


if __name__ == "__main__":
    main()
