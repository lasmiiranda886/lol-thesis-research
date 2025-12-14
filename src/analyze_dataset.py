# src/analyze_dataset.py

"""
Analyse des Champ-Select-Datensatzes:
- Welche Spalten gibt es? (Champions + Matchups)
- Verteilung der Zielvariable (blue_win)
- Wie sind Train/Test aufgeteilt (Größe, Klassenverteilung)?
- Basis-Korrelationen mit blue_win (v.a. für matchup_wr_* sinnvoll)
- Verteilung der Matchup-Winrates in Train vs. Test
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/interim/aggregate/matches_soloq_champs_with_matchups.parquet"


def main():
    p = Path(DATA_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {p}")

    df = pd.read_parquet(p)
    print(f"Rohdaten shape: {df.shape}\n")

    print("Spaltenliste:")
    print(df.columns.tolist())
    print()

    # Zielvariable als int (0/1)
    df["blue_win_int"] = df["blue_win"].astype(int)

    # 1) Zielverteilung
    target_counts = df["blue_win"].value_counts()
    target_props = df["blue_win"].value_counts(normalize=True)
    print("Verteilung blue_win (gesamt):")
    print(pd.DataFrame({"count": target_counts, "proportion": target_props}))
    print()

    # 2) Features identifizieren
    champ_cols = [c for c in df.columns if c.startswith("championid_")]
    matchup_cols = [c for c in df.columns if c.startswith("matchup_wr_")]
    other_cols = [c for c in df.columns if c not in champ_cols + matchup_cols + ["matchId", "blue_win", "blue_win_int"]]

    print(f"Anzahl Champion-ID-Spalten: {len(champ_cols)}")
    print(f"Anzahl Matchup-Spalten:     {len(matchup_cols)}")
    print(f"Weitere numerische Spalten: {other_cols}")
    print()

    # 3) Train/Test-Split wie im Modell
    feature_cols_all = champ_cols + other_cols + matchup_cols
    X = df[feature_cols_all]
    y = df["blue_win_int"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Train/Test-Größen:")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test  shape: X={X_test.shape}, y={y_test.shape}")
    print()

    print("blue_win-Verteilung im Training:")
    print(y_train.value_counts(normalize=True).rename("proportion"))
    print()

    print("blue_win-Verteilung im Test:")
    print(y_test.value_counts(normalize=True).rename("proportion"))
    print()

    # 4) Korrelationen matchup_wr_* mit blue_win
    # (Champion-IDs sind nominale IDs, dort ist Pearson-Korrelation wenig sinnvoll)
    if matchup_cols:
        corr_matchups = df[matchup_cols + ["blue_win_int"]].corr(numeric_only=True)["blue_win_int"].sort_values(ascending=False)
        print("Korrelationen (Pearson) von matchup_wr_* mit blue_win:")
        print(corr_matchups)
        print()

    # 5) Verteilungen der Matchup-Winrates in Train vs. Test
    #    -> prüfen, ob sie ähnlich aussehen (kein komischer Shift)
    if matchup_cols:
        desc_train = X_train[matchup_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        desc_test = X_test[matchup_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]

        print("Deskriptive Statistiken der Matchup-Winrates (Train):")
        print(desc_train)
        print()

        print("Deskriptive Statistiken der Matchup-Winrates (Test):")
        print(desc_test)
        print()

        # Optional: Differenzen der Mittelwerte
        diff_means = desc_test["mean"] - desc_train["mean"]
        print("Differenz der Mittelwerte (Test - Train) pro matchup_wr_*:")
        print(diff_means)
        print()

    # 6) (Optional) einfache „Korrelation“ Champion-ID vs. blue_win
    #    Achtung: das ist keine echte Korrelation im statistischen Sinn, weil IDs nominal sind,
    #    aber man sieht zumindest, ob bestimmte IDs stark mit Siegen/Verlusten assoziiert sind.
    champ_winrates = {}
    for col in champ_cols:
        # Winrate der blauen Seite nach Champion-ID in dieser Spalte
        tmp = df[[col, "blue_win_int"]].groupby(col)["blue_win_int"].mean()
        champ_winrates[col] = tmp.describe()

    print("Beispiel: Verteilung der Champion-basierten Winrates (aggregiert je Spalte):")
    for col, stats in champ_winrates.items():
        print(f"\nSpalte: {col}")
        print(stats)
        # um die Ausgabe zu kürzen, brechen wir nach 2-3 Spalten ab
        if col.endswith("mid") or col.endswith("supp"):
            break


if __name__ == "__main__":
    main()
