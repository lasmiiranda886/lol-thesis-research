# src/teamcomp_features.py

"""
Erzeugt Teamcomp-Features pro Match auf Basis von:
- matches_soloq_champs.parquet (Match-Level mit Champion-IDs pro Lane)
- data/static/champion_features.parquet (Champion-Level-Features aus Data Dragon)

Output:
- Match-Level-Datei mit zusätzlichen Teamcomp-Features, z.B.:
  blue_tankiness_sum, red_tankiness_sum, tankiness_diff
  blue_engage_sum,   red_engage_sum,   engage_diff
  blue_ap_share,     red_ap_share,     ap_diff
  blue_late_scale,   red_late_scale,   late_scale_diff
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

LANES: List[str] = ["top", "jungle", "mid", "adc", "supp"]
SIDES: List[str] = ["blue", "red"]


def required_champion_columns() -> List[str]:
    cols = []
    for lane in LANES:
        for side in SIDES:
            cols.append(f"championid_{side}_{lane}")
    return cols


def basic_sanity_check(df: pd.DataFrame) -> None:
    req = ["matchId", "blue_win"] + required_champion_columns()
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten im Input-DataFrame: {missing}")


def load_matches(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Matches-Datei nicht gefunden: {p}")
    df = pd.read_parquet(p)
    basic_sanity_check(df)
    return df


def load_champion_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Champion-Feature-Datei nicht gefunden: {p}")
    df = pd.read_parquet(p)

    # Wir erwarten mind. diese Spalten:
    needed = [
        "championId",
        "championName",
        "tankiness_score",
        "damage_ap_share",
        "damage_ad_share",
        "late_scaling_score",
        "tag_tank",
        "tag_fighter",
        "tag_mage",
        "tag_assassin",
        "tag_marksman",
        "tag_support",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten in champion_features: {missing}")



    # Engage-Score heuristisch aus Rollen ableiten,
    # falls noch nicht existiert.
    if "engage_score" not in df.columns:
        # Roh-Score: Tanks & Supports stark, Fighter mittel, Mage/Assassin etwas
        engage_raw = (
            0.8 * df["tag_tank"]
            + 0.7 * df["tag_support"]
            + 0.5 * df["tag_fighter"]
            + 0.3 * df["tag_mage"]
            + 0.3 * df["tag_assassin"]
        )
        # Auf 0–1 normalisieren (Maximalwert ca. 0.8+0.7+0.5+0.3+0.3 = 2.6)
        max_raw = engage_raw.max()
        if max_raw > 0:
            df["engage_score"] = engage_raw / max_raw
        else:
            df["engage_score"] = 0.0

    return df


def build_teamcomp_features(
    matches: pd.DataFrame, champ_feats: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregiert Champion-Features pro Team & Match.
    Vorgehen (vektorisiert):
      - matches (wide) → long: 1 Zeile pro (matchId, side, lane, championId)
      - Merge mit champ_feats auf championId
      - groupby(matchId, side) und aggregieren (sum/mean)
      - zurückpivotieren: eine Zeile pro Match, blue_*/red_* + Diff-Spalten
    """

    # Wide → Long: 10 Zeilen pro Match (eine pro Lane/Side)
    long_rows = []
    for lane in LANES:
        for side in SIDES:
            col = f"championid_{side}_{lane}"
            tmp = matches[["matchId", col]].copy()
            tmp = tmp.rename(columns={col: "championId"})
            tmp["side"] = side
            tmp["lane"] = lane
            long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True)

    # Manche Zeilen können NaN-championId haben → rauswerfen
    long_df = long_df.dropna(subset=["championId"]).copy()
    long_df["championId"] = long_df["championId"].astype(int)

    # Merge mit Champion-Features
    merged = long_df.merge(
        champ_feats,
        on="championId",
        how="left",
        suffixes=("", "_cf"),
    )

    # Safety: Zeilen ohne Champion-Feature (sollte nicht passieren) entfernen
    merged = merged.dropna(subset=["tankiness_score"])

    # Aggregation pro (matchId, side)
    agg = (
        merged.groupby(["matchId", "side"])
        .agg(
            tankiness_sum=("tankiness_score", "sum"),
            ap_share_mean=("damage_ap_share", "mean"),
            late_scale_mean=("late_scaling_score", "mean"),
            engage_sum=("engage_score", "sum"),
        )
        .reset_index()
    )

    # Wide: side → Spalten
    wide = agg.pivot(index="matchId", columns="side")

    # MultiIndex-Spalten in einfache Namen umwandeln
    # (metric, side) -> f"{side}_{metric}", z.B. ("tankiness_sum","blue") -> "blue_tankiness_sum"
    wide.columns = [f"{side}_{metric}" for metric, side in wide.columns]
    wide = wide.reset_index()

    # Nach dem Umbenennen haben wir Spalten wie:
    #   blue_tankiness_sum, red_tankiness_sum,
    #   blue_ap_share_mean, red_ap_share_mean,
    #   blue_late_scale_mean, red_late_scale_mean,
    #   blue_engage_sum, red_engage_sum

    # Mean-Spalten für AP-Share und Late-Scaling in "schöne" Namen umbenennen
    wide = wide.rename(
        columns={
            "blue_ap_share_mean": "blue_ap_share",
            "red_ap_share_mean": "red_ap_share",
            "blue_late_scale_mean": "blue_late_scale",
            "red_late_scale_mean": "red_late_scale",
        }
    )

    # Diff-Spalten berechnen
    wide["tankiness_diff"] = wide["blue_tankiness_sum"] - wide["red_tankiness_sum"]
    wide["ap_diff"] = wide["blue_ap_share"] - wide["red_ap_share"]
    wide["late_scale_diff"] = wide["blue_late_scale"] - wide["red_late_scale"]
    wide["engage_diff"] = wide["blue_engage_sum"] - wide["red_engage_sum"]


    # Diff-Spalten berechnen
    wide["tankiness_diff"] = (
        wide["blue_tankiness_sum"] - wide["red_tankiness_sum"]
    )
    wide["ap_diff"] = wide["blue_ap_share"] - wide["red_ap_share"]
    wide["late_scale_diff"] = (
        wide["blue_late_scale"] - wide["red_late_scale"]
    )
    wide["engage_diff"] = wide["blue_engage_sum"] - wide["red_engage_sum"]

    # Mit originalem Match-DataFrame mergen (um blue_win und Champ-IDs zu behalten)
    result = matches.merge(wide, on="matchId", how="left")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Erzeuge Teamcomp-Features aus Champion-Static-Features und Matchdaten."
    )
    parser.add_argument(
        "--matches-path",
        type=str,
        default="data/interim/aggregate/matches_soloq_champs.parquet",
        help="Pfad zur Match-Level-Datei mit Champion-IDs pro Lane.",
    )
    parser.add_argument(
        "--champion-features-path",
        type=str,
        default="data/static/champion_features.parquet",
        help="Pfad zur Champion-Feature-Tabelle (Parquet).",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/interim/aggregate/matches_soloq_with_teamcomp.parquet",
        help="Zielpfad für die erweiterte Match-Level-Datei.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Lade Matches von {args.matches_path} ...")
    matches = load_matches(args.matches_path)
    print("Matches shape:", matches.shape)

    print(f"Lade Champion-Features von {args.champion_features_path} ...")
    champ_feats = load_champion_features(args.champion_features_path)
    print("Champion-Features shape:", champ_feats.shape)

    print("\nBaue Teamcomp-Features ...")
    result = build_teamcomp_features(matches, champ_feats)
    print("Erweitertes Match-DataFrame shape:", result.shape)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"\nGespeichert unter: {out_path}")


if __name__ == "__main__":
    main()
