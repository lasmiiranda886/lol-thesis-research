#!/usr/bin/env python3
import pandas as pd
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_load(path: str):
    p = Path(path)
    print(f"\n[{path}]")
    if not p.exists():
        print("  -> Datei NICHT gefunden")
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"  -> Fehler beim Laden: {e}")
        return None
    print(f"  -> shape: {df.shape}")
    return df


def analyze_rank_cache(df_rank: pd.DataFrame, label: str):
    print_header(f"RANK-CACHE ({label})")
    if df_rank is None or df_rank.empty:
        print("  -> leer oder nicht vorhanden")
        return

    # Zeile gilt als "hat Rank", wenn mind. eins von tier/rank_div/leaguePoints nicht NaN ist
    cols_rank_info = [c for c in ["tier", "rank_div", "leaguePoints"] if c in df_rank.columns]
    if cols_rank_info:
        df_rank = df_rank.copy()
        df_rank["has_rank"] = df_rank[cols_rank_info].notna().any(axis=1)
    else:
        df_rank["has_rank"] = False

    total = len(df_rank)
    nonempty = int(df_rank["has_rank"].sum())
    uniq_players = df_rank[["puuid", "platform"]].drop_duplicates().shape[0]

    print(f"  Gesamtzeilen Rank-Cache:         {total}")
    print(f"  Davon mit Rank-Infos:            {nonempty} ({nonempty / total:.1%})")
    print(f"  Eindeutige (puuid, platform):    {uniq_players}")

    print("\n  Rank-Infos pro Plattform:")
    per_plat = df_rank.groupby("platform")["has_rank"].agg(
        total="count",
        with_rank="sum"
    )
    per_plat["pct_with_rank"] = per_plat["with_rank"] / per_plat["total"]
    print(per_plat.sort_values("total", ascending=False))


def analyze_mastery_cache(df_mast: pd.DataFrame, label: str):
    print_header(f"MASTERY-CACHE ({label})")
    if df_mast is None or df_mast.empty:
        print("  -> leer oder nicht vorhanden")
        return

    df_mast = df_mast.copy()
    cols_mast_info = [c for c in ["cm_level", "cm_points"] if c in df_mast.columns]
    if cols_mast_info:
        df_mast["has_mastery"] = df_mast[cols_mast_info].notna().any(axis=1)
    else:
        df_mast["has_mastery"] = False

    total = len(df_mast)
    nonempty = int(df_mast["has_mastery"].sum())
    uniq_pairs = df_mast[["puuid", "platform", "championId"]].drop_duplicates().shape[0]

    print(f"  Gesamtzeilen Mastery-Cache:          {total}")
    print(f"  Davon mit Mastery-Infos:             {nonempty} ({nonempty / total:.1%})")
    print(f"  Eindeutige (puuid, platform, champ): {uniq_pairs}")

    print("\n  Mastery-Infos pro Plattform:")
    per_plat = df_mast.groupby("platform")["has_mastery"].agg(
        total="count",
        with_mastery="sum"
    )
    per_plat["pct_with_mastery"] = per_plat["with_mastery"] / per_plat["total"]
    print(per_plat.sort_values("total", ascending=False))


def main():
    # ---------------------------------------------------------------------
    # 1) MATCHES / PARTICIPANTS
    # ---------------------------------------------------------------------
    print_header("MATCHES / PARTICIPANTS")

    # 1a) aggregate/participants_soloq.parquet (falls vorhanden)
    df_soloq = safe_load("data/interim/aggregate/participants_soloq.parquet")
    if df_soloq is not None:
        print("\n[participants_soloq]")
        n_rows = len(df_soloq)
        n_matches = df_soloq["matchId"].nunique() if "matchId" in df_soloq.columns else None
        print(f"  Zeilen (participants_soloq): {n_rows}")
        if n_matches is not None:
            print(f"  Eindeutige Matches:        {n_matches}")

        if {"matchId", "platform"} <= set(df_soloq.columns):
            mpp = df_soloq.groupby("platform")["matchId"].nunique().sort_values(ascending=False)
            print("\n  Matches pro Plattform (unique matchId):")
            print(mpp)

    # 1b) participants_stream_dataset (alle Parts, robust gegen kaputte Parts)
    parts_dir = Path("data/interim/participants_stream_dataset/parts")
    if parts_dir.exists():
        print_header("participants_stream_dataset (Parts)")
        files = sorted(parts_dir.glob("*.parquet"))
        print(f"  Gefundene Part-Dateien: {len(files)}")

        total_rows = 0
        uniq_matches_parts = []
        good_parts = 0
        skipped_parts = 0

        for fp in files:
            try:
                df_part = pd.read_parquet(fp, columns=["matchId", "platform"])
            except Exception as e:
                print(f"  -> SKIP {fp.name}: hat kein (matchId, platform) Schema ({e})")
                skipped_parts += 1
                continue

            good_parts += 1
            total_rows += len(df_part)
            uniq_matches_parts.append(df_part.drop_duplicates(["matchId", "platform"]))

        print(f"\n  Verwendbare Parts (mit matchId+platform): {good_parts}")
        print(f"  Übersprungene Parts (anderes Schema):     {skipped_parts}")

        if uniq_matches_parts:
            df_m = pd.concat(uniq_matches_parts, ignore_index=True).drop_duplicates(["matchId", "platform"])
            print(f"\n  Gesamtzeilen (Summe über verwendbare Parts): {total_rows}")
            print(f"  Eindeutige (matchId, platform):              {df_m.shape[0]}")

            mpp_stream = df_m.groupby("platform")["matchId"].nunique().sort_values(ascending=False)
            print("\n  Matches pro Plattform (unique matchId) [Stream-Parts]:")
            print(mpp_stream)
        else:
            print("\n  Keine verwendbaren Parts mit matchId+platform gefunden.")
    else:
        print("\n[data/interim/participants_stream_dataset/parts] nicht gefunden.")

    # ---------------------------------------------------------------------
    # 2) RANK-CACHES
    # ---------------------------------------------------------------------
    df_rank_main = safe_load("data/cache/rank_by_puuid.parquet")
    analyze_rank_cache(df_rank_main, "data/cache/rank_by_puuid.parquet")

    # Optional: dev-rank-cache, falls vorhanden
    df_rank_dev = safe_load("data/cache_dev/rank_by_puuid.parquet")
    if df_rank_dev is not None:
        analyze_rank_cache(df_rank_dev, "data/cache_dev/rank_by_puuid.parquet")

    # ---------------------------------------------------------------------
    # 3) MASTERY-CACHES
    # ---------------------------------------------------------------------
    df_mast_main = safe_load("data/cache/mastery_by_puuid.parquet")
    analyze_mastery_cache(df_mast_main, "data/cache/mastery_by_puuid.parquet")

    # Optional: dev-mastery-cache, falls vorhanden
    df_mast_dev = safe_load("data/cache_dev/mastery_by_puuid.parquet")
    if df_mast_dev is not None:
        analyze_mastery_cache(df_mast_dev, "data/cache_dev/mastery_by_puuid.parquet")


if __name__ == "__main__":
    main()
