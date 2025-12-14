#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================================
# INPUTS
# ============================================================================
BASE_PATH = Path("data/interim/aggregate/matches_soloq_with_teamcomp_mastery_star_rank_plus_hero.parquet")
CTX_PATH  = Path("data/interim/aggregate/match_context_map.parquet")

# Nur nötig, falls hero_puuid nicht schon im BASE_PATH drin ist
PART_PATH = Path("data/interim/aggregate/participants_soloq_clean.parquet")

# OUTPUT (neue Datei, nichts wird überschrieben)
OUT_PATH  = Path("data/interim/aggregate/hero_dataset.parquet")

# ============================================================================
# Helpers
# ============================================================================
def ensure_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

def coerce_bool_to_int(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype("int8")
    return s.fillna(0).astype(int).astype("int8")

# ============================================================================
# Main
# ============================================================================
def main():
    ensure_exists(BASE_PATH)
    ensure_exists(CTX_PATH)

    print("Loading base:", BASE_PATH)
    df = pd.read_parquet(BASE_PATH)
    print("Base shape:", df.shape)

    print("Loading context:", CTX_PATH)
    ctx = pd.read_parquet(CTX_PATH)
    print("Context shape:", ctx.shape)

    # --- Sanity: unique matchId
    if df["matchId"].duplicated().any():
        raise ValueError("Base dataset has duplicate matchId rows (should be 1 row per match).")
    if ctx["matchId"].duplicated().any():
        raise ValueError("Context map has duplicate matchId rows (should be 1 row per match).")

    # --- Required hero columns check
    required_hero = ["hero_teamId", "hero_is_blue", "blue_win", "hero_teamPosition", "hero_championId"]
    missing = [c for c in required_hero if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in base dataset: {missing}")

    df = df.copy()

    # ------------------------------------------------------------------------
    # Build hero_win (target) from hero perspective
    # ------------------------------------------------------------------------
    df["blue_win_int"] = coerce_bool_to_int(df["blue_win"])

    df["hero_is_blue"] = df["hero_is_blue"].fillna(0).astype("int8")
    if "hero_is_red" not in df.columns:
        df["hero_is_red"] = (1 - df["hero_is_blue"]).astype("int8")
    else:
        df["hero_is_red"] = df["hero_is_red"].fillna(0).astype("int8")

    df["hero_win"] = np.where(df["hero_is_blue"] == 1, df["blue_win_int"], 1 - df["blue_win_int"]).astype("int8")

    # ------------------------------------------------------------------------
    # Ensure hero_puuid exists (DEBUG only, exclude from training later)
    # ------------------------------------------------------------------------
    if "hero_puuid" not in df.columns:
        ensure_exists(PART_PATH)
        print("hero_puuid not in base -> reconstructing from participants:", PART_PATH)

        # Minimal columns for a safe join
        part_cols = ["matchId", "puuid", "teamId", "teamPosition", "championId"]
        dfp = pd.read_parquet(PART_PATH, columns=part_cols)

        # Rename to align with hero_* keys
        dfp = dfp.rename(columns={
            "puuid": "hero_puuid",
            "teamId": "hero_teamId",
            "teamPosition": "hero_teamPosition",
            "championId": "hero_championId",
        })

        # Join keys must uniquely identify the hero row
        hero_keys = ["matchId", "hero_teamId", "hero_teamPosition", "hero_championId"]

        # Sanity: within participants, per match there should be max 1 row for exact hero_keys
        dup = dfp.duplicated(subset=hero_keys, keep=False)
        if dup.any():
            # This would be unusual; fail hard to avoid silent wrong IDs
            ex = dfp.loc[dup, hero_keys + ["hero_puuid"]].head(20)
            raise ValueError(
                "Duplicate hero-key rows in participants (cannot uniquely recover hero_puuid).\n"
                f"Sample:\n{ex}"
            )

        hero_map = dfp[hero_keys + ["hero_puuid"]].copy()

        before = df.shape[0]
        df = df.merge(hero_map, on=hero_keys, how="left", validate="one_to_one")
        assert df.shape[0] == before

        miss = df["hero_puuid"].isna().mean()
        print("hero_puuid missing fraction after reconstruction:", miss)
        if miss > 0:
            # fail hard: we expect 0.0 since we merged hero_* from same participants source originally
            raise ValueError(f"hero_puuid could not be recovered for {miss:.2%} of matches (expected 0%).")
    else:
        print("hero_puuid already present in base parquet.")

    # ------------------------------------------------------------------------
    # Merge in match context (time, queue, mode, bans, etc.)
    # ------------------------------------------------------------------------
    out = df.merge(ctx, on="matchId", how="left", validate="one_to_one")
    print("After context merge:", out.shape)

    miss_ctx = out["gameStartTimestamp"].isna().mean() if "gameStartTimestamp" in out.columns else None
    print("Missing context fraction (gameStartTimestamp isna mean):", miss_ctx)

    # convenience column
    if "season_year" in out.columns:
        out["season_year_current_game"] = out["season_year"].astype("Int64")

    # ------------------------------------------------------------------------
    # OPTIONAL sanity checks (do not drop rows, only warn)
    # ------------------------------------------------------------------------
    if "queueId" in out.columns:
        frac_not_420 = (out["queueId"] != 420).mean()
        if frac_not_420 > 0:
            print(f"WARNING: queueId != 420 fraction: {frac_not_420:.4%}")

    if "mapId" in out.columns:
        frac_not_11 = (out["mapId"] != 11).mean()
        if frac_not_11 > 0:
            print(f"WARNING: mapId != 11 fraction: {frac_not_11:.4%}")

    # ------------------------------------------------------------------------
    # Final write
    # ------------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print("\nWROTE:", OUT_PATH)
    print("Final shape:", out.shape)

    preview_cols = [c for c in [
        "matchId",
        "hero_puuid",
        "hero_win",
        "blue_win",
        "hero_teamId",
        "hero_is_blue",
        "hero_teamPosition",
        "hero_championId",
        "hero_platform",
        "hero_patch",
        "season_year",
        "gameStartTimestamp",
        "queueId",
        "mapId",
        "gameMode",
        "has_bans",
    ] if c in out.columns]

    print("\nPreview:")
    print(out[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
