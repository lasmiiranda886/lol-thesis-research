#!/usr/bin/env python3
"""
Feature Engineering: Hero Champion-spezifische Winrate

Berechnet für jeden Hero in jedem Match:
- hero_champ_wr: Winrate des Hero auf seinem aktuellen Champion (aus FRÜHEREN Matches)
- hero_champ_games: Anzahl Spiele auf diesem Champion (vor diesem Match)
- hero_overall_wr: Gesamtwinrate des Hero (aus FRÜHEREN Matches)
- hero_overall_games: Anzahl Spiele gesamt (vor diesem Match)

Leakage-frei: Nur Matches VOR dem aktuellen Match werden berücksichtigt.
Smoothing: Beta-Prior für kleine Sample Sizes.

Input:
    data/interim/aggregate/participants_soloq_clean.parquet
    data/interim/aggregate/match_context_map.parquet
    data/interim/aggregate/hero_dataset.parquet

Output:
    data/interim/aggregate/hero_dataset.parquet (erweitert)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Lade alle benötigten Daten."""
    base = Path("data/interim/aggregate")
    
    parts = pd.read_parquet(base / "participants_soloq_clean.parquet")
    ctx = pd.read_parquet(base / "match_context_map.parquet")
    hero = pd.read_parquet(base / "hero_dataset.parquet")
    
    print(f"Participants: {parts.shape}")
    print(f"Match context: {ctx.shape}")
    print(f"Hero dataset: {hero.shape}")
    
    return parts, ctx, hero


def build_player_history(parts: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    """
    Erstelle eine Tabelle mit allen Spieler-Match-Kombinationen inkl. Timestamp.
    """
    # Nur relevante Spalten
    parts_slim = parts[['matchId', 'puuid', 'championId', 'win']].copy()
    ctx_slim = ctx[['matchId', 'gameStartTimestamp']].copy()
    
    # Merge
    df = parts_slim.merge(ctx_slim, on='matchId', how='left')
    
    # Sort by timestamp (wichtig für kumulative Berechnung)
    df = df.sort_values(['puuid', 'gameStartTimestamp']).reset_index(drop=True)
    
    print(f"Player history: {df.shape}")
    print(f"Unique players: {df['puuid'].nunique()}")
    
    return df


def compute_historical_stats(history: pd.DataFrame, alpha: float = 2.0, beta: float = 2.0) -> pd.DataFrame:
    """
    Berechne für jedes Match die HISTORISCHEN Stats (nur frühere Matches).
    
    Verwendet cumsum/cumcount mit shift(1) um das aktuelle Match auszuschliessen.
    
    Smoothing: (wins + alpha) / (games + alpha + beta)
    """
    df = history.copy()
    df['win_int'] = df['win'].astype(int)
    
    # --- Champion-spezifische Stats ---
    # Gruppiere nach Spieler + Champion, sortiert nach Zeit
    df = df.sort_values(['puuid', 'championId', 'gameStartTimestamp']).reset_index(drop=True)
    
    grp_champ = df.groupby(['puuid', 'championId'], sort=False)
    
    # cumsum gibt Wert INKL. aktuellem Match, shift(1) macht es zu "vor diesem Match"
    df['champ_wins_before'] = grp_champ['win_int'].cumsum().groupby([df['puuid'], df['championId']]).shift(1).fillna(0)
    df['champ_games_before'] = grp_champ.cumcount()  # 0-indexed, also bereits "Spiele davor"
    
    # Smoothed Winrate
    df['hero_champ_wr'] = (df['champ_wins_before'] + alpha) / (df['champ_games_before'] + alpha + beta)
    df['hero_champ_games'] = df['champ_games_before'].astype(int)
    
    # --- Overall Stats (alle Champions) ---
    df = df.sort_values(['puuid', 'gameStartTimestamp']).reset_index(drop=True)
    
    grp_player = df.groupby('puuid', sort=False)
    
    df['overall_wins_before'] = grp_player['win_int'].cumsum().groupby(df['puuid']).shift(1).fillna(0)
    df['overall_games_before'] = grp_player.cumcount()
    
    df['hero_overall_wr'] = (df['overall_wins_before'] + alpha) / (df['overall_games_before'] + alpha + beta)
    df['hero_overall_games'] = df['overall_games_before'].astype(int)
    
    # Cleanup
    result = df[['matchId', 'puuid', 'championId', 
                 'hero_champ_wr', 'hero_champ_games',
                 'hero_overall_wr', 'hero_overall_games']].copy()
    
    return result


def merge_to_hero_dataset(hero: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Merge die berechneten Stats ins Hero Dataset.
    """
    # Rename für Merge
    stats_renamed = stats.rename(columns={
        'puuid': 'hero_puuid',
        'championId': 'hero_championId'
    })
    
    # Merge auf matchId + hero_puuid + hero_championId
    hero_new = hero.merge(
        stats_renamed,
        on=['matchId', 'hero_puuid', 'hero_championId'],
        how='left'
    )
    
    # Check für fehlende Werte
    missing = hero_new['hero_champ_wr'].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have missing hero_champ_wr")
        # Fallback auf Prior
        hero_new['hero_champ_wr'] = hero_new['hero_champ_wr'].fillna(0.5)
        hero_new['hero_champ_games'] = hero_new['hero_champ_games'].fillna(0)
        hero_new['hero_overall_wr'] = hero_new['hero_overall_wr'].fillna(0.5)
        hero_new['hero_overall_games'] = hero_new['hero_overall_games'].fillna(0)
    
    return hero_new


def main():
    print("=" * 60)
    print("Feature Engineering: Hero Champion Winrate")
    print("=" * 60)
    
    # 1. Load data
    parts, ctx, hero = load_data()
    
    # 2. Build player history with timestamps
    print("\nBuilding player history...")
    history = build_player_history(parts, ctx)
    
    # 3. Compute historical stats (leakage-free)
    print("\nComputing historical stats...")
    stats = compute_historical_stats(history, alpha=2.0, beta=2.0)
    
    # Sanity check
    print(f"\nStats sample:")
    print(stats.head(10))
    print(f"\nhero_champ_games distribution:")
    print(stats['hero_champ_games'].describe())
    
    # 4. Merge to hero dataset
    print("\nMerging to hero dataset...")
    hero_new = merge_to_hero_dataset(hero, stats)
    
    # 5. Verify
    print(f"\nNew hero dataset shape: {hero_new.shape}")
    print(f"New columns: {[c for c in hero_new.columns if c not in hero.columns]}")
    
    # Stats
    print(f"\nhero_champ_wr stats:")
    print(hero_new['hero_champ_wr'].describe())
    print(f"\nhero_champ_games stats:")
    print(hero_new['hero_champ_games'].describe())
    
    # 6. Save
    out_path = Path("data/interim/aggregate/hero_dataset.parquet")
    
    # Backup first
    backup_path = out_path.with_suffix('.parquet.bak')
    if out_path.exists():
        import shutil
        shutil.copy(out_path, backup_path)
        print(f"\nBackup saved to: {backup_path}")
    
    hero_new.to_parquet(out_path, index=False)
    print(f"Saved to: {out_path}")
    
    # Final column list
    print(f"\nFinal columns ({len(hero_new.columns)}):")
    print(list(hero_new.columns))


if __name__ == "__main__":
    main()