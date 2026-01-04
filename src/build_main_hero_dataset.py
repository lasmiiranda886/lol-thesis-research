"""
Build Main-Hero Dataset
=======================
Creates a hero dataset where heroes are selected based on:
- Minimum 50 games in dataset
- At least 70% of games on their top 3 champions

This selects "mains" / "OTPs" who specialize in few champions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

MIN_GAMES = 50
MIN_CONCENTRATION = 0.70  # 70% on top 3 champions

DATA_DIR = Path('data/interim/aggregate')

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("BUILD MAIN-HERO DATASET")
    print(f"Started: {datetime.now()}")
    print("="*70)
    print(f"\nCriteria: >= {MIN_GAMES} games, >= {MIN_CONCENTRATION*100:.0f}% on top 3 champions")

    # 1. Load participants
    print("\n" + "="*60)
    print("1. LOADING DATA")
    print("="*60)

    participants = pd.read_parquet(DATA_DIR / 'participants_global_2025.parquet')
    print(f"Total matches: {participants['matchId'].nunique():,}")
    print(f"Total players: {participants['puuid'].nunique():,}")

    # Load existing train/test split to maintain consistency
    train_random = pd.read_parquet(DATA_DIR / 'hero_dataset_random_train_final.parquet')
    test_random = pd.read_parquet(DATA_DIR / 'hero_dataset_random_test_final.parquet')

    train_matches = set(train_random['matchId'].unique())
    test_matches = set(test_random['matchId'].unique())

    print(f"Train matches (from random split): {len(train_matches):,}")
    print(f"Test matches (from random split): {len(test_matches):,}")

    # 2. Identify Main-Heroes
    print("\n" + "="*60)
    print("2. IDENTIFYING MAIN-HEROES")
    print("="*60)

    # Count games per player
    games_per_player = participants.groupby('puuid').size()
    eligible_puuids = set(games_per_player[games_per_player >= MIN_GAMES].index)
    print(f"Players with >= {MIN_GAMES} games: {len(eligible_puuids):,}")

    eligible = participants[participants['puuid'].isin(eligible_puuids)]

    # Calculate champion concentration
    champ_counts = eligible.groupby(['puuid', 'championName']).size().reset_index(name='count')

    def top3_concentration(group):
        sorted_counts = group.sort_values('count', ascending=False)
        top3_sum = sorted_counts['count'].head(3).sum()
        total = sorted_counts['count'].sum()
        return top3_sum / total

    print("Calculating champion concentration...")
    concentrations = champ_counts.groupby('puuid').apply(top3_concentration, include_groups=False)
    main_puuids = set(concentrations[concentrations >= MIN_CONCENTRATION].index)

    print(f"Main-Heroes (>= {MIN_CONCENTRATION*100:.0f}% Top3): {len(main_puuids):,}")

    # Get stats for main-heroes
    main_player_stats = games_per_player[games_per_player.index.isin(main_puuids)]
    print(f"  Avg games: {main_player_stats.mean():.1f}")
    print(f"  Avg concentration: {concentrations[concentrations.index.isin(main_puuids)].mean()*100:.1f}%")

    # 3. Select Main-Heroes from matches
    print("\n" + "="*60)
    print("3. SELECTING HEROES FROM MATCHES")
    print("="*60)

    # Filter to matches with at least one main-hero
    main_hero_matches = participants[participants['puuid'].isin(main_puuids)]

    # For each match, select the main-hero with highest games (most experienced)
    # If multiple main-heroes in same match, pick one with most games
    main_hero_matches = main_hero_matches.copy()
    main_hero_matches['player_games'] = main_hero_matches['puuid'].map(games_per_player)

    # Group by match and select hero with most games
    def select_best_hero(group):
        # Sort by games (descending), take first
        return group.sort_values('player_games', ascending=False).iloc[0]

    print("Selecting best main-hero per match...")
    hero_rows = main_hero_matches.groupby('matchId').apply(select_best_hero, include_groups=False)
    hero_rows = hero_rows.reset_index()  # Keep matchId as column

    print(f"Total main-hero matches: {len(hero_rows):,}")

    # Split into train/test based on existing split
    train_heroes = hero_rows[hero_rows['matchId'].isin(train_matches)]
    test_heroes = hero_rows[hero_rows['matchId'].isin(test_matches)]

    print(f"Train: {len(train_heroes):,}")
    print(f"Test: {len(test_heroes):,}")

    # 4. Build hero features
    print("\n" + "="*60)
    print("4. BUILDING HERO FEATURES")
    print("="*60)

    # We'll use the same feature columns as the random dataset
    feature_cols = [c for c in train_random.columns if c not in ['matchId']]

    # Create hero dataset by joining with participants data
    # We need to get the same features for our main-heroes

    # Get match-level info for all matches
    all_matches = set(hero_rows['matchId'].unique())
    match_participants = participants[participants['matchId'].isin(all_matches)]

    # Build hero dataset with same structure as random
    print("Building feature matrix from existing random dataset structure...")

    # The easiest way is to merge with the random datasets based on matchId + puuid
    # But those might have different heroes selected, so we need to rebuild features

    # Let's load the base participants with all features
    # and select our main-heroes from there

    # Actually, let's load both random datasets and just filter to matches
    # where our main-hero was selected

    # Check overlap: how many of our main-hero matches are in random with same hero?
    random_all = pd.concat([train_random, test_random])

    # Merge to find matches where main-hero was also selected in random
    merged = hero_rows[['matchId', 'puuid']].merge(
        random_all[['matchId', 'puuid']],
        on=['matchId', 'puuid'],
        how='inner'
    )
    print(f"Matches where main-hero = random hero: {len(merged):,} / {len(hero_rows):,}")

    # For simplicity, let's use these overlapping matches
    # This ensures we have the same feature calculation
    overlap_matches = set(merged['matchId'].unique())

    train_main = random_all[
        (random_all['matchId'].isin(overlap_matches)) &
        (random_all['matchId'].isin(train_matches))
    ].copy()

    test_main = random_all[
        (random_all['matchId'].isin(overlap_matches)) &
        (random_all['matchId'].isin(test_matches))
    ].copy()

    print(f"\nFinal Main-Hero Dataset:")
    print(f"  Train: {len(train_main):,} matches")
    print(f"  Test: {len(test_main):,} matches")

    # 5. Verify main-hero criteria on final dataset
    print("\n" + "="*60)
    print("5. VERIFYING MAIN-HERO CRITERIA")
    print("="*60)

    # Check that all heroes in final dataset are main-heroes
    train_hero_puuids = set(train_main['puuid'].unique())
    test_hero_puuids = set(test_main['puuid'].unique())

    train_are_mains = len(train_hero_puuids & main_puuids) / len(train_hero_puuids) * 100
    test_are_mains = len(test_hero_puuids & main_puuids) / len(test_hero_puuids) * 100

    print(f"Train heroes that are main-heroes: {train_are_mains:.1f}%")
    print(f"Test heroes that are main-heroes: {test_are_mains:.1f}%")

    # Hero stats
    print(f"\n--- Hero Stats ---")
    print(f"Train hero_total_games mean: {train_main['hero_total_games'].mean():.1f}")
    print(f"Test hero_total_games mean: {test_main['hero_total_games'].mean():.1f}")
    print(f"Train hero_personal_champ_games mean: {train_main['hero_personal_champ_games'].mean():.1f}")
    print(f"Test hero_personal_champ_games mean: {test_main['hero_personal_champ_games'].mean():.1f}")

    # 6. Save datasets
    print("\n" + "="*60)
    print("6. SAVING DATASETS")
    print("="*60)

    train_main.to_parquet(DATA_DIR / 'hero_dataset_main_train_final.parquet', index=False)
    test_main.to_parquet(DATA_DIR / 'hero_dataset_main_test_final.parquet', index=False)

    print(f"Saved: hero_dataset_main_train_final.parquet ({len(train_main):,} rows)")
    print(f"Saved: hero_dataset_main_test_final.parquet ({len(test_main):,} rows)")

    # 7. Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nRandom Hero Dataset:")
    print(f"  Train: {len(train_random):,} matches")
    print(f"  Test: {len(test_random):,} matches")

    print(f"\nMain-Hero Dataset (>= {MIN_GAMES}G, >= {MIN_CONCENTRATION*100:.0f}% Top3):")
    print(f"  Train: {len(train_main):,} matches")
    print(f"  Test: {len(test_main):,} matches")

    print(f"\nFiles saved:")
    print(f"  - hero_dataset_random_train_final.parquet")
    print(f"  - hero_dataset_random_test_final.parquet")
    print(f"  - hero_dataset_main_train_final.parquet")
    print(f"  - hero_dataset_main_test_final.parquet")

    print(f"\nFinished: {datetime.now()}")
    print("="*70)

    return train_main, test_main


if __name__ == '__main__':
    train, test = main()
