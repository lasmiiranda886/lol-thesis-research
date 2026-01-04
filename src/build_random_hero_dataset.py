"""
Build Random Hero Dataset (V8-Random)
======================================
Creates a comparison dataset where the hero is randomly selected
from all 10 players (no 100-games filter).

This demonstrates the impact of the hero selection criteria on model performance.

Usage:
    python src/build_random_hero_dataset.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

BASE_PATH = Path(__file__).parent.parent / 'data/interim/aggregate'
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_region(platform):
    """Map platform to region."""
    europe = ['euw1', 'eun1', 'tr1', 'ru']
    americas = ['na1', 'br1', 'la1', 'la2']
    asia = ['kr', 'jp1']
    # sea includes oc1, me1, etc.

    platform = platform.lower()
    if platform in europe:
        return 'europe'
    elif platform in americas:
        return 'americas'
    elif platform in asia:
        return 'asia'
    else:
        return 'sea'


def build_hero_features(hero_row, match_participants):
    """Build features for a single hero from match data."""
    features = {}

    # Hero player stats
    features['hero_rank_numeric'] = hero_row['rank_numeric'] if pd.notna(hero_row['rank_numeric']) else 4
    features['hero_lp'] = hero_row['leaguePoints'] if pd.notna(hero_row['leaguePoints']) else 50
    features['hero_is_blue_feat'] = 1 if hero_row['teamId'] == 100 else 0
    features['hero_total_games'] = hero_row['total_games'] if pd.notna(hero_row['total_games']) else 100
    features['hero_winrate'] = hero_row['winrate'] if pd.notna(hero_row['winrate']) else 0.5
    features['hero_cm_points_log'] = np.log1p(hero_row['cm_points']) if pd.notna(hero_row['cm_points']) else 10
    features['hero_cm_level_feat'] = hero_row['cm_level'] if pd.notna(hero_row['cm_level']) else 5

    # WR-Rank mismatch
    expected_wr = {1: 0.45, 2: 0.48, 3: 0.50, 4: 0.50, 5: 0.51, 6: 0.52, 7: 0.53, 8: 0.55}
    rank = features['hero_rank_numeric']
    features['hero_wr_rank_mismatch'] = features['hero_winrate'] - expected_wr.get(rank, 0.5)

    # Smurf detection
    features['is_potential_smurf'] = 1 if (features['hero_total_games'] < 100 and features['hero_winrate'] > 0.65) else 0
    features['smurf_score'] = features['hero_winrate'] if features['hero_total_games'] < 100 else 0

    # Region features
    region = get_region(hero_row['platform'])
    features['region_europe'] = 1 if region == 'europe' else 0
    features['region_americas'] = 1 if region == 'americas' else 0
    features['region_asia'] = 1 if region == 'asia' else 0
    features['region_sea'] = 1 if region == 'sea' else 0

    # Team composition (simplified - just champion IDs)
    hero_team = match_participants[match_participants['teamId'] == hero_row['teamId']]
    enemy_team = match_participants[match_participants['teamId'] != hero_row['teamId']]

    features['hero_championId'] = hero_row['championId']
    features['hero_teamId'] = hero_row['teamId']

    # Store champion IDs for later feature calculation
    for i, (_, row) in enumerate(hero_team.iterrows()):
        features[f'ally_{i}_championId'] = row['championId']
    for i, (_, row) in enumerate(enemy_team.iterrows()):
        features[f'enemy_{i}_championId'] = row['championId']

    # Target
    features['hero_win'] = 1 if hero_row['win'] else 0

    # Metadata
    features['matchId'] = hero_row['matchId']
    features['platform'] = hero_row['platform']
    features['hero_puuid'] = hero_row['puuid']

    return features


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("BUILD RANDOM HERO DATASET (V8-Random)")
    print(f"Started: {datetime.now()}")
    print("=" * 70)

    # Load V8 dataset to get the same matches
    print("\n[1] Loading V8 dataset to identify matches...")
    v8_train = pd.read_parquet(BASE_PATH / 'hero_dataset_train_v8.parquet')
    v8_test = pd.read_parquet(BASE_PATH / 'hero_dataset_test_v8.parquet')

    v8_matches = set(v8_train['matchId'].unique()) | set(v8_test['matchId'].unique())
    print(f"    V8 matches: {len(v8_matches):,}")

    # Load raw participants data
    print("\n[2] Loading participants data...")
    participants = pd.read_parquet(BASE_PATH / 'participants_enriched.parquet')
    print(f"    Total participants: {len(participants):,}")

    # Filter to V8 matches
    participants = participants[participants['matchId'].isin(v8_matches)]
    print(f"    Participants in V8 matches: {len(participants):,}")
    print(f"    Unique matches: {participants['matchId'].nunique():,}")

    # Build random hero dataset
    print("\n[3] Building random hero dataset...")
    print("    (Randomly selecting 1 hero per match - NO 100-games filter)")

    hero_data = []
    matches_processed = 0

    for match_id, match_df in participants.groupby('matchId'):
        if len(match_df) != 10:
            continue

        # RANDOM HERO SELECTION - key difference from V8!
        hero_idx = np.random.randint(0, 10)
        hero_row = match_df.iloc[hero_idx]

        features = build_hero_features(hero_row, match_df)
        hero_data.append(features)

        matches_processed += 1
        if matches_processed % 25000 == 0:
            print(f"    Processed {matches_processed:,} matches...")

    hero_df = pd.DataFrame(hero_data)
    print(f"    Total hero samples: {len(hero_df):,}")

    # Temporal split (same as V8)
    print("\n[4] Temporal train/test split...")
    hero_df = hero_df.sort_values('matchId')  # matchId is roughly temporal

    split_idx = int(len(hero_df) * 0.8)
    train_df = hero_df.iloc[:split_idx].copy()
    test_df = hero_df.iloc[split_idx:].copy()

    print(f"    Train: {len(train_df):,}")
    print(f"    Test: {len(test_df):,}")

    # Compare hero characteristics
    print("\n[5] Hero characteristics comparison...")
    print(f"\n    {'Metric':<25} {'V8 (100+ games)':<20} {'Random':<20}")
    print("    " + "-" * 65)
    print(f"    {'Mean total_games':<25} {382.9:<20.1f} {train_df['hero_total_games'].mean():<20.1f}")
    print(f"    {'Median total_games':<25} {257.0:<20.1f} {train_df['hero_total_games'].median():<20.1f}")
    print(f"    {'Mean winrate':<25} {0.514:<20.3f} {train_df['hero_winrate'].mean():<20.3f}")
    print(f"    {'% with <100 games':<25} {'0%':<20} {(train_df['hero_total_games'] < 100).mean()*100:<19.1f}%")

    # Rank distribution
    print("\n    Rank distribution:")
    rank_names = {1: 'Iron', 2: 'Bronze', 3: 'Silver', 4: 'Gold',
                  5: 'Platinum', 6: 'Emerald', 7: 'Diamond', 8: 'Master+'}

    v8_rank_dist = {1: 5.2, 2: 10.2, 3: 22.7, 4: 22.9, 5: 19.2, 6: 16.6, 7: 3.3, 8: 0.0}
    random_rank_dist = train_df['hero_rank_numeric'].value_counts(normalize=True).sort_index() * 100

    print(f"    {'Rank':<12} {'V8':<12} {'Random':<12}")
    print("    " + "-" * 35)
    for rank in range(1, 9):
        v8_pct = v8_rank_dist.get(rank, 0)
        random_pct = random_rank_dist.get(rank, 0)
        print(f"    {rank_names[rank]:<12} {v8_pct:<11.1f}% {random_pct:<11.1f}%")

    # Define features for model (simplified - hero stats only for quick comparison)
    print("\n[6] Training models for comparison...")

    feature_cols = [
        'hero_rank_numeric', 'hero_lp', 'hero_is_blue_feat',
        'hero_total_games', 'hero_winrate', 'hero_cm_points_log',
        'hero_cm_level_feat', 'hero_wr_rank_mismatch',
        'is_potential_smurf', 'smurf_score',
        'region_europe', 'region_americas', 'region_asia', 'region_sea'
    ]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['hero_win']
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['hero_win']

    # Train LightGBM (fast)
    print("    Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_SEED,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    lgb_acc = accuracy_score(y_test, (lgb_pred > 0.5).astype(int))

    print(f"    LightGBM AUC: {lgb_auc:.4f}, Accuracy: {lgb_acc:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n    {'Dataset':<20} {'AUC':<12} {'Features':<15} {'Hero Selection'}")
    print("    " + "-" * 65)
    print(f"    {'V8 (Original)':<20} {'0.5898':<12} {'48 (full)':<15} {'100+ games'}")
    print(f"    {'V8-Random':<20} {lgb_auc:<12.4f} {'14 (hero only)':<15} {'Random'}")

    print("\n    Note: V8-Random uses only hero features (14) vs V8's full 48 features.")
    print("    The AUC difference shows the impact of hero selection quality.")

    # Save datasets
    print("\n[7] Saving datasets...")
    train_df.to_parquet(BASE_PATH / 'hero_dataset_train_v8_random.parquet', index=False)
    test_df.to_parquet(BASE_PATH / 'hero_dataset_test_v8_random.parquet', index=False)

    with open(BASE_PATH / 'features_v8_random.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"    Saved to {BASE_PATH}/")

    print("\n" + "=" * 70)
    print(f"COMPLETED: {datetime.now()}")
    print("=" * 70)

    return {
        'v8_random_auc': lgb_auc,
        'v8_random_acc': lgb_acc,
        'train_size': len(train_df),
        'test_size': len(test_df)
    }


if __name__ == '__main__':
    results = main()
