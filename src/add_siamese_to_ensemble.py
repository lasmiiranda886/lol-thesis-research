#!/usr/bin/env python3
"""
Quick script to add Siamese scores to hero dataset for ensemble
Uses the small Siamese model (best test generalization)
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Config - SMALL version
CONFIG = {
    'emb_dim': 32,
    'hidden_dim': 128,
    'dropout': 0.3,
    'epochs': 25,
    'lr': 0.001,
    'batch_size': 1024,
    'n_folds': 5,
    'early_stopping_patience': 7
}

BASE_PATH = '/Users/teefix/Desktop/lol-data-pipeline/data/interim/aggregate'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SiameseNetwork(nn.Module):
    def __init__(self, n_features, emb_dim=32, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.champion_emb = nn.Embedding(n_features, emb_dim)
        self.team_encoder = nn.Sequential(
            nn.Linear(5 * emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.elo_emb = nn.Embedding(11, 16)
        self.side_emb = nn.Embedding(2, 8)
        compare_input = hidden_dim * 3 + 16 + 8
        self.compare_net = nn.Sequential(
            nn.Linear(compare_input, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, champs, elo, side, return_logits=True):
        batch_size = champs.shape[0]
        emb = self.champion_emb(champs)
        hero_emb = emb[:, :5, :].view(batch_size, -1)
        enemy_emb = emb[:, 5:, :].view(batch_size, -1)
        hero_enc = self.team_encoder(hero_emb)
        enemy_enc = self.team_encoder(enemy_emb)
        diff = torch.abs(hero_enc - enemy_enc)
        elo_bucket = torch.clamp(elo.long(), 0, 10).squeeze(1)
        elo_enc = self.elo_emb(elo_bucket)
        side_enc = self.side_emb(side.squeeze(1))
        combined = torch.cat([hero_enc, enemy_enc, diff, elo_enc, side_enc], dim=1)
        logits = self.compare_net(combined)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def load_data():
    """Load all data for training and test"""
    print("Loading data...")

    # EUW + Enriched 2023-2024
    euw_df = pd.read_parquet(f'{BASE_PATH}/participants_soloq_clean.parquet')
    euw_2023_2024 = euw_df[euw_df['gameVersion'].str.match(r'^1[34]\.')].copy()

    enriched_df = pd.read_parquet(f'{BASE_PATH}/participants_enriched.parquet')
    enriched_2023_2024 = enriched_df[enriched_df['gameVersion'].str.match(r'^1[34]\.')].copy()

    # 2025 data
    participants_2025 = pd.read_parquet(f'{BASE_PATH}/participants_global_2025.parquet')
    train_hero = pd.read_parquet(f'{BASE_PATH}/hero_global_2025_random_train.parquet')
    test_hero = pd.read_parquet(f'{BASE_PATH}/hero_global_2025_random_test.parquet')

    train_match_ids = set(train_hero['matchId'].unique())
    test_match_ids = set(test_hero['matchId'].unique())

    train_2025 = participants_2025[participants_2025['matchId'].isin(train_match_ids)].copy()
    test_2025 = participants_2025[participants_2025['matchId'].isin(test_match_ids)].copy()

    # Combine training
    all_train = pd.concat([euw_2023_2024, enriched_2023_2024, train_2025], ignore_index=True)
    all_train = all_train.drop_duplicates(subset=['matchId', 'puuid'])

    print(f"  Train: {all_train['matchId'].nunique():,} matches")
    print(f"  Test: {test_2025['matchId'].nunique():,} matches")

    return all_train, test_2025, train_match_ids, test_match_ids


def prepare_match_features(participants_df, desc=""):
    """Prepare match-level features - vectorized"""
    print(f"Preparing features {desc}...")

    elo_map = {'IRON': 1, 'BRONZE': 2, 'SILVER': 3, 'GOLD': 4, 'PLATINUM': 5,
               'EMERALD': 6, 'DIAMOND': 7, 'MASTER': 8, 'GRANDMASTER': 9, 'CHALLENGER': 10}
    pos_order = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

    results = []
    start = time.time()

    for match_id, match_df in participants_df.groupby('matchId'):
        if len(match_df) != 10:
            continue

        team_data = {}
        for team_id, team_df in match_df.groupby('teamId'):
            team_df = team_df.sort_values('teamPosition', key=lambda x: x.map(pos_order))
            if len(team_df) != 5:
                continue
            champs = team_df['championName'].tolist()
            elo_val = team_df['rank_tier'].map(elo_map).mean()
            win = team_df['win'].iloc[0]
            team_data[team_id] = {'champs': champs, 'elo': elo_val, 'win': win}

        if len(team_data) != 2:
            continue

        team_ids = list(team_data.keys())
        for i, hero_team_id in enumerate(team_ids):
            enemy_team_id = team_ids[1 - i]
            results.append({
                'matchId': match_id,
                'hero_team_id': hero_team_id,
                'hero_top': team_data[hero_team_id]['champs'][0],
                'hero_jg': team_data[hero_team_id]['champs'][1],
                'hero_mid': team_data[hero_team_id]['champs'][2],
                'hero_bot': team_data[hero_team_id]['champs'][3],
                'hero_sup': team_data[hero_team_id]['champs'][4],
                'enemy_top': team_data[enemy_team_id]['champs'][0],
                'enemy_jg': team_data[enemy_team_id]['champs'][1],
                'enemy_mid': team_data[enemy_team_id]['champs'][2],
                'enemy_bot': team_data[enemy_team_id]['champs'][3],
                'enemy_sup': team_data[enemy_team_id]['champs'][4],
                'elo': team_data[hero_team_id]['elo'],
                'is_blue_side': 1 if hero_team_id == 100 else 0,
                'win': int(team_data[hero_team_id]['win'])
            })

        if len(results) % 100000 == 0:
            print(f"   {len(results)//2:,} matches ({time.time()-start:.0f}s)")

    result_df = pd.DataFrame(results)
    result_df['elo'] = result_df['elo'].fillna(5.0).clip(1, 10)
    print(f"   Done: {len(result_df):,} samples from {len(result_df)//2:,} matches")
    return result_df


def create_encoder(match_features_df):
    pos_types = ['TOP', 'JG', 'MID', 'BOT', 'SUP']
    cols = ['hero_top', 'hero_jg', 'hero_mid', 'hero_bot', 'hero_sup',
            'enemy_top', 'enemy_jg', 'enemy_mid', 'enemy_bot', 'enemy_sup']
    all_combos = set()
    for col, pos in zip(cols, pos_types * 2):
        for champ in match_features_df[col].unique():
            all_combos.add(f"{pos}_{champ}")
    encoder = LabelEncoder()
    encoder.fit(list(all_combos))
    return encoder


def encode_features(match_df, encoder):
    pos_map = {'hero_top': 'TOP', 'hero_jg': 'JG', 'hero_mid': 'MID',
               'hero_bot': 'BOT', 'hero_sup': 'SUP',
               'enemy_top': 'TOP', 'enemy_jg': 'JG', 'enemy_mid': 'MID',
               'enemy_bot': 'BOT', 'enemy_sup': 'SUP'}
    known = set(encoder.classes_)
    encoded = {}
    for col, pos in pos_map.items():
        combos = match_df[col].apply(lambda x: f"{pos}_{x}")
        fallback = [c for c in known if c.startswith(pos + '_')][0]
        combos = combos.apply(lambda x: x if x in known else fallback)
        encoded[col + '_enc'] = encoder.transform(combos)
    return pd.DataFrame(encoded)


def main():
    print("=" * 60)
    print("ADD SIAMESE SCORES TO ENSEMBLE")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    # Load data
    train_data, test_data, train_2025_ids, test_match_ids = load_data()

    # Prepare features
    train_features = prepare_match_features(train_data, "(train)")

    # Create encoder
    encoder = create_encoder(train_features)
    print(f"Encoder: {len(encoder.classes_)} unique combos")

    # Encode
    encoded = encode_features(train_features, encoder)
    X_champs = encoded.values
    X_elo = train_features['elo'].values.reshape(-1, 1)
    X_side = train_features['is_blue_side'].values.reshape(-1, 1)
    y = train_features['win'].values

    n_features = len(encoder.classes_)

    # Cross-validation for OOF predictions
    print("\n" + "=" * 60)
    print("TRAINING SIAMESE WITH CV FOR OOF PREDICTIONS")
    print("=" * 60)

    kfold = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_champs, y)):
        print(f"\n--- Fold {fold + 1}/{CONFIG['n_folds']} ---")
        start = time.time()

        X_train = torch.LongTensor(X_champs[train_idx]).to(device)
        X_elo_train = torch.FloatTensor(X_elo[train_idx]).to(device)
        X_side_train = torch.LongTensor(X_side[train_idx]).to(device)
        y_train = torch.FloatTensor(y[train_idx]).to(device)

        X_val = torch.LongTensor(X_champs[val_idx]).to(device)
        X_elo_val = torch.FloatTensor(X_elo[val_idx]).to(device)
        X_side_val = torch.LongTensor(X_side[val_idx]).to(device)
        y_val = y[val_idx]

        model = SiameseNetwork(n_features, CONFIG['emb_dim'], CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)

        batch_size = CONFIG['batch_size']
        n_batches = (len(train_idx) + batch_size - 1) // batch_size

        best_val_auc = 0
        best_state = None
        patience = 0

        for epoch in range(CONFIG['epochs']):
            model.train()
            indices = torch.randperm(len(train_idx))

            for i in range(n_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]
                optimizer.zero_grad()
                logits = model(X_train[batch_idx], X_elo_train[batch_idx], X_side_train[batch_idx])
                loss = criterion(logits, y_train[batch_idx].unsqueeze(1))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val, X_elo_val, X_side_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val, val_probs)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if patience >= CONFIG['early_stopping_patience']:
                break

        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val, X_elo_val, X_side_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

        oof_predictions[val_idx] = val_probs
        fold_aucs.append(best_val_auc)
        print(f"  Fold {fold+1} AUC: {best_val_auc:.4f} ({(time.time()-start)/60:.1f}min)")

    oof_auc = roc_auc_score(y, oof_predictions)
    print(f"\nOOF AUC: {oof_auc:.4f}")

    # Save OOF predictions for 2025 train matches
    print("\n" + "=" * 60)
    print("SAVING OOF PREDICTIONS")
    print("=" * 60)

    train_features['siamese_oof_score'] = oof_predictions
    train_2025_oof = train_features[train_features['matchId'].isin(train_2025_ids)].copy()

    # Get one prediction per match (from hero team perspective, blue side)
    train_2025_scores = train_2025_oof.groupby('matchId').agg({
        'siamese_oof_score': 'first',
        'hero_team_id': 'first'
    }).reset_index()

    print(f"2025 Train OOF scores: {len(train_2025_scores):,} matches")

    # Train final model on all data
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    X_all = torch.LongTensor(X_champs).to(device)
    X_elo_all = torch.FloatTensor(X_elo).to(device)
    X_side_all = torch.LongTensor(X_side).to(device)
    y_all = torch.FloatTensor(y).to(device)

    final_model = SiameseNetwork(n_features, CONFIG['emb_dim'], CONFIG['hidden_dim'], CONFIG['dropout']).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)

    for epoch in range(CONFIG['epochs']):
        final_model.train()
        indices = torch.randperm(len(y))
        for i in range((len(y) + batch_size - 1) // batch_size):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            logits = final_model(X_all[batch_idx], X_elo_all[batch_idx], X_side_all[batch_idx])
            loss = criterion(logits, y_all[batch_idx].unsqueeze(1))
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1}")

    # Test predictions
    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    test_features = prepare_match_features(test_data, "(test)")
    test_encoded = encode_features(test_features, encoder)

    X_test = torch.LongTensor(test_encoded.values).to(device)
    X_test_elo = torch.FloatTensor(test_features['elo'].values.reshape(-1, 1)).to(device)
    X_test_side = torch.LongTensor(test_features['is_blue_side'].values.reshape(-1, 1)).to(device)
    y_test = test_features['win'].values

    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(X_test, X_test_elo, X_test_side)
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

    test_auc = roc_auc_score(y_test, test_probs)
    print(f"Test AUC: {test_auc:.4f}")

    test_features['siamese_score'] = test_probs
    test_scores = test_features.groupby('matchId').agg({
        'siamese_score': 'first',
        'hero_team_id': 'first'
    }).reset_index()

    print(f"Test scores: {len(test_scores):,} matches")

    # Save scores
    print("\n" + "=" * 60)
    print("SAVING SCORES")
    print("=" * 60)

    # Save as pickle for ensemble integration
    train_scores_dict = {
        'match_scores': dict(zip(train_2025_scores['matchId'], train_2025_scores['siamese_oof_score'])),
        'oof_auc': oof_auc
    }

    test_scores_dict = {
        'match_scores': dict(zip(test_scores['matchId'], test_scores['siamese_score'])),
        'test_auc': test_auc
    }

    with open(f'{BASE_PATH}/nn_siamese_small_train_scores.pkl', 'wb') as f:
        pickle.dump(train_scores_dict, f)

    with open(f'{BASE_PATH}/nn_siamese_small_test_scores.pkl', 'wb') as f:
        pickle.dump(test_scores_dict, f)

    print(f"Saved: nn_siamese_small_train_scores.pkl")
    print(f"Saved: nn_siamese_small_test_scores.pkl")

    # Merge into hero datasets
    print("\n" + "=" * 60)
    print("MERGING INTO HERO DATASETS")
    print("=" * 60)

    # Load hero datasets
    train_hero = pd.read_parquet(f'{BASE_PATH}/hero_global_2025_random_train.parquet')
    test_hero = pd.read_parquet(f'{BASE_PATH}/hero_global_2025_random_test.parquet')

    print(f"Train hero before: {len(train_hero):,} rows, {len(train_hero.columns)} cols")
    print(f"Test hero before: {len(test_hero):,} rows, {len(test_hero.columns)} cols")

    # Add siamese scores
    train_hero['siamese_score'] = train_hero['matchId'].map(train_scores_dict['match_scores'])
    test_hero['siamese_score'] = test_hero['matchId'].map(test_scores_dict['match_scores'])

    # Check coverage
    train_coverage = train_hero['siamese_score'].notna().sum() / len(train_hero) * 100
    test_coverage = test_hero['siamese_score'].notna().sum() / len(test_hero) * 100

    print(f"Train coverage: {train_coverage:.1f}%")
    print(f"Test coverage: {test_coverage:.1f}%")

    # Fill missing with 0.5 (neutral)
    train_hero['siamese_score'] = train_hero['siamese_score'].fillna(0.5)
    test_hero['siamese_score'] = test_hero['siamese_score'].fillna(0.5)

    # Save updated datasets
    train_hero.to_parquet(f'{BASE_PATH}/hero_global_2025_random_train_siamese.parquet', index=False)
    test_hero.to_parquet(f'{BASE_PATH}/hero_global_2025_random_test_siamese.parquet', index=False)

    print(f"\nSaved: hero_global_2025_random_train_siamese.parquet")
    print(f"Saved: hero_global_2025_random_test_siamese.parquet")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Siamese OOF AUC: {oof_auc:.4f}")
    print(f"Siamese Test AUC: {test_auc:.4f}")
    print(f"Train coverage: {train_coverage:.1f}%")
    print(f"Test coverage: {test_coverage:.1f}%")
    print(f"\nFinished: {datetime.now()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
