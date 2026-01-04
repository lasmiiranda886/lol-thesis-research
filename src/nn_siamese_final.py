#!/usr/bin/env python3
"""
Siamese Network - Final Training
With user-specified configuration for production use.
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Configuration
CONFIG = {
    'emb_dim': 48,
    'hidden_dim': 256,
    'compare_hidden': 128,
    'dropout': 0.35,
    'epochs': 50,
    'lr': 0.001,
    'batch_size': 1024,
    'n_folds': 5,
    'early_stopping_patience': 10
}

print("=" * 60)
print("SIAMESE NETWORK - FINAL TRAINING")
print(f"Started: {datetime.now()}")
print("=" * 60)
print(f"\nConfiguration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# ============================================================================
# Data Loading
# ============================================================================

def load_all_training_data():
    """Load all training data"""
    base_path = 'data/interim/aggregate'

    print("\n" + "=" * 60)
    print("LOADING ALL TRAINING DATA")
    print("=" * 60)

    # 1. EUW 2023-2024
    print("\n1. Loading EUW data...")
    euw_df = pd.read_parquet(f'{base_path}/participants_soloq_clean.parquet')
    euw_2023_2024 = euw_df[euw_df['gameVersion'].str.match(r'^1[34]\.')].copy()
    print(f"   EUW 2023-2024: {euw_2023_2024['matchId'].nunique():,} matches")

    # 2. Enriched (Global) 2023-2024
    print("\n2. Loading Enriched (Global) data...")
    enriched_df = pd.read_parquet(f'{base_path}/participants_enriched.parquet')
    enriched_2023_2024 = enriched_df[enriched_df['gameVersion'].str.match(r'^1[34]\.')].copy()
    print(f"   Enriched 2023-2024: {enriched_2023_2024['matchId'].nunique():,} matches")

    # 3. Load 2025 participants
    print("\n3. Loading 2025 participants...")
    participants_2025 = pd.read_parquet(f'{base_path}/participants_global_2025.parquet')
    print(f"   2025 All: {participants_2025['matchId'].nunique():,} matches")

    # 4. Get train/test split
    print("\n4. Getting train/test split...")
    train_hero = pd.read_parquet(f'{base_path}/hero_global_2025_random_train.parquet')
    test_hero = pd.read_parquet(f'{base_path}/hero_global_2025_random_test.parquet')

    train_match_ids = set(train_hero['matchId'].unique())
    test_match_ids = set(test_hero['matchId'].unique())

    print(f"   2025 Train: {len(train_match_ids):,} matches")
    print(f"   2025 Test: {len(test_match_ids):,} matches")

    train_2025 = participants_2025[participants_2025['matchId'].isin(train_match_ids)].copy()
    test_2025 = participants_2025[participants_2025['matchId'].isin(test_match_ids)].copy()

    # 5. Combine training data
    print("\n5. Combining training data...")
    all_train = pd.concat([euw_2023_2024, enriched_2023_2024, train_2025], ignore_index=True)
    all_train = all_train.drop_duplicates(subset=['matchId', 'puuid'])

    print(f"\n=== TOTAL TRAINING DATA ===")
    print(f"   Matches: {all_train['matchId'].nunique():,}")
    print(f"   Rows: {len(all_train):,}")

    return all_train, test_2025, train_match_ids, test_match_ids


def prepare_match_features(participants_df):
    """Convert participant data to match-level features"""

    print("\nPreparing match features...")
    start_time = time.time()

    elo_map = {
        'IRON': 1, 'BRONZE': 2, 'SILVER': 3, 'GOLD': 4,
        'PLATINUM': 5, 'EMERALD': 6, 'DIAMOND': 7,
        'MASTER': 8, 'GRANDMASTER': 9, 'CHALLENGER': 10
    }

    pos_order = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

    results = []
    grouped = participants_df.groupby('matchId')
    total_matches = len(grouped)
    processed = 0

    for match_id, match_df in grouped:
        if len(match_df) != 10:
            continue

        team_data = defaultdict(lambda: {'champs': [None]*5, 'elo': None, 'win': None})

        for _, row in match_df.iterrows():
            team_id = row['teamId']
            pos = row.get('teamPosition', row.get('individualPosition', 'MIDDLE'))
            pos_idx = pos_order.get(pos, 2)

            team_data[team_id]['champs'][pos_idx] = row['championName']

            if row.get('rank_tier'):
                team_data[team_id]['elo'] = elo_map.get(row['rank_tier'], 5)

            team_data[team_id]['win'] = row['win']

        if len(team_data) != 2:
            continue

        valid = True
        for tid, tdata in team_data.items():
            if None in tdata['champs']:
                valid = False
                break

        if not valid:
            continue

        team_ids = list(team_data.keys())

        for i, hero_team_id in enumerate(team_ids):
            enemy_team_id = team_ids[1 - i]
            is_blue_side = 1 if hero_team_id == 100 else 0

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
                'elo': team_data[hero_team_id]['elo'] or 5,
                'is_blue_side': is_blue_side,
                'win': int(team_data[hero_team_id]['win'])
            })

        processed += 1
        if processed % 50000 == 0:
            elapsed = time.time() - start_time
            print(f"   Processed {processed:,} / {total_matches:,} ({elapsed:.1f}s)")

    result_df = pd.DataFrame(results)
    result_df['elo'] = result_df['elo'].fillna(5.0).clip(1, 10)

    elapsed = time.time() - start_time
    print(f"   Done! {len(result_df):,} samples from {processed:,} matches ({elapsed:.1f}s)")

    return result_df


def create_encoder(match_features_df):
    """Create position-aware champion encoder"""
    pos_cols = ['hero_top', 'hero_jg', 'hero_mid', 'hero_bot', 'hero_sup',
                'enemy_top', 'enemy_jg', 'enemy_mid', 'enemy_bot', 'enemy_sup']
    pos_types = ['TOP', 'JG', 'MID', 'BOT', 'SUP'] * 2

    all_combos = set()
    for col, pos in zip(pos_cols, pos_types):
        combos = match_features_df[col].apply(lambda x: f"{pos}_{x}")
        all_combos.update(combos.unique())

    encoder = LabelEncoder()
    encoder.fit(list(all_combos))
    print(f"   Unique position-champion combos: {len(encoder.classes_)}")

    return encoder


def encode_features(match_df, encoder):
    """Encode all features for models"""
    pos_cols = ['hero_top', 'hero_jg', 'hero_mid', 'hero_bot', 'hero_sup',
                'enemy_top', 'enemy_jg', 'enemy_mid', 'enemy_bot', 'enemy_sup']
    pos_types = ['TOP', 'JG', 'MID', 'BOT', 'SUP'] * 2

    known = set(encoder.classes_)
    encoded = []

    for col, pos in zip(pos_cols, pos_types):
        combos = match_df[col].apply(lambda x: f"{pos}_{x}")
        fallback = [c for c in known if c.startswith(pos + '_')][0]
        combos = combos.apply(lambda x: x if x in known else fallback)
        encoded.append(encoder.transform(combos))

    X_champs = np.column_stack(encoded)
    X_elo = match_df['elo'].values.reshape(-1, 1)
    X_side = match_df['is_blue_side'].values.reshape(-1, 1)
    y = match_df['win'].values
    match_ids = match_df['matchId'].values

    return X_champs, X_elo, X_side, y, match_ids


# ============================================================================
# Siamese Network Model
# ============================================================================

class SiameseNetwork(nn.Module):
    """
    Siamese Network for team comparison.
    Uses shared weights to encode both teams, then compares representations.
    """

    def __init__(self, n_features, emb_dim=48, hidden_dim=256, compare_hidden=128, dropout=0.35):
        super().__init__()

        self.champion_emb = nn.Embedding(n_features, emb_dim)

        # Shared team encoder (applied to both hero and enemy teams)
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

        # Elo embedding
        self.elo_emb = nn.Embedding(11, 16)

        # Side embedding
        self.side_emb = nn.Embedding(2, 8)

        # Comparison network
        # Input: hero_enc (hidden_dim) + enemy_enc (hidden_dim) + |diff| (hidden_dim) + elo (16) + side (8)
        compare_input = hidden_dim * 3 + 16 + 8

        self.compare_net = nn.Sequential(
            nn.Linear(compare_input, compare_hidden),
            nn.BatchNorm1d(compare_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(compare_hidden, compare_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(compare_hidden // 2, 1)
        )

    def forward(self, champs, elo, side):
        """
        champs: (batch, 10) - position-encoded champion IDs [hero_5 + enemy_5]
        elo: (batch, 1) - elo value 1-10
        side: (batch, 1) - blue side (1) or red side (0)
        """
        hero_champs = champs[:, :5]
        enemy_champs = champs[:, 5:]

        # Embed champions
        hero_emb = self.champion_emb(hero_champs)    # (batch, 5, emb_dim)
        enemy_emb = self.champion_emb(enemy_champs)  # (batch, 5, emb_dim)

        # Flatten to team representation
        hero_flat = hero_emb.view(hero_emb.size(0), -1)   # (batch, 5*emb_dim)
        enemy_flat = enemy_emb.view(enemy_emb.size(0), -1)

        # Encode teams with SHARED weights
        hero_enc = self.team_encoder(hero_flat)   # (batch, hidden_dim)
        enemy_enc = self.team_encoder(enemy_flat)  # (batch, hidden_dim)

        # Compute difference (symmetric)
        diff = torch.abs(hero_enc - enemy_enc)  # (batch, hidden_dim)

        # Elo embedding
        elo_idx = elo.long().clamp(0, 10).squeeze(1)
        elo_enc = self.elo_emb(elo_idx)  # (batch, 16)

        # Side embedding
        side_idx = side.long().squeeze(1)
        side_enc = self.side_emb(side_idx)  # (batch, 8)

        # Combine all features
        combined = torch.cat([hero_enc, enemy_enc, diff, elo_enc, side_enc], dim=1)

        # Final prediction
        logits = self.compare_net(combined)

        return logits


# ============================================================================
# Training
# ============================================================================

def train_siamese_cv(X_champs, X_elo, X_side, y, match_ids, n_features, train_2025_ids, config):
    """Train Siamese with Cross-Validation for OOF predictions"""

    print("\n" + "=" * 60)
    print("TRAINING SIAMESE WITH CROSS-VALIDATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Features: {n_features}, Samples: {len(y):,}")

    kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y))
    fold_aucs = []
    fold_models = []

    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_champs, y)):
        print(f"\n--- Fold {fold + 1}/{config['n_folds']} ---")
        fold_start = time.time()

        # Prepare data
        X_champs_train = torch.LongTensor(X_champs[train_idx]).to(device)
        X_elo_train = torch.FloatTensor(X_elo[train_idx]).to(device)
        X_side_train = torch.LongTensor(X_side[train_idx]).to(device)
        y_train = torch.FloatTensor(y[train_idx]).to(device)

        X_champs_val = torch.LongTensor(X_champs[val_idx]).to(device)
        X_elo_val = torch.FloatTensor(X_elo[val_idx]).to(device)
        X_side_val = torch.LongTensor(X_side[val_idx]).to(device)
        y_val = y[val_idx]

        # Initialize model
        model = SiameseNetwork(
            n_features=n_features,
            emb_dim=config['emb_dim'],
            hidden_dim=config['hidden_dim'],
            compare_hidden=config['compare_hidden'],
            dropout=config['dropout']
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=False
        )

        batch_size = config['batch_size']
        n_batches = (len(train_idx) + batch_size - 1) // batch_size

        best_val_auc = 0
        best_state = None
        patience = 0

        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_loss = 0
            indices = torch.randperm(len(train_idx), device=device)

            for i in range(n_batches):
                batch_idx = indices[i*batch_size:(i+1)*batch_size]

                optimizer.zero_grad()
                logits = model(
                    X_champs_train[batch_idx],
                    X_elo_train[batch_idx],
                    X_side_train[batch_idx]
                )
                loss = criterion(logits, y_train[batch_idx].unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_champs_val, X_elo_val, X_side_val)
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_auc = roc_auc_score(y_val, val_probs)

            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 5 == 0 or epoch == config['epochs'] - 1:
                print(f"  Epoch {epoch+1}: loss={train_loss/n_batches:.4f}, val_auc={val_auc:.4f}, best={best_val_auc:.4f}")

            if patience >= config['early_stopping_patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model and get final predictions
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        with torch.no_grad():
            val_logits = model(X_champs_val, X_elo_val, X_side_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()

        oof_predictions[val_idx] = val_probs
        fold_aucs.append(best_val_auc)
        fold_models.append(best_state)

        fold_time = time.time() - fold_start
        print(f"  Fold {fold+1} Final AUC: {best_val_auc:.4f} ({fold_time/60:.1f}min)")

    total_time = time.time() - total_start
    oof_auc = roc_auc_score(y, oof_predictions)

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"OOF AUC: {oof_auc:.4f}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Std: {np.std(fold_aucs):.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")

    return oof_predictions, oof_auc, fold_aucs, fold_models


def train_final_model(X_champs, X_elo, X_side, y, n_features, config):
    """Train final model on all data"""

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_champs_t = torch.LongTensor(X_champs).to(device)
    X_elo_t = torch.FloatTensor(X_elo).to(device)
    X_side_t = torch.LongTensor(X_side).to(device)
    y_t = torch.FloatTensor(y).to(device)

    model = SiameseNetwork(
        n_features=n_features,
        emb_dim=config['emb_dim'],
        hidden_dim=config['hidden_dim'],
        compare_hidden=config['compare_hidden'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    batch_size = config['batch_size']
    n_batches = (len(y) + batch_size - 1) // batch_size

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        indices = torch.randperm(len(y), device=device)

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]

            optimizer.zero_grad()
            logits = model(X_champs_t[batch_idx], X_elo_t[batch_idx], X_side_t[batch_idx])
            loss = criterion(logits, y_t[batch_idx].unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch+1}: loss={train_loss/n_batches:.4f}")

    return model


def predict(model, X_champs, X_elo, X_side, batch_size=1024):
    """Make predictions with model"""
    device = next(model.parameters()).device

    X_champs_t = torch.LongTensor(X_champs).to(device)
    X_elo_t = torch.FloatTensor(X_elo).to(device)
    X_side_t = torch.LongTensor(X_side).to(device)

    model.eval()
    predictions = []

    n_samples = len(X_champs)
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)

            logits = model(X_champs_t[start:end], X_elo_t[start:end], X_side_t[start:end])
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.extend(probs)

    return np.array(predictions)


# ============================================================================
# Main
# ============================================================================

def main():
    # Load data
    train_data, test_data, train_2025_ids, test_2025_ids = load_all_training_data()

    # Prepare features
    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)

    train_features = prepare_match_features(train_data)

    print("\nCreating encoder...")
    encoder = create_encoder(train_features)

    print("\nEncoding training features...")
    X_champs, X_elo, X_side, y, match_ids = encode_features(train_features, encoder)
    n_features = len(encoder.classes_)

    # Train with CV
    oof_predictions, oof_auc, fold_aucs, fold_models = train_siamese_cv(
        X_champs, X_elo, X_side, y, match_ids,
        n_features, train_2025_ids, CONFIG
    )

    # Save OOF predictions for 2025 train
    print("\n" + "=" * 60)
    print("SAVING OOF PREDICTIONS FOR 2025 TRAIN")
    print("=" * 60)

    train_features['siamese_oof_score'] = oof_predictions
    train_2025_oof = train_features[train_features['matchId'].isin(train_2025_ids)].copy()

    # Aggregate to match level (take first perspective)
    train_2025_scores = train_2025_oof.groupby('matchId').agg({
        'siamese_oof_score': 'first',
        'hero_team_id': 'first'
    }).reset_index()

    print(f"2025 Train OOF scores: {len(train_2025_scores):,} matches")

    oof_scores_dict = {
        'match_scores': dict(zip(train_2025_scores['matchId'], train_2025_scores['siamese_oof_score'])),
        'oof_auc': oof_auc,
        'fold_aucs': fold_aucs,
        'config': CONFIG
    }

    with open('data/interim/aggregate/nn_siamese_train_scores.pkl', 'wb') as f:
        pickle.dump(oof_scores_dict, f)
    print("Saved: nn_siamese_train_scores.pkl")

    # Train final model
    final_model = train_final_model(X_champs, X_elo, X_side, y, n_features, CONFIG)

    # Prepare test features
    print("\n" + "=" * 60)
    print("PREPARING TEST FEATURES")
    print("=" * 60)

    test_features = prepare_match_features(test_data)
    X_champs_test, X_elo_test, X_side_test, y_test, match_ids_test = encode_features(test_features, encoder)

    # Predict on test
    print("\n" + "=" * 60)
    print("PREDICTING ON TEST DATA")
    print("=" * 60)

    test_predictions = predict(final_model, X_champs_test, X_elo_test, X_side_test)
    test_auc = roc_auc_score(y_test, test_predictions)
    print(f"Test AUC: {test_auc:.4f}")

    # Aggregate test scores to match level
    test_features['siamese_score'] = test_predictions
    test_match_scores = test_features.groupby('matchId').agg({
        'siamese_score': 'first',
        'hero_team_id': 'first'
    }).reset_index()

    print(f"Test scores: {len(test_match_scores):,} matches")

    test_scores_dict = {
        'match_scores': dict(zip(test_match_scores['matchId'], test_match_scores['siamese_score'])),
        'test_auc': test_auc,
        'config': CONFIG
    }

    with open('data/interim/aggregate/nn_siamese_test_scores.pkl', 'wb') as f:
        pickle.dump(test_scores_dict, f)
    print("Saved: nn_siamese_test_scores.pkl")

    # Save model and encoder
    torch.save(final_model.state_dict(), 'models/nn_siamese_model.pt')
    with open('models/nn_siamese_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("Saved: nn_siamese_model.pt, nn_siamese_encoder.pkl")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"OOF AUC: {oof_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print(f"\nFinished: {datetime.now()}")
    print("=" * 60)

    # Save summary
    summary = {
        'oof_auc': oof_auc,
        'test_auc': test_auc,
        'fold_aucs': fold_aucs,
        'config': CONFIG,
        'finished': str(datetime.now())
    }
    with open('models/nn_siamese_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
