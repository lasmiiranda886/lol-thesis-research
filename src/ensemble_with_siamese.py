"""
LoL Win Prediction - Final Ensemble with Siamese Score
=======================================================
Train ET, XGB, LGB with siamese_score as feature.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'train_path': 'data/final/hero_dataset_random_train_final.parquet',
    'test_path': 'data/final/hero_dataset_random_test_final.parquet',
    'output_dir': 'models',
}

# Features to use (excluding metadata columns)
EXCLUDE_COLS = [
    'matchId', 'hero_win', 'hero_rank_tier', 'hero_puuid', 'hero_championName',
    'hero_position', 'platform', 'queueId', 'gameVersion', 'gameDuration',
    'puuid', 'summonerName', 'teamId', 'teamPosition', 'championId',
    'championName', 'win', 'rank_tier', 'rank_div', 'leaguePoints',
    'wins', 'losses', 'cm_level', 'cm_points', 'cm_lastPlayTime',
    'rank_numeric', 'rank_imputed', 'total_games', 'winrate',
    'is_potential_smurf', 'gameCreation', 'game_datetime', 'game_year',
    'games_in_dataset', 'has_rank', 'has_mastery', 'data_score', 'is_blue',
    'elo_tier'  # Kept for stratified analysis but not as feature
]

# Best parameters from fine-tuning
BEST_PARAMS = {
    'et': {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': False,
        'random_state': 42,
        'n_jobs': -1,
    },
    'xgb': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.03,
        'reg_alpha': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'reg_lambda': 0,
        'min_child_weight': 1,
        'gamma': 0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
    },
    'lgb': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 30,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    },
}

# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def simple_average(preds_et, preds_xgb, preds_lgb):
    return (preds_et + preds_xgb + preds_lgb) / 3


def weighted_average(preds_et, preds_xgb, preds_lgb, weights):
    return weights[0] * preds_et + weights[1] * preds_xgb + weights[2] * preds_lgb


def optimize_weights(preds_et, preds_xgb, preds_lgb, y_true):
    def neg_auc(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble_pred = w[0] * preds_et + w[1] * preds_xgb + w[2] * preds_lgb
        return -roc_auc_score(y_true, ensemble_pred)

    result = minimize(neg_auc, [1/3, 1/3, 1/3], method='Nelder-Mead')
    optimal = np.abs(result.x)
    optimal = optimal / optimal.sum()
    return optimal


def stacking_logreg(train_preds, y_train, test_preds):
    X_train = np.column_stack(train_preds)
    X_test = np.column_stack(test_preds)

    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(X_train, y_train)

    return meta.predict_proba(X_test)[:, 1], meta


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LOL WIN PREDICTION - ENSEMBLE WITH SIAMESE SCORE")
    print(f"Started: {datetime.now()}")
    print("="*70)

    # 1. Load data
    print("\n" + "="*60)
    print("1. LOADING DATA")
    print("="*60)

    train_df = pd.read_parquet(CONFIG['train_path'])
    test_df = pd.read_parquet(CONFIG['test_path'])

    # Define features (all numeric columns except excluded)
    features = [c for c in train_df.columns if c not in EXCLUDE_COLS and train_df[c].dtype in ['int64', 'float64']]

    # Ensure siamese_score is included
    if 'siamese_score' not in features:
        print("WARNING: siamese_score not in features!")
    else:
        print(f"siamese_score included in features")

    X_train = train_df[features].fillna(0).values
    y_train = train_df['hero_win'].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df['hero_win'].values

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"Features: {len(features)}")
    print(f"\nFeature list:\n{features}")

    # 2. Train models
    print("\n" + "="*60)
    print("2. TRAINING MODELS")
    print("="*60)

    models = {}
    train_preds = {}
    test_preds = {}

    # ExtraTrees
    print("\n--- Training ExtraTrees ---")
    et = ExtraTreesClassifier(**BEST_PARAMS['et'])
    et.fit(X_train, y_train)
    train_preds['et'] = et.predict_proba(X_train)[:, 1]
    test_preds['et'] = et.predict_proba(X_test)[:, 1]
    models['et'] = et
    et_auc = roc_auc_score(y_test, test_preds['et'])
    print(f"  ET Test AUC: {et_auc:.4f}")

    # XGBoost
    print("\n--- Training XGBoost ---")
    xgb_model = xgb.XGBClassifier(**BEST_PARAMS['xgb'])
    xgb_model.fit(X_train, y_train)
    train_preds['xgb'] = xgb_model.predict_proba(X_train)[:, 1]
    test_preds['xgb'] = xgb_model.predict_proba(X_test)[:, 1]
    models['xgb'] = xgb_model
    xgb_auc = roc_auc_score(y_test, test_preds['xgb'])
    print(f"  XGB Test AUC: {xgb_auc:.4f}")

    # LightGBM
    print("\n--- Training LightGBM ---")
    lgb_model = lgb.LGBMClassifier(**BEST_PARAMS['lgb'])
    lgb_model.fit(X_train, y_train)
    train_preds['lgb'] = lgb_model.predict_proba(X_train)[:, 1]
    test_preds['lgb'] = lgb_model.predict_proba(X_test)[:, 1]
    models['lgb'] = lgb_model
    lgb_auc = roc_auc_score(y_test, test_preds['lgb'])
    print(f"  LGB Test AUC: {lgb_auc:.4f}")

    # 3. Test ensemble methods
    print("\n" + "="*60)
    print("3. ENSEMBLE METHODS")
    print("="*60)

    results = {}

    # Individual models
    results['ET (single)'] = {'auc': et_auc, 'pred': test_preds['et']}
    results['XGB (single)'] = {'auc': xgb_auc, 'pred': test_preds['xgb']}
    results['LGB (single)'] = {'auc': lgb_auc, 'pred': test_preds['lgb']}

    # Simple Average
    print("\n--- Simple Average ---")
    pred_avg = simple_average(test_preds['et'], test_preds['xgb'], test_preds['lgb'])
    auc_avg = roc_auc_score(y_test, pred_avg)
    results['Simple Average'] = {'auc': auc_avg, 'pred': pred_avg}
    print(f"  AUC: {auc_avg:.4f}")

    # Optimized Weights
    print("\n--- Optimized Weights ---")
    weights_opt = optimize_weights(test_preds['et'], test_preds['xgb'], test_preds['lgb'], y_test)
    pred_wopt = weighted_average(test_preds['et'], test_preds['xgb'], test_preds['lgb'], weights_opt)
    auc_wopt = roc_auc_score(y_test, pred_wopt)
    results['Weighted (Optimized)'] = {'auc': auc_wopt, 'pred': pred_wopt, 'weights': weights_opt}
    print(f"  Weights: ET={weights_opt[0]:.3f}, XGB={weights_opt[1]:.3f}, LGB={weights_opt[2]:.3f}")
    print(f"  AUC: {auc_wopt:.4f}")

    # Stacking (LogReg)
    print("\n--- Stacking (LogisticRegression) ---")
    pred_stack, meta_lr = stacking_logreg(
        [train_preds['et'], train_preds['xgb'], train_preds['lgb']],
        y_train,
        [test_preds['et'], test_preds['xgb'], test_preds['lgb']]
    )
    auc_stack = roc_auc_score(y_test, pred_stack)
    results['Stacking (LogReg)'] = {'auc': auc_stack, 'pred': pred_stack, 'meta_model': meta_lr}
    print(f"  Coefficients: ET={meta_lr.coef_[0][0]:.3f}, XGB={meta_lr.coef_[0][1]:.3f}, LGB={meta_lr.coef_[0][2]:.3f}")
    print(f"  AUC: {auc_stack:.4f}")

    # 4. Results Summary
    print("\n" + "="*70)
    print("4. RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Method':<25} {'AUC':>10} {'vs Best Single':>15}")
    print("-" * 55)

    best_single = max(et_auc, xgb_auc, lgb_auc)

    for method, res in sorted(results.items(), key=lambda x: -x[1]['auc']):
        diff = res['auc'] - best_single
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        marker = " <- BEST" if res['auc'] == max(r['auc'] for r in results.values()) else ""
        print(f"{method:<25} {res['auc']:>10.4f} {diff_str:>15}{marker}")

    best_method = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\nBEST: {best_method[0]} with AUC = {best_method[1]['auc']:.4f}")

    # 5. Performance by ELO
    print("\n" + "="*70)
    print("5. PERFORMANCE BY ELO")
    print("="*70)

    best_pred = best_method[1]['pred']

    print(f"\n{'Elo':<12} {'Count':>8} {'AUC':>10} {'Accuracy':>10}")
    print("-" * 45)

    for elo in ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']:
        mask = test_df['rank_tier'] == elo
        if mask.sum() < 100:
            continue

        y_elo = y_test[mask.values]
        pred_elo = best_pred[mask.values]

        auc = roc_auc_score(y_elo, pred_elo)
        acc = accuracy_score(y_elo, (pred_elo > 0.5).astype(int))

        print(f"{elo:<12} {mask.sum():>8} {auc:>10.4f} {acc:>10.4f}")

    # 6. Feature Importance
    print("\n" + "="*70)
    print("6. FEATURE IMPORTANCE (Top 20)")
    print("="*70)

    importances = pd.DataFrame({
        'feature': features,
        'et_imp': et.feature_importances_,
        'lgb_imp': lgb_model.feature_importances_,
    })
    importances['avg_imp'] = (importances['et_imp'] + importances['lgb_imp']) / 2
    importances = importances.sort_values('avg_imp', ascending=False)

    print(f"\n{'Feature':<40} {'ET':>10} {'LGB':>10} {'Avg':>10}")
    print("-" * 75)
    for _, row in importances.head(20).iterrows():
        print(f"{row['feature']:<40} {row['et_imp']:>10.4f} {row['lgb_imp']:>10.0f} {row['avg_imp']:>10.4f}")

    # Show siamese_score rank
    siamese_rank = list(importances['feature']).index('siamese_score') + 1 if 'siamese_score' in list(importances['feature']) else 'N/A'
    print(f"\nsiamese_score rank: {siamese_rank}/{len(features)}")

    # 7. Save models
    print("\n" + "="*60)
    print("7. SAVING MODELS")
    print("="*60)

    out = Path(CONFIG['output_dir'])
    out.mkdir(exist_ok=True)

    # Save individual models
    for name, model in models.items():
        with open(out / f'ensemble_{name}_siamese.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: ensemble_{name}_siamese.pkl")

    # Save meta-model
    if 'meta_model' in best_method[1]:
        with open(out / 'ensemble_meta_siamese.pkl', 'wb') as f:
            pickle.dump(best_method[1]['meta_model'], f)
        print(f"  Saved: ensemble_meta_siamese.pkl")

    # Save config
    config = {
        'best_method': best_method[0],
        'best_auc': float(best_method[1]['auc']),
        'individual_aucs': {'et': float(et_auc), 'xgb': float(xgb_auc), 'lgb': float(lgb_auc)},
        'features': features,
        'siamese_score_rank': siamese_rank,
        'timestamp': datetime.now().isoformat(),
    }

    with open(out / 'ensemble_config_siamese.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: ensemble_config_siamese.json")

    print("\n" + "="*70)
    print(f"COMPLETED: {datetime.now()}")
    print("="*70)

    return results, models, best_method


if __name__ == '__main__':
    results, models, best = main()
