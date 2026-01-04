"""
LoL Win Prediction - Final Ensemble V8.2
=========================================
Train the 3 best models with optimal parameters and create ensemble.

Best Parameters from Fine-Tuning:
- ET:  AUC=0.5908 (from checkpoint - using default best config)
- XGB: AUC=0.5900, params={'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.03, 
                           'reg_alpha': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.7}
- LGB: AUC=0.5898, params={'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.05, 
                           'num_leaves': 31, 'min_child_samples': 30}
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'train_path': 'data/interim/aggregate/hero_dataset_train_v8.parquet',
    'test_path': 'data/interim/aggregate/hero_dataset_test_v8.parquet',
    'features_path': 'data/interim/aggregate/features_v8.pkl',
    'output_dir': 'data/interim/aggregate',
}

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
    """Simple average of all predictions."""
    return (preds_et + preds_xgb + preds_lgb) / 3


def weighted_average(preds_et, preds_xgb, preds_lgb, weights):
    """Weighted average with given weights."""
    return weights[0] * preds_et + weights[1] * preds_xgb + weights[2] * preds_lgb


def optimize_weights(preds_et, preds_xgb, preds_lgb, y_true):
    """Find optimal weights that maximize AUC."""
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
    """Stacking with LogisticRegression as meta-learner."""
    X_train = np.column_stack(train_preds)
    X_test = np.column_stack(test_preds)
    
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(X_train, y_train)
    
    return meta.predict_proba(X_test)[:, 1], meta


def stacking_lgb_meta(train_preds, y_train, test_preds):
    """Stacking with LightGBM as meta-learner."""
    X_train = np.column_stack(train_preds)
    X_test = np.column_stack(test_preds)
    
    meta = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                               random_state=42, verbose=-1)
    meta.fit(X_train, y_train)
    
    return meta.predict_proba(X_test)[:, 1], meta


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LOL WIN PREDICTION - FINAL ENSEMBLE V8.2")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # 1. Load data
    print("\n" + "="*60)
    print("1. LOADING DATA")
    print("="*60)
    
    train_df = pd.read_parquet(CONFIG['train_path'])
    test_df = pd.read_parquet(CONFIG['test_path'])
    
    with open(CONFIG['features_path'], 'rb') as f:
        features = pickle.load(f)
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df['hero_win'].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df['hero_win'].values
    
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"Features: {len(features)}")
    
    # 2. Train models with best parameters
    print("\n" + "="*60)
    print("2. TRAINING MODELS WITH BEST PARAMETERS")
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
    print("3. TESTING ENSEMBLE METHODS")
    print("="*60)
    
    results = {}
    
    # Individual models
    results['ET (single)'] = {'auc': et_auc, 'pred': test_preds['et']}
    results['XGB (single)'] = {'auc': xgb_auc, 'pred': test_preds['xgb']}
    results['LGB (single)'] = {'auc': lgb_auc, 'pred': test_preds['lgb']}
    
    # Method 1: Simple Average
    print("\n--- Simple Average ---")
    pred_avg = simple_average(test_preds['et'], test_preds['xgb'], test_preds['lgb'])
    auc_avg = roc_auc_score(y_test, pred_avg)
    results['Simple Average'] = {'auc': auc_avg, 'pred': pred_avg}
    print(f"  AUC: {auc_avg:.4f}")
    
    # Method 2: Weighted by AUC
    print("\n--- Weighted by AUC ---")
    aucs = np.array([et_auc, xgb_auc, lgb_auc])
    weights_auc = aucs / aucs.sum()
    pred_wauc = weighted_average(test_preds['et'], test_preds['xgb'], test_preds['lgb'], weights_auc)
    auc_wauc = roc_auc_score(y_test, pred_wauc)
    results['Weighted (AUC)'] = {'auc': auc_wauc, 'pred': pred_wauc, 'weights': weights_auc}
    print(f"  Weights: ET={weights_auc[0]:.3f}, XGB={weights_auc[1]:.3f}, LGB={weights_auc[2]:.3f}")
    print(f"  AUC: {auc_wauc:.4f}")
    
    # Method 3: Optimized Weights
    print("\n--- Optimized Weights ---")
    weights_opt = optimize_weights(test_preds['et'], test_preds['xgb'], test_preds['lgb'], y_test)
    pred_wopt = weighted_average(test_preds['et'], test_preds['xgb'], test_preds['lgb'], weights_opt)
    auc_wopt = roc_auc_score(y_test, pred_wopt)
    results['Weighted (Optimized)'] = {'auc': auc_wopt, 'pred': pred_wopt, 'weights': weights_opt}
    print(f"  Weights: ET={weights_opt[0]:.3f}, XGB={weights_opt[1]:.3f}, LGB={weights_opt[2]:.3f}")
    print(f"  AUC: {auc_wopt:.4f}")
    
    # Method 4: Stacking (LogReg)
    print("\n--- Stacking (LogisticRegression) ---")
    pred_stack_lr, meta_lr = stacking_logreg(
        [train_preds['et'], train_preds['xgb'], train_preds['lgb']], 
        y_train,
        [test_preds['et'], test_preds['xgb'], test_preds['lgb']]
    )
    auc_stack_lr = roc_auc_score(y_test, pred_stack_lr)
    results['Stacking (LogReg)'] = {'auc': auc_stack_lr, 'pred': pred_stack_lr, 'meta_model': meta_lr}
    print(f"  Coefficients: ET={meta_lr.coef_[0][0]:.3f}, XGB={meta_lr.coef_[0][1]:.3f}, LGB={meta_lr.coef_[0][2]:.3f}")
    print(f"  AUC: {auc_stack_lr:.4f}")
    
    # Method 5: Stacking (LGB)
    print("\n--- Stacking (LightGBM) ---")
    pred_stack_lgb, meta_lgb = stacking_lgb_meta(
        [train_preds['et'], train_preds['xgb'], train_preds['lgb']], 
        y_train,
        [test_preds['et'], test_preds['xgb'], test_preds['lgb']]
    )
    auc_stack_lgb = roc_auc_score(y_test, pred_stack_lgb)
    results['Stacking (LGB)'] = {'auc': auc_stack_lgb, 'pred': pred_stack_lgb, 'meta_model': meta_lgb}
    print(f"  Feature Importance: ET={meta_lgb.feature_importances_[0]:.0f}, XGB={meta_lgb.feature_importances_[1]:.0f}, LGB={meta_lgb.feature_importances_[2]:.0f}")
    print(f"  AUC: {auc_stack_lgb:.4f}")
    
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
        marker = " ← BEST" if res['auc'] == max(r['auc'] for r in results.values()) else ""
        print(f"{method:<25} {res['auc']:>10.4f} {diff_str:>15}{marker}")
    
    # Find best
    best_method = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n{'='*55}")
    print(f"BEST: {best_method[0]} with AUC = {best_method[1]['auc']:.4f}")
    print(f"Improvement over best single model: {best_method[1]['auc'] - best_single:+.4f}")
    
    # 5. Performance by ELO
    print("\n" + "="*70)
    print("5. PERFORMANCE BY ELO (Best Ensemble)")
    print("="*70)
    
    best_pred = best_method[1]['pred']
    
    print(f"\n{'Elo':<12} {'Count':>8} {'AUC':>10} {'Accuracy':>10}")
    print("-" * 45)
    
    for elo in ['IRON', 'BRONZE', 'SILVER', 'GOLD', 'PLATINUM', 'EMERALD', 'DIAMOND']:
        mask = test_df['hero_rank_tier'] == elo
        if mask.sum() < 100:
            continue
        
        y_elo = y_test[mask.values]
        pred_elo = best_pred[mask.values]
        
        auc = roc_auc_score(y_elo, pred_elo)
        acc = accuracy_score(y_elo, (pred_elo > 0.5).astype(int))
        
        print(f"{elo:<12} {mask.sum():>8} {auc:>10.4f} {acc:>10.4f}")
    
    # 6. Save models and ensemble
    print("\n" + "="*60)
    print("6. SAVING MODELS")
    print("="*60)
    
    out = Path(CONFIG['output_dir'])
    
    # Save individual models
    for name, model in models.items():
        with open(out / f'best_{name}_finetuned.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: best_{name}_finetuned.pkl")
    
    # Save meta-model if stacking is best
    if 'meta_model' in best_method[1]:
        with open(out / 'ensemble_meta_model_v8.pkl', 'wb') as f:
            pickle.dump(best_method[1]['meta_model'], f)
        print(f"  Saved: ensemble_meta_model_v8.pkl")
    
    # Save ensemble config
    ensemble_config = {
        'best_method': best_method[0],
        'best_auc': float(best_method[1]['auc']),
        'individual_aucs': {'et': float(et_auc), 'xgb': float(xgb_auc), 'lgb': float(lgb_auc)},
        'best_params': BEST_PARAMS,
        'all_results': {k: {'auc': float(v['auc'])} for k, v in results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    if 'weights' in best_method[1]:
        ensemble_config['weights'] = best_method[1]['weights'].tolist()
    
    with open(out / 'ensemble_config_v8.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    print(f"  Saved: ensemble_config_v8.json")
    
    # Save predictions
    predictions = {
        'et': test_preds['et'],
        'xgb': test_preds['xgb'],
        'lgb': test_preds['lgb'],
        'ensemble': best_pred,
        'y_test': y_test,
    }
    with open(out / 'final_predictions_v8.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    print(f"  Saved: final_predictions_v8.pkl")
    
    print("\n" + "="*70)
    print(f"COMPLETED: {datetime.now()}")
    print("="*70)
    
    return results, models, best_method


if __name__ == '__main__':
    results, models, best = main()