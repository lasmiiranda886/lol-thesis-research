"""
LoL Win Prediction - Hyperparameter Fine-Tuning V8.1 (Memory Optimized)
========================================================================
- Garbage collection after each model
- Only keeps best model in memory
- Resume from checkpoint support
- Reduced ExtraTrees configs (memory hog)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import pickle
import json
import time
import gc
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb

# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    'train_path': 'data/interim/aggregate/hero_dataset_train_v8.parquet',
    'test_path': 'data/interim/aggregate/hero_dataset_test_v8.parquet',
    'features_path': 'data/interim/aggregate/features_v8.pkl',
    'output_dir': 'data/interim/aggregate',
    'checkpoint_interval': 25,
    
    # Resume from checkpoint if exists
    'resume_from_checkpoint': True,
}

# =============================================================================
# PARAMETER CONFIGURATIONS (REDUCED FOR MEMORY)
# =============================================================================

def generate_et_configs(n_configs=400):
    """Generate ExtraTrees configs - REDUCED to avoid OOM."""
    configs = []
    
    # 1. Fine grid around best (n_est=300, depth=15, min_split=10)
    for n_est in [250, 300, 350, 400]:
        for depth in [12, 14, 15, 16, 18, 20]:
            for min_split in [8, 10, 12, 15]:
                for min_leaf in [1, 2, 3]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'max_features': 'sqrt',
                    })
    
    # 2. Bootstrap variations
    for n_est in [300, 400]:
        for depth in [15, 20]:
            for bootstrap in [True, False]:
                for cw in [None, 'balanced']:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': 10,
                        'min_samples_leaf': 2,
                        'max_features': 'sqrt',
                        'bootstrap': bootstrap,
                        'class_weight': cw,
                    })
    
    # 3. max_features exploration
    for n_est in [300, 400]:
        for depth in [15, 20]:
            for max_feat in ['sqrt', 'log2', 0.5, 0.7, 0.8]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'min_samples_split': 10,
                    'min_samples_leaf': 2,
                    'max_features': max_feat,
                })
    
    # Remove duplicates
    unique = []
    seen = set()
    for c in configs:
        # Set defaults
        c.setdefault('bootstrap', False)
        c.setdefault('class_weight', None)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique[:n_configs]


def generate_lgb_configs(n_configs=600):
    """Generate LightGBM configs."""
    configs = []
    
    # 1. Fine grid around best
    for n_est in [175, 200, 225, 250, 300]:
        for depth in [5, 6, 7, 8]:
            for lr in [0.03, 0.04, 0.05, 0.06]:
                for leaves in [25, 31, 40, 50]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr,
                        'num_leaves': leaves,
                        'min_child_samples': 30,
                    })
    
    # 2. Regularization
    for n_est in [200, 300]:
        for depth in [6, 8]:
            for reg_a in [0, 0.05, 0.1, 0.3]:
                for reg_l in [0, 0.05, 0.1, 0.3]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': 0.05,
                        'num_leaves': 31,
                        'min_child_samples': 30,
                        'reg_alpha': reg_a,
                        'reg_lambda': reg_l,
                    })
    
    # 3. Subsampling
    for n_est in [200, 300, 400]:
        for subsample in [0.7, 0.8, 0.9]:
            for colsample in [0.7, 0.8, 0.9]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'min_child_samples': 30,
                    'subsample': subsample,
                    'colsample_bytree': colsample,
                })
    
    # 4. Lower LR + more estimators
    for n_est in [400, 500, 600]:
        for depth in [6, 8]:
            for lr in [0.01, 0.02, 0.03]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'learning_rate': lr,
                    'num_leaves': 31,
                    'min_child_samples': 30,
                })
    
    # Remove duplicates
    unique = []
    seen = set()
    for c in configs:
        c.setdefault('reg_alpha', 0)
        c.setdefault('reg_lambda', 0)
        c.setdefault('subsample', 1.0)
        c.setdefault('colsample_bytree', 1.0)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique[:n_configs]


def generate_xgb_configs(n_configs=600):
    """Generate XGBoost configs."""
    configs = []
    
    # 1. Fine grid around best
    for n_est in [250, 300, 350, 400]:
        for depth in [5, 6, 7, 8]:
            for lr in [0.02, 0.03, 0.04, 0.05]:
                for reg_a in [0.05, 0.1, 0.15, 0.2]:
                    configs.append({
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'learning_rate': lr,
                        'reg_alpha': reg_a,
                    })
    
    # 2. reg_lambda exploration
    for n_est in [300, 400]:
        for depth in [6, 8]:
            for reg_l in [0, 0.1, 0.5, 1.0, 2.0]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'learning_rate': 0.03,
                    'reg_alpha': 0.1,
                    'reg_lambda': reg_l,
                })
    
    # 3. Subsampling
    for n_est in [300, 400]:
        for subsample in [0.6, 0.7, 0.8, 0.9]:
            for colsample in [0.6, 0.7, 0.8, 0.9]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': 6,
                    'learning_rate': 0.03,
                    'reg_alpha': 0.1,
                    'subsample': subsample,
                    'colsample_bytree': colsample,
                })
    
    # 4. min_child_weight + gamma
    for n_est in [300, 400]:
        for mcw in [1, 3, 5, 10]:
            for gamma in [0, 0.1, 0.5]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': 6,
                    'learning_rate': 0.03,
                    'reg_alpha': 0.1,
                    'min_child_weight': mcw,
                    'gamma': gamma,
                })
    
    # 5. Higher estimators
    for n_est in [500, 600, 800]:
        for depth in [6, 8]:
            for lr in [0.01, 0.02]:
                configs.append({
                    'n_estimators': n_est,
                    'max_depth': depth,
                    'learning_rate': lr,
                    'reg_alpha': 0.1,
                })
    
    # Remove duplicates
    unique = []
    seen = set()
    for c in configs:
        c.setdefault('reg_alpha', 0)
        c.setdefault('reg_lambda', 0)
        c.setdefault('subsample', 1.0)
        c.setdefault('colsample_bytree', 1.0)
        c.setdefault('min_child_weight', 1)
        c.setdefault('gamma', 0)
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return unique[:n_configs]


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(results, model_name, path, best_model=None, best_params=None):
    """Save checkpoint with results."""
    checkpoint = {
        'model': model_name,
        'n_configs': len(results),
        'results': results,
        'best_auc': max(r['test_auc'] for r in results) if results else 0,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
    }
    checkpoint_file = f"{path}/checkpoint_{model_name}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)
    
    # Save best model separately
    if best_model is not None:
        model_file = f"{path}/best_{model_name}_finetuned.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
    
    return checkpoint_file


def load_checkpoint(model_name, path):
    """Load checkpoint if exists."""
    checkpoint_file = f"{path}/checkpoint_{model_name}.json"
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# TRAINING (MEMORY OPTIMIZED)
# =============================================================================

def run_fine_tuning(model_name, model_class, configs, X_train, y_train, X_test, y_test, 
                    base_params=None, output_path=None, resume=True):
    """Run fine-tuning with memory optimization."""
    print(f"\n{'='*70}")
    print(f"FINE-TUNING {model_name.upper()} ({len(configs)} configurations)")
    print(f"{'='*70}")
    
    # Check for existing checkpoint
    start_idx = 0
    results = []
    best_auc = 0
    best_model = None
    best_params = None
    best_pred = None
    
    if resume and output_path:
        checkpoint = load_checkpoint(model_name, output_path)
        if checkpoint:
            results = checkpoint['results']
            start_idx = checkpoint['n_configs']
            # Handle old checkpoint format
            if 'best_auc' in checkpoint:
                best_auc = checkpoint['best_auc']
            elif results:
                best_auc = max(r['test_auc'] for r in results)
            best_params = checkpoint.get('best_params')
            print(f"  Resuming from checkpoint: {start_idx} configs done, best AUC: {best_auc:.4f}")
    
    start_time = time.time()
    
    for i, params in enumerate(configs):
        if i < start_idx:
            continue
        
        try:
            # Build model
            full_params = {**base_params, **params} if base_params else params
            model = model_class(**full_params)
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict_proba(X_train)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, train_pred)
            test_auc = roc_auc_score(y_test, test_pred)
            test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
            
            result = {
                'config_id': i + 1,
                'params': params,
                'train_auc': round(train_auc, 5),
                'test_auc': round(test_auc, 5),
                'test_acc': round(test_acc, 5),
            }
            results.append(result)
            
            # Track best
            if test_auc > best_auc:
                best_auc = test_auc
                best_params = params
                best_model = model
                best_pred = test_pred.copy()
            else:
                # Delete model to free memory
                del model
            
            # Clear predictions
            del train_pred, test_pred
            
            # Progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = len(configs) - i - 1
                eta = elapsed / (i + 1 - start_idx) * remaining if i > start_idx else 0
                print(f"  [{i+1}/{len(configs)}] Best AUC: {best_auc:.4f} | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            
            # Checkpoint
            if output_path and (i + 1) % CONFIG['checkpoint_interval'] == 0:
                save_checkpoint(results, model_name, output_path, best_model, best_params)
            
            # CRITICAL: Garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"  [{i+1}/{len(configs)}] FAILED: {str(e)[:60]}")
            gc.collect()
    
    # Final save
    if output_path:
        save_checkpoint(results, model_name, output_path, best_model, best_params)
    
    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed/60:.1f} minutes")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Best params: {best_params}")
    
    return results, best_model, best_params, best_pred


def print_top_results(results, model_name, n=15):
    """Print top results."""
    sorted_results = sorted(results, key=lambda x: -x['test_auc'])
    
    print(f"\n{'='*70}")
    print(f"TOP {n} {model_name.upper()} CONFIGURATIONS")
    print(f"{'='*70}")
    
    print(f"\n{'Rank':<6} {'Config':<8} {'Train AUC':<12} {'Test AUC':<12} {'Test Acc':<12}")
    print("-" * 55)
    
    for rank, res in enumerate(sorted_results[:n], 1):
        print(f"{rank:<6} {res['config_id']:<8} {res['train_auc']:<12.4f} {res['test_auc']:<12.4f} "
              f"{res['test_acc']:<12.4f}")
    
    return sorted_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LOL WIN PREDICTION - HYPERPARAMETER FINE-TUNING V8.1")
    print("(Memory Optimized + Checkpoint Resume)")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load data
    print("\n" + "="*60)
    print("1. LOADING DATA")
    print("="*60)
    
    train_df = pd.read_parquet(CONFIG['train_path'])
    test_df = pd.read_parquet(CONFIG['test_path'])
    
    with open(CONFIG['features_path'], 'rb') as f:
        features = pickle.load(f)
    
    print(f"Train: {len(train_df):,} rows")
    print(f"Test: {len(test_df):,} rows")
    print(f"Features: {len(features)}")
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df['hero_win'].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df['hero_win'].values
    
    # Free dataframes
    del train_df
    gc.collect()
    
    # Generate configs
    print("\n" + "="*60)
    print("2. GENERATING CONFIGURATIONS")
    print("="*60)
    
    et_configs = generate_et_configs(n_configs=400)
    lgb_configs = generate_lgb_configs(n_configs=600)
    xgb_configs = generate_xgb_configs(n_configs=600)
    
    print(f"ExtraTrees configs: {len(et_configs)}")
    print(f"LightGBM configs: {len(lgb_configs)}")
    print(f"XGBoost configs: {len(xgb_configs)}")
    print(f"Total: {len(et_configs) + len(lgb_configs) + len(xgb_configs)}")
    
    out = CONFIG['output_dir']
    all_best = {}
    
    # Fine-tune ExtraTrees
    et_base = {'random_state': 42, 'n_jobs': -1}
    et_results, et_model, et_params, et_pred = run_fine_tuning(
        'et', ExtraTreesClassifier, et_configs,
        X_train, y_train, X_test, y_test,
        base_params=et_base, output_path=out,
        resume=CONFIG['resume_from_checkpoint']
    )
    print_top_results(et_results, 'ExtraTrees')
    all_best['et'] = {'model': et_model, 'params': et_params, 'pred': et_pred,
                      'auc': max(r['test_auc'] for r in et_results)}
    
    # Clean up
    gc.collect()
    
    # Fine-tune LightGBM
    lgb_base = {'random_state': 42, 'verbose': -1, 'n_jobs': -1}
    lgb_results, lgb_model, lgb_params, lgb_pred = run_fine_tuning(
        'lgb', lgb.LGBMClassifier, lgb_configs,
        X_train, y_train, X_test, y_test,
        base_params=lgb_base, output_path=out,
        resume=CONFIG['resume_from_checkpoint']
    )
    print_top_results(lgb_results, 'LightGBM')
    all_best['lgb'] = {'model': lgb_model, 'params': lgb_params, 'pred': lgb_pred,
                       'auc': max(r['test_auc'] for r in lgb_results)}
    
    gc.collect()
    
    # Fine-tune XGBoost
    xgb_base = {'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0, 'n_jobs': -1}
    xgb_results, xgb_model, xgb_params, xgb_pred = run_fine_tuning(
        'xgb', xgb.XGBClassifier, xgb_configs,
        X_train, y_train, X_test, y_test,
        base_params=xgb_base, output_path=out,
        resume=CONFIG['resume_from_checkpoint']
    )
    print_top_results(xgb_results, 'XGBoost')
    all_best['xgb'] = {'model': xgb_model, 'params': xgb_params, 'pred': xgb_pred,
                       'auc': max(r['test_auc'] for r in xgb_results)}
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL BEST MODELS")
    print("="*70)
    
    for name, info in sorted(all_best.items(), key=lambda x: -x[1]['auc']):
        print(f"\n{name.upper()}: AUC = {info['auc']:.4f}")
        print(f"  Params: {info['params']}")
    
    # Save predictions for ensemble
    print("\n" + "="*60)
    print("SAVING PREDICTIONS FOR ENSEMBLE")
    print("="*60)
    
    predictions = {
        'et': all_best['et']['pred'],
        'lgb': all_best['lgb']['pred'],
        'xgb': all_best['xgb']['pred'],
        'y_test': y_test,
    }
    
    with open(f"{out}/best_predictions_v8.pkl", 'wb') as f:
        pickle.dump(predictions, f)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_models': {
            name: {'auc': info['auc'], 'params': info['params']}
            for name, info in all_best.items()
        }
    }
    with open(f"{out}/finetuning_summary_v8.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSaved to {out}/")
    
    print("\n" + "="*70)
    print(f"COMPLETED: {datetime.now()}")
    print("="*70)
    
    return all_best


if __name__ == '__main__':
    best_models = main()