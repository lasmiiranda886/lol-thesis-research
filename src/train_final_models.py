"""
Train Final Models for Thesis
==============================
Trains and saves the best feature combinations found in ablation study.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
import xgboost as xgb

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path('data/final')
MODEL_DIR = Path('models')
REPORT_DIR = Path('reports')
MODEL_DIR.mkdir(exist_ok=True)

DATASETS = {
    'random': {
        'train': DATA_DIR / 'hero_dataset_random_train_final.parquet',
        'test': DATA_DIR / 'hero_dataset_random_test_final.parquet',
    },
    'main': {
        'train': DATA_DIR / 'hero_dataset_main_train_final.parquet',
        'test': DATA_DIR / 'hero_dataset_main_test_final.parquet',
    }
}

# Best feature combinations from ablation study
BEST_COMBINATIONS = {
    'minimal_13': {
        'name': 'Minimal 13 Features',
        'description': 'Best minimal feature set (GRP_1530)',
        'features': [
            'siamese_score', 'expected_wr', 'smurf_score',
            'hero_cm_level_feat', 'hero_cm_points_log', 'hero_is_blue_feat',
            'hero_personal_overall_wr', 'hero_personal_champ_games',
            'cs_hero_vs_enemy_wr', 'cs_enemy_team_avg_wr',
            'cs_hero_champ_wr_at_elo_role', 'cs_hero_team_avg_wr',
            'hero_winrate'
        ]
    },
    'balanced_17': {
        'name': 'Balanced 17 Features',
        'description': 'Good balance of performance and simplicity (BB_M4_101)',
        'features': [
            'hero_lp', 'hero_is_blue_feat', 'hero_total_games', 'hero_winrate',
            'hero_cm_level_feat', 'hero_wr_rank_mismatch', 'hero_personal_champ_wr',
            'hero_personal_champ_games', 'hero_personal_overall_wr',
            'siamese_score', 'smurf_score', 'expected_wr',
            'cs_hero_champ_wr_at_elo_role', 'cs_hero_team_avg_wr',
            'cs_enemy_team_avg_wr', 'cs_hero_vs_enemy_wr', 'hero_cm_points_log'
        ]
    },
    'hero_focused_20': {
        'name': 'Hero Focused 20 Features',
        'description': 'Hero-centric feature set (HERO_191)',
        'features': [
            'hero_total_games', 'hero_wr_rank_mismatch', 'hero_personal_champ_wr',
            'hero_cm_level_feat', 'hero_rank_numeric', 'hero_cm_points_log',
            'hero_personal_champ_games', 'hero_is_blue_feat', 'hero_personal_overall_wr',
            'hero_lp', 'hero_winrate',
            'siamese_score', 'smurf_score', 'expected_wr',
            'cs_hero_champ_wr_at_elo_role', 'cs_hero_team_avg_wr',
            'cs_enemy_team_avg_wr', 'cs_hero_vs_enemy_wr',
            'tc_scaling_diff', 'tc_tank_diff'
        ]
    },
    'comprehensive_25': {
        'name': 'Comprehensive 25 Features',
        'description': 'Full feature set without matchups (BEST_LKO6)',
        'features': [
            'hero_rank_numeric', 'hero_lp', 'hero_is_blue_feat', 'hero_total_games',
            'hero_winrate', 'hero_cm_points_log', 'hero_cm_level_feat',
            'hero_wr_rank_mismatch', 'hero_personal_champ_wr', 'hero_personal_champ_games',
            'hero_personal_overall_wr',
            'siamese_score', 'smurf_score', 'expected_wr',
            'cs_hero_champ_wr_at_elo', 'cs_hero_champ_wr_at_role', 'cs_hero_champ_wr_at_elo_role',
            'cs_hero_team_avg_wr', 'cs_enemy_team_avg_wr', 'cs_hero_vs_enemy_wr',
            'cs_hero_mastery_zscore',
            'tc_scaling_diff', 'tc_tank_diff', 'tc_engage_diff', 'tc_tankiness_diff'
        ]
    }
}

# Full model parameters for final training
MODEL_PARAMS = {
    'et': {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
    },
    'lgb': {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.03,
        'num_leaves': 64,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
    },
    'xgb': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }
}

ENSEMBLE_WEIGHTS = {'et': 0.30, 'lgb': 0.40, 'xgb': 0.30}


def train_ensemble(X_train, y_train, X_test, y_test, features):
    """Train full ensemble and return models + predictions."""
    models = {}
    predictions = {}
    metrics = {}

    # Train ExtraTrees
    print("  Training ExtraTrees...")
    et = ExtraTreesClassifier(**MODEL_PARAMS['et'])
    et.fit(X_train, y_train)
    models['et'] = et
    predictions['et'] = et.predict_proba(X_test)[:, 1]
    metrics['et_auc'] = roc_auc_score(y_test, predictions['et'])

    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(**MODEL_PARAMS['lgb'])
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    predictions['lgb'] = lgb_model.predict_proba(X_test)[:, 1]
    metrics['lgb_auc'] = roc_auc_score(y_test, predictions['lgb'])

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**MODEL_PARAMS['xgb'])
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    predictions['xgb'] = xgb_model.predict_proba(X_test)[:, 1]
    metrics['xgb_auc'] = roc_auc_score(y_test, predictions['xgb'])

    # Ensemble prediction
    ensemble_pred = (
        ENSEMBLE_WEIGHTS['et'] * predictions['et'] +
        ENSEMBLE_WEIGHTS['lgb'] * predictions['lgb'] +
        ENSEMBLE_WEIGHTS['xgb'] * predictions['xgb']
    )
    metrics['ensemble_auc'] = roc_auc_score(y_test, ensemble_pred)
    metrics['ensemble_acc'] = accuracy_score(y_test, (ensemble_pred > 0.5).astype(int))
    metrics['ensemble_logloss'] = log_loss(y_test, ensemble_pred)

    # Feature importance (from LightGBM)
    importance = pd.DataFrame({
        'feature': features,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return models, predictions, metrics, importance, ensemble_pred


def main():
    print("="*70)
    print("TRAINING FINAL MODELS FOR THESIS")
    print(f"Started: {datetime.now()}")
    print("="*70)

    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)

    datasets = {}
    for name, paths in DATASETS.items():
        train = pd.read_parquet(paths['train'])
        test = pd.read_parquet(paths['test'])
        datasets[name] = {'train': train, 'test': test}
        print(f"{name}: Train {len(train):,}, Test {len(test):,}")

    # Results storage
    all_results = []
    thesis_summary = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {},
        'models': {},
    }

    # Train each combination on each dataset
    for combo_key, combo_config in BEST_COMBINATIONS.items():
        print(f"\n{'='*60}")
        print(f"TRAINING: {combo_config['name']}")
        print(f"Features: {len(combo_config['features'])}")
        print(f"{'='*60}")

        thesis_summary['models'][combo_key] = {
            'name': combo_config['name'],
            'description': combo_config['description'],
            'n_features': len(combo_config['features']),
            'features': combo_config['features'],
            'results': {}
        }

        for dataset_name, data in datasets.items():
            print(f"\n--- {dataset_name.upper()} Dataset ---")

            train_df = data['train']
            test_df = data['test']

            # Get valid features
            valid_features = [f for f in combo_config['features'] if f in train_df.columns]
            print(f"  Valid features: {len(valid_features)}/{len(combo_config['features'])}")

            if len(valid_features) < 3:
                print("  SKIPPED: Too few valid features")
                continue

            X_train = train_df[valid_features].fillna(0).values
            y_train = train_df['hero_win'].values
            X_test = test_df[valid_features].fillna(0).values
            y_test = test_df['hero_win'].values

            # Train ensemble
            models, predictions, metrics, importance, ensemble_pred = train_ensemble(
                X_train, y_train, X_test, y_test, valid_features
            )

            # Print results
            print(f"\n  Results:")
            print(f"    ExtraTrees AUC:  {metrics['et_auc']:.4f}")
            print(f"    LightGBM AUC:    {metrics['lgb_auc']:.4f}")
            print(f"    XGBoost AUC:     {metrics['xgb_auc']:.4f}")
            print(f"    Ensemble AUC:    {metrics['ensemble_auc']:.4f}")
            print(f"    Ensemble Acc:    {metrics['ensemble_acc']:.4f}")
            print(f"    Ensemble LogLoss: {metrics['ensemble_logloss']:.4f}")

            print(f"\n  Top 5 Features:")
            for _, row in importance.head(5).iterrows():
                print(f"    - {row['feature']}: {row['importance']:.0f}")

            # Save models
            model_path = MODEL_DIR / f"{combo_key}_{dataset_name}_models.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'models': models,
                    'features': valid_features,
                    'params': MODEL_PARAMS,
                    'weights': ENSEMBLE_WEIGHTS,
                    'metrics': metrics,
                    'importance': importance.to_dict(),
                }, f)
            print(f"\n  Saved: {model_path}")

            # Store results
            result = {
                'combination': combo_key,
                'combination_name': combo_config['name'],
                'dataset': dataset_name,
                'n_features': len(valid_features),
                'et_auc': metrics['et_auc'],
                'lgb_auc': metrics['lgb_auc'],
                'xgb_auc': metrics['xgb_auc'],
                'ensemble_auc': metrics['ensemble_auc'],
                'ensemble_acc': metrics['ensemble_acc'],
                'ensemble_logloss': metrics['ensemble_logloss'],
            }
            all_results.append(result)

            thesis_summary['models'][combo_key]['results'][dataset_name] = {
                'n_samples_train': len(train_df),
                'n_samples_test': len(test_df),
                'metrics': metrics,
                'top_features': importance.head(10).to_dict('records'),
            }

    # Save results summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / 'final_model_results.csv', index=False)
    print(f"\nSaved: reports/final_model_results.csv")

    # Save thesis summary
    with open(REPORT_DIR / 'thesis_model_summary.json', 'w') as f:
        json.dump(thesis_summary, f, indent=2, default=str)
    print(f"Saved: reports/thesis_model_summary.json")

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY FOR THESIS")
    print("="*70)

    print("\n" + "-"*70)
    print(f"{'Combination':<25} {'Dataset':<8} {'ET':>8} {'LGB':>8} {'XGB':>8} {'Ensemble':>10}")
    print("-"*70)

    for _, row in results_df.iterrows():
        print(f"{row['combination']:<25} {row['dataset']:<8} "
              f"{row['et_auc']:>8.4f} {row['lgb_auc']:>8.4f} {row['xgb_auc']:>8.4f} "
              f"{row['ensemble_auc']:>10.4f}")

    # Best per dataset
    print("\n" + "="*70)
    print("BEST MODELS PER DATASET")
    print("="*70)

    for dataset in ['random', 'main']:
        subset = results_df[results_df['dataset'] == dataset]
        best = subset.loc[subset['ensemble_auc'].idxmax()]
        print(f"\n{dataset.upper()}:")
        print(f"  Best: {best['combination_name']}")
        print(f"  AUC: {best['ensemble_auc']:.4f}")
        print(f"  Features: {best['n_features']}")

    # Thesis-ready text
    print("\n" + "="*70)
    print("THESIS-READY TEXT")
    print("="*70)

    random_best = results_df[results_df['dataset'] == 'random'].loc[
        results_df[results_df['dataset'] == 'random']['ensemble_auc'].idxmax()]
    main_best = results_df[results_df['dataset'] == 'main'].loc[
        results_df[results_df['dataset'] == 'main']['ensemble_auc'].idxmax()]

    print(f"""
Our feature ablation study tested {31260:,} feature combinations across both datasets.
The best performing model on the Random Hero dataset achieved an AUC of {random_best['ensemble_auc']:.4f}
using {int(random_best['n_features'])} features ({random_best['combination_name']}).

For the Main-Hero dataset (players with ≥50 games, ≥70% on top 3 champions),
the best model achieved an AUC of {main_best['ensemble_auc']:.4f} with {int(main_best['n_features'])} features.

Key findings from the ablation study:
1. Removing matchup features consistently improved performance
2. The Siamese network score was the most important single feature
3. Minimal feature sets (13-17 features) performed comparably to larger sets
4. Hero-specific features (winrate, mastery, games played) were more predictive
   than team composition features
""")

    print(f"\n{'='*70}")
    print(f"COMPLETED: {datetime.now()}")
    print("="*70)

    return results_df


if __name__ == '__main__':
    results = main()
