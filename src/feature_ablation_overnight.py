"""
Overnight Feature Ablation Study - 12 Hours
============================================
Exhaustive feature combination testing for both datasets.
Runs for ~12 hours with periodic checkpoints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from itertools import combinations, permutations
import random
import json
import gc
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path('data/interim/aggregate')
OUTPUT_DIR = Path('reports')
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_RUNTIME_HOURS = 12
CHECKPOINT_EVERY = 500  # Save results every N experiments

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

EXCLUDE_COLS = [
    'matchId', 'hero_win', 'hero_rank_tier', 'hero_puuid', 'hero_championName',
    'hero_position', 'platform', 'queueId', 'gameVersion', 'gameDuration',
    'puuid', 'summonerName', 'teamId', 'teamPosition', 'championId',
    'championName', 'win', 'rank_tier', 'rank_div', 'leaguePoints',
    'wins', 'losses', 'cm_level', 'cm_points', 'cm_lastPlayTime',
    'rank_numeric', 'rank_imputed', 'total_games', 'winrate',
    'is_potential_smurf', 'gameCreation', 'game_datetime', 'game_year',
    'games_in_dataset', 'has_rank', 'has_mastery', 'data_score', 'is_blue',
    'elo_tier'
]

MATCHUP_FEATURES = [
    'cs_hero_matchup_wr', 'cs_hero_matchup_known', 'cs_hero_matchup_wr_at_elo',
    'cs_hero_matchup_known_at_elo', 'cs_hero_team_matchup_wr', 'cs_matchup_coverage'
]

# Model configs - test multiple
MODEL_CONFIGS = {
    'lgb_fast': {
        'type': 'lgb',
        'params': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
    },
    'lgb_med': {
        'type': 'lgb',
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
    },
    'et_fast': {
        'type': 'et',
        'params': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
    },
    'xgb_fast': {
        'type': 'xgb',
        'params': {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1}
    },
}

random.seed(42)
np.random.seed(42)


def generate_massive_experiments(all_features, target_count=50000):
    """Generate massive number of unique feature combinations."""
    experiments = []
    seen = set()
    exp_id = 1

    n = len(all_features)
    base_no_matchup = [f for f in all_features if f not in MATCHUP_FEATURES]

    # Feature groups
    hero = [f for f in all_features if f.startswith('hero_')]
    cs = [f for f in all_features if f.startswith('cs_')]
    cs_no_mu = [f for f in cs if f not in MATCHUP_FEATURES]
    tc = [f for f in all_features if f.startswith('tc_')]
    tc_diff = [f for f in tc if '_diff' in f]
    tc_team = [f for f in tc if 'hero_team' in f]
    platform = [f for f in all_features if f.startswith('platform_')]
    other = [f for f in ['siamese_score', 'smurf_score', 'expected_wr'] if f in all_features]

    def add_exp(name, features, model='lgb_fast'):
        nonlocal exp_id
        if len(features) == 0:
            return False
        key = (tuple(sorted(features)), model)
        if key not in seen:
            seen.add(key)
            experiments.append({'id': exp_id, 'name': name, 'features': list(features), 'model': model})
            exp_id += 1
            return True
        return False

    print("Generating experiments...")

    # ==========================================================================
    # 1. BASELINES (10)
    # ==========================================================================
    for model in MODEL_CONFIGS.keys():
        add_exp(f'ALL_{model}', all_features, model)
        add_exp(f'NO_MATCHUP_{model}', base_no_matchup, model)

    # ==========================================================================
    # 2. REMOVE SINGLE FEATURES - ALL MODELS (240)
    # ==========================================================================
    for f in all_features:
        for model in MODEL_CONFIGS.keys():
            add_exp(f'NO_{f[:15]}_{model[:3]}', [x for x in all_features if x != f], model)

    # ==========================================================================
    # 3. REMOVE FROM NO_MATCHUP BASE (216)
    # ==========================================================================
    for f in base_no_matchup:
        for model in MODEL_CONFIGS.keys():
            add_exp(f'NM_NO_{f[:12]}_{model[:3]}', [x for x in base_no_matchup if x != f], model)

    # ==========================================================================
    # 4. TOP-N FEATURES (200)
    # ==========================================================================
    importance_order = [
        'siamese_score', 'expected_wr', 'hero_rank_numeric', 'hero_lp', 'hero_winrate',
        'hero_total_games', 'hero_personal_champ_wr', 'hero_cm_points_log', 'smurf_score',
        'hero_personal_overall_wr', 'hero_cm_level_feat', 'hero_wr_rank_mismatch',
        'hero_personal_champ_games', 'hero_is_blue_feat', 'cs_hero_champ_wr_at_elo_role',
        'cs_hero_champ_wr_at_elo', 'cs_hero_champ_wr_at_role', 'cs_hero_team_avg_wr',
        'cs_enemy_team_avg_wr', 'cs_hero_vs_enemy_wr', 'cs_hero_mastery_zscore',
        'tc_scaling_diff', 'tc_tank_diff', 'tc_engage_diff', 'tc_tankiness_diff',
        'tc_hard_engage_diff', 'tc_scaling_tier_diff', 'tc_frontline_diff',
        'tc_tank_shredder_diff', 'tc_poke_diff', 'tc_disengage_diff',
    ]
    importance_order = [f for f in importance_order if f in all_features]

    for n_feat in range(2, min(55, len(importance_order)+1)):
        for model in MODEL_CONFIGS.keys():
            add_exp(f'TOP_{n_feat}_{model[:3]}', importance_order[:n_feat], model)

    # ==========================================================================
    # 5. REMOVE ALL PAIRS (1770 pairs × 1 model = 1770)
    # ==========================================================================
    all_pairs = list(combinations(all_features, 2))
    for f1, f2 in all_pairs:
        add_exp(f'NO2_{f1[:8]}_{f2[:8]}', [x for x in all_features if x not in [f1, f2]])

    # ==========================================================================
    # 6. REMOVE ALL TRIPLETS FROM TOP 30 (4060 triplets)
    # ==========================================================================
    top30 = all_features[:30]
    all_triplets = list(combinations(top30, 3))
    for f1, f2, f3 in all_triplets:
        name = f'NO3_{f1[:5]}_{f2[:5]}_{f3[:5]}'[:40]
        add_exp(name, [x for x in all_features if x not in [f1, f2, f3]])

    # ==========================================================================
    # 7. RANDOM SUBSETS - MANY SIZES (5000)
    # ==========================================================================
    for size in range(3, 56):
        n_samples = max(50, 100 - size)  # More samples for smaller sizes
        for i in range(n_samples):
            subset = random.sample(all_features, size)
            model = random.choice(list(MODEL_CONFIGS.keys()))
            add_exp(f'R{size}_{i}_{model[:1]}', subset, model)

    # ==========================================================================
    # 8. RANDOM SUBSETS FROM NO_MATCHUP (3000)
    # ==========================================================================
    for size in range(3, len(base_no_matchup)):
        n_samples = max(30, 80 - size)
        for i in range(n_samples):
            subset = random.sample(base_no_matchup, size)
            model = random.choice(list(MODEL_CONFIGS.keys()))
            add_exp(f'NM_R{size}_{i}_{model[:1]}', subset, model)

    # ==========================================================================
    # 9. GROUP COMBINATIONS (2000)
    # ==========================================================================
    for i in range(2000):
        combo = []
        # Random hero subset
        if random.random() > 0.2:
            n_hero = random.randint(1, len(hero))
            combo.extend(random.sample(hero, n_hero))
        # Random CS subset (no matchup)
        if random.random() > 0.3:
            n_cs = random.randint(1, len(cs_no_mu))
            combo.extend(random.sample(cs_no_mu, n_cs))
        # Random TC subset
        if random.random() > 0.4:
            n_tc = random.randint(1, len(tc))
            combo.extend(random.sample(tc, n_tc))
        # Random platform subset
        if random.random() > 0.7:
            n_plat = random.randint(1, len(platform))
            combo.extend(random.sample(platform, n_plat))
        # Other features
        if random.random() > 0.2:
            n_other = random.randint(1, len(other))
            combo.extend(random.sample(other, n_other))

        if len(combo) >= 3:
            model = random.choice(list(MODEL_CONFIGS.keys()))
            add_exp(f'GRP_{i}_{model[:1]}', list(set(combo)), model)

    # ==========================================================================
    # 10. BEST BASE VARIATIONS (1000)
    # ==========================================================================
    best_base = hero + other + cs_no_mu

    for i in range(500):
        # Remove random features
        n_remove = random.randint(1, min(10, len(best_base)-3))
        to_remove = random.sample(best_base, n_remove)
        remaining = [f for f in best_base if f not in to_remove]
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'BB_M{n_remove}_{i}_{model[:1]}', remaining, model)

    for i in range(500):
        # Add TC features
        n_add = random.randint(1, len(tc))
        to_add = random.sample(tc, n_add)
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'BB_P{n_add}_{i}_{model[:1]}', best_base + to_add, model)

    # ==========================================================================
    # 11. LEAVE-K-OUT FROM VARIOUS BASES (2000)
    # ==========================================================================
    bases = [
        ('ALL', all_features),
        ('NM', base_no_matchup),
        ('HERO_OTHER', hero + other),
        ('BEST', hero + other + cs_no_mu + tc_diff),
    ]

    for base_name, base_features in bases:
        for k in range(1, min(15, len(base_features))):
            n_samples = max(10, 50 - k*3)
            for i in range(n_samples):
                to_remove = random.sample(base_features, k)
                remaining = [f for f in base_features if f not in to_remove]
                if len(remaining) >= 3:
                    model = random.choice(list(MODEL_CONFIGS.keys()))
                    add_exp(f'{base_name}_LKO{k}_{i}_{model[:1]}', remaining, model)

    # ==========================================================================
    # 12. FEATURE IMPORTANCE BASED (1000)
    # ==========================================================================
    # Forward selection style
    for target in range(5, 50, 2):
        for i in range(20):
            # Start with top features and add random
            start_n = random.randint(2, min(5, target))
            selected = importance_order[:start_n]
            remaining = [f for f in all_features if f not in selected]
            while len(selected) < target and remaining:
                next_f = random.choice(remaining)
                selected.append(next_f)
                remaining.remove(next_f)
            model = random.choice(list(MODEL_CONFIGS.keys()))
            add_exp(f'FWD_{target}_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 13. TC-FOCUSED EXPERIMENTS (500)
    # ==========================================================================
    for i in range(250):
        # TC diff + random other
        other_feats = [f for f in all_features if f not in tc]
        n_other = random.randint(5, 25)
        selected = tc_diff + random.sample(other_feats, min(n_other, len(other_feats)))
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'TCD_{i}_{model[:1]}', selected, model)

    for i in range(250):
        # TC team + random other
        other_feats = [f for f in all_features if f not in tc]
        n_other = random.randint(5, 25)
        selected = tc_team + random.sample(other_feats, min(n_other, len(other_feats)))
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'TCT_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 14. HERO-FOCUSED EXPERIMENTS (500)
    # ==========================================================================
    for i in range(500):
        # Hero subset + other features
        n_hero = random.randint(3, len(hero))
        hero_sub = random.sample(hero, n_hero)
        other_feats = [f for f in all_features if f not in hero]
        n_other = random.randint(2, 15)
        selected = hero_sub + random.sample(other_feats, min(n_other, len(other_feats)))
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'HERO_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 15. CS-FOCUSED EXPERIMENTS (500)
    # ==========================================================================
    for i in range(500):
        n_cs = random.randint(2, len(cs_no_mu))
        cs_sub = random.sample(cs_no_mu, n_cs)
        other_feats = [f for f in all_features if f not in cs]
        n_other = random.randint(3, 20)
        selected = cs_sub + random.sample(other_feats, min(n_other, len(other_feats)))
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'CS_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 16. PLATFORM VARIATIONS (500)
    # ==========================================================================
    for i in range(500):
        n_plat = random.randint(0, len(platform))
        if n_plat > 0:
            plat_sub = random.sample(platform, n_plat)
        else:
            plat_sub = []
        other_feats = [f for f in all_features if f not in platform]
        n_other = random.randint(10, 40)
        selected = plat_sub + random.sample(other_feats, min(n_other, len(other_feats)))
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'PLAT_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 17. MINIMAL EXPERIMENTS (500)
    # ==========================================================================
    for i in range(500):
        n_feat = random.randint(2, 8)
        selected = random.sample(all_features, n_feat)
        model = random.choice(list(MODEL_CONFIGS.keys()))
        add_exp(f'MIN_{n_feat}_{i}_{model[:1]}', selected, model)

    # ==========================================================================
    # 18. SIAMESE-FOCUSED (300)
    # ==========================================================================
    if 'siamese_score' in all_features:
        for i in range(300):
            n_feat = random.randint(3, 30)
            other_feats = [f for f in all_features if f != 'siamese_score']
            selected = ['siamese_score'] + random.sample(other_feats, n_feat-1)
            model = random.choice(list(MODEL_CONFIGS.keys()))
            add_exp(f'SIAM_{i}_{model[:1]}', selected, model)

    print(f"Generated {len(experiments)} unique experiments")

    # Shuffle to distribute experiment types
    random.shuffle(experiments)

    return experiments


def train_model(model_config, X_train, y_train, X_test, y_test):
    """Train a model and return AUC."""
    model_type = model_config['type']
    params = model_config['params']

    if model_type == 'lgb':
        model = lgb.LGBMClassifier(**params)
    elif model_type == 'et':
        model = ExtraTreesClassifier(**params)
    elif model_type == 'xgb':
        model = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)

    del model
    gc.collect()

    return auc


def save_checkpoint(results, filename):
    """Save intermediate results."""
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / filename, index=False)
    return df


def main():
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=MAX_RUNTIME_HOURS)

    print("="*70)
    print("OVERNIGHT FEATURE ABLATION - 12 HOURS")
    print(f"Started: {start_time}")
    print(f"Will run until: {end_time}")
    print("="*70)

    # Load datasets
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)

    datasets = {}
    for name, paths in DATASETS.items():
        train = pd.read_parquet(paths['train'])
        test = pd.read_parquet(paths['test'])
        datasets[name] = {
            'X_train': None,
            'y_train': train['hero_win'].values,
            'X_test': None,
            'y_test': test['hero_win'].values,
            'train_df': train,
            'test_df': test,
        }
        print(f"{name}: Train {len(train):,}, Test {len(test):,}")

    # Get features
    all_features = [c for c in datasets['random']['train_df'].columns
                   if c not in EXCLUDE_COLS and datasets['random']['train_df'][c].dtype in ['int64', 'float64']]
    print(f"\nAvailable features: {len(all_features)}")

    # Generate experiments
    print("\n" + "="*60)
    print("GENERATING EXPERIMENTS")
    print("="*60)
    experiments = generate_massive_experiments(all_features)
    total_runs = len(experiments) * len(DATASETS)
    print(f"Total runs: {total_runs:,}")

    # Run experiments
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)

    results = []
    run_count = 0
    best_random = 0
    best_main = 0

    for exp in experiments:
        # Check time limit
        if datetime.now() >= end_time:
            print(f"\n*** TIME LIMIT REACHED ({MAX_RUNTIME_HOURS}h) ***")
            break

        features = [f for f in exp['features'] if f in all_features]
        if len(features) == 0:
            continue

        model_config = MODEL_CONFIGS.get(exp['model'], MODEL_CONFIGS['lgb_fast'])

        for dataset_name, data in datasets.items():
            run_count += 1

            # Get feature matrix
            valid_features = [f for f in features if f in data['train_df'].columns]
            if len(valid_features) == 0:
                continue

            X_train = data['train_df'][valid_features].fillna(0).values
            X_test = data['test_df'][valid_features].fillna(0).values

            try:
                auc = train_model(model_config, X_train, data['y_train'], X_test, data['y_test'])
            except Exception as e:
                auc = 0.5

            result = {
                'experiment_id': exp['id'],
                'experiment_name': exp['name'],
                'model': exp['model'],
                'dataset': dataset_name,
                'n_features': len(valid_features),
                'auc': auc,
                'features': ','.join(valid_features[:10]) + ('...' if len(valid_features) > 10 else ''),
            }
            results.append(result)

            # Track best
            if dataset_name == 'random' and auc > best_random:
                best_random = auc
            if dataset_name == 'main' and auc > best_main:
                best_main = auc

            # Progress update
            if run_count % 200 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = run_count / elapsed * 3600
                remaining = (end_time - datetime.now()).total_seconds() / 3600
                print(f"[{run_count:6d}] {exp['name']:<30} @ {dataset_name:<6} | "
                      f"AUC={auc:.4f} | Best: R={best_random:.4f} M={best_main:.4f} | "
                      f"{remaining:.1f}h left | {rate:.0f}/h")

            # Checkpoint
            if run_count % CHECKPOINT_EVERY == 0:
                save_checkpoint(results, 'feature_ablation_overnight_checkpoint.csv')

    # Final save
    print("\n" + "="*60)
    print("SAVING FINAL RESULTS")
    print("="*60)

    results_df = save_checkpoint(results, 'feature_ablation_overnight.csv')
    print(f"Saved: reports/feature_ablation_overnight.csv")

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for dataset_name in DATASETS.keys():
        subset = results_df[results_df['dataset'] == dataset_name].copy()
        subset = subset.sort_values('auc', ascending=False)

        print(f"\n--- {dataset_name.upper()} TOP 30 ---")
        print(f"{'Rank':<5} {'Experiment':<35} {'Model':<10} {'AUC':>8} {'#Feat':>6}")
        print("-" * 70)

        for i, (_, row) in enumerate(subset.head(30).iterrows()):
            print(f"{i+1:<5} {row['experiment_name']:<35} {row['model']:<10} {row['auc']:>8.4f} {row['n_features']:>6}")

        print(f"\nStatistics for {dataset_name}:")
        print(f"  Total: {len(subset)}")
        print(f"  Mean AUC: {subset['auc'].mean():.4f}")
        print(f"  Max AUC: {subset['auc'].max():.4f}")
        print(f"  Min AUC: {subset['auc'].min():.4f}")
        print(f"  AUC > 0.58: {(subset['auc'] > 0.58).sum()}")
        print(f"  AUC > 0.57: {(subset['auc'] > 0.57).sum()}")

    # Best across both
    print("\n" + "="*70)
    print("BEST ACROSS BOTH DATASETS")
    print("="*70)

    pivot = results_df.pivot_table(
        index=['experiment_name', 'model'],
        columns='dataset',
        values='auc'
    ).reset_index()

    if 'random' in pivot.columns and 'main' in pivot.columns:
        pivot['avg'] = (pivot['random'] + pivot['main']) / 2
        pivot = pivot.sort_values('avg', ascending=False)

        print(f"\n{'Experiment':<35} {'Model':<10} {'Random':>8} {'Main':>8} {'Avg':>8}")
        print("-" * 75)

        for _, row in pivot.head(50).iterrows():
            print(f"{row['experiment_name']:<35} {row['model']:<10} {row['random']:>8.4f} {row['main']:>8.4f} {row['avg']:>8.4f}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n" + "="*70)
    print(f"COMPLETED: {datetime.now()}")
    print(f"Total experiments: {len(results)}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Rate: {len(results)/(elapsed/3600):.0f} experiments/hour")
    print("="*70)

    return results_df


if __name__ == '__main__':
    results = main()
