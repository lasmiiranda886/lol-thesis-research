"""
Additional Thesis Analysis - Chapter 4 Extended Results
========================================================
Generates additional figures and statistics for the thesis:
1. ROC Curves
2. AUC by Elo Tier
3. Champion Winrates by Region and Elo
4. Blue/Red Side Analysis
5. Performance by Role
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from pathlib import Path

# Paths
DATA_DIR = Path('data/interim/aggregate')
MODEL_DIR = Path('models')
OUTPUT_DIR = Path('thesis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme (consistent with other thesis figures)
COLORS = {
    'random': '#2ecc71',
    'main': '#3498db',
    'primary': '#2c3e50',
    'secondary': '#e74c3c',
    'tertiary': '#f39c12'
}

# Global model data (loaded once)
MODEL_DATA = {}


def load_data():
    """Load test datasets."""
    random_test = pd.read_parquet(DATA_DIR / 'hero_dataset_random_test_final.parquet')
    main_test = pd.read_parquet(DATA_DIR / 'hero_dataset_main_test_final.parquet')
    return random_test, main_test


def load_models():
    """Load ensemble models from the ablation study."""
    global MODEL_DATA

    # Load the hero_focused_20 models (closest to our final feature set)
    with open(MODEL_DIR / 'hero_focused_20_random_models.pkl', 'rb') as f:
        MODEL_DATA['random'] = pickle.load(f)
    with open(MODEL_DIR / 'hero_focused_20_main_models.pkl', 'rb') as f:
        MODEL_DATA['main'] = pickle.load(f)

    print(f"Loaded models with {len(MODEL_DATA['random']['features'])} features")
    return MODEL_DATA['random'], MODEL_DATA['main']


def get_features():
    """Get the feature list from loaded models."""
    if 'random' in MODEL_DATA:
        return MODEL_DATA['random']['features']
    return []


def get_ensemble_predictions(X, model_data):
    """Get weighted ensemble predictions from model dict."""
    models = model_data['models']
    weights = model_data.get('weights', {'et': 0.30, 'lgb': 0.40, 'xgb': 0.30})

    pred_et = models['et'].predict_proba(X)[:, 1]
    pred_lgb = models['lgb'].predict_proba(X)[:, 1]
    pred_xgb = models['xgb'].predict_proba(X)[:, 1]

    return weights['et'] * pred_et + weights['lgb'] * pred_lgb + weights['xgb'] * pred_xgb


def fig_roc_curves():
    """Generate ROC curves for both datasets."""
    print("Generating ROC curves...")

    random_test, main_test = load_data()
    random_model, main_model = load_models()
    features = get_features()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Random Hero ROC
    X_random = random_test[features].fillna(0)
    y_random = random_test['hero_win'].astype(int)
    y_pred_random = get_ensemble_predictions(X_random, random_model)

    fpr_r, tpr_r, _ = roc_curve(y_random, y_pred_random)
    auc_r = auc(fpr_r, tpr_r)

    axes[0].plot(fpr_r, tpr_r, color=COLORS['random'], lw=2,
                 label=f'Ensemble (AUC = {auc_r:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')
    axes[0].fill_between(fpr_r, tpr_r, alpha=0.2, color=COLORS['random'])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC-Kurve: Random Hero Dataset')
    axes[0].legend(loc='lower right')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Main-Hero ROC
    X_main = main_test[features].fillna(0)
    y_main = main_test['hero_win'].astype(int)
    y_pred_main = get_ensemble_predictions(X_main, main_model)

    fpr_m, tpr_m, _ = roc_curve(y_main, y_pred_main)
    auc_m = auc(fpr_m, tpr_m)

    axes[1].plot(fpr_m, tpr_m, color=COLORS['main'], lw=2,
                 label=f'Ensemble (AUC = {auc_m:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')
    axes[1].fill_between(fpr_m, tpr_m, alpha=0.2, color=COLORS['main'])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC-Kurve: Main-Hero Dataset')
    axes[1].legend(loc='lower right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'roc_curves.pdf')
    plt.close()
    print(f"✓ Saved: roc_curves.png/pdf (Random AUC={auc_r:.4f}, Main AUC={auc_m:.4f})")

    return auc_r, auc_m


def fig_auc_by_elo():
    """Calculate and visualize AUC by Elo tier."""
    print("Calculating AUC by Elo tier...")

    random_test, main_test = load_data()
    features = get_features()

    # Map rank_numeric to tier name
    rank_map = {
        3: 'Silver', 4: 'Gold', 5: 'Platinum', 6: 'Emerald', 7: 'Diamond'
    }

    results = {'tier': [], 'random_auc': [], 'main_auc': [], 'random_n': [], 'main_n': []}

    for rank_num, tier_name in rank_map.items():
        # Random dataset
        mask_r = random_test['hero_rank_numeric'] == rank_num
        if mask_r.sum() >= 100:
            X_r = random_test.loc[mask_r, features].fillna(0)
            y_r = random_test.loc[mask_r, 'hero_win'].astype(int)
            y_pred_r = get_ensemble_predictions(X_r, MODEL_DATA['random'])
            auc_r = roc_auc_score(y_r, y_pred_r)
            n_r = mask_r.sum()
        else:
            auc_r = np.nan
            n_r = mask_r.sum()

        # Main dataset
        mask_m = main_test['hero_rank_numeric'] == rank_num
        if mask_m.sum() >= 100:
            X_m = main_test.loc[mask_m, features].fillna(0)
            y_m = main_test.loc[mask_m, 'hero_win'].astype(int)
            y_pred_m = get_ensemble_predictions(X_m, MODEL_DATA['main'])
            auc_m = roc_auc_score(y_m, y_pred_m)
            n_m = mask_m.sum()
        else:
            auc_m = np.nan
            n_m = mask_m.sum()

        results['tier'].append(tier_name)
        results['random_auc'].append(auc_r)
        results['main_auc'].append(auc_m)
        results['random_n'].append(n_r)
        results['main_n'].append(n_m)

        print(f"  {tier_name}: Random AUC={auc_r:.4f} (n={n_r}), Main AUC={auc_m:.4f} (n={n_m})")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results['tier']))
    width = 0.35

    # Filter out NaN values for plotting
    random_aucs = [a if not np.isnan(a) else 0 for a in results['random_auc']]
    main_aucs = [a if not np.isnan(a) else 0 for a in results['main_auc']]

    bars1 = ax.bar(x - width/2, random_aucs, width, label='Random Hero', color=COLORS['random'])
    bars2 = ax.bar(x + width/2, main_aucs, width, label='Main-Hero', color=COLORS['main'])

    # Add baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')

    ax.set_xlabel('Elo Tier')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Modell-Performance nach Elo-Tier')
    ax.set_xticks(x)
    ax.set_xticklabels(results['tier'])
    ax.legend()
    ax.set_ylim(0.48, 0.62)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'auc_by_elo.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'auc_by_elo.pdf')
    plt.close()
    print("✓ Saved: auc_by_elo.png/pdf")

    return pd.DataFrame(results)


def fig_champion_winrate_by_elo():
    """Analyze champion winrate variation by Elo."""
    print("Analyzing champion winrates by Elo...")

    random_train = pd.read_parquet(DATA_DIR / 'hero_dataset_random_train_final.parquet')

    # Get top 20 most played champions
    top_champs = random_train['championName'].value_counts().head(20).index.tolist()

    # Calculate winrate by elo for top champions
    rank_map = {3: 'Silver', 4: 'Gold', 5: 'Platinum', 6: 'Emerald'}

    results = []
    for champ in top_champs:
        champ_data = random_train[random_train['championName'] == champ]
        for rank_num, tier_name in rank_map.items():
            tier_data = champ_data[champ_data['hero_rank_numeric'] == rank_num]
            if len(tier_data) >= 50:
                wr = tier_data['hero_win'].mean()
                results.append({
                    'champion': champ,
                    'tier': tier_name,
                    'winrate': wr,
                    'games': len(tier_data)
                })

    df = pd.DataFrame(results)

    # Find champions with biggest elo-dependent variance
    variance_by_champ = df.groupby('champion')['winrate'].var().sort_values(ascending=False)
    high_var_champs = variance_by_champ.head(8).index.tolist()

    print(f"  Champions with highest elo-dependent winrate variance:")
    for champ in high_var_champs[:5]:
        champ_df = df[df['champion'] == champ]
        min_wr = champ_df['winrate'].min()
        max_wr = champ_df['winrate'].max()
        print(f"    {champ}: {min_wr:.1%} - {max_wr:.1%} (diff: {(max_wr-min_wr)*100:.1f}%)")

    # Create heatmap-style figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pivot for heatmap
    pivot_df = df[df['champion'].isin(high_var_champs)].pivot(
        index='champion', columns='tier', values='winrate'
    )
    # Reorder columns
    pivot_df = pivot_df[['Silver', 'Gold', 'Platinum', 'Emerald']]

    # Plot as grouped bar chart
    x = np.arange(len(high_var_champs))
    width = 0.2
    tiers = ['Silver', 'Gold', 'Platinum', 'Emerald']
    colors_tier = ['#95a5a6', '#f1c40f', '#1abc9c', '#9b59b6']

    for i, (tier, color) in enumerate(zip(tiers, colors_tier)):
        values = [pivot_df.loc[champ, tier] if champ in pivot_df.index else 0.5
                  for champ in high_var_champs]
        ax.bar(x + i*width, values, width, label=tier, color=color)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Champion')
    ax.set_ylabel('Winrate')
    ax.set_title('Champion-Winrate nach Elo-Tier (höchste Varianz)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(high_var_champs, rotation=45, ha='right')
    ax.legend(title='Elo Tier')
    ax.set_ylim(0.40, 0.60)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_elo.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_elo.pdf')
    plt.close()
    print("✓ Saved: champion_winrate_by_elo.png/pdf")

    return df


def fig_champion_winrate_by_region():
    """Analyze champion winrate variation by region."""
    print("Analyzing champion winrates by region...")

    random_train = pd.read_parquet(DATA_DIR / 'hero_dataset_random_train_final.parquet')

    # Get top regions
    top_regions = random_train['platform'].value_counts().head(6).index.tolist()

    # Get top 15 most played champions
    top_champs = random_train['championName'].value_counts().head(15).index.tolist()

    results = []
    for champ in top_champs:
        champ_data = random_train[random_train['championName'] == champ]
        for region in top_regions:
            region_data = champ_data[champ_data['platform'] == region]
            if len(region_data) >= 30:
                wr = region_data['hero_win'].mean()
                results.append({
                    'champion': champ,
                    'region': region.upper(),
                    'winrate': wr,
                    'games': len(region_data)
                })

    df = pd.DataFrame(results)

    # Find champions with biggest regional variance
    variance_by_champ = df.groupby('champion')['winrate'].var().sort_values(ascending=False)
    high_var_champs = variance_by_champ.head(6).index.tolist()

    print(f"  Champions with highest regional winrate variance:")
    for champ in high_var_champs[:3]:
        champ_df = df[df['champion'] == champ]
        min_wr = champ_df['winrate'].min()
        max_wr = champ_df['winrate'].max()
        min_region = champ_df.loc[champ_df['winrate'].idxmin(), 'region']
        max_region = champ_df.loc[champ_df['winrate'].idxmax(), 'region']
        print(f"    {champ}: {min_wr:.1%} ({min_region}) - {max_wr:.1%} ({max_region})")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    pivot_df = df[df['champion'].isin(high_var_champs)].pivot(
        index='champion', columns='region', values='winrate'
    )

    x = np.arange(len(high_var_champs))
    width = 0.12
    regions = [r.upper() for r in top_regions]
    colors_region = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, (region, color) in enumerate(zip(regions, colors_region)):
        values = [pivot_df.loc[champ, region] if (champ in pivot_df.index and region in pivot_df.columns) else 0.5
                  for champ in high_var_champs]
        ax.bar(x + i*width, values, width, label=region, color=color)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Champion')
    ax.set_ylabel('Winrate')
    ax.set_title('Champion-Winrate nach Region (höchste Varianz)')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(high_var_champs, rotation=45, ha='right')
    ax.legend(title='Region', loc='upper right')
    ax.set_ylim(0.40, 0.60)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.pdf')
    plt.close()
    print("✓ Saved: champion_winrate_by_region.png/pdf")

    return df


def fig_blue_red_side():
    """Analyze blue vs red side win rates and prediction accuracy."""
    print("Analyzing blue/red side...")

    random_test, main_test = load_data()
    features = get_features()

    results = {'side': [], 'dataset': [], 'winrate': [], 'auc': [], 'n': []}

    for dataset_name, test_df, model_data in [('Random', random_test, MODEL_DATA['random']),
                                               ('Main', main_test, MODEL_DATA['main'])]:
        for side, side_val in [('Blue', 1), ('Red', 0)]:
            mask = test_df['hero_is_blue_feat'] == side_val
            side_df = test_df[mask]

            wr = side_df['hero_win'].mean()

            X = side_df[features].fillna(0)
            y = side_df['hero_win'].astype(int)
            y_pred = get_ensemble_predictions(X, model_data)
            auc_score = roc_auc_score(y, y_pred)

            results['side'].append(side)
            results['dataset'].append(dataset_name)
            results['winrate'].append(wr)
            results['auc'].append(auc_score)
            results['n'].append(len(side_df))

            print(f"  {dataset_name} {side}: WR={wr:.1%}, AUC={auc_score:.4f}, n={len(side_df)}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = pd.DataFrame(results)

    # Winrate comparison
    x = np.arange(2)
    width = 0.35

    random_wr = df[(df['dataset'] == 'Random')]['winrate'].values
    main_wr = df[(df['dataset'] == 'Main')]['winrate'].values

    axes[0].bar(x - width/2, random_wr, width, label='Random Hero', color=COLORS['random'])
    axes[0].bar(x + width/2, main_wr, width, label='Main-Hero', color=COLORS['main'])
    axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Seite')
    axes[0].set_ylabel('Winrate')
    axes[0].set_title('Winrate nach Seite')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Blue', 'Red'])
    axes[0].legend()
    axes[0].set_ylim(0.45, 0.55)

    # Add value labels
    for i, (r_wr, m_wr) in enumerate(zip(random_wr, main_wr)):
        axes[0].annotate(f'{r_wr:.1%}', xy=(i - width/2, r_wr), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=9)
        axes[0].annotate(f'{m_wr:.1%}', xy=(i + width/2, m_wr), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=9)

    # AUC comparison
    random_auc = df[(df['dataset'] == 'Random')]['auc'].values
    main_auc = df[(df['dataset'] == 'Main')]['auc'].values

    axes[1].bar(x - width/2, random_auc, width, label='Random Hero', color=COLORS['random'])
    axes[1].bar(x + width/2, main_auc, width, label='Main-Hero', color=COLORS['main'])
    axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Seite')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('Modell-AUC nach Seite')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Blue', 'Red'])
    axes[1].legend()
    axes[1].set_ylim(0.50, 0.62)

    # Add value labels
    for i, (r_auc, m_auc) in enumerate(zip(random_auc, main_auc)):
        axes[1].annotate(f'{r_auc:.3f}', xy=(i - width/2, r_auc), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=9)
        axes[1].annotate(f'{m_auc:.3f}', xy=(i + width/2, m_auc), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'blue_red_side.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'blue_red_side.pdf')
    plt.close()
    print("✓ Saved: blue_red_side.png/pdf")

    return df


def fig_auc_by_role():
    """Calculate and visualize AUC by role (position)."""
    print("Calculating AUC by role...")

    random_test, main_test = load_data()
    features = get_features()

    roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    role_labels = ['Top', 'Jungle', 'Mid', 'ADC', 'Support']

    results = {'role': [], 'random_auc': [], 'main_auc': [], 'random_n': [], 'main_n': []}

    for role, label in zip(roles, role_labels):
        # Random dataset
        mask_r = random_test['teamPosition'] == role
        if mask_r.sum() >= 100:
            X_r = random_test.loc[mask_r, features].fillna(0)
            y_r = random_test.loc[mask_r, 'hero_win'].astype(int)
            y_pred_r = get_ensemble_predictions(X_r, MODEL_DATA['random'])
            auc_r = roc_auc_score(y_r, y_pred_r)
        else:
            auc_r = np.nan

        # Main dataset
        mask_m = main_test['teamPosition'] == role
        if mask_m.sum() >= 100:
            X_m = main_test.loc[mask_m, features].fillna(0)
            y_m = main_test.loc[mask_m, 'hero_win'].astype(int)
            y_pred_m = get_ensemble_predictions(X_m, MODEL_DATA['main'])
            auc_m = roc_auc_score(y_m, y_pred_m)
        else:
            auc_m = np.nan

        results['role'].append(label)
        results['random_auc'].append(auc_r)
        results['main_auc'].append(auc_m)
        results['random_n'].append(mask_r.sum())
        results['main_n'].append(mask_m.sum())

        print(f"  {label}: Random AUC={auc_r:.4f} (n={mask_r.sum()}), Main AUC={auc_m:.4f} (n={mask_m.sum()})")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(role_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, results['random_auc'], width, label='Random Hero', color=COLORS['random'])
    bars2 = ax.bar(x + width/2, results['main_auc'], width, label='Main-Hero', color=COLORS['main'])

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')

    ax.set_xlabel('Rolle')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Modell-Performance nach Rolle')
    ax.set_xticks(x)
    ax.set_xticklabels(role_labels)
    ax.legend()
    ax.set_ylim(0.50, 0.62)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'auc_by_role.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'auc_by_role.pdf')
    plt.close()
    print("✓ Saved: auc_by_role.png/pdf")

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Additional Thesis Analysis - Chapter 4 Extended Results")
    print("=" * 60)

    # Generate all figures
    print("\n--- ROC Curves ---")
    fig_roc_curves()

    print("\n--- AUC by Elo Tier ---")
    elo_results = fig_auc_by_elo()

    print("\n--- Champion Winrates by Elo ---")
    fig_champion_winrate_by_elo()

    print("\n--- Champion Winrates by Region ---")
    fig_champion_winrate_by_region()

    print("\n--- Blue/Red Side Analysis ---")
    side_results = fig_blue_red_side()

    print("\n--- AUC by Role ---")
    role_results = fig_auc_by_role()

    print("\n" + "=" * 60)
    print(f"All additional figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
