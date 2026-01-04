"""
Champion Variability Charts - Multiple Visualization Types
==========================================================
Generates 4 different chart types for champion winrate analysis:
1. Heatmap - Overview of all champions across categories
2. Slope Chart - Clear change visualization between two points
3. Diverging Bar Chart - Centered at 50% baseline
4. Dot Plot with Range - Shows min/max/mean variation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
import seaborn as sns

# Paths
DATA_DIR = Path('data/interim/aggregate')
OUTPUT_DIR = Path('thesis/figures/chapter4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'positive': '#2ecc71',  # Green - above 50%
    'negative': '#e74c3c',  # Red - below 50%
    'neutral': '#95a5a6',   # Gray
    'primary': '#2c3e50',
    'secondary': '#3498db',
}

# Elo tier colors
ELO_COLORS = {
    'Silver': '#95a5a6',
    'Gold': '#f1c40f',
    'Platinum': '#1abc9c',
    'Emerald': '#9b59b6',
    'Diamond': '#3498db'
}

# Region colors
REGION_COLORS = {
    'OC1': '#e74c3c',
    'KR': '#3498db',
    'JP1': '#2ecc71',
    'BR1': '#f39c12',
    'EUW1': '#9b59b6',
    'NA1': '#1abc9c'
}


def load_data():
    """Load training data for champion analysis."""
    random_train = pd.read_parquet(DATA_DIR / 'hero_dataset_random_train_final.parquet')
    return random_train


def get_champion_elo_data(df, min_games=50):
    """Calculate champion winrates by elo tier."""
    rank_map = {3: 'Silver', 4: 'Gold', 5: 'Platinum', 6: 'Emerald', 7: 'Diamond'}

    # Get top 20 most played champions
    top_champs = df['championName'].value_counts().head(20).index.tolist()

    results = []
    for champ in top_champs:
        champ_data = df[df['championName'] == champ]
        for rank_num, tier_name in rank_map.items():
            tier_data = champ_data[champ_data['hero_rank_numeric'] == rank_num]
            if len(tier_data) >= min_games:
                wr = tier_data['hero_win'].mean()
                results.append({
                    'champion': champ,
                    'tier': tier_name,
                    'tier_order': rank_num,
                    'winrate': wr,
                    'games': len(tier_data)
                })

    return pd.DataFrame(results)


def get_champion_region_data(df, min_games=30):
    """Calculate champion winrates by region."""
    top_regions = df['platform'].value_counts().head(6).index.tolist()
    top_champs = df['championName'].value_counts().head(15).index.tolist()

    results = []
    for champ in top_champs:
        champ_data = df[df['championName'] == champ]
        for region in top_regions:
            region_data = champ_data[champ_data['platform'] == region]
            if len(region_data) >= min_games:
                wr = region_data['hero_win'].mean()
                results.append({
                    'champion': champ,
                    'region': region.upper(),
                    'winrate': wr,
                    'games': len(region_data)
                })

    return pd.DataFrame(results)


def get_high_variance_champions(df, group_col, n=8):
    """Get champions with highest winrate variance."""
    variance = df.groupby('champion')['winrate'].agg(['var', 'min', 'max', 'mean'])
    variance['range'] = variance['max'] - variance['min']
    variance = variance.sort_values('range', ascending=False)
    return variance.head(n).index.tolist()


# =============================================================================
# CHAMPION WINRATE BY ELO - 4 CHART TYPES
# =============================================================================

def fig_elo_heatmap(df, champions):
    """Heatmap showing champion winrates across elo tiers."""
    print("  Creating Elo Heatmap...")

    # Pivot data
    pivot = df[df['champion'].isin(champions)].pivot(
        index='champion', columns='tier', values='winrate'
    )
    pivot = pivot[['Silver', 'Gold', 'Platinum', 'Emerald', 'Diamond']]
    pivot = pivot.reindex(champions)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create diverging colormap centered at 0.5
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap=cmap, center=0.5,
                vmin=0.44, vmax=0.56, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Winrate', 'format': '%.0f%%'})

    ax.set_title('Champion-Winrate nach Elo-Tier (Heatmap)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Elo Tier', fontsize=12)
    ax.set_ylabel('Champion', fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_elo_heatmap.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_elo_heatmap.pdf')
    plt.close()
    print("    Saved: champion_elo_heatmap.png/pdf")


def fig_elo_slope_chart(df, champions):
    """Slope chart showing change from Silver to Diamond."""
    print("  Creating Elo Slope Chart...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to Silver and Diamond only
    df_filtered = df[df['champion'].isin(champions) & df['tier'].isin(['Silver', 'Diamond'])]
    pivot = df_filtered.pivot(index='champion', columns='tier', values='winrate')

    # Sort by change magnitude
    pivot['change'] = pivot['Diamond'] - pivot['Silver']
    pivot = pivot.sort_values('change', ascending=False)

    # Draw slopes
    for i, (champ, row) in enumerate(pivot.iterrows()):
        silver_wr = row['Silver']
        diamond_wr = row['Diamond']
        change = row['change']

        # Color based on direction
        color = COLORS['positive'] if change > 0 else COLORS['negative']

        # Draw line
        ax.plot([0, 1], [silver_wr, diamond_wr], color=color, linewidth=2.5, alpha=0.8)

        # Add points
        ax.scatter([0], [silver_wr], color=ELO_COLORS['Silver'], s=100, zorder=5)
        ax.scatter([1], [diamond_wr], color=ELO_COLORS['Diamond'], s=100, zorder=5)

        # Add champion labels
        ax.annotate(champ, xy=(-0.05, silver_wr), ha='right', va='center', fontsize=10)
        ax.annotate(f'{change:+.1%}', xy=(1.05, diamond_wr), ha='left', va='center',
                   fontsize=10, color=color, fontweight='bold')

    # Add baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.42, 0.58)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Silver', 'Diamond'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Winrate', fontsize=12)
    ax.set_title('Champion-Winrate: Silver vs. Diamond (Slope Chart)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['positive'], label='Steigt mit Elo'),
        Patch(facecolor=COLORS['negative'], label='Sinkt mit Elo')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_elo_slope.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_elo_slope.pdf')
    plt.close()
    print("    Saved: champion_elo_slope.png/pdf")


def fig_elo_diverging_bar(df, champions):
    """Diverging bar chart centered at 50%."""
    print("  Creating Elo Diverging Bar Chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Silver winrates
    silver_data = df[(df['champion'].isin(champions)) & (df['tier'] == 'Silver')].copy()
    silver_data['deviation'] = silver_data['winrate'] - 0.5
    silver_data = silver_data.sort_values('deviation')

    colors_silver = [COLORS['positive'] if d > 0 else COLORS['negative'] for d in silver_data['deviation']]

    axes[0].barh(silver_data['champion'], silver_data['deviation'] * 100, color=colors_silver)
    axes[0].axvline(x=0, color='black', linewidth=1)
    axes[0].set_xlabel('Abweichung von 50% (Prozentpunkte)', fontsize=11)
    axes[0].set_title('Silver', fontsize=13, fontweight='bold')
    axes[0].set_xlim(-8, 8)
    axes[0].grid(True, axis='x', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(silver_data.iterrows()):
        val = row['deviation'] * 100
        offset = 0.3 if val >= 0 else -0.3
        axes[0].annotate(f'{row["winrate"]:.1%}', xy=(val + offset, i),
                        va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # Diamond winrates
    diamond_data = df[(df['champion'].isin(champions)) & (df['tier'] == 'Diamond')].copy()
    diamond_data['deviation'] = diamond_data['winrate'] - 0.5
    diamond_data = diamond_data.sort_values('deviation')

    colors_diamond = [COLORS['positive'] if d > 0 else COLORS['negative'] for d in diamond_data['deviation']]

    axes[1].barh(diamond_data['champion'], diamond_data['deviation'] * 100, color=colors_diamond)
    axes[1].axvline(x=0, color='black', linewidth=1)
    axes[1].set_xlabel('Abweichung von 50% (Prozentpunkte)', fontsize=11)
    axes[1].set_title('Diamond', fontsize=13, fontweight='bold')
    axes[1].set_xlim(-8, 8)
    axes[1].grid(True, axis='x', alpha=0.3)

    # Add value labels
    for i, (_, row) in enumerate(diamond_data.iterrows()):
        val = row['deviation'] * 100
        offset = 0.3 if val >= 0 else -0.3
        axes[1].annotate(f'{row["winrate"]:.1%}', xy=(val + offset, i),
                        va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.suptitle('Champion-Winrate: Abweichung von 50% (Diverging Bar)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_elo_diverging.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'champion_elo_diverging.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: champion_elo_diverging.png/pdf")


def fig_elo_dot_range(df, champions):
    """Dot plot with range showing min/max/mean."""
    print("  Creating Elo Dot Plot with Range...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate stats per champion
    stats = df[df['champion'].isin(champions)].groupby('champion')['winrate'].agg(['min', 'max', 'mean'])
    stats['range'] = stats['max'] - stats['min']
    stats = stats.sort_values('range', ascending=True)

    y_positions = np.arange(len(stats))

    # Draw ranges
    for i, (champ, row) in enumerate(stats.iterrows()):
        # Range line
        ax.hlines(y=i, xmin=row['min'], xmax=row['max'], color=COLORS['primary'], linewidth=3, alpha=0.6)

        # Min/Max points
        ax.scatter([row['min']], [i], color=COLORS['negative'], s=120, zorder=5, marker='o')
        ax.scatter([row['max']], [i], color=COLORS['positive'], s=120, zorder=5, marker='o')

        # Mean point
        ax.scatter([row['mean']], [i], color=COLORS['primary'], s=80, zorder=6, marker='D')

        # Range annotation
        ax.annotate(f'{row["range"]*100:.1f}%', xy=(row['max'] + 0.005, i),
                   va='center', ha='left', fontsize=9, fontweight='bold')

    # Add baseline
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50% Baseline')

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats.index)
    ax.set_xlabel('Winrate', fontsize=12)
    ax.set_ylabel('Champion', fontsize=12)
    ax.set_title('Champion-Winrate Varianz über Elo-Tiers (Dot Plot)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.42, 0.62)
    ax.grid(True, axis='x', alpha=0.3)

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['negative'],
                  markersize=10, label='Minimum (niedrigste Elo)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['positive'],
                  markersize=10, label='Maximum (höchste Elo)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['primary'],
                  markersize=8, label='Durchschnitt'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_elo_dotrange.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_elo_dotrange.pdf')
    plt.close()
    print("    Saved: champion_elo_dotrange.png/pdf")


# =============================================================================
# CHAMPION WINRATE BY REGION - 4 CHART TYPES
# =============================================================================

def fig_region_heatmap(df, champions):
    """Heatmap showing champion winrates across regions."""
    print("  Creating Region Heatmap...")

    # Pivot data
    pivot = df[df['champion'].isin(champions)].pivot(
        index='champion', columns='region', values='winrate'
    )
    pivot = pivot.reindex(champions)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create diverging colormap centered at 0.5
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap=cmap, center=0.5,
                vmin=0.42, vmax=0.58, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Winrate'})

    ax.set_title('Champion-Winrate nach Region (Heatmap)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Region', fontsize=12)
    ax.set_ylabel('Champion', fontsize=12)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_region_heatmap.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_region_heatmap.pdf')
    plt.close()
    print("    Saved: champion_region_heatmap.png/pdf")


def fig_region_slope_chart(df, champions):
    """Slope chart comparing two key regions (BR1 vs KR)."""
    print("  Creating Region Slope Chart...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to BR1 and KR (contrasting regions)
    df_filtered = df[df['champion'].isin(champions) & df['region'].isin(['BR1', 'KR'])]
    pivot = df_filtered.pivot(index='champion', columns='region', values='winrate')

    # Remove champions without both regions
    pivot = pivot.dropna()

    # Sort by change magnitude
    pivot['change'] = pivot['KR'] - pivot['BR1']
    pivot = pivot.sort_values('change', ascending=False)

    # Draw slopes
    for i, (champ, row) in enumerate(pivot.iterrows()):
        br1_wr = row['BR1']
        kr_wr = row['KR']
        change = row['change']

        color = COLORS['positive'] if change > 0 else COLORS['negative']

        ax.plot([0, 1], [br1_wr, kr_wr], color=color, linewidth=2.5, alpha=0.8)
        ax.scatter([0], [br1_wr], color=REGION_COLORS['BR1'], s=100, zorder=5)
        ax.scatter([1], [kr_wr], color=REGION_COLORS['KR'], s=100, zorder=5)

        ax.annotate(champ, xy=(-0.05, br1_wr), ha='right', va='center', fontsize=10)
        ax.annotate(f'{change:+.1%}', xy=(1.05, kr_wr), ha='left', va='center',
                   fontsize=10, color=color, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.40, 0.60)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BR1 (Brasilien)', 'KR (Korea)'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Winrate', fontsize=12)
    ax.set_title('Champion-Winrate: BR1 vs. KR (Slope Chart)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLORS['positive'], label='Besser in KR'),
        Patch(facecolor=COLORS['negative'], label='Besser in BR1')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_region_slope.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_region_slope.pdf')
    plt.close()
    print("    Saved: champion_region_slope.png/pdf")


def fig_region_diverging_bar(df, champions):
    """Diverging bar chart for regions centered at 50%."""
    print("  Creating Region Diverging Bar Chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # BR1 winrates
    br1_data = df[(df['champion'].isin(champions)) & (df['region'] == 'BR1')].copy()
    br1_data['deviation'] = br1_data['winrate'] - 0.5
    br1_data = br1_data.sort_values('deviation')

    colors_br1 = [COLORS['positive'] if d > 0 else COLORS['negative'] for d in br1_data['deviation']]

    axes[0].barh(br1_data['champion'], br1_data['deviation'] * 100, color=colors_br1)
    axes[0].axvline(x=0, color='black', linewidth=1)
    axes[0].set_xlabel('Abweichung von 50% (Prozentpunkte)', fontsize=11)
    axes[0].set_title('BR1 (Brasilien)', fontsize=13, fontweight='bold')
    axes[0].set_xlim(-10, 10)
    axes[0].grid(True, axis='x', alpha=0.3)

    for i, (_, row) in enumerate(br1_data.iterrows()):
        val = row['deviation'] * 100
        offset = 0.3 if val >= 0 else -0.3
        axes[0].annotate(f'{row["winrate"]:.1%}', xy=(val + offset, i),
                        va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    # KR winrates
    kr_data = df[(df['champion'].isin(champions)) & (df['region'] == 'KR')].copy()
    kr_data['deviation'] = kr_data['winrate'] - 0.5
    kr_data = kr_data.sort_values('deviation')

    colors_kr = [COLORS['positive'] if d > 0 else COLORS['negative'] for d in kr_data['deviation']]

    axes[1].barh(kr_data['champion'], kr_data['deviation'] * 100, color=colors_kr)
    axes[1].axvline(x=0, color='black', linewidth=1)
    axes[1].set_xlabel('Abweichung von 50% (Prozentpunkte)', fontsize=11)
    axes[1].set_title('KR (Korea)', fontsize=13, fontweight='bold')
    axes[1].set_xlim(-10, 10)
    axes[1].grid(True, axis='x', alpha=0.3)

    for i, (_, row) in enumerate(kr_data.iterrows()):
        val = row['deviation'] * 100
        offset = 0.3 if val >= 0 else -0.3
        axes[1].annotate(f'{row["winrate"]:.1%}', xy=(val + offset, i),
                        va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    plt.suptitle('Champion-Winrate: Abweichung von 50% nach Region (Diverging Bar)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_region_diverging.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'champion_region_diverging.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: champion_region_diverging.png/pdf")


def fig_region_dot_range(df, champions):
    """Dot plot with range showing regional variation."""
    print("  Creating Region Dot Plot with Range...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate stats per champion
    stats = df[df['champion'].isin(champions)].groupby('champion')['winrate'].agg(['min', 'max', 'mean'])
    stats['range'] = stats['max'] - stats['min']
    stats = stats.sort_values('range', ascending=True)

    y_positions = np.arange(len(stats))

    for i, (champ, row) in enumerate(stats.iterrows()):
        ax.hlines(y=i, xmin=row['min'], xmax=row['max'], color=COLORS['secondary'], linewidth=3, alpha=0.6)
        ax.scatter([row['min']], [i], color=COLORS['negative'], s=120, zorder=5, marker='o')
        ax.scatter([row['max']], [i], color=COLORS['positive'], s=120, zorder=5, marker='o')
        ax.scatter([row['mean']], [i], color=COLORS['secondary'], s=80, zorder=6, marker='D')
        ax.annotate(f'{row["range"]*100:.1f}%', xy=(row['max'] + 0.005, i),
                   va='center', ha='left', fontsize=9, fontweight='bold')

    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(stats.index)
    ax.set_xlabel('Winrate', fontsize=12)
    ax.set_ylabel('Champion', fontsize=12)
    ax.set_title('Champion-Winrate Varianz über Regionen (Dot Plot)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.40, 0.62)
    ax.grid(True, axis='x', alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['negative'],
                  markersize=10, label='Minimum (schlechteste Region)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['positive'],
                  markersize=10, label='Maximum (beste Region)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['secondary'],
                  markersize=8, label='Durchschnitt'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_region_dotrange.png', dpi=150)
    plt.savefig(OUTPUT_DIR / 'champion_region_dotrange.pdf')
    plt.close()
    print("    Saved: champion_region_dotrange.png/pdf")


def fig_elo_stacked_diverging_clean(df_raw):
    """Clean diverging bar chart with verified data - excludes Diamond (insufficient data)."""
    print("  Creating Clean Elo Diverging Chart...")

    # Use only Silver-Emerald (Diamond has <1% of data)
    tiers = ['Silver', 'Gold', 'Platinum', 'Emerald']
    tier_colors = {'Silver': '#95a5a6', 'Gold': '#f1c40f', 'Platinum': '#1abc9c', 'Emerald': '#9b59b6'}
    rank_map = {'Silver': 3, 'Gold': 4, 'Platinum': 5, 'Emerald': 6}

    # Find champions with highest variance AND sufficient data (min 200 games per tier)
    top_champs = df_raw['championName'].value_counts().head(30).index.tolist()
    results = []

    for champ in top_champs:
        champ_data = df_raw[df_raw['championName'] == champ]
        champ_row = {'champion': champ, 'tiers': {}}
        valid_tiers = 0

        for tier_name, rank_num in rank_map.items():
            tier_data = champ_data[champ_data['hero_rank_numeric'] == rank_num]
            if len(tier_data) >= 200:  # Minimum 200 games for reliability
                wr = tier_data['hero_win'].mean()
                champ_row['tiers'][tier_name] = {'winrate': wr, 'games': len(tier_data)}
                valid_tiers += 1

        if valid_tiers == 4:  # Need all 4 tiers
            wrs = [champ_row['tiers'][t]['winrate'] for t in tiers]
            champ_row['range'] = max(wrs) - min(wrs)
            results.append(champ_row)

    # Sort by variance and take top 5
    results_sorted = sorted(results, key=lambda x: x['range'], reverse=True)[:5]

    if not results_sorted:
        print("    No champions with sufficient data")
        return

    # Calculate max deviation for x-axis
    all_devs = []
    for r in results_sorted:
        for tier in tiers:
            if tier in r['tiers']:
                all_devs.append(abs((r['tiers'][tier]['winrate'] - 0.5) * 100))
    max_dev = max(all_devs) if all_devs else 10
    x_limit = max(12, int(max_dev + 5))  # Extra space for labels

    # Create chart - WIDER for labels
    n_champs = len(results_sorted)
    fig, axes = plt.subplots(n_champs, 1, figsize=(14, 2.0 * n_champs))
    if n_champs == 1:
        axes = [axes]

    for idx, champ_data in enumerate(results_sorted):
        ax = axes[idx]
        champ = champ_data['champion']

        deviations = []
        colors = []
        labels = []

        for tier in tiers:
            if tier in champ_data['tiers']:
                wr = champ_data['tiers'][tier]['winrate']
                games = champ_data['tiers'][tier]['games']
                deviations.append((wr - 0.5) * 100)
                colors.append(tier_colors[tier])
                labels.append((wr, games))
            else:
                deviations.append(0)
                colors.append('#cccccc')
                labels.append((None, 0))

        # Create horizontal bars
        bars = ax.barh(tiers, deviations, color=colors, edgecolor='black', linewidth=0.5, height=0.6)

        # Add value labels - ALWAYS next to bar end (outside)
        for bar, (wr, games), dev in zip(bars, labels, deviations):
            if wr is not None:
                label_text = f'{wr*100:.1f}% (n={games:,})'
                # Always position right next to the bar
                if dev >= 0:
                    x_pos = dev + 0.5
                    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                           va='center', ha='left', fontsize=9)
                else:
                    x_pos = dev - 0.5
                    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                           va='center', ha='right', fontsize=9)

        # Styling
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(f'{champ} (Δ {champ_data["range"]*100:.1f}%)', fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='y', labelsize=10)

        if idx == n_champs - 1:
            ax.set_xlabel('Abweichung von 50% Winrate (Prozentpunkte)', fontsize=10)

        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle('Champion-Winrate nach Elo-Tier (Silver–Emerald, min. 200 Spiele)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_elo.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_elo.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: champion_winrate_by_elo.png/pdf")


def fig_elo_stacked_diverging(df, champions):
    """Diverging bar chart: Champion on top, Elo tiers below as horizontal bars."""
    print("  Creating Stacked Elo Diverging Chart (OLD)...")
    # Keep old function but don't use it
    pass


def fig_region_stacked_diverging_clean(df_raw):
    """Clean diverging bar chart for regions with verified data."""
    print("  Creating Clean Region Diverging Chart...")

    # Get top regions by sample size
    top_regions = df_raw['platform'].value_counts().head(5).index.tolist()
    region_colors = {'br1': '#009739', 'euw1': '#003399', 'na1': '#B31942',
                     'kr': '#003478', 'oc1': '#00843D', 'jp1': '#BC002D',
                     'eun1': '#0052B4', 'tr1': '#E30A17'}

    # Find champions with highest regional variance AND sufficient data
    top_champs = df_raw['championName'].value_counts().head(30).index.tolist()
    results = []

    for champ in top_champs:
        champ_data = df_raw[df_raw['championName'] == champ]
        champ_row = {'champion': champ, 'regions': {}}
        valid_regions = 0

        for region in top_regions:
            region_data = champ_data[champ_data['platform'] == region]
            if len(region_data) >= 150:  # Minimum 150 games for reliability
                wr = region_data['hero_win'].mean()
                champ_row['regions'][region] = {'winrate': wr, 'games': len(region_data)}
                valid_regions += 1

        if valid_regions >= 4:  # Need at least 4 regions
            wrs = [champ_row['regions'][r]['winrate'] for r in champ_row['regions']]
            champ_row['range'] = max(wrs) - min(wrs)
            results.append(champ_row)

    # Sort by variance and take top 5
    results_sorted = sorted(results, key=lambda x: x['range'], reverse=True)[:5]

    if not results_sorted:
        print("    No champions with sufficient data")
        return

    # Calculate max deviation for x-axis
    all_devs = []
    for r in results_sorted:
        for region in r['regions']:
            all_devs.append(abs((r['regions'][region]['winrate'] - 0.5) * 100))
    max_dev = max(all_devs) if all_devs else 10
    x_limit = max(12, int(max_dev + 5))  # Extra space for labels

    # Create chart - WIDER for labels
    n_champs = len(results_sorted)
    fig, axes = plt.subplots(n_champs, 1, figsize=(14, 2.0 * n_champs))
    if n_champs == 1:
        axes = [axes]

    for idx, champ_data in enumerate(results_sorted):
        ax = axes[idx]
        champ = champ_data['champion']

        # Sort regions by winrate for this champion
        sorted_regions = sorted(champ_data['regions'].keys(),
                               key=lambda r: champ_data['regions'][r]['winrate'])

        deviations = []
        colors = []
        labels = []
        region_labels = []

        for region in sorted_regions:
            wr = champ_data['regions'][region]['winrate']
            games = champ_data['regions'][region]['games']
            deviations.append((wr - 0.5) * 100)
            colors.append(region_colors.get(region, '#666666'))
            labels.append((wr, games))
            region_labels.append(region.upper())

        # Create horizontal bars
        bars = ax.barh(region_labels, deviations, color=colors, edgecolor='black', linewidth=0.5, height=0.6)

        # Add value labels - ALWAYS next to bar end (outside)
        for bar, (wr, games), dev in zip(bars, labels, deviations):
            label_text = f'{wr*100:.1f}% (n={games:,})'
            # Always position right next to the bar
            if dev >= 0:
                x_pos = dev + 0.5
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha='left', fontsize=9)
            else:
                x_pos = dev - 0.5
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha='right', fontsize=9)

        # Styling
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(f'{champ} (Δ {champ_data["range"]*100:.1f}%)', fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='y', labelsize=10)

        if idx == n_champs - 1:
            ax.set_xlabel('Abweichung von 50% Winrate (Prozentpunkte)', fontsize=10)

        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle('Champion-Winrate nach Region (min. 150 Spiele pro Region)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: champion_winrate_by_region.png/pdf")


def fig_elo_region_combined(df_full, elo_tier='Platinum'):
    """Diverging bar chart: Champion winrate by region within a specific Elo tier."""
    print(f"  Creating Combined Elo-Region Chart for {elo_tier}...")

    # Map elo tier to numeric
    rank_map = {'Silver': 3, 'Gold': 4, 'Platinum': 5, 'Emerald': 6, 'Diamond': 7}
    rank_num = rank_map.get(elo_tier, 5)

    # Filter to specific elo
    df_elo = df_full[df_full['hero_rank_numeric'] == rank_num].copy()

    # Get top regions by sample size
    top_regions = df_elo['platform'].value_counts().head(6).index.tolist()

    # Get champions with highest regional variance within this elo
    results = []
    top_champs = df_elo['championName'].value_counts().head(30).index.tolist()

    for champ in top_champs:
        champ_data = df_elo[df_elo['championName'] == champ]
        region_wrs = []
        for region in top_regions:
            region_data = champ_data[champ_data['platform'] == region]
            if len(region_data) >= 20:
                wr = region_data['hero_win'].mean()
                region_wrs.append({'champion': champ, 'region': region.upper(), 'winrate': wr, 'games': len(region_data)})

        if len(region_wrs) >= 4:  # Need at least 4 regions for meaningful comparison
            results.extend(region_wrs)

    if not results:
        print("    No data available for this elo tier")
        return

    df_combined = pd.DataFrame(results)

    # Find champions with highest variance
    variance = df_combined.groupby('champion')['winrate'].agg(['var', 'min', 'max'])
    variance['range'] = variance['max'] - variance['min']
    high_var_champs = variance.sort_values('range', ascending=False).head(6).index.tolist()

    # Calculate max deviation for dynamic x-axis
    all_devs = [abs((r['winrate'] - 0.5) * 100) for r in results if r['champion'] in high_var_champs]
    max_dev = max(all_devs) if all_devs else 10
    x_limit = max(12, int(max_dev + 5))  # Extra space for labels

    # Create chart - WIDER for labels
    n_champs = len(high_var_champs)
    fig, axes = plt.subplots(n_champs, 1, figsize=(14, 2.4 * n_champs))
    if n_champs == 1:
        axes = [axes]

    region_colors = {
        'BR1': '#009739', 'EUW1': '#003399', 'NA1': '#B31942',
        'KR': '#003478', 'OC1': '#00843D', 'JP1': '#BC002D',
        'EUN1': '#0052B4', 'TR1': '#E30A17', 'LA1': '#006847', 'LA2': '#002B7F'
    }

    for idx, champ in enumerate(high_var_champs):
        ax = axes[idx]
        champ_data = df_combined[df_combined['champion'] == champ].sort_values('winrate')

        regions = champ_data['region'].tolist()
        values = champ_data['winrate'].tolist()
        games_list = champ_data['games'].tolist()
        deviations = [(v - 0.5) * 100 for v in values]
        colors = [region_colors.get(r, '#666666') for r in regions]

        # Create horizontal bars
        bars = ax.barh(regions, deviations, color=colors, edgecolor='black', linewidth=0.5, height=0.65)

        # Add value labels - ALWAYS next to bar end (outside)
        for bar, val, dev, games in zip(bars, values, deviations, games_list):
            label_text = f'{val*100:.1f}% (n={games:,})'
            # Always position right next to the bar
            if dev >= 0:
                x_pos = dev + 0.5
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha='left', fontsize=9)
            else:
                x_pos = dev - 0.5
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha='right', fontsize=9)

        # Styling
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlim(-x_limit, x_limit)
        champ_range = variance.loc[champ, 'range'] * 100
        ax.set_title(f'{champ} (Δ {champ_range:.1f}%)', fontsize=12, fontweight='bold', loc='left')
        ax.tick_params(axis='y', labelsize=10)

        if idx == n_champs - 1:
            ax.set_xlabel('Abweichung von 50% Winrate (Prozentpunkte)', fontsize=10)

        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle(f'Champion-Winrate nach Region ({elo_tier}, min. 20 Spiele pro Region)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'champion_winrate_by_region_{elo_tier.lower()}.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f'champion_winrate_by_region_{elo_tier.lower()}.pdf', bbox_inches='tight')
    plt.close()
    print(f"    Saved: champion_winrate_by_region_{elo_tier.lower()}.png/pdf")


def fig_region_stacked_diverging(df, champions):
    """Diverging bar chart: Champion on top, Regions below as horizontal bars."""
    print("  Creating Stacked Region Diverging Chart...")

    n_champs = min(len(champions), 6)
    selected_champs = champions[:n_champs]

    fig, axes = plt.subplots(n_champs, 1, figsize=(12, 2.4 * n_champs))
    if n_champs == 1:
        axes = [axes]

    # Get available regions sorted by sample size
    regions = df['region'].value_counts().head(5).index.tolist()
    region_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Calculate max deviation for dynamic x-axis
    all_deviations = []
    for champ in selected_champs:
        champ_data = df[df['champion'] == champ]
        for region in regions:
            region_data = champ_data[champ_data['region'] == region]
            if len(region_data) > 0:
                wr = region_data['winrate'].values[0]
                all_deviations.append(abs((wr - 0.5) * 100))

    max_dev = max(all_deviations) if all_deviations else 10
    x_limit = max(12, int(max_dev + 4))

    for idx, champ in enumerate(selected_champs):
        ax = axes[idx]
        champ_data = df[df['champion'] == champ]

        values = []
        deviations = []
        colors = []
        available_regions = []
        games_list = []

        for region, color in zip(regions, region_colors):
            region_data = champ_data[champ_data['region'] == region]
            if len(region_data) > 0:
                wr = region_data['winrate'].values[0]
                games = region_data['games'].values[0] if 'games' in region_data.columns else 0
                values.append(wr)
                deviations.append((wr - 0.5) * 100)
                colors.append(color)
                available_regions.append(region)
                games_list.append(games)

        if not available_regions:
            continue

        # Create horizontal bars
        bars = ax.barh(available_regions, deviations, color=colors, edgecolor='black', linewidth=0.5, height=0.65)

        # Add value labels - uniform positioning
        for bar, val, dev, games in zip(bars, values, deviations, games_list):
            label_text = f'{val:.1%}'
            if games > 0:
                label_text += f' (n={games:,})'

            if abs(dev) > 4:
                # Inside bar
                x_pos = dev * 0.5
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha='center', fontsize=9, fontweight='bold',
                       color='white' if abs(dev) > 6 else 'black')
            else:
                # Outside bar
                x_pos = dev + (1.5 if dev >= 0 else -1.5)
                ha = 'left' if dev >= 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                       va='center', ha=ha, fontsize=9, fontweight='bold')

        # Styling
        ax.axvline(x=0, color='black', linewidth=1.5)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_title(champ, fontsize=13, fontweight='bold', loc='left', pad=5)
        ax.tick_params(axis='y', labelsize=10)

        if idx == n_champs - 1:
            ax.set_xlabel('Abweichung von 50% (Prozentpunkte)', fontsize=11)

        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

    plt.suptitle('Champion-Winrate nach Region', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'champion_winrate_by_region.pdf', bbox_inches='tight')
    plt.close()
    print("    Saved: champion_winrate_by_region.png/pdf")


def main():
    print("=" * 70)
    print("Champion Variability Charts - Multiple Visualization Types")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Get champion data by elo
    print("\n--- Champion Winrate by Elo ---")
    elo_df = get_champion_elo_data(df)
    elo_champions = get_high_variance_champions(elo_df, 'tier', n=10)
    print(f"  Selected {len(elo_champions)} champions with highest elo variance")

    # Generate chart types for Elo
    fig_elo_heatmap(elo_df, elo_champions)
    fig_elo_slope_chart(elo_df, elo_champions)
    fig_elo_diverging_bar(elo_df, elo_champions)
    fig_elo_dot_range(elo_df, elo_champions)
    fig_elo_stacked_diverging_clean(df)  # Use raw data for verified numbers

    # Get champion data by region
    print("\n--- Champion Winrate by Region ---")
    region_df = get_champion_region_data(df)
    region_champions = get_high_variance_champions(region_df, 'region', n=10)
    print(f"  Selected {len(region_champions)} champions with highest regional variance")

    # Generate chart types for Region
    fig_region_heatmap(region_df, region_champions)
    fig_region_slope_chart(region_df, region_champions)
    fig_region_diverging_bar(region_df, region_champions)
    fig_region_dot_range(region_df, region_champions)
    fig_region_stacked_diverging_clean(df)  # Use raw data for verified numbers

    # Generate combined Elo-Region chart (uses raw data)
    print("\n--- Combined Elo-Region Analysis ---")
    fig_elo_region_combined(df, elo_tier='Platinum')
    fig_elo_region_combined(df, elo_tier='Emerald')

    print("\n" + "=" * 70)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 70)

    # Summary
    print("\nGenerated files:")
    print("  ELO Charts:")
    print("    - champion_elo_heatmap.png/pdf")
    print("    - champion_elo_slope.png/pdf")
    print("    - champion_elo_diverging.png/pdf")
    print("    - champion_elo_dotrange.png/pdf")
    print("  REGION Charts:")
    print("    - champion_region_heatmap.png/pdf")
    print("    - champion_region_slope.png/pdf")
    print("    - champion_region_diverging.png/pdf")
    print("    - champion_region_dotrange.png/pdf")


if __name__ == '__main__':
    main()
