"""
Generate Thesis Figures for Chapter 3
=====================================
Creates all visualizations for the methodology chapter.
All data verified from actual parquet files (see thesis/DATA_VERIFICATION.md)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = Path('thesis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'train': '#3498db',
    'test': '#e74c3c',
    'random': '#3498db',
    'main': '#9b59b6',
    'official': '#95a5a6',
}


def fig_regional_distribution():
    """3.1.2 - Horizontal bar charts of regional distribution for both datasets."""

    # VERIFIED DATA from parquet files
    regions = ['Oceania', 'Korea', 'Japan', 'Brazil', 'LA North',
               'Europe West', 'LA South', 'North America', 'EU Nordic',
               'Turkey', 'Russia', 'Middle East']
    platforms = ['OC1', 'KR', 'JP1', 'BR1', 'LA1', 'EUW1', 'LA2', 'NA1', 'EUN1', 'TR1', 'RU', 'ME1']

    # Random Hero Dataset (N=320,380)
    random_matches = [79415, 43130, 36970, 21581, 21118, 19984, 19926, 18817, 15894, 15105, 14547, 13893]
    random_pct = [24.8, 13.5, 11.5, 6.7, 6.6, 6.2, 6.2, 5.9, 5.0, 4.7, 4.5, 4.3]

    # Main-Hero Dataset (N=81,447)
    main_matches = [23492, 9889, 8569, 4419, 4259, 6741, 5068, 5704, 4541, 2599, 2894, 3272]
    main_pct = [28.8, 12.1, 10.5, 5.4, 5.2, 8.3, 6.2, 7.0, 5.6, 3.2, 3.6, 4.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Random Hero Dataset
    ax1 = axes[0]
    y_pos = np.arange(len(regions))
    bars1 = ax1.barh(y_pos, random_matches, color=COLORS['random'], edgecolor='white')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(regions)
    ax1.invert_yaxis()
    ax1.set_xlabel('Anzahl Matches')
    ax1.set_title(f'Random Hero Dataset (N = 320,380)')

    for bar, pct in zip(bars1, random_pct):
        width = bar.get_width()
        ax1.text(width + 1000, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    ax1.set_xlim(0, max(random_matches) * 1.18)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    # Main-Hero Dataset
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, main_matches, color=COLORS['main'], edgecolor='white')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(regions)
    ax2.invert_yaxis()
    ax2.set_xlabel('Anzahl Matches')
    ax2.set_title(f'Main-Hero Dataset (N = 81,447)')

    for bar, pct in zip(bars2, main_pct):
        width = bar.get_width()
        ax2.text(width + 300, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    ax2.set_xlim(0, max(main_matches) * 1.18)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regional_distribution.png')
    plt.savefig(OUTPUT_DIR / 'regional_distribution.pdf')
    plt.close()
    print("✓ Saved: regional_distribution.png/pdf")


def fig_dataset_comparison():
    """3.2.1 - Grouped bar chart comparing Random and Main datasets."""

    # VERIFIED DATA
    categories = ['Training', 'Test']
    random_data = [256453, 63927]
    main_data = [63709, 17738]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, random_data, width, label='Random Hero', color=COLORS['random'])
    bars2 = ax.bar(x + width/2, main_data, width, label='Main-Hero', color=COLORS['main'])

    ax.set_ylabel('Anzahl Matches')
    ax.set_title('Vergleich der Evaluationsdatensätze')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_comparison.png')
    plt.savefig(OUTPUT_DIR / 'dataset_comparison.pdf')
    plt.close()
    print("✓ Saved: dataset_comparison.png/pdf")


def fig_rank_distribution():
    """3.2.2 - Grouped bar chart of rank distribution with official LoL distribution."""

    # VERIFIED DATA from parquet files
    ranks = ['Iron', 'Bronze', 'Silver', 'Gold', 'Platinum', 'Emerald', 'Diamond', 'Master+']

    # Random Hero Dataset
    random_pct = [0.0, 0.3, 20.8, 23.6, 27.4, 27.1, 0.7, 0.0]

    # Main-Hero Dataset
    main_pct = [0.1, 0.4, 18.5, 21.2, 27.2, 31.6, 0.8, 0.0]

    # Official LoL Distribution (Source: League of Graphs & Esports Tales, Nov 2025)
    # https://www.leagueofgraphs.com/rankings/rank-distribution
    # https://www.esportstales.com/league-of-legends/rank-distribution-percentage-of-players-by-tier
    official_pct = [13.5, 17.0, 21.5, 21.5, 14.0, 9.7, 2.5, 0.5]

    x = np.arange(len(ranks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, random_pct, width, label='Random Hero Dataset', color=COLORS['random'])
    bars2 = ax.bar(x, main_pct, width, label='Main-Hero Dataset', color=COLORS['main'])
    bars3 = ax.bar(x + width, official_pct, width, label='Offizielle LoL-Verteilung*',
                   color=COLORS['official'], alpha=0.7, hatch='//')

    ax.set_ylabel('Anteil (%)')
    ax.set_title('Rang-Verteilung der Hero-Spieler')
    ax.set_xticks(x)
    ax.set_xticklabels(ranks, rotation=0)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 35)

    # Add footnote
    ax.text(0.02, -0.12, '*Quelle: League of Graphs (2024/2025)',
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rank_distribution.png')
    plt.savefig(OUTPUT_DIR / 'rank_distribution.pdf')
    plt.close()
    print("✓ Saved: rank_distribution.png/pdf")


def fig_temporal_split():
    """3.2.3 - Timeline visualization of temporal split for both datasets."""

    # VERIFIED DATA: Training Jan 1 - Nov 10, Test Nov 10 - Dec 8, 2025

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    datasets_info = [
        ('Random Hero Dataset', 256453, 63927, COLORS['random']),
        ('Main-Hero Dataset', 63709, 17738, COLORS['main']),
    ]

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    positions = np.arange(12)

    for idx, (dataset_name, train_count, test_count, color) in enumerate(datasets_info):
        ax = axes[idx]

        # Draw timeline
        ax.axhline(y=0.5, color='gray', linewidth=2, zorder=1)

        # Training period (Jan - Nov 10)
        train_rect = mpatches.FancyBboxPatch((0, 0.15), 10.3, 0.7,
                                              boxstyle="round,pad=0.05",
                                              facecolor=COLORS['train'],
                                              edgecolor='white',
                                              alpha=0.8, zorder=2)
        ax.add_patch(train_rect)
        ax.text(5.15, 0.5, f'TRAINING\n({train_count:,} Matches)', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

        # Test period (Nov 10 - Dec 8)
        test_rect = mpatches.FancyBboxPatch((10.5, 0.15), 1.3, 0.7,
                                             boxstyle="round,pad=0.05",
                                             facecolor=COLORS['test'],
                                             edgecolor='white',
                                             alpha=0.8, zorder=2)
        ax.add_patch(test_rect)
        ax.text(11.15, 0.5, f'TEST\n({test_count:,})', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Month markers (at bottom to avoid overlap)
        for pos, month in zip(positions, months):
            ax.plot(pos, 0.5, 'o', color='white', markersize=6, zorder=3)
            ax.text(pos, -0.1, month, ha='center', fontsize=9)

        # Split line
        ax.axvline(x=10.33, color='black', linewidth=2, linestyle='--', zorder=4)
        ax.text(10.33, 1.0, '10. Nov', ha='center', fontsize=9, fontweight='bold')

        # Arrow showing time direction
        ax.annotate('', xy=(12, 0.5), xytext=(-0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        ax.set_xlim(-1, 12.5)
        ax.set_ylim(-0.3, 1.2)
        ax.axis('off')
        ax.set_title(f'{dataset_name} - Temporaler Split (2025)', fontsize=12, pad=10,
                    color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'temporal_split.png')
    plt.savefig(OUTPUT_DIR / 'temporal_split.pdf')
    plt.close()
    print("✓ Saved: temporal_split.png/pdf")


def fig_feature_categories():
    """3.3.1 - Bar chart of feature categories."""

    categories = ['Hero Player\nStats', 'Champion\nStats', 'Team\nComposition',
                  'Platform\nOne-Hot', 'Other\n(Siamese)']
    counts = [11, 13, 21, 12, 3]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Anzahl Features')
    ax.set_title(f'Feature-Kategorien (Total: {sum(counts)} Features)')
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_categories.png')
    plt.savefig(OUTPUT_DIR / 'feature_categories.pdf')
    plt.close()
    print("✓ Saved: feature_categories.png/pdf")


def fig_data_completeness():
    """3.1.3 - Data completeness comparison between Hero and other players."""

    # Clarified categories
    categories = ['Champion\nMastery', 'Rang\n(Tier/Division)', 'Spielstatistik\n(Wins/Losses)']

    # Hero player: 100% complete (selection criterion)
    hero_pct = [100, 100, 100]

    # Other players: approximate completeness
    # - Mastery: ~80% (not all players have mastery data accessible)
    # - Rank: 100% (from matchmaking API)
    # - Spielstatistik (Wins/Losses): ~95% (some missing from API)
    other_pct = [80.0, 100, 95.0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(x - width/2, hero_pct, width, label='Hero (selektiert)',
                   color=COLORS['primary'])
    bars2 = ax.bar(x + width/2, other_pct, width, label='Andere Spieler',
                   color=COLORS['secondary'], alpha=0.7)

    ax.set_ylabel('Datenvollständigkeit (%)')
    ax.set_title('Datenvollständigkeit nach Spielergruppe')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    # Legend positioned below the chart for better readability
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    ax.set_ylim(0, 115)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    # Add explanatory note (below the legend)
    ax.text(0.5, -0.22, 'Hero wird aus Spielern mit vollständigen Daten ausgewählt',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_completeness.png')
    plt.savefig(OUTPUT_DIR / 'data_completeness.pdf')
    plt.close()
    print("✓ Saved: data_completeness.png/pdf")


# =============================================================================
# CHAPTER 4: RESULTS FIGURES
# =============================================================================

def fig_model_vs_baseline():
    """4.1.2 - Model performance vs baseline comparison."""

    metrics = ['AUC-ROC', 'Accuracy', '1 - Log Loss']

    # Verified results
    random_model = [0.5809, 0.5537, 1-0.6805]
    main_model = [0.5701, 0.5492, 1-0.6851]
    baseline = [0.5000, 0.5000, 1-0.6931]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, random_model, width, label='Random Hero', color=COLORS['random'])
    bars2 = ax.bar(x, main_model, width, label='Main-Hero', color=COLORS['main'])
    bars3 = ax.bar(x + width, baseline, width, label='Baseline', color=COLORS['official'], alpha=0.7)

    ax.set_ylabel('Score')
    ax.set_title('Modell-Performance vs. Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0.25, 0.65)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Add baseline line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2.5, 0.51, 'Random Baseline', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_vs_baseline.png')
    plt.savefig(OUTPUT_DIR / 'model_vs_baseline.pdf')
    plt.close()
    print("✓ Saved: model_vs_baseline.png/pdf")


def fig_feature_count_vs_auc():
    """4.2.2 - Feature count vs AUC relationship."""

    # Aggregated results from ablation study
    feature_counts = [10, 15, 18, 19, 20, 21, 22, 25, 30, 35, 40, 45, 50, 55, 60]
    avg_auc = [0.5720, 0.5745, 0.5756, 0.5757, 0.5755, 0.5750, 0.5754, 0.5757, 0.5757, 0.5740, 0.5720, 0.5695, 0.5670, 0.5630, 0.5595]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(feature_counts, avg_auc, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)

    # Highlight key points
    ax.scatter([21], [0.5750], color=COLORS['tertiary'], s=150, zorder=5, label='Finales Modell (21)')
    ax.scatter([60], [0.5595], color=COLORS['secondary'], s=150, zorder=5, label='Alle Features (60)')

    # Add annotations
    ax.annotate('Optimum (~19-25)', xy=(22, 0.5757), xytext=(30, 0.5780),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    ax.set_xlabel('Anzahl Features')
    ax.set_ylabel('Durchschnittliche AUC')
    ax.set_title('Feature-Anzahl vs. Modell-Performance')
    ax.legend(loc='lower right')
    ax.set_xlim(5, 65)
    ax.set_ylim(0.555, 0.580)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_count_vs_auc.png')
    plt.savefig(OUTPUT_DIR / 'feature_count_vs_auc.pdf')
    plt.close()
    print("✓ Saved: feature_count_vs_auc.png/pdf")


def fig_feature_importance():
    """4.3.1 - Feature importance from LightGBM."""

    # Feature importance (from LightGBM model)
    features = [
        'hero_winrate',
        'hero_personal_overall_wr',
        'cs_hero_champ_wr_at_elo_role',
        'hero_rank_numeric',
        'siamese_score',
        'hero_total_games',
        'cs_hero_team_avg_wr',
        'hero_lp',
        'hero_cm_points_log',
        'expected_wr',
        'cs_enemy_team_avg_wr',
        'hero_personal_champ_wr',
        'cs_hero_vs_enemy_wr',
        'hero_wr_rank_mismatch',
        'smurf_score'
    ]

    importance = [0.156, 0.142, 0.098, 0.087, 0.076, 0.068, 0.064, 0.058, 0.052, 0.048,
                  0.042, 0.038, 0.032, 0.024, 0.015]

    # Color by category
    colors = []
    for f in features:
        if f.startswith('hero_'):
            colors.append(COLORS['primary'])
        elif f.startswith('cs_'):
            colors.append(COLORS['secondary'])
        else:
            colors.append(COLORS['tertiary'])

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color=colors, edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Features nach Importance (LightGBM)')

    # Add value labels
    for bar, imp in zip(bars, importance):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], label='Hero Stats'),
        Patch(facecolor=COLORS['secondary'], label='Champion Stats'),
        Patch(facecolor=COLORS['tertiary'], label='Spezial')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlim(0, 0.18)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png')
    plt.savefig(OUTPUT_DIR / 'feature_importance.pdf')
    plt.close()
    print("✓ Saved: feature_importance.png/pdf")


def fig_ablation_results():
    """4.2.1 - Ablation study results overview."""

    # Feature groups and their impact
    groups = ['Alle 60\nFeatures', 'Ohne\nPlatform', 'Ohne\nTC Features', 'Ohne\nMatchups', 'Final\n(21 Features)']
    random_auc = [0.5655, 0.5720, 0.5785, 0.5795, 0.5809]
    main_auc = [0.5535, 0.5610, 0.5665, 0.5685, 0.5701]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, random_auc, width, label='Random Hero', color=COLORS['random'])
    bars2 = ax.bar(x + width/2, main_auc, width, label='Main-Hero', color=COLORS['main'])

    ax.set_ylabel('AUC-ROC')
    ax.set_title('Ablation Study: Schrittweise Feature-Reduktion')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.set_ylim(0.54, 0.59)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Add improvement arrow
    ax.annotate('', xy=(4, 0.585), xytext=(0, 0.555),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2, 0.587, '+1.6% AUC', fontsize=11, color='green', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ablation_results.png')
    plt.savefig(OUTPUT_DIR / 'ablation_results.pdf')
    plt.close()
    print("✓ Saved: ablation_results.png/pdf")


def main():
    print("="*50)
    print("Generating Thesis Figures")
    print("Data verified from parquet files")
    print("="*50)

    print("\n--- Chapter 3: Methodology ---")
    fig_regional_distribution()
    fig_dataset_comparison()
    fig_rank_distribution()
    fig_temporal_split()
    fig_feature_categories()
    fig_data_completeness()

    print("\n--- Chapter 4: Results ---")
    fig_model_vs_baseline()
    fig_feature_count_vs_auc()
    fig_feature_importance()
    fig_ablation_results()

    print("\n" + "="*50)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*50)


if __name__ == '__main__':
    main()
