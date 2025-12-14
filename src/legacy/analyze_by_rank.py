import pandas as pd
import json
import os

# Lade die Match-Details
df = pd.read_parquet("./match_details_extracted.parquet")

# Lade die Original-Daten mit Rang-Info
df_euw = pd.read_parquet("./data_euw1/matches.parquet") if os.path.exists("./data_euw1/matches.parquet") else pd.DataFrame()
df_na = pd.read_parquet("./data_na1/matches.parquet") if os.path.exists("./data_na1/matches.parquet") else pd.DataFrame()
df_kr = pd.read_parquet("./data_kr/matches.parquet") if os.path.exists("./data_kr/matches.parquet") else pd.DataFrame()

# Kombiniere Rang-Informationen
all_ranks = pd.concat([df_euw, df_na, df_kr], ignore_index=True)

# Merge mit Match-Details
df_with_rank = df.merge(
    all_ranks[['puuid', 'tier', 'matchId']].drop_duplicates(),
    on=['puuid', 'matchId'],
    how='left'
)

print("="*60)
print("CHAMPION WIN RATES BY TIER")
print("="*60)

# Win-Rate pro Champion und Tier
for tier in df_with_rank['tier'].dropna().unique():
    tier_data = df_with_rank[df_with_rank['tier'] == tier]
    print(f"\n{tier}:")
    
    # Top 5 Champions nach Win-Rate (min. 5 Spiele)
    champ_stats = tier_data.groupby('championName').agg({
        'win': ['sum', 'count', 'mean']
    }).round(3)
    champ_stats.columns = ['wins', 'games', 'winrate']
    champ_stats = champ_stats[champ_stats['games'] >= 5]
    champ_stats = champ_stats.sort_values('winrate', ascending=False)
    
    print("Top 5 Win-Rate Champions:")
    print(champ_stats.head())
    print(f"Unique Champions played: {len(champ_stats)}")

print("\n" + "="*60)
print("POSITION PREFERENCES BY TIER")
print("="*60)

position_by_tier = df_with_rank.groupby(['tier', 'teamPosition']).size().unstack(fill_value=0)
position_pct = position_by_tier.div(position_by_tier.sum(axis=1), axis=0) * 100
print(position_pct.round(1))

print("\n" + "="*60)
print("KDA STATISTICS BY TIER")
print("="*60)

kda_by_tier = df_with_rank.groupby('tier').agg({
    'kills': 'mean',
    'deaths': 'mean',
    'assists': 'mean',
    'goldEarned': 'mean',
    'totalDamageDealt': 'mean'
}).round(1)

kda_by_tier['KDA'] = ((kda_by_tier['kills'] + kda_by_tier['assists']) / kda_by_tier['deaths']).round(2)
print(kda_by_tier)

# Speichere für weitere Analyse
df_with_rank.to_parquet("./complete_dataset.parquet", index=False)
print(f"\nComplete dataset saved with {len(df_with_rank)} entries")

# Champion Mastery Correlation würde hier kommen wenn wir die Daten hätten
print("\n" + "="*60)
print("NEXT STEPS FOR YOUR THESIS:")
print("="*60)
print("""
1. Sammle mehr Daten (Ziel: 10.000+ Matches)
2. Hole Champion Mastery Daten für Spieler
3. Analysiere:
   - Korrelation zwischen Mastery Points und Win-Rate
   - Champion-Komplexität vs Rang
   - Meta-Shifts über Zeit (gameVersion)
   - Lane-Matchup Win-Rates
   - First-Pick vs Counter-Pick Advantage
""")
