import pandas as pd
import numpy as np
import json
import os
from glob import glob

# Lade Daten
if os.path.exists("analysis_ready_data.parquet"):
    df = pd.read_parquet("analysis_ready_data.parquet")
    print(f"Daten geladen: {len(df)} Einträge")
else:
    print("analysis_ready_data.parquet nicht gefunden!")
    print("Führe zuerst aus: python3 prepare_analysis.py")
    exit()

print("="*60)
print("BACHELORARBEIT - HAUPTANALYSE")
print("="*60)

# 1. Win-Rate nach Champion-Erfahrung
print("\n1. WIN-RATE NACH CHAMPION-MASTERY:")
print("-"*40)

# Gruppiere nach Mastery-Level
mastery_bins = [0, 10000, 50000, 100000, 250000, 500000, float('inf')]
mastery_labels = ['0-10k', '10-50k', '50-100k', '100-250k', '250-500k', '500k+']

df['mastery_bracket'] = pd.cut(df['championPoints'].fillna(0), bins=mastery_bins, labels=mastery_labels)

mastery_winrate = df.groupby('mastery_bracket').agg({
    'win': ['mean', 'count']
}).round(3)

print(mastery_winrate)

# 2. Learning Curve - Win-Rate über Zeit
print("\n2. LEARNING CURVE ANALYSE:")
print("-"*40)

# Für jeden Spieler-Champion, zähle Spiele chronologisch
player_champ_games = []
for puuid in df['puuid'].unique()[:100]:  # Sample für Performance
    for champ in df[df['puuid']==puuid]['championName'].unique():
        games = df[(df['puuid']==puuid) & (df['championName']==champ)].copy()
        games['game_number'] = range(1, len(games)+1)
        player_champ_games.append(games)

if player_champ_games:
    df_learning = pd.concat(player_champ_games)
    
    # Win-Rate nach Spiel-Nummer
    for n in [1, 5, 10, 20, 50, 100]:
        wr = df_learning[df_learning['game_number']==n]['win'].mean()
        count = df_learning[df_learning['game_number']==n].shape[0]
        if count > 0:
            print(f"Spiel #{n}: {wr:.1%} Win-Rate (n={count})")

# 3. Champion-Pool Analyse
print("\n3. CHAMPION-POOL GRÖSSE:")
print("-"*40)

champion_pools = df.groupby('puuid')['championName'].nunique()
print(f"Durchschnitt: {champion_pools.mean():.1f} Champions")
print(f"Median: {champion_pools.median():.0f} Champions")
print(f"One-Tricks (≤3 Champions): {(champion_pools <= 3).mean():.1%}")

# 4. Performance-Korrelation
print("\n4. MASTERY vs PERFORMANCE:")
print("-"*40)

df_clean = df.dropna(subset=['championPoints', 'kills', 'deaths', 'assists'])
if len(df_clean) > 0:
    df_clean['kda'] = (df_clean['kills'] + df_clean['assists']) / df_clean['deaths'].clip(lower=1)
    
    correlation = df_clean[['championPoints', 'kda', 'win']].corr()
    print("Korrelation Mastery-Points zu:")
    print(f"- KDA: {correlation.loc['championPoints', 'kda']:.3f}")
    print(f"- Win: {correlation.loc['championPoints', 'win']:.3f}")

print("\n" + "="*60)
print("EMPFEHLUNG FÜR THESIS:")
print("Diese Daten zeigen bereits interessante Muster!")
print("Mit 1M+ Matches werden die Ergebnisse statistisch robust.")
print("="*60)
