import pandas as pd
import numpy as np

# Lade vorhandene Daten
df = pd.read_parquet("./match_details_extracted.parquet")
all_ranks = pd.concat([
    pd.read_parquet("./data_euw1/matches.parquet"),
    pd.read_parquet("./data_na1/matches.parquet"),
    pd.read_parquet("./data_kr/matches.parquet")
], ignore_index=True)

print("="*60)
print("VERFÜGBARE DATEN")
print("="*60)

print("\n✓ HABEN WIR:")
print(f"- Match IDs: {df['matchId'].nunique()}")
print(f"- Spieler (PUUIDs): {df['puuid'].nunique()}")
print(f"- Champions gespielt: {df['championName'].nunique()}")
print(f"- Win/Loss pro Spiel: JA")
print(f"- Rang-Information (Gold, Plat): JA")
print(f"- KDA, Gold, Damage: JA")

print("\n✗ FEHLT NOCH:")
print("- Spielerfahrung pro Champion (wie viele Spiele mit Champion X)")
print("- Champion Mastery Points")
print("- Match-Historie pro Spieler (für N-Spiele Analyse)")

print("\n" + "="*60)
print("BEISPIEL: Können wir 'Spielerfahrung' berechnen?")
print("="*60)

# Simuliere Spielerfahrung durch Zählung im aktuellen Dataset
player_champ_exp = df.groupby(['puuid', 'championName']).size().reset_index(name='games_in_dataset')

# Merge zurück
df_with_exp = df.merge(player_champ_exp, on=['puuid', 'championName'])

# Beispiel: Win-Rate nach Erfahrung
print("\nWin-Rate nach Anzahl Spiele (im Dataset):")
exp_groups = df_with_exp.groupby('games_in_dataset')['win'].agg(['mean', 'count'])
print(exp_groups.head(10))

print("\n" + "="*60)
print("WAS WIR BRAUCHEN:")
print("="*60)

print("""
1. MATCH HISTORY: 
   - Wir brauchen die KOMPLETTE Match-Historie jedes Spielers
   - Nicht nur die 10 letzten Matches
   - API Call: /lol/match/v5/matches/by-puuid/{puuid}/ids?count=100&start=0

2. CHAMPION MASTERY:
   - API Call: /lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}
   - Gibt uns: championLevel, championPoints, tokensEarned

3. MEHR MATCHES:
   - Aktuell nur 300 Matches
   - Ziel: 10.000+ für statistische Signifikanz
""")

# Test: Können wir trotzdem eine Basis-Analyse machen?
print("\n" + "="*60)
print("MÖGLICHE ANALYSE MIT AKTUELLEN DATEN:")
print("="*60)

# Gruppe nach Tier
tier_analysis = all_ranks.groupby('tier')['puuid'].nunique()
print(f"\nSpieler pro Tier:")
print(tier_analysis)

# Zeige was möglich wäre
sample_player = df['puuid'].iloc[0]
sample_matches = df[df['puuid'] == sample_player]
print(f"\nBeispiel Spieler: {sample_matches['championName'].value_counts().to_dict()}")
print(f"Problem: Nur {len(sample_matches)} Matches von diesem Spieler im Dataset!")
