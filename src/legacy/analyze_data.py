import pandas as pd

print("=" * 80)
print("DATENSTRUKTUR-ANALYSE")
print("=" * 80)

# 1. League Entries Details
print("\n1. LEAGUE ENTRIES - Spielerdaten")
print("-" * 40)
df = pd.read_parquet("data_lol/league_entries_sample.parquet")
print("Spalten und Datentypen:")
for col in df.columns:
    print(f"  {col:20} {str(df[col].dtype):15} Beispiel: {df[col].iloc[0]}")

print("\nErste 3 komplette Zeilen:")
print(df.head(3).to_string())

# 2. Match IDs Struktur
print("\n2. MATCH IDs - Struktur")
print("-" * 40)
df_matches = pd.read_parquet("data_lol/match_ids.parquet")
print(f"Erste 5 Match-IDs für einen Spieler:")
sample_puuid = df_matches['puuid'].iloc[0]
sample_matches = df_matches[df_matches['puuid'] == sample_puuid]['matchId'].head(5)
for mid in sample_matches:
    print(f"  {mid}")

# 3. Statistiken
print("\n3. STATISTIKEN")
print("-" * 40)
df_entries = pd.read_parquet("data_lol/league_entries_sample.parquet")
df_entries['winrate'] = df_entries['wins'] / (df_entries['wins'] + df_entries['losses']) * 100
print(f"Win-Rate Verteilung:")
print(f"  Durchschnittliche Win-Rate: {df_entries['winrate'].mean():.1f}%")
print(f"  Höchste Win-Rate: {df_entries['winrate'].max():.1f}%")
print(f"  Niedrigste Win-Rate: {df_entries['winrate'].min():.1f}%")

print(f"\nLeague Points Verteilung:")
print(f"  Durchschnitt: {df_entries['leaguePoints'].mean():.0f} LP")
print(f"  Maximum: {df_entries['leaguePoints'].max()} LP")
print(f"  Minimum: {df_entries['leaguePoints'].min()} LP")

print(f"\nSpiele pro Spieler:")
df_entries['total_games'] = df_entries['wins'] + df_entries['losses']
print(f"  Durchschnitt: {df_entries['total_games'].mean():.0f} Spiele")
print(f"  Maximum: {df_entries['total_games'].max()} Spiele")
print(f"  Minimum: {df_entries['total_games'].min()} Spiele")
