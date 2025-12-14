import pandas as pd

print("\n=== LEAGUE ENTRIES ===")
df_entries = pd.read_parquet("data_lol/league_entries_sample.parquet")
print(f"Anzahl Spieler: {len(df_entries)}")
print(f"Spalten: {df_entries.columns.tolist()}")
print("\nVerteilung nach Rang:")
print(df_entries.groupby(['tier', 'division']).size().sort_index())
print("\nBeispiel-Einträge:")
print(df_entries.head(3))

print("\n=== MATCH IDs ===")
df_matches = pd.read_parquet("data_lol/match_ids.parquet")
print(f"Anzahl Match-IDs: {len(df_matches)}")
print(f"Unique Spieler: {df_matches['puuid'].nunique()}")
print(f"Matches pro Spieler (Durchschnitt): {len(df_matches) / df_matches['puuid'].nunique():.1f}")
print("\nBeispiel Match-IDs:")
print(df_matches.head(3))
