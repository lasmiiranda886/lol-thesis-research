import pandas as pd
import json
import os
from glob import glob

print("Vorbereitung der Analyse-Daten...")

# Sammle alle Match-Details die bereits heruntergeladen sind
match_files = glob("data_collection/matches/*.json")
print(f"Gefundene Match-Dateien: {len(match_files)}")

if len(match_files) > 100:  # Wenn genug Daten da sind
    
    # Extrahiere relevante Daten
    data_rows = []
    
    for file_path in match_files[:1000]:  # Erste 1000 für Test
        with open(file_path, 'r') as f:
            match = json.load(f)
        
        match_id = os.path.basename(file_path).replace('.json', '')
        info = match.get('info', {})
        
        for participant in info.get('participants', []):
            data_rows.append({
                'matchId': match_id,
                'puuid': participant.get('puuid'),
                'championId': participant.get('championId'),
                'championName': participant.get('championName'),
                'champLevel': participant.get('champLevel'),
                'win': participant.get('win'),
                'kills': participant.get('kills'),
                'deaths': participant.get('deaths'),
                'assists': participant.get('assists'),
                'totalDamageDealt': participant.get('totalDamageDealt'),
                'totalMinionsKilled': participant.get('totalMinionsKilled'),
                'teamPosition': participant.get('teamPosition'),
                'gameMode': info.get('gameMode'),
                'gameDuration': info.get('gameDuration')
            })
    
    # Speichere als DataFrame
    df_matches = pd.DataFrame(data_rows)
    
    # Lade Mastery Daten
    mastery_data = []
    mastery_files = glob("data_collection/mastery/*.json")
    
    for file_path in mastery_files[:1000]:
        puuid = os.path.basename(file_path).replace('.json', '')
        with open(file_path, 'r') as f:
            masteries = json.load(f)
        
        for mastery in masteries:
            mastery_data.append({
                'puuid': puuid,
                'championId': mastery.get('championId'),
                'championLevel': mastery.get('championLevel'),
                'championPoints': mastery.get('championPoints')
            })
    
    df_mastery = pd.DataFrame(mastery_data)
    
    # Merge Daten
    df_complete = df_matches.merge(
        df_mastery, 
        on=['puuid', 'championId'], 
        how='left'
    )
    
    # Speichere für Analyse
    df_complete.to_parquet("analysis_ready_data.parquet")
    
    print(f"\nAnalyse-Daten bereit!")
    print(f"- Matches: {len(df_matches)}")
    print(f"- Mit Mastery-Daten: {df_complete['championPoints'].notna().sum()}")
    print(f"- Unique Spieler: {df_complete['puuid'].nunique()}")
    print(f"- Unique Champions: {df_complete['championName'].nunique()}")
    
else:
    print("Warte auf mehr Daten... (mindestens 100 Matches benötigt)")
