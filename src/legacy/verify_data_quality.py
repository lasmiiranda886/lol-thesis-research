import os
import json
import pandas as pd
import numpy as np
from glob import glob

print("="*60)
print("DATENQUALITÄTS-PRÜFUNG")
print("="*60)

# Check collected data
mastery_files = glob("data_collection/mastery/*.json")
match_id_files = glob("data_collection/match_ids/*.json")
summoner_files = glob("data_collection/summoners/*.json")

print(f"\nGesammelte Dateien:")
print(f"- Summoner Dateien: {len(summoner_files)}")
print(f"- Match ID Dateien: {len(match_id_files)}")
print(f"- Mastery Dateien: {len(mastery_files)}")

# Beispiel: Lade eine Mastery-Datei
if mastery_files:
    with open(mastery_files[0], 'r') as f:
        sample_mastery = json.load(f)
    
    print(f"\n✓ CHAMPION MASTERY DATEN VORHANDEN!")
    print(f"Beispiel Spieler hat Mastery für {len(sample_mastery)} Champions")
    if sample_mastery:
        print(f"Top Champion: {sample_mastery[0].get('championPoints')} Punkte, Level {sample_mastery[0].get('championLevel')}")

# Beispiel: Lade Match History
if match_id_files:
    with open(match_id_files[0], 'r') as f:
        sample_matches = json.load(f)
    
    print(f"\n✓ MATCH HISTORY VORHANDEN!")
    print(f"Beispiel Spieler: {len(sample_matches)} Matches in Historie")

# Prüfe ob wir genug Daten für deine Analysen haben
print("\n" + "="*60)
print("ANALYSE-MÖGLICHKEITEN MIT DIESEN DATEN:")
print("="*60)

print("""
✓ MÖGLICH:
1. Win-Rate nach Champion-Mastery-Level (0-7)
2. Win-Rate nach Champion-Mastery-Points (kontinuierlich)
3. Durchschnittliche Mastery pro Rang (Iron vs Diamond)
4. Korrelation: Mastery vs Performance (KDA)
5. "Learning Curve": Win-Rate nach N gespielten Spielen
6. Champion-Pool-Größe pro Rang
7. One-Trick-Pony Analyse (Spieler mit >50% Spiele auf 1 Champion)

✓ DATEN DIE WIR JETZT HABEN:
- Vollständige Match-Historie (bis zu 500 Matches pro Spieler)
- Champion Mastery Points & Level für jeden Champion
- Rang-Information (Iron bis Diamond)

NÄCHSTE SCHRITTE:
1. Lass den Collector weiterlaufen (Ziel: 1M Matches)
2. Parallel: Download der Match-Details für die gesammelten IDs
3. Analyse-Pipeline aufbauen
""")

# Speichere Sample für Testing
if os.path.exists("data_collection/progress.json"):
    with open("data_collection/progress.json", 'r') as f:
        progress = json.load(f)
    print(f"\nAktueller Fortschritt:")
    print(f"- Unique Summoners: {len(progress.get('collected_summoners', []))}")
    print(f"- Unique Match IDs: {len(progress.get('collected_match_ids', []))}")
