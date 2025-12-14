import pandas as pd
import json
import os
from glob import glob

print("="*60)
print("DATENSTAND FÜR CHAMPION SELECT PREDICTION MODEL")
print("="*60)

# Check aktueller Stand
with open("data_collection/progress.json", 'r') as f:
    progress = json.load(f)

match_files = glob("data_collection/matches/*.json")
mastery_files = glob("data_collection/mastery/*.json")

print(f"\n✓ HABEN:")
print(f"- Unique Match IDs: {len(progress.get('collected_match_ids', [])):,}")
print(f"- Heruntergeladene Matches: {len(match_files):,}")
print(f"- Spieler mit Mastery-Daten: {len(mastery_files):,}")

print(f"\n✗ FEHLT für dein Model:")
print("1. SPIELER-RANG-ZUORDNUNG für JEDEN Spieler in JEDEM Match")
print("2. TEAM-ZUSAMMENSETZUNG (wer spielt mit/gegen wen)")
print("3. PRE-GAME DATEN (Mastery VOR dem Match, nicht danach)")

# Lade Sample Match für Struktur-Check
if match_files:
    with open(match_files[0], 'r') as f:
        sample = json.load(f)
    
    print(f"\n📊 MATCH-STRUKTUR ANALYSE:")
    info = sample.get('info', {})
    
    # Check Team-Daten
    teams = info.get('teams', [])
    if teams:
        print(f"- Teams vorhanden: JA ({len(teams)} Teams)")
    
    # Check Participant-Daten
    participants = info.get('participants', [])
    if participants:
        p = participants[0]
        print(f"- Spieler pro Match: {len(participants)}")
        print(f"- Haben wir PUUID: {'puuid' in p}")
        print(f"- Haben wir Team-ID: {'teamId' in p}")
        print(f"- Haben wir Champion: {'championName' in p}")
        
        # KRITISCH: Haben wir Rang-Info?
        if 'tier' in p or 'rank' in p:
            print(f"- Rang im Match: JA")
        else:
            print(f"- Rang im Match: NEIN ⚠️")

print("\n" + "="*60)
print("NÄCHSTE SCHRITTE FÜR DEIN MODEL:")
print("="*60)
print("""
1. RANG-DATEN SAMMELN:
   - Für JEDEN Spieler den aktuellen Rang abrufen
   - API: /lol/league/v4/entries/by-puuid/{puuid}

2. MATCH-TIMELINE DATEN:
   - Pre-Game Mastery-Stand
   - Champion Select Reihenfolge

3. TEAM-COMPOSITION ANALYSE:
   - 5v5 Team-Matchups
   - Durchschnittliche Mastery pro Team
   - Rang-Verteilung pro Team
""")

# Berechne wie viele Daten wir für statistisch robuste Vorhersagen brauchen
print("\n📈 DATEN-ANFORDERUNGEN:")
print("-"*40)

champions = 168  # Anzahl Champions in LoL
ranks = 7  # Iron bis Diamond
min_samples_per_combo = 30  # Minimum für Statistik

total_needed = champions * ranks * min_samples_per_combo
print(f"Minimum benötigte Datenpunkte: {total_needed:,}")
print(f"Aktuell haben wir: ~{len(match_files)*10:,} Spieler-Champion-Kombinationen")
print(f"Fortschritt: {(len(match_files)*10/total_needed)*100:.1f}%")
