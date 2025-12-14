import pandas as pd
import requests
import time
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Erweiterte Regionen-Liste
ALL_REGIONS = {
    "euw1": "europe",
    "eun1": "europe", 
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "tr1": "europe",
    "ru": "europe"
}

# Alle Ränge für maximale Abdeckung
ALL_TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
ALL_DIVISIONS = ["I", "II", "III", "IV"]

print("="*60)
print("DATENSAMMLUNG KALKULATION")
print("="*60)

# Berechne theoretisches Maximum
total_regions = len(ALL_REGIONS)
total_tier_div_combos = len(ALL_TIERS) * len(ALL_DIVISIONS)
players_per_bucket = 200  # Max von API
matches_per_player = 100  # Letzte 100 Matches

theoretical_players = total_regions * total_tier_div_combos * players_per_bucket
theoretical_matches = theoretical_players * matches_per_player

print(f"""
Theoretisches Maximum:
- Regionen: {total_regions}
- Rang-Kombinationen: {total_tier_div_combos}
- Spieler pro Bucket: {players_per_bucket}
- Matches pro Spieler: {matches_per_player}

TOTAL:
- Spieler: {theoretical_players:,}
- Matches: {theoretical_matches:,}
- Unique Matches (geschätzt 30% Overlap): {int(theoretical_matches * 0.7):,}
""")

# Rate Limit Berechnung
requests_needed = theoretical_players + theoretical_matches + (theoretical_matches * 0.7)  # Players + Match IDs + Match Details
hours_needed = requests_needed / (2000 * 6)  # 2000 requests per 10 seconds

print(f"""
Zeit-Schätzung bei Personal Key Rate Limits:
- Benötigte Requests: {int(requests_needed):,}
- Geschätzte Zeit: {hours_needed:.1f} Stunden ({hours_needed/24:.1f} Tage)
""")

print(f"""
EMPFEHLUNG FÜR 1 MILLION MATCHES:

1. PHASE 1 - Breite Datensammlung (1-2 Tage):
   - Alle Regionen
   - Alle Ränge  
   - 50 Spieler pro Rang
   - Nur Match IDs sammeln
   
2. PHASE 2 - Match Details (3-5 Tage):
   - Unique Matches herunterladen
   - Mit Caching arbeiten
   - Mehrere Instanzen parallel
   
3. PHASE 3 - Champion Mastery (1 Tag):
   - Für alle gesammelten Spieler
   - Mastery Points + Level

KRITISCHE DATEN DIE WIR SAMMELN MÜSSEN:
- Match History: VOLLSTÄNDIG (nicht nur letzte 10)
- Champion Mastery: FÜR JEDEN SPIELER
- Summoner Stats: Totale Spiele, Account Level
""")

# Speichere Konfiguration
config = {
    "api_key": API_KEY,
    "regions": ALL_REGIONS,
    "tiers": ALL_TIERS,
    "divisions": ALL_DIVISIONS,
    "target_matches": 1000000,
    "created_at": datetime.now().isoformat()
}

with open("collection_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nKonfiguration gespeichert in collection_config.json")
print("\nBereit für Massen-Datensammlung? (Führe 'start_collection.py' aus)")
