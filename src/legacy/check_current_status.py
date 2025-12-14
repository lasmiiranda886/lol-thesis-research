import json
import os

with open("data_collection/progress.json", 'r') as f:
    progress = json.load(f)

match_ids = progress.get('collected_match_ids', [])
print(f"Aktuelle Match IDs: {len(match_ids):,}")

# Check bereits heruntergeladene Matches
existing_matches = len([f for f in os.listdir("data_collection/matches") if f.endswith('.json')]) if os.path.exists("data_collection/matches") else 0
print(f"Bereits heruntergeladen: {existing_matches:,}")
print(f"Noch zu downloaden: {len(match_ids) - existing_matches:,}")
