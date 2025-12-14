import os
import json
import time
import pandas as pd
from datetime import datetime

API_KEY = os.getenv("RIOT_API_KEY")
SAVE_DIR = "./data_lol"
PROGRESS_FILE = f"{SAVE_DIR}/progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"processed_matches": [], "last_run": None}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def main():
    progress = load_progress()
    print(f"Bereits verarbeitet: {len(progress['processed_matches'])} Matches")
    
    # Lade bereits gesammelte Match-IDs
    df_matches = pd.read_parquet(f"{SAVE_DIR}/match_ids.parquet")
    new_matches = df_matches[~df_matches['matchId'].isin(progress['processed_matches'])]
    
    print(f"Neue zu verarbeitende Matches: {len(new_matches)}")
    print(f"Start: {datetime.now()}")
    
    # Hier würde der Download-Code kommen
    # Für jetzt nur Status anzeigen
    
if __name__ == "__main__":
    main()
