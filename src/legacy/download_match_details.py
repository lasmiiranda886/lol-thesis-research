import os
import json
import pandas as pd
import requests
from tqdm import tqdm
import time

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Lade alle gesammelten Match IDs
if os.path.exists("data_collection/progress.json"):
    with open("data_collection/progress.json", 'r') as f:
        progress = json.load(f)
    match_ids = progress.get('collected_match_ids', [])[:1000]  # Erste 1000 für Test
else:
    print("Keine Match IDs gefunden!")
    exit()

print(f"Downloading details for {len(match_ids)} matches...")

session = requests.Session()
session.headers.update({"X-Riot-Token": API_KEY})

os.makedirs("data_collection/matches", exist_ok=True)

for match_id in tqdm(match_ids):
    cache_file = f"data_collection/matches/{match_id}.json"
    
    if os.path.exists(cache_file):
        continue
    
    # Determine routing from match ID
    if match_id.startswith("EUW"):
        routing = "europe"
    elif match_id.startswith("NA"):
        routing = "americas"
    elif match_id.startswith("KR") or match_id.startswith("JP"):
        routing = "asia"
    else:
        routing = "europe"  # default
    
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    
    try:
        r = session.get(url, timeout=10)
        if r.status_code == 200:
            with open(cache_file, 'w') as f:
                json.dump(r.json(), f)
        elif r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 60)))
        time.sleep(0.05)  # Rate limiting
    except:
        continue

print("Download complete!")
