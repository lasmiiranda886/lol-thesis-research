import pandas as pd
import requests
import time
from tqdm import tqdm
import os
import json

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Lade die Match-IDs
df_all = pd.read_parquet("./all_regions_combined.parquet")
print(f"Loaded {len(df_all)} match IDs from {df_all['platform'].nunique()} regions")

# Rate limiting
request_times = []

def wait_for_rate_limit():
    global request_times
    now = time.time()
    request_times = [t for t in request_times if now - t < 10]
    
    if len(request_times) >= 200:  # 2000 per 10 seconds / 10 = 200 per second
        time.sleep(0.1)
    
    request_times.append(now)

def get_match_details(match_id, region):
    routing = {
        "euw1": "europe",
        "na1": "americas", 
        "kr": "asia"
    }.get(region, "europe")
    
    wait_for_rate_limit()
    
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": API_KEY}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 60))
            print(f"\n[Rate Limited] Waiting {retry_after}s...")
            time.sleep(retry_after)
            return get_match_details(match_id, region)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"\nError getting {match_id}: {e}")
    
    return None

# Sammle Match-Details
os.makedirs("./match_details", exist_ok=True)
all_matches = []

# Gruppiere nach Region für effizientes Routing
for region in df_all['platform'].unique():
    region_matches = df_all[df_all['platform'] == region]['matchId'].unique()
    print(f"\nProcessing {len(region_matches)} matches from {region}")
    
    for match_id in tqdm(region_matches[:100], desc=region):  # Nur 100 pro Region für Test
        
        # Check ob bereits heruntergeladen
        cache_file = f"./match_details/{match_id}.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                match_data = json.load(f)
        else:
            match_data = get_match_details(match_id, region)
            if match_data:
                # Cache für spätere Verwendung
                with open(cache_file, 'w') as f:
                    json.dump(match_data, f)
        
        if match_data:
            # Extrahiere wichtige Daten
            info = match_data.get('info', {})
            for participant in info.get('participants', []):
                all_matches.append({
                    'matchId': match_id,
                    'region': region,
                    'puuid': participant.get('puuid'),
                    'championId': participant.get('championId'),
                    'championName': participant.get('championName'),
                    'teamPosition': participant.get('teamPosition'),
                    'win': participant.get('win'),
                    'kills': participant.get('kills'),
                    'deaths': participant.get('deaths'),
                    'assists': participant.get('assists'),
                    'totalDamageDealt': participant.get('totalDamageDealt'),
                    'goldEarned': participant.get('goldEarned'),
                    'champLevel': participant.get('champLevel'),
                    'gameDuration': info.get('gameDuration'),
                    'gameVersion': info.get('gameVersion')
                })

# Speichere Ergebnisse
df_matches = pd.DataFrame(all_matches)
df_matches.to_parquet("./match_details_extracted.parquet", index=False)

print(f"\n{'='*50}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*50}")
print(f"Total participants: {len(df_matches)}")
print(f"Unique matches: {df_matches['matchId'].nunique()}")
print(f"Unique champions: {df_matches['championName'].nunique()}")
print(f"\nTop 10 Champions by popularity:")
print(df_matches['championName'].value_counts().head(10))
print(f"\nWin rates by position:")
print(df_matches.groupby('teamPosition')['win'].mean().sort_values(ascending=False))
