import json
import os
import requests
import time
from glob import glob
from tqdm import tqdm

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"
session = requests.Session()
session.headers.update({"X-Riot-Token": API_KEY})

os.makedirs("data_collection/ranks", exist_ok=True)

# Sammle alle einzigartigen PUUIDs aus den heruntergeladenen Matches
match_files = glob("data_collection/matches/*.json")
all_puuids = set()

print("Extrahiere PUUIDs aus Matches...")
for file in match_files[:1000]:  # Erste 1000 Matches
    with open(file, 'r') as f:
        match = json.load(f)
    for p in match.get('info', {}).get('participants', []):
        if p.get('puuid'):
            all_puuids.add(p['puuid'])

print(f"Gefundene unique PUUIDs: {len(all_puuids)}")

# Hole Rang für jeden Spieler
def get_player_rank(puuid, region="euw1"):
    cache_file = f"data_collection/ranks/{puuid}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
    
    try:
        r = session.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Finde Ranked Solo Queue
            for entry in data:
                if entry.get('queueType') == 'RANKED_SOLO_5x5':
                    with open(cache_file, 'w') as f:
                        json.dump(entry, f)
                    return entry
        elif r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 60)))
            return get_player_rank(puuid, region)
    except:
        pass
    
    return None

# Sammle Rang-Daten
rank_data = {}
for puuid in tqdm(list(all_puuids), desc="Collecting ranks"):
    rank_info = get_player_rank(puuid)
    if rank_info:
        rank_data[puuid] = {
            'tier': rank_info.get('tier'),
            'rank': rank_info.get('rank'),
            'leaguePoints': rank_info.get('leaguePoints'),
            'wins': rank_info.get('wins'),
            'losses': rank_info.get('losses')
        }
    time.sleep(0.05)  # Rate limiting

# Speichere zusammengefasste Rang-Daten
with open("data_collection/all_player_ranks.json", 'w') as f:
    json.dump(rank_data, f)

print(f"\nRang-Daten gesammelt für {len(rank_data)} Spieler")
print(f"Beispiel: {list(rank_data.values())[0] if rank_data else 'Keine Daten'}")
