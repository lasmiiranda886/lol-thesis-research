import json
import os
import requests
import time
from tqdm import tqdm
import sys

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"  # Dein Personal Key

def download_missing_matches():
    # Lade Progress
    with open("data_collection/progress.json", 'r') as f:
        progress = json.load(f)
    
    all_match_ids = progress.get('collected_match_ids', [])
    
    # Check was bereits heruntergeladen wurde
    existing = set()
    if os.path.exists("data_collection/matches"):
        existing = {f.replace('.json', '') for f in os.listdir("data_collection/matches") if f.endswith('.json')}
    
    missing = [mid for mid in all_match_ids if mid not in existing]
    
    print(f"Total Match IDs: {len(all_match_ids):,}")
    print(f"Bereits heruntergeladen: {len(existing):,}")
    print(f"Noch zu downloaden: {len(missing):,}")
    
    if not missing:
        print("Alle Matches bereits heruntergeladen!")
        return
    
    session = requests.Session()
    session.headers.update({"X-Riot-Token": API_KEY})
    
    # Download mit Resume-Funktion
    for match_id in tqdm(missing[:10000]):  # Max 10k pro Durchlauf
        # Bestimme Region
        if match_id.startswith("EUW"):
            routing = "europe"
        elif match_id.startswith("NA"):
            routing = "americas"
        elif match_id.startswith("KR") or match_id.startswith("JP"):
            routing = "asia"
        elif match_id.startswith("BR") or match_id.startswith("LA"):
            routing = "americas"
        elif match_id.startswith("OC"):
            routing = "sea"
        else:
            routing = "europe"
        
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                with open(f"data_collection/matches/{match_id}.json", 'w') as f:
                    json.dump(r.json(), f)
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"\nRate limited, waiting {wait}s...")
                time.sleep(wait + 1)
            elif r.status_code == 404:
                continue  # Match existiert nicht mehr
            
            time.sleep(0.05)  # 20 requests/second mit Personal Key
            
        except KeyboardInterrupt:
            print("\nDownload unterbrochen. Fortschritt wurde gespeichert.")
            break
        except Exception as e:
            continue
    
    print("\nDownload-Batch abgeschlossen!")

if __name__ == "__main__":
    while True:
        download_missing_matches()
        print("\nWarte 60 Sekunden vor nächstem Batch...")
        time.sleep(60)
