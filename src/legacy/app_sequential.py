# Sequential Multi-Region Pipeline - Respektiert Rate Limits
import os
import time
import random
import pandas as pd
from tqdm import tqdm
import requests

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Regionen nacheinander verarbeiten
REGIONS = {
    "euw1": "europe",
    "na1": "americas", 
    "kr": "asia",
}

# Konservative Rate Limits
REQUESTS_PER_SECOND = 20  # Sehr konservativ
REQUESTS_PER_10_MIN = 1000  # Track für 10-Minuten-Limit

class RateLimiter:
    def __init__(self):
        self.requests_made = []
        
    def wait_if_needed(self):
        now = time.time()
        # Entferne alte Requests (älter als 10 Minuten)
        self.requests_made = [t for t in self.requests_made if now - t < 600]
        
        # Check 10-Minuten-Limit
        if len(self.requests_made) >= REQUESTS_PER_10_MIN:
            wait_time = 600 - (now - self.requests_made[0])
            if wait_time > 0:
                print(f"[Rate Limit] Waiting {wait_time:.0f}s for 10-min window...")
                time.sleep(wait_time + 1)
        
        # Check Sekunden-Limit
        recent = [t for t in self.requests_made if now - t < 1]
        if len(recent) >= REQUESTS_PER_SECOND:
            time.sleep(1)
        
        self.requests_made.append(now)

rate_limiter = RateLimiter()

def riot_get(url, params=None):
    rate_limiter.wait_if_needed()
    headers = {"X-Riot-Token": API_KEY}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", 120))
            print(f"[429] Rate limited! Waiting {retry_after}s...")
            time.sleep(retry_after)
            return riot_get(url, params)  # Retry
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_region(platform, routing):
    print(f"\n{'='*50}")
    print(f"Processing {platform}")
    print(f"{'='*50}")
    
    save_dir = f"./data_{platform}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Nur 2 Tiers, 1 Division für schnellen Test
    tiers = ["GOLD", "PLATINUM"]
    divisions = ["III"]
    
    all_data = []
    
    for tier in tiers:
        for division in divisions:
            print(f"[{platform}] Getting {tier} {division}...")
            url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
            entries = riot_get(url, {"page": 1}) or []
            
            if entries:
                # Nur 20 Spieler pro Tier/Division
                entries = entries[:20]
                
                for e in tqdm(entries, desc=f"{tier} {division}"):
                    if not e.get("puuid"):
                        continue
                    
                    # Hole Match IDs
                    match_url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{e['puuid']}/ids"
                    match_ids = riot_get(match_url, {"count": 10, "queue": 420}) or []
                    
                    for mid in match_ids[:10]:  # Nur 10 Matches pro Spieler
                        all_data.append({
                            "platform": platform,
                            "tier": tier,
                            "puuid": e["puuid"],
                            "matchId": mid
                        })
    
    df = pd.DataFrame(all_data)
    df.to_parquet(f"{save_dir}/matches.parquet", index=False)
    print(f"[{platform}] Saved {len(df)} match IDs")
    return df

def main():
    all_results = []
    
    for platform, routing in REGIONS.items():
        df = process_region(platform, routing)
        all_results.append(df)
        time.sleep(10)  # Pause zwischen Regionen
    
    # Kombiniere alle
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_parquet("./all_regions_combined.parquet", index=False)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total Match IDs: {len(df_all)}")
    print(f"Regions: {df_all['platform'].unique()}")
    print(f"Unique Matches: {df_all['matchId'].nunique()}")

if __name__ == "__main__":
    main()
