# Multi-Region LoL Data Pipeline
import os
import time
import random
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
from tqdm import tqdm

# NEUER API KEY
API_KEY = os.getenv("RIOT_API_KEY", "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12")

# MULTI-REGION KONFIGURATION
REGIONS = {
    "euw1": "europe",      # EU West
    "eun1": "europe",      # EU Nordic & East
    "na1": "americas",     # North America
    "br1": "americas",     # Brazil
    "la1": "americas",     # Latin America North
    "la2": "americas",     # Latin America South
    "kr": "asia",          # Korea
    "jp1": "asia",         # Japan
}

# ERWEITERTE RATE LIMITS FÜR PERSONAL KEY
REQS_PER_SEC = 200  # 2000/10 seconds
MAX_WORKERS = 10    # Mehr parallele Arbeiter

TIERS = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
DIVISIONS = ["IV"]  # Erstmal nur eine Division pro Tier für schnelleren Test

TARGET_SUMMONERS_PER_BUCKET = 50
MAX_MATCH_IDS_PER_PUUID = 50
MIN_YEAR = 2024

print(f"""
========================================
MULTI-REGION PRODUCTION PIPELINE
========================================
Regions: {', '.join(REGIONS.keys())}
API Key: {API_KEY[:20]}...
Rate Limit: {REQS_PER_SEC} req/s
Workers: {MAX_WORKERS}
========================================
""")

_bucket = {"tokens": REQS_PER_SEC, "last": time.time()}
_lock = threading.Lock()

def rate_limit():
    with _lock:
        now = time.time()
        elapsed = now - _bucket["last"]
        refill = elapsed * REQS_PER_SEC
        _bucket["tokens"] = min(REQS_PER_SEC, _bucket["tokens"] + refill)
        _bucket["last"] = now
        if _bucket["tokens"] < 1:
            time.sleep((1 - _bucket["tokens"]) / REQS_PER_SEC)
            _bucket["tokens"] = 0
        else:
            _bucket["tokens"] -= 1

def riot_get(url: str, params: Optional[dict] = None, retry: int = 3) -> Any:
    params = params or {}
    headers = {"X-Riot-Token": API_KEY}
    for attempt in range(retry):
        rate_limit()
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 1))
                print(f"[Rate Limited] Waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            r.raise_for_status()
        except Exception as e:
            if attempt == retry - 1:
                print(f"[Error] {url}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None

def process_region(platform: str, region_routing: str) -> pd.DataFrame:
    """Verarbeitet eine einzelne Region"""
    print(f"\n[{platform}] Starting region processing...")
    
    save_dir = f"./data_{platform}"
    os.makedirs(save_dir, exist_ok=True)
    
    all_summoners = []
    
    # Sammle Summoners
    for tier in TIERS:
        for division in DIVISIONS:
            url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
            entries = riot_get(url, {"page": 1}) or []
            
            if entries:
                random.shuffle(entries)
                entries = entries[:TARGET_SUMMONERS_PER_BUCKET]
                
                for e in entries:
                    if e.get("puuid"):
                        all_summoners.append({
                            "platform": platform,
                            "tier": tier,
                            "division": division,
                            "puuid": e["puuid"],
                            "wins": e.get("wins"),
                            "losses": e.get("losses")
                        })
    
    if not all_summoners:
        print(f"[{platform}] No summoners found")
        return pd.DataFrame()
    
    df_summoners = pd.DataFrame(all_summoners)
    print(f"[{platform}] Found {len(df_summoners)} summoners")
    
    # Sammle Match IDs
    all_match_ids = []
    for puuid in tqdm(df_summoners["puuid"].tolist(), desc=f"[{platform}] Match IDs"):
        url = f"https://{region_routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        match_ids = riot_get(url, {"count": MAX_MATCH_IDS_PER_PUUID, "queue": 420}) or []
        for mid in match_ids:
            all_match_ids.append({
                "platform": platform,
                "puuid": puuid,
                "matchId": mid
            })
    
    df_matches = pd.DataFrame(all_match_ids)
    print(f"[{platform}] Found {len(df_matches)} match IDs")
    
    # Speichere Zwischenergebnisse
    df_summoners.to_parquet(f"{save_dir}/summoners.parquet", index=False)
    df_matches.to_parquet(f"{save_dir}/match_ids.parquet", index=False)
    
    return df_matches

def main():
    # Verarbeite alle Regionen parallel
    with ThreadPoolExecutor(max_workers=len(REGIONS)) as executor:
        futures = {
            executor.submit(process_region, platform, routing): platform 
            for platform, routing in REGIONS.items()
        }
        
        results = []
        for future in as_completed(futures):
            platform = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
                    print(f"[{platform}] ✓ Complete")
            except Exception as e:
                print(f"[{platform}] ✗ Failed: {e}")
    
    # Kombiniere alle Ergebnisse
    if results:
        df_all = pd.concat(results, ignore_index=True)
        df_all.to_parquet("./all_regions_matches.parquet", index=False)
        print(f"\n========================================")
        print(f"TOTAL MATCH IDs: {len(df_all)}")
        print(f"Unique Matches: {df_all['matchId'].nunique()}")
        print(f"========================================")

if __name__ == "__main__":
    main()
