#!/usr/bin/env python3
import os
import json
import time
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import hashlib

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Load config
with open("collection_config.json", "r") as f:
    config = json.load(f)

class DataCollector:
    def __init__(self):
        self.api_key = API_KEY
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})
        self.request_count = 0
        self.start_time = time.time()
        
        # Create directories
        os.makedirs("data_collection", exist_ok=True)
        os.makedirs("data_collection/match_ids", exist_ok=True)
        os.makedirs("data_collection/matches", exist_ok=True)
        os.makedirs("data_collection/mastery", exist_ok=True)
        os.makedirs("data_collection/summoners", exist_ok=True)
        
        # Load progress
        self.progress_file = "data_collection/progress.json"
        self.progress = self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "collected_summoners": [],
            "collected_match_ids": [],
            "processed_matches": [],
            "last_region": None,
            "last_tier": None,
            "last_division": None
        }
    
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def rate_limit_wait(self):
        # Personal key: 2000 requests per 10 seconds
        self.request_count += 1
        if self.request_count % 100 == 0:
            elapsed = time.time() - self.start_time
            if elapsed < 5:  # If we're going too fast
                time.sleep(5 - elapsed)
            self.start_time = time.time()
    
    def riot_request(self, url, params=None):
        self.rate_limit_wait()
        try:
            r = self.session.get(url, params=params, timeout=10)
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", 60))
                print(f"\n[Rate Limited] Waiting {retry_after}s...")
                time.sleep(retry_after + 1)
                return self.riot_request(url, params)
            if r.status_code == 200:
                return r.json()
            return None
        except Exception as e:
            print(f"\nError: {e}")
            return None
    
    def collect_summoners(self, region, tier, division):
        """Phase 1: Collect summoner PUUIDs"""
        cache_key = f"{region}_{tier}_{division}"
        cache_file = f"data_collection/summoners/{cache_key}.json"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
        summoners = []
        
        for page in range(1, 6):  # Get up to 5 pages
            data = self.riot_request(url, {"page": page})
            if not data:
                break
            summoners.extend(data)
            if len(data) < 205:  # Less than full page, no more data
                break
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(summoners, f)
        
        return summoners
    
    def collect_match_history(self, puuid, region_routing):
        """Phase 2: Collect match IDs for a player"""
        cache_file = f"data_collection/match_ids/{puuid}.json"
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        url = f"https://{region_routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        
        all_matches = []
        for start in range(0, 500, 100):  # Get up to 500 matches per player
            matches = self.riot_request(url, {
                "queue": 420,  # Ranked Solo
                "start": start,
                "count": 100
            })
            if not matches:
                break
            all_matches.extend(matches)
            if len(matches) < 100:
                break
        
        # Save cache
        with open(cache_file, 'w') as f:
            json.dump(all_matches, f)
        
        return all_matches
    
    def collect_champion_mastery(self, puuid, region):
        """Phase 3: Collect champion mastery"""
        cache_file = f"data_collection/mastery/{puuid}.json"
        
        if os.path.exists(cache_file):
            return
        
        url = f"https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}"
        mastery = self.riot_request(url)
        
        if mastery:
            with open(cache_file, 'w') as f:
                json.dump(mastery, f)
    
    def run_collection(self, target_matches=100000):
        """Main collection loop"""
        print("="*60)
        print("STARTING MASSIVE DATA COLLECTION")
        print(f"Target: {target_matches:,} matches")
        print("="*60)
        
        unique_matches = set(self.progress.get("collected_match_ids", []))
        unique_summoners = set(self.progress.get("collected_summoners", []))
        
        # Phase 1: Collect summoners and match IDs
        for region, routing in config["regions"].items():
            if len(unique_matches) >= target_matches:
                break
                
            print(f"\n[{region}] Processing region...")
            
            for tier in config["tiers"]:
                if len(unique_matches) >= target_matches:
                    break
                    
                for division in config["divisions"]:
                    if len(unique_matches) >= target_matches:
                        break
                    
                    # Skip if already processed
                    if (self.progress.get("last_region") == region and 
                        self.progress.get("last_tier") == tier and 
                        self.progress.get("last_division") == division):
                        continue
                    
                    print(f"[{region}] {tier} {division}")
                    
                    # Get summoners
                    summoners = self.collect_summoners(region, tier, division)
                    
                    # Collect match history for each summoner
                    for summoner in tqdm(summoners[:50], desc=f"{tier} {division}"):
                        if summoner.get("puuid") in unique_summoners:
                            continue
                        
                        puuid = summoner.get("puuid")
                        if not puuid:
                            continue
                        
                        # Get match history
                        matches = self.collect_match_history(puuid, routing)
                        unique_matches.update(matches)
                        unique_summoners.add(puuid)
                        
                        # Get mastery
                        self.collect_champion_mastery(puuid, region)
                        
                        # Save progress periodically
                        if len(unique_summoners) % 100 == 0:
                            self.progress["collected_summoners"] = list(unique_summoners)
                            self.progress["collected_match_ids"] = list(unique_matches)
                            self.progress["last_region"] = region
                            self.progress["last_tier"] = tier
                            self.progress["last_division"] = division
                            self.save_progress()
                            
                            print(f"\nProgress: {len(unique_summoners):,} summoners, {len(unique_matches):,} unique matches")
        
        print("\n" + "="*60)
        print("COLLECTION COMPLETE")
        print(f"Total summoners: {len(unique_summoners):,}")
        print(f"Total unique matches: {len(unique_matches):,}")
        print("="*60)
        
        # Save final results
        pd.DataFrame({"match_id": list(unique_matches)}).to_parquet("data_collection/all_match_ids.parquet")
        pd.DataFrame({"puuid": list(unique_summoners)}).to_parquet("data_collection/all_summoners.parquet")

if __name__ == "__main__":
    collector = DataCollector()
    # Start with 100k for test, dann später 1M
    collector.run_collection(target_matches=100000)
