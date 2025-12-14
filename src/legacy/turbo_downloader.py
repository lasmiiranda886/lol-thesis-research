import json
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

API_KEY = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"

# Rate Limiter für 2000/10s = 200/s
class RateLimiter:
    def __init__(self, max_per_second=180):  # Etwas unter Limit für Sicherheit
        self.max_per_second = max_per_second
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Entferne alte Calls (älter als 1 Sekunde)
            self.calls = [t for t in self.calls if now - t < 1]
            
            if len(self.calls) >= self.max_per_second:
                sleep_time = 1 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < 1]
            
            self.calls.append(now)

rate_limiter = RateLimiter(180)  # 180 requests/second

def download_match(match_id):
    """Download single match with proper routing"""
    
    # Check if already exists
    if os.path.exists(f"data_collection/matches/{match_id}.json"):
        return "exists"
    
    # Determine routing
    if match_id.startswith("EUW") or match_id.startswith("EUN"):
        routing = "europe"
    elif match_id.startswith("NA"):
        routing = "americas"
    elif match_id.startswith("KR") or match_id.startswith("JP"):
        routing = "asia"
    elif match_id.startswith("BR") or match_id.startswith("LA"):
        routing = "americas"
    elif match_id.startswith("OC"):
        routing = "sea"
    elif match_id.startswith("TR") or match_id.startswith("RU"):
        routing = "europe"
    else:
        routing = "europe"
    
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    
    rate_limiter.wait_if_needed()
    
    try:
        r = requests.get(url, headers={"X-Riot-Token": API_KEY}, timeout=5)
        if r.status_code == 200:
            with open(f"data_collection/matches/{match_id}.json", 'w') as f:
                json.dump(r.json(), f)
            return "success"
        elif r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 10)))
            return "rate_limited"
        elif r.status_code == 404:
            return "not_found"
        else:
            return f"error_{r.status_code}"
    except Exception as e:
        return f"exception"

def main():
    # Load all match IDs
    with open("data_collection/progress.json", 'r') as f:
        progress = json.load(f)
    
    all_match_ids = progress.get('collected_match_ids', [])
    
    # Check what's already downloaded
    existing = set()
    if os.path.exists("data_collection/matches"):
        existing = {f.replace('.json', '') for f in os.listdir("data_collection/matches") if f.endswith('.json')}
    
    missing = [mid for mid in all_match_ids if mid not in existing]
    
    print(f"="*60)
    print(f"TURBO DOWNLOADER - Personal Key Rate Limits")
    print(f"="*60)
    print(f"Total Match IDs: {len(all_match_ids):,}")
    print(f"Already downloaded: {len(existing):,}")
    print(f"To download: {len(missing):,}")
    print(f"Rate limit: 180 requests/second")
    print(f"Estimated time: {len(missing)/180/60:.1f} minutes")
    print(f"="*60)
    
    if not missing:
        print("All matches already downloaded!")
        return
    
    # Download mit Thread Pool für maximale Geschwindigkeit
    with ThreadPoolExecutor(max_workers=50) as executor:
        # Submit first batch
        futures = {executor.submit(download_match, match_id): match_id 
                  for match_id in missing[:50000]}  # 50k Batch
        
        # Process results with progress bar
        results = {"success": 0, "exists": 0, "not_found": 0, "rate_limited": 0, "error": 0}
        
        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "success":
                    results["success"] += 1
                elif result == "exists":
                    results["exists"] += 1
                elif result == "not_found":
                    results["not_found"] += 1
                elif result == "rate_limited":
                    results["rate_limited"] += 1
                else:
                    results["error"] += 1
                
                pbar.update(1)
                pbar.set_postfix(results)
    
    print(f"\nDownload complete!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()
