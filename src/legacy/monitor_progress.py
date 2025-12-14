import os
import json
import time
from datetime import datetime

def get_stats():
    # Progress file
    with open("data_collection/progress.json", 'r') as f:
        progress = json.load(f)
    
    # Count files
    match_details = len([f for f in os.listdir("data_collection/matches") if f.endswith('.json')]) if os.path.exists("data_collection/matches") else 0
    mastery_files = len([f for f in os.listdir("data_collection/mastery") if f.endswith('.json')]) if os.path.exists("data_collection/mastery") else 0
    
    return {
        "summoners": len(progress.get('collected_summoners', [])),
        "match_ids": len(progress.get('collected_match_ids', [])),
        "match_details": match_details,
        "mastery_data": mastery_files
    }

# Monitor loop
print("Monitoring progress... (Ctrl+C to stop)\n")
while True:
    stats = get_stats()
    
    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
          f"Summoners: {stats['summoners']:,} | "
          f"Match IDs: {stats['match_ids']:,} | "
          f"Downloaded: {stats['match_details']:,} | "
          f"Mastery: {stats['mastery_data']:,}", end="")
    
    time.sleep(10)
