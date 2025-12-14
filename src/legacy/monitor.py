#!/usr/bin/env python3
import os
import sys
import pandas as pd
from datetime import datetime
import time

def check_status():
    data_dir = "./data_lol"
    
    print(f"\n{'='*60}")
    print(f"LoL Pipeline Status: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if os.path.exists(data_dir):
        total_size = 0
        file_stats = []
        
        for file in os.listdir(data_dir):
            if file.endswith('.parquet'):
                path = f"{data_dir}/{file}"
                try:
                    df = pd.read_parquet(path)
                    size = os.path.getsize(path) / 1024 / 1024  # MB
                    total_size += size
                    mtime = os.path.getmtime(path)
                    last_modified = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    
                    file_stats.append({
                        'file': file,
                        'rows': len(df),
                        'size_mb': size,
                        'last_modified': last_modified
                    })
                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        # Sort by modification time
        file_stats.sort(key=lambda x: x['last_modified'], reverse=True)
        
        print(f"\nDateien-Übersicht:")
        print(f"{'Datei':<30} | {'Zeilen':>10} | {'Größe (MB)':>12} | {'Letzte Änderung':<20}")
        print("-" * 80)
        
        for stat in file_stats:
            print(f"{stat['file']:<30} | {stat['rows']:>10,} | {stat['size_mb']:>12.2f} | {stat['last_modified']:<20}")
        
        print(f"\nGesamt-Speicherplatz: {total_size:.2f} MB")
    else:
        print("Keine Daten gefunden!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        while True:
            os.system('clear')  # Clear screen
            check_status()
            time.sleep(30)
    else:
        check_status()
