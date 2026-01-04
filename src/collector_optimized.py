#!/usr/bin/env python3
"""
Data Directory Scanner
Scannt das komplette Projektverzeichnis und gibt eine Übersicht über alle Daten.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import json

def get_size_str(size_bytes):
    """Konvertiert Bytes in lesbare Größe"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def scan_directory(base_path):
    """Scannt ein Verzeichnis rekursiv"""
    
    results = {
        "base_path": str(base_path),
        "summary": {
            "total_files": 0,
            "total_size": 0,
            "by_extension": defaultdict(lambda: {"count": 0, "size": 0}),
        },
        "directories": {},
        "important_files": [],
        "parquet_files": [],
        "json_dirs": [],
        "pkl_files": [],
    }
    
    # Wichtige Dateitypen
    important_extensions = {'.parquet', '.pkl', '.json', '.csv', '.txt', '.py'}
    
    for root, dirs, files in os.walk(base_path):
        rel_root = os.path.relpath(root, base_path)
        
        # Skip versteckte Verzeichnisse und node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '__pycache__']
        
        dir_info = {
            "file_count": len(files),
            "subdir_count": len(dirs),
            "files_by_type": defaultdict(int),
            "total_size": 0,
        }
        
        json_count = 0
        
        for f in files:
            filepath = os.path.join(root, f)
            try:
                size = os.path.getsize(filepath)
            except:
                size = 0
            
            ext = os.path.splitext(f)[1].lower()
            
            results["summary"]["total_files"] += 1
            results["summary"]["total_size"] += size
            results["summary"]["by_extension"][ext]["count"] += 1
            results["summary"]["by_extension"][ext]["size"] += size
            
            dir_info["files_by_type"][ext] += 1
            dir_info["total_size"] += size
            
            if ext == '.json':
                json_count += 1
            
            # Wichtige Dateien erfassen
            if ext == '.parquet':
                results["parquet_files"].append({
                    "path": os.path.relpath(filepath, base_path),
                    "size": get_size_str(size),
                    "size_bytes": size
                })
            elif ext == '.pkl':
                results["pkl_files"].append({
                    "path": os.path.relpath(filepath, base_path),
                    "size": get_size_str(size),
                    "size_bytes": size
                })
        
        # JSON-Verzeichnisse (viele JSON-Dateien = Match-Daten)
        if json_count > 100:
            results["json_dirs"].append({
                "path": rel_root,
                "json_count": json_count,
                "total_size": get_size_str(dir_info["total_size"])
            })
        
        results["directories"][rel_root] = {
            "file_count": dir_info["file_count"],
            "subdir_count": dir_info["subdir_count"],
            "total_size": get_size_str(dir_info["total_size"]),
            "files_by_type": dict(dir_info["files_by_type"])
        }
    
    # Sortiere nach Größe
    results["parquet_files"].sort(key=lambda x: x["size_bytes"], reverse=True)
    results["pkl_files"].sort(key=lambda x: x["size_bytes"], reverse=True)
    results["json_dirs"].sort(key=lambda x: x["json_count"], reverse=True)
    
    return results

def print_report(results):
    """Gibt einen formatierten Bericht aus"""
    
    print("=" * 80)
    print("DATA DIRECTORY SCAN REPORT")
    print("=" * 80)
    print(f"\nBase Path: {results['base_path']}")
    print(f"Total Files: {results['summary']['total_files']:,}")
    print(f"Total Size: {get_size_str(results['summary']['total_size'])}")
    
    print("\n" + "-" * 40)
    print("FILES BY EXTENSION:")
    print("-" * 40)
    ext_stats = sorted(
        results['summary']['by_extension'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    for ext, stats in ext_stats[:15]:
        ext_name = ext if ext else "(no extension)"
        print(f"  {ext_name:12} : {stats['count']:>8,} files, {get_size_str(stats['size']):>12}")
    
    print("\n" + "-" * 40)
    print("PARQUET FILES (Data):")
    print("-" * 40)
    for pf in results['parquet_files'][:20]:
        print(f"  {pf['size']:>12} : {pf['path']}")
    if len(results['parquet_files']) > 20:
        print(f"  ... and {len(results['parquet_files']) - 20} more parquet files")
    
    print("\n" + "-" * 40)
    print("PKL FILES (Models/Caches):")
    print("-" * 40)
    for pf in results['pkl_files'][:15]:
        print(f"  {pf['size']:>12} : {pf['path']}")
    if len(results['pkl_files']) > 15:
        print(f"  ... and {len(results['pkl_files']) - 15} more pkl files")
    
    print("\n" + "-" * 40)
    print("JSON DIRECTORIES (Match Data):")
    print("-" * 40)
    for jd in results['json_dirs'][:10]:
        print(f"  {jd['json_count']:>8,} JSONs, {jd['total_size']:>12} : {jd['path']}")
    
    print("\n" + "-" * 40)
    print("TOP-LEVEL DIRECTORY STRUCTURE:")
    print("-" * 40)
    
    # Nur erste 2 Ebenen zeigen
    for dir_path, info in sorted(results['directories'].items()):
        depth = dir_path.count(os.sep)
        if depth <= 1 and info['file_count'] > 0:
            print(f"  {dir_path}/")
            print(f"    Files: {info['file_count']}, Subdirs: {info['subdir_count']}, Size: {info['total_size']}")
            if info['files_by_type']:
                types_str = ", ".join([f"{k}: {v}" for k, v in sorted(info['files_by_type'].items(), key=lambda x: -x[1])[:5]])
                print(f"    Types: {types_str}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_data_directory.py <path_to_scan>")
        print("\nExample:")
        print("  python scan_data_directory.py /path/to/lol-data-pipeline")
        sys.exit(1)
    
    base_path = Path(sys.argv[1])
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        sys.exit(1)
    
    print(f"Scanning {base_path}...")
    results = scan_directory(base_path)
    
    # Print report
    print_report(results)
    
    # Save JSON for detailed analysis
    output_file = "scan_results.json"
    
    # Convert defaultdict to dict for JSON serialization
    results['summary']['by_extension'] = dict(results['summary']['by_extension'])
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    print("You can share this file for further analysis.")

if __name__ == "__main__":
    main()