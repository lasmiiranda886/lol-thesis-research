#!/usr/bin/env python3
"""
Simpler, funktionierender Collector - testet erst die API
"""

import requests
import time
import pandas as pd
from pathlib import Path
import json

def test_api_key(api_key):
    """Testet ob der API Key funktioniert"""
    headers = {"X-Riot-Token": api_key}
    
    # Test 1: Account API (sollte immer funktionieren)
    test_url = "https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/Faker/T1"
    r = requests.get(test_url, headers=headers)
    print(f"Account API Test: {r.status_code}")
    if r.status_code != 200:
        print(f"Error: {r.text}")
        return False
    
    # Test 2: Ein bekanntes Match
    test_match = "EUW1_6851875459"  # Ein recent Match
    match_url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{test_match}"
    r = requests.get(match_url, headers=headers)
    print(f"Match API Test: {r.status_code}")
    
    return r.status_code == 200

def get_puuid_from_name(name, tag, api_key):
    """Holt PUUID von einem bekannten Spieler"""
    headers = {"X-Riot-Token": api_key}
    url = f"https://europe.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{name}/{tag}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json().get('puuid')
    return None

def simple_collect(api_key, num_matches=100):
    """Einfacher, robuster Collector"""
    headers = {"X-Riot-Token": api_key}
    
    # Bekannte High-Elo Spieler (manuell, aber funktioniert garantiert)
    known_players = [
        ("Caps", "EUW"),
        ("Jankos", "EUW"), 
        ("Rekkles", "EUW"),
        ("Bwipo", "EUW"),
        ("Upset", "EUW"),
    ]
    
    all_match_ids = set()
    all_participants = []
    
    print("=== SIMPLE COLLECTOR ===")
    
    # Schritt 1: PUUIDs holen
    puuids = []
    for name, tag in known_players:
        puuid = get_puuid_from_name(name, tag, api_key)
        if puuid:
            puuids.append(puuid)
            print(f"Found {name}#{tag}: {puuid[:8]}...")
        time.sleep(0.1)  # Rate limit respect
    
    if not puuids:
        print("ERROR: Konnte keine Spieler finden. API Key prüfen!")
        return pd.DataFrame()
    
    # Schritt 2: Match IDs sammeln
    print(f"\nCollecting matches from {len(puuids)} players...")
    for puuid in puuids:
        url = f"https://europe.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&count=20"
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            matches = r.json()
            all_match_ids.update(matches)
            print(f"  Found {len(matches)} matches")
        else:
            print(f"  Error {r.status_code} getting matches")
        
        time.sleep(0.05)  # Rate limit
        
        if len(all_match_ids) >= num_matches:
            break
    
    match_ids = list(all_match_ids)[:num_matches]
    print(f"\nProcessing {len(match_ids)} unique matches...")
    
    # Schritt 3: Match Details holen
    for i, match_id in enumerate(match_ids):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(match_ids)}")
        
        # Region aus Match ID
        prefix = match_id.split('_')[0]
        if prefix.startswith('EUW') or prefix.startswith('EUN'):
            region = 'europe'
        elif prefix.startswith('NA'):
            region = 'americas'
        elif prefix.startswith('KR'):
            region = 'asia'
        else:
            region = 'europe'  # Default
        
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            data = r.json()
            info = data.get('info', {})
            
            # Nur Ranked Solo/Duo
            if info.get('queueId') != 420:
                continue
            
            for p in info.get('participants', []):
                all_participants.append({
                    'matchId': match_id,
                    'puuid': p.get('puuid'),
                    'summonerName': p.get('summonerName'),
                    'championId': p.get('championId'),
                    'championName': p.get('championName'),
                    'win': p.get('win'),
                    'kills': p.get('kills'),
                    'deaths': p.get('deaths'),
                    'assists': p.get('assists'),
                    'teamPosition': p.get('teamPosition'),
                    'totalDamageDealt': p.get('totalDamageDealtToChampions'),
                    'goldEarned': p.get('goldEarned'),
                })
        
        time.sleep(0.05)  # Rate limit - WICHTIG!
    
    # Schritt 4: Rank Daten (vereinfacht)
    print(f"\nEnriching with rank data...")
    df = pd.DataFrame(all_participants)
    
    if df.empty:
        print("No data collected!")
        return df
    
    # Unique PUUIDs für Rank lookup
    unique_puuids = df['puuid'].unique()[:50]  # Limitiert für Test
    
    rank_data = {}
    for puuid in unique_puuids:
        url = f"https://euw1.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            entries = r.json()
            for entry in entries:
                if entry.get('queueType') == 'RANKED_SOLO_5x5':
                    rank_data[puuid] = {
                        'tier': entry.get('tier'),
                        'rank': entry.get('rank'),
                        'lp': entry.get('leaguePoints')
                    }
                    break
        
        time.sleep(0.05)
    
    # Rank Daten hinzufügen
    df['tier'] = df['puuid'].map(lambda x: rank_data.get(x, {}).get('tier'))
    df['rank'] = df['puuid'].map(lambda x: rank_data.get(x, {}).get('rank'))
    df['lp'] = df['puuid'].map(lambda x: rank_data.get(x, {}).get('lp'))
    
    return df

def main():
    api_key = "RGAPI-c4a287e8-aa5c-492b-aec4-099b4625af12"
    
    print("Testing API Key...")
    if not test_api_key(api_key):
        print("API Key scheint nicht zu funktionieren!")
        print("\nMögliche Lösungen:")
        print("1. Neuen API Key generieren: https://developer.riotgames.com/")
        print("2. Warten (Rate Limit)")
        print("3. VPN nutzen falls IP geblockt")
        return
    
    print("API Key works!\n")
    
    # Daten sammeln
    df = simple_collect(api_key, num_matches=50)
    
    if not df.empty:
        # Speichern
        output_path = Path("data/test/simple_test.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        
        print(f"\n=== SUCCESS ===")
        print(f"Collected {len(df)} participants")
        print(f"Unique matches: {df['matchId'].nunique()}")
        print(f"Rank coverage: {df['tier'].notna().mean():.1%}")
        print(f"Saved to: {output_path}")
        
        print(f"\nRank distribution:")
        print(df['tier'].value_counts())
    else:
        print("Collection failed!")

if __name__ == "__main__":
    main()