#!/usr/bin/env python3
"""
collect_match_ids_global.py

Ziel:
- Seeds (PUUIDs) aus allen Plattformen (euw1, na1, kr, ...) über league-exp-v4 sammeln
- nur gewünschte Ranks (z.B. SILVER/GOLD, Div IV/III/II/I)
- aus diesen Seeds Match-IDs per match-v5/by-puuid ziehen (Queue 420)
- Matches pro Routing-Region (europe/americas/asia/sea) balanciert sammeln

Wichtig:
- league-exp-Einträge enthalten direkt 'puuid' (kein summonerId).
"""

import os
import sys
import time
import random
import threading
from pathlib import Path
from typing import Dict, List

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from tqdm import tqdm


# ---------------- Routing & Plattformen ----------------

ROUTING_BY_PLATFORM: Dict[str, str] = {
    "euw1": "europe",
    "eun1": "europe",
    "tr1":  "europe",
    "ru":   "europe",
    "me1":  "europe",

    "na1":  "americas",
    "br1":  "americas",
    "la1":  "americas",
    "la2":  "americas",

    "oc1":  "sea",

    "kr":   "asia",
    "jp1":  "asia",
}

DEFAULT_PLATFORMS = list(ROUTING_BY_PLATFORM.keys())
DEFAULT_TIERS = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER"]
DEFAULT_DIVISIONS = ["IV", "III", "II", "I"]
DEFAULT_QUEUE = "RANKED_SOLO_5x5"
DEFAULT_QUEUE_ID = 420  # match-v5: Ranked Solo/Duo


# ---------------- Rate Limiter & HTTP ----------------

class RateLimiter:
    """Einfacher Rate-Limiter (thread-sicher)."""

    def __init__(self, rps: float):
        rps = max(float(rps), 1e-6)
        self.min_interval = 1.0 / rps
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            delta = now - self.last_call
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
                now = time.time()
            self.last_call = now


def http_get(
    url: str,
    headers: dict,
    limiter: RateLimiter,
    timeout: float = 10.0,
    retries: int = 3,
):
    """HTTP GET mit einfachem Retry + Rate-Limit."""
    last_err = None
    for i in range(retries + 1):
        limiter.wait()
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
        except Exception as e:
            last_err = f"exc:{type(e).__name__}"
            time.sleep(0.2 * (i + 1))
            continue

        if r.status_code == 200:
            try:
                return r.json(), None
            except Exception:
                return None, "json"

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                wait = float(ra)
            except (TypeError, ValueError):
                wait = 1.0
            time.sleep(wait)
            last_err = "429"
            continue

        if 500 <= r.status_code < 600:
            last_err = f"err:{r.status_code}"
            time.sleep(0.2 * (i + 1))
            continue

        # 4xx (außer 429): direkt abbrechen
        return None, f"err:{r.status_code}"

    return None, last_err or "err:unknown"


# ---------------- Seeds aus league-exp ----------------

def collect_seeds_for_platform(
    plat: str,
    queue: str,
    tiers: List[str],
    divisions: List[str],
    max_seeds_per_platform: int,
    rps_league: float,
    headers: dict,
) -> pd.DataFrame:
    """
    Holt Seeds aus *allen* (tier, division)-Kombinationen.

    Strategie:
    - Für jede (tier, div)-Kombi genau eine Seite (page=1) league-exp holen.
    - Aus dieser Page zufällig bis zu seeds_per_combo Seeds samplen.
    - Danach deduplizieren und ggf. auf max_seeds_per_platform runter-samplen.

    Rückgabe-DataFrame-Spalten:
        - platform
        - routing_region
        - puuid
        - tier
        - division
    """
    routing_region = ROUTING_BY_PLATFORM.get(plat, "unknown")
    limiter = RateLimiter(rps_league)

    rows = []
    req_count = 0

    print(f"[League] Plattform {plat}: sammle Seeds aus allen Tiers/Divisionen ...")

    # alle (Tier,Div)-Kombis – optional mischen, damit Reihenfolge nicht immer SILVERIV zuerst ist
    combos = [(t, d) for t in tiers for d in divisions]
    random.shuffle(combos)

    # Ziel-Samples pro Kombi (mindestens 1)
    n_combos = max(1, len(combos))
    seeds_per_combo = max(1, max_seeds_per_platform // n_combos)

    for tier, div in combos:
        url = (
            f"https://{plat}.api.riotgames.com/"
            f"lol/league-exp/v4/entries/{queue}/{tier}/{div}?page=1"
        )

        data, err = http_get(url, headers, limiter, timeout=8, retries=2)
        req_count += 1

        if err is not None:
            # 404 (kein Tier/Div auf der Plattform) oder ähnliches -> einfach nächste Kombi
            # print(f"[League] {plat} {tier}{div} page1: {err}")
            continue

        if not isinstance(data, list) or not data:
            continue

        # zufällige Auswahl aus dieser einen Page
        k = min(seeds_per_combo, len(data))
        picked = random.sample(data, k=k)

        print(
            f"[League] {plat} {tier}{div} page1: {len(data)} entries, "
            f"sampled {k} seeds"
        )

        for e in picked:
            puuid = e.get("puuid")
            if not puuid:
                continue
            rows.append(
                {
                    "platform": plat,
                    "routing_region": routing_region,
                    "puuid": puuid,
                    "tier": tier,
                    "division": div,
                }
            )

    if not rows:
        print(f"[League] {plat}: 0 Seeds gesammelt (Requests: {req_count})")
        return pd.DataFrame(
            columns=["platform", "routing_region", "puuid", "tier", "division"]
        )

    df = pd.DataFrame(rows)

    # Dedupe auf PUUID-Ebene pro Plattform
    df = df.drop_duplicates(subset=["platform", "puuid"])

    # Zufällig auf max_seeds_per_platform reduzieren
    if len(df) > max_seeds_per_platform:
        df = df.sample(n=max_seeds_per_platform, random_state=None).reset_index(drop=True)

    tier_counts = df["tier"].value_counts().to_dict()
    print(
        f"[League] {plat}: {len(df)} Seeds (Requests: {req_count}), "
        f"Tiers: {tier_counts}"
    )
    return df


# ---------------- Matches aus match-v5 ----------------

def fetch_matches_for_seed(
    puuid: str,
    routing_region: str,
    max_per_seed: int,
    queue_id: int,
    headers: dict,
    limiter: RateLimiter,
) -> List[str]:
    """Holt bis zu max_per_seed Match-IDs für einen PUUID (nur bestimmte Queue)."""
    url = (
        f"https://{routing_region}.api.riotgames.com/lol/match/v5/"
        f"matches/by-puuid/{puuid}/ids?queue={queue_id}&start=0&count={max_per_seed}"
    )
    data, err = http_get(url, headers, limiter)
    if err is not None or not isinstance(data, list):
        return []
    return [str(mid) for mid in data]


def collect_matches_for_region(
    region: str,
    seeds_region: pd.DataFrame,
    n_per_region: int,
    max_per_seed: int,
    rps_match: float,
    queue_id: int,
    headers: dict,
    workers: int,
) -> pd.DataFrame:
    """Sammelt Match-IDs für eine Routing-Region über alle Seeds dieser Region."""
    limiter = RateLimiter(rps_match)

    records = seeds_region.to_dict(orient="records")
    random.shuffle(records)

    rows = []
    seen_ids = set()

    if not records:
        print(f"[Match] Region {region}: keine Seeds, überspringe.")
        return pd.DataFrame(columns=["match_id", "routing_region"])

    print(f"[Match] Region {region}: starte Match-Sammlung mit {len(records)} Seeds.")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for rec in records:
            if len(seen_ids) >= n_per_region:
                break
            puuid = rec["puuid"]
            fut = ex.submit(
                fetch_matches_for_seed,
                puuid,
                region,
                max_per_seed,
                queue_id,
                headers,
                limiter,
            )
            futures[fut] = rec

        with tqdm(total=len(futures), desc=f"Matches {region}", unit="seed") as pbar:
            for fut in as_completed(futures):
                rec = futures[fut]
                mids = fut.result()
                for mid in mids:
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        rows.append(
                            {
                                "match_id": mid,
                                "routing_region": region,
                                "seed_platform": rec["platform"],
                                "seed_puuid": rec["puuid"],
                                "seed_tier": rec.get("tier"),
                                "seed_division": rec.get("division"),
                            }
                        )
                        if len(seen_ids) >= n_per_region:
                            break
                pbar.update(1)
                if len(seen_ids) >= n_per_region:
                    break

    print(f"[Match] Region {region}: {len(seen_ids)} unique match IDs gesammelt.")
    if not rows:
        return pd.DataFrame(columns=["match_id", "routing_region"])

    return pd.DataFrame(rows)


# ---------------- Main ----------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Ausgabedatei (Parquet) für alle Match-IDs.",
    )
    ap.add_argument(
        "--n-per-region",
        type=int,
        default=1000,
        help="Zielanzahl Match-IDs pro Routing-Region (europe/americas/asia/sea).",
    )
    ap.add_argument(
        "--max-per-seed",
        type=int,
        default=50,
        help="Maximale Anzahl Matches pro Seed-PUUID.",
    )
    ap.add_argument(
        "--max-seeds-per-platform",
        type=int,
        default=40,
        help="Maximale Anzahl Seeds (PUUIDs) pro Plattform (euw1, na1, ...).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=32,
        help="ThreadPool-Größe für parallele Requests.",
    )
    ap.add_argument(
        "--rps-league",
        type=float,
        default=0.5,
        help="RPS-Limit für league-exp-Requests pro Plattform.",
    )
    ap.add_argument(
        "--rps-summoner",
        type=float,
        default=1.0,
        help="(Derzeit ungenutzt, nur aus Kompatibilitätsgründen vorhanden.)",
    )
    ap.add_argument(
        "--rps-match",
        type=float,
        default=1.0,
        help="RPS-Limit für match-v5-Requests pro Routing-Region.",
    )
    ap.add_argument(
        "--tiers",
        type=str,
        default=",".join(DEFAULT_TIERS),
        help="Kommagetrennte Tiers (z.B. 'SILVER,GOLD,PLATINUM,EMERALD,DIAMOND,MASTER').",
    )
    ap.add_argument(
        "--divisions",
        type=str,
        default=",".join(DEFAULT_DIVISIONS),
        help="Kommagetrennte Divisionen (z.B. 'IV,III,II,I').",
    )
    ap.add_argument(
        "--platforms",
        type=str,
        default=",".join(DEFAULT_PLATFORMS),
        help="Kommagetrennte Plattformen (z.B. 'euw1,eun1,na1,kr').",
    )
    ap.add_argument(
        "--queue",
        type=str,
        default=DEFAULT_QUEUE,
        help="League-Queue für league-exp (z.B. RANKED_SOLO_5x5).",
    )
    ap.add_argument(
        "--queue-id",
        type=int,
        default=DEFAULT_QUEUE_ID,
        help="Queue-ID-Filter für match-v5/by-puuid (z.B. 420 für SoloQ).",
    )

    args = ap.parse_args()

    api_key = os.environ.get("RIOT_API_KEY")
    if not api_key:
        sys.exit("RIOT_API_KEY nicht gesetzt (export RIOT_API_KEY=...).")

    headers = {"X-Riot-Token": api_key}

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    platforms = [p for p in platforms if p in ROUTING_BY_PLATFORM]
    if not platforms:
        platforms = DEFAULT_PLATFORMS

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    divisions = [d.strip().upper() for d in args.divisions.split(",") if d.strip()]

    print(f"Plattformen: {platforms}")
    print(f"Tiers: {tiers}")
    print(f"Divisionen: {divisions}")
    print(f"match-v5 Queue-ID: {args.queue_id}")

    # -------- Seeds parallel über Plattformen sammeln --------
    print("\n[League] Sammle Seeds parallel über Plattformen ...")
    all_seed_dfs: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(args.workers, len(platforms))) as ex:
        futures = {
            ex.submit(
                collect_seeds_for_platform,
                plat,
                args.queue,
                tiers,
                divisions,
                args.max_seeds_per_platform,
                args.rps_league,
                headers,
            ): plat
            for plat in platforms
        }

        for fut in as_completed(futures):
            plat = futures[fut]
            try:
                df_seeds = fut.result()
            except Exception as e:
                print(f"[League] {plat}: Fehler beim Seed-Collect: {e}")
                continue
            if df_seeds is not None and not df_seeds.empty:
                all_seed_dfs.append(df_seeds)

    if not all_seed_dfs:
        print(
            "Keine Seeds aus league-/league-exp-Endpoints gesammelt – "
            "bitte API-Key / Produktrechte / Tiers/Divisions prüfen."
        )
        return

    seeds_df = pd.concat(all_seed_dfs, ignore_index=True)

    # Safety: routing_region evtl. füllen, falls leer
    if "routing_region" not in seeds_df.columns:
        seeds_df["routing_region"] = seeds_df["platform"].map(ROUTING_BY_PLATFORM)

    print("\nSeed-Zusammenfassung (Plattform x Region):")
    print(seeds_df.groupby(["platform", "routing_region"]).size())

    # -------- Matches pro Routing-Region sammeln --------
    all_match_dfs: List[pd.DataFrame] = []

    regions = sorted(seeds_df["routing_region"].unique())
    print("\n[Match] Sammle Matches parallel über Routing-Regionen ...")

    # Wie viele Worker bekommt jede Region intern für ihre Seeds
    workers_per_region = max(2, args.workers // max(1, len(regions)))

    from concurrent.futures import ThreadPoolExecutor, as_completed  # falls oben schon importiert, kannst du diese Zeile natürlich weglassen

    with ThreadPoolExecutor(max_workers=len(regions)) as ex:
        future_to_region = {}
        for region, seeds_region in seeds_df.groupby("routing_region"):
            fut = ex.submit(
                collect_matches_for_region,
                region=region,
                seeds_region=seeds_region,
                n_per_region=args.n_per_region,
                max_per_seed=args.max_per_seed,
                rps_match=args.rps_match,
                queue_id=args.queue_id,
                headers=headers,
                workers=workers_per_region,
            )
            future_to_region[fut] = region

        for fut in as_completed(future_to_region):
            region = future_to_region[fut]
            try:
                df_region = fut.result()
            except Exception as e:
                print(f"[Match] Region {region}: Fehler in collect_matches_for_region: {e}")
                continue
            if not df_region.empty:
                all_match_dfs.append(df_region)

    if not all_match_dfs:
        print("Keine Match-IDs gesammelt – evtl. Queue leer oder Fehler bei match-v5 Calls.")
        return

    out_df = pd.concat(all_match_dfs, ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["match_id"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"\nGespeichert: {out_path}")
    print(out_df["routing_region"].value_counts())



if __name__ == "__main__":
    main()
