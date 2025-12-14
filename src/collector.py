#!/usr/bin/env python3
# Streaming collector for LoL: downloads matches and, in parallel, fetches rank (by PUUID) and mastery (by PUUID+champ).
# - Interrupt-safe (Ctrl+C) with partial saves
# - Progress bars per stage
# - On-disk caches to avoid re-fetching
# - Tunable rate limits via CLI, chunked dataset writes for large runs
#
# Wichtig:
#   --n bedeutet: "bis zu n Matches pro Region (routing: europe/americas/asia/sea) **pro Zyklus**".
#
# Beispiel (kurzer Test-Run, moderat, ein Zyklus):
#   (venv) export RIOT_API_KEY=YOUR-KEY
#   (venv) python src/collector.py --n 250 --cycles 1 \
#     --write-mode dataset --dataset-dir data/interim/participants_stream_dataset \
#     --only-queue 420 --rps-match 1.7 --rps-league 0.8 --rps-mastery 0.8 \
#     --batch-size 200 --match-workers 32 --meta-workers 64 \
#     --linger-seconds 300
#
# Beispiel (endloser Run – viele Zyklen; Achtung: API-Limits!):
#   (venv) python src/collector.py --n 250 --cycles 0 \
#     --write-mode dataset --dataset-dir data/interim/participants_stream_dataset \
#     --only-queue 420 --rps-match 1.7 --rps-league 0.8 --rps-mastery 0.8 \
#     --batch-size 200 --match-workers 32 --meta-workers 64 \
#     --linger-seconds 300

import os
import sys
import json
import time
import signal
import threading
import random
from pathlib import Path
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import pandas as pd
import requests
from tqdm import tqdm

# ---------------- routes / helpers ----------------
REGIONAL_BY_PREFIX = {
    "EUW": "europe", "EUN": "europe", "TR": "europe", "RU": "europe", "ME": "europe",
    "NA": "americas", "BR": "americas", "LA": "americas", "OC": "sea",
    "KR": "asia", "JP": "asia",
}
PLATFORM_BY_PREFIX = {
    "EUW": "euw1", "EUN": "eun1", "TR": "tr1", "RU": "ru", "ME": "me1",
    "NA": "na1", "BR": "br1", "LA": "la1", "OC": "oc1",
    "KR": "kr", "JP": "jp1",
}


def _prefix(mid: str) -> str:
    head = mid.split("_", 1)[0]
    letters = "".join([c for c in head if c.isalpha()])
    return letters[:3].upper()


def _norm_platform(p: Optional[str], mid_prefix: str) -> str:
    return str(p).lower() if p else PLATFORM_BY_PREFIX.get(mid_prefix, "euw1")


def _platform_to_region(platform: str) -> str:
    letters = "".join(c for c in str(platform) if c.isalpha()).upper()
    pref = letters[:3]
    return REGIONAL_BY_PREFIX.get(pref, "unknown")


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------- rate limiter ----------------
class RateLimiter:
    """Einfacher Rate-Limiter über minimale Zeitabstände.

    rps = gewünschte Requests pro Sekunde (float).
    Beispiel:
      rps = 0.8  -> mindestens 1.25 Sekunden Abstand zwischen zwei Aufrufen
      rps = 10   -> mindestens 0.1 Sekunden Abstand
    """

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


# ---------------- global stop flag ----------------
stop_event = threading.Event()


def on_sigint(sig, frame):
    stop_event.set()
    print("\n[CTRL-C] Stop requested — finishing current ops and saving partial…", file=sys.stderr)


signal.signal(signal.SIGINT, on_sigint)


# ---------------- HTTP helper ----------------
def http_get(url: str, headers: dict, limiter: RateLimiter, timeout=6, retries=2):
    for i in range(retries + 1):
        if stop_event.is_set():
            return None, "stopped"
        limiter.wait()
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
        except Exception as e:
            if i == retries:
                return None, f"exc:{type(e).__name__}"
            time.sleep(0.2 * (i + 1))
            continue
        if r.status_code == 200:
            try:
                return r.json(), None
            except Exception:
                return None, "json"
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra else 0.5)
            continue
        if 500 <= r.status_code < 600:
            if i == retries:
                return None, f"err:{r.status_code}"
            time.sleep(0.2 * (i + 1))
            continue
        return None, f"err:{r.status_code}"
    return None, "err:retries"


# ---------------- caches (on disk) ----------------
def _load_kv(path: Path, cols: List[str]) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame(columns=cols)


def _save_unique(df: pd.DataFrame, path: Path, subset: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:
            pass
    df = df.drop_duplicates(subset=subset, keep="last")
    df.to_parquet(path, index=False)


# ---------------- write helper (single or dataset mode) ----------------
def write_output(df: pd.DataFrame, outp: Path, partial: bool, save_prefix: str,
                 write_mode: str, dataset_dir: str) -> Path:
    outp.parent.mkdir(parents=True, exist_ok=True)
    if write_mode == "single":
        out_final = (outp.parent / f"{save_prefix}_PARTIAL_{ts()}.parquet") if partial else outp
        df.to_parquet(out_final, index=False)
        return out_final
    # dataset mode (chunked)
    ds_dir = Path(dataset_dir)
    (ds_dir / "parts").mkdir(parents=True, exist_ok=True)
    seq = int(time.time() * 1000) % 10_000_000
    fname = f"{save_prefix}{'_PARTIAL' if partial else ''}_{ts()}_{seq}.parquet"
    out_final = ds_dir / "parts" / fname
    df.to_parquet(out_final, index=False)
    # append to manifest
    mani = ds_dir / "_manifest.csv"
    with open(mani, "a") as fh:
        fh.write(str(out_final) + "\n")
    return out_final


# ---------------- main pipeline (eine Runde) ----------------
def main(ids="data/interim/all_match_ids.parquet",
         n=2000,
         raw_dir="data/raw/matches",
         out="data/interim/participants_stream.parquet",
         save_prefix="participants_stream",
         scope="current",
         batch_size=300,
         max_workers_match=32,
         max_workers_meta=64,
         rps_match=0.5,
         rps_league=0.8,
         rps_mastery=0.8,
         write_mode="single",
         dataset_dir="data/interim/participants_stream_dataset",
         only_queue: Optional[int] = 420,
         linger_seconds: float = 120.0,
         rank_enabled: bool = True,
         skip_downloads: bool = False,
         cache_dir: str = "data/cache",
         extra_known_cache_dir: Optional[str] = None,
         mastery_enabled: bool = True,
         ):

    api_key = os.environ.get("RIOT_API_KEY")
    if not api_key:
        raise SystemExit("Missing RIOT_API_KEY")
    headers = {"X-Riot-Token": api_key}
    # Prepare dirs (brauchen wir gleich für "only new matches")
    rawp = Path(raw_dir)
    rawp.mkdir(parents=True, exist_ok=True)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load IDs und wähle bis zu n Matches PRO Region (routing: europe/americas/asia/sea)
    # ------------------------------------------------------------------
    ids_df = pd.read_parquet(ids)

    if "match_id" not in ids_df.columns:
        raise SystemExit(f"{ids} enthält keine Spalte 'match_id'.")

    ids_df["match_id"] = ids_df["match_id"].astype(str)

    # routing-Region:
    # - Falls das globale Match-ID-File 'routing_region' enthält, nutzen wir das.
    # - Sonst bestimmen wir die Region wie bisher aus dem Match-ID-Präfix (EUW, NA, ...).
    if "routing_region" in ids_df.columns:
        ids_df["region"] = ids_df["routing_region"].astype(str)
    else:
        ids_df["prefix"] = ids_df["match_id"].map(_prefix)
        ids_df["region"] = ids_df["prefix"].map(REGIONAL_BY_PREFIX)

    # Optional: nur "neue" Matches (für die es noch KEIN JSON im raw_dir gibt)
    if not skip_downloads:
        ids_df["json_exists"] = ids_df["match_id"].apply(
            lambda mid: (rawp / f"{mid}.json").exists()
        )
        before = len(ids_df)
        ids_df = ids_df[~ids_df["json_exists"]].copy()
        dropped = before - len(ids_df)
        print(f"Ignoriere {dropped} Match-IDs mit existierendem JSON (nur neue Matches).")

    usable = ids_df.dropna(subset=["region"])
    if usable.empty:
        raise SystemExit(f"No usable match IDs with known region in {ids}")

    region_groups = usable.groupby("region", sort=True)
    mids_by_region: Dict[str, List[str]] = {}
    target_per_region = Counter()  # wie viele Matches pro routing-Region in dieser Runde

    print(f"Lade Match-IDs aus {ids} – Ziel: bis zu {n} Matches pro Region (routing).")
    for region, g in region_groups:
        take = min(len(g), n)
        mids_region = g.head(take)["match_id"].tolist()
        mids_by_region[region] = mids_region
        target_per_region[region] = take
        print(f"  Region {region:8s}: {take:6d} von {len(g):6d} IDs ausgewählt")

    # IDs aus allen Regionen interleaven, damit alle Router gleichzeitig genutzt werden
    mids: List[str] = []
    if mids_by_region:
        max_len = max(len(lst) for lst in mids_by_region.values())
        regions_sorted = sorted(mids_by_region.keys())
        for i in range(max_len):
            for region in regions_sorted:
                lst = mids_by_region[region]
                if i < len(lst):
                    mids.append(lst[i])

    if not mids:
        raise SystemExit(f"No match IDs selected from {ids}")

    print(f"Gesamt ausgewählte Match-IDs (nur neue): {len(mids)}")

    # Caches (Root konfigurierbar)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    rank_cache_path = cache_root / "rank_by_puuid.parquet"
    mastery_cache_path = cache_root / "mastery_by_puuid.parquet"

    # Caches laden
    rank_df = _load_kv(
        rank_cache_path,
        ["puuid", "platform", "tier", "rank_div", "leaguePoints", "wins", "losses", "updated_ts"],
    )
    # Leere Rank-Zeilen ignorieren (werden neu abgefragt)
    if not rank_df.empty:
        rank_cols = ["tier", "rank_div", "leaguePoints", "wins", "losses"]
        mask_good = rank_df[rank_cols].notna().any(axis=1)
        bad = (~mask_good).sum()
        if bad > 0:
            print(f"Warnung: {bad} leere Rank-Cache-Zeilen werden ignoriert (werden neu abgefragt).")
        rank_df = rank_df[mask_good].copy()

    mastery_df = _load_kv(
        mastery_cache_path,
        ["puuid", "platform", "championId", "cm_level", "cm_points", "cm_lastPlayTime", "updated_ts"],
    )

    rank_known_keys = set(zip(rank_df["puuid"].astype(str), rank_df["platform"].astype(str)))
    mast_known_keys = set(
        zip(
            mastery_df["puuid"].astype(str),
            mastery_df["platform"].astype(str),
            mastery_df["championId"].astype("Int64"),
        )
    )

    # Optional: zusätzliches read-only Cache-Verzeichnis für bekannte Masteries
    if extra_known_cache_dir is not None:
        extra_root = Path(extra_known_cache_dir)
        extra_mastery_path = extra_root / "mastery_by_puuid.parquet"
        if extra_mastery_path.exists():
            try:
                extra_df = pd.read_parquet(extra_mastery_path)
                extra_keys = set(
                    zip(
                        extra_df["puuid"].astype(str),
                        extra_df["platform"].astype(str),
                        extra_df["championId"].astype("Int64"),
                    )
                )
                before = len(mast_known_keys)
                mast_known_keys |= extra_keys
                added = len(mast_known_keys) - before
                print(f"Extra-known mastery keys geladen aus {extra_mastery_path}: {added} neue Keys.")
            except Exception as e:
                print(f"Warnung: Konnte extra-known Cache {extra_mastery_path} nicht laden: {e}", file=sys.stderr)
        else:
            print(f"Hinweis: extra-known mastery Cache {extra_mastery_path} existiert nicht.")

    # In-memory stores (werden später auch in Caches geschrieben)
    rank_store: Dict[Tuple[str, str], dict] = {}
    mast_store: Dict[Tuple[str, str, int], dict] = {}

    # --- Stats für Rank/Mastery pro Plattform & Errors ---
    stats_lock = threading.Lock()
    platform_rank_total = Counter()   # wie viele Rank-Tasks pro Plattform enqueued
    platform_mast_total = Counter()   # wie viele Mastery-Tasks pro Plattform enqueued
    stats_rank_success = Counter()    # wie viele Rank-Requests pro Plattform erfolgreich
    stats_mast_success = Counter()    # wie viele Mastery-Requests pro Plattform erfolgreich
    stats_rank_error = Counter()      # Fehlercodes (err:429, err:500, exc:Timeout, ...)
    stats_mast_error = Counter()

    # Rate limiter pro Router
    # Match: pro REGIONAL router (europe / americas / asia / sea)
    match_limiters: Dict[str, RateLimiter] = {}

    def get_match_limiter(region: str) -> RateLimiter:
        if region not in match_limiters:
            match_limiters[region] = RateLimiter(rps_match)
        return match_limiters[region]

    # League & Mastery: pro Plattform-Router (euw1 / na1 / kr / ...)
    league_limiters: Dict[str, RateLimiter] = {} if rank_enabled else {}
    mastery_limiters: Dict[str, RateLimiter] = {}

    def get_league_limiter(platform: str) -> Optional[RateLimiter]:
        if not rank_enabled:
            return None
        if platform not in league_limiters:
            league_limiters[platform] = RateLimiter(rps_league)
        return league_limiters[platform]

    def get_mastery_limiter(platform: str) -> RateLimiter:
        if platform not in mastery_limiters:
            mastery_limiters[platform] = RateLimiter(rps_mastery)
        return mastery_limiters[platform]

    # Queues für Meta-Lookups
    rank_queue: deque[Tuple[str, str]] = deque()             # (puuid, platform)
    mast_queue: deque[Tuple[str, str, int]] = deque()        # (puuid, platform, championId)
    rank_lock = threading.Lock()
    mast_lock = threading.Lock()

    # -------------- WORKERS für LEAGUE & MASTERY --------------
    def worker_rank():
        nonlocal stats_rank_success, stats_rank_error
        while not stop_event.is_set():
            try:
                with rank_lock:
                    item = rank_queue.popleft() if rank_queue else None
                if item is None:
                    time.sleep(0.01)
                    continue

                puuid, platform = item
                key = (puuid, platform)
                if key in rank_store or key in rank_known_keys:
                    continue

                url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
                limiter = get_league_limiter(platform)
                if limiter is None:
                    continue

                data, err = http_get(url, headers, limiter)

                if err is None and isinstance(data, list):
                    # Erfolgreich – auch wenn Liste leer (unranked) ist
                    with stats_lock:
                        stats_rank_success[platform] += 1

                    if data:
                        solo = None
                        for e in data:
                            if str(e.get("queueType", "")).upper() == "RANKED_SOLO_5X5":
                                # prefer solo queue
                                solo = e
                                break
                        e = solo or data[0]
                        rank_store[key] = {
                            "puuid": puuid,
                            "platform": platform,
                            "tier": e.get("tier"),
                            "rank_div": e.get("rank"),
                            "leaguePoints": e.get("leaguePoints"),
                            "wins": e.get("wins"),
                            "losses": e.get("losses"),
                            "updated_ts": datetime.utcnow().isoformat(),
                        }
                    else:
                        # Spieler hat keine League-Einträge (unranked) -> explizit als unranked speichern
                        rank_store[key] = {
                            "puuid": puuid,
                            "platform": platform,
                            "tier": None,
                            "rank_div": None,
                            "leaguePoints": None,
                            "wins": None,
                            "losses": None,
                            "updated_ts": datetime.utcnow().isoformat(),
                        }
                else:
                    # Fehler -> NICHT in Rank-Cache speichern, damit späterer Run erneut versuchen kann
                    if err is not None:
                        with stats_lock:
                            stats_rank_error[err] += 1
                    continue

            except IndexError:
                time.sleep(0.01)

    def worker_mastery():
        nonlocal stats_mast_success, stats_mast_error
        while not stop_event.is_set():
            try:
                with mast_lock:
                    item = mast_queue.popleft() if mast_queue else None
                if item is None:
                    time.sleep(0.01)
                    continue
                puuid, platform, cid = item
                key = (puuid, platform, cid)
                if key in mast_store or key in mast_known_keys:
                    continue
                url = (
                    f"https://{platform}.api.riotgames.com/lol/champion-mastery/v4/"
                    f"champion-masteries/by-puuid/{puuid}/by-champion/{cid}"
                )
                limiter = get_mastery_limiter(platform)
                data, err = http_get(url, headers, limiter)
                if err is None and isinstance(data, dict):
                    with stats_lock:
                        stats_mast_success[platform] += 1
                    mast_store[key] = {
                        "puuid": puuid,
                        "platform": platform,
                        "championId": cid,
                        "cm_level": data.get("championLevel"),
                        "cm_points": data.get("championPoints"),
                        "cm_lastPlayTime": data.get("lastPlayTime"),
                        "updated_ts": datetime.utcnow().isoformat(),
                    }
                else:
                    # Fehler -> NICHT speichern; späterer Run versucht es erneut
                    if err is not None:
                        with stats_lock:
                            stats_mast_error[err] += 1
                    continue
            except IndexError:
                time.sleep(0.01)

    # Meta-Worker starten: Aufteilung nach RPS-Gewicht (league vs. mastery)
    if rank_enabled:
        league_weight = max(float(rps_league), 0.0)
    else:
        league_weight = 0.0

    # Wenn mastery_enabled=False -> keine Mastery-Worker / keine neuen Mastery-Calls
    if mastery_enabled:
        mastery_weight = max(float(rps_mastery), 0.0)
    else:
        mastery_weight = 0.0

    total_weight = league_weight + mastery_weight

    if total_weight <= 0:
        # Fallback: alles auf Mastery
        n_master = max_workers_meta
        n_rank = 0
    else:
        share_rank = league_weight / total_weight if rank_enabled else 0.0
        n_rank = int(max_workers_meta * share_rank)
        n_master = max_workers_meta - n_rank
        if n_master <= 0:
            n_master = 1
            n_rank = max_workers_meta - 1 if rank_enabled and max_workers_meta > 1 else 0

    print(f"Meta-Worker: mastery={n_master}, rank={n_rank}")

    if rank_enabled and n_rank > 0:
        for _ in range(n_rank):
            threading.Thread(target=worker_rank, daemon=True).start()
    else:
        n_rank = 0

    if mastery_enabled and n_master > 0:
        for _ in range(n_master):
            threading.Thread(target=worker_mastery, daemon=True).start()
    else:
        n_master = 0

    # -------------- DOWNLOAD + EXTRACT --------------
    was_interrupted = False

    def dl_match(mid: str) -> Tuple[str, str, str]:
        """Lädt ein Match-JSON (falls noch nicht vorhanden) und gibt (status, mid, region) zurück."""
        if stop_event.is_set():
            return "stopped", mid, "unknown"
        pref = _prefix(mid)
        region = REGIONAL_BY_PREFIX.get(pref, "europe")
        f = rawp / f"{mid}.json"
        if f.exists():
            return "cache", mid, region
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{mid}"
        limiter = get_match_limiter(region)
        data, err = http_get(url, headers, limiter, timeout=6, retries=2)
        if err is None and isinstance(data, dict):
            f.write_text(json.dumps(data), encoding="utf-8")
            return "ok", mid, region
        return (err or "fail"), mid, region

    # Step 1 + 2: Downloads (optional) & JSON-Auswahl
    if skip_downloads:
        print("Skip downloads: verwende nur vorhandene JSONs im raw_dir.")
        if scope == "all":
            json_files = list(rawp.glob("*.json"))
        else:
            json_files = [rawp / f"{mid}.json" for mid in mids if (rawp / f"{mid}.json").exists()]
        random.shuffle(json_files)
        print(f"JSONs to parse: {len(json_files)}")
    else:
        stats = Counter()
        downloads_per_region = Counter()
        with tqdm(total=len(mids), desc="Match downloads", unit="m") as pbar:
            for i in range(0, len(mids), batch_size):
                if stop_event.is_set():
                    break
                batch = mids[i:i + batch_size]
                with ThreadPoolExecutor(max_workers=max_workers_match) as ex:
                    futs = {ex.submit(dl_match, mid): mid for mid in batch}
                    try:
                        for fut in as_completed(futs):
                            tag, mid, region = fut.result()
                            stats[tag] += 1
                            if tag in ("ok", "cache"):
                                downloads_per_region[region] += 1
                            pbar.update(1)
                            # Per-Region Fortschritt im Postfix
                            postfix = {
                                reg: f"{downloads_per_region[reg]}/{target_per_region[reg]}"
                                for reg in sorted(target_per_region.keys())
                            }
                            pbar.set_postfix(postfix)
                            if stop_event.is_set():
                                ex.shutdown(wait=False, cancel_futures=True)
                                break
                    except KeyboardInterrupt:
                        stop_event.set()
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
        print("Download stats:", dict(stats))

        was_interrupted = stop_event.is_set()
        if was_interrupted:
            stop_event.clear()

        if scope == "all":
            json_files = list(rawp.glob("*.json"))
        else:
            json_files = [rawp / f"{mid}.json" for mid in mids if (rawp / f"{mid}.json").exists()]
        random.shuffle(json_files)
        print(f"JSONs to parse: {len(json_files)}")

    if not json_files:
        out_final = write_output(
            pd.DataFrame(), outp, partial=True, save_prefix=save_prefix,
            write_mode=write_mode, dataset_dir=dataset_dir,
        )
        print(f"No JSONs to parse; wrote empty partial → {out_final}")
        return

    # Step 3: Participants extrahieren & Meta-Queues füllen
    rows: List[dict] = []
    kept = 0
    with tqdm(total=len(json_files), desc="Extract participants", unit="f") as pbar:
        for fp in json_files:
            if stop_event.is_set():
                break
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pbar.update(1)
                continue
            info = obj.get("info", {})
            meta = obj.get("metadata", {})
            mid = str(meta.get("matchId") or fp.stem)
            pref = _prefix(mid)
            platform = _norm_platform(info.get("platformId"), pref)

            # SoloQ-Filter
            if only_queue is not None and info.get("queueId") != only_queue:
                pbar.update(1)
                continue

            parts = info.get("participants", []) or []
            for p in parts:
                puuid = p.get("puuid")
                cid = p.get("championId")
                rows.append(
                    {
                        "matchId": mid,
                        "platform": platform,
                        "queueId": info.get("queueId"),
                        "gameVersion": info.get("gameVersion"),
                        "gameDuration": info.get("gameDuration"),
                        "puuid": puuid,
                        "summonerName": p.get("summonerName"),
                        "teamId": p.get("teamId"),
                        "teamPosition": p.get("teamPosition"),
                        "championId": cid,
                        "championName": p.get("championName"),
                        "win": bool(p.get("win")),
                    }
                )
                # Mastery (Priorität) – nur wenn aktiviert
                if mastery_enabled and puuid is not None and cid is not None:
                    keym = (str(puuid), platform, int(cid))
                    if keym not in mast_known_keys:
                        with mast_lock:
                            mast_queue.append(keym)
                            platform_mast_total[platform] += 1

                # Rank (optional)
                if rank_enabled and puuid:
                    key = (str(puuid), platform)
                    if key not in rank_known_keys:
                        with rank_lock:
                            rank_queue.append(key)
                            platform_rank_total[platform] += 1
            kept += 1
            pbar.update(1)
    df = pd.DataFrame(rows)
    print(f"Participants parsed: {df.shape}  (matches kept: {kept})")

    if df.empty:
        out_final = write_output(
            df, outp, partial=True, save_prefix=save_prefix,
            write_mode=write_mode, dataset_dir=dataset_dir,
        )
        print(f"\nSaved → {out_final}  (empty partial; interrupted or filtered away)")
        return

    # Meta-Queue-Größen messen
    with mast_lock:
        initial_mast_pending = len(mast_queue)
    with rank_lock:
        initial_rank_pending = len(rank_queue) if rank_enabled else 0

    print(
        f"Meta tasks enqueued: mastery={initial_mast_pending}, "
        f"rank={initial_rank_pending}, total={initial_mast_pending + initial_rank_pending}"
    )

    if platform_mast_total:
        print("\nMastery tasks per platform (enqueued):")
        for plat, total in sorted(platform_mast_total.items()):
            reg = _platform_to_region(plat)
            print(f"  {plat:5s} [{reg:8s}] : {total:6d}")
    if platform_rank_total:
        print("\nRank tasks per platform (enqueued):")
        for plat, total in sorted(platform_rank_total.items()):
            reg = _platform_to_region(plat)
            print(f"  {plat:5s} [{reg:8s}] : {total:6d}")

    # Step 4: warten, bis ~95 % der Meta-Tasks erledigt oder Zeitlimit
    target_frac_mast = 0.95
    target_frac_rank = 0.95

    max_wait = float(linger_seconds)
    if max_wait <= 0:
        t_end = float("inf")
    else:
        t_end = time.time() + max_wait

    # ---------- Fall 1: Mastery aktiviert -> wie bisher auf Mastery warten ----------
    if mastery_enabled and initial_mast_pending > 0:
        print(
            f"Warte, bis {target_frac_mast*100:.1f}% der Mastery-Tasks verarbeitet sind "
            f"oder max. {max_wait/3600:.2f} Stunden erreicht sind ..."
        )

        processed_prev_mast = 0

        with tqdm(
            desc="Finalizing mastery (target 95%)",
            total=initial_mast_pending,
            unit="m_task",
        ) as pbar:
            while not stop_event.is_set():
                now = time.time()
                with mast_lock:
                    mq = len(mast_queue)
                if rank_enabled:
                    with rank_lock:
                        rq = len(rank_queue)
                else:
                    rq = 0

                processed_mast = max(initial_mast_pending - mq, 0)
                processed_rank = max(initial_rank_pending - rq, 0)

                frac_mast = (
                    processed_mast / initial_mast_pending if initial_mast_pending > 0 else 1.0
                )

                delta_mast = max(0, processed_mast - processed_prev_mast)
                if delta_mast > 0:
                    pbar.update(delta_mast)
                    processed_prev_mast = processed_mast

                # Region-Statistik für Mastery & Rank
                region_mast_total = Counter()
                region_mast_done = Counter()
                region_rank_total = Counter()
                region_rank_done = Counter()
                with stats_lock:
                    for plat, total in platform_mast_total.items():
                        reg = _platform_to_region(plat)
                        region_mast_total[reg] += total
                        region_mast_done[reg] += stats_mast_success.get(plat, 0)
                    for plat, total in platform_rank_total.items():
                        reg = _platform_to_region(plat)
                        region_rank_total[reg] += total
                        region_rank_done[reg] += stats_rank_success.get(plat, 0)

                postfix = {
                    "mast_done": processed_mast,
                    "mast_total": initial_mast_pending,
                    "mast_frac": f"{frac_mast:.3f}",
                    "rank_done": processed_rank,
                    "rank_total": initial_rank_pending,
                    "rank_q": rq,
                }
                # pro Region kurze Strings (z.B. mast_europe: 123/456)
                for reg in sorted(region_mast_total.keys()):
                    tot = region_mast_total[reg]
                    done = region_mast_done.get(reg, 0)
                    postfix[f"mast_{reg}"] = f"{done}/{tot}"
                for reg in sorted(region_rank_total.keys()):
                    tot = region_rank_total[reg]
                    done = region_rank_done.get(reg, 0)
                    postfix[f"rank_{reg}"] = f"{done}/{tot}"

                pbar.set_postfix(postfix)

                if frac_mast >= target_frac_mast:
                    print("Ziel 95% Mastery erreicht.")
                    break

                if now >= t_end:
                    print("Maximale Wartezeit erreicht, bevor 95% Mastery vollständig waren.")
                    break

                time.sleep(0.2)

    # ---------- Fall 2: Mastery deaktiviert, aber Rank aktiv -> auf Rank warten ----------
    elif (not mastery_enabled) and rank_enabled and initial_rank_pending > 0:
        print(
            f"Mastery-Lookups sind deaktiviert – warte auf Rank-Tasks "
            f"bis {target_frac_rank*100:.1f}% erledigt sind oder max. {max_wait/3600:.2f} Stunden."
        )

        processed_prev_rank = 0

        with tqdm(
            desc="Finalizing rank (target 95%)",
            total=initial_rank_pending,
            unit="r_task",
        ) as pbar:
            while not stop_event.is_set():
                now = time.time()
                with rank_lock:
                    rq = len(rank_queue)

                processed_rank = max(initial_rank_pending - rq, 0)
                frac_rank = (
                    processed_rank / initial_rank_pending if initial_rank_pending > 0 else 1.0
                )

                delta_rank = max(0, processed_rank - processed_prev_rank)
                if delta_rank > 0:
                    pbar.update(delta_rank)
                    processed_prev_rank = processed_rank

                # Region-Statistik Rank
                region_rank_total = Counter()
                region_rank_done = Counter()
                with stats_lock:
                    for plat, total in platform_rank_total.items():
                        reg = _platform_to_region(plat)
                        region_rank_total[reg] += total
                        region_rank_done[reg] += stats_rank_success.get(plat, 0)

                postfix = {
                    "rank_done": processed_rank,
                    "rank_total": initial_rank_pending,
                    "rank_frac": f"{frac_rank:.3f}",
                    "rank_q": rq,
                }
                for reg in sorted(region_rank_total.keys()):
                    tot = region_rank_total[reg]
                    done = region_rank_done.get(reg, 0)
                    postfix[f"rank_{reg}"] = f"{done}/{tot}"

                pbar.set_postfix(postfix)

                if frac_rank >= target_frac_rank:
                    print("Ziel 95% Rank erreicht.")
                    break

                if now >= t_end:
                    print("Maximale Wartezeit erreicht, bevor 95% Rank vollständig waren.")
                    break

                time.sleep(0.2)

    else:
        print("Keine Mastery-/Rank-Tasks enqueued – nichts zu warten.")

    # Step 5: Caches materialisieren und joinen
    if rank_enabled and rank_store:
        rdf = pd.DataFrame(rank_store.values())
        _save_unique(rdf, rank_cache_path, subset=["puuid", "platform"])
        rank_known_keys |= set(zip(rdf["puuid"].astype(str), rdf["platform"].astype(str)))

    if mastery_enabled and mast_store:
        mdf = pd.DataFrame(mast_store.values())
        _save_unique(mdf, mastery_cache_path, subset=["puuid", "platform", "championId"])
        mast_known_keys |= set(
            zip(
                mdf["puuid"].astype(str),
                mdf["platform"].astype(str),
                mdf["championId"].astype("Int64", errors="ignore"),
            )
        )

    # Caches zum Join laden
    if rank_enabled:
        rank_join = _load_kv(
            rank_cache_path,
            ["puuid", "platform", "tier", "rank_div", "leaguePoints", "wins", "losses", "updated_ts"],
        )
        rank_join = rank_join[
            ["puuid", "platform", "tier", "rank_div", "leaguePoints", "wins", "losses"]
        ].rename(columns={"tier": "rank_tier"})
    else:
        rank_join = None

    mast_join = _load_kv(
        mastery_cache_path,
        ["puuid", "platform", "championId", "cm_level", "cm_points", "cm_lastPlayTime", "updated_ts"],
    )
    mast_join = mast_join[
        ["puuid", "platform", "championId", "cm_level", "cm_points", "cm_lastPlayTime"]
    ]

    if rank_enabled and rank_join is not None and not rank_join.empty:
        df = df.merge(rank_join, on=["puuid", "platform"], how="left")
    else:
        df["rank_tier"] = None
        df["rank_div"] = None
        df["leaguePoints"] = None
        df["wins"] = None
        df["losses"] = None

    df = df.merge(mast_join, on=["puuid", "platform", "championId"], how="left")

    # Step 6: speichern
    partial = was_interrupted or stop_event.is_set()
    out_final = write_output(
        df,
        outp,
        partial=partial,
        save_prefix=save_prefix,
        write_mode=write_mode,
        dataset_dir=dataset_dir,
    )
    print(f"\nSaved → {out_final}  shape={df.shape}")

    cov_rank = df[["rank_tier", "rank_div", "leaguePoints"]].notna().mean().round(3).to_dict()
    cov_mast = df[["cm_level", "cm_points"]].notna().mean().round(3).to_dict()
    print("Coverage:")
    print("  Rank  :", cov_rank)
    print("  Master:", cov_mast)

    # ---- Zusätzliche Übersicht: Rank/Mastery-Fortschritt pro Plattform & Region ----
    if platform_rank_total:
        print("\nRank progress per platform:")
        for plat, total in sorted(platform_rank_total.items()):
            done = stats_rank_success.get(plat, 0)
            frac = done / total if total > 0 else 0.0
            reg = _platform_to_region(plat)
            print(f"  {plat:5s} [{reg:8s}] : {done:6d}/{total:6d} ({frac:5.1%})")

        region_rank_total = Counter()
        region_rank_done = Counter()
        for plat, total in platform_rank_total.items():
            reg = _platform_to_region(plat)
            region_rank_total[reg] += total
            region_rank_done[reg] += stats_rank_success.get(plat, 0)
        print("\nRank progress per region:")
        for reg, total in sorted(region_rank_total.items()):
            done = region_rank_done.get(reg, 0)
            frac = done / total if total > 0 else 0.0
            print(f"  {reg:8s}: {done:6d}/{total:6d} ({frac:5.1%})")

    if platform_mast_total:
        print("\nMastery progress per platform:")
        for plat, total in sorted(platform_mast_total.items()):
            done = stats_mast_success.get(plat, 0)
            frac = done / total if total > 0 else 0.0
            reg = _platform_to_region(plat)
            print(f"  {plat:5s} [{reg:8s}] : {done:6d}/{total:6d} ({frac:5.1%})")

        region_mast_total = Counter()
        region_mast_done = Counter()
        for plat, total in platform_mast_total.items():
            reg = _platform_to_region(plat)
            region_mast_total[reg] += total
            region_mast_done[reg] += stats_mast_success.get(plat, 0)
        print("\nMastery progress per region:")
        for reg, total in sorted(region_mast_total.items()):
            done = region_mast_done.get(reg, 0)
            frac = done / total if total > 0 else 0.0
            print(f"  {reg:8s}: {done:6d}/{total:6d} ({frac:5.1%})")

    if stats_rank_error:
        print("\nRank error stats (err_code: count):")
        for err, cnt in stats_rank_error.most_common():
            print(f"  {err:10s}: {cnt}")

    if stats_mast_error:
        print("\nMastery error stats (err_code: count):")
        for err, cnt in stats_mast_error.most_common():
            print(f"  {err:10s}: {cnt}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str, default="data/interim/all_match_ids.parquet")
    ap.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Zielanzahl Matches pro Region (routing: europe/americas/asia/sea) **pro Zyklus**.",
    )
    ap.add_argument("--raw-dir", type=str, default="data/raw/matches")
    ap.add_argument("--out", type=str, default="data/interim/participants_stream.parquet")
    ap.add_argument("--save-prefix", type=str, default="participants_stream")
    ap.add_argument("--scope", type=str, choices=["current", "all"], default="current")
    ap.add_argument("--batch-size", type=int, default=300)
    ap.add_argument("--match-workers", type=int, default=32)
    ap.add_argument("--meta-workers", type=int, default=64)
    ap.add_argument(
        "--rps-match",
        type=float,
        default=0.5,
        help="Match-v5 RPS pro Regional-Router (europe/americas/asia/sea). "
             "Beispiel ~1.7 ≈ 200 Requests alle 2 Minuten.",
    )
    ap.add_argument(
        "--rps-league",
        type=float,
        default=0.8,
        help="League-v4 RPS pro Plattform-Router (euw1/na1/kr/...).",
    )
    ap.add_argument(
        "--rps-mastery",
        type=float,
        default=0.8,
        help="Mastery-v4 RPS pro Plattform-Router (euw1/na1/kr/...).",
    )
    ap.add_argument(
        "--write-mode",
        type=str,
        choices=["single", "dataset"],
        default="single",
    )
    ap.add_argument("--dataset-dir", type=str, default="data/interim/participants_stream_dataset")
    ap.add_argument(
        "--only-queue",
        type=int,
        default=420,
        help="Nur diese Queue-ID sammeln (z.B. 420 für Ranked Solo/Duo).",
    )
    ap.add_argument(
        "--linger-seconds",
        type=float,
        default=120.0,
        help="Maximale Wartezeit in Sekunden in der Meta-Phase (Mastery/Rank).",
    )
    ap.add_argument(
        "--disable-rank",
        action="store_true",
        help="Keine Rank-Lookups (nur Mastery).",
    )
    ap.add_argument(
        "--disable-mastery",
        action="store_true",
        help="Keine Mastery-Lookups (nur Rank).",
    )
    ap.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Keine neuen Matches laden, nur vorhandene JSONs im raw_dir verwenden.",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Verzeichnis für Rank/Mastery-Caches.",
    )
    ap.add_argument(
        "--extra-known-cache-dir",
        type=str,
        default=None,
        help=(
            "Optionales zusätzliches Cache-Verzeichnis, dessen Mastery-Keys "
            "als bereits bekannt gelten (read-only)."
        ),
    )
    ap.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Wie oft der Collector hintereinander laufen soll. 1 = einmalig (Default), "
             "0 = unendlich viele Zyklen.",
    )
    ap.add_argument(
        "--sleep-between-cycles",
        type=float,
        default=0.0,
        help="Pause in Sekunden zwischen zwei Zyklen (z.B. 120 für 2 Minuten).",
    )

    args = ap.parse_args()

    cycle = 0
    while True:
        cycle += 1
        print(f"\n==================== Collector cycle {cycle} ====================")
        main(
            ids=args.ids,
            n=args.n,
            raw_dir=args.raw_dir,
            out=args.out,
            save_prefix=args.save_prefix,
            scope=args.scope,
            batch_size=args.batch_size,
            max_workers_match=args.match_workers,
            max_workers_meta=args.meta_workers,
            rps_match=args.rps_match,
            rps_league=args.rps_league,
            rps_mastery=args.rps_mastery,
            write_mode=args.write_mode,
            dataset_dir=args.dataset_dir,
            only_queue=args.only_queue,
            linger_seconds=args.linger_seconds,
            rank_enabled=not args.disable_rank,
            skip_downloads=args.skip_downloads,
            cache_dir=args.cache_dir,
            extra_known_cache_dir=args.extra_known_cache_dir,
            mastery_enabled=not args.disable_mastery,
        )

        if args.cycles > 0 and cycle >= args.cycles:
            break
        if args.sleep_between_cycles > 0:
            time.sleep(args.sleep_between_cycles)
        if stop_event.is_set():
            # Falls während eines Zyklus CTRL-C gedrückt wurde, komplett raus.
            break
