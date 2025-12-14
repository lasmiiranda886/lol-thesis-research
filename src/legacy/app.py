    # app.py
    # ------------------------------------------------------------
    # LoL Ranked (Queue 420) Datenpipeline (Silber–Master),
    # robust für entries mit PUUID-only (keine Summoner-ID nötig).
    # ------------------------------------------------------------
    import os
    import time
    import random
    import threading
    from typing import List, Any, Optional, Tuple
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    import pandas as pd
    from tqdm import tqdm

    # ---------------------------
    # Konfiguration
    # ---------------------------
    API_KEY = os.getenv("RIOT_API_KEY")
    assert API_KEY, "Bitte die Umgebungsvariable RIOT_API_KEY setzen!"

    PLATFORM = "euw1"          # v4-APIs (summoner/league/mastery)
    REGION_ROUTING = "europe"  # v5-APIs (match)

    TIERS = ["SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND"]
    DIVISIONS = ["I", "II", "III", "IV"]

    INCLUDE_MASTER = False
    MASTER_SAMPLE_FRAC = 0.30

    TARGET_SUMMONERS_PER_BUCKET = 20
    MAX_MATCH_IDS_PER_PUUID = 30
    MIN_YEAR = 2023
    SAVE_DIR = "./data_lol"
    os.makedirs(SAVE_DIR, exist_ok=True)

    MAX_WORKERS = 8
    REQUEST_TIMEOUT = 20
    REQS_PER_SEC = 12

    _bucket = {"tokens": REQS_PER_SEC, "last": time.time()}
    _lock = threading.Lock()

    RUN_DEBUG_PROBE = True
    USE_BOOTSTRAP = False
    BOOTSTRAP_SUMMONER_NAME = "Faker"

    # ---------------------------
    # Rate limit & HTTP helper
    # ---------------------------
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

    def riot_get(url: str, params: Optional[dict] = None, retry: int = 5) -> Any:
        params = params or {}
        headers = {"X-Riot-Token": API_KEY}
        for attempt in range(retry):
            rate_limit()
            try:
                r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 2 ** attempt))); continue
                if 500 <= r.status_code < 600:
                    time.sleep(2 ** attempt); continue
                r.raise_for_status()
            except requests.RequestException:
                if attempt == retry - 1:
                    raise
                time.sleep(2 ** attempt)
        return None

    # ---------------------------
    # Debug-Probe (optional)
    # ---------------------------
    def _debug_probe():
        url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/SILVER/IV"
        headers = {"X-Riot-Token": API_KEY}
        try:
            r = requests.get(url, params={"page": 1}, headers=headers, timeout=REQUEST_TIMEOUT)
            print("[DEBUG] entries status:", r.status_code)
            print("[DEBUG] content-type:", r.headers.get("Content-Type", ""))
            try:
                j = r.json()
                if isinstance(j, list):
                    print("[DEBUG] entries type: list, len:", len(j))
                    if j and isinstance(j[0], dict):
                        print("[DEBUG] sample keys:", list(j[0].keys())[:8])
                else:
                    print("[DEBUG] entries type:", type(j).__name__, "status obj:", j.get("status", {}))
            except Exception:
                print("[DEBUG] text snippet:", r.text[:240])
        except Exception as e:
            print("[DEBUG] probe error:", repr(e))

    # ---------------------------
    # Endpunkte
    # ---------------------------
    def league_entries(queue: str, tier: str, division: str, page: int = 1) -> List[dict]:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier}/{division}"
        data = riot_get(url, params={"page": page}) or []
        if isinstance(data, dict):
            print(f"[WARN] league/v4/entries error:", data.get("status", {}))
            return []
        # akzeptiere Einträge mit PUUID und/oder SummonerId
        return [e for e in data if isinstance(e, dict) and ("puuid" in e or "summonerId" in e)]

    def league_entries_exp(queue: str, tier: str, division: str, page: int = 1) -> List[dict]:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}"
        data = riot_get(url, params={"page": page}) or []
        if isinstance(data, dict):
            print(f"[WARN] league-exp/v4/entries error:", data.get("status", {}))
            return []
        return [e for e in data if isinstance(e, dict) and ("puuid" in e or "summonerId" in e)]

    def league_master_list(queue: str) -> dict:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/{queue}"
        return riot_get(url) or {}

    def summoner_by_name(name: str) -> Optional[dict]:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{name}"
        return riot_get(url)

    def summoner_by_id(encrypted_summoner_id: str) -> Optional[dict]:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/summoner/v4/summoners/{encrypted_summoner_id}"
        return riot_get(url)

    def summoner_by_puuid(puuid: str) -> Optional[dict]:
        url = f"https://{PLATFORM}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return riot_get(url)

    # Champion Mastery – **direkt per PUUID** (Summoner-ID deprecated/entfernt)
    def mastery_by_puuid(puuid: str) -> List[dict]:
        # Doku/Hinweise der Riot DevRel: by-PUUID wird unterstützt (Summoner-ID deprecated) :contentReference[oaicite:1]{index=1}
        url = f"https://{PLATFORM}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}"
        return riot_get(url) or []

    # Match-V5
    def match_ids_by_puuid(puuid: str, start: int = 0, count: int = 100, queue: Optional[int] = 420) -> List[str]:
        params = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        return riot_get(url, params=params) or []

    def match_by_id(match_id: str) -> Optional[dict]:
        url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return riot_get(url)

    def timeline_by_id(match_id: str) -> Optional[dict]:
        url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return riot_get(url)

    # ---------------------------
    # 1) Spieler-Sampling (PUUID-first)
    # ---------------------------
    def sample_summoners(queue: str = "RANKED_SOLO_5x5") -> pd.DataFrame:
        rows = []
        for tier in TIERS:
            for division in DIVISIONS:
                page, bucket = 1, []
                while len(bucket) < TARGET_SUMMONERS_PER_BUCKET:
                    entries = league_entries(queue, tier, division, page=page)
                    if not entries:
                        alt = league_entries_exp(queue, tier, division, page=page)
                        if not alt:
                            break
                        entries = alt
                    bucket.extend(entries); page += 1

                if not bucket:
                    print(f"[INFO] Keine Entries für {tier} {division} erhalten (Page 1 leer) – weiter.")
                    continue

                random.shuffle(bucket)
                bucket = bucket[:TARGET_SUMMONERS_PER_BUCKET]

                for e in bucket:
                    rows.append({
                        "tier": e.get("tier", tier),
                        "division": e.get("rank", division),
                        "puuid": e.get("puuid"),
                        "summonerId": e.get("summonerId"),  # kann fehlen
                        "leaguePoints": e.get("leaguePoints"),
                        "wins": e.get("wins"),
                        "losses": e.get("losses"),
                    })

        if INCLUDE_MASTER:
            ml = league_master_list(queue) or {}
            entries = ml.get("entries", [])
            if isinstance(entries, list) and entries:
                random.shuffle(entries)
                for e in entries[: max(1, int(len(entries) * MASTER_SAMPLE_FRAC))]:
                    rows.append({
                        "tier": "MASTER", "division": None,
                        "puuid": None, "summonerId": e.get("summonerId"),
                        "leaguePoints": e.get("leaguePoints"),
                        "wins": e.get("wins"), "losses": e.get("losses"),
                    })
            else:
                print("[WARN] master_list['entries'] leer/ungültig – Master übersprungen.")

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("Keine gültigen Summoner-Einträge erhalten (prüfe API-Key/Rate-Limit/Region).")

        # Wenn möglich: PUUID priorisieren, sonst versuchen, aus summonerId zu resolven (best effort)
        if df["puuid"].isna().any():
            needing = df[df["puuid"].isna() & df["summonerId"].notna()].copy()
            if not needing.empty:
                print(f"[INFO] Resolving {len(needing)} fehlende PUUIDs via summoner-v4 …")
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = {ex.submit(summoner_by_id, sid): sid for sid in needing["summonerId"].tolist()}
                    results = {}
                    for fut in tqdm(as_completed(futs), total=len(futs), desc="resolve puuid"):
                        data = fut.result()
                        if data and "puuid" in data:
                            results[futs[fut]] = data["puuid"]
                if results:
                    df.loc[df["puuid"].isna() & df["summonerId"].isin(results.keys()), "puuid"] = \
                        df.loc[df["puuid"].isna() & df["summonerId"].isin(results.keys()), "summonerId"].map(results)

        # am Ende nur Zeilen mit PUUID behalten (für Match-V5/Mastery)
        df = df[df["puuid"].notna()].copy()
        before = len(df)
        df = df.drop_duplicates(subset=["puuid"])
        print(f"[INFO] Gesampelte Summoner: {before} (nach Deduplikation nach PUUID: {len(df)})")
        return df

    def bootstrap_from_summoner_name(name: str) -> pd.DataFrame:
        s = summoner_by_name(name)
        if not s:
            raise RuntimeError(f"Summoner '{name}' nicht gefunden (PLATFORM={PLATFORM}).")
        return pd.DataFrame([{
            "tier": None, "division": None,
            "puuid": s["puuid"], "summonerId": s.get("id"),
            "leaguePoints": None, "wins": None, "losses": None
        }])

    # ---------------------------
    # 2) Match-IDs & Matches
    # ---------------------------
    def fetch_match_ids_for_puuid(puuid: str, max_ids: int = MAX_MATCH_IDS_PER_PUUID) -> List[str]:
        ids, start, batch = [], 0, 100
        while len(ids) < max_ids:
            got = match_ids_by_puuid(puuid, start=start, count=batch, queue=420)
            if not got: break
            ids.extend(got); start += batch
            if len(got) < batch: break
        return ids[:max_ids]

    def harvest_match_ids(df_summoners: pd.DataFrame) -> pd.DataFrame:
        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_match_ids_for_puuid, puuid): puuid for puuid in df_summoners["puuid"].tolist()}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Match IDs"):
                try:
                    ids = fut.result()
                    for mid in ids:
                        rows.append({"puuid": futs[fut], "matchId": mid})
                except Exception:
                    pass
        df = pd.DataFrame(rows).drop_duplicates(subset=["matchId"])
        if df.empty:
            print("[WARN] Keine Match-IDs geholt – evtl. zu wenige Summoner oder leere History.")
        return df

    def match_fetch_full(match_id: str, with_timeline: bool = False) -> Optional[Tuple[dict, Optional[dict]]]:
        m = match_by_id(match_id)
        if not m or not isinstance(m, dict) or "metadata" not in m:
            return None
        t = timeline_by_id(match_id) if with_timeline else None
        return m, t

    def harvest_matches(df_match_ids: pd.DataFrame, with_timeline: bool = False) -> pd.DataFrame:
        rows = []
        mids = df_match_ids["matchId"].tolist() if not df_match_ids.empty else []
        if not mids:
            return pd.DataFrame(rows)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(match_fetch_full, mid, with_timeline): mid for mid in mids}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Matches"):
                try:
                    res = fut.result()
                    if not res: continue
                    match, timeline = res
                    rows.append({"matchId": futs[fut], "match": match, "timeline": timeline})
                except Exception:
                    pass
        return pd.DataFrame(rows)

    # ---------------------------
    # 3) Feature-Engineering
    # ---------------------------
    LANE_ORDER = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

    def extract_participants_table(match_obj: dict) -> pd.DataFrame:
        info = match_obj.get("info", {})
        game_end_ts = info.get("gameEndTimestamp") or info.get("gameCreation")
        game_year = pd.to_datetime(game_end_ts, unit="ms", utc=True).year if game_end_ts else None
        parts = info.get("participants", [])
        rows = []
        for p in parts:
            rows.append({
                "matchId": match_obj.get("metadata", {}).get("matchId"),
                "gameCreation": info.get("gameCreation"),
                "gameDuration": info.get("gameDuration"),
                "gameVersion": info.get("gameVersion"),
                "queueId": info.get("queueId"),
                "game_year": game_year,
                "teamId": p.get("teamId"),
                "win": bool(p.get("win")),
                "puuid": p.get("puuid"),
                "summonerId": p.get("summonerId"),
                "championId": p.get("championId"),
                "championName": p.get("championName"),
                "teamPosition": p.get("teamPosition") or p.get("individualPosition"),
                "kills": p.get("kills"), "deaths": p.get("deaths"), "assists": p.get("assists"),
                # Platzhalter – echte Early-Lane-Stats ggf. aus Timeline ableiten
                "goldDiffAt10": p.get("challenges", {}).get("goldPerMinute", None),
                "csDiffAt10": p.get("challenges", {}).get("laneMinionsFirst10Minutes", None)
            })
        return pd.DataFrame(rows)

    def build_lane_matchups(df_parts: pd.DataFrame) -> pd.DataFrame:
        df = df_parts[(df_parts["queueId"] == 420) & (df_parts["teamPosition"].isin(LANE_ORDER))].copy()
        left = df.rename(columns={"teamPosition":"lane","championId":"champA_id","championName":"champA","teamId":"teamA","win":"winA","puuid":"puuidA"})
        right = df.rename(columns={"teamPosition":"lane","championId":"champB_id","championName":"champB","teamId":"teamB","win":"winB","puuid":"puuidB"})
        merged = pd.merge(
            left[["matchId","lane","teamA","winA","champA_id","champA","puuidA","kills","deaths","assists","gameDuration","gameVersion","game_year"]],
            right[["matchId","lane","teamB","winB","champB_id","champB","puuidB"]],
            on=["matchId","lane"], how="inner"
        )
        merged = merged[merged["teamA"] != merged["teamB"]].copy()
        merged["A_wins_game"] = merged["winA"].astype(int)
        return merged[["matchId","lane","champA_id","champA","champB_id","champB","A_wins_game","puuidA","puuidB","kills","deaths","assists","gameDuration","gameVersion","game_year"]]

    def aggregate_lane_matchups(df_lane: pd.DataFrame, min_games: int = 30) -> pd.DataFrame:
        grp = df_lane.groupby(["lane","champA_id","champB_id"], as_index=False).agg(
            games=("A_wins_game","size"), wins=("A_wins_game","sum")
        )
        grp["p_win"] = grp["wins"] / grp["games"]
        return grp[grp["games"] >= min_games].sort_values(["lane","games"], ascending=[True, False])

    def player_champion_stats(df_parts: pd.DataFrame) -> pd.DataFrame:
        df = df_parts[df_parts["queueId"] == 420].copy()
        grp = df.groupby(["puuid","championId","championName"], as_index=False).agg(
            games=("win","size"), wins=("win","sum")
        )
        grp["winrate"] = grp["wins"] / grp["games"]
        return grp

    def attach_champion_mastery(df_summoners: pd.DataFrame) -> pd.DataFrame:
        """Champion-Mastery per **PUUID** abrufen (empfohlen & unterstützt)."""
        rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(mastery_by_puuid, puuid): puuid for puuid in df_summoners["puuid"].tolist()}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Mastery (by PUUID)"):
                puuid = futs[fut]
                try:
                    mastery = fut.result() or []
                    for m in mastery:
                        rows.append({
                            "puuid": puuid,
                            "championId": m.get("championId"),
                            "championLevel": m.get("championLevel"),
                            "championPoints": m.get("championPoints"),
                            "lastPlayTime": m.get("lastPlayTime")
                        })
                except Exception:
                    pass
        return pd.DataFrame(rows)

    # ---------------------------
    # 4) Orchestrierung
    # ---------------------------
    def main(with_timeline: bool = False):
        if RUN_DEBUG_PROBE:
            print("== Debug-Probe league/v4/entries =="); _debug_probe(); print("== Ende Debug-Probe ==\n")

        # 1) Spieler-Stichprobe (oder Bootstrap)
        if USE_BOOTSTRAP:
            print(f"[INFO] Bootstrap über Summoner-Namen '{BOOTSTRAP_SUMMONER_NAME}' aktiviert.")
            df_entries = bootstrap_from_summoner_name(BOOTSTRAP_SUMMONER_NAME)
        else:
            df_entries = sample_summoners(queue="RANKED_SOLO_5x5")
        df_entries.to_parquet(f"{SAVE_DIR}/league_entries_sample.parquet", index=False)

        # 2) (Kein Summoner-ID-Resolve mehr nötig) – wir behalten PUUID-only
        df_summ = df_entries[["puuid"]].drop_duplicates().copy()
        df_summ.to_parquet(f"{SAVE_DIR}/summoners.parquet", index=False)

        # 3) Match-IDs & Matches
        df_mids = harvest_match_ids(df_summ)
        df_mids.to_parquet(f"{SAVE_DIR}/match_ids.parquet", index=False)

        df_matches = harvest_matches(df_mids, with_timeline=with_timeline)
        df_matches.to_parquet(f"{SAVE_DIR}/matches_raw.parquet", index=False)

        # 4) Participants & Lane-Matchups
        parts_rows = []
        for rec in tqdm(df_matches.itertuples(index=False), total=len(df_matches), desc="Participants"):
            ptab = extract_participants_table(rec.match)
            if MIN_YEAR is not None:
                ptab = ptab[(ptab["game_year"].isna()) | (ptab["game_year"] >= MIN_YEAR)]
            parts_rows.append(ptab)
        df_parts = pd.concat(parts_rows, ignore_index=True) if parts_rows else pd.DataFrame()
        if df_parts.empty:
            print("[WARN] participants leer – evtl. keine Matches geladen?")
        else:
            df_parts.to_parquet(f"{SAVE_DIR}/participants.parquet", index=False)

            df_lane = build_lane_matchups(df_parts)
            df_lane.to_parquet(f"{SAVE_DIR}/lane_matchups.parquet", index=False)

            df_lane_agg = aggregate_lane_matchups(df_lane, min_games=30)
            df_lane_agg.to_parquet(f"{SAVE_DIR}/lane_matchup_agg.parquet", index=False)

            # 5) Spieler-Proficiency (Mastery jetzt by-PUUID)
            df_mastery = attach_champion_mastery(df_summ)
            df_mastery.to_parquet(f"{SAVE_DIR}/champion_mastery.parquet", index=False)

            df_player_champ = player_champion_stats(df_parts)
            df_player_champ.to_parquet(f"{SAVE_DIR}/player_champion_stats.parquet", index=False)

        print("Fertig. Dateien gespeichert in:", SAVE_DIR)

    if __name__ == "__main__":
        main(with_timeline=False)
