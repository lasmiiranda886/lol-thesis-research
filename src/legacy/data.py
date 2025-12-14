import json
from pathlib import Path
import pandas as pd

# 1) Take a small, safe subset to validate schema
matches_dir = Path("data_collection/matches")
files = sorted(matches_dir.glob("*.json"))
subset = files[:500]  # adjust if you want

rows = []
for fp in subset:
    with open(fp, "r") as f:
        m = json.load(f)
    meta = m.get("metadata", {})
    info = m.get("info", {})
    match_id = meta.get("matchId")
    queue_id = info.get("queueId")
    game_version = info.get("gameVersion")
    for p in info.get("participants", []) or []:
        rows.append({
            "matchId": match_id,
            "queueId": queue_id,
            "gameVersion": game_version,
            "teamId": p.get("teamId"),
            "win": bool(p.get("win")),
            "championId": p.get("championId"),
            "championName": p.get("championName"),
            "teamPosition": p.get("teamPosition"),  # TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY or None
            "lane": p.get("lane"),                  # legacy field, may be None
            "role": p.get("role"),                  # legacy field, may be None
        })

df = pd.DataFrame(rows)

# 2) Quick sanity checks
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSample rows:")
print(df.sample(min(5, len(df)), random_state=0))

print("\nCounts by teamPosition and win:")
print(df.groupby(["teamPosition","win"]).size().unstack(fill_value=0).head(10))

print("\nTop champions in subset:")
print(df["championName"].value_counts().head(10))
