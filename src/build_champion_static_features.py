# src/build_champion_static_features.py

"""
Erzeugt Champion-Level-Features aus Riot Static Data (Data Dragon).

- Lädt champion.json für eine gegebene Patch-Version.
- Extrahiert:
  - Champion-ID (numeric), Name
  - Rollen-Tags (Tank/Fighter/Mage/Assassin/Marksman/Support) als One-Hot
  - Basis- und Per-Level-Stats (HP, Armor, MR) und berechnet Werte auf Level 13
  - einfache Tankiness-Heuristik (effektive HP gegen Phys/Mag, normalisiert)
  - grobe Damage-Profile (AP/AD) aus den Tags
  - grobe Late-Scaling-Heuristik aus per-Level-Stats, normalisiert

Output:
  data/static/champion_features.parquet
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List

import requests
import pandas as pd
import numpy as np


TAG_LIST = ["Tank", "Fighter", "Mage", "Assassin", "Marksman", "Support"]


def fetch_champion_data(dd_version: str, language: str = "en_US") -> Dict[str, Any]:
    """
    Lädt champion.json für die angegebene Data-Dragon-Version und Sprache.

    dd_version: z.B. "14.1.1"
    language: z.B. "en_US" oder "de_DE"
    """
    url = f"https://ddragon.leagueoflegends.com/cdn/{dd_version}/data/{language}/champion.json"
    print(f"Lade Champion-Static-Data von:\n  {url}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data["data"]  # dict: name -> champion_obj


def build_champion_features(champ_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Erzeugt ein DataFrame mit Champion-Level-Features aus dem
    von Data Dragon geladenen champion.json-Content.
    """
    rows: List[Dict[str, Any]] = []

    LEVEL_REF = 13  # Referenzlevel für Tankiness-Berechnung

    for champ_name, obj in champ_data.items():
        # Numeric Champion-ID (string -> int)
        champ_id = int(obj["key"])
        name = obj["name"]
        tags = obj.get("tags", [])
        stats = obj.get("stats", {})

        # Basis-Stats
        base_hp = stats.get("hp", 0.0)
        hp_per_level = stats.get("hpperlevel", 0.0)
        base_armor = stats.get("armor", 0.0)
        armor_per_level = stats.get("armorperlevel", 0.0)
        base_mr = stats.get("spellblock", 0.0)
        mr_per_level = stats.get("spellblockperlevel", 0.0)

        # Werte auf Referenzlevel (z.B. 13)
        lvl_delta = LEVEL_REF - 1
        hp_lvl = base_hp + hp_per_level * lvl_delta
        armor_lvl = base_armor + armor_per_level * lvl_delta
        mr_lvl = base_mr + mr_per_level * lvl_delta

        # Effektive HP (sehr grobe Näherung)
        ehp_phys = hp_lvl * (1.0 + armor_lvl / 100.0)
        ehp_magic = hp_lvl * (1.0 + mr_lvl / 100.0)
        tankiness_raw = 0.5 * (ehp_phys + ehp_magic)

        # Grobe Damage-Profil-Heuristik aus den Tags
        # (kann später durch bessere Heuristik/Manualliste ergänzt werden)
        # Start: neutral
        ap_share = 0.5
        ad_share = 0.5

        if "Mage" in tags:
            ap_share = 0.8
            ad_share = 0.2
        elif "Marksman" in tags:
            ap_share = 0.2
            ad_share = 0.8
        elif "Assassin" in tags:
            ap_share = 0.4
            ad_share = 0.6
        elif "Fighter" in tags:
            ap_share = 0.3
            ad_share = 0.7
        elif "Tank" in tags:
            ap_share = 0.3
            ad_share = 0.7
        elif "Support" in tags:
            ap_share = 0.6
            ad_share = 0.4

        # Late-Game-Scaling grob aus per-level Stats ableiten
        base_ad = stats.get("attackdamage", 0.0)
        ad_per_level = stats.get("attackdamageperlevel", 0.0)
        as_per_level_pct = stats.get("attackspeedperlevel", 0.0)  # in %, z.B. 2.5

        # Verhältnis "per-level" zu "base" als proxy für Scaling
        ad_scale = (ad_per_level / base_ad) if base_ad > 0 else 0.0
        hp_scale = (hp_per_level / base_hp) if base_hp > 0 else 0.0
        as_scale = as_per_level_pct / 100.0

        scaling_raw = ad_scale + hp_scale + as_scale

        row: Dict[str, Any] = {
            "championId": champ_id,
            "championName": name,
            "tags": ",".join(tags),
            # Level-Stats
            "hp_lvl13": hp_lvl,
            "armor_lvl13": armor_lvl,
            "mr_lvl13": mr_lvl,
            "tankiness_raw": tankiness_raw,
            # Damage-Profil
            "damage_ap_share": ap_share,
            "damage_ad_share": ad_share,
            # Scaling-Rohwert
            "late_scaling_raw": scaling_raw,
        }

        # One-Hot für Rollen
        for t in TAG_LIST:
            row[f"tag_{t.lower()}"] = 1 if t in tags else 0

        rows.append(row)

    df = pd.DataFrame(rows)

    # Tankiness normalisieren auf 0–1
    t_min = df["tankiness_raw"].min()
    t_max = df["tankiness_raw"].max()
    if t_max > t_min:
        df["tankiness_score"] = (df["tankiness_raw"] - t_min) / (t_max - t_min)
    else:
        df["tankiness_score"] = 0.5  # fallback

    # Late-Scaling normalisieren auf 0–1
    s_min = df["late_scaling_raw"].min()
    s_max = df["late_scaling_raw"].max()
    if s_max > s_min:
        df["late_scaling_score"] = (df["late_scaling_raw"] - s_min) / (s_max - s_min)
    else:
        df["late_scaling_score"] = 0.5

    # Aufräumen: Rohwerte optional behalten, können für Analysen hilfreich sein
    # Wenn du es schlanker möchtest, kannst du tankiness_raw/late_scaling_raw später droppen.
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baue Champion-Static-Features aus Riot Data Dragon."
    )
    parser.add_argument(
        "--dd-version",
        type=str,
        default="14.1.1",  # ggf. an deine Datenversion anpassen
        help="Data-Dragon-Version, z.B. 14.1.1",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en_US",
        help="Sprache für die Static Data, z.B. en_US oder de_DE.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/static/champion_features.parquet",
        help="Zielpfad für die Champion-Feature-Tabelle (Parquet).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    champ_data = fetch_champion_data(args.dd_version, args.language)
    df_feats = build_champion_features(champ_data)

    print("Beispielzeilen:")
    print(df_feats.head())

    print(f"\nSpeichere Champion-Features nach: {out_path}")
    df_feats.to_parquet(out_path, index=False)
    print("Fertig.")


if __name__ == "__main__":
    main()
