# Feature Documentation - LoL Win Prediction V8

## Übersicht

| Kategorie | Anzahl | Prefix |
|-----------|--------|--------|
| Hero Player Stats | 8 | `hero_` |
| Champion Stats | 13 | `cs_` |
| Team Composition | 21 | `tc_` |
| Smurf Detection | 3 | - |
| Region | 4 | `region_` |
| **Total** | **48** | |

---

## 1. Hero Player Stats (8 Features)

Features die direkt vom "Hero" (der Spieler dessen Perspektive wir predicten) stammen.

| Feature | Beschreibung | Berechnung | Quelle |
|---------|--------------|------------|--------|
| `hero_rank_numeric` | Numerischer Rank | 1=Iron, 2=Bronze, 3=Silver, 4=Gold, 5=Platinum, 6=Emerald, 7=Diamond, 8=Master+ | Riot League API |
| `hero_lp` | League Points | 0-100 LP innerhalb der Division | Riot League API |
| `hero_is_blue_feat` | Blue Side Indicator | 1 wenn Blue Side, 0 wenn Red Side | `teamId == 100` |
| `hero_total_games` | Ranked Games gespielt | `wins + losses` | Riot League API |
| `hero_winrate` | Aktuelle Winrate | `wins / (wins + losses)` | Berechnet |
| `hero_cm_points_log` | Champion Mastery (log) | `log(mastery_points + 1)` | Riot Mastery API |
| `hero_cm_level_feat` | Champion Mastery Level | 1-7 (Mastery Level) | Riot Mastery API |
| `hero_wr_rank_mismatch` | Winrate-Rank Diskrepanz | `actual_winrate - expected_winrate_at_rank` | Berechnet |

**Code-Referenz:** `src/merge_hero_datasets.py` (Zeilen 117-130)

---

## 2. Champion Stats (13 Features) - Prefix `cs_`

Historische Champion-Statistiken basierend auf 400k+ Matches.

### 2.1 Champion Winrates

| Feature | Beschreibung | Berechnung |
|---------|--------------|------------|
| `cs_hero_champ_wr_at_elo` | Champion WR bei Hero's Elo | Bayesian-smoothed WR für Champion @ Elo-Tier |
| `cs_hero_champ_wr_at_role` | Champion WR auf Hero's Role | Bayesian-smoothed WR für Champion @ Role |
| `cs_hero_champ_wr_at_elo_role` | Champion WR bei Elo+Role | Feinste Granularität: Champion @ Elo @ Role |

### 2.2 Team Champion Strength

| Feature | Beschreibung | Berechnung |
|---------|--------------|------------|
| `cs_hero_team_avg_wr` | Team Durchschnitts-WR | Mean WR aller 5 Team-Champions @ Hero's Elo |
| `cs_enemy_team_avg_wr` | Enemy Durchschnitts-WR | Mean WR aller 5 Enemy-Champions @ Hero's Elo |
| `cs_hero_vs_enemy_wr` | Team vs Enemy Differenz | `team_avg_wr - enemy_avg_wr` |

### 2.3 Matchup Stats

| Feature | Beschreibung | Berechnung |
|---------|--------------|------------|
| `cs_hero_matchup_wr` | Hero's Lane Matchup WR | WR von Hero's Champion vs Lane-Gegner |
| `cs_hero_matchup_known` | Matchup-Daten Flag | 1 wenn echte Matchup-Daten, 0 wenn geschätzt |
| `cs_hero_matchup_wr_at_elo` | Matchup WR @ Elo | Elo-spezifische Matchup WR |
| `cs_hero_matchup_known_at_elo` | Elo-Matchup Flag | 1 wenn Elo-spezifische Daten vorhanden |
| `cs_hero_team_matchup_wr` | Team Matchup Score | Gewichteter Durchschnitt aller 5 Lane-Matchups |
| `cs_matchup_coverage` | Matchup-Abdeckung | % der Matchups mit echten Daten (nicht geschätzt) |

### 2.4 Mastery

| Feature | Beschreibung | Berechnung |
|---------|--------------|------------|
| `cs_hero_mastery_zscore` | Mastery Z-Score | `(hero_mastery - avg_mastery_at_elo) / std_mastery` |

**Fallback-Hierarchie für Matchups:**
1. Matchup @ Elo @ Role (wenn ≥20 Games)
2. Matchup @ Role (wenn ≥20 Games, Elo-übergreifend)
3. Champion WR Differential: `estimated_wr = 0.5 + (champ_wr - enemy_wr)` (clipped [0.3, 0.7])

**Code-Referenz:** `src/champion_stats_features.py` (Zeilen 230-312)

---

## 3. Team Composition Features (21 Features) - Prefix `tc_`

Basierend auf Champion-Klassifizierungen aus `data/static/champion_categories.json`.

### 3.1 Differenz-Features (Hero Team - Enemy Team)

| Feature | Beschreibung | Werte |
|---------|--------------|-------|
| `tc_engage_diff` | Engage-Potenzial Differenz | Summe HARD_ENGAGE Score |
| `tc_tank_diff` | Tank-Anzahl Differenz | Anzahl Tank-Tag Champions |
| `tc_scaling_diff` | Scaling Differenz | Summe SCALING_TIER (0-3) |
| `tc_tankiness_diff` | Tankiness Differenz | Normalisierte Tankiness (0-1) |
| `tc_hard_engage_diff` | Hard Engage Differenz | Anzahl S-Tier Engage Champions |
| `tc_scaling_tier_diff` | Scaling Tier Differenz | Late-Game Scaling Score |
| `tc_frontline_diff` | Frontline Differenz | Anzahl Frontline Champions |
| `tc_tank_shredder_diff` | Tank Shredder Differenz | Anti-Tank Champions |
| `tc_poke_diff` | Poke Differenz | Poke-fähige Champions |
| `tc_disengage_diff` | Disengage Differenz | Counter-Engage Champions |

### 3.2 Hero Team Scores (Absolut)

| Feature | Beschreibung |
|---------|--------------|
| `tc_hero_team_engage` | Engage Score Hero's Team |
| `tc_hero_team_tanks` | Tank Count Hero's Team |
| `tc_hero_team_scaling` | Scaling Score Hero's Team |
| `tc_hero_team_tankiness` | Tankiness Score Hero's Team |
| `tc_hero_team_hard_engage` | Hard Engage Count |
| `tc_hero_team_scaling_tier` | Scaling Tier Sum |
| `tc_hero_team_frontline` | Frontline Count |
| `tc_hero_team_tank_shredder` | Tank Shredder Count |
| `tc_hero_team_poke` | Poke Count |
| `tc_hero_team_disengage` | Disengage Count |

### 3.3 Spezial-Features

| Feature | Beschreibung |
|---------|--------------|
| `tc_armor_stacker_vs_ad` | Armor Stacker Vorteil gegen AD-heavy Teams |

**Champion Categories Schema:**
```json
{
  "HARD_ENGAGE": {
    "S": ["Malphite", "Leona", "Amumu"],
    "A": ["Nautilus", "Alistar", "Rakan"],
    "B": ["Jarvan IV", "Sejuani"]
  },
  "SCALING_TIER": {
    "3": ["Kassadin", "Kayle", "Veigar"],
    "2": ["Jinx", "Viktor", "Azir"],
    "1": ["Vayne", "Kog'Maw"],
    "0": ["Renekton", "Pantheon", "Lee Sin"]
  },
  "TANK_SHREDDER": ["Vayne", "Kog'Maw", "Fiora", "Gwen"],
  "FRONTLINE": ["Ornn", "Sion", "Maokai", "Darius"],
  "POKE": ["Xerath", "Jayce", "Zoe", "Nidalee"],
  "DISENGAGE": ["Janna", "Gragas", "Anivia"]
}
```

**Code-Referenz:** `src/build_champion_static_features.py`, `data/static/champion_categories.json`

---

## 4. Smurf Detection (3 Features)

| Feature | Beschreibung | Berechnung |
|---------|--------------|------------|
| `is_potential_smurf` | Smurf-Indikator | `1 if (total_games < 100 and winrate > 0.65) else 0` |
| `smurf_score` | Smurf-Stärke | `winrate if total_games < 100 else 0` |
| `hero_wr_rank_mismatch` | Performance vs Erwartung | `actual_winrate - expected_winrate_at_rank` |

**Logik:** Neue Accounts mit hoher Winrate sind wahrscheinlich Smurfs.

---

## 5. Region Features (4 Features)

One-Hot Encoding der Spielregion.

| Feature | Regionen |
|---------|----------|
| `region_europe` | EUW, EUNE |
| `region_americas` | NA, BR, LAN, LAS |
| `region_asia` | KR, JP |
| `region_sea` | SEA, OCE, TW, VN, TH, PH, SG |

**Quelle:** Riot API `platform` Field

---

## Bayesian Smoothing

### Problem
Champion/Matchup Winrates mit wenigen Games sind unzuverlässig.
- 2 Wins / 2 Games = 100% WR (unrealistisch)

### Lösung: Shrinkage towards Prior

```python
def bayesian_smoothed_rate(wins, games, prior_rate=0.5, prior_strength=10):
    """
    Shrinkage: (wins + prior_rate * prior_strength) / (games + prior_strength)
    """
    if games == 0:
        return prior_rate
    return (wins + prior_rate * prior_strength) / (games + prior_strength)
```

### Beispiele (prior_strength=10)

| Reale Daten | Raw WR | Smoothed WR | Effekt |
|-------------|--------|-------------|--------|
| 0 Wins / 0 Games | N/A | 0.500 | Neutral (Prior) |
| 1 Win / 1 Game | 1.000 | 0.545 | Stark korrigiert |
| 2 Wins / 2 Games | 1.000 | 0.583 | Korrigiert |
| 10 Wins / 10 Games | 1.000 | 0.750 | Moderat korrigiert |
| 60 Wins / 100 Games | 0.600 | 0.591 | Leicht korrigiert |
| 600 Wins / 1000 Games | 0.600 | 0.600 | Fast keine Korrektur |

### Prior Strength nach Feature-Typ

| Feature | Prior Strength | Begründung |
|---------|---------------|------------|
| Champion @ Elo | 15 | Moderate Datenmenge |
| Champion @ Role | 15 | Moderate Datenmenge |
| Champion @ Elo @ Role | 15 | Feinste Granularität |
| Matchup @ Role | 30 | Weniger Daten, stärkere Regularisierung |
| Matchup @ Elo @ Role | 30 | Sehr wenig Daten pro Kombination |

---

## Daten-Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ RIOT API                                                     │
│  - Match-V5: Match Details, 10 Participants                 │
│  - League-V4: Rank, LP, Wins/Losses                         │
│  - Champion-Mastery-V4: Mastery Points, Level               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. COLLECTOR (src/collector.py)                              │
│    - Sammelt Match-Daten von Riot API                       │
│    - Extrahiert 10 Participants pro Match                   │
│    - Holt Rank + Mastery für jeden Spieler                  │
│    Output: data/raw/matches/, data/raw/ranks/, data/raw/mastery/
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MERGE HERO DATASETS (src/merge_hero_datasets.py)         │
│    - Wählt 1 "Hero" pro Match (zufällig, Silver-Diamond)    │
│    - Berechnet hero_* Features                              │
│    - Erstellt Team-Composition Spalten (10 Champion IDs)    │
│    Output: data/interim/aggregate/hero_dataset.parquet      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CHAMPION STATS (src/champion_stats_features.py)          │
│    - Fitted NUR auf Train-Daten (Data Leakage Prevention)   │
│    - Berechnet Bayesian-smoothed Winrates                   │
│    - Berechnet Lane-Matchup WRs mit Fallback                │
│    - Berechnet Mastery Z-Scores                             │
│    Output: cs_* Features                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. TEAM COMP FEATURES                                        │
│    - Lädt champion_categories.json (manuell kuratiert)      │
│    - Lädt champion_features.parquet (Riot Data Dragon)      │
│    - Berechnet tc_* Differenzen und Scores                  │
│    Output: tc_* Features                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. TRAIN/TEST SPLIT (temporal)                               │
│    - hero_dataset_train_v8.parquet: 120,872 Samples         │
│    - hero_dataset_test_v8.parquet: 30,218 Samples           │
│    - features_v8.pkl: 48 Feature-Namen                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Importance (V8 Baseline)

Top 10 Features nach durchschnittlicher Importance (ExtraTrees + LightGBM + XGBoost):

| Rank | Feature | Avg. Importance |
|------|---------|-----------------|
| 1 | `cs_hero_team_matchup_wr` | 9.76% |
| 2 | `hero_rank_numeric` | 7.52% |
| 3 | `cs_hero_matchup_wr` | 5.89% |
| 4 | `cs_hero_champ_wr_at_elo` | 4.21% |
| 5 | `hero_winrate` | 3.98% |
| 6 | `tc_scaling_diff` | 3.45% |
| 7 | `cs_hero_vs_enemy_wr` | 3.21% |
| 8 | `hero_total_games` | 2.87% |
| 9 | `tc_tankiness_diff` | 2.65% |
| 10 | `cs_hero_mastery_zscore` | 2.41% |

Bottom 5 Features (Kandidaten für Entfernung):

| Rank | Feature | Avg. Importance |
|------|---------|-----------------|
| 44 | `tc_hero_team_disengage` | 0.89% |
| 45 | `region_americas` | 0.78% |
| 46 | `tc_armor_stacker_vs_ad` | 0.71% |
| 47 | `region_asia` | 0.62% |
| 48 | `region_sea` | 0.58% |

---

## Modell-Performance (V8)

| Modell | AUC | Accuracy |
|--------|-----|----------|
| ExtraTrees | 0.5891 | 54.2% |
| LightGBM | 0.5903 | 54.5% |
| XGBoost | 0.5912 | 54.6% |
| **Ensemble (Avg)** | **0.5921** | **54.7%** |

**Elo-Breakdown:**
| Elo Tier | AUC | Insight |
|----------|-----|---------|
| LOW (Iron-Silver) | 0.579 | Mastery = Smurf indicator |
| MID (Gold-Plat) | 0.597 | Balanced features |
| HIGH (Emerald+) | 0.604 | Matchup knowledge matters |

---

## Wichtige Constraints (Thesis)

### Was der Spieler in Champion Select sehen kann:
- Eigenen Rank, LP, Wins/Losses
- Eigene Champion Mastery
- Alle 10 Champion-Picks (aber NICHT wer sie spielt)
- Blue/Red Side

### Was NICHT verfügbar ist:
- Ranks der 9 anderen Spieler (seit 2023 von Riot versteckt)
- Mastery der 9 anderen Spieler
- Match-MMR oder Durchschnitts-Elo

---

**Letzte Aktualisierung:** 2024-12-22
**Version:** V8 (48 Features, AUC 0.5921)
