import json

print("=" * 80)
print("VORSCHAU: Welche Champion-Daten werden gesammelt")
print("=" * 80)

print("""
Nach Abschluss der Pipeline wirst du folgende Daten haben:

1. PARTICIPANTS.PARQUET - Pro Spieler/Match:
   - matchId, puuid, summonerId
   - championId, championName (z.B. "Jinx", "LeeSin")
   - teamPosition (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)
   - kills, deaths, assists
   - win (True/False)
   - goldDiffAt10, csDiffAt10
   - gameDuration, gameVersion

2. PLAYER_CHAMPION_STATS.PARQUET - Aggregiert pro Spieler+Champion:
   - puuid
   - championId, championName
   - games (Anzahl Spiele auf diesem Champion)
   - wins (Anzahl Siege)
   - winrate (Berechnet)

3. LANE_MATCHUPS.PARQUET - Champion vs Champion:
   - lane (TOP, JUNGLE, etc.)
   - champA_id, champA (z.B. "Darius")
   - champB_id, champB (z.B. "Garen")
   - A_wins_game (1 oder 0)
   - kills, deaths, assists

4. CHAMPION_MASTERY.PARQUET - Spieler-Erfahrung:
   - puuid
   - championId
   - championLevel (1-7, Mastery Level)
   - championPoints (Erfahrungspunkte auf dem Champion)
   - lastPlayTime

BEISPIEL-ANALYSE DIE DU MACHEN KANNST:
- "Zeige Win-Rate von Jinx in Gold-Rang bei Spielern mit 50+ Games"
- "Welche Champions haben die höchste Win-Rate gegen Yasuo in Midlane?"
- "Korrelation zwischen Champion-Mastery und Win-Rate"
- "Top 10 Champions nach Rang sortiert"
""")

print("\nDie Daten werden automatisch verknüpfbar sein über:")
print("- puuid (Spieler-ID)")
print("- championId (Champion-ID)")
print("- matchId (Match-ID)")
