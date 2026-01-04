"""
Champion Statistics Feature Engineering
========================================
Berechnet Champion-basierte Features aus historischen Daten.

WICHTIG: Alle Stats werden NUR aus dem Train-Split berechnet um Data Leakage zu vermeiden!

Features:
1. Champion WR @ Elo - Wie stark ist der Champion in diesem Elo-Tier?
2. Champion WR @ Role - Wie stark ist der Champion auf dieser Position?
3. Mastery Uplift - Ist der Spieler über/unter dem typischen Mastery-Level für diesen Champ?
4. Team Average Champion Strength - Durchschnittliche Stärke aller 5 Champions

Alle Features nutzen Bayesian Smoothing um bei wenigen Samples zum Globalen Mean zu tendieren.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def bayesian_smoothed_rate(
    successes: int, 
    trials: int, 
    prior_rate: float = 0.5, 
    prior_strength: int = 10
) -> float:
    """
    Berechnet eine geglättete Rate mit Bayesian Shrinkage.
    
    Bei wenigen Trials → tendiert zu prior_rate (meist 0.5)
    Bei vielen Trials → tendiert zur echten Rate
    
    Args:
        successes: Anzahl Erfolge (Wins)
        trials: Anzahl Versuche (Games)
        prior_rate: Prior-Annahme (default 50% WR)
        prior_strength: "Wie viele Games brauchen wir um dem Wert zu vertrauen"
                       Höher = mehr Smoothing
    
    Returns:
        Geglättete Rate zwischen 0 und 1
    
    Examples:
        - 1 Win / 1 Game: (1 + 0.5*10) / (1 + 10) = 0.545 (nicht 1.0!)
        - 60 Wins / 100 Games: (60 + 5) / (110) = 0.59 (fast echt)
        - 0 Wins / 0 Games: 0.5 (prior)
    """
    if trials == 0:
        return prior_rate
    
    return (successes + prior_rate * prior_strength) / (trials + prior_strength)


def bayesian_smoothed_mean(
    values: pd.Series,
    prior_mean: float,
    prior_strength: int = 10
) -> float:
    """
    Berechnet einen geglätteten Durchschnitt mit Bayesian Shrinkage.
    
    Args:
        values: Serie von Werten
        prior_mean: Prior-Annahme für den Durchschnitt
        prior_strength: Wie viele Samples brauchen wir um dem Wert zu vertrauen
    
    Returns:
        Geglätteter Durchschnitt
    """
    n = len(values)
    if n == 0:
        return prior_mean
    
    sample_mean = values.mean()
    return (sample_mean * n + prior_mean * prior_strength) / (n + prior_strength)


class ChampionStatsBuilder:
    """
    Berechnet Champion-basierte Statistiken aus historischen Daten.
    
    WICHTIG: fit() muss NUR auf Train-Daten aufgerufen werden!
    """
    
    def __init__(
        self,
        min_games_for_matchup: int = 20,
        wr_prior_strength: int = 15,
        mastery_prior_strength: int = 10
    ):
        """
        Args:
            min_games_for_matchup: Minimum Games für Matchup-Stats (ansonsten Fallback)
            wr_prior_strength: Smoothing-Stärke für Winrates
            mastery_prior_strength: Smoothing-Stärke für Mastery-Durchschnitte
        """
        self.min_games_for_matchup = min_games_for_matchup
        self.wr_prior_strength = wr_prior_strength
        self.mastery_prior_strength = mastery_prior_strength
        
        # Statistiken (werden in fit() berechnet)
        self.global_wr = 0.5
        self.global_mastery_mean = 0.0
        self.global_mastery_std = 1.0
        
        # Champion @ Elo Stats
        self.champ_elo_stats: Dict[Tuple[int, str], dict] = {}
        
        # Champion @ Role Stats
        self.champ_role_stats: Dict[Tuple[int, str], dict] = {}
        
        # Champion @ Elo @ Role Stats (feinste Granularität)
        self.champ_elo_role_stats: Dict[Tuple[int, str, str], dict] = {}
        
        # Matchup Stats (Lane-Matchups)
        # Key: (champion_id, enemy_champion_id, role)
        self.matchup_role_stats: Dict[Tuple[int, int, str], dict] = {}
        
        # Matchup @ Elo @ Role (feinste Granularität)
        # Key: (champion_id, enemy_champion_id, elo, role)
        self.matchup_elo_role_stats: Dict[Tuple[int, int, str, str], dict] = {}
        
        self._is_fitted = False
    
    def fit(self, participants_df: pd.DataFrame) -> 'ChampionStatsBuilder':
        """
        Berechnet alle Statistiken aus den Trainingsdaten.
        
        Args:
            participants_df: DataFrame mit allen Participants (10 pro Match)
                           Muss enthalten: championId, win, rank_tier, teamPosition, cm_points
        
        Returns:
            self
        """
        print("Fitting ChampionStatsBuilder...")
        df = participants_df.copy()
        
        # Datenbereinigung
        df['championId'] = df['championId'].astype(int)
        df['win'] = df['win'].astype(int)
        df['rank_tier'] = df['rank_tier'].fillna('GOLD').astype(str).str.upper()
        df['teamPosition'] = df['teamPosition'].fillna('UNKNOWN').astype(str).str.upper()
        df['cm_points'] = df['cm_points'].fillna(0).astype(float)
        
        # Globale Stats
        self.global_wr = df['win'].mean()
        self.global_mastery_mean = df['cm_points'].mean()
        self.global_mastery_std = df['cm_points'].std()
        
        print(f"  Global WR: {self.global_wr:.3f}")
        print(f"  Global Mastery Mean: {self.global_mastery_mean:,.0f}")
        print(f"  Global Mastery Std: {self.global_mastery_std:,.0f}")
        
        # 1. Champion @ Elo Stats
        print("  Computing Champion @ Elo stats...")
        champ_elo = df.groupby(['championId', 'rank_tier']).agg({
            'win': ['sum', 'count'],
            'cm_points': ['mean', 'std', 'count']
        }).reset_index()
        champ_elo.columns = ['championId', 'rank_tier', 'wins', 'games', 
                            'mastery_mean', 'mastery_std', 'mastery_count']
        
        for _, row in champ_elo.iterrows():
            key = (int(row['championId']), row['rank_tier'])
            self.champ_elo_stats[key] = {
                'wr': bayesian_smoothed_rate(
                    row['wins'], row['games'], 
                    self.global_wr, self.wr_prior_strength
                ),
                'games': int(row['games']),
                'mastery_mean': bayesian_smoothed_mean(
                    pd.Series([row['mastery_mean']]) if pd.notna(row['mastery_mean']) else pd.Series([]),
                    self.global_mastery_mean,
                    self.mastery_prior_strength
                ),
                'mastery_std': row['mastery_std'] if pd.notna(row['mastery_std']) else self.global_mastery_std
            }
        
        print(f"    {len(self.champ_elo_stats)} Champion-Elo combinations")
        
        # 2. Champion @ Role Stats
        print("  Computing Champion @ Role stats...")
        champ_role = df.groupby(['championId', 'teamPosition']).agg({
            'win': ['sum', 'count'],
            'cm_points': ['mean', 'std']
        }).reset_index()
        champ_role.columns = ['championId', 'teamPosition', 'wins', 'games', 
                             'mastery_mean', 'mastery_std']
        
        for _, row in champ_role.iterrows():
            key = (int(row['championId']), row['teamPosition'])
            self.champ_role_stats[key] = {
                'wr': bayesian_smoothed_rate(
                    row['wins'], row['games'],
                    self.global_wr, self.wr_prior_strength
                ),
                'games': int(row['games']),
                'mastery_mean': row['mastery_mean'] if pd.notna(row['mastery_mean']) else self.global_mastery_mean,
                'mastery_std': row['mastery_std'] if pd.notna(row['mastery_std']) else self.global_mastery_std
            }
        
        print(f"    {len(self.champ_role_stats)} Champion-Role combinations")
        
        # 3. Champion @ Elo @ Role Stats (feinste Granularität)
        print("  Computing Champion @ Elo @ Role stats...")
        champ_elo_role = df.groupby(['championId', 'rank_tier', 'teamPosition']).agg({
            'win': ['sum', 'count'],
            'cm_points': ['mean', 'std']
        }).reset_index()
        champ_elo_role.columns = ['championId', 'rank_tier', 'teamPosition', 
                                  'wins', 'games', 'mastery_mean', 'mastery_std']
        
        for _, row in champ_elo_role.iterrows():
            key = (int(row['championId']), row['rank_tier'], row['teamPosition'])
            self.champ_elo_role_stats[key] = {
                'wr': bayesian_smoothed_rate(
                    row['wins'], row['games'],
                    self.global_wr, self.wr_prior_strength
                ),
                'games': int(row['games']),
                'mastery_mean': row['mastery_mean'] if pd.notna(row['mastery_mean']) else self.global_mastery_mean,
                'mastery_std': row['mastery_std'] if pd.notna(row['mastery_std']) else self.global_mastery_std
            }
        
        print(f"    {len(self.champ_elo_role_stats)} Champion-Elo-Role combinations")
        
        # 4. Matchup Stats (Lane-Matchups)
        print("  Computing Matchup stats...")
        
        # Wir brauchen die Gegner-Champions pro Lane
        # Dafür müssen wir die Matches gruppieren und Blue vs Red vergleichen
        
        # Erstelle Mapping: matchId -> {role -> {blue_champ, red_champ, blue_win}}
        match_lanes = {}
        for _, row in df.iterrows():
            match_id = row['matchId']
            role = row['teamPosition']
            champ_id = int(row['championId'])
            team_id = row.get('teamId', 100)  # 100 = Blue, 200 = Red
            win = row['win']
            elo = row['rank_tier']
            
            if match_id not in match_lanes:
                match_lanes[match_id] = {}
            
            if role not in match_lanes[match_id]:
                match_lanes[match_id][role] = {'elo': elo}
            
            if team_id == 100:  # Blue
                match_lanes[match_id][role]['blue_champ'] = champ_id
                match_lanes[match_id][role]['blue_win'] = win
            else:  # Red
                match_lanes[match_id][role]['red_champ'] = champ_id
        
        # Aggregiere Matchup-Stats
        matchup_data = []  # (champ, enemy_champ, role, elo, win)
        
        for match_id, lanes in match_lanes.items():
            for role, data in lanes.items():
                if 'blue_champ' in data and 'red_champ' in data and 'blue_win' in data:
                    blue_champ = data['blue_champ']
                    red_champ = data['red_champ']
                    blue_win = data['blue_win']
                    elo = data.get('elo', 'GOLD')
                    
                    # Blue's perspective
                    matchup_data.append((blue_champ, red_champ, role, elo, blue_win))
                    # Red's perspective (inverted)
                    matchup_data.append((red_champ, blue_champ, role, elo, 1 - blue_win))
        
        matchup_df = pd.DataFrame(matchup_data, columns=['champ', 'enemy', 'role', 'elo', 'win'])
        
        # 4a. Matchup @ Role (Elo-übergreifend, robuster)
        matchup_role = matchup_df.groupby(['champ', 'enemy', 'role']).agg({
            'win': ['sum', 'count']
        }).reset_index()
        matchup_role.columns = ['champ', 'enemy', 'role', 'wins', 'games']
        
        for _, row in matchup_role.iterrows():
            if row['games'] >= self.min_games_for_matchup:
                key = (int(row['champ']), int(row['enemy']), row['role'])
                self.matchup_role_stats[key] = {
                    'wr': bayesian_smoothed_rate(
                        row['wins'], row['games'],
                        self.global_wr, self.wr_prior_strength * 2  # Stärkeres Smoothing
                    ),
                    'games': int(row['games'])
                }
        
        print(f"    {len(self.matchup_role_stats)} Matchup-Role combinations (min {self.min_games_for_matchup} games)")
        
        # 4b. Matchup @ Elo @ Role (feinste Granularität)
        matchup_elo_role = matchup_df.groupby(['champ', 'enemy', 'elo', 'role']).agg({
            'win': ['sum', 'count']
        }).reset_index()
        matchup_elo_role.columns = ['champ', 'enemy', 'elo', 'role', 'wins', 'games']
        
        for _, row in matchup_elo_role.iterrows():
            if row['games'] >= self.min_games_for_matchup:
                key = (int(row['champ']), int(row['enemy']), row['elo'], row['role'])
                self.matchup_elo_role_stats[key] = {
                    'wr': bayesian_smoothed_rate(
                        row['wins'], row['games'],
                        self.global_wr, self.wr_prior_strength * 2
                    ),
                    'games': int(row['games'])
                }
        
        print(f"    {len(self.matchup_elo_role_stats)} Matchup-Elo-Role combinations (min {self.min_games_for_matchup} games)")
        
        self._is_fitted = True
        print("  Done!")
        
        return self
    
    def get_champion_wr_at_elo(self, champion_id: int, elo: str) -> float:
        """Holt die Champion-Winrate für ein bestimmtes Elo."""
        key = (int(champion_id), elo.upper())
        if key in self.champ_elo_stats:
            return self.champ_elo_stats[key]['wr']
        return self.global_wr
    
    def get_champion_wr_at_role(self, champion_id: int, role: str) -> float:
        """Holt die Champion-Winrate für eine bestimmte Rolle."""
        key = (int(champion_id), role.upper())
        if key in self.champ_role_stats:
            return self.champ_role_stats[key]['wr']
        return self.global_wr
    
    def get_champion_wr_at_elo_role(self, champion_id: int, elo: str, role: str) -> float:
        """Holt die Champion-Winrate für Elo+Rolle Kombination."""
        key = (int(champion_id), elo.upper(), role.upper())
        if key in self.champ_elo_role_stats:
            return self.champ_elo_role_stats[key]['wr']
        # Fallback: Elo-only
        return self.get_champion_wr_at_elo(champion_id, elo)
    
    def get_mastery_percentile(
        self, 
        champion_id: int, 
        elo: str, 
        player_mastery: float
    ) -> float:
        """
        Berechnet wie der Spieler-Mastery relativ zum Durchschnitt in diesem Elo ist.
        
        Returns:
            Z-Score: 0 = Durchschnitt, >0 = überdurchschnittlich, <0 = unterdurchschnittlich
        """
        key = (int(champion_id), elo.upper())
        
        if key in self.champ_elo_stats:
            stats = self.champ_elo_stats[key]
            mean = stats['mastery_mean']
            std = stats['mastery_std']
        else:
            mean = self.global_mastery_mean
            std = self.global_mastery_std
        
        if std == 0 or pd.isna(std):
            std = self.global_mastery_std
        
        if std == 0:
            return 0.0
        
        return (player_mastery - mean) / std
    
    def get_matchup_wr(
        self, 
        champion_id: int, 
        enemy_champion_id: int, 
        role: str, 
        elo: str
    ) -> Tuple[float, str]:
        """
        Holt die Matchup-Winrate mit Fallback-Hierarchie.
        
        Fallback:
        1. Matchup @ Elo @ Role (beste Granularität)
        2. Matchup @ Role (Elo-übergreifend)
        3. Champion WR @ Elo - Enemy WR @ Elo (wenn Matchup nie gesehen)
        
        Returns:
            (winrate, source) - source zeigt welcher Fallback verwendet wurde
        """
        champ = int(champion_id)
        enemy = int(enemy_champion_id)
        role_upper = role.upper()
        elo_upper = elo.upper()
        
        # 1. Versuche Matchup @ Elo @ Role
        key_elo_role = (champ, enemy, elo_upper, role_upper)
        if key_elo_role in self.matchup_elo_role_stats:
            return self.matchup_elo_role_stats[key_elo_role]['wr'], 'matchup_elo_role'
        
        # 2. Fallback: Matchup @ Role (Elo-übergreifend)
        key_role = (champ, enemy, role_upper)
        if key_role in self.matchup_role_stats:
            return self.matchup_role_stats[key_role]['wr'], 'matchup_role'
        
        # 3. Fallback: Champion WR Differenz
        champ_wr = self.get_champion_wr_at_elo(champ, elo_upper)
        enemy_wr = self.get_champion_wr_at_elo(enemy, elo_upper)
        
        # Schätze Matchup-WR basierend auf Champion-Stärke-Differenz
        # Wenn beide 50% haben → 50%
        # Wenn ich 55% habe und Gegner 45% → ich sollte ~55% haben
        estimated_wr = 0.5 + (champ_wr - enemy_wr)
        estimated_wr = np.clip(estimated_wr, 0.3, 0.7)  # Begrenze auf realistische Werte
        
        return estimated_wr, 'estimated'
    
    def get_estimated_mastery(self, champion_id: int, elo: str) -> float:
        """
        Schätzt die Mastery eines Spielers basierend auf Champion und Elo.
        
        Verwendet den MEDIAN (robuster als Mean bei Mastery-Verteilungen).
        Da wir nur Mean gespeichert haben, nutzen wir den als Proxy.
        
        Returns:
            Geschätzte Mastery Points
        """
        key = (int(champion_id), elo.upper())
        if key in self.champ_elo_stats:
            return self.champ_elo_stats[key]['mastery_mean']
        return self.global_mastery_mean
    
    def compute_features_for_match(
        self,
        hero_champion_id: int,
        hero_elo: str,
        hero_role: str,
        hero_mastery: float,
        blue_champion_ids: list,
        red_champion_ids: list,
        blue_roles: list,
        red_roles: list,
        hero_is_blue: bool
    ) -> Dict[str, float]:
        """
        Berechnet alle Champion-Stats Features für ein Match aus Hero-Perspektive.
        
        Args:
            hero_champion_id: Champion ID des Hero
            hero_elo: Elo-Tier des Hero (z.B. "GOLD")
            hero_role: Position des Hero (z.B. "TOP")
            hero_mastery: Mastery Points des Hero auf diesem Champion
            blue_champion_ids: Liste der 5 Blue Team Champion IDs [top, jgl, mid, adc, sup]
            red_champion_ids: Liste der 5 Red Team Champion IDs [top, jgl, mid, adc, sup]
            blue_roles: Liste der Rollen ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
            red_roles: Liste der Rollen
            hero_is_blue: True wenn Hero im Blue Team ist
        
        Returns:
            Dictionary mit Features
        """
        features = {}
        
        # Standard-Rollen falls nicht angegeben
        if not blue_roles or len(blue_roles) != 5:
            blue_roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        if not red_roles or len(red_roles) != 5:
            red_roles = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
        
        # =================================================================
        # 1. Hero's Champion Strength (WR-basiert)
        # =================================================================
        features['hero_champ_wr_at_elo'] = self.get_champion_wr_at_elo(hero_champion_id, hero_elo)
        features['hero_champ_wr_at_role'] = self.get_champion_wr_at_role(hero_champion_id, hero_role)
        features['hero_champ_wr_at_elo_role'] = self.get_champion_wr_at_elo_role(
            hero_champion_id, hero_elo, hero_role
        )
        
        # =================================================================
        # 2. Hero Mastery Uplift (ECHTE Mastery vs. Durchschnitt)
        # =================================================================
        features['hero_mastery_zscore'] = self.get_mastery_percentile(
            hero_champion_id, hero_elo, hero_mastery
        )
        features['hero_mastery_zscore'] = np.clip(features['hero_mastery_zscore'], -3, 3)
        
        expected_mastery = self.get_estimated_mastery(hero_champion_id, hero_elo)
        features['hero_mastery_vs_expected'] = hero_mastery - expected_mastery
        features['hero_mastery_vs_expected_log'] = np.sign(features['hero_mastery_vs_expected']) * \
            np.log1p(abs(features['hero_mastery_vs_expected']))
        
        # =================================================================
        # 3. Team Average Champion Strength (WR-basiert)
        # =================================================================
        blue_wrs = [self.get_champion_wr_at_elo(cid, hero_elo) 
                   for cid in blue_champion_ids if pd.notna(cid)]
        red_wrs = [self.get_champion_wr_at_elo(cid, hero_elo) 
                  for cid in red_champion_ids if pd.notna(cid)]
        
        features['blue_team_avg_champ_wr'] = np.mean(blue_wrs) if blue_wrs else self.global_wr
        features['red_team_avg_champ_wr'] = np.mean(red_wrs) if red_wrs else self.global_wr
        features['team_champ_wr_diff'] = features['blue_team_avg_champ_wr'] - features['red_team_avg_champ_wr']
        
        # =================================================================
        # 4. MATCHUP FEATURES (Lane-Matchups)
        # =================================================================
        
        # Hero's Lane Matchup
        hero_role_upper = hero_role.upper()
        
        # Finde Hero's Position in der Liste
        if hero_is_blue:
            hero_team_champs = blue_champion_ids
            enemy_team_champs = red_champion_ids
            hero_team_roles = blue_roles
            enemy_team_roles = red_roles
        else:
            hero_team_champs = red_champion_ids
            enemy_team_champs = blue_champion_ids
            hero_team_roles = red_roles
            enemy_team_roles = blue_roles
        
        # Finde den Gegner auf der gleichen Lane
        hero_lane_enemy = None
        for i, role in enumerate(enemy_team_roles):
            if role.upper() == hero_role_upper and i < len(enemy_team_champs):
                hero_lane_enemy = enemy_team_champs[i]
                break
        
        # Hero's Lane Matchup WR
        if hero_lane_enemy is not None and pd.notna(hero_lane_enemy):
            matchup_wr, matchup_source = self.get_matchup_wr(
                hero_champion_id, int(hero_lane_enemy), hero_role, hero_elo
            )
            features['hero_matchup_wr'] = matchup_wr
            features['hero_matchup_source'] = 1.0 if matchup_source == 'matchup_elo_role' else \
                                              0.5 if matchup_source == 'matchup_role' else 0.0
        else:
            features['hero_matchup_wr'] = self.global_wr
            features['hero_matchup_source'] = 0.0
        
        # All Lane Matchups (Blue perspective)
        lane_matchup_sum = 0.0
        lane_matchup_count = 0
        matchup_sources_sum = 0.0
        
        for i, (b_champ, r_champ, b_role) in enumerate(zip(blue_champion_ids, red_champion_ids, blue_roles)):
            if pd.notna(b_champ) and pd.notna(r_champ):
                mu_wr, mu_source = self.get_matchup_wr(int(b_champ), int(r_champ), b_role, hero_elo)
                lane_matchup_sum += mu_wr
                lane_matchup_count += 1
                matchup_sources_sum += 1.0 if mu_source == 'matchup_elo_role' else \
                                       0.5 if mu_source == 'matchup_role' else 0.0
        
        if lane_matchup_count > 0:
            features['blue_avg_lane_matchup_wr'] = lane_matchup_sum / lane_matchup_count
            features['matchup_confidence'] = matchup_sources_sum / lane_matchup_count
        else:
            features['blue_avg_lane_matchup_wr'] = self.global_wr
            features['matchup_confidence'] = 0.0
        
        features['red_avg_lane_matchup_wr'] = 1.0 - features['blue_avg_lane_matchup_wr']
        features['lane_matchup_diff'] = features['blue_avg_lane_matchup_wr'] - 0.5
        
        # =================================================================
        # 5. GESCHÄTZTE Team Mastery (basierend auf Hero's Elo)
        # =================================================================
        blue_estimated_mastery = [self.get_estimated_mastery(cid, hero_elo) 
                                  for cid in blue_champion_ids if pd.notna(cid)]
        red_estimated_mastery = [self.get_estimated_mastery(cid, hero_elo) 
                                 for cid in red_champion_ids if pd.notna(cid)]
        
        features['blue_team_est_mastery_avg'] = np.mean(blue_estimated_mastery) if blue_estimated_mastery else self.global_mastery_mean
        features['red_team_est_mastery_avg'] = np.mean(red_estimated_mastery) if red_estimated_mastery else self.global_mastery_mean
        features['team_est_mastery_diff'] = features['blue_team_est_mastery_avg'] - features['red_team_est_mastery_avg']
        
        features['team_est_mastery_diff_log'] = np.sign(features['team_est_mastery_diff']) * \
            np.log1p(abs(features['team_est_mastery_diff']))
        
        # =================================================================
        # 6. Aus Hero-Perspektive aggregieren
        # =================================================================
        if hero_is_blue:
            features['hero_team_avg_champ_wr'] = features['blue_team_avg_champ_wr']
            features['enemy_team_avg_champ_wr'] = features['red_team_avg_champ_wr']
            features['hero_team_est_mastery'] = features['blue_team_est_mastery_avg']
            features['enemy_team_est_mastery'] = features['red_team_est_mastery_avg']
            features['hero_team_lane_matchup'] = features['blue_avg_lane_matchup_wr']
        else:
            features['hero_team_avg_champ_wr'] = features['red_team_avg_champ_wr']
            features['enemy_team_avg_champ_wr'] = features['blue_team_avg_champ_wr']
            features['hero_team_est_mastery'] = features['red_team_est_mastery_avg']
            features['enemy_team_est_mastery'] = features['blue_team_est_mastery_avg']
            features['hero_team_lane_matchup'] = features['red_avg_lane_matchup_wr']
        
        features['hero_vs_enemy_team_wr'] = features['hero_team_avg_champ_wr'] - features['enemy_team_avg_champ_wr']
        features['hero_vs_enemy_team_mastery'] = features['hero_team_est_mastery'] - features['enemy_team_est_mastery']
        features['hero_vs_enemy_team_mastery_log'] = np.sign(features['hero_vs_enemy_team_mastery']) * \
            np.log1p(abs(features['hero_vs_enemy_team_mastery']))
        
        # =================================================================
        # 7. Hero's relativer Beitrag zum Team
        # =================================================================
        features['hero_champ_vs_team_avg'] = (
            features['hero_champ_wr_at_elo'] - features['hero_team_avg_champ_wr']
        )
        
        features['hero_mastery_vs_team_est'] = hero_mastery - features['hero_team_est_mastery']
        features['hero_mastery_vs_team_est_log'] = np.sign(features['hero_mastery_vs_team_est']) * \
            np.log1p(abs(features['hero_mastery_vs_team_est']))
        
        # =================================================================
        # 8. Combined Scores
        # =================================================================
        mastery_boost = features['hero_mastery_zscore'] * 0.02
        features['hero_adjusted_wr'] = features['hero_champ_wr_at_elo'] * (1 + mastery_boost)
        
        # Matchup-adjusted WR
        matchup_boost = (features['hero_matchup_wr'] - 0.5) * 0.5  # 50% weight auf Matchup
        features['hero_matchup_adjusted_wr'] = features['hero_champ_wr_at_elo'] + matchup_boost
        
        return features


def add_champion_stats_to_dataset(
    hero_df: pd.DataFrame,
    participants_df: pd.DataFrame,
    train_indices: pd.Index,
    test_indices: pd.Index
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fügt Champion-Stats Features zu Train und Test DataFrames hinzu.
    
    WICHTIG: Stats werden NUR aus Train-Daten berechnet!
    
    Args:
        hero_df: Hero-Dataset (1 Zeile pro Match)
        participants_df: Alle Participants (10 Zeilen pro Match)
        train_indices: Indices für Train-Split
        test_indices: Indices für Test-Split
    
    Returns:
        (train_df_with_features, test_df_with_features)
    """
    # 1. Finde Train-Match-IDs
    train_match_ids = set(hero_df.loc[train_indices, 'matchId'].unique())
    
    # 2. Filtere participants auf Train-Matches
    train_participants = participants_df[participants_df['matchId'].isin(train_match_ids)]
    
    print(f"Train matches: {len(train_match_ids):,}")
    print(f"Train participants: {len(train_participants):,}")
    
    # 3. Fit Stats Builder nur auf Train-Daten
    builder = ChampionStatsBuilder()
    builder.fit(train_participants)
    
    # 4. Standard-Rollen Mapping
    role_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    
    # 4. Berechne Features für alle Matches
    def compute_features_for_row(row):
        return builder.compute_features_for_match(
            hero_champion_id=int(row['hero_championId']),
            hero_elo=str(row.get('hero_rank_tier', 'GOLD')),
            hero_role=str(row.get('hero_teamPosition', 'UNKNOWN')),
            hero_mastery=float(row.get('hero_cm_points', 0)),
            blue_champion_ids=[
                row.get('championid_blue_top'),
                row.get('championid_blue_jungle'),
                row.get('championid_blue_mid'),
                row.get('championid_blue_adc'),
                row.get('championid_blue_supp'),
            ],
            red_champion_ids=[
                row.get('championid_red_top'),
                row.get('championid_red_jungle'),
                row.get('championid_red_mid'),
                row.get('championid_red_adc'),
                row.get('championid_red_supp'),
            ],
            blue_roles=role_order,
            red_roles=role_order,
            hero_is_blue=bool(row.get('hero_is_blue', True))
        )
    
    print("\nComputing features for train set...")
    train_features = hero_df.loc[train_indices].apply(compute_features_for_row, axis=1)
    train_features_df = pd.DataFrame(train_features.tolist(), index=train_indices)
    train_features_df.columns = [f'cs_{c}' for c in train_features_df.columns]
    
    print("Computing features for test set...")
    test_features = hero_df.loc[test_indices].apply(compute_features_for_row, axis=1)
    test_features_df = pd.DataFrame(test_features.tolist(), index=test_indices)
    test_features_df.columns = [f'cs_{c}' for c in test_features_df.columns]
    
    # 5. Merge
    train_df = pd.concat([hero_df.loc[train_indices], train_features_df], axis=1)
    test_df = pd.concat([hero_df.loc[test_indices], test_features_df], axis=1)
    
    print(f"\nAdded {len(train_features_df.columns)} champion stats features")
    
    return train_df, test_df, builder


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing ChampionStatsBuilder...")
    
    # Lade Daten
    parts = pd.read_parquet('data/interim/aggregate/participants_soloq_clean.parquet')
    hero = pd.read_parquet('data/interim/aggregate/hero_dataset.parquet')
    
    print(f"Participants: {len(parts):,}")
    print(f"Hero matches: {len(hero):,}")
    
    # Simuliere Train/Test Split
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(hero.index, test_size=0.2, random_state=42)
    
    # Fit und Transform
    train_df, test_df, builder = add_champion_stats_to_dataset(
        hero, parts, train_idx, test_idx
    )
    
    print("\nTrain shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    
    # Zeige neue Features
    cs_cols = [c for c in train_df.columns if c.startswith('cs_')]
    print("\nNeue Champion Stats Features:")
    print(train_df[cs_cols].describe())