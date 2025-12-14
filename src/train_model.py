#!/usr/bin/env python3
"""
Trainingsskript (Hero-Level) für die Bachelorarbeit:
Vorhersage der Gewinnwahrscheinlichkeit in der Championauswahl aus Sicht eines "Hero"-Spielers.

Input:
    data/interim/aggregate/hero_dataset.parquet  (MINIMAL, 25 cols)

WICHTIG:
- hero_puuid darf im Parquet bleiben (Debug), wird IMMER aus Features ausgeschlossen.
- Keine Legacy-Features.
- Matchup-Priors werden leakage-frei NUR aus TRAIN berechnet.
- Matchups mit n < min_count werden auf global_default (train blue_win mean) gesetzt.
- Smoothing: Beta(alpha, beta)

Feature Sets:
- PICKS_ONLY: 10 Champion Picks + hero_is_blue
- HERO_ONLY: nur Hero-Eigenschaften (Rank/Mastery/Role/Side)
- PICKS_HERO: PICKS_ONLY + HERO_ONLY
- PICKS_HERO_MATCHUPS: PICKS_HERO + matchup_wr_hero_{lane} + matchup_log_n_{lane}

Output:
    data/models/baseline_results_hero_clean.csv
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
DATA_PATH = "data/interim/aggregate/hero_dataset.parquet"

ID_COL = "matchId"
TARGET_COL = "hero_win"
DEBUG_ID_COLS = ["hero_puuid"]

LANES = ["top", "jungle", "mid", "adc", "supp"]
BLUE_CHAMP_COLS = [f"championid_blue_{lane}" for lane in LANES]
RED_CHAMP_COLS = [f"championid_red_{lane}" for lane in LANES]
ALL_CHAMP_COLS = BLUE_CHAMP_COLS + RED_CHAMP_COLS

# Hero columns (erlaubt; in deinem minimal parquet vorhanden)
HERO_COLS = [
    "hero_is_blue",
    "hero_teamPosition",
    "hero_championId",
    "hero_rank_tier",
    "hero_rank_div",
    "hero_leaguePoints",
    "hero_wins",
    "hero_losses",
    "hero_cm_points",
    "hero_cm_level",
    "hero_cm_lastPlayTime",
]

# Matchup features (werden dynamisch hinzugefügt)
MATCHUP_WR_BLUE_COLS = [f"matchup_wr_blue_{lane}" for lane in LANES]
MATCHUP_N_COLS       = [f"matchup_n_{lane}" for lane in LANES]
MATCHUP_WR_HERO_COLS = [f"matchup_wr_hero_{lane}" for lane in LANES]
MATCHUP_LOGN_COLS    = [f"matchup_log_n_{lane}" for lane in LANES]


# -----------------------------------------------------------------------------
# Experiments (reduced for speed)
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str
    feature_set: str
    model_type: str
    params: Optional[Dict] = None


def make_experiments() -> List[ExperimentConfig]:
    """
    Schnell-Setup:
    - Dummy
    - LogReg (1x)
    - HGB (1x)
    - RF  (1x)
    Für: HERO_ONLY, PICKS_ONLY, PICKS_HERO, PICKS_HERO_MATCHUPS
    """
    exps: List[ExperimentConfig] = []

    # Dummy baseline (nur damit LogLoss/AUC sanity bleibt)
    exps.append(ExperimentConfig("DUMMY_majority__PICKS_HERO", "PICKS_HERO", "dummy", {"strategy": "most_frequent"}))

    # 3 "echte" Modelle, jeweils 4 Feature-Sets
    feature_sets = ["HERO_ONLY", "PICKS_ONLY", "PICKS_HERO", "PICKS_HERO_MATCHUPS"]

    # Logistic Regression
    for fs in feature_sets:
        exps.append(ExperimentConfig(f"LOGREG__{fs}", fs, "logreg", {"C": 1.0}))

    # HistGradientBoosting (oft am besten in deinem Setup)
    for fs in feature_sets:
        exps.append(ExperimentConfig(f"HGB__{fs}", fs, "hgb", {"max_iter": 500, "learning_rate": 0.06}))

    # RandomForest (ein Setting)
    for fs in feature_sets:
        exps.append(ExperimentConfig(f"RF__{fs}", fs, "rf", {"n_estimators": 300, "max_depth": 12}))

    return exps


# -----------------------------------------------------------------------------
# Feature columns
# -----------------------------------------------------------------------------
def _existing(cols: List[str], available: List[str]) -> List[str]:
    return [c for c in cols if c in available]


def build_feature_columns(feature_set: str, available_cols: List[str]) -> List[str]:
    fs = feature_set.upper()

    hero_side = _existing(["hero_is_blue"], available_cols)

    if fs == "PICKS_ONLY":
        cols = _existing(ALL_CHAMP_COLS, available_cols) + hero_side

    elif fs == "HERO_ONLY":
        cols = _existing(HERO_COLS, available_cols)

    elif fs == "PICKS_HERO":
        cols = _existing(ALL_CHAMP_COLS, available_cols) + hero_side + _existing(HERO_COLS, available_cols)

    elif fs == "PICKS_HERO_MATCHUPS":
        cols = _existing(ALL_CHAMP_COLS, available_cols) + hero_side + _existing(HERO_COLS, available_cols)
        cols += _existing(MATCHUP_WR_HERO_COLS, available_cols)
        cols += _existing(MATCHUP_LOGN_COLS, available_cols)

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # Safety: remove debug/id/target
    cols = [c for c in cols if c not in [ID_COL, TARGET_COL, *DEBUG_ID_COLS]]

    # Dedup preserve order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# -----------------------------------------------------------------------------
# Matchups (leakage-free + smoothing + min_count=20)
# -----------------------------------------------------------------------------
def compute_matchup_tables(
    train_df: pd.DataFrame,
    *,
    min_count: int = 20,
    alpha: float = 8.0,
    beta: float = 8.0,
) -> Tuple[Dict[str, Dict[Tuple[int, int], float]], Dict[str, Dict[Tuple[int, int], int]]]:
    """
    Returns per lane:
      - wr_map[(b,r)] = smoothed P(blue_win | pair)
      - n_map[(b,r)]  = count(pair)
    Only keeps pairs with n >= min_count (others default to global).
    """
    if "blue_win" not in train_df.columns:
        raise ValueError("Need 'blue_win' to compute matchup tables.")

    wr_maps: Dict[str, Dict[Tuple[int, int], float]] = {}
    n_maps: Dict[str, Dict[Tuple[int, int], int]] = {}

    for lane in LANES:
        b = f"championid_blue_{lane}"
        r = f"championid_red_{lane}"
        if b not in train_df.columns or r not in train_df.columns:
            continue

        g = train_df.groupby([b, r])["blue_win"]
        cnt = g.size()
        wins = g.sum()

        keep = cnt[cnt >= min_count].index
        if len(keep) == 0:
            wr_maps[lane] = {}
            n_maps[lane] = {}
            continue

        cnt_k = cnt.loc[keep].astype(float)
        wins_k = wins.loc[keep].astype(float)

        smoothed = (wins_k + alpha) / (cnt_k + alpha + beta)

        wr_maps[lane] = smoothed.to_dict()
        n_maps[lane] = cnt.loc[keep].astype(int).to_dict()

    return wr_maps, n_maps


def add_matchup_features(
    df: pd.DataFrame,
    wr_maps: dict,
    n_maps: dict,
    global_default: float,
) -> pd.DataFrame:
    """
    Adds:
      matchup_wr_blue_{lane}
      matchup_n_{lane}
    Unknown/filtered pairs -> global_default, n=0.
    """
    df = df.copy()

    for lane in LANES:
        b = f"championid_blue_{lane}"
        r = f"championid_red_{lane}"
        wr_col = f"matchup_wr_blue_{lane}"
        n_col  = f"matchup_n_{lane}"

        if b not in df.columns or r not in df.columns:
            continue

        idx = pd.MultiIndex.from_frame(df[[b, r]])

        wr_map = wr_maps.get(lane, {})
        n_map = n_maps.get(lane, {})

        if wr_map:
            s_wr = pd.Series(wr_map)
            df[wr_col] = s_wr.reindex(idx).values
            df[wr_col] = df[wr_col].fillna(global_default)
        else:
            df[wr_col] = global_default

        if n_map:
            s_n = pd.Series(n_map)
            df[n_col] = s_n.reindex(idx).values
            df[n_col] = df[n_col].fillna(0).astype("int32")
        else:
            df[n_col] = 0

    return df


def add_matchup_hero_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts blue-based matchup to hero-based:
      matchup_wr_hero = matchup_wr_blue if hero blue else 1 - matchup_wr_blue
    Adds:
      matchup_log_n = log1p(matchup_n)
    """
    df = df.copy()
    hero_blue = df["hero_is_blue"].astype(int).values

    for lane in LANES:
        wrb = f"matchup_wr_blue_{lane}"
        n   = f"matchup_n_{lane}"
        wrh = f"matchup_wr_hero_{lane}"
        lgn = f"matchup_log_n_{lane}"

        if wrb in df.columns:
            v = df[wrb].astype(float).values
            df[wrh] = np.where(hero_blue == 1, v, 1.0 - v)

        if n in df.columns:
            df[lgn] = np.log1p(df[n].astype(float))

    return df


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def make_model(model_type: str, params: Optional[Dict] = None):
    params = params or {}
    mt = model_type.lower()

    if mt == "dummy":
        base = {"strategy": "most_frequent"}
        base.update(params)
        return DummyClassifier(**base)

    if mt == "logreg":
        base = {"max_iter": 2000, "solver": "lbfgs", "C": 1.0}
        base.update(params)
        lr = LogisticRegression(**base)
        return make_pipeline(StandardScaler(with_mean=True, with_std=True), lr)

    if mt == "rf":
        base = {"n_estimators": 300, "max_depth": 12, "n_jobs": -1, "random_state": 42}
        base.update(params)
        return RandomForestClassifier(**base)

    if mt == "hgb":
        base = {"max_iter": 500, "learning_rate": 0.06, "random_state": 42}
        base.update(params)
        return HistGradientBoostingClassifier(**base)

    raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return float("nan")


def _safe_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    eps = 1e-7
    y_proba = np.clip(y_proba, eps, 1 - eps)
    try:
        return float(log_loss(y_true, y_proba))
    except ValueError:
        return float("nan")


def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]
    if hasattr(model, "decision_function"):
        from scipy.special import expit
        return expit(model.decision_function(X))
    return model.predict(X).astype(float)


# -----------------------------------------------------------------------------
# Load / preprocess
# -----------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_parquet(path)

    required = [ID_COL, TARGET_COL, "hero_is_blue", "blue_win"] + ALL_CHAMP_COLS + HERO_COLS
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in dataset (expected minimal parquet): {miss}")

    df[TARGET_COL] = df[TARGET_COL].astype("int8")
    df["blue_win"] = df["blue_win"].astype("int8")

    return df


def prepare_splits(
    df: pd.DataFrame,
    feature_set: str,
    test_size: float,
    val_size: float,
    random_state: int,
    matchup_alpha: float = 8.0,
    matchup_beta: float = 8.0,
    matchup_min_count: int = 20,
):
    # 1) test split
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COL],
        random_state=random_state,
    )

    # 2) val split (relative)
    effective_val_frac = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=effective_val_frac,
        stratify=df_trainval[TARGET_COL],
        random_state=random_state,
    )

    # 3) leakage-free matchup tables (nur wenn benötigt)
    needs_matchups = feature_set.upper() == "PICKS_HERO_MATCHUPS"
    if needs_matchups:
        global_default = float(df_train["blue_win"].astype(int).mean())

        wr_maps, n_maps = compute_matchup_tables(
            df_train,
            min_count=matchup_min_count,
            alpha=matchup_alpha,
            beta=matchup_beta,
        )

        df_train = add_matchup_features(df_train, wr_maps, n_maps, global_default)
        df_test  = add_matchup_features(df_test,  wr_maps, n_maps, global_default)
        df_val   = add_matchup_features(df_val,   wr_maps, n_maps, global_default)

        df_train = add_matchup_hero_perspective(df_train)
        df_test  = add_matchup_hero_perspective(df_test)
        df_val   = add_matchup_hero_perspective(df_val)

        print(f"Matchups kept (n>={matchup_min_count}):", {k: len(v) for k, v in wr_maps.items()})
        print("global_default (train blue_win mean):", global_default)

    # 4) feature columns
    available_cols = list(df_train.columns)
    feature_cols = build_feature_columns(feature_set, available_cols)

    # 5) cleaning / encoding
    # champ cols -> int32, missing -> -1
    for col in ALL_CHAMP_COLS:
        if col in feature_cols:
            for d in (df_train, df_test, df_val):
                d[col] = d[col].replace([np.inf, -np.inf], np.nan).fillna(-1).astype("int32")

    # categorical-like -> stable train mapping
    def is_cat(series: pd.Series) -> bool:
        # in minimal parquet: hero_teamPosition, hero_rank_tier, hero_rank_div könnten object sein
        return pd.api.types.is_object_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype)

    cat_cols = [c for c in feature_cols if is_cat(df_train[c])]

    for c in cat_cols:
        train_vals = df_train[c].astype("object").fillna("NA")
        cats = pd.Index(train_vals.unique())
        mapping = {k: i for i, k in enumerate(cats)}
        for d in (df_train, df_test, df_val):
            d[c] = d[c].astype("object").fillna("NA").map(mapping).fillna(-1).astype("int32")

    # remaining numeric -> fillna(0)
    for c in feature_cols:
        if c in ALL_CHAMP_COLS or c in cat_cols:
            continue
        for d in (df_train, df_test, df_val):
            d[c] = d[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train = df_train[feature_cols].copy()
    X_test  = df_test[feature_cols].copy()
    X_val   = df_val[feature_cols].copy()

    y_train = df_train[TARGET_COL].astype(int).values
    y_test  = df_test[TARGET_COL].astype(int).values
    y_val   = df_val[TARGET_COL].astype(int).values

    return X_train, X_test, X_val, y_train, y_test, y_val, feature_cols


# -----------------------------------------------------------------------------
# Train & eval
# -----------------------------------------------------------------------------
def train_and_evaluate(cfg: ExperimentConfig, X_train, X_test, X_val, y_train, y_test, y_val):
    model = make_model(cfg.model_type, cfg.params)
    model.fit(X_train, y_train)

    p_tr = get_proba(model, X_train)
    p_te = get_proba(model, X_test)
    p_va = get_proba(model, X_val)

    pred_tr = (p_tr >= 0.5).astype(int)
    pred_te = (p_te >= 0.5).astype(int)
    pred_va = (p_va >= 0.5).astype(int)

    metrics = {
        "experiment": cfg.name,
        "feature_set": cfg.feature_set,
        "model_type": cfg.model_type,
        "params": json.dumps(cfg.params or {}, sort_keys=True),

        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_val": int(len(y_val)),
        "n_features": int(X_train.shape[1]),

        "auc_train": _safe_auc(y_train, p_tr),
        "auc_test":  _safe_auc(y_test,  p_te),
        "auc_val":   _safe_auc(y_val,   p_va),

        "logloss_train": _safe_logloss(y_train, p_tr),
        "logloss_test":  _safe_logloss(y_test,  p_te),
        "logloss_val":   _safe_logloss(y_val,   p_va),

        "acc_train": float(accuracy_score(y_train, pred_tr)),
        "acc_test":  float(accuracy_score(y_test,  pred_te)),
        "acc_val":   float(accuracy_score(y_val,   pred_va)),
    }
    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=DATA_PATH)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)

    p.add_argument("--matchup-alpha", type=float, default=8.0)
    p.add_argument("--matchup-beta", type=float, default=8.0)
    p.add_argument("--matchup-min-count", type=int, default=20)

    return p.parse_args()


def main():
    args = parse_args()
    df = load_data(args.data_path)

    print("Loaded dataset:", df.shape)
    if "hero_puuid" in df.columns:
        print("INFO: hero_puuid present (debug only) -> excluded from training features.")

    # hard sanity: matchId unique?
    if df[ID_COL].duplicated().any():
        dup = int(df[ID_COL].duplicated().sum())
        raise RuntimeError(f"Found duplicated matchId rows: {dup}. Fix this before training (hard leakage risk).")

    experiments = make_experiments()
    results = []

    for cfg in experiments:
        print("\n" + "=" * 90)
        print(f"Experiment: {cfg.name} | FS={cfg.feature_set} | model={cfg.model_type} | params={cfg.params}")

        try:
            X_tr, X_te, X_va, y_tr, y_te, y_va, feat_cols = prepare_splits(
                df,
                feature_set=cfg.feature_set,
                test_size=args.test_size,
                val_size=args.val_size,
                random_state=args.random_state,
                matchup_alpha=args.matchup_alpha,
                matchup_beta=args.matchup_beta,
                matchup_min_count=args.matchup_min_count,
            )
            metrics = train_and_evaluate(cfg, X_tr, X_te, X_va, y_tr, y_te, y_va)

        except Exception as e:
            print(f"FAIL {cfg.name}: {type(e).__name__}: {e}")
            continue

        results.append(metrics)

        print(f"AUC  train/test/val: {metrics['auc_train']:.4f} / {metrics['auc_test']:.4f} / {metrics['auc_val']:.4f}")
        print(f"LL   train/test/val: {metrics['logloss_train']:.4f} / {metrics['logloss_test']:.4f} / {metrics['logloss_val']:.4f}")
        print(f"ACC  train/test/val: {metrics['acc_train']:.4f} / {metrics['acc_test']:.4f} / {metrics['acc_val']:.4f}")
        print(f"#features: {metrics['n_features']} | cols={feat_cols}")

    if not results:
        print("No experiments succeeded.")
        return

    out_dir = "data/models"
    os.makedirs(out_dir, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values(by="auc_val", ascending=False)
    out_path = os.path.join(out_dir, "baseline_results_hero_clean.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 90)
    print("Saved results:", out_path)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
