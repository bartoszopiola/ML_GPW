"""
Phase 4 — training the final production models.

Trains two models on ALL available data (no test holdout):
  final_model_iter1.txt  — regressor (excess target), same setup as Iter1
  final_model_iter3.txt  — LambdaRank, same setup as Iter3

Feature selection: mean gain across the 6 walk-forward windows (from the iter1/iter3 importance CSVs).

Approach:
  1. Split all (trainable) data 90/10: train_main / val_last90
  2. Train with early_stopping on val → best_iter
  3. Retrain on the WHOLE set with num_boost_round = best_iter
  4. Save the final model

Output:
  ml_data/final_model_iter1.txt
  ml_data/final_model_iter3.txt
  ml_data/final_model_meta.json   (dates, features, best_iter, params)

Usage:
    python train_final_model.py
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
INPUT_PARQUET = OUT_DIR / "dataset_ready_for_ml.parquet"

FI_ITER1_CSV = OUT_DIR / "phase3b_iter1_feature_importance.csv"
FI_ITER3_CSV = OUT_DIR / "phase3b_iter3_feature_importance.csv"

MODEL_ITER1  = OUT_DIR / "final_model_iter1.txt"
MODEL_ITER3  = OUT_DIR / "final_model_iter3.txt"
META_JSON    = OUT_DIR / "final_model_meta.json"

TRAIN_START = pd.Timestamp("2021-04-01")
VAL_DAYS = 90       # last N days used as validation for early stopping
TOP_K = 30

NUM_RELEVANCE_BINS = 5
EARLY_STOPPING = 50

LGB_PARAMS_ITER1 = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "force_col_wise": True,
    "seed": 42,
}

LGB_PARAMS_ITER3 = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "lambdarank_truncation_level": 10,
    "label_gain": list(range(NUM_RELEVANCE_BINS)),
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "force_col_wise": True,
    "seed": 42,
}

BRANZA_TO_SUPER = {
    "Banki": "Finanse", "Ubezpieczenia": "Finanse",
    "Wierzytelności": "Finanse", "Rynek kapitałowy": "Finanse",
    "Działalność inwestycyjna": "Finanse",
    "Informatyka": "Tech_Gaming", "Gry": "Tech_Gaming",
    "Nowe technologie": "Tech_Gaming",
    "Górnictwo": "Surowce_Energia", "Hutnictwo": "Surowce_Energia",
    "Energia": "Surowce_Energia", "Paliwa i gaz": "Surowce_Energia",
    "Chemia": "Surowce_Energia",
    "Budownictwo": "Bud_Nieruch", "Nieruchomości": "Bud_Nieruch",
    "Odzież i kosmetyki": "Konsument", "Handel internetowy": "Konsument",
    "Sieci handlowe": "Konsument", "Artykuły spożywcze": "Konsument",
    "Handel hurtowy": "Konsument", "Wyposażenie domu": "Konsument",
    "Rekreacja i wypoczynek": "Konsument",
    "Biotechnologia": "Healthcare", "Produkcja leków": "Healthcare",
    "Sprzęt i materiały medyczne": "Healthcare",
    "Dystrybucja leków": "Healthcare", "Pozostałe": "Healthcare",
    "Przemysł elektromaszynowy": "Przemysl", "Motoryzacja": "Przemysl",
    "Drewno i papier": "Przemysl", "Recykling": "Przemysl",
    "Transport i logistyka": "Przemysl",
    "Media": "Media_Telecom", "Telekomunikacja": "Media_Telecom",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("final_model")


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["super_sektor"] = df["branza"].map(BRANZA_TO_SUPER).fillna("Inne")
    valid = (df["is_liquid"] == True) & df["target_return"].notna()
    sector_avg = (df.loc[valid, ["date", "super_sektor", "target_return"]]
                  .groupby(["date", "super_sektor"])["target_return"]
                  .mean().rename("sector_avg_return").reset_index())
    df = df.merge(sector_avg, on=["date", "super_sektor"], how="left")
    df["target_excess"] = df["target_return"] - df["sector_avg_return"]
    df["target_rank_excess"] = np.nan
    valid_rank = (df["is_liquid"] == True) & df["target_excess"].notna()
    ranks = df.loc[valid_rank, ["date", "target_excess"]].groupby("date")["target_excess"].rank(pct=True)
    df.loc[valid_rank, "target_rank_excess"] = ranks.values
    df["target_relevance"] = np.nan
    df.loc[valid_rank, "target_relevance"] = (
        (df.loc[valid_rank, "target_rank_excess"] * NUM_RELEVANCE_BINS)
        .clip(0, NUM_RELEVANCE_BINS - 1)
        .astype(int)
    )
    return df


def select_features_from_fi(fi_csv: Path, fallback_csv: Path = None) -> list:
    csv = fi_csv if fi_csv.exists() else fallback_csv
    if csv is None or not csv.exists():
        raise FileNotFoundError(f"Feature importance CSV not found: {fi_csv}")
    fi = pd.read_csv(csv)
    avg = (fi.groupby("feature")["gain_pct"]
           .mean()
           .sort_values(ascending=False)
           .head(TOP_K))
    features = avg.index.tolist()
    log.info(f"  Top-3 features: {features[:3]}")
    return features


def get_trainable(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return (df[(df["is_liquid"] == True)
               & df[target_col].notna()
               & (df["feature_coverage"] >= 0.50)
               & (df["date"] >= TRAIN_START)]
            .sort_values("date")
            .reset_index(drop=True))


def compute_ic(pred, target, dates):
    dfr = pd.DataFrame({"p": pred, "t": target, "d": dates})
    ics = [spearmanr(g["p"], g["t"])[0] for _, g in dfr.groupby("d") if len(g) >= 5]
    ics = [v for v in ics if not np.isnan(v)]
    return float(np.mean(ics)) if ics else np.nan


def train_with_retrain(params, X_main, y_main, X_val, y_val,
                       X_all, y_all, feature_names,
                       group_main=None, group_val=None, group_all=None):
    """
    1. Trains on X_main with early stopping on X_val → best_iter
    2. Retrains on X_all (main+val) with num_boost_round=best_iter
    Returns the final model (trained on the whole set).
    """
    ds_main = lgb.Dataset(X_main, label=y_main, group=group_main,
                          feature_name=feature_names, free_raw_data=False)
    ds_val  = lgb.Dataset(X_val,  label=y_val,  group=group_val,
                          reference=ds_main, feature_name=feature_names)

    m_es = lgb.train(params, ds_main,
                     num_boost_round=2000,
                     valid_sets=[ds_val], valid_names=["val"],
                     callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    best_iter = m_es.best_iteration
    log.info(f"  Early stopping best_iter={best_iter}")

    ds_all = lgb.Dataset(X_all, label=y_all, group=group_all,
                         feature_name=feature_names, free_raw_data=False)
    m_final = lgb.train(params, ds_all, num_boost_round=best_iter)
    return m_final, best_iter


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("PHASE 4 — TRAINING THE FINAL PRODUCTION MODELS")
    log.info("=" * 70)

    log.info(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    log.info(f"  {len(df):,} rows, {df['date'].nunique()} dates, "
             f"{df['ticker'].nunique()} companies")

    log.info("Computing the excess target and relevance...")
    df = add_targets(df)
    log.info(f"  target_rank_excess: {df['target_rank_excess'].notna().sum():,} non-NaN")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "train_start": str(TRAIN_START.date()),
    }

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL ITER1 — regressor, excess target
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "─" * 70)
    log.info("ITER1: Regressor (excess target)")
    log.info("─" * 70)

    features_i1 = select_features_from_fi(FI_ITER1_CSV)
    trainable_i1 = get_trainable(df, "target_rank_excess")
    train_end_i1 = trainable_i1["date"].max()
    val_cut_i1   = train_end_i1 - pd.Timedelta(days=VAL_DAYS)

    df_main_i1 = trainable_i1[trainable_i1["date"] <= val_cut_i1]
    df_val_i1  = trainable_i1[trainable_i1["date"] >  val_cut_i1]
    log.info(f"  Train main: {len(df_main_i1):,} rows  ({df_main_i1['date'].min().date()} → {df_main_i1['date'].max().date()})")
    log.info(f"  Val:        {len(df_val_i1):,} rows  ({df_val_i1['date'].min().date()} → {df_val_i1['date'].max().date()})")
    log.info(f"  Total (all):{len(trainable_i1):,} rows  → final retrain")

    # Keep only the features available in the dataset
    features_i1 = [f for f in features_i1 if f in df.columns]
    log.info(f"  Features: {len(features_i1)}")

    X_main = df_main_i1[features_i1].astype("float32")
    y_main = df_main_i1["target_rank_excess"].astype("float32")
    X_val  = df_val_i1[features_i1].astype("float32")
    y_val  = df_val_i1["target_rank_excess"].astype("float32")
    X_all  = trainable_i1[features_i1].astype("float32")
    y_all  = trainable_i1["target_rank_excess"].astype("float32")

    model_i1, best_i1 = train_with_retrain(
        LGB_PARAMS_ITER1, X_main, y_main, X_val, y_val, X_all, y_all, features_i1
    )
    model_i1.save_model(str(MODEL_ITER1))
    log.info(f"  Saved: {MODEL_ITER1}")

    # Check on the validation set
    pred_val_i1 = model_i1.predict(X_val)
    ic_val_i1 = compute_ic(pred_val_i1, y_val.values, df_val_i1["date"])
    log.info(f"  IC check (val, excess):      {ic_val_i1:+.4f}")

    meta["iter1"] = {
        "features": features_i1,
        "best_iter": best_i1,
        "train_end": str(train_end_i1.date()),
        "val_ic_excess": round(ic_val_i1, 4) if not np.isnan(ic_val_i1) else None,
        "params": LGB_PARAMS_ITER1,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL ITER3 — LambdaRank, excess relevance bins
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "─" * 70)
    log.info("ITER3: LambdaRank (relevance bins)")
    log.info("─" * 70)

    features_i3 = select_features_from_fi(FI_ITER3_CSV, fallback_csv=FI_ITER1_CSV)
    trainable_i3 = get_trainable(df, "target_relevance")
    train_end_i3 = trainable_i3["date"].max()
    val_cut_i3   = train_end_i3 - pd.Timedelta(days=VAL_DAYS)

    df_main_i3 = trainable_i3[trainable_i3["date"] <= val_cut_i3].reset_index(drop=True)
    df_val_i3  = trainable_i3[trainable_i3["date"] >  val_cut_i3].reset_index(drop=True)
    log.info(f"  Train main: {len(df_main_i3):,} rows")
    log.info(f"  Val:        {len(df_val_i3):,} rows")

    features_i3 = [f for f in features_i3 if f in df.columns]
    log.info(f"  Features: {len(features_i3)}")

    groups_main = df_main_i3.groupby("date", sort=False).size().values
    groups_val  = df_val_i3.groupby("date", sort=False).size().values
    groups_all  = trainable_i3.groupby("date", sort=False).size().values

    X_main3 = df_main_i3[features_i3].astype("float32")
    y_main3 = df_main_i3["target_relevance"].astype(int).values
    X_val3  = df_val_i3[features_i3].astype("float32")
    y_val3  = df_val_i3["target_relevance"].astype(int).values
    X_all3  = trainable_i3[features_i3].astype("float32")
    y_all3  = trainable_i3["target_relevance"].astype(int).values

    model_i3, best_i3 = train_with_retrain(
        LGB_PARAMS_ITER3,
        X_main3, y_main3, X_val3, y_val3, X_all3, y_all3, features_i3,
        group_main=groups_main, group_val=groups_val, group_all=groups_all
    )

    # LambdaRank with small groups (~130 companies/day) stops too early.
    # If early stopping gave < 10 trees, use a minimum of 20 (the walk-forward average).
    if best_i3 < 10:
        log.info(f"  best_iter={best_i3} suspiciously low for LambdaRank — retraining with min 20")
        best_i3 = max(best_i3, 20)
        ds_all3 = lgb.Dataset(X_all3, label=y_all3, group=groups_all,
                              feature_name=features_i3, free_raw_data=False)
        model_i3 = lgb.train(LGB_PARAMS_ITER3, ds_all3, num_boost_round=best_i3)
    model_i3.save_model(str(MODEL_ITER3))
    log.info(f"  Saved: {MODEL_ITER3}")

    # Check iter3 IC on val (vs target_rank)
    pred_val_i3 = model_i3.predict(X_val3)
    ic_val_i3 = compute_ic(pred_val_i3, df_val_i3["target_rank_excess"].values, df_val_i3["date"])
    log.info(f"  IC check (val, excess):      {ic_val_i3:+.4f}")

    meta["iter3"] = {
        "features": features_i3,
        "best_iter": best_i3,
        "train_end": str(train_end_i3.date()),
        "val_ic_excess": round(ic_val_i3, 4) if not np.isnan(ic_val_i3) else None,
        "params": LGB_PARAMS_ITER3,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE METADATA
    # ══════════════════════════════════════════════════════════════════════════
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info(f"\n  Metadane: {META_JSON}")

    log.info("\n" + "=" * 70)
    log.info("DONE — final models saved")
    log.info("=" * 70)
    log.info(f"  Iter1 ({best_i1} drzew): {MODEL_ITER1}")
    log.info(f"  Iter3 ({best_i3} drzew): {MODEL_ITER3}")
    log.info(f"  Meta:                    {META_JSON}")
    log.info("")
    log.info("  Now run: python predict_today.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
