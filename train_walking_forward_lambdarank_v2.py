"""
Step 3B - Iteration 3: LambdaRank v2 (fixed hyperparameters)

What changes vs Iter 2 (diagnosis: the model stopped after 1-4 boosting rounds):
  - min_data_in_leaf: 300 -> 30  (the key fix: per-day groups have ~50-100
    companies, so 300 blocked useful tree splits)
  - NUM_RELEVANCE_BINS: 32 -> 5  (a simpler ranking problem: 5 classes instead of 32)
  - NUM_BOOST_ROUND: 500 -> 2000 (same as Iter 0; early_stopping=50 finds the optimum)

Everything else (excess target, walk-forward, feature selection, learning_rate=0.03) is unchanged.

Output:
  - phase3b_iter3_per_window_results.csv
  - phase3b_iter3_feature_importance.csv
  - phase3b_iter3_oos_predictions.parquet
  - phase3b_iter3_report.txt

Usage:
    python train_walking_forward_lambdarank_v2.py
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
INPUT_PARQUET = OUT_DIR / "dataset_ready_for_ml.parquet"

RESULTS_CSV = OUT_DIR / "phase3b_iter3_per_window_results.csv"
IMPORTANCE_CSV = OUT_DIR / "phase3b_iter3_feature_importance.csv"
OOS_PREDS_PARQUET = OUT_DIR / "phase3b_iter3_oos_predictions.parquet"
REPORT_TXT = OUT_DIR / "phase3b_iter3_report.txt"

BRANZA_TO_SUPER = {
    "Banki": "Finanse", "Ubezpieczenia": "Finanse", "Wierzytelnosci": "Finanse",
    "Rynek kapitalowy": "Finanse", "Dzialalnosc inwestycyjna": "Finanse",
    "Informatyka": "Tech_Gaming", "Gry": "Tech_Gaming", "Nowe technologie": "Tech_Gaming",
    "Gornictwo": "Surowce_Energia", "Hutnictwo": "Surowce_Energia",
    "Energia": "Surowce_Energia", "Paliwa i gaz": "Surowce_Energia",
    "Chemia": "Surowce_Energia",
    "Budownictwo": "Bud_Nieruch", "Nieruchomosci": "Bud_Nieruch",
    "Odziej i kosmetyki": "Konsument", "Handel internetowy": "Konsument",
    "Sieci handlowe": "Konsument", "Artykuly spozywcze": "Konsument",
    "Handel hurtowy": "Konsument", "Wyposazenie domu": "Konsument",
    "Rekreacja i wypoczynek": "Konsument",
    "Biotechnologia": "Healthcare", "Produkcja lekow": "Healthcare",
    "Sprzet i materialy medyczne": "Healthcare", "Dystrybucja lekow": "Healthcare",
    "Pozostale": "Healthcare",
    "Przemysl elektromaszynowy": "Przemysl", "Motoryzacja": "Przemysl",
    "Drewno i papier": "Przemysl", "Recykling": "Przemysl",
    "Transport i logistyka": "Przemysl",
    "Media": "Media_Telecom", "Telekomunikacja": "Media_Telecom",
    # original Polish spelling (both versions work, to be safe)
    "Wierzytelności": "Finanse", "Działalność inwestycyjna": "Finanse",
    "Górnictwo": "Surowce_Energia", "Nieruchomości": "Bud_Nieruch",
    "Odzież i kosmetyki": "Konsument", "Artykuły spożywcze": "Konsument",
    "Wyposażenie domu": "Konsument", "Produkcja leków": "Healthcare",
    "Sprzęt i materiały medyczne": "Healthcare", "Dystrybucja leków": "Healthcare",
    "Pozostałe": "Healthcare", "Przemysł elektromaszynowy": "Przemysl",
    "Rynek kapitałowy": "Finanse",
}

TRAIN_START = pd.Timestamp("2021-04-01")
WINDOWS = [
    {"train_end": "2023-03-31", "test_start": "2023-05-08", "test_end": "2023-10-31"},
    {"train_end": "2023-09-30", "test_start": "2023-11-06", "test_end": "2024-04-30"},
    {"train_end": "2024-03-31", "test_start": "2024-05-08", "test_end": "2024-10-31"},
    {"train_end": "2024-09-30", "test_start": "2024-11-06", "test_end": "2025-04-30"},
    {"train_end": "2025-03-31", "test_start": "2025-05-08", "test_end": "2025-10-31"},
    {"train_end": "2025-09-30", "test_start": "2025-11-06", "test_end": "2026-03-24"},
]

EXCLUDED_FEATURES = {
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "ma5_ma20_ratio", "close_ma20_ratio", "close_ma50_ratio",
    "macd_hist", "macd_hist_change", "rsi_change_5d",
    "atr_norm", "bb_width_norm", "volume_log_change_5d",
    "pos_52w", "gap", "up_streak",
    "body_norm", "upper_shadow", "lower_shadow",
    "is_monday", "is_friday",
    "rank_ret_60d", "rank_rsi_14",
}

TOP_K_FEATURES_PER_WINDOW = 30

# === KEY CHANGES vs Iter 2 ===
NUM_RELEVANCE_BINS = 5   # 32 -> 5: a simpler problem (bottom20/20-40/40-60/60-80/top20)

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "lambdarank_truncation_level": 10,  # top-10 (consistent with the top-5 portfolio)
    "label_gain": list(range(NUM_RELEVANCE_BINS)),  # [0,1,2,3,4]
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_data_in_leaf": 30,   # 300 -> 30: THE KEY FIX
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "force_col_wise": True,
    "seed": 42,
}

NUM_BOOST_ROUND = 2000   # 500 -> 2000
EARLY_STOPPING = 50
TOP_N_PORTFOLIO = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("walk_fwd_iter3")


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def add_super_sektor_and_excess_target(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Attaching super_sektor and computing the sector-neutral excess target...")
    df = df.copy()

    df["super_sektor"] = df["branza"].map(BRANZA_TO_SUPER)
    df["super_sektor"] = df["super_sektor"].fillna("Inne")

    valid_mask = (df["is_liquid"] == True) & df["target_return"].notna()
    valid_df = df.loc[valid_mask, ["date", "super_sektor", "target_return"]].copy()

    sector_avg = (valid_df.groupby(["date", "super_sektor"])["target_return"]
                  .mean()
                  .rename("sector_avg_return")
                  .reset_index())

    df = df.merge(sector_avg, on=["date", "super_sektor"], how="left")
    df["target_excess"] = df["target_return"] - df["sector_avg_return"]

    df["target_rank_excess"] = np.nan
    valid_for_rank = (df["is_liquid"] == True) & df["target_excess"].notna()
    valid_df_rank = df.loc[valid_for_rank, ["date", "target_excess"]].copy()
    ranks = valid_df_rank.groupby("date")["target_excess"].rank(pct=True)
    df.loc[valid_for_rank, "target_rank_excess"] = ranks.values

    df["target_relevance"] = np.nan
    df.loc[valid_for_rank, "target_relevance"] = (
        (df.loc[valid_for_rank, "target_rank_excess"] * NUM_RELEVANCE_BINS)
        .clip(0, NUM_RELEVANCE_BINS - 1)
        .astype(int)
    )

    n_excess = df["target_rank_excess"].notna().sum()
    log.info(f"  target_rank_excess for {n_excess:,} rows")
    log.info(f"  target_relevance: {NUM_RELEVANCE_BINS} bins (0=bottom, {NUM_RELEVANCE_BINS-1}=top)")

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "ticker", "yf_symbol", "date", "branza", "super_sektor",
        "period_end", "available_from",
        "open", "high", "low", "close", "volume",
        "turnover_pln", "turnover_30d_avg", "zero_vol_60d", "is_liquid",
        "target_return", "target_rank", "target_top20",
        "target_excess", "target_rank_excess", "target_relevance", "sector_avg_return",
        "split", "is_trainable", "feature_coverage",
    }
    raw_skewed = {
        "ps", "net_debt_ebitda", "net_fin_debt",
        "eps", "sps", "op_eps",
        "total_assets", "equity", "net_income", "shares", "market_cap",
    }
    feature_cols = []
    for c in df.columns:
        if c in exclude or c in raw_skewed:
            continue
        if c in EXCLUDED_FEATURES:
            continue
        feature_cols.append(c)
    return feature_cols


def compute_ic(pred: np.ndarray, target: np.ndarray, dates: pd.Series) -> dict:
    df = pd.DataFrame({"pred": pred, "target": target, "date": dates.values})
    daily_ic = []
    for date, group in df.groupby("date"):
        if len(group) >= 5:
            ic, _ = spearmanr(group["pred"], group["target"])
            if not np.isnan(ic):
                daily_ic.append(ic)
    if not daily_ic:
        return {"mean": np.nan, "std": np.nan, "ir": np.nan, "n_days": 0}
    arr = np.array(daily_ic)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ir": float(arr.mean() / arr.std()) if arr.std() > 0 else np.nan,
        "n_days": len(arr),
    }


def compute_topn_returns(pred: np.ndarray, returns: np.ndarray,
                         dates: pd.Series, top_n: int) -> dict:
    df = pd.DataFrame({"pred": pred, "ret": returns, "date": dates.values})
    daily = []
    for date, group in df.groupby("date"):
        if len(group) < top_n:
            continue
        top = group.nlargest(top_n, "pred")
        daily.append(top["ret"].mean())
    if not daily:
        return {"mean_return": np.nan, "n_days": 0, "win_rate": np.nan}
    arr = np.array(daily)
    return {
        "mean_return": float(arr.mean()),
        "median_return": float(np.median(arr)),
        "n_days": len(arr),
        "win_rate": float((arr > 0).mean()),
        "std": float(arr.std()),
    }


def universe_baseline(returns: np.ndarray, dates: pd.Series) -> dict:
    df = pd.DataFrame({"ret": returns, "date": dates.values})
    daily = df.groupby("date")["ret"].mean()
    return {
        "mean_return": float(daily.mean()),
        "median_return": float(daily.median()),
        "n_days": len(daily),
        "win_rate": float((daily > 0).mean()),
    }


def greenblatt_predictions(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d["pe_pos"] = d["pe"].where(d["pe"] > 0, np.inf).fillna(np.inf)
    d["roe_clean"] = d["roe"].fillna(d["roe"].median())
    rank_pe_inv = d.groupby("date")["pe_pos"].rank(pct=True, ascending=False)
    rank_roe = d.groupby("date")["roe_clean"].rank(pct=True, ascending=True)
    return (rank_pe_inv + rank_roe).values


def make_groups_for_lambdarank(df_in: pd.DataFrame) -> np.ndarray:
    """
    LambdaRank needs a 'group' parameter: a vector with the number of items
    in each group. Each unique 'date' is one ranking group.
    Requires df to be sorted by date.
    """
    if not df_in["date"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by date for LambdaRank")
    counts = df_in.groupby("date", sort=False).size().values
    return counts


# ─── WALKING-FORWARD ───────────────────────────────────────────────────────────

def train_one_window(df: pd.DataFrame, feature_cols_full: list[str],
                     window_idx: int, window: dict) -> dict:
    train_end = pd.Timestamp(window["train_end"])
    test_start = pd.Timestamp(window["test_start"])
    test_end = pd.Timestamp(window["test_end"])
    val_start = train_end - timedelta(days=90)

    log.info("=" * 70)
    log.info(f"WINDOW {window_idx + 1}/{len(WINDOWS)}")
    log.info(f"  Train: {TRAIN_START.date()} -> {val_start.date()}")
    log.info(f"  Val:   {val_start.date()} -> {train_end.date()}")
    log.info(f"  Test:  {test_start.date()} -> {test_end.date()}")
    log.info("=" * 70)

    trainable = df[
        (df["is_liquid"] == True)
        & df["target_relevance"].notna()
        & (df["feature_coverage"] >= 0.50)
    ].sort_values("date").reset_index(drop=True)

    train_mask = (trainable["date"] >= TRAIN_START) & (trainable["date"] < val_start)
    val_mask = (trainable["date"] >= val_start) & (trainable["date"] <= train_end)
    test_mask = (trainable["date"] >= test_start) & (trainable["date"] <= test_end)

    df_tr = trainable[train_mask].copy().reset_index(drop=True)
    df_va = trainable[val_mask].copy().reset_index(drop=True)
    df_te = trainable[test_mask].copy().reset_index(drop=True)

    log.info(f"  Rows: train={len(df_tr):,}  val={len(df_va):,}  test={len(df_te):,}")

    if len(df_tr) < 1000 or len(df_va) < 100 or len(df_te) < 100:
        log.warning(f"  Window {window_idx + 1}: not enough data - skipping")
        return None

    groups_tr = make_groups_for_lambdarank(df_tr)
    groups_va = make_groups_for_lambdarank(df_va)
    log.info(f"  Train groups (days): {len(groups_tr)}, avg group size: {groups_tr.mean():.1f}")
    log.info(f"  Val groups (days):   {len(groups_va)}, avg group size: {groups_va.mean():.1f}")

    # === STEP 1: all features for feature selection ===
    X_tr_full = df_tr[feature_cols_full].astype("float32")
    y_tr = df_tr["target_relevance"].astype(int).values
    X_va_full = df_va[feature_cols_full].astype("float32")
    y_va = df_va["target_relevance"].astype(int).values

    log.info(f"  Step 1: training on {len(feature_cols_full)} features for feature selection...")
    train_set = lgb.Dataset(X_tr_full, label=y_tr, group=groups_tr,
                            feature_name=feature_cols_full, free_raw_data=False)
    val_set = lgb.Dataset(X_va_full, label=y_va, group=groups_va,
                          reference=train_set, feature_name=feature_cols_full)

    model_full = lgb.train(
        LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )

    importance = model_full.feature_importance(importance_type="gain")
    fi_df = pd.DataFrame({"feature": feature_cols_full, "gain": importance})
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)
    selected_features = fi_df.head(TOP_K_FEATURES_PER_WINDOW)["feature"].tolist()
    log.info(f"  Step 1 best_iter={model_full.best_iteration}, selected top-{TOP_K_FEATURES_PER_WINDOW} features")

    # === STEP 2: retrain on the selected features ===
    X_tr = df_tr[selected_features].astype("float32")
    X_va = df_va[selected_features].astype("float32")
    X_te = df_te[selected_features].astype("float32")

    train_set = lgb.Dataset(X_tr, label=y_tr, group=groups_tr,
                            feature_name=selected_features, free_raw_data=False)
    val_set = lgb.Dataset(X_va, label=y_va, group=groups_va,
                          reference=train_set, feature_name=selected_features)

    model = lgb.train(
        LGB_PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    log.info(f"  Step 2 best_iter={model.best_iteration}")

    pred_te = model.predict(X_te, num_iteration=model.best_iteration)

    # === Metrics ===
    ic_test_excess = compute_ic(pred_te, df_te["target_rank_excess"].values, df_te["date"])
    ic_test_raw = compute_ic(pred_te, df_te["target_rank"].values, df_te["date"])

    universe = universe_baseline(df_te["target_return"].values, df_te["date"])
    topn = compute_topn_returns(pred_te, df_te["target_return"].values,
                                df_te["date"], TOP_N_PORTFOLIO)

    greenblatt_pred = greenblatt_predictions(df_te)
    ic_greenblatt = compute_ic(greenblatt_pred, df_te["target_rank"].values, df_te["date"])
    topn_greenblatt = compute_topn_returns(greenblatt_pred, df_te["target_return"].values,
                                           df_te["date"], TOP_N_PORTFOLIO)

    log.info(f"  RESULTS (TEST):")
    log.info(f"    Universe avg:       {universe['mean_return']:+.4f}  win_rate={universe['win_rate']:.1%}")
    log.info(f"    Model IC (excess):  {ic_test_excess['mean']:+.4f}  IR={ic_test_excess['ir']:+.3f}")
    log.info(f"    Model IC (raw):     {ic_test_raw['mean']:+.4f}")
    log.info(f"    Model top-5:        {topn['mean_return']:+.4f}  alpha={topn['mean_return'] - universe['mean_return']:+.4f}  win={topn['win_rate']:.1%}")
    log.info(f"    Greenblatt IC:      {ic_greenblatt['mean']:+.4f}")
    log.info(f"    Greenblatt top-5:   {topn_greenblatt['mean_return']:+.4f}  alpha={topn_greenblatt['mean_return'] - universe['mean_return']:+.4f}")

    model_wins_ic = ic_test_raw["mean"] > ic_greenblatt["mean"]
    model_wins_topn = topn["mean_return"] > topn_greenblatt["mean_return"]
    if model_wins_ic and model_wins_topn:
        log.info(f"    MODEL WINS (IC + top5)")
    elif model_wins_ic or model_wins_topn:
        log.info(f"    TIE (wins on one metric)")
    else:
        log.info(f"    Greenblatt wins")

    oos = df_te[["ticker", "date", "branza", "super_sektor", "close",
                 "target_return", "target_rank", "target_rank_excess"]].copy()
    oos["pred_ml"] = pred_te
    oos["pred_greenblatt"] = greenblatt_pred
    oos["window"] = window_idx + 1

    fi_selected = pd.DataFrame({
        "window": window_idx + 1,
        "feature": selected_features,
        "gain": model.feature_importance(importance_type="gain"),
    })
    fi_selected["gain_pct"] = 100 * fi_selected["gain"] / fi_selected["gain"].sum()

    return {
        "window": window_idx + 1,
        "train_end": train_end.date(),
        "test_start": test_start.date(),
        "test_end": test_end.date(),
        "n_train": len(df_tr),
        "n_val": len(df_va),
        "n_test": len(df_te),
        "best_iter": model.best_iteration,
        "model_ic_excess_mean": ic_test_excess["mean"],
        "model_ic_raw_mean": ic_test_raw["mean"],
        "model_top5_return": topn["mean_return"],
        "model_top5_winrate": topn["win_rate"],
        "model_top5_alpha": topn["mean_return"] - universe["mean_return"],
        "greenblatt_ic_mean": ic_greenblatt["mean"],
        "greenblatt_top5_return": topn_greenblatt["mean_return"],
        "greenblatt_top5_alpha": topn_greenblatt["mean_return"] - universe["mean_return"],
        "universe_mean_return": universe["mean_return"],
        "model_wins_ic": model_wins_ic,
        "model_wins_topn": model_wins_topn,
        "model_wins_both": model_wins_ic and model_wins_topn,
        "_oos_preds": oos,
        "_feature_importance": fi_selected,
    }


def main():
    log.info("=" * 70)
    log.info("STEP 3B - ITERATION 3: LAMBDARANK v2 (fixed hyperparameters)")
    log.info("=" * 70)
    log.info(f"  CHANGES vs Iter 2:")
    log.info(f"    - min_data_in_leaf: 300 -> 30  (fix: groups of ~50-100 companies per day)")
    log.info(f"    - NUM_RELEVANCE_BINS: 32 -> {NUM_RELEVANCE_BINS}  (a simpler ranking problem)")
    log.info(f"    - NUM_BOOST_ROUND: 500 -> {NUM_BOOST_ROUND}  (early stopping finds the optimum)")
    log.info("")

    log.info(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)

    df = add_super_sektor_and_excess_target(df)

    feature_cols = get_feature_cols(df)
    log.info(f"  Features: {len(feature_cols)}")
    log.info("")

    all_results = []
    all_oos_preds = []
    all_importance = []

    for i, window in enumerate(WINDOWS):
        result = train_one_window(df, feature_cols, i, window)
        if result is None:
            continue
        all_oos_preds.append(result.pop("_oos_preds"))
        all_importance.append(result.pop("_feature_importance"))
        all_results.append(result)
        log.info("")

    if not all_results:
        log.error("No window was trained!")
        return

    # === SUMMARY ===
    log.info("=" * 70)
    log.info("SUMMARY - ITERATION 3 (LambdaRank v2)")
    log.info("=" * 70)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_CSV, index=False)
    pd.concat(all_importance, ignore_index=True).to_csv(IMPORTANCE_CSV, index=False)
    pd.concat(all_oos_preds, ignore_index=True).to_parquet(OOS_PREDS_PARQUET, index=False)

    log.info(f"\n  Per-window results:")
    log.info(
        f"  {'Win':>4} {'Test period':>22} {'best_it':>7} {'IC exc':>8} {'IC raw':>7} "
        f"{'GB IC':>7} {'Top-5':>8} {'GB t5':>8} {'Verdict':>8}")
    log.info(f"  {'-'*4} {'-'*22} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    n_wins_ic = 0
    n_wins_topn = 0
    n_wins_both = 0
    for r in all_results:
        period = f"{r['test_start']}to{r['test_end']}"
        verdict = "WIN" if r["model_wins_both"] else (
            "tie" if r["model_wins_ic"] or r["model_wins_topn"] else "LOSS")
        log.info(
            f"  {r['window']:>4} {period:>22} {r['best_iter']:>7} "
            f"{r['model_ic_excess_mean']:>+8.4f} {r['model_ic_raw_mean']:>+7.4f} "
            f"{r['greenblatt_ic_mean']:>+7.4f} {r['model_top5_return']:>+8.4f} "
            f"{r['greenblatt_top5_return']:>+8.4f} {verdict:>8}")
        n_wins_ic += int(r["model_wins_ic"])
        n_wins_topn += int(r["model_wins_topn"])
        n_wins_both += int(r["model_wins_both"])

    avg_ic_raw = results_df["model_ic_raw_mean"].mean()
    avg_top5 = results_df["model_top5_return"].mean()

    log.info(f"\n  Averages across all windows (n={len(all_results)}):")
    log.info(f"    Model IC (excess):     {results_df['model_ic_excess_mean'].mean():+.4f}")
    log.info(f"    Model IC (raw):        {avg_ic_raw:+.4f}")
    log.info(f"    Greenblatt IC:         {results_df['greenblatt_ic_mean'].mean():+.4f}")
    log.info(f"    Model top-5 ret:       {avg_top5:+.4f}")
    log.info(f"    Greenblatt top-5:      {results_df['greenblatt_top5_return'].mean():+.4f}")
    log.info(f"    Universe avg:          {results_df['universe_mean_return'].mean():+.4f}")
    log.info(f"")
    log.info(f"    Model wins IC:    {n_wins_ic}/{len(all_results)}")
    log.info(f"    Model wins top-5: {n_wins_topn}/{len(all_results)}")
    log.info(f"    Model wins BOTH:  {n_wins_both}/{len(all_results)}")

    log.info("")
    log.info("=" * 70)
    log.info("VERDICT vs AMBITIOUS GOAL (IC raw>0.07 + beats Greenblatt >=4/6)")
    log.info("=" * 70)
    target_ic_met = avg_ic_raw > 0.07
    target_wins_met = n_wins_both >= 4
    log.info(f"  Mean raw IC > 0.07:     {avg_ic_raw:+.4f}  {'OK' if target_ic_met else 'X'}")
    log.info(f"  Wins (both) >=4/6:      {n_wins_both}/6   {'OK' if target_wins_met else 'X'}")
    log.info("")
    if target_ic_met and target_wins_met:
        log.info(f"  AMBITIOUS GOAL REACHED!")
    elif avg_ic_raw > 0.05 and n_wins_both >= 3:
        log.info(f"  Conservative goal reached.")
    elif avg_ic_raw > 0.03:
        log.info(f"  Weak signal.")
    else:
        log.info(f"  No signal.")

    log.info(f"")
    log.info(f"  COMPARISON OF ALL ITERATIONS:")
    log.info(f"  {'Iter':<8} {'IC raw':>9} {'wins both':>11} {'top-5 ret':>11}")
    log.info(f"  {'-'*8} {'-'*9} {'-'*11} {'-'*11}")
    log.info(f"  {'Iter 0':<8} {0.0687:>+9.4f} {3:>11}/6 {0.0176:>+11.4f}  (regressor + raw target)")
    log.info(f"  {'Iter 1':<8} {0.0515:>+9.4f} {1:>11}/6 {0.0463:>+11.4f}  (regressor + excess target)")
    log.info(f"  {'Iter 2':<8} {0.0266:>+9.4f} {1:>11}/6 {0.0241:>+11.4f}  (LambdaRank, min_leaf=300, bins=32)")
    log.info(f"  {'Iter 3':<8} {avg_ic_raw:>+9.4f} {n_wins_both:>11}/6 {avg_top5:>+11.4f}  (LambdaRank v2, min_leaf=30, bins=5)")

    log.info("\n  Top-15 features (mean gain across windows):")
    fi_all = pd.concat(all_importance, ignore_index=True)
    fi_avg = (fi_all.groupby("feature")
              .agg(mean_gain_pct=("gain_pct", "mean"),
                   appears_in_n=("window", "nunique"))
              .sort_values("mean_gain_pct", ascending=False)
              .head(15))
    for feat, row in fi_avg.iterrows():
        log.info(f"    {feat:30s}  avg_gain={row['mean_gain_pct']:>5.2f}%  "
                 f"(in {int(row['appears_in_n'])}/{len(all_results)} windows)")

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"Step 3B Iteration 3 - LambdaRank v2 (min_leaf=30, bins=5)\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Changes vs Iter 2:\n")
        f.write(f"  min_data_in_leaf: 300 -> 30\n")
        f.write(f"  NUM_RELEVANCE_BINS: 32 -> {NUM_RELEVANCE_BINS}\n")
        f.write(f"  NUM_BOOST_ROUND: 500 -> {NUM_BOOST_ROUND}\n\n")
        f.write(f"Per-window results:\n")
        f.write(results_df.to_string(index=False) + "\n\n")
        f.write(f"\nMean across windows:\n")
        f.write(f"  Model IC raw:     {avg_ic_raw:+.4f}\n")
        f.write(f"  Greenblatt IC:    {results_df['greenblatt_ic_mean'].mean():+.4f}\n")
        f.write(f"  Wins both:        {n_wins_both}/{len(all_results)}\n\n")
        f.write(f"\nComparison vs previous iterations:\n")
        f.write(f"  Iter 0: IC=+0.0687, wins=3/6, top-5=+0.0176\n")
        f.write(f"  Iter 1: IC=+0.0515, wins=1/6, top-5=+0.0463\n")
        f.write(f"  Iter 2: IC=+0.0266, wins=1/6, top-5=+0.0241\n")
        f.write(f"  Iter 3: IC={avg_ic_raw:+.4f}, wins={n_wins_both}/6, top-5={avg_top5:+.4f}\n")

    log.info(f"")
    log.info(f"  Per-window:        {RESULTS_CSV}")
    log.info(f"  Feature importance:{IMPORTANCE_CSV}")
    log.info(f"  OOS predictions:   {OOS_PREDS_PARQUET}")
    log.info(f"  Full report:       {REPORT_TXT}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
