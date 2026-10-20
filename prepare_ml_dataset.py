"""
Step 2C: target + train/val/test split + final ML dataset

Loads dataset_features.parquet and:
  1. Computes the target — 20-day open-to-open forward return
  2. Computes target_rank (percentile among liquid companies on a given day)
  3. Computes target_top20 (binary flag for the top 20% of the ranking)
  4. Defines a time-based split: train / validation / test (with 25-day gaps)
  5. Filters "tradeable" rows (liquid + has a target + minimum feature coverage)
  6. Output: dataset_ready_for_ml.parquet + report

Split configuration:
  - Train:      2021-04-01 → 2024-09-30
  - Gap:        25 trading days
  - Validation: 2024-10-30 → 2025-04-30
  - Gap:        25 trading days
  - Test:       2025-06-04 → 2026-04-04 (up to 20 days before the end of available data)
  - Holdout:    2026-04-05 → 2026-04-24 (no target, "for predicting the future")

Usage:
    python prepare_ml_dataset.py
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
INPUT_PARQUET = OUT_DIR / "dataset_features.parquet"
OUTPUT_PARQUET = OUT_DIR / "dataset_ready_for_ml.parquet"
SPLIT_REPORT = OUT_DIR / "split_report.csv"

# Target
TARGET_HORIZON = 20  # trading days
TARGET_TOP_PCT = 0.20  # top 20% of the ranking = 1, the rest = 0

# Splits — dates in YYYY-MM-DD format
TRAIN_START = "2021-04-01"
TRAIN_END = "2024-09-30"
VAL_START = "2024-10-30"  # ~25 trading days after TRAIN_END
VAL_END = "2025-04-30"
TEST_START = "2025-06-04"  # ~25 trading days after VAL_END
# TEST_END = max(date) - 20 trading days (computed dynamically)

# Filtering: minimum % of non-NaN features for a row to be used in training
MIN_FEATURE_COVERAGE = 0.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_ml")


# ─── STEP 1: TARGET ────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the target as the open-to-open forward return over TARGET_HORIZON days.

    Realistic: the model gets the signal at the end of day t (close[t]),
    buys at the open of day t+1 (open[t+1]) and sells at the open of day t+1+H (open[t+1+H]).

    target_return = open[t+1+H] / open[t+1] - 1
    """
    log.info(f"Computing target (open-to-open, horizon={TARGET_HORIZON} days)...")
    df = df.sort_values(["ticker", "date"]).copy()

    # buy_open  = tomorrow's open = open shifted by -1
    # sell_open = open in 1+H days = open shifted by -(1+H)
    df["buy_open"] = df.groupby("ticker", sort=False)["open"].shift(-1)
    df["sell_open"] = df.groupby("ticker", sort=False)["open"].shift(-(1 + TARGET_HORIZON))

    df["target_return"] = df["sell_open"] / df["buy_open"] - 1

    # Clean up helper columns
    df = df.drop(columns=["buy_open", "sell_open"])

    n_with_target = df["target_return"].notna().sum()
    log.info(f"  Rows with a target: {n_with_target:,}/{len(df):,} ({100 * n_with_target / len(df):.1f}%)")
    return df


def add_target_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    target_rank = percentile rank of target_return among liquid companies on a given day.
    target_top20 = 1 if rank >= 0.80, 0 otherwise.

    KEY: the rank is computed ONLY among rows where:
      - is_liquid == True
      - target_return is not NaN
    All other rows → NaN.
    """
    log.info("Computing target_rank (percentile) and target_top20 (binary)...")
    df = df.copy()

    # Mask: liquid only + has a target
    valid_mask = (df["is_liquid"] == True) & df["target_return"].notna()

    # Initialize with NaN
    df["target_rank"] = np.nan
    df["target_top20"] = np.nan

    # Rank among liquid rows
    valid_df = df.loc[valid_mask, ["date", "target_return"]].copy()
    ranks = valid_df.groupby("date")["target_return"].rank(pct=True)
    df.loc[valid_mask, "target_rank"] = ranks.values
    df.loc[valid_mask, "target_top20"] = (df.loc[valid_mask, "target_rank"] >= (1 - TARGET_TOP_PCT)).astype(float)

    n_rank = df["target_rank"].notna().sum()
    n_top = (df["target_top20"] == 1).sum()
    log.info(f"  target_rank for {n_rank:,} rows")
    log.info(
        f"  target_top20=1 for {n_top:,} rows ({100 * n_top / n_rank:.1f}% of ranked — sanity: should be ~{TARGET_TOP_PCT * 100:.0f}%)")
    return df


# ─── STEP 2: SPLITS ────────────────────────────────────────────────────────────

def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns each row a label: 'train', 'val', 'test', 'gap', 'holdout'.
    """
    log.info("Assigning time-based splits...")
    df = df.copy()

    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)

    # TEST_END = last date with a meaningful target (i.e. max_date minus a 20+1 trading-day margin)
    # For simplicity: a date with an available target_return (not NaN, not holdout)
    # Initialize everything as 'gap' first — a safe default
    df["split"] = "gap"

    # Train
    df.loc[(df["date"] >= train_start) & (df["date"] <= train_end), "split"] = "train"

    # Validation
    df.loc[(df["date"] >= val_start) & (df["date"] <= val_end), "split"] = "val"

    # Test: from TEST_START up to the last date where the target is still available
    last_with_target = df.loc[df["target_return"].notna(), "date"].max()
    df.loc[(df["date"] >= test_start) & (df["date"] <= last_with_target), "split"] = "test"

    # Holdout: after the last date with a target (the model can predict on these, but they can't be evaluated)
    df.loc[df["date"] > last_with_target, "split"] = "holdout"

    # Statistics
    counts = df["split"].value_counts()
    for split, n in counts.items():
        dates = df.loc[df["split"] == split, "date"]
        if len(dates) > 0:
            log.info(f"  {split:8s}  {n:>7,} rows  {dates.min().date()} → {dates.max().date()}")
        else:
            log.info(f"  {split:8s}  {n:>7,} rows")

    return df


# ─── STEP 3: TRAINING FILTERS ──────────────────────────────────────────────────

def add_trainable_flag(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    A row is 'trainable' if:
      - is_liquid == True
      - target_rank is not NaN
      - at least MIN_FEATURE_COVERAGE of the features are non-NaN
    """
    log.info("Computing feature coverage per row...")
    df = df.copy()

    # Number of non-NaN features per row
    feature_data = df[feature_cols]
    n_features = len(feature_cols)
    df["feature_coverage"] = feature_data.notna().sum(axis=1) / n_features

    df["is_trainable"] = (
            (df["is_liquid"] == True)
            & df["target_rank"].notna()
            & (df["feature_coverage"] >= MIN_FEATURE_COVERAGE)
    )

    n_trainable = df["is_trainable"].sum()
    log.info(f"  Trainable rows: {n_trainable:,} (of {len(df):,}, {100 * n_trainable / len(df):.1f}%)")
    log.info(f"  Median feature coverage: {df['feature_coverage'].median():.1%}")

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    List of feature columns used for training.
    Skips: base columns (ticker, date, OHLCV), targets, splits, raw fundamentals,
    metadata (yf_symbol, branza, period_end etc.)
    """
    exclude = {
        # identifiers and time
        "ticker", "yf_symbol", "date", "branza",
        "period_end", "available_from",
        # OHLCV raw
        "open", "high", "low", "close", "volume",
        "turnover_pln", "turnover_30d_avg", "zero_vol_60d", "is_liquid",
        # Targets and metadata
        "target_return", "target_rank", "target_top20",
        "split", "is_trainable", "feature_coverage",
    }
    # Raw fundamental metrics are skipped too (their log/change versions are in the features)
    raw_fund = {
        "pe", "pbv", "ps", "ev_ebitda", "ev_sales",
        "roe", "roa", "roic", "margin_net", "margin_op",
        "debt_total", "debt_to_equity", "net_debt_ebitda", "net_fin_debt",
        "current_ratio", "quick_ratio", "cash_ratio",
        "eps", "bvps", "sps", "op_eps",
        "total_assets", "equity", "net_income", "shares", "market_cap",
    }
    # But some of them we DO want as features (the ones without log/change versions)
    # i.e. all RAW ratios - they are good features on their own!
    # We only exclude the skewed ones (which have log_*) and those with *_yoy/*_qoq versions
    keep_raw_fund = {
        "pe", "pbv", "ps", "ev_ebitda", "ev_sales",
        "roe", "roa", "roic", "margin_net", "margin_op",
        "debt_total", "debt_to_equity", "net_debt_ebitda",
        "current_ratio", "quick_ratio", "cash_ratio",
        "bvps", "op_eps",
    }

    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c in raw_fund and c not in keep_raw_fund:
            continue  # raw absolute values (eps, sps, total_assets, market_cap, shares, equity, net_income, net_fin_debt)
        feature_cols.append(c)

    return feature_cols


# ─── STEP 4: SPLIT REPORT ──────────────────────────────────────────────────────

def make_split_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Diagnostic report per split."""
    rows = []
    for split in ["train", "val", "test", "gap", "holdout"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            continue
        trainable = sub[sub.get("is_trainable", False) == True]
        rows.append({
            "split": split,
            "rows_total": len(sub),
            "rows_trainable": len(trainable),
            "tickers": sub["ticker"].nunique(),
            "tickers_trainable": trainable["ticker"].nunique(),
            "date_min": sub["date"].min().date(),
            "date_max": sub["date"].max().date(),
            "median_target_return": float(trainable["target_return"].median()) if len(trainable) > 0 else np.nan,
            "n_target_top20": int((trainable["target_top20"] == 1).sum()) if len(trainable) > 0 else 0,
        })
    return pd.DataFrame(rows)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("STEP 2C: TARGET + SPLIT + FINALIZATION")
    log.info("=" * 70)

    # Load
    log.info(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    log.info(f"  {len(df):,} rows, {df['ticker'].nunique()} companies, "
             f"{df['date'].min().date()} → {df['date'].max().date()}")

    # 1. Target
    df = add_target(df)
    df = add_target_rank(df)

    # 2. Splits
    df = assign_split(df)

    # 3. Trainable flag
    feature_cols = get_feature_cols(df)
    log.info(f"\nFeature list ({len(feature_cols)}):")
    log.info(f"  {feature_cols}")
    df = add_trainable_flag(df, feature_cols)

    # 4. Save
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log.info(f"\n✓ Saved: {OUTPUT_PARQUET}")

    # 5. Split report
    report = make_split_report(df, feature_cols)
    report.to_csv(SPLIT_REPORT, index=False)

    # === SUMMARY ===
    log.info("=" * 70)
    log.info("FINAL REPORT")
    log.info("=" * 70)
    log.info(f"  Total rows:              {len(df):,}")
    log.info(f"  Trainable rows:          {df['is_trainable'].sum():,}")
    log.info(f"  ML features:             {len(feature_cols)}")
    log.info(f"")
    log.info(f"  Splity:")
    for _, row in report.iterrows():
        log.info(f"    {row['split']:8s}  total={row['rows_total']:>7,}  trainable={row['rows_trainable']:>7,}  "
                 f"tickers={row['tickers']:>3}  {row['date_min']} → {row['date_max']}")
    log.info(f"")

    # Sanity check of median target returns per split
    log.info(f"  Median target_return per split (trainable only):")
    for _, row in report.iterrows():
        if pd.notna(row["median_target_return"]):
            log.info(f"    {row['split']:8s}  median={row['median_target_return']:+.4f}  "
                     f"n_top20={row['n_target_top20']:,}")
    log.info(f"")

    # Sanity: target_rank distribution
    trainable = df[df["is_trainable"] == True]
    if len(trainable) > 0:
        log.info(f"  target_rank distribution (trainable):")
        log.info(f"    min={trainable['target_rank'].min():.4f}  "
                 f"p25={trainable['target_rank'].quantile(0.25):.4f}  "
                 f"med={trainable['target_rank'].median():.4f}  "
                 f"p75={trainable['target_rank'].quantile(0.75):.4f}  "
                 f"max={trainable['target_rank'].max():.4f}")

        # Feature coverage for trainable rows
        log.info(f"")
        log.info(f"  Feature coverage (trainable):")
        log.info(f"    median: {trainable['feature_coverage'].median():.1%}")
        log.info(f"    p25: {trainable['feature_coverage'].quantile(0.25):.1%}")
        log.info(f"    min: {trainable['feature_coverage'].min():.1%}")

    # Last sanity check: does KGHM have trainable rows in every split?
    log.info(f"")
    log.info(f"  Sanity check: KGHM across splits:")
    for split in ["train", "val", "test", "holdout"]:
        sub = df[(df["ticker"] == "KGHM") & (df["split"] == split)]
        trainable_n = sub.get("is_trainable", pd.Series(dtype=bool)).sum() if len(sub) > 0 else 0
        log.info(f"    {split:8s}  total={len(sub):>5}  trainable={trainable_n:>5}")

    log.info(f"")
    log.info(f"  Output:               {OUTPUT_PARQUET}")
    log.info(f"  Split report:         {SPLIT_REPORT}")
    log.info("=" * 70)
    log.info(f"\n  ✓ PHASE 2 COMPLETE. Dataset ready for ML.")
    log.info(f"  → Next step: Phase 3 (LightGBM ranker + walk-forward backtest)")


if __name__ == "__main__":
    main()