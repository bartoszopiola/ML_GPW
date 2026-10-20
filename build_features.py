"""
Step 2B: Feature Engineering

Loads dataset_merged.parquet (171k rows, OHLCV + fundamentals + sector)
and computes ~75 ML features in 5 groups:

  A. Technical (~25)         - per-ticker time series (returns, MA, RSI, MACD, BB, ATR, vol)
  B. Fundamental level (~30) - raw metrics from the merge + log transforms for skewed ones
  C. Fundamental change (~9) - YoY and QoQ deltas on quarters (KEY: profitability trend,
                               growth, detecting dilution)
  D. Cross-sectional (~12)   - rank within the universe + rank within the sector, per day
  E. Date features (4)       - day_of_week, month, is_monday, is_friday

Output: dataset_features.parquet

Usage:
    python build_features.py
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
INPUT_PARQUET = OUT_DIR / "dataset_merged.parquet"
OUTPUT_PARQUET = OUT_DIR / "dataset_features.parquet"
FEATURE_LIST_TXT = OUT_DIR / "feature_list.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_features")

# Fundamental metrics for which we compute YoY/QoQ changes
CHANGE_METRICS = {
    "roe":         ("yoy", "qoq"),
    "roa":         ("yoy",),
    "margin_net":  ("yoy", "qoq"),
    "margin_op":   ("yoy",),
    "eps":         ("yoy", "qoq"),
    "sps":         ("yoy",),
    "shares":      ("yoy",),       # dilution!
    "pe":          ("qoq",),       # valuation re-rating
    "debt_to_equity": ("yoy",),    # rising debt
}

# Absolute (skewed) metrics → sign-preserving log transform
LOG_METRICS = ["market_cap", "total_assets", "equity", "net_income", "shares", "net_fin_debt"]

# Ranks within the whole universe and within the sector
RANK_METRICS_UNIVERSE = [
    "ret_5d", "ret_20d", "ret_60d",
    "pe", "pbv", "ev_ebitda",
    "roe", "roic", "margin_net",
    "market_cap", "volume_ratio", "rsi_14",
]
RANK_METRICS_BRANZA = [
    "pe", "pbv", "roe", "margin_net", "ret_20d", "market_cap",
]


# ─── A. TECHNICAL FEATURES (per ticker) ───────────────────────────────────────

def _compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes time-series features per ticker.
    Requires: df sorted by date, a single ticker.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # Mask: days with volume==0 are "silent" - returns from them should be NaN
    silent_day = (volume == 0)

    # === Returns ===
    for n in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{n}d"] = close.pct_change(n)
    # On silent days a zero return is misleading - mark it as NaN
    df.loc[silent_day, ["ret_1d", "ret_3d", "ret_5d"]] = np.nan

    # === Moving averages ratios ===
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma5_ma20_ratio"] = ma5 / ma20
    df["ma20_ma50_ratio"] = ma20 / ma50
    df["close_ma20_ratio"] = close / ma20
    df["close_ma50_ratio"] = close / ma50

    # === RSI ===
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_change_5d"] = df["rsi_14"] - df["rsi_14"].shift(5)

    # === MACD ===
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    df["macd_hist_change"] = df["macd_hist"].diff()
    df["macd_hist_norm"] = df["macd_hist"] / close

    # === Bollinger Bands ===
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width_norm"] = (bb_upper - bb_lower) / sma20

    # === ATR (volatility) ===
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_norm"] = atr / close

    # === Volume features ===
    vol_ma20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (vol_ma20 + 1e-9)
    df["volume_log_change_5d"] = np.log1p(volume) - np.log1p(volume.shift(5))
    df.loc[silent_day, "volume_ratio"] = np.nan

    # === 52-week position ===
    high_52w = close.rolling(252).max()
    low_52w = close.rolling(252).min()
    df["pos_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-9)

    # === Gap ===
    df["gap"] = open_ / close.shift() - 1

    # === Up streak ===
    is_up = (close.diff() > 0).astype(int)
    df["up_streak"] = is_up.groupby((is_up != is_up.shift()).cumsum()).cumcount() + 1
    df.loc[is_up == 0, "up_streak"] = 0

    # === Candlestick body/shadow (normalized) ===
    rng = (high - low + 1e-9)
    df["body_norm"] = (close - open_).abs() / rng
    df["upper_shadow"] = (high - df[["open", "close"]].max(axis=1)) / rng
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - low) / rng

    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("A. Technical features (per ticker)...")
    frames = []
    for ticker, group in tqdm(df.groupby("ticker", sort=False), desc="  Technical"):
        group = group.sort_values("date").reset_index(drop=True)
        frames.append(_compute_technical(group))
    return pd.concat(frames, ignore_index=True)


# ─── B. FUNDAMENTAL LEVEL FEATURES ─────────────────────────────────────────────

def _signed_log(x):
    """Sign-preserving log: sign(x) * log1p(|x|). Works for negative values."""
    return np.sign(x) * np.log1p(np.abs(x))


def add_fundamental_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log transforms for skewed fundamentals.
    The raw fundamental values are already in df (from the merge step).
    """
    log.info("B. Fundamental level features (log transforms)...")
    df = df.copy()
    for col in LOG_METRICS:
        if col in df.columns:
            df[f"log_{col}"] = _signed_log(df[col])
    log.info(f"  Added {len([c for c in LOG_METRICS if c in df.columns])} log-transforms")
    return df


# ─── C. FUNDAMENTAL CHANGE FEATURES (per ticker, on quarters) ──────────────────

def _compute_change_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each company, computes QoQ and YoY changes at the quarter level,
    then propagates them to the daily rows.

    Note: computed on unique (ticker, period_end) pairs, NOT on daily rows
    (otherwise the forward fill would smear the shift and create a fake gradient).
    """
    log.info("C. Fundamental change features (YoY, QoQ)...")

    # Extract unique (ticker, period_end, [metrics]) and sort by quarter
    metric_cols = list({m for m in CHANGE_METRICS.keys() if m in merged.columns})
    if not metric_cols:
        log.warning("  No metrics to compute changes for")
        return merged

    quarterly = (merged[["ticker", "period_end"] + metric_cols]
                 .dropna(subset=["period_end"])
                 .drop_duplicates(subset=["ticker", "period_end"])
                 .sort_values(["ticker", "period_end"])
                 .reset_index(drop=True))
    log.info(f"  Unique (ticker × quarter): {len(quarterly):,}")

    # Per ticker, compute shift(-1) on quarters (QoQ) and shift(-4) (YoY)
    # NB: shift within a group in pandas - we use groupby
    # SAFE_DIV_THRESHOLD: if the denominator (previous period) is close to zero,
    # pct_change would be Inf - return NaN instead
    SAFE_DIV_THRESHOLD = 0.01

    def _safe_pct_change(numer: pd.Series, denom: pd.Series) -> pd.Series:
        """Returns numer/denom - 1, but NaN if |denom| < threshold."""
        result = numer / denom - 1
        # Mask rows where denom is close to zero or NaN/Inf
        bad_mask = denom.abs() < SAFE_DIV_THRESHOLD
        result = result.where(~bad_mask, np.nan)
        # Also: replace any remaining Inf/-Inf with NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        return result

    new_cols = {}
    for metric, change_types in CHANGE_METRICS.items():
        if metric not in metric_cols:
            continue

        if "qoq" in change_types:
            # QoQ change: current quarter vs the previous one
            grouped = quarterly.groupby("ticker", sort=False)[metric]
            # For ratios: plain difference (safe, no division)
            # For absolute values (eps, shares): pct change with safe division
            if metric in ("eps", "sps", "shares"):
                new_cols[f"{metric}_qoq"] = _safe_pct_change(quarterly[metric], grouped.shift(1))
            else:
                new_cols[f"{metric}_qoq"] = quarterly[metric] - grouped.shift(1)

        if "yoy" in change_types:
            grouped = quarterly.groupby("ticker", sort=False)[metric]
            if metric in ("eps", "sps", "shares"):
                new_cols[f"{metric}_yoy"] = _safe_pct_change(quarterly[metric], grouped.shift(4))
            else:
                new_cols[f"{metric}_yoy"] = quarterly[metric] - grouped.shift(4)

    # Attach the change columns to quarterly
    for col, vals in new_cols.items():
        quarterly[col] = vals

    # Winsorize change features: pct_change can have extreme values
    # (when the previous period was close to zero but > threshold). Trees are robust, but
    # we cap extreme values so they don't distort the model's ranking.
    PCT_CHANGE_CLIP = 5.0  # ±500% is a practical maximum for a meaningful change
    pct_change_features = [c for c in new_cols.keys()
                           if any(c.startswith(p) for p in ("eps_", "sps_", "shares_"))]
    for c in pct_change_features:
        quarterly[c] = quarterly[c].clip(lower=-PCT_CHANGE_CLIP, upper=PCT_CHANGE_CLIP)

    # Now: merge the newly computed change_metrics with merged on (ticker, period_end)
    change_cols = list(new_cols.keys())
    quarterly_changes = quarterly[["ticker", "period_end"] + change_cols]

    result = merged.merge(quarterly_changes, on=["ticker", "period_end"], how="left")
    log.info(f"  Added {len(change_cols)} change features: {sorted(change_cols)}")
    return result


def add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    return _compute_change_metrics(df)


# ─── D. CROSS-SECTIONAL FEATURES ───────────────────────────────────────────────

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per day (and optionally per sector), computes the percentile rank of each metric.
    Computed ONLY among rows with is_liquid=True (so the ranking isn't
    distorted by illiquid companies).
    """
    log.info("D. Cross-sectional features (rank within universe + sector)...")
    df = df.copy()

    # Liquid mask - only these rows take part in the ranking
    liquid_mask = df["is_liquid"] == True

    # Ranking within the whole universe (per day)
    for metric in RANK_METRICS_UNIVERSE:
        if metric not in df.columns:
            continue
        rank_col = f"rank_{metric}"
        # Initialize with NaN
        df[rank_col] = np.nan
        # Rank only among liquid rows, the rest stays NaN
        liquid_df = df.loc[liquid_mask, ["date", metric]].copy()
        ranks = liquid_df.groupby("date")[metric].rank(pct=True)
        df.loc[liquid_mask, rank_col] = ranks.values

    # Ranking within the sector (per day × sector)
    for metric in RANK_METRICS_BRANZA:
        if metric not in df.columns:
            continue
        rank_col = f"rank_{metric}_branza"
        df[rank_col] = np.nan
        liquid_df = df.loc[liquid_mask, ["date", "branza", metric]].copy()
        ranks = liquid_df.groupby(["date", "branza"])[metric].rank(pct=True)
        df.loc[liquid_mask, rank_col] = ranks.values

    n_universe = sum(1 for m in RANK_METRICS_UNIVERSE if m in df.columns)
    n_branza = sum(1 for m in RANK_METRICS_BRANZA if m in df.columns)
    log.info(f"  Added {n_universe} universe ranks + {n_branza} sector ranks")
    return df


# ─── E. DATE FEATURES ──────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("E. Date features...")
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    return df


# ─── DIAGNOSTICS ───────────────────────────────────────────────────────────────

def feature_coverage_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature coverage report, liquid days only."""
    liquid = df[df["is_liquid"] == True]
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        n_total = len(liquid)
        n_notna = liquid[col].notna().sum()
        pct = 100 * n_notna / n_total if n_total > 0 else 0
        rows.append({
            "feature": col,
            "n_values_liquid": int(n_notna),
            "n_total_liquid": int(n_total),
            "pct_coverage_liquid": round(pct, 1),
            "median": float(liquid[col].median()) if n_notna > 0 else np.nan,
            "p1": float(liquid[col].quantile(0.01)) if n_notna > 0 else np.nan,
            "p99": float(liquid[col].quantile(0.99)) if n_notna > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("STEP 2B: FEATURE ENGINEERING")
    log.info("=" * 70)

    # Load
    log.info(f"Loading {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    log.info(f"  {len(df):,} rows, {df['ticker'].nunique()} companies, "
             f"{df['date'].min().date()} → {df['date'].max().date()}")

    # Number of input columns (original - base)
    base_cols = list(df.columns)

    # A. Technical
    df = add_technical_features(df)

    # B. Fundamental level (log transforms)
    df = add_fundamental_level_features(df)

    # C. Fundamental change
    df = add_change_features(df)

    # D. Cross-sectional ranks (must come after everything else, so ranks cover all metrics)
    df = add_cross_sectional_features(df)

    # E. Date features
    df = add_date_features(df)

    # === SAFETY: replace any remaining Inf/-Inf with NaN ===
    # LightGBM doesn't handle Inf; NaN is fine.
    feature_cols_so_far = [c for c in df.columns if c not in base_cols]
    inf_count = 0
    for c in feature_cols_so_far:
        if df[c].dtype.kind == 'f':  # float columns only
            n_inf = np.isinf(df[c]).sum()
            if n_inf > 0:
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)
                inf_count += n_inf
    if inf_count > 0:
        log.info(f"  Safety: replaced {inf_count} Inf values with NaN")

    # === Feature list (all new columns added on top of the base ones) ===
    all_cols = list(df.columns)
    feature_cols = [c for c in all_cols if c not in base_cols]

    # === Save ===
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log.info(f"\n✓ Saved: {OUTPUT_PARQUET}")

    # === Write the feature list to txt ===
    with FEATURE_LIST_TXT.open("w", encoding="utf-8") as f:
        f.write(f"# Feature list ({len(feature_cols)})\n")
        f.write(f"# Generated: {df['date'].max().date()}\n\n")
        for c in feature_cols:
            f.write(f"{c}\n")

    # === Coverage report ===
    cov = feature_coverage_report(df, feature_cols)
    cov_csv = OUT_DIR / "feature_coverage.csv"
    cov.to_csv(cov_csv, index=False)

    # === SUMMARY ===
    log.info("=" * 70)
    log.info("REPORT")
    log.info("=" * 70)
    log.info(f"  Total rows:           {len(df):,}")
    log.info(f"  Liquid rows:          {df['is_liquid'].sum():,} "
             f"({100*df['is_liquid'].mean():.1f}%)")
    log.info(f"  Total features:       {len(feature_cols)}")
    log.info(f"  Base columns:         {len(base_cols)}")
    log.info(f"  All columns:          {len(all_cols)}")
    log.info(f"")

    # Feature groups
    technical_features = [c for c in feature_cols if any(c.startswith(p) for p in
        ["ret_", "ma", "close_ma", "rsi", "macd", "bb_", "atr_", "volume", "pos_52", "gap",
         "up_streak", "body_", "upper_", "lower_"])]
    log_features = [c for c in feature_cols if c.startswith("log_")]
    change_features = [c for c in feature_cols if c.endswith("_yoy") or c.endswith("_qoq")]
    rank_features = [c for c in feature_cols if c.startswith("rank_")]
    rank_branza = [c for c in rank_features if c.endswith("_branza")]
    rank_universe = [c for c in rank_features if not c.endswith("_branza")]
    date_features = [c for c in feature_cols if c in
        ["day_of_week", "month", "is_monday", "is_friday"]]

    log.info(f"  Feature groups:")
    log.info(f"    A. Technical:               {len(technical_features):3d}")
    log.info(f"    B. Log transforms:          {len(log_features):3d}")
    log.info(f"    C. Fundamental changes:     {len(change_features):3d}")
    log.info(f"    D. Rank in universe:        {len(rank_universe):3d}")
    log.info(f"       Rank in sector:          {len(rank_branza):3d}")
    log.info(f"    E. Date:                    {len(date_features):3d}")
    log.info(f"    Razem:                      {len(feature_cols):3d}")
    log.info(f"")

    # Coverage warnings: features with <50% coverage on liquid days
    poor_coverage = cov[cov["pct_coverage_liquid"] < 50]
    if len(poor_coverage) > 0:
        log.warning(f"  Features with <50% coverage (on liquid days):")
        for _, row in poor_coverage.iterrows():
            log.warning(f"    {row['feature']:30s} {row['pct_coverage_liquid']:.1f}%")
    else:
        log.info(f"  All features have >=50% coverage on liquid days ✓")

    # Sanity check: show 10 selected features for KGHM 2024-09-16
    log.info(f"\n  Sanity check: KGHM ~2024-09-15:")
    kghm = df[df["ticker"] == "KGHM"].copy()
    target = pd.Timestamp("2024-09-15")
    kghm["dist"] = (kghm["date"] - target).abs()
    if len(kghm) > 0:
        s = kghm.sort_values("dist").iloc[0]
        log.info(f"    date={s['date'].date()}  close={s['close']:.2f}")
        sample_features = [
            "ret_5d", "ret_20d", "rsi_14", "macd_hist_norm", "bb_position",
            "volume_ratio", "log_market_cap",
            "roe_yoy", "margin_net_yoy", "shares_yoy",
            "rank_pe", "rank_pe_branza", "rank_roe_branza",
        ]
        for f in sample_features:
            if f in df.columns:
                v = s.get(f, None)
                if pd.notna(v):
                    log.info(f"    {f:25s} = {v:.4f}")
                else:
                    log.info(f"    {f:25s} = NaN")

    log.info(f"")
    log.info(f"  Output:               {OUTPUT_PARQUET}")
    log.info(f"  Feature list:         {FEATURE_LIST_TXT}")
    log.info(f"  Coverage report:      {cov_csv}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()