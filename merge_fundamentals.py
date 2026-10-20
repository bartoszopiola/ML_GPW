"""
Step 2A: merge OHLCV + fundamentals + sector into one wide-format dataset.

What it does:
  1. Loads:
     - dataset_raw.parquet (OHLCV, ~171k rows, daily)
     - dataset_fundamentals_long.parquet (~474k rows, quarterly, long format)
     - fundamentals_fetch_report.csv (sector per company)
  2. Filters fundamentals down to 25 selected numeric metrics
  3. Filters out pre-IPO data (Kurs=0, Liczba akcji=0)
  4. Pivots long → wide (one metric = one column)
  5. Winsorizes outliers (1st-99th percentile per metric)
  6. As-of merge: for every OHLCV day takes the latest available quarter
     (available_from <= date), per ticker
  7. Computes market_cap = close × shares (TODAY's price × shares from the latest report)
  8. Adds the sector as a categorical column
  9. Output: dataset_merged.parquet + diagnostic report

Usage:
    python merge_fundamentals.py
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")

OHLCV_PARQUET = OUT_DIR / "dataset_raw.parquet"
FUNDAMENTALS_LONG = OUT_DIR / "dataset_fundamentals_long.parquet"
BRANZA_REPORT = OUT_DIR / "fundamentals_fetch_report.csv"

OUTPUT_PARQUET = OUT_DIR / "dataset_merged.parquet"
DIAG_REPORT = OUT_DIR / "merge_diagnostics.csv"

# Mapping: Polish biznesradar name → short English name (ASCII)
METRIC_MAPPING = {
    # Market valuation (5)
    "Cena / Zysk":                   "pe",
    "Cena / Wartość księgowa":       "pbv",
    "Cena / Przychody ze sprzedaży": "ps",
    "EV / EBITDA":                   "ev_ebitda",
    "EV / Przychody ze sprzedaży":   "ev_sales",
    # Profitability (5)
    "ROE":                           "roe",
    "ROA":                           "roa",
    "ROIC":                          "roic",
    "Marża zysku netto":             "margin_net",
    "Marża zysku operacyjnego":      "margin_op",
    # Debt (4)
    "Zadłużenie ogólne":             "debt_total",
    "Zadłużenie kapitału własnego":  "debt_to_equity",
    "Zadłużenie netto / EBITDA":     "net_debt_ebitda",
    "Zadłużenie finansowe netto":    "net_fin_debt",
    # Liquidity (3)
    "Płynność bieżąca":              "current_ratio",
    "Płynność szybka":               "quick_ratio",
    "Płynność gotówkowa":            "cash_ratio",
    # Per share (4)
    "Zysk na akcję":                 "eps",
    "Wartość księgowa na akcję":     "bvps",
    "Przychody ze sprzedaży na akcję": "sps",
    "Zysk operacyjny na akcję":      "op_eps",
    # Absolute values (4)
    "Aktywa razem":                  "total_assets",
    "Kapitał własny akcjonariuszy jednostki dominującej": "equity",
    "Zysk netto":                    "net_income",
    "Liczba akcji":                  "shares",
}

# Winsorization percentiles - quality vs extreme outliers
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# Metrics where 0 is a meaningful value (not an outlier)
ZERO_OK_METRICS = {"net_fin_debt", "net_income"}

# Metrics whose pre-IPO value of 0 should become NaN (pre-IPO filter)
PREIPO_FILTER_METRICS = ["Kurs", "Liczba akcji"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_fund")


# ─── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────

def load_data():
    log.info("Loading OHLCV...")
    ohlcv = pd.read_parquet(OHLCV_PARQUET)
    log.info(f"  OHLCV: {len(ohlcv):,} rows, {ohlcv['ticker'].nunique()} companies, "
             f"{ohlcv['date'].min().date()} → {ohlcv['date'].max().date()}")

    log.info("Loading fundamentals (long)...")
    fund_long = pd.read_parquet(FUNDAMENTALS_LONG)
    log.info(f"  Fund long: {len(fund_long):,} rows, {fund_long['ticker'].nunique()} companies, "
             f"{fund_long['metric'].nunique()} metrics")

    log.info("Loading sectors...")
    branza_df = pd.read_csv(BRANZA_REPORT)[["ticker", "branza"]].copy()
    branza_df["branza"] = branza_df["branza"].fillna("Inne").astype(str).str.strip()
    log.info(f"  Sectors: {branza_df['branza'].nunique()} unique across {len(branza_df)} companies")

    return ohlcv, fund_long, branza_df


# ─── STEP 2: FILTER AND PIVOT FUNDAMENTALS ─────────────────────────────────────

def filter_preipo(fund_long: pd.DataFrame) -> pd.DataFrame:
    """
    Removes quarters where Kurs=0 or Liczba akcji=0 (pre-IPO).
    Whole quarters (company × period_end) are dropped to keep things consistent.
    """
    bad = fund_long[
        (fund_long["metric"].isin(PREIPO_FILTER_METRICS)) &
        (fund_long["value"] == 0)
    ][["ticker", "period_end"]].drop_duplicates()

    if len(bad) == 0:
        return fund_long

    # Anti-join: keep only the (ticker, period_end) pairs that are NOT in bad
    merged = fund_long.merge(
        bad.assign(_drop=True), on=["ticker", "period_end"], how="left"
    )
    cleaned = merged[merged["_drop"].isna()].drop(columns="_drop")
    log.info(f"  Pre-IPO filter: removed {len(bad)} (ticker × period_end) → "
             f"{len(fund_long) - len(cleaned):,} rows")
    return cleaned


def pivot_fundamentals(fund_long: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: each metric becomes a column."""
    # Keep only the selected metrics
    target_metrics = list(METRIC_MAPPING.keys())
    selected = fund_long[fund_long["metric"].isin(target_metrics)].copy()
    log.info(f"  Metric filter: {len(fund_long):,} → {len(selected):,} "
             f"(from {fund_long['metric'].nunique()} → {selected['metric'].nunique()} unique metrics)")

    # Some metrics can appear in several source_subpages → drop duplicates
    # (e.g. "Marża zysku netto" can be in both wskazniki_rynkowe and wskazniki_rentownosci)
    selected = selected.drop_duplicates(
        subset=["ticker", "period_end", "metric"], keep="first"
    )

    # Pivot
    wide = selected.pivot_table(
        index=["ticker", "period_end", "available_from"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()

    # Rename columns to the short names
    wide = wide.rename(columns=METRIC_MAPPING)

    # Force a fixed column order (predictability)
    metric_cols = list(METRIC_MAPPING.values())
    existing_metrics = [c for c in metric_cols if c in wide.columns]
    wide = wide[["ticker", "period_end", "available_from"] + existing_metrics]

    log.info(f"  Wide format: {len(wide):,} rows, {len(existing_metrics)} metrics")
    return wide


# ─── STEP 3: WINSORIZATION ─────────────────────────────────────────────────────

def winsorize_metrics(fund_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clips each metric to the [WINSOR_LOWER, WINSOR_UPPER] percentile range.
    Returns: (df_winsorized, stats_per_metric)
    """
    df = fund_wide.copy()
    metric_cols = [c for c in METRIC_MAPPING.values() if c in df.columns]

    stats_rows = []
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        n_before = vals.count()
        lo = vals.quantile(WINSOR_LOWER)
        hi = vals.quantile(WINSOR_UPPER)

        clipped_lo = (df[col] < lo).sum()
        clipped_hi = (df[col] > hi).sum()

        df[col] = df[col].clip(lower=lo, upper=hi)

        stats_rows.append({
            "metric": col,
            "n_values": int(n_before),
            "min_orig": float(vals.min()),
            "max_orig": float(vals.max()),
            "p1_clip_to": float(lo),
            "p99_clip_to": float(hi),
            "n_clipped_low": int(clipped_lo),
            "n_clipped_high": int(clipped_hi),
            "median": float(vals.median()),
        })

    stats_df = pd.DataFrame(stats_rows)
    log.info(f"  Winsorized {len(metric_cols)} metrics to the [{WINSOR_LOWER:.0%}, {WINSOR_UPPER:.0%}] percentile range")
    return df, stats_df


# ─── STEP 4: AS-OF MERGE PER METRIC ────────────────────────────────────────────

def asof_merge(ohlcv: pd.DataFrame, fund_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Per-metric forward fill: for every OHLCV day and every metric, takes the
    latest non-NaN value of that metric where available_from <= date.

    Why per metric? Different metrics are published at different frequencies:
      - price, ROE, margins: quarterly
      - total_assets, equity, net_income: annually only
    A plain merge_asof would pick the latest "record", which for annual metrics
    would give NaN in 3 out of 4 quarters. Here the forward fill runs per column.

    Implementation:
      1. The old merge_asof strategy provides period_end (latest quarter, for diagnostics).
      2. For each metric we build a separate "long" view ticker × available_from × value
         (only where value != NaN), then merge_asof it with OHLCV.
    """
    metric_cols = [c for c in METRIC_MAPPING.values() if c in fund_wide.columns]

    # Sort OHLCV
    ohlcv_s = ohlcv.sort_values(["date"]).reset_index(drop=True)

    # First: attach period_end + available_from of the "latest quarter" - for diagnostics
    # (which quarters, how densely fundamentals cover OHLCV)
    diag_fund = fund_wide[["ticker", "period_end", "available_from"]].sort_values("available_from")
    merged = pd.merge_asof(
        ohlcv_s,
        diag_fund,
        left_on="date",
        right_on="available_from",
        by="ticker",
        direction="backward",
    )

    # Per metric: run a separate merge_asof on the NON-NULL values of each column
    log.info(f"  Per-metric forward fill ({len(metric_cols)} metrics)...")
    for col in metric_cols:
        # Take only the rows where this metric is available
        sub = (fund_wide[["ticker", "available_from", col]]
               .dropna(subset=[col])
               .sort_values("available_from")
               .reset_index(drop=True))

        # merge_asof on the single column (as _value)
        sub = sub.rename(columns={col: f"{col}_val", "available_from": f"{col}_avail"})

        merged = pd.merge_asof(
            merged.sort_values("date"),
            sub.sort_values(f"{col}_avail"),
            left_on="date",
            right_on=f"{col}_avail",
            by="ticker",
            direction="backward",
        )
        # Replace the original column and drop the helper ones
        merged[col] = merged[f"{col}_val"]
        merged = merged.drop(columns=[f"{col}_val", f"{col}_avail"])

    # Diagnostics: how many rows have any fundamental at all
    has_any_fund = merged[metric_cols].notna().any(axis=1)
    log.info(f"  Per-metric merge: {has_any_fund.sum():,}/{len(merged):,} rows with ≥1 metric "
             f"({100*has_any_fund.sum()/len(merged):.1f}%)")

    return merged


# ─── STEP 5: ADD MARKET CAP AND SECTOR ─────────────────────────────────────────

def add_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market cap = current close (today) × shares (from the latest report).
    Uses the current price, not the price on the report date.
    """
    if "shares" not in df.columns:
        log.warning("  No 'shares' column - skipping market_cap")
        return df

    df = df.copy()
    df["market_cap"] = df["close"] * df["shares"]
    log.info(f"  Market cap computed for {df['market_cap'].notna().sum():,} rows")
    return df


def add_branza(df: pd.DataFrame, branza_df: pd.DataFrame) -> pd.DataFrame:
    """Attaches the branza (sector) column per ticker."""
    df = df.merge(branza_df, on="ticker", how="left")
    log.info(f"  Sector attached, {df['branza'].notna().sum():,} rows")
    return df


# ─── STEP 6: REORDER + DIAGNOSTICS ─────────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    # Standard column order for readability
    base_cols = ["ticker", "yf_symbol", "date", "branza",
                 "open", "high", "low", "close", "volume",
                 "turnover_pln", "turnover_30d_avg", "zero_vol_60d", "is_liquid"]
    fund_cols = ["period_end", "available_from"] + list(METRIC_MAPPING.values())
    derived_cols = ["market_cap"]

    all_cols = []
    for c in base_cols + fund_cols + derived_cols:
        if c in df.columns and c not in all_cols:
            all_cols.append(c)

    # Append any remaining columns
    for c in df.columns:
        if c not in all_cols:
            all_cols.append(c)

    return df[all_cols]


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage statistics per metric."""
    metric_cols = [c for c in METRIC_MAPPING.values() if c in df.columns] + ["market_cap", "branza"]
    rows = []
    total = len(df)
    for c in metric_cols:
        n_notna = df[c].notna().sum()
        pct = 100 * n_notna / total if total > 0 else 0
        rows.append({
            "metric": c,
            "n_values": int(n_notna),
            "n_total": total,
            "pct_coverage": round(pct, 1),
        })
    return pd.DataFrame(rows).sort_values("pct_coverage", ascending=False)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("STEP 2A: MERGE OHLCV + FUNDAMENTALS")
    log.info("=" * 70)

    # 1. Load
    ohlcv, fund_long, branza_df = load_data()

    # 2. Filter pre-IPO and pivot
    log.info("\nFiltering pre-IPO and pivoting long → wide...")
    fund_long_clean = filter_preipo(fund_long)
    fund_wide = pivot_fundamentals(fund_long_clean)

    # 3. Winsorize
    log.info("\nWinsorization...")
    fund_wide_winsor, winsor_stats = winsorize_metrics(fund_wide)

    # 4. As-of merge
    log.info("\nAs-of merge OHLCV + fundamentals...")
    merged = asof_merge(ohlcv, fund_wide_winsor)

    # 5. Derived columns
    log.info("\nComputing derived columns...")
    merged = add_market_cap(merged)
    merged = add_branza(merged, branza_df)

    # 6. Finalize
    final = finalize(merged)

    # 7. Save
    final.to_parquet(OUTPUT_PARQUET, index=False)
    log.info(f"\n✓ Saved: {OUTPUT_PARQUET}")

    # 8. Diagnostics
    cov = coverage_report(final)
    cov.to_csv(DIAG_REPORT, index=False)

    # 9. Summary print
    log.info("=" * 70)
    log.info("REPORT")
    log.info("=" * 70)
    log.info(f"  Total rows:           {len(final):,}")
    log.info(f"  Companies:            {final['ticker'].nunique()}")
    log.info(f"  Date range:           {final['date'].min().date()} → {final['date'].max().date()}")
    log.info(f"  Rows with period_end: {final['period_end'].notna().sum():,} "
             f"({100*final['period_end'].notna().mean():.1f}%)")
    log.info(f"  Liquid rows:          {final['is_liquid'].sum():,} "
             f"({100*final['is_liquid'].mean():.1f}%)")
    log.info(f"")
    log.info(f"  Sectors ({final['branza'].nunique()}):")
    top_br = final.drop_duplicates("ticker")["branza"].value_counts().head(10)
    for b, n in top_br.items():
        log.info(f"    {b:30s} {n}")
    log.info(f"")
    log.info(f"  Coverage per metric (top 10 + bottom 5):")
    for _, row in cov.head(10).iterrows():
        log.info(f"    {row['metric']:20s} {row['pct_coverage']:5.1f}%  "
                 f"({row['n_values']:,}/{row['n_total']:,})")
    log.info(f"    ...")
    for _, row in cov.tail(5).iterrows():
        log.info(f"    {row['metric']:20s} {row['pct_coverage']:5.1f}%  "
                 f"({row['n_values']:,}/{row['n_total']:,})")
    log.info(f"")
    log.info(f"  Output:               {OUTPUT_PARQUET}")
    log.info(f"  Coverage report:      {DIAG_REPORT}")
    log.info("=" * 70)

    # 10. Sample sanity check
    log.info("\nSanity check: KGHM around 2024-09-15 (nearest trading day):")
    kghm = final[final["ticker"] == "KGHM"].copy()
    if len(kghm) > 0:
        target = pd.Timestamp("2024-09-15")
        kghm["dist"] = (kghm["date"] - target).abs()
        s = kghm.sort_values("dist").iloc[0]
        log.info(f"  date={s['date'].date()}  close={s['close']:.2f}  branza={s['branza']}")
        log.info(f"  latest quarter: period_end={s['period_end'].date() if pd.notna(s['period_end']) else None}, "
                 f"available_from={s['available_from'].date() if pd.notna(s['available_from']) else None}")
        for col in ["pe", "pbv", "roe", "margin_net", "debt_to_equity",
                    "total_assets", "equity", "net_income", "shares", "market_cap"]:
            v = s.get(col, None)
            if pd.notna(v):
                if abs(v) > 1000:
                    log.info(f"  {col:18s} = {v:,.0f}")
                else:
                    log.info(f"  {col:18s} = {v:.4f}")
            else:
                log.info(f"  {col:18s} = NaN")


if __name__ == "__main__":
    main()