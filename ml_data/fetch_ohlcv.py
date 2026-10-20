"""
Step 1A: OHLCV downloader with a cache and a dynamic liquidity filter.

Features:
- 5 years of history (configurable)
- Parquet cache per company (a second run only downloads the missing days)
- Dynamic liquidity filter: 30-day average turnover ≥ MIN_AVG_TURNOVER_PLN
- Sanity checks: splits, price jumps, days with zero turnover
- Detailed final report: days per company, problems, exclusions

Usage:
    python fetch_ohlcv.py                  # normal run, fetches the missing data
    python fetch_ohlcv.py --force-refresh  # ignores the cache, downloads everything again
    python fetch_ohlcv.py --report-only    # no downloads, report from the cache only
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from tickers import TICKER_MAP

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS = 365 * 5                 # 5 years
MIN_HISTORY_DAYS = 60                   # minimum for a company to be included at all
MIN_AVG_TURNOVER_PLN = 100_000          # liquidity threshold (30-day average turnover in PLN)
TURNOVER_WINDOW = 30                    # window for the average turnover
MAX_ZERO_VOLUME_DAYS_60D = 5            # max days with zero turnover in the last 60 days

OUT_DIR = Path("ml_data")
CACHE_DIR = OUT_DIR / "ohlcv_cache"     # per-company cache lives here
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

RAW_PARQUET = OUT_DIR / "dataset_raw.parquet"
REPORT_CSV = OUT_DIR / "fetch_report.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_ohlcv")


# ─── DOWNLOAD PER COMPANY ──────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def _load_cache(name: str) -> Optional[pd.DataFrame]:
    p = _cache_path(name)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as e:
        log.warning(f"Cache for {name} is corrupted, ignoring it: {e}")
        return None


def _save_cache(name: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(name), index=False)


def _fetch_single(name: str, yf_symbol: str, start: datetime, end: datetime,
                  force_refresh: bool = False) -> tuple[Optional[pd.DataFrame], str]:
    """
    Returns (DataFrame with OHLCV, status_message).
    Status: "ok", "ok_cached", "ok_incremental", "skip_too_short", "error: ..."
    """
    cached = None if force_refresh else _load_cache(name)

    # If we have a cache and the last date is fresh (< 3 days old), leave it alone
    if cached is not None and not cached.empty:
        last_cached = pd.to_datetime(cached["date"].max())
        # Download incrementally from the day after the last cached one
        fetch_start = last_cached + timedelta(days=1)
        if fetch_start.date() >= end.date():
            return cached, "ok_cached"
    else:
        fetch_start = start

    try:
        df_new = yf.Ticker(yf_symbol).history(
            start=fetch_start, end=end, auto_adjust=True
        )
    except Exception as e:
        # If we have a cache, it's better than nothing
        if cached is not None:
            return cached, f"warn_using_cache_only: {e}"
        return None, f"error: {e}"

    if df_new.empty and cached is None:
        return None, "skip_no_data"

    if not df_new.empty:
        df_new = df_new.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df_new["date"] = pd.to_datetime(df_new["date"]).dt.tz_localize(None)
        df_new = df_new[["date", "open", "high", "low", "close", "volume"]]

    # Combine cache + new data
    if cached is not None and not df_new.empty:
        df = pd.concat([cached, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        status = "ok_incremental"
    elif cached is not None:
        df = cached
        status = "ok_cached"
    else:
        df = df_new
        status = "ok"

    # Hard filter on the minimum history length
    if len(df) < MIN_HISTORY_DAYS:
        return None, f"skip_too_short ({len(df)} days)"

    # Save the cache (always, even for incremental runs)
    df_to_cache = df.copy()
    _save_cache(name, df_to_cache)

    return df, status


# ─── SANITY CHECKS ─────────────────────────────────────────────────────────────

def _detect_anomalies(df: pd.DataFrame, name: str) -> list[str]:
    """Detects potential data problems."""
    issues = []

    # Price jumps > 30% in one day (possibly a badly handled split)
    daily_ret = df["close"].pct_change().abs()
    big_jumps = (daily_ret > 0.30).sum()
    if big_jumps > 0:
        issues.append(f"big_jumps={big_jumps}")

    # Days with zero turnover
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        issues.append(f"zero_volume_days={zero_vol}")

    # Very short history (less than 252 days = ~1 year)
    if len(df) < 252:
        issues.append(f"short_history={len(df)}d")

    return issues


# ─── LIQUIDITY FILTER (per row, dynamic) ───────────────────────────────────────

def _add_liquidity_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns:
      - turnover_pln: daily turnover in PLN (close * volume)
      - turnover_30d_avg: 30-day average turnover
      - is_liquid: whether it meets the liquidity threshold
    """
    df = df.copy()
    df["turnover_pln"] = df["close"] * df["volume"]
    df["turnover_30d_avg"] = (
        df.groupby("ticker")["turnover_pln"]
          .transform(lambda x: x.rolling(TURNOVER_WINDOW, min_periods=10).mean())
    )

    # Number of zero-volume days in the last 60 days (per ticker)
    df["zero_vol_60d"] = (
        df.groupby("ticker")["volume"]
          .transform(lambda x: (x == 0).rolling(60, min_periods=20).sum())
    )

    df["is_liquid"] = (
        (df["turnover_30d_avg"] >= MIN_AVG_TURNOVER_PLN) &
        (df["zero_vol_60d"] <= MAX_ZERO_VOLUME_DAYS_60D)
    )
    return df


# ─── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def fetch_all(force_refresh: bool = False, report_only: bool = False) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS + 30)

    yf_tickers = {n: s for n, s in TICKER_MAP.items() if s is not None}
    log.info(
        f"Universe: {len(yf_tickers)} companies | "
        f"Okres: {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} "
        f"({LOOKBACK_DAYS} days)"
    )

    if report_only:
        log.info("Report-only mode: loading the existing cache without downloading")

    frames = []
    report_rows = []

    for name, yf_symbol in tqdm(yf_tickers.items(), desc="OHLCV"):
        if report_only:
            df = _load_cache(name)
            status = "ok_cached" if df is not None else "skip_no_cache"
        else:
            df, status = _fetch_single(name, yf_symbol, start_date, end_date,
                                       force_refresh=force_refresh)

        report_rows.append({
            "ticker": name,
            "yf_symbol": yf_symbol,
            "status": status,
            "rows": len(df) if df is not None else 0,
            "first_date": df["date"].min() if df is not None and not df.empty else None,
            "last_date": df["date"].max() if df is not None and not df.empty else None,
            "anomalies": ";".join(_detect_anomalies(df, name)) if df is not None else "",
        })

        if df is None or df.empty:
            continue

        df = df.copy()
        df["ticker"] = name
        df["yf_symbol"] = yf_symbol
        frames.append(df[["ticker", "yf_symbol", "date", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise RuntimeError("No data was downloaded!")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Add liquidity flags (on the whole dataset at once)
    result = _add_liquidity_flag(result)

    # Save
    result.to_parquet(RAW_PARQUET, index=False)

    # Report
    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT_CSV, index=False)

    _print_summary(result, report)
    return result


def _print_summary(df: pd.DataFrame, report: pd.DataFrame) -> None:
    n_total = len(report)
    n_ok = (report["status"].str.startswith("ok")).sum()
    n_skip = (report["status"].str.startswith("skip")).sum()
    n_err = (report["status"].str.startswith("error")).sum()

    # Liquidity: how many companies are liquid "today" (the latest date)
    last_date = df["date"].max()
    today_snapshot = df[df["date"] == last_date]
    n_liquid_today = today_snapshot["is_liquid"].sum()

    # Anomalies
    with_anomalies = report[report["anomalies"] != ""]

    log.info("=" * 70)
    log.info(f"OHLCV DOWNLOAD REPORT")
    log.info("=" * 70)
    log.info(f"  Input universe:        {n_total} companies")
    log.info(f"  Downloaded OK:         {n_ok}")
    log.info(f"  Skipped:               {n_skip}")
    log.info(f"  Errors:                {n_err}")
    log.info(f"  Total rows:            {len(df):,}")
    log.info(f"  Date range:            {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")
    log.info(f"")
    log.info(f"  Liquid on {last_date:%Y-%m-%d}:  {n_liquid_today}/{today_snapshot['ticker'].nunique()} companies")
    log.info(f"  (threshold: 30-day average turnover ≥ {MIN_AVG_TURNOVER_PLN:,} PLN)")
    log.info(f"")

    if len(with_anomalies) > 0:
        log.info(f"  Companies with anomalies: {len(with_anomalies)}")
        for _, row in with_anomalies.head(15).iterrows():
            log.info(f"    {row['ticker']:15s} {row['anomalies']}")
        if len(with_anomalies) > 15:
            log.info(f"    ... and {len(with_anomalies) - 15} more (see {REPORT_CSV})")

    if n_skip > 0 or n_err > 0:
        log.info(f"")
        log.info(f"  Problematic companies:")
        problematic = report[~report["status"].str.startswith("ok")]
        for _, row in problematic.iterrows():
            log.info(f"    {row['ticker']:15s} {row['status']}")

    log.info("=" * 70)
    log.info(f"Full report: {REPORT_CSV}")
    log.info(f"Dataset:      {RAW_PARQUET}")


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore cache, download everything again")
    parser.add_argument("--report-only", action="store_true",
                        help="Report from the cache only, no downloads")
    args = parser.parse_args()

    fetch_all(force_refresh=args.force_refresh, report_only=args.report_only)


if __name__ == "__main__":
    main()