"""
Step 1B-2: fundamentals scraper for biznesradar.pl

Features:
- Downloads 6 subpages per company: market value ratios, profitability, debt,
  liquidity, income statement, balance sheet
- Cache per company per subpage with a TTL (1d for market ratios, 7d for the rest)
- Rate limiting (1 req/sec) + retry with backoff
- Parser for "report-table" tables with Polish number formatting
- Point-in-time: adds available_from = period_end + margin (B1 hybrid)
  - Q1, Q3: +75 days
  - Q2 (half-year): +95 days
  - Q4 (annual): +125 days
- Output: dataset_fundamentals_long.parquet (long format)
- Test mode: --test (only 4 companies: KGHM, PKO, CDR, DOM)

Usage:
    python fetch_fundamentals.py --test         # test on 4 companies
    python fetch_fundamentals.py                # all 140
    python fetch_fundamentals.py --force        # ignore cache
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
CACHE_DIR = OUT_DIR / "br_cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OUTPUT_PARQUET = OUT_DIR / "dataset_fundamentals_long.parquet"
REPORT_CSV = OUT_DIR / "fundamentals_fetch_report.csv"
MAPPING_FILE = Path("biznesradar_mapping.json")

# Subpages to scrape
SUBPAGES = {
    "wskazniki_rynkowe": "wskazniki-wartosci-rynkowej",
    "wskazniki_rentownosci": "wskazniki-rentownosci",
    "wskazniki_zadluzenia": "wskazniki-zadluzenia",
    "wskazniki_plynnosci": "wskazniki-plynnosci",
    "rachunek_zyskow": "raporty-finansowe-rachunek-zyskow-i-strat",
    "bilans": "raporty-finansowe-bilans",
}

# Cache TTL (days)
CACHE_TTL_DAYS = {
    "wskazniki_rynkowe": 1,    # P/E depends on price - refresh daily
    "default": 7,              # reports come out quarterly, so weekly is enough
}

# Point-in-time margins (B1 hybrid)
PIT_MARGINS = {
    1: 75,   # Q1 (period_end = March 31)  → ~June 14
    2: 95,   # Q2 (half-year, June 30)     → ~October 3
    3: 75,   # Q3 (September 30)           → ~December 14
    4: 125,  # Q4 (annual, December 31)  → ~May 5
}

BASE_URL = "https://www.biznesradar.pl/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

RATE_LIMIT_SEC = 1.0
RETRY_COUNT = 3
RETRY_BACKOFF_SEC = 5.0

TEST_TICKERS = ["KGHM", "PKOBP", "CDPROJEKT", "DOMDEV"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("br_scraper")


# ─── CACHE ─────────────────────────────────────────────────────────────────────

def _cache_path(slug: str, subpage_type: str) -> Path:
    company_dir = CACHE_DIR / slug
    company_dir.mkdir(exist_ok=True)
    return company_dir / f"{subpage_type}.html"


def _meta_path(slug: str, subpage_type: str) -> Path:
    return CACHE_DIR / slug / f"{subpage_type}.meta.json"


def _is_cache_fresh(slug: str, subpage_type: str) -> bool:
    """Checks whether the cache exists and is fresh (according to TTL)."""
    cache_p = _cache_path(slug, subpage_type)
    meta_p = _meta_path(slug, subpage_type)
    if not cache_p.exists() or not meta_p.exists():
        return False

    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
    except Exception:
        return False

    ttl_days = CACHE_TTL_DAYS.get(subpage_type, CACHE_TTL_DAYS["default"])
    age = datetime.now() - fetched_at
    return age < timedelta(days=ttl_days)


def _save_cache(slug: str, subpage_type: str, html: str) -> None:
    _cache_path(slug, subpage_type).write_text(html, encoding="utf-8")
    _meta_path(slug, subpage_type).write_text(
        json.dumps({"fetched_at": datetime.now().isoformat()}),
        encoding="utf-8",
    )


def _load_cache(slug: str, subpage_type: str) -> Optional[str]:
    p = _cache_path(slug, subpage_type)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


# ─── FETCHING ──────────────────────────────────────────────────────────────────

def fetch_subpage(slug: str, subpage_type: str, force: bool = False) -> Optional[str]:
    """
    Fetches the HTML of a given subpage for a company, using the cache if fresh.
    Returns the HTML as a string, or None on error.
    """
    if not force and _is_cache_fresh(slug, subpage_type):
        return _load_cache(slug, subpage_type)

    url_part = SUBPAGES[subpage_type]
    url = f"{BASE_URL}{url_part}/{slug}"

    last_err = None
    for attempt in range(RETRY_COUNT):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                html = r.text
                _save_cache(slug, subpage_type, html)
                return html
            elif r.status_code == 429:
                # rate limited - wait longer
                wait = RETRY_BACKOFF_SEC * (attempt + 1) * 2
                log.warning(f"  {slug}/{subpage_type}: 429, waiting {wait}s")
                time.sleep(wait)
            elif r.status_code in (502, 503, 504):
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            else:
                last_err = f"HTTP {r.status_code}"
                break
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))

    log.warning(f"  {slug}/{subpage_type}: failed - {last_err}")
    # Fallback: use the cache even if stale
    return _load_cache(slug, subpage_type)


# ─── PARSING ───────────────────────────────────────────────────────────────────

# Polish month abbreviations to numbers (biznesradar uses e.g. "mar 24", "cze 24")
PL_MONTHS = {
    "sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6,
    "lip": 7, "sie": 8, "wrz": 9, "paź": 10, "paz": 10, "lis": 11, "gru": 12,
}

# Last day of each "publication quarter" based on the period's closing month
QUARTER_END_DAY = {3: 31, 6: 30, 9: 30, 12: 31}


def parse_period_header(text: str) -> Optional[date]:
    """
    Parses a biznesradar column header.

    Examples:
      '2024/Q1(mar 24)'    → date(2024, 3, 31)
      '2024(gru 24)'       → date(2024, 12, 31)  (annual)
      '2024/Q4(gru 24)'    → date(2024, 12, 31)
    """
    if not text:
        return None
    text = text.strip()

    # Match the pattern: year and optional quarter + month in parentheses
    # Format: "2024/Q1(mar 24)" or "2024(gru 24)"
    m = re.match(r"(\d{4})(?:/Q(\d))?\(([a-ząćęłńóśźż]{3})\s*(\d{2})\)", text, re.IGNORECASE)
    if not m:
        return None

    year = int(m.group(1))
    month_abbr = m.group(3).lower()
    if month_abbr not in PL_MONTHS:
        return None

    month = PL_MONTHS[month_abbr]
    day = QUARTER_END_DAY.get(month, 31)

    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_value(text: str) -> Optional[float]:
    """
    Parses a Polish-formatted number from biznesradar.

    Examples:
      '31,29'           → 31.29
      '24,42%'          → 0.2442    (percent as a fraction)
      '63 620 002 000'  → 63620002000.0
      '-1,52%'          → -0.0152
      ''                → None
      'r/r'             → None
      '~branża'         → None
    """
    if not text:
        return None
    text = text.strip()

    # Skip non-numeric markers
    if text in ("", "-", "—", "N/A", "n/a", "r/r", "k/k", "~branża"):
        return None
    if text.startswith("~") or text.startswith("r/r") or text.startswith("k/k"):
        return None

    # Check whether it's a percentage
    is_percent = "%" in text

    # Clean up: remove spaces (thousands separator), &nbsp;, the % sign, +
    cleaned = (text
               .replace("\xa0", "")    # non-breaking space
               .replace(" ", "")
               .replace("%", "")
               .replace("+", "")
               .replace(",", "."))

    # Check whether it's a number (after cleaning)
    if not re.match(r"^-?\d+\.?\d*$", cleaned):
        return None

    try:
        val = float(cleaned)
        if is_percent:
            val = val / 100.0
        return val
    except ValueError:
        return None


def extract_first_value_from_cell(cell) -> Optional[float]:
    """
    Extracts the first meaningful numeric value from a <td> cell.

    Cells in biznesradar's report-table can contain:
      - just the value: '31,29'
      - value + sector average: '31,29 ~branża 28,40'
      - value + sector + y/y + q/q: a more nested structure
    We always take the FIRST number = the main value.
    """
    # First try to parse the whole cell text (if it's simple)
    full_text = cell.get_text(" ", strip=True)
    if not full_text:
        return None

    # Find ALL numbers and take the first NON-percentage one, or the first one if all are %
    # Pattern: optional minus, digits (with spaces), optional decimal part with a comma, optional %
    # Example: '31,29', '63 620 002 000', '-1,52%', '+24,42%'
    matches = re.findall(r"[+\-]?\d[\d\s\xa0]*(?:,\d+)?%?", full_text)
    for raw in matches:
        # Skip if it's part of "~branża 28,40" - we want the REAL first value
        val = parse_value(raw)
        if val is not None:
            return val
    return None


def parse_report_table(html: str) -> dict[date, dict[str, float]]:
    """
    Parses a biznesradar 'report-table'.

    Returns a dict: {period_end: {metric_name: value}}
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the report-table
    tables = soup.find_all("table", class_="report-table")
    if not tables:
        return {}

    result = {}

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # First row = headers (period_ends)
        header_cells = rows[0].find_all(["th", "td"])
        period_dates = []
        for cell in header_cells:
            txt = cell.get_text(strip=True)
            period_dates.append(parse_period_header(txt))

        # Remaining rows = metrics
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) < 2:
                continue

            metric_name = cells[0].get_text(strip=True)
            if not metric_name or len(metric_name) < 2:
                continue

            # Parse every cell (except the first one, which holds the metric name)
            for i, cell in enumerate(cells[1:], start=1):
                if i >= len(period_dates):
                    break
                period = period_dates[i]
                if period is None:
                    continue

                value = extract_first_value_from_cell(cell)
                if value is None:
                    continue

                if period not in result:
                    result[period] = {}
                # If the same metric already exists for this date (from another table),
                # don't overwrite it - keep the first one found
                if metric_name not in result[period]:
                    result[period][metric_name] = value

    return result


def parse_branza(html: str) -> Optional[str]:
    """Extracts the company's sector from the profileSummary table."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table", class_="profileSummary"):
        text = table.get_text(" | ", strip=True)
        m = re.search(r"Branża:\s*\|?\s*([^\|]+?)(?:\s*\||$)", text)
        if m:
            return m.group(1).strip()
    return None


# ─── POINT-IN-TIME ─────────────────────────────────────────────────────────────

def add_available_from(period_end: date) -> date:
    """
    Adds the margin to period_end to get the report publication date (PIT).
    """
    margin_days = PIT_MARGINS.get(period_end.month // 3 if period_end.month % 3 == 0 else 1, 75)
    # To be safe: map to the correct quarter
    quarter_map = {3: 1, 6: 2, 9: 3, 12: 4}
    q = quarter_map.get(period_end.month, 1)
    margin_days = PIT_MARGINS[q]
    return period_end + timedelta(days=margin_days)


# ─── CONSOLIDATION ─────────────────────────────────────────────────────────────

def fetch_company(platform_name: str, slug: str, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Fetches all subpages for one company, parses them and returns a long-format DF.

    Returns: (DataFrame, status_dict)
    """
    rows = []
    status = {
        "platform_name": platform_name,
        "slug": slug,
        "subpages_ok": 0,
        "subpages_failed": 0,
        "n_periods": 0,
        "n_metrics": 0,
        "branza": None,
        "errors": [],
    }

    for subpage_type in SUBPAGES:
        # Rate limit (cache misses only)
        cache_was_fresh = _is_cache_fresh(slug, subpage_type) and not force

        html = fetch_subpage(slug, subpage_type, force=force)
        if html is None:
            status["subpages_failed"] += 1
            status["errors"].append(f"{subpage_type}: no html")
            continue
        status["subpages_ok"] += 1

        # Try to parse branza (one of subpages will have it)
        if status["branza"] is None:
            status["branza"] = parse_branza(html)

        # Parse report tables
        try:
            parsed = parse_report_table(html)
        except Exception as e:
            status["errors"].append(f"{subpage_type}: parse error {e}")
            continue

        # Convert to long format rows
        for period_end, metrics in parsed.items():
            available_from = add_available_from(period_end)
            for metric_name, value in metrics.items():
                rows.append({
                    "ticker": platform_name,
                    "period_end": period_end,
                    "available_from": available_from,
                    "source_subpage": subpage_type,
                    "metric": metric_name,
                    "value": value,
                })

        # Rate limit only if we actually fetched (not cache)
        if not cache_was_fresh:
            time.sleep(RATE_LIMIT_SEC)

    df = pd.DataFrame(rows)
    if not df.empty:
        status["n_periods"] = df["period_end"].nunique()
        status["n_metrics"] = df["metric"].nunique()

    return df, status


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Only 4 test companies: KGHM, PKO, CDR, DOM")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache, download everything again")
    args = parser.parse_args()

    # Load the mapping
    if not MAPPING_FILE.exists():
        log.error(f"Missing {MAPPING_FILE}. Run resolve_biznesradar.py first")
        return
    mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
    log.info(f"Loaded mapping: {len(mapping)} companies")

    if args.test:
        mapping = {k: v for k, v in mapping.items() if k in TEST_TICKERS}
        log.info(f"TEST MODE: only {list(mapping.keys())}")

    log.info(f"Fetching {len(SUBPAGES)} subpages × {len(mapping)} companies = "
             f"{len(SUBPAGES) * len(mapping)} requests")
    log.info(f"Cache: {'DISABLED (--force)' if args.force else 'ENABLED'}")

    all_dfs = []
    statuses = []

    for platform_name, slug in tqdm(mapping.items(), desc="Companies"):
        df, status = fetch_company(platform_name, slug, force=args.force)
        if not df.empty:
            all_dfs.append(df)
        statuses.append(status)

    if not all_dfs:
        log.error("No data was downloaded!")
        return

    full = pd.concat(all_dfs, ignore_index=True)

    # Convert dates to proper datetime
    full["period_end"] = pd.to_datetime(full["period_end"])
    full["available_from"] = pd.to_datetime(full["available_from"])

    # Save
    full.to_parquet(OUTPUT_PARQUET, index=False)
    log.info(f"✓ Saved: {OUTPUT_PARQUET}")

    # Status report
    status_df = pd.DataFrame([{
        "ticker": s["platform_name"],
        "slug": s["slug"],
        "subpages_ok": s["subpages_ok"],
        "subpages_failed": s["subpages_failed"],
        "n_periods": s["n_periods"],
        "n_metrics": s["n_metrics"],
        "branza": s["branza"],
        "errors": "; ".join(s["errors"][:3]) if s["errors"] else "",
    } for s in statuses])
    status_df.to_csv(REPORT_CSV, index=False)

    # Summary
    log.info("=" * 70)
    log.info("FUNDAMENTALS DOWNLOAD REPORT")
    log.info("=" * 70)
    log.info(f"  Companies processed:  {len(statuses)}")
    log.info(f"  Companies with data:  {sum(1 for s in statuses if s['n_periods'] > 0)}")
    log.info(f"  Total rows:           {len(full):,}")
    log.info(f"  Company × quarters:   {full[['ticker', 'period_end']].drop_duplicates().shape[0]:,}")
    log.info(f"  Unique metrics:       {full['metric'].nunique()}")
    log.info(f"  Date range:           {full['period_end'].min().date()} → {full['period_end'].max().date()}")
    log.info(f"")

    # Sector distribution
    branze = status_df[status_df["branza"].notna()]["branza"].value_counts().head(10)
    if len(branze) > 0:
        log.info(f"  Top sectors:")
        for b, n in branze.items():
            log.info(f"    {b:30s} {n}")

    # Companies with problems
    problematic = status_df[(status_df["subpages_failed"] > 0) | (status_df["n_periods"] == 0)]
    if len(problematic) > 0:
        log.info(f"")
        log.info(f"  Companies with problems: {len(problematic)}")
        for _, row in problematic.head(10).iterrows():
            log.info(f"    {row['ticker']:15s} OK={row['subpages_ok']}/6 "
                     f"periods={row['n_periods']} metrics={row['n_metrics']}")

    # Per-company summary
    if args.test:
        log.info(f"")
        log.info(f"  TEST mode details:")
        for _, row in status_df.iterrows():
            log.info(f"    {row['ticker']:12s}  {row['n_periods']:3d} quarters × "
                     f"{row['n_metrics']:3d} metrics  ({row['branza']})")

    log.info(f"")
    log.info(f"  Output:               {OUTPUT_PARQUET}")
    log.info(f"  Report:               {REPORT_CSV}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()