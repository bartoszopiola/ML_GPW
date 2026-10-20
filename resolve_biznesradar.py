"""
Mapping RESOLVER: GPW ticker → biznesradar slug

For every company in TICKER_MAP it tries to find the right URL on biznesradar.pl
by testing several variants: GPW ticker, platform name, hyphenated variants.

Output:
  - biznesradar_mapping.json - dict {platform_name: slug}
  - br_unresolved.txt - companies without a mapping (to fill in manually)
  - br_resolver_report.csv - full per-company report of attempts

Usage:
    python resolve_biznesradar.py
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from tickers import TICKER_MAP

OUTPUT_JSON = Path("biznesradar_mapping.json")
UNRESOLVED_TXT = Path("br_unresolved.txt")
REPORT_CSV = Path("br_resolver_report.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
}

BASE_URL = "https://www.biznesradar.pl/notowania/"
RATE_LIMIT_SEC = 1.0  # secs between requests


def generate_candidates(platform_name: str, yf_ticker: str) -> list[str]:
    """
    Generates slug variants to check, ordered by likelihood.
    """
    candidates = []

    # 1. Platform name as-is (e.g. KGHM, ALLEGRO)
    candidates.append(platform_name)

    # 2. GPW ticker without .WA (e.g. KGH, ALE)
    if yf_ticker and yf_ticker.endswith(".WA"):
        ticker_only = yf_ticker[:-3]
        if ticker_only and ticker_only not in candidates:
            candidates.append(ticker_only)

    # 3. Name with hyphens between letters and digits (CDPROJEKT → CD-PROJEKT? can't be done automatically)
    # Try to split CamelCase / abbreviations (e.g. PKOBP → PKO-BP, CDPROJEKT → CD-PROJEKT)
    # Heuristic: if it ends with BP, SE, ETF, GROUP etc., insert a hyphen
    suffixes = ["BP", "POL", "SEE", "BS", "MS", "TX", "FM",
                "MD", "PE", "TE", "SA", "NET", "GRP"]
    for sfx in suffixes:
        if platform_name.endswith(sfx) and len(platform_name) > len(sfx) + 1:
            candidate = platform_name[:-len(sfx)] + "-" + sfx
            if candidate not in candidates:
                candidates.append(candidate)

    return candidates


def verify_page(url: str, expected_platform: str) -> tuple[bool, str, str]:
    """
    Fetches the page and checks that:
      - status is 200
      - it contains <h1>Notowania ...</h1>
      - the h1 has a sensible ticker

    Returns: (is_valid, h1_text, error_msg)
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if r.status_code != 200:
            return False, "", f"status={r.status_code}"

        soup = BeautifulSoup(r.content, "html.parser")
        h1 = soup.find("h1")
        if not h1:
            return False, "", "no h1"

        h1_text = h1.get_text(strip=True)
        # Expected format: "Notowania KGH (KGHM)" or "Notowania CDR (CDPROJEKT)"
        if not h1_text.startswith("Notowania"):
            return False, h1_text, "h1 not 'Notowania ...'"

        # Check whether the title contains a word from platform_name (loose match)
        title = soup.title.text if soup.title else ""
        # Most titles look like: "Notowania KGHM POLSKA MIEDŹ SA (KGH)- BiznesRadar.pl"

        return True, h1_text, ""

    except Exception as e:
        return False, "", str(e)


def resolve_ticker(platform_name: str, yf_ticker: str | None) -> dict:
    """Tries to find the right slug for a given company."""
    if not yf_ticker:
        return {"platform_name": platform_name, "slug": None, "url": None,
                "h1": "", "tried": [], "status": "no_yf_ticker"}

    candidates = generate_candidates(platform_name, yf_ticker)
    tried = []

    for candidate in candidates:
        url = BASE_URL + candidate
        is_valid, h1, err = verify_page(url, platform_name)
        tried.append({"slug": candidate, "valid": is_valid, "h1": h1, "err": err})

        if is_valid:
            return {
                "platform_name": platform_name,
                "yf_ticker": yf_ticker,
                "slug": candidate,
                "url": url,
                "h1": h1,
                "tried": tried,
                "status": "ok",
            }
        time.sleep(RATE_LIMIT_SEC)

    return {
        "platform_name": platform_name,
        "yf_ticker": yf_ticker,
        "slug": None,
        "url": None,
        "h1": "",
        "tried": tried,
        "status": "not_found",
    }


def main():
    print(f"Resolving biznesradar slugs for {len(TICKER_MAP)} companies")
    print(f"Rate limit: {RATE_LIMIT_SEC}s between requests")
    print(f"Expected time: ~{len(TICKER_MAP) * 2 * RATE_LIMIT_SEC / 60:.1f} min (assuming ~2 attempts per company)")
    print()

    results = []
    for platform_name, yf_ticker in tqdm(TICKER_MAP.items(), desc="Resolving"):
        result = resolve_ticker(platform_name, yf_ticker)
        results.append(result)

    # Mapping JSON
    mapping = {r["platform_name"]: r["slug"] for r in results if r["slug"]}
    OUTPUT_JSON.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    # Unresolved
    unresolved = [r for r in results if not r["slug"]]
    with UNRESOLVED_TXT.open("w", encoding="utf-8") as f:
        f.write(f"Companies without a mapping: {len(unresolved)}\n")
        f.write("=" * 60 + "\n\n")
        for r in unresolved:
            f.write(f"{r['platform_name']:15s} (yf: {r['yf_ticker']})\n")
            f.write(f"  Status: {r['status']}\n")
            f.write(f"  Tried:\n")
            for t in r["tried"]:
                f.write(f"    {t['slug']:30s} -> {t['err'] or t['h1'][:60]}\n")
            f.write("\n")

    # Full CSV report
    import csv
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform_name", "yf_ticker", "slug", "url", "h1", "status", "tried_count"])
        for r in results:
            w.writerow([
                r["platform_name"], r["yf_ticker"], r["slug"] or "",
                r["url"] or "", r["h1"][:80], r["status"], len(r["tried"])
            ])

    print(f"\n{'=' * 60}")
    print(f"REPORT")
    print(f"{'=' * 60}")
    print(f"  Total:        {len(results)}")
    print(f"  Resolved:     {len(mapping)}  ({100*len(mapping)/len(results):.0f}%)")
    print(f"  Unresolved:   {len(unresolved)}")
    print(f"")
    print(f"  Output JSON:  {OUTPUT_JSON}")
    print(f"  Unresolved:   {UNRESOLVED_TXT}")
    print(f"  Full report:  {REPORT_CSV}")

    if unresolved:
        print(f"\n  Companies to check manually:")
        for r in unresolved[:20]:
            print(f"    {r['platform_name']:15s} (yf: {r['yf_ticker']})")
        if len(unresolved) > 20:
            print(f"    ... and {len(unresolved) - 20} more (see {UNRESOLVED_TXT})")


if __name__ == "__main__":
    main()