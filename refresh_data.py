"""
Refreshes the data and generates predictions.

Pipeline:
  1. fetch_ohlcv        — new prices from yfinance (cache-aware, fetches only the delta)
  2. fetch_fundamentals — new ratios from biznesradar.pl (cache TTL 1-7d)
  3. merge_fundamentals — OHLCV + fundamentals → dataset_merged
  4. build_features     — computes all features → dataset_features
  5. prepare_ml_dataset — joins, filters, adds targets → dataset_ready_for_ml
  6. predict_today      — ML ensemble → top-5 picks

Usage:
    python refresh_data.py                  # full pipeline
    python refresh_data.py --skip-fetch     # rebuild only (merge..predict), no downloads
    python refresh_data.py --skip-ohlcv     # skip prices (fundamentals + rebuild only)
    python refresh_data.py --predict-only   # only predict on the current dataset
    python refresh_data.py --no-predict     # everything except the prediction
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
ML_DATA = ROOT / "ml_data"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def run_step(name: str, script: Path, extra_args: list = None) -> bool:
    """Runs a single pipeline step. Returns True if it succeeded."""
    args = extra_args or []
    print(f"\n{BOLD}[{_ts()}] ── {name} {'─'*(55 - len(name))}{RESET}")

    if not script.exists():
        print(f"  {RED}ERROR: script does not exist: {script}{RESET}")
        return False

    cmd = [sys.executable, str(script)] + args
    t0 = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=False,  # logs go to stdout live
    )

    elapsed = time.time() - t0
    ok = result.returncode == 0

    if ok:
        print(f"  {GREEN}✓ OK{RESET}  ({elapsed:.0f}s)")
    else:
        print(f"  {RED}✗ ERROR (exit={result.returncode}){RESET}  ({elapsed:.0f}s)")

    return ok


def check_file_age(path: Path) -> str:
    if not path.exists():
        return f"{RED}missing{RESET}"
    age_h = (time.time() - path.stat().st_mtime) / 3600
    if age_h < 2:
        return f"{GREEN}{age_h:.1f}h temu{RESET}"
    elif age_h < 24:
        return f"{YELLOW}{age_h:.1f}h temu{RESET}"
    else:
        return f"{RED}{age_h/24:.0f}d temu{RESET}"


def print_status():
    """Shows the current status of the key files."""
    files = [
        (ML_DATA / "dataset_raw.parquet",          "OHLCV (dataset_raw)"),
        (ML_DATA / "dataset_fundamentals_long.parquet", "Fundamentals"),
        (ML_DATA / "dataset_merged.parquet",        "Merged"),
        (ML_DATA / "dataset_features.parquet",      "Features"),
        (ML_DATA / "dataset_ready_for_ml.parquet",  "Dataset ML"),
        (ML_DATA / "final_model_iter1.txt",         "Model Iter1"),
        (ML_DATA / "final_model_iter3.txt",         "Model Iter3"),
        (ML_DATA / "predictions_today.csv",         "Predictions"),
    ]
    print(f"\n{BOLD}File status:{RESET}")
    for path, label in files:
        size = f"{path.stat().st_size / 1e6:.1f}MB" if path.exists() else "—"
        age  = check_file_age(path)
        print(f"  {label:<30}  {size:>8}  {age}")


def main():
    parser = argparse.ArgumentParser(description="Refresh GPW ML data")
    parser.add_argument("--skip-fetch",     action="store_true",
                        help="Skip fetching OHLCV and fundamentals")
    parser.add_argument("--skip-ohlcv",     action="store_true",
                        help="Skip only the OHLCV fetch")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Skip the fundamentals fetch")
    parser.add_argument("--predict-only",   action="store_true",
                        help="Prediction only, no data updates")
    parser.add_argument("--no-predict",     action="store_true",
                        help="Skip the prediction step")
    parser.add_argument("--status",         action="store_true",
                        help="Only show the status of the files and exit")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*65}")
    print(f"  GPW ML — REFRESH DATA   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*65}{RESET}")

    print_status()

    if args.status:
        return

    if not (ML_DATA / "final_model_iter1.txt").exists():
        print(f"\n{YELLOW}WARNING: final models are missing. "
              f"Run first: python train_final_model.py{RESET}")

    t_total = time.time()
    results = {}

    # ── STEP DEFINITIONS ──────────────────────────────────────────────────────

    steps = []

    if not args.predict_only:
        if not args.skip_fetch and not args.skip_ohlcv:
            steps.append((
                "1. Fetching OHLCV prices",
                ML_DATA / "fetch_ohlcv.py",
                [],
            ))

        if not args.skip_fetch and not args.skip_fundamentals:
            steps.append((
                "2. Fetching fundamentals (biznesradar)",
                ROOT / "fetch_fundamentals.py",
                [],
            ))

        steps += [
            ("3. Merge OHLCV + fundamentals",   ROOT / "merge_fundamentals.py",   []),
            ("4. Building features",          ROOT / "build_features.py",       []),
            ("5. Preparing the ML dataset",   ROOT / "prepare_ml_dataset.py",   []),
        ]

    if not args.no_predict:
        steps.append((
            "6. Generating predictions",
            ROOT / "predict_today.py",
            [],
        ))

    if not steps:
        print(f"\n{YELLOW}No steps to run.{RESET}")
        return

    # ── RUN ───────────────────────────────────────────────────────────────────

    print(f"\n{BOLD}Plan: {len(steps)} {'step' if len(steps)==1 else 'steps'}{RESET}")
    for name, _, _ in steps:
        print(f"  • {name}")

    failed_at = None
    for name, script, extra in steps:
        ok = run_step(name, script, extra)
        results[name] = ok
        if not ok:
            failed_at = name
            break

    # ── SUMMARY ───────────────────────────────────────────────────────────────

    elapsed_total = time.time() - t_total
    print(f"\n{BOLD}{'='*65}")
    print(f"  SUMMARY   ({elapsed_total:.0f}s total)")
    print(f"{'='*65}{RESET}")

    for name, ok in results.items():
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon}  {name}")

    skipped = len(steps) - len(results)
    if skipped > 0:
        for name, _, _ in steps[len(results):]:
            print(f"  {YELLOW}–{RESET}  {name}  (skipped after an error)")

    if failed_at is None:
        print(f"\n{GREEN}{BOLD}All OK.{RESET}")
        print(f"Predykcje: {ML_DATA / 'predictions_today.txt'}")
    else:
        print(f"\n{RED}{BOLD}Pipeline stopped at: {failed_at}{RESET}")
        print(f"Check the logs above and run it again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
