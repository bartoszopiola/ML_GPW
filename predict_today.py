"""
Phase 4 — today's prediction (portfolio picks).

Loads the final models and ranks all liquid companies
on the latest date in the dataset.

Ensemble: rank-percentile(Iter1) * 0.5 + rank-percentile(Iter3) * 0.5
(same as in eval_ensemble.py, which gave top-5 = +4.75%)

Output:
  - top-15 table in the terminal
  - ml_data/predictions_today.csv   (full ranking)
  - ml_data/predictions_today.txt   (text report)

Usage:
    python predict_today.py
    python predict_today.py --top 10
    python predict_today.py --date 2025-11-15   # a specific date
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

OUT_DIR      = Path("ml_data")
INPUT_PARQUET = OUT_DIR / "dataset_ready_for_ml.parquet"
MODEL_ITER1   = OUT_DIR / "final_model_iter1.txt"
MODEL_ITER3   = OUT_DIR / "final_model_iter3.txt"
META_JSON     = OUT_DIR / "final_model_meta.json"
OUTPUT_CSV    = OUT_DIR / "predictions_today.csv"
OUTPUT_TXT    = OUT_DIR / "predictions_today.txt"

TOP_N_PORTFOLIO = 5
SHOW_TOP_N = 15

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


def load_models():
    for p in [MODEL_ITER1, MODEL_ITER3, META_JSON]:
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            print("Run first: python train_final_model.py")
            sys.exit(1)
    m1 = lgb.Booster(model_file=str(MODEL_ITER1))
    m3 = lgb.Booster(model_file=str(MODEL_ITER3))
    with META_JSON.open(encoding="utf-8") as f:
        meta = json.load(f)
    return m1, m3, meta


def greenblatt_score(df: pd.DataFrame) -> pd.Series:
    pe_pos   = df["pe"].where(df["pe"] > 0, np.inf).fillna(np.inf)
    roe_fill = df["roe"].fillna(df["roe"].median())
    rank_pe  = pe_pos.rank(pct=True, ascending=False)
    rank_roe = roe_fill.rank(pct=True, ascending=True)
    return rank_pe + rank_roe


def fmt_val(v, fmt=".2f", na="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return f"{v:{fmt}}"


def fmt_pct(v, na="—"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return na
    return f"{v*100:+.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=SHOW_TOP_N)
    parser.add_argument("--date", type=str, default=None,
                        help="Prediction date YYYY-MM-DD (default: the latest date in the dataset)")
    args = parser.parse_args()

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    m1, m3, meta = load_models()
    features_i1 = meta["iter1"]["features"]
    features_i3 = meta["iter3"]["features"]
    trained_at  = meta.get("trained_at", "?")
    print(f"  Iter1: {meta['iter1']['best_iter']} drzew, "
          f"trained up to {meta['iter1']['train_end']}")
    print(f"  Iter3: {meta['iter3']['best_iter']} drzew, "
          f"trained up to {meta['iter3']['train_end']}")
    print(f"  Models trained:     {trained_at[:19]}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading {INPUT_PARQUET}...")
    df_full = pd.read_parquet(INPUT_PARQUET)
    df_full["date"] = pd.to_datetime(df_full["date"])

    if args.date:
        pred_date = pd.Timestamp(args.date)
        if pred_date not in df_full["date"].values:
            dates_avail = df_full[df_full["is_liquid"] == True]["date"].sort_values()
            nearest = dates_avail[dates_avail <= pred_date].max()
            print(f"  Data {args.date} niedostepna, uzywam: {nearest.date()}")
            pred_date = nearest
    else:
        pred_date = df_full[df_full["is_liquid"] == True]["date"].max()

    print(f"  Prediction date: {pred_date.date()}")

    df_day = df_full[
        (df_full["date"] == pred_date) &
        (df_full["is_liquid"] == True)
    ].copy().reset_index(drop=True)

    print(f"  Liquid companies: {len(df_day)}")

    if len(df_day) < 10:
        print("ERROR: not enough companies on this date.")
        sys.exit(1)

    # ── Greenblatt ────────────────────────────────────────────────────────────
    df_day["gb_score"] = greenblatt_score(df_day)
    df_day["gb_rank"]  = df_day["gb_score"].rank(ascending=False).astype(int)

    # ── ML predictions ───────────────────────────────────────────────────────
    # Iter1
    feats_i1_avail = [f for f in features_i1 if f in df_day.columns]
    missing_i1 = set(features_i1) - set(feats_i1_avail)
    if missing_i1:
        print(f"  Warning: {len(missing_i1)} missing features for Iter1 "
              f"(filling with NaN): {list(missing_i1)[:5]}")
    X_i1 = df_day.reindex(columns=features_i1).astype("float32")
    df_day["pred_i1"] = m1.predict(X_i1)
    df_day["rank_i1"] = df_day["pred_i1"].rank(pct=True)

    # Iter3
    feats_i3_avail = [f for f in features_i3 if f in df_day.columns]
    missing_i3 = set(features_i3) - set(feats_i3_avail)
    if missing_i3:
        print(f"  Warning: {len(missing_i3)} missing features for Iter3: "
              f"{list(missing_i3)[:5]}")
    X_i3 = df_day.reindex(columns=features_i3).astype("float32")
    df_day["pred_i3"] = m3.predict(X_i3)
    df_day["rank_i3"] = df_day["pred_i3"].rank(pct=True)

    # Ensemble
    df_day["ensemble_score"] = 0.5 * df_day["rank_i1"] + 0.5 * df_day["rank_i3"]
    df_day["ml_rank"] = df_day["ensemble_score"].rank(ascending=False, method="first").astype(int)
    df_day = df_day.sort_values("ml_rank").reset_index(drop=True)

    # ── Output table ──────────────────────────────────────────────────────────
    display_cols = [
        "ml_rank", "ticker", "branza", "close",
        "pe", "roe", "ev_ebitda", "debt_to_equity",
        "rank_i1", "rank_i3", "ensemble_score", "gb_rank",
    ]
    # Attach ret_60d if available (trend context)
    if "ret_60d" in df_day.columns:
        display_cols.append("ret_60d")

    out = df_day[display_cols].head(args.top).copy()

    # Formatting for display
    header = (f"\n{'='*90}\n"
              f"ML PICKS — {pred_date.date()}\n"
              f"Ensemble: 50% Iter1 (regressor) + 50% Iter3 (LambdaRank)\n"
              f"{'='*90}\n")

    col_header = (f"{'Rank':>4}  {'Ticker':<10}  {'Sector':<26}  "
                  f"{'Price':>7}  {'PE':>7}  {'ROE':>7}  {'EV/E':>7}  "
                  f"{'D/E':>6}  {'r1%':>5}  {'r3%':>5}  {'GB':>5}"
                  f"{'  Ret60d':>8}")
    sep = "─" * 90

    lines = [header, col_header, sep]

    for i, row in out.iterrows():
        rank    = int(row["ml_rank"])
        ticker  = str(row["ticker"])
        branza  = str(row.get("branza", ""))[:25]
        price   = fmt_val(row.get("close"), ".2f")
        pe      = fmt_val(row.get("pe"), ".1f")
        roe     = fmt_pct(row.get("roe"))
        ev_e    = fmt_val(row.get("ev_ebitda"), ".1f")
        de      = fmt_val(row.get("debt_to_equity"), ".2f")
        r1_pct  = f"{row['rank_i1']*100:.0f}"
        r3_pct  = f"{row['rank_i3']*100:.0f}"
        gb      = str(int(row["gb_rank"]))
        ret60   = fmt_pct(row["ret_60d"]) if "ret_60d" in row else ""

        marker = " <<<" if rank <= TOP_N_PORTFOLIO else ""
        line = (f"{rank:>4}  {ticker:<10}  {branza:<26}  "
                f"{price:>7}  {pe:>7}  {roe:>7}  {ev_e:>7}  "
                f"{de:>6}  {r1_pct:>5}  {r3_pct:>5}  {gb:>5}"
                f"  {ret60:>6}{marker}")
        lines.append(line)

        if rank == TOP_N_PORTFOLIO:
            lines.append(f"{'─'*40} (outside the portfolio) {'─'*31}")

    footer = [
        sep,
        f"  r1% = Iter1 rank-percentyl  |  r3% = Iter3 rank-percentyl  |  "
        f"GB = Greenblatt rank (1=best)",
        f"  <<< = portfolio position (top {TOP_N_PORTFOLIO})",
        f"  Universe: {len(df_day)} liquid companies",
        sep,
    ]
    lines.extend(footer)
    report = "\n".join(lines)
    print(report)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_day.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"Predictions generated: {datetime.now().isoformat()}\n")
        f.write(f"Models trained at: {trained_at}\n")
        f.write(report + "\n")
    print(f"\n  Full ranking: {OUTPUT_CSV}")
    print(f"  Report:        {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
