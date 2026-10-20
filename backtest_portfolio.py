"""
Phase 3C - portfolio backtest with transaction costs

Simulates a monthly rebalance of a top-5 portfolio based on the ML signal (ensemble)
and compares it with Greenblatt and the WIG benchmark.

Portfolio parameters:
  - Starting capital: 10,000 PLN
  - 5 equal-weighted positions
  - Rebalance every REBALANCE_FREQ trading days (~monthly)
  - Buffer: keep a stock if it's still in the top (N + BUFFER) = top 10
  - Broker commission: max(0.39%, min 5 PLN) per trade
  - Bid-ask spread: 0.1% per side (entry and exit)

Input data:
  - phase3b_ensemble_oos_predictions.parquet (OOS predictions from eval_ensemble.py)

Outputs:
  - phase3c_period_returns.csv     — per-period returns of each strategy
  - phase3c_equity_curve.csv       — portfolio value after each rebalance
  - phase3c_report.txt             — full report

Usage:
    python backtest_portfolio.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── PARAMETERS ───────────────────────────────────────────────────────────────

OUT_DIR = Path("ml_data")
ENSEMBLE_PARQUET = OUT_DIR / "phase3b_ensemble_oos_predictions.parquet"

PERIOD_CSV = OUT_DIR / "phase3c_period_returns.csv"
EQUITY_CSV = OUT_DIR / "phase3c_equity_curve.csv"
REPORT_TXT = OUT_DIR / "phase3c_report.txt"

PORTFOLIO_VALUE_START = 10_000   # PLN
N_POSITIONS = 5
BUFFER = 5                       # keep if in top N+BUFFER = top 10
REBALANCE_FREQ = 20              # rebalance every this many trading days

COMMISSION_PCT = 0.0039          # 0.39%
COMMISSION_MIN_PLN = 5.0         # min 5 PLN
SPREAD_PCT = 0.001               # 0.1% per side (entry or exit)

BELKA_RATE = 0.19                # 19% Polish capital gains tax ("Belka" tax)


# ─── TRANSACTION COSTS ────────────────────────────────────────────────────────

def transaction_cost(n_buys: int, n_sells: int, position_value: float) -> float:
    """
    Total transaction cost (PLN).
    Each trade (buy or sell) costs:
      - commission: max(COMMISSION_PCT * value, COMMISSION_MIN_PLN)
      - spread: SPREAD_PCT * value
    """
    n_legs = n_buys + n_sells
    if n_legs == 0:
        return 0.0
    commission = n_legs * max(COMMISSION_PCT * position_value, COMMISSION_MIN_PLN)
    spread = n_legs * SPREAD_PCT * position_value
    return commission + spread


# ─── BACKTEST ─────────────────────────────────────────────────────────────────

def run_backtest(df_oos: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    """
    Simulates a portfolio of N_POSITIONS stocks with a monthly rebalance.

    Buffer logic:
      - On rebalance date t: get the ML ranking for all stocks
      - Keep current positions that are still in the top (N + BUFFER)
      - Fill free slots with new candidates from the top N (not already held)
      - Only buy/sell the actual changes

    Per-period return = mean target_return of the selected stocks - transaction costs
    """
    # All OOS dates, sorted (continuous across all windows)
    all_dates = sorted(df_oos["date"].unique())

    # Rebalance dates every REBALANCE_FREQ days
    rebal_dates = [all_dates[i] for i in range(0, len(all_dates), REBALANCE_FREQ)]

    portfolio: set[str] = set()
    portfolio_value = float(PORTFOLIO_VALUE_START)
    periods = []

    for idx, date in enumerate(rebal_dates):
        day_data = df_oos[
            (df_oos["date"] == date) &
            df_oos[pred_col].notna() &
            df_oos["target_return"].notna()
        ].copy()

        if len(day_data) < N_POSITIONS:
            continue  # too few stocks on this day (shouldn't happen)

        # Sort by prediction, descending
        day_data = day_data.sort_values(pred_col, ascending=False).reset_index(drop=True)
        top_n = set(day_data.head(N_POSITIONS)["ticker"])
        top_buf = set(day_data.head(N_POSITIONS + BUFFER)["ticker"])

        # Buffer: keep positions that are still in the top N+BUFFER
        if not portfolio:
            new_portfolio = top_n
        else:
            keep = portfolio & top_buf
            # Fill free slots with the best candidates not already held
            candidates = [t for t in day_data["ticker"] if t not in keep]
            additions = set(candidates[: N_POSITIONS - len(keep)])
            new_portfolio = keep | additions

        sells = portfolio - new_portfolio
        buys = new_portfolio - portfolio

        # Transaction cost
        pos_size = portfolio_value / N_POSITIONS
        cost_pln = transaction_cost(len(buys), len(sells), pos_size)
        cost_pct = cost_pln / portfolio_value

        # Return: mean target_return of the selected positions
        selected = day_data[day_data["ticker"].isin(new_portfolio)]
        gross_ret = float(selected["target_return"].mean())
        net_ret = gross_ret - cost_pct

        portfolio_value *= (1 + net_ret)
        portfolio = new_portfolio

        periods.append({
            "date": date,
            "model": label,
            "portfolio": ", ".join(sorted(new_portfolio)),
            "n_positions": len(new_portfolio),
            "n_buys": len(buys),
            "n_sells": len(sells),
            "turnover_pct": len(buys) / N_POSITIONS,
            "cost_pln": round(cost_pln, 2),
            "cost_pct": cost_pct,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "portfolio_value": portfolio_value,
        })

    return pd.DataFrame(periods)


def universe_ew(df_oos: pd.DataFrame) -> pd.DataFrame:
    """
    Passive strategy: equal-weighted universe (all stocks, no selection).
    No transaction costs (a proxy for an index).
    """
    all_dates = sorted(df_oos["date"].unique())
    rebal_dates = [all_dates[i] for i in range(0, len(all_dates), REBALANCE_FREQ)]

    pv = float(PORTFOLIO_VALUE_START)
    periods = []

    for date in rebal_dates:
        day_data = df_oos[(df_oos["date"] == date) & df_oos["target_return"].notna()]
        if len(day_data) == 0:
            continue
        ret = float(day_data["target_return"].mean())
        pv *= (1 + ret)
        periods.append({
            "date": date,
            "model": "Universe EW",
            "gross_return": ret,
            "net_return": ret,
            "portfolio_value": pv,
        })

    return pd.DataFrame(periods)


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}

    returns = df["net_return"].values
    values = df["portfolio_value"].values
    n = len(returns)
    total_days = n * REBALANCE_FREQ   # approximation

    total_ret = values[-1] / PORTFOLIO_VALUE_START - 1
    ann_ret = (1 + total_ret) ** (252.0 / total_days) - 1

    # Sharpe (20-day periods → annualized with sqrt(252/20))
    sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 / REBALANCE_FREQ)
              if np.std(returns) > 0 else np.nan)

    # Max drawdown on the equity curve
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / peak
    max_dd = float(dd.min())

    win_rate = float((returns > 0).mean())

    # Costs and turnover (only for strategies that trade)
    avg_cost = float(df["cost_pct"].mean()) if "cost_pct" in df.columns else 0.0
    total_cost = float(df["cost_pct"].sum()) if "cost_pct" in df.columns else 0.0
    avg_turnover = float(df["turnover_pct"].mean()) if "turnover_pct" in df.columns else np.nan

    # Belka tax (simplified: 19% of the total gain)
    gross_gain = values[-1] - PORTFOLIO_VALUE_START
    belka_pln = max(gross_gain * BELKA_RATE, 0.0)
    after_belka = values[-1] - belka_pln
    ret_after_belka = after_belka / PORTFOLIO_VALUE_START - 1
    ann_after_belka = (1 + ret_after_belka) ** (252.0 / total_days) - 1

    return {
        "n_periods": n,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "avg_cost_pct": avg_cost,
        "total_cost_pct": total_cost,
        "final_value_pln": values[-1],
        "belka_pln": belka_pln,
        "final_after_belka_pln": after_belka,
        "ann_return_after_belka": ann_after_belka,
    }


# ─── WIG BENCHMARK (optional) ─────────────────────────────────────────────────

def fetch_wig_benchmark(rebal_dates: list[pd.Timestamp]) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        start = min(rebal_dates) - pd.Timedelta("5D")
        end = max(rebal_dates) + pd.Timedelta("35D")

        wig = yf.download("^WIG", start=str(start.date()), end=str(end.date()),
                          auto_adjust=True, progress=False)
        if len(wig) < 30:
            return None

        close = wig["Close"].squeeze()
        close.index = pd.to_datetime(close.index)
        all_wig_dates = sorted(close.index)

        rows = []
        pv = float(PORTFOLIO_VALUE_START)
        for date in rebal_dates:
            # Find the nearest WIG trading day >= date
            future = [d for d in all_wig_dates if d >= pd.Timestamp(date)]
            if len(future) < REBALANCE_FREQ + 1:
                continue
            t0 = future[0]
            t20 = future[REBALANCE_FREQ]  # 20 WIG trading days later
            ret = float(close[t20] / close[t0] - 1)
            pv *= (1 + ret)
            rows.append({
                "date": date,
                "model": "WIG",
                "gross_return": ret,
                "net_return": ret,
                "portfolio_value": pv,
            })

        return pd.DataFrame(rows) if rows else None

    except Exception as e:
        print(f"  [!] WIG fetch failed: {e}")
        return None


# ─── REPORT ───────────────────────────────────────────────────────────────────

def format_report(all_metrics: dict[str, dict], period_dfs: dict[str, pd.DataFrame],
                  per_window_summary: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 3C — PORTFOLIO BACKTEST WITH TRANSACTION COSTS")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Parameters:")
    lines.append(f"  Starting capital:   {PORTFOLIO_VALUE_START:,.0f} PLN")
    lines.append(f"  Positions:          {N_POSITIONS}")
    lines.append(f"  Rebalance buffer:   top {N_POSITIONS + BUFFER} (keep if in top {N_POSITIONS + BUFFER})")
    lines.append(f"  Rebalance every:    {REBALANCE_FREQ} trading days")
    lines.append(f"  Commission:         max({COMMISSION_PCT*100:.2f}%, min {COMMISSION_MIN_PLN:.0f} PLN)")
    lines.append(f"  Spread bid-ask:     {SPREAD_PCT*100:.1f}% per side")
    lines.append(f"  Belka tax:          {BELKA_RATE*100:.0f}% of the total gain (simplified)")
    lines.append("")

    # Main comparison table
    lines.append("─" * 72)
    lines.append("STRATEGY COMPARISON")
    lines.append("─" * 72)
    header = (f"  {'Strategy':<22} {'Tot ret':>8} {'Ann ret':>8} {'Sharpe':>7} "
              f"{'MaxDD':>7} {'WinRate':>8} {'After tax':>9}")
    lines.append(header)
    lines.append(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")

    for label, m in all_metrics.items():
        if not m:
            continue
        sharpe_str = f"{m['sharpe']:>7.2f}" if not np.isnan(m['sharpe']) else f"{'N/A':>7}"
        belka_str = f"{m['ann_return_after_belka']:>+9.2%}"
        lines.append(
            f"  {label:<22} {m['total_return']:>+8.2%} {m['ann_return']:>+8.2%} "
            f"{sharpe_str} {m['max_drawdown']:>7.2%} {m['win_rate']:>8.1%} {belka_str}"
        )
    lines.append("")

    # Transaction costs
    lines.append("─" * 72)
    lines.append("TRANSACTION COSTS")
    lines.append("─" * 72)
    for label, m in all_metrics.items():
        if "avg_turnover" not in m or np.isnan(m.get("avg_turnover", np.nan)):
            continue
        lines.append(
            f"  {label:<22}  avg turnover={m['avg_turnover']:.0%}/period  "
            f"avg cost={m['avg_cost_pct']:.3%}/period  "
            f"total cost drag={m['total_cost_pct']:.2%}"
        )
    lines.append("")

    # Final portfolio values
    lines.append("─" * 72)
    lines.append("PORTFOLIO VALUE AT THE END OF THE PERIOD (start: 10,000 PLN)")
    lines.append("─" * 72)
    for label, m in all_metrics.items():
        if not m:
            continue
        gain = m["final_value_pln"] - PORTFOLIO_VALUE_START
        lines.append(
            f"  {label:<22}  {m['final_value_pln']:>8,.0f} PLN  "
            f"(gross gain: {gain:>+,.0f} PLN, after tax: {m['final_after_belka_pln']:>,.0f} PLN)"
        )
    lines.append("")

    # Per-window top-5 returns
    if not per_window_summary.empty:
        lines.append("─" * 72)
        lines.append("RETURNS PER TEST WINDOW (mean top-5 per day x N periods in the window)")
        lines.append("─" * 72)
        lines.append(per_window_summary.to_string(index=False))
        lines.append("")

    # Per-period details for the ML ensemble
    if "ML Ensemble" in period_dfs:
        df_ml = period_dfs["ML Ensemble"]
        lines.append("─" * 72)
        lines.append("REBALANCE DETAILS — ML ENSEMBLE")
        lines.append("─" * 72)
        lines.append(f"  {'Date':>12} {'Portfolio (top 5)':>45} {'Buy':>4} {'Gross':>7} {'Cost%':>6} {'Net':>7} {'Value PLN':>10}")
        lines.append(f"  {'-'*12} {'-'*45} {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*10}")
        for _, row in df_ml.iterrows():
            port_str = row["portfolio"][:45]
            lines.append(
                f"  {str(row['date'].date()):>12} {port_str:>45} "
                f"{row['n_buys']:>4} {row['gross_return']:>+7.3%} "
                f"{row['cost_pct']:>6.3%} {row['net_return']:>+7.3%} "
                f"{row['portfolio_value']:>10,.0f}"
            )
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("PHASE 3C — PORTFOLIO BACKTEST WITH TRANSACTION COSTS")
    print("=" * 72)

    print(f"\nLoading {ENSEMBLE_PARQUET}...")
    df = pd.read_parquet(ENSEMBLE_PARQUET)
    print(f"  {len(df):,} rows, {df['date'].nunique()} unique dates, "
          f"{df['ticker'].nunique()} companies")
    print(f"  Okres: {df['date'].min().date()} -> {df['date'].max().date()}")

    # ─── RUN STRATEGIES ────────────────────────────────────────────────────────

    print("\nRunning the backtest...")
    print("  [1/4] ML Ensemble...")
    df_ml = run_backtest(df, "pred_ensemble", "ML Ensemble")

    print("  [2/4] Greenblatt...")
    df_gb = run_backtest(df, "pred_gb_norm", "Greenblatt")

    print("  [3/4] Universe EW (passive benchmark)...")
    df_ew = universe_ew(df)

    print("  [4/4] WIG benchmark (yfinance)...")
    rebal_dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())[::REBALANCE_FREQ]]
    df_wig = fetch_wig_benchmark(rebal_dates)
    if df_wig is not None:
        print(f"        WIG: {len(df_wig)} periods fetched.")
    else:
        print("        WIG unavailable — skipping.")

    # ─── METRICS ──────────────────────────────────────────────────────────────

    strategy_dfs = {"ML Ensemble": df_ml, "Greenblatt": df_gb, "Universe EW": df_ew}
    if df_wig is not None:
        strategy_dfs["WIG"] = df_wig

    all_metrics: dict[str, dict] = {}
    for label, df_s in strategy_dfs.items():
        all_metrics[label] = compute_metrics(df_s)

    # ─── SUMMARY PRINT ─────────────────────────────────────────────────────────

    print(f"\n  {'Strategy':<22} {'Tot ret':>8} {'Ann ret':>8} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'WinRate':>8} {'After tax':>9}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")
    for label, m in all_metrics.items():
        if not m:
            continue
        sharpe_str = f"{m['sharpe']:>7.2f}" if not np.isnan(m['sharpe']) else f"{'N/A':>7}"
        print(f"  {label:<22} {m['total_return']:>+8.2%} {m['ann_return']:>+8.2%} "
              f"{sharpe_str} {m['max_drawdown']:>7.2%} {m['win_rate']:>8.1%} "
              f"{m['ann_return_after_belka']:>+9.2%}")

    print()
    for label, m in all_metrics.items():
        if "avg_turnover" not in m or np.isnan(m.get("avg_turnover", np.nan)):
            continue
        print(f"  {label:<22}  turnover={m['avg_turnover']:.0%}/period  "
              f"avg_cost={m['avg_cost_pct']:.3%}/period  "
              f"total_cost_drag={m['total_cost_pct']:.2%}")

    print(f"\n  Portfolio value as of {df_ml['date'].max().date() if len(df_ml) > 0 else '?'}:")
    for label, m in all_metrics.items():
        if not m:
            continue
        print(f"  {label:<22}  {m['final_value_pln']:>8,.0f} PLN "
              f"(after tax: {m['final_after_belka_pln']:>,.0f} PLN)")

    # ─── PER-WINDOW SUMMARY ────────────────────────────────────────────────────

    # Assign windows to rebalance dates
    date_to_window = df.groupby("date")["window"].first()
    rows_per_window = []
    for label, df_s in {"ML Ensemble": df_ml, "Greenblatt": df_gb}.items():
        if df_s.empty:
            continue
        df_s2 = df_s.copy()
        df_s2["window"] = df_s2["date"].map(date_to_window)
        for w, g in df_s2.groupby("window"):
            rows_per_window.append({
                "window": w,
                "model": label,
                "n_periods": len(g),
                "avg_net_return": g["net_return"].mean(),
                "total_net_return": (1 + g["net_return"]).prod() - 1,
                "win_rate": (g["net_return"] > 0).mean(),
            })

    pw_df = pd.DataFrame(rows_per_window)
    if not pw_df.empty:
        pw_pivot = pw_df.pivot_table(
            index="window",
            columns="model",
            values=["avg_net_return", "total_net_return"],
        ).round(4)
        print(f"\n  Per-window cumulative net return:")
        print(f"  {'Win':>4} {'ML Ensemble':>13} {'Greenblatt':>13}  {'ML wins?':>10}")
        print(f"  {'-'*4} {'-'*13} {'-'*13}  {'-'*10}")
        ml_wins = 0
        for w in sorted(pw_df["window"].unique()):
            ml_r = pw_df[(pw_df["window"] == w) & (pw_df["model"] == "ML Ensemble")]["total_net_return"]
            gb_r = pw_df[(pw_df["window"] == w) & (pw_df["model"] == "Greenblatt")]["total_net_return"]
            ml_v = float(ml_r.values[0]) if len(ml_r) else np.nan
            gb_v = float(gb_r.values[0]) if len(gb_r) else np.nan
            win = "ML" if ml_v > gb_v else "GB"
            if ml_v > gb_v:
                ml_wins += 1
            print(f"  {w:>4} {ml_v:>+13.2%} {gb_v:>+13.2%}  {win:>10}")
        total_wins = len(sorted(pw_df["window"].unique()))
        print(f"  {'':>4} {'':>13} {'':>13}  ML wygrywa: {ml_wins}/{total_wins}")

    # ─── SAVE ──────────────────────────────────────────────────────────────────

    all_periods = pd.concat(
        [df_s for df_s in strategy_dfs.values() if not df_s.empty],
        ignore_index=True,
    )
    all_periods.to_csv(PERIOD_CSV, index=False)

    # Equity curve (wide table)
    equity_rows = []
    for label, df_s in strategy_dfs.items():
        for _, row in df_s.iterrows():
            equity_rows.append({
                "date": row["date"],
                "model": label,
                "portfolio_value": row["portfolio_value"],
                "net_return": row["net_return"],
            })
    equity_df = pd.DataFrame(equity_rows)
    equity_pivot = equity_df.pivot_table(
        index="date", columns="model", values="portfolio_value"
    ).round(2)
    equity_pivot.to_csv(EQUITY_CSV)

    print(f"\n  Saved:")
    print(f"    {PERIOD_CSV}")
    print(f"    {EQUITY_CSV}")

    # ─── REPORT TXT ────────────────────────────────────────────────────────────

    per_window_df = pw_df if not pw_df.empty else pd.DataFrame()
    report_str = format_report(all_metrics, {"ML Ensemble": df_ml, "Greenblatt": df_gb},
                                per_window_df)
    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"    {REPORT_TXT}")

    # ─── VERDICT ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 72)
    print("VERDICT: DOES THE ML ENSEMBLE BEAT WIG AFTER COSTS AND TAX?")
    print("=" * 72)
    ml_m = all_metrics.get("ML Ensemble", {})
    wig_m = all_metrics.get("WIG") or all_metrics.get("Universe EW", {})
    bench_label = "WIG" if "WIG" in all_metrics else "Universe EW"
    if ml_m and wig_m:
        print(f"  ML Ensemble  ann. (after tax): {ml_m['ann_return_after_belka']:>+.2%}")
        print(f"  {bench_label:<12} ann.:            {wig_m['ann_return']:>+.2%}")
        alpha = ml_m["ann_return_after_belka"] - wig_m["ann_return"]
        print(f"  Alpha (ML after tax vs {bench_label}):  {alpha:>+.2%}")
        if alpha > 0:
            print(f"  ML ENSEMBLE BEATS THE BENCHMARK even after costs and tax!")
        else:
            print(f"  ML ENSEMBLE does not beat the benchmark after costs and tax.")
    print("=" * 72)


if __name__ == "__main__":
    main()
