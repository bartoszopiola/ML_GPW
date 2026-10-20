"""
Backtest chart for Phase 3C.

Generates plot_backtest.png with:
  - Top panel: equity curves (ML Ensemble, Greenblatt, Universe EW)
  - Bottom panel: drawdown

Usage:
    python plot_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("ml_data")
EQUITY_CSV = OUT_DIR / "phase3c_equity_curve.csv"
PERIOD_CSV = OUT_DIR / "phase3c_period_returns.csv"
OUTPUT_PNG = OUT_DIR / "plot_backtest.png"

PORTFOLIO_START = 10_000

WINDOW_BOUNDARIES = [
    ("2023-05-08", "2023-10-31"),
    ("2023-11-06", "2024-04-30"),
    ("2024-05-08", "2024-10-31"),
    ("2024-11-06", "2025-04-30"),
    ("2025-05-08", "2025-10-31"),
    ("2025-11-06", "2026-03-24"),
]

COLORS = {
    "ML Ensemble": "#2563EB",    # blue
    "Greenblatt":  "#DC2626",    # red
    "Universe EW": "#6B7280",    # grey
}

WINDOW_COLORS = ["#EFF6FF", "#FFF7ED", "#F0FDF4", "#FDF4FF", "#FFFBEB", "#F0F9FF"]


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    return (series - peak) / peak * 100  # percent


def format_pct(x, pos):
    return f"{x:+.0f}%"


def format_kpln(x, pos):
    return f"{x/1000:.1f}k"


def main():
    # ─── Data ──────────────────────────────────────────────────────────────────
    equity = pd.read_csv(EQUITY_CSV, index_col=0, parse_dates=True)
    periods = pd.read_csv(PERIOD_CSV, parse_dates=["date"])

    # Fill in dates — start from day 0
    t0 = pd.Timestamp("2023-05-08") - pd.Timedelta("1D")
    start_row = pd.DataFrame(
        {col: PORTFOLIO_START for col in equity.columns},
        index=[t0],
    )
    equity = pd.concat([start_row, equity]).sort_index()

    drawdowns = {col: compute_drawdown(equity[col]) for col in equity.columns}

    # Metrics for the legend
    def annualized(series):
        total = series.iloc[-1] / PORTFOLIO_START - 1
        n_days = (series.index[-1] - series.index[0]).days
        return (1 + total) ** (365 / n_days) - 1

    def max_dd(series):
        return compute_drawdown(series).min()

    def sharpe(model_label):
        df_m = periods[periods["model"] == model_label]
        if df_m.empty or "net_return" not in df_m.columns:
            return np.nan
        r = df_m["net_return"].values
        return np.mean(r) / np.std(r) * np.sqrt(252 / 20) if np.std(r) > 0 else np.nan

    # ─── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    fig.patch.set_facecolor("#FAFAFA")
    for ax in (ax1, ax2):
        ax.set_facecolor("#FAFAFA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")
        ax.tick_params(colors="#4B5563", labelsize=9)
        ax.yaxis.label.set_color("#374151")
        ax.xaxis.label.set_color("#374151")
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7, zorder=0)

    # ─── Test window bands ─────────────────────────────────────────────────────
    for i, (ws, we) in enumerate(WINDOW_BOUNDARIES):
        ts, te = pd.Timestamp(ws), pd.Timestamp(we)
        for ax in (ax1, ax2):
            ax.axvspan(ts, te, alpha=0.35, color=WINDOW_COLORS[i], zorder=1)
        # Window label on the top panel
        mid = ts + (te - ts) / 2
        ax1.text(mid, ax1.get_ylim()[1] if i == 0 else 1,
                 f"W{i+1}", ha="center", va="bottom",
                 fontsize=7.5, color="#9CA3AF", fontweight="bold")

    # ─── Equity curves ─────────────────────────────────────────────────────────
    order = ["ML Ensemble", "Universe EW", "Greenblatt"]
    lw_map = {"ML Ensemble": 2.5, "Greenblatt": 1.8, "Universe EW": 1.5}
    ls_map = {"ML Ensemble": "-", "Greenblatt": "--", "Universe EW": ":"}
    zorder_map = {"ML Ensemble": 5, "Greenblatt": 4, "Universe EW": 3}

    for col in order:
        if col not in equity.columns:
            continue
        s = equity[col]
        ann = annualized(s)
        sp = sharpe(col)
        mdd = max_dd(s)
        label = (f"{col}  |  ann: {ann:+.1%}  "
                 f"Sharpe: {sp:.2f}  MaxDD: {mdd:.1%}")
        ax1.plot(
            s.index, s.values,
            color=COLORS[col], linewidth=lw_map[col],
            linestyle=ls_map[col], label=label,
            zorder=zorder_map[col],
        )

    # Starting line
    ax1.axhline(PORTFOLIO_START, color="#9CA3AF", linewidth=0.8,
                linestyle="-", alpha=0.6, zorder=2)

    # End point labels
    last_date = equity.index[-1]
    for col in order:
        if col not in equity.columns:
            continue
        val = equity[col].iloc[-1]
        ax1.annotate(
            f"{val/1000:.1f}k PLN",
            xy=(last_date, val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=COLORS[col], fontweight="bold",
            va="center",
        )

    ax1.set_ylabel("Portfolio value (PLN)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(format_kpln))
    ax1.set_title(
        "GPW portfolio backtest — Phase 3C\n"
        "5 positions · rebalance every 20 days · commission 0.39% · spread 0.1%",
        fontsize=12, fontweight="bold", color="#1F2937", pad=12,
    )

    leg = ax1.legend(
        loc="upper left", fontsize=8.2,
        framealpha=0.92, edgecolor="#D1D5DB",
        facecolor="white",
    )
    for text in leg.get_texts():
        text.set_color("#374151")

    # Window labels (after the axes are set)
    ax1.autoscale_view()
    ymin1, ymax1 = ax1.get_ylim()
    for i, (ws, we) in enumerate(WINDOW_BOUNDARIES):
        mid = pd.Timestamp(ws) + (pd.Timestamp(we) - pd.Timestamp(ws)) / 2
        ax1.text(mid, ymax1 * 0.99, f"W{i+1}",
                 ha="center", va="top",
                 fontsize=7.5, color="#9CA3AF", fontweight="bold")

    # ─── Drawdown panel ────────────────────────────────────────────────────────
    for col in order:
        if col not in equity.columns:
            continue
        dd = drawdowns[col]
        ax2.plot(dd.index, dd.values,
                 color=COLORS[col], linewidth=lw_map[col],
                 linestyle=ls_map[col], zorder=zorder_map[col])
        ax2.fill_between(dd.index, dd.values, 0,
                          color=COLORS[col], alpha=0.08, zorder=2)

    ax2.axhline(0, color="#9CA3AF", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # ─── ML rebalances (markers) ───────────────────────────────────────────────
    ml_periods = periods[periods["model"] == "ML Ensemble"].copy()
    ml_turnover = ml_periods[ml_periods["n_buys"] > 0]
    for _, row in ml_turnover.iterrows():
        ax1.axvline(row["date"], color=COLORS["ML Ensemble"],
                    alpha=0.15, linewidth=0.7, zorder=2)

    # ─── Window annotations on the X axis ──────────────────────────────────────
    # Per-window cumulative returns as text above the band
    ml_w = ml_periods.copy()
    ml_w["window"] = None
    for i, (ws, we) in enumerate(WINDOW_BOUNDARIES):
        mask = (ml_w["date"] >= pd.Timestamp(ws)) & (ml_w["date"] <= pd.Timestamp(we))
        ml_w.loc[mask, "window"] = i + 1

    ymin2, ymax2 = ax2.get_ylim()
    for i, (ws, we) in enumerate(WINDOW_BOUNDARIES):
        w_data = ml_w[ml_w["window"] == i + 1]
        if w_data.empty:
            continue
        cum_ret = (1 + w_data["net_return"]).prod() - 1
        mid = pd.Timestamp(ws) + (pd.Timestamp(we) - pd.Timestamp(ws)) / 2
        ax2.text(mid, ymax2 * 0.92, f"{cum_ret:+.0%}",
                 ha="center", va="top", fontsize=7.5,
                 color=COLORS["ML Ensemble"], fontweight="bold")

    # ─── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Chart saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
