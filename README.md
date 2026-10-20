# ML_GPW

A machine learning stock-ranking pipeline for the Warsaw Stock Exchange (GPW). Every day it scores ~140 Polish companies and picks the 5 with the best expected 20-day return relative to their sector. The goal was a tool that needs very little time per month and can be checked against a simple, well-known benchmark: Greenblatt's "magic formula".

The project covers the whole path: data collection (prices + scraped fundamentals), point-in-time feature engineering, walk-forward model evaluation, a portfolio backtest with transaction costs and tax, and a daily prediction script.

## Pipeline

```
fetch_ohlcv.py          prices from yfinance (5 years, per-company cache, liquidity filter)
fetch_fundamentals.py   fundamentals scraped from biznesradar.pl (6 report pages per company)
        ↓
merge_fundamentals.py   point-in-time merge of prices and fundamentals
build_features.py       technical, fundamental, change and cross-sectional features
prepare_ml_dataset.py   20-day target, time-based splits, filtering
        ↓
train_walking_forward_*.py   walk-forward evaluation (6 windows)
backtest_portfolio.py        monthly top-5 portfolio with costs and tax
train_final_model.py         final models trained on all data
predict_today.py             today's ranking
```

`refresh_data.py` runs the data steps and the prediction in one go.

## Data

- **Universe**: 140 GPW companies (`tickers.py`), 139 with price history.
- **Prices**: daily OHLCV from yfinance, 2021-03 to 2026-04 (~171k rows).
- **Fundamentals**: 474k rows covering 135 metrics and up to 22 years of quarterly and annual reports scraped from biznesradar.pl. 25 metrics are used: valuation, profitability, debt, liquidity, per-share values.
- **Liquidity filter**: a company is tradeable on a given day only if its 30-day average turnover is at least 100,000 PLN.

### Avoiding look-ahead bias

Financial reports are published weeks after the quarter ends. Each report is only made available to the model after a publication margin:

| Report | Available after period end |
|---|---|
| Q1, Q3 | +75 days |
| Q2 (half-year) | +95 days |
| Q4 (annual) | +125 days |

The fundamentals are forward-filled per metric rather than per report, because some values (e.g. total assets, net income) are published only annually.

## Features

About 80 features in five groups:

- **Technical**: returns over 1-60 days, moving average ratios, RSI, MACD, Bollinger Bands, ATR, volume, 52-week position, candlestick shape
- **Fundamental levels**: P/E, EV/EBITDA, ROE, ROA, margins, debt ratios, plus log transforms for skewed values
- **Fundamental changes**: quarter-on-quarter and year-on-year changes (profitability trends, dilution)
- **Cross-sectional ranks**: percentile of each metric within the whole universe and within the company's sector, per day
- **Calendar**: day of week, month

## Target and validation

- **Target**: 20-day open-to-open forward return. The signal is computed after the close on day *t*, the stock is bought at the open of *t+1* and sold 20 trading days later.
- **Sector-neutral target** (from iteration 1): the return minus the average return of the company's super-sector (8 groups). The model learns which company will beat its sector, not which sector will go up.
- **Walk-forward validation**: 6 windows, each tested on ~6 months of data the model has never seen (2023-05 to 2026-03). A 25-day gap separates training and test data so overlapping 20-day returns can't leak.
- **Model**: LightGBM, as a regressor and as a LambdaRank ranker. In each window, features are selected by importance and the model is retrained on the selected set.

## Results

Mean over the 6 walk-forward windows. IC is the daily Spearman correlation between the model's scores and actual returns. Top-5 is the average 20-day return of the 5 highest-ranked companies.

| Approach | Mean IC | Top-5 20-day return |
|---|---|---|
| Iter 0: regressor, raw target | +0.069 | +1.76% |
| Iter 1: regressor, sector-neutral target | +0.052 | +4.63% |
| Iter 2: LambdaRank | +0.027 | +2.41% |
| Iter 3: LambdaRank, tuned | +0.037 | +4.40% |
| Greenblatt (P/E + ROE rank) | +0.079 | +3.33% |

What came out of this:

- **The simple Greenblatt ranking has the highest IC.** No model beat it at ranking the whole universe.
- **IC and top-5 return don't always move together.** The sector-neutral target lowered IC but almost tripled the top-5 return. For a 5-stock portfolio, only the top of the ranking matters.
- The final signal is an equal-weight ensemble of Iter 1 and Iter 3 (average of their rank percentiles).
- Iterations 0 and 2 and the ensemble evaluation were run with earlier scripts that are not included in this repository. Their numbers come from the experiment log. The ensemble's out-of-sample predictions used by the backtest are included in `ml_data/`.

### Portfolio backtest

Monthly rebalance of 5 equal-weight positions from the ensemble's out-of-sample predictions. A stock is kept while it stays in the top 10, to limit turnover. Costs: 0.39% commission (min 5 PLN) and 0.1% spread per side. The 19% Polish capital gains tax is applied in simplified form to the total gain. Starting capital: 10,000 PLN. Period: 2023-05 to 2026-03.

| Strategy | Total return | Annual return | Sharpe | Max drawdown | After tax (annual) |
|---|---|---|---|---|---|
| ML ensemble | +329% | +66.5% | 2.05 | -7.8% | +57.6% |
| Greenblatt top-5 | +68% | +20.0% | 1.19 | -13.0% | +16.7% |
| Equal-weight universe | +80% | +22.9% | 1.82 | -7.6% | +19.2% |

### Limitations

These backtest numbers are almost certainly too optimistic:

- **Survivorship bias**: the universe is a list of companies that are listed today, so companies that went bankrupt or were delisted during the period are missing.
- **Selection bias**: the ensemble and its settings were chosen after looking at the walk-forward results, so the test windows are not fully independent.
- **Small caps**: the portfolio often holds small, volatile companies, where real trades would move prices more than the 0.1% spread assumes.
- **Short period**: about 3 years and 36 rebalances. One very strong window (late 2025 to early 2026) drives a large part of the result.

The next step is paper trading, comparing the live picks with the backtest before using real money.

## Usage

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# 1. (one-off) map tickers to biznesradar URLs - biznesradar_mapping.json is already included
python resolve_biznesradar.py

# 2. download data and build the dataset
python refresh_data.py --no-predict

# 3. walk-forward evaluation
python train_walking_forward_excess.py
python train_walking_forward_lambdarank_v2.py

# 4. portfolio backtest and chart
python backtest_portfolio.py
python plot_backtest.py

# 5. train the final models and rank today's companies
python train_final_model.py
python predict_today.py
```

After the first run, `python refresh_data.py` updates the data and prints the current top picks. Downloads are cached, so later runs only fetch new data.

The large datasets (`ml_data/dataset_*.parquet`) and download caches are not in the repository. Step 2 rebuilds them. The trained models, reports, out-of-sample predictions and the backtest chart are included.

## Project structure

```
tickers.py                    company list (GPW name → Yahoo Finance symbol)
biznesradar_mapping.json      company → biznesradar.pl URL
resolve_biznesradar.py        builds the mapping above
fetch_fundamentals.py         fundamentals scraper
merge_fundamentals.py         point-in-time merge
build_features.py             feature engineering
prepare_ml_dataset.py         target, splits, filtering
train_walking_forward_*.py    walk-forward experiments
backtest_portfolio.py         portfolio backtest
plot_backtest.py              backtest chart
train_final_model.py          final models
predict_today.py              daily ranking
refresh_data.py               runs the whole pipeline
ml_data/
  fetch_ohlcv.py              price downloader
  final_model_*.txt           trained LightGBM models
  phase3*_report.txt          experiment and backtest reports
  plot_backtest.png           backtest chart
```

## Disclaimer

Educational project. Backtest results don't guarantee future returns and nothing here is investment advice.