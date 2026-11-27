# Cross-Asset Macro Signal

A desktop dashboard for **stocks, bonds, gold and the US dollar**. It shows how the markets are behaving now, and forecasts **next month's risk** (how wildly each market will swing) with models that are tested year by year against a simple rule.

Python and Exploratory Data Analysis (EDA) work together: the EDA scripts find what is predictable (risk, not direction), that finding chooses the models, and the same year-by-year test judges them. Direction models are kept as a documented negative result.

Pure Python. No Kaggle data. Educational tool, **not investment advice**.

For a detailed, beginner-friendly explanation of the six screens, calculations, source files, models and presentation roles, read [`PROJECT_GUIDE.pdf`](PROJECT_GUIDE.pdf).

## Simple localhost website

For a simple browser demo for a college presentation:

```bash
python web_server.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

No extra web packages are required. This page reads the existing CSV files in `data/`.

## What you see in the app

| Screen | What it gives you |
|---|---|
| Dashboard | Market mood banner, five indicator cards (fear meter, yield curve, stocks vs bonds, last 12 months, next-month risk), and a chart with Growth / Drawdown / Risk views, period buttons and hover |
| Markets | Table of price, 1m/3m/12m change, risk now, fall from peak and next-month risk, plus who-moves-with-whom charts |
| Risk Forecast | Pick a market and a model; see forecast vs what happened and a scoreboard of all models against the naive rule |
| Put to Work | Tests the forecast in money terms: hold less when forecast risk is high, and compare return, worst fall and Sharpe ratio with buy and hold |
| Build Your Mix | Split money across the four markets; see past growth, worst fall and expected risk next month |
| Reality Check | Risk forecasts vs direction predictions, side by side |

## How the project works

1. `build_dataset.py` downloads prices, VIX, oil, high-yield bonds (HYG) and FRED data (rates, inflation expectations, financial conditions, jobless claims, unemployment, CPI). Slow economic series are shifted by their publication delay so the model never sees numbers before they were released.
2. `eda.py` and `eda_risk.py` explore the data: volatility clustering, VIX vs later risk, signal screening, economic backdrop tests.
3. `risk_features.py` builds the risk signals and the target: volatility over the next 21 days.
4. `forecast.py` compares five models: naive, HAR (memory of recent risk), HAR plus fear and credit signals, boosted trees, and a blend. Each year from 2015 is forecast using only earlier years.
5. `forecast.py` also tests whether each gain is real (Diebold-Mariano test and bootstrap range) and builds a wild-month alert (logistic regression on the risk signals). `vol_target.py` then tests the forecast in money terms.
6. The app shows the results. Everything is honest: if a model does not beat the naive rule, the app says so.

## What “the model” means in this project

This project does **not** use ChatGPT, another hosted AI service, or a downloaded pretrained model. It trains ordinary statistical and machine-learning models on this computer using **scikit-learn** (listed in `requirements.txt`). There are no saved model-weight files: the forecast scripts fit the models again when the data is updated.

The desktop app's main prediction is **next-month market risk**, not whether a market will go up or down. Its files and models are:

| File | What it does |
|---|---|
| `build_dataset.py` | Downloads prices from Yahoo Finance and economic series from FRED. It writes the input price/economic data. |
| `macro.py` | Delays each economic value until its release date, so historical forecasts do not accidentally use information that was not available yet. |
| `risk_features.py` | Turns prices and economic data into signals such as recent volatility and VIX, and creates the answer to predict: volatility over the following 21 trading days. |
| `forecast.py` | Trains and compares the five risk forecasts below, testing each calendar year using only earlier years. It writes forecast scores, predictions, alerts, and the chart under `data/`. |
| `vol_target.py` | Uses those forecasts to test a simple risk-managed investment strategy; it does not train another forecast model. |

The five forecasts in `forecast.py` are:

1. **Naive** — assumes the next month's volatility will equal the latest measured volatility. This is the comparison rule, not a learned model.
2. **HAR** — linear regression using each market's recent 5-, 21-, and 63-day volatility.
3. **HARX** — scaled ridge regression using recent risk and selected fear, credit, and economic signals.
4. **Boosted** — scikit-learn's `HistGradientBoostingRegressor`, which learns non-linear combinations of the risk signals.
5. **Blend** — averages the three learned models' predictions on the log-volatility scale.

`forecast.py` also fits a **logistic-regression alert** to estimate whether a month will be unusually volatile (above the 75th percentile of training-period risk). This alert is separate from the five volatility estimates.

There is a second, separate experiment about **market direction**:

| File | What it does |
|---|---|
| `signals.py` | Builds the direction signals and up/down labels in `data/features.csv`. |
| `walk_forward.py` | Tests logistic regression, random forest, and gradient boosting year by year. The updater runs it, and the app's Reality Check uses its results. |
| `model.py` | A separate direction-model experiment with those three classifier types and a single train/test split. It is not part of the app's update pipeline. |
| `backtest.py` | An optional analysis script that uses the predictions written by `model.py`; it is not part of the app's update pipeline. |

`eda.py` and `eda_risk.py` are explanatory analysis scripts, not models needed to start the app. The desktop interface is `run.py` → `gui.py`; `web_server.py` and the `web/` folder provide a separate, simple browser demo. Both interfaces read prepared files from `data/`.

## Install and run

You need **Python 3.10 or newer** (with tkinter, which is included in the normal python.org installers).

1. Unzip the folder anywhere.
2. Start it:
   - **Windows:** double-click `launch_windows.bat`
   - **Mac / Linux:** run `./launch_mac_linux.sh`
   - **Any system:** open a terminal in the folder and run `python run.py`
3. If packages are missing, the launcher offers to install them. Or run `python -m pip install -r requirements.txt` yourself.
4. On the very first start there is no data yet. The app asks to download it (internet needed, a few minutes, first time only).

Keep the terminal open while the app runs; closing it closes the app.

## Keeping the data fresh

Press **Update data** in the app. It downloads the latest data, rebuilds the risk signals, re-runs the year-by-year forecast test and the direction test, and restarts the app (a few minutes). If anything fails, your old data is kept and the reason is shown. A log is saved in `data/update_log.txt`.

The top bar turns amber when the data is more than 5 days old.

## Data sources

- Prices and VIX: Yahoo Finance, through the `yfinance` library (SPY, TLT, GLD, UUP, ^VIX, CL=F, HYG)
- Economic data: FRED, the St. Louis Fed (DGS10, DGS2, T10Y2Y, DFF, T10YIE, NFCI, ICSA, UNRATE, CPIAUCSL). HYG comes from Yahoo. Optional series that fail to download are skipped

These are free for personal and educational use. Check their terms before any other use, especially commercial use. Yahoo Finance data is not licensed for resale.

## Files

| File | Purpose |
|---|---|
| `run.py` | Launcher with dependency checks |
| `gui.py` | The window |
| `logic.py` | All calculations and wording, with no window code (this is what the tests cover) |
| `updater.py` | Runs the update steps in the background |
| `build_dataset.py` | Downloads and cleans data (`market.csv`, `dataset.csv`) |
| `macro.py` | Shifts economic series by their release delay |
| `risk_features.py` | Builds risk signals and next-month volatility targets |
| `forecast.py` | Year-by-year test of the five risk models; latest forecasts |
| `vol_target.py` | Risk-managed backtest: forecast turned into positions, compared with buy and hold |
| `eda_risk.py` | EDA behind the risk forecast (saved in `data/eda/`) |
| `signals.py` | Builds the direction signals (`features.csv`) |
| `walk_forward.py` | Year-by-year model test (`walk_forward_summary.csv`), direction result, shown on Reality Check |
| `eda.py`, `model.py`, `backtest.py` | Analysis scripts for the course report. The app does not need them |
| `tests/` | Automated tests |

Analysis scripts run from this folder: `python eda.py`, `python model.py`, `python backtest.py`.

## Presentation split (three members)

Each member can focus on three files. The files are grouped by what they do, so you can explain one part of the project without presenting every line in every file.

| Member | Files | Simple presentation outline |
|---|---|---|
| **Shlok — machine learning** | `risk_features.py`, `forecast.py`, `walk_forward.py` | Explain how past prices become risk signals and a future-volatility target; compare the five risk forecasts in `forecast.py`; then show how `walk_forward.py` tests direction classifiers using earlier years only. |
| **Jeet — desktop app** | `gui.py`, `logic.py`, `updater.py` | Show the screens in `gui.py`; explain that `logic.py` keeps calculations and user-facing explanations separate from the window; finish with how `updater.py` runs the data/model scripts in order. |
| **Dhruvit — easiest demo** | `web_server.py`, `web/index.html`, `web/app.js` | Start the browser demo; describe how the HTML lays out the page, the JavaScript requests prepared results from the local Python server, and the server returns JSON from `data/`. `web/style.css` only controls appearance and can be mentioned briefly. |

Keep the other files as supporting material rather than merging them into the presentation files: `build_dataset.py` and `macro.py` prepare safe historical inputs; `signals.py` builds direction inputs; `vol_target.py` tests risk-managed positions; `eda.py` and `eda_risk.py` explore the data; `model.py` and `backtest.py` are optional direction-analysis experiments. `run.py` starts the desktop app. These separate responsibilities make the code easier to find, test, and explain; combining modules would create larger files without reducing the actual work the app needs to do.

## Tests

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests cover the calculations, wording, data loading, the update runner, release-lag handling, a no-lookahead check on the signals, and the whole pipeline on simulated data. The window itself is not covered by automated tests.

## Troubleshooting

- **Nothing appears:** the window may be behind other windows. Check the taskbar.
- **"tkinter is missing":** reinstall Python from python.org and keep tcl/tk ticked. On Linux run `sudo apt install python3-tk`.
- **Update fails:** check your internet connection, then read `data/update_log.txt`. Your old data stays in place.

## Limits

- Past results do not predict future returns. Nothing here is advice.
- Build Your Mix uses real past prices in US dollars, ignores fees and taxes, and rebalances the mix every day.
- Statistical tests assume independent days; volatility clustering breaks that, so p-values are approximate.
- The dollar fund (UUP) and the other funds are proxies for the real markets.

## Licence

MIT. See `LICENSE`. Replace "The project authors" in that file with your team's names.



conda activate base
hash -r
/opt/anaconda3/bin/python3.14 run.py# Cross-Asset-MacroSignal-Model
# Cross-Asset-MacroSignal-Model
