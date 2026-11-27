import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ASSETS = ["SPY", "TLT", "GLD", "UUP"]
COST = 0.0005

for name in ["data/forecast_predictions.csv", "data/forecast_latest.csv"]:
    if not os.path.exists(name):
        sys.exit("Could not find " + name + ". Run forecast.py first.")

market_path = "data/market.csv" if os.path.exists("data/market.csv") else "data/dataset.csv"
market = pd.read_csv(market_path, index_col=0, parse_dates=True)
predictions = pd.read_csv("data/forecast_predictions.csv", index_col=0, parse_dates=True)
latest = pd.read_csv("data/forecast_latest.csv").set_index("asset")

returns = market[ASSETS].pct_change()
past_vol = returns.rolling(21).std() * np.sqrt(252) * 100
target_vol = past_vol.expanding(252).median()


def stats_of(daily):
    curve = (1 + daily).cumprod()
    years = len(daily) / 252
    yearly = curve.iloc[-1] ** (1 / years) - 1
    swing = daily.std() * np.sqrt(252)
    return {
        "total_return": (curve.iloc[-1] - 1) * 100,
        "yearly_return": yearly * 100,
        "yearly_swing": swing * 100,
        "sharpe": yearly / swing if swing > 0 else 0,
        "worst_fall": (curve / curve.cummax() - 1).min() * 100,
    }


def run_strategy(forecast, asset, dates):
    position = (target_vol[asset].reindex(dates) / forecast.reindex(dates)).clip(0, 1)
    position = position.fillna(1.0)
    # Use yesterday's position for today's return so today's return cannot affect today's choice.
    held = position.shift(1).fillna(1.0)
    daily = held * returns[asset].reindex(dates) - COST * position.diff().abs().shift(1).fillna(0)
    return daily, held


rows = []
curves = pd.DataFrame()
for asset in ASSETS:
    best = latest.loc[asset, "best_model"]
    dates = predictions.index[predictions.index > predictions.index[0]]
    dates = dates.intersection(returns.index)
    hold = returns[asset].reindex(dates)
    naive_daily, naive_held = run_strategy(predictions[asset + "_naive"], asset, dates)
    model_daily, model_held = run_strategy(predictions[asset + "_" + best], asset, dates)
    for label, daily, held in [("buy_and_hold", hold, pd.Series(1.0, index=dates)), ("naive_rule", naive_daily, naive_held), ("risk_forecast", model_daily, model_held)]:
        row = {"asset": asset, "strategy": label, "average_exposure": held.mean() * 100}
        row.update(stats_of(daily))
        rows.append(row)
        curves[asset + "_" + label] = (1 + daily).cumprod()

results = pd.DataFrame(rows)
results.to_csv("data/vol_target_results.csv", index=False)
curves.to_csv("data/vol_target_curves.csv")

print("Risk-managed investing: hold less when the forecast says risk is high")
print("position = (usual risk) / (forecast risk), never above 100% (no borrowing), cost 0.05% per change")
print()
print(results.round(2).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, asset in zip(axes.flatten(), ASSETS):
    ax.plot(curves[asset + "_buy_and_hold"], color="tab:gray", label="Buy and hold")
    ax.plot(curves[asset + "_risk_forecast"], color="tab:red", label="Using the risk forecast")
    ax.set_title(asset)
    ax.legend()
plt.tight_layout()
plt.savefig("data/vol_target.png")
