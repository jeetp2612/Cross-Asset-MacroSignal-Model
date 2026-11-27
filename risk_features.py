import os
import sys

import numpy as np
import pandas as pd

HORIZON = 21
ASSETS = ["SPY", "TLT", "GLD", "UUP"]

path = "data/market.csv"
if not os.path.exists(path):
    path = "data/dataset.csv"
if not os.path.exists(path):
    sys.exit("Could not find data/market.csv. Run build_dataset.py first.")

data = pd.read_csv(path, index_col=0, parse_dates=True)
returns = data[ASSETS].pct_change()
scale = np.sqrt(252) * 100

features = pd.DataFrame(index=data.index)

for asset in ASSETS:
    daily = returns[asset]
    features[asset + "_absret"] = daily.abs() * 100
    features[asset + "_vol5"] = daily.rolling(5).std() * scale
    features[asset + "_vol21"] = daily.rolling(21).std() * scale
    features[asset + "_vol63"] = daily.rolling(63).std() * scale
    features[asset + "_ret21"] = data[asset].pct_change(21) * 100
    features[asset + "_dd252"] = (data[asset] / data[asset].rolling(252).max() - 1) * 100

oil_returns = data["OIL"].pct_change()
oil_returns[oil_returns.abs() > 1] = np.nan
features["OIL_vol21"] = oil_returns.rolling(21, min_periods=15).std() * scale

features["VIX"] = data["VIX"]
features["VIX_chg5"] = data["VIX"].pct_change(5) * 100
features["VIX_vs_avg63"] = data["VIX"] / data["VIX"].rolling(63).mean()
features["VIX_minus_vol"] = data["VIX"] - features["SPY_vol21"]

features["CURVE"] = data["CURVE_10Y_2Y"]
features["CURVE_chg21"] = data["CURVE_10Y_2Y"].diff(21)
features["US10Y_chg21"] = data["US10Y"].diff(21)
features["US2Y_chg21"] = data["US2Y"].diff(21)
features["FED_chg63"] = data["FED_FUNDS"].diff(63)

if "HYG" in data.columns:
    hyg = data["HYG"].pct_change()
    features["HYG_vol21"] = hyg.rolling(21).std() * scale
    features["HYG_ret21"] = data["HYG"].pct_change(21) * 100
    features["CREDIT_STRESS"] = features["HYG_ret21"] - features["TLT_ret21"]

if "INFL_EXPECT" in data.columns:
    features["INFL_EXPECT"] = data["INFL_EXPECT"]
    features["INFL_EXPECT_chg21"] = data["INFL_EXPECT"].diff(21)

if "FIN_COND" in data.columns:
    features["FIN_COND"] = data["FIN_COND"]
    features["FIN_COND_chg21"] = data["FIN_COND"].diff(21)

if "CLAIMS" in data.columns:
    features["CLAIMS_chg21"] = data["CLAIMS"].pct_change(21) * 100

if "UNRATE" in data.columns:
    features["UNRATE_chg63"] = data["UNRATE"].diff(63)

if "CPI" in data.columns:
    features["CPI_yoy"] = (data["CPI"] / data["CPI"].shift(252) - 1) * 100

features = features.replace([np.inf, -np.inf], np.nan).dropna()

targets = pd.DataFrame(index=data.index)
for asset in ASSETS:
    # Shift the 21-day rolling volatility back to the day before that period starts.
    targets[asset + "_fwdvol"] = returns[asset].rolling(HORIZON).std().shift(-HORIZON) * scale

final = features.join(targets.loc[features.index])
final.index.name = "date"

os.makedirs("data", exist_ok=True)
temporary = "data/risk_features.csv.tmp"
final.to_csv(temporary)
os.replace(temporary, "data/risk_features.csv")

print("rows:", len(final))
print("number of risk signals:", len(features.columns))
print("from", final.index.min().date(), "to", final.index.max().date())
print("latest day with a known answer:", final.dropna().index.max().date())
print()
print("signals:", ", ".join(features.columns))
print()
print("average next-month volatility (annual %):")
print(targets.loc[final.index].mean().round(2))
