import pandas as pd

data = pd.read_csv("data/dataset.csv", index_col=0, parse_dates=True)

features = pd.DataFrame(index=data.index)

for asset in ["SPY", "TLT", "GLD", "UUP", "OIL"]:
    features[asset + "_ret_1d"] = data[asset].pct_change(1)
    features[asset + "_ret_5d"] = data[asset].pct_change(5)
    features[asset + "_ret_21d"] = data[asset].pct_change(21)
    features[asset + "_ret_63d"] = data[asset].pct_change(63)

features["SPY_vol_21d"] = data["SPY"].pct_change().rolling(21).std()
features["TLT_vol_21d"] = data["TLT"].pct_change().rolling(21).std()

features["VIX_level"] = data["VIX"]
features["VIX_chg_5d"] = data["VIX"].pct_change(5)
features["VIX_vs_avg"] = data["VIX"] / data["VIX"].rolling(63).mean()

features["CURVE"] = data["CURVE_10Y_2Y"]
features["CURVE_chg_21d"] = data["CURVE_10Y_2Y"].diff(21)
features["US10Y_chg_5d"] = data["US10Y"].diff(5)
features["US10Y_chg_21d"] = data["US10Y"].diff(21)
features["US2Y_chg_5d"] = data["US2Y"].diff(5)
features["FED_FUNDS"] = data["FED_FUNDS"]

features["GOLD_STOCK_ratio"] = data["GLD"] / data["SPY"]
features["GOLD_STOCK_chg_21d"] = features["GOLD_STOCK_ratio"].pct_change(21)

labels = data[["SPY_up", "TLT_up", "GLD_up", "UUP_up"]]

final = features.join(labels)
final = final.dropna()

final.to_csv("data/features.csv")

print("rows:", len(final))
print("number of signal columns:", len(features.columns))
print("from", final.index.min().date(), "to", final.index.max().date())
print()
print("share of days the price went up over the next 21 days:")
print(labels.loc[final.index].mean().round(3))
