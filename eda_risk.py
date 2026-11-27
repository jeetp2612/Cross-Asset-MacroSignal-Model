import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ASSETS = ["SPY", "TLT", "GLD", "UUP"]
MAX_LAG = 60

os.makedirs("data/eda", exist_ok=True)

if not os.path.exists("data/risk_features.csv"):
    sys.exit("Could not find data/risk_features.csv. Run risk_features.py first.")

market_path = "data/market.csv" if os.path.exists("data/market.csv") else "data/dataset.csv"
market = pd.read_csv(market_path, index_col=0, parse_dates=True)
risk = pd.read_csv("data/risk_features.csv", index_col=0, parse_dates=True)
returns = market[ASSETS].pct_change().dropna()
known = risk.dropna()

print("=" * 60)
print("SECTION 1: volatility clustering (does today's risk tell us about tomorrow's?)")
print("=" * 60)

lags = list(range(1, MAX_LAG + 1))
acf_table = pd.DataFrame({"lag": lags})
for asset in ASSETS:
    acf_table[asset + "_returns"] = [returns[asset].autocorr(lag) for lag in lags]
    acf_table[asset + "_abs_returns"] = [returns[asset].abs().autocorr(lag) for lag in lags]
acf_table.to_csv("data/eda/risk_acf.csv", index=False)

band = 1.96 / np.sqrt(len(returns))
print("95% noise band: +/-", round(band, 3))
for asset in ASSETS:
    print(
        asset,
        "| returns lag 1:", round(acf_table.loc[0, asset + "_returns"], 3),
        "| absolute returns lag 1:", round(acf_table.loc[0, asset + "_abs_returns"], 3),
        "lag 21:", round(acf_table.loc[20, asset + "_abs_returns"], 3),
        "lag 60:", round(acf_table.loc[59, asset + "_abs_returns"], 3),
    )

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for asset in ASSETS:
    axes[0].plot(lags, acf_table[asset + "_returns"], label=asset)
    axes[1].plot(lags, acf_table[asset + "_abs_returns"], label=asset)
for ax, title in zip(axes, ["Returns: no memory (direction)", "Absolute returns: long memory (risk)"]):
    ax.axhspan(-band, band, color="grey", alpha=0.25, label="noise band")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("lag (trading days)")
    ax.set_ylabel("autocorrelation")
    ax.legend()
plt.tight_layout()
plt.savefig("data/eda/risk_acf.png")
plt.close()

print()
print("=" * 60)
print("SECTION 2: fear (VIX) versus the risk that actually followed")
print("=" * 60)

pair = known[["VIX", "SPY_fwdvol"]]
pearson = stats.pearsonr(pair["VIX"], pair["SPY_fwdvol"])
spearman = stats.spearmanr(pair["VIX"], pair["SPY_fwdvol"])
slope, intercept = np.polyfit(pair["VIX"], pair["SPY_fwdvol"], 1)
premium = pair["VIX"] - pair["SPY_fwdvol"]
print("rows used:", len(pair))
print("Pearson correlation:", round(pearson[0], 3), "| Spearman:", round(spearman[0], 3))
print("each extra VIX point goes with", round(slope, 2), "more points of realised volatility")
print("average gap (VIX minus realised volatility):", round(premium.mean(), 2), "| VIX was above realised risk in", round((premium > 0).mean() * 100, 1), "% of months")

buckets = pd.cut(pair["VIX"], bins=[0, 15, 25, 1000], labels=["calm (<15)", "normal (15-25)", "fearful (>25)"])
bucket_table = pair.groupby(buckets, observed=True)["SPY_fwdvol"].agg(["count", "mean", "median", "std"])
bucket_table.to_csv("data/eda/risk_vix_buckets.csv")
print()
print("next-month SPY volatility by VIX level:")
print(bucket_table.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(pair["VIX"], pair["SPY_fwdvol"], s=6, alpha=0.4)
x = np.linspace(pair["VIX"].min(), pair["VIX"].max(), 50)
axes[0].plot(x, intercept + slope * x, color="red", label="fitted line")
axes[0].plot(x, x, color="grey", linestyle="--", label="VIX equals realised risk")
axes[0].set_xlabel("VIX today")
axes[0].set_ylabel("SPY volatility over the next 21 days (annual %)")
axes[0].set_title("Fear today versus risk that followed")
axes[0].legend()
groups = [pair.loc[buckets == name, "SPY_fwdvol"] for name in buckets.cat.categories]
axes[1].boxplot(groups)
axes[1].set_xticks(range(1, len(groups) + 1))
axes[1].set_xticklabels(list(buckets.cat.categories))
axes[1].set_ylabel("SPY volatility over the next 21 days (annual %)")
axes[1].set_title("Next-month risk by fear level")
plt.tight_layout()
plt.savefig("data/eda/risk_vix_vs_fwdvol.png")
plt.close()

print()
print("=" * 60)
print("SECTION 3: which signals go with next month's risk? (feature screening)")
print("=" * 60)

signal_columns = [c for c in known.columns if not c.endswith("_fwdvol")]
screen_rows = []
for column in signal_columns:
    result = stats.spearmanr(known[column], known["SPY_fwdvol"])
    screen_rows.append({"signal": column, "spearman": result[0], "p_value": result[1]})
screen = pd.DataFrame(screen_rows)
screen["strength"] = screen["spearman"].abs()
screen = screen.sort_values("strength", ascending=False).drop(columns="strength")
screen.to_csv("data/eda/risk_signal_screen.csv", index=False)
print("top 12 signals by absolute Spearman correlation with SPY next-month volatility:")
print(screen.head(12).round(3).to_string(index=False))
print()
print("weakest 5:")
print(screen.tail(5).round(3).to_string(index=False))
print("(rows overlap by 20 days, so the p-values are optimistic: use the sizes, not the stars)")

top = screen.head(15).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top["signal"], top["spearman"], color=["tab:red" if v > 0 else "tab:blue" for v in top["spearman"]])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Spearman correlation with SPY volatility over the next 21 days")
plt.title("Signals that go with next month's risk")
plt.tight_layout()
plt.savefig("data/eda/risk_signal_screen.png")
plt.close()

persistence_rows = []
for asset in ASSETS:
    result = stats.spearmanr(known[asset + "_vol21"], known[asset + "_fwdvol"])
    persistence_rows.append({"asset": asset, "spearman_now_vs_next_month": result[0]})
persistence = pd.DataFrame(persistence_rows)
persistence.to_csv("data/eda/risk_persistence.csv", index=False)
print()
print("risk persistence (this month's volatility versus next month's):")
print(persistence.round(3).to_string(index=False))

print()
print("=" * 60)
print("SECTION 4: does the economic backdrop change next month's risk?")
print("=" * 60)

regime_rows = []


def compare(name, mask):
    first = known.loc[mask, "SPY_fwdvol"]
    second = known.loc[~mask, "SPY_fwdvol"]
    if len(first) < 30 or len(second) < 30:
        return
    test = stats.mannwhitneyu(first, second, alternative="two-sided")
    regime_rows.append({
        "backdrop": name,
        "days_yes": len(first),
        "days_no": len(second),
        "median_vol_yes": first.median(),
        "median_vol_no": second.median(),
        "ratio": first.median() / second.median(),
        "p_value": test.pvalue,
    })


compare("yield curve inverted", known["CURVE"] < 0)
if "FIN_COND" in known:
    compare("financial conditions tighter than median", known["FIN_COND"] > known["FIN_COND"].median())
if "CLAIMS_chg21" in known:
    compare("jobless claims rising", known["CLAIMS_chg21"] > 0)
if "CREDIT_STRESS" in known:
    compare("high-yield bonds lagging safe bonds", known["CREDIT_STRESS"] < 0)
if "INFL_EXPECT" in known:
    compare("inflation expectations above median", known["INFL_EXPECT"] > known["INFL_EXPECT"].median())
if "UNRATE_chg63" in known:
    compare("unemployment rising", known["UNRATE_chg63"] > 0)
compare("fear (VIX) above 25", known["VIX"] > 25)

regimes = pd.DataFrame(regime_rows)
regimes.to_csv("data/eda/risk_regime_tests.csv", index=False)
print(regimes.round(3).to_string(index=False))
print("(Mann-Whitney U test; rows overlap, so p-values are approximate)")

if not regimes.empty:
    ordered = regimes.sort_values("ratio")
    plt.figure(figsize=(9, 4.5))
    plt.barh(ordered["backdrop"], ordered["ratio"], color="tab:orange")
    plt.axvline(1, color="black", linewidth=0.8)
    plt.xlabel("typical next-month SPY volatility: backdrop present / absent")
    plt.title("Which backdrops go with wilder months?")
    plt.tight_layout()
    plt.savefig("data/eda/risk_regimes.png")
    plt.close()

print()
print("=" * 60)
print("KEY NUMBERS")
print("=" * 60)
print("absolute-return memory at lag 21 (SPY):", round(acf_table.loc[20, "SPY_abs_returns"], 3))
print("return memory at lag 1 (SPY):", round(acf_table.loc[0, "SPY_returns"], 3))
print("VIX vs next-month SPY volatility, Spearman:", round(spearman[0], 3))
print("strongest signal:", screen.iloc[0]["signal"], round(screen.iloc[0]["spearman"], 3))
print("files written to data/eda/: risk_acf, risk_vix_vs_fwdvol, risk_signal_screen, risk_regimes (png) and matching csv files")
