import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("data/eda", exist_ok=True)

data = pd.read_csv("data/dataset.csv", index_col=0, parse_dates=True)

assets = ["SPY", "TLT", "GLD", "UUP", "OIL"]
four = ["SPY", "TLT", "GLD", "UUP"]
labels = ["low (<15)", "medium (15-25)", "high (>25)"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig("data/eda/" + name, dpi=120)
    plt.close(fig)


def draw_heatmap(table, title, filename, percent):
    fig, ax = plt.subplots(figsize=(1.2 * len(table.columns) + 3, 0.5 * len(table.index) + 3))
    if percent:
        limit = table.abs().values.max()
        colors = "RdYlGn"
    else:
        limit = 1
        colors = "coolwarm"
    image = ax.imshow(table.values, cmap=colors, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.values[i, j]
            if percent:
                text = f"{value * 100:.0f}%"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    save(fig, filename)


print("=== 1. OVERVIEW ===")
print("rows:", len(data), " columns:", len(data.columns))
print("from", data.index.min().date(), "to", data.index.max().date())
print("missing values:", int(data.isna().sum().sum()))
print("duplicate dates:", int(data.index.duplicated().sum()))
print("lowest oil price:", round(data["OIL"].min(), 2), "on", data["OIL"].idxmin().date())
print()
summary = data[assets + ["VIX", "US10Y", "US2Y", "CURVE_10Y_2Y", "FED_FUNDS"]].describe().T
print(summary.round(3).to_string())
summary.to_csv("data/eda/01_summary_statistics.csv")

returns = data[assets].pct_change().dropna()
clean = returns[(returns.abs() < 1).all(axis=1)]
print()
print("days removed because one asset moved more than 100% in a day:", len(returns) - len(clean))

print()
print("=== 2. DAILY RETURNS: SHAPE AND OUTLIERS ===")
shape_table = pd.DataFrame({
    "mean_pct": clean.mean() * 100,
    "std_pct": clean.std() * 100,
    "min_pct": clean.min() * 100,
    "max_pct": clean.max() * 100,
    "skew": clean.skew(),
    "excess_kurtosis": clean.kurt(),
    "annual_volatility_pct": clean.std() * (252 ** 0.5) * 100,
})
print(shape_table.round(3).to_string())
shape_table.to_csv("data/eda/02_return_statistics.csv")

outlier_rows = []
for asset in assets:
    r = returns[asset]
    z = (r - r.mean()) / r.std()
    q1 = r.quantile(0.25)
    q3 = r.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = ((r < q1 - 1.5 * iqr) | (r > q3 + 1.5 * iqr)).sum()
    outlier_rows.append({
        "asset": asset,
        "days_beyond_3_std": int((z.abs() > 3).sum()),
        "days_outside_iqr_fences": int(iqr_outliers),
        "share_outside_iqr_pct": iqr_outliers / len(r) * 100,
        "largest_one_day_move_pct": r.abs().max() * 100,
    })
outliers = pd.DataFrame(outlier_rows)
print()
print(outliers.round(2).to_string(index=False))
outliers.to_csv("data/eda/03_outliers.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
panels = [(a + " daily return (%)", clean[a] * 100) for a in assets]
panels.append(("VIX daily change (points)", data["VIX"].diff().dropna()))
for ax, (title, series) in zip(axes.flatten(), panels):
    ax.hist(series, bins=60)
    ax.set_title(title)
save(fig, "01_return_histograms.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([clean[a] * 100 for a in assets])
ax.set_xticklabels(assets)
ax.set_title("Daily returns (%) by asset")
save(fig, "02_return_boxplots.png")

fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(clean["SPY"], dist="norm", plot=ax)
ax.set_title("SPY daily returns against a normal distribution")
save(fig, "03_spy_qq_plot.png")

print()
print("=== 3. OVER TIME ===")
rebased = data[four] / data[four].iloc[0] * 100
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(rebased)
ax1.legend(four)
ax1.set_title("Prices rebased to 100 at the start")
ax2.plot(data["OIL"], color="black")
ax2.set_title("Oil price")
save(fig, "04_prices_over_time.png")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["VIX"], color="black", linewidth=1)
ax.fill_between(data.index, 0, data["VIX"].max(), where=(data["VIX"] > 30).values, color="red", alpha=0.3)
ax.set_title("VIX (red = above 30)")
save(fig, "05_vix_over_time.png")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data["CURVE_10Y_2Y"], color="black", linewidth=1)
ax.fill_between(data.index, data["CURVE_10Y_2Y"], 0, where=(data["CURVE_10Y_2Y"] < 0).values, color="red", alpha=0.4)
ax.axhline(0, color="gray")
ax.set_title("Yield curve, 10-year minus 2-year (red = inverted)")
save(fig, "06_yield_curve.png")

vix_high_share = (data["VIX"] > 30).mean() * 100
inverted_share = (data["CURVE_10Y_2Y"] < 0).mean() * 100
print("share of days with VIX above 30:", round(vix_high_share, 1), "%")
print("share of days with an inverted yield curve:", round(inverted_share, 1), "%")

print()
print("=== 4. RELATIONSHIPS ===")
relations = clean.copy()
relations["VIX_chg"] = data["VIX"].diff().loc[clean.index]
relations["US10Y_chg"] = data["US10Y"].diff().loc[clean.index]
corr = relations.corr()
print(corr.round(2).to_string())
corr.to_csv("data/eda/04_correlations.csv")
draw_heatmap(corr, "Correlation of daily moves", "07_correlation_heatmap.png", False)

relations["UUP_pct"] = relations["UUP"] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax, x_name, x_label, y_name in [(ax1, "VIX_chg", "VIX daily change (points)", "SPY"), (ax2, "UUP_pct", "UUP daily return (%)", "GLD")]:
    x = relations[x_name]
    y = relations[y_name]
    ax.scatter(x, y * 100, s=4, alpha=0.4)
    slope, intercept = np.polyfit(x, y * 100, 1)
    ax.plot(x, slope * x + intercept, color="red")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_name + " daily return (%)")
    ax.set_title(y_name + " against " + x_name)
save(fig, "08_scatter_plots.png")

rolling = pd.DataFrame({
    "SPY vs TLT": clean["SPY"].rolling(252).corr(clean["TLT"]),
    "SPY vs GLD": clean["SPY"].rolling(252).corr(clean["GLD"]),
    "SPY vs UUP": clean["SPY"].rolling(252).corr(clean["UUP"]),
})
fig, ax = plt.subplots(figsize=(12, 5))
rolling.plot(ax=ax)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("1-year rolling correlation with stocks (SPY)")
save(fig, "09_rolling_correlation.png")

print()
print("=== 5. REGIMES ===")
vix_regime = pd.cut(data["VIX"].shift(1), bins=[0, 15, 25, 200], labels=labels).loc[clean.index]
mean_table = clean.groupby(vix_regime, observed=True).mean() * 100
std_table = clean.groupby(vix_regime, observed=True).std() * 100
days_table = vix_regime.value_counts().reindex(labels).dropna().astype(int)
print("average daily return (%) the day after VIX was in each range:")
print(mean_table.round(3).to_string())
print()
print("daily volatility (%) the day after VIX was in each range:")
print(std_table.round(3).to_string())
print()
print("number of days in each range:")
print(days_table.to_string())
pd.concat([mean_table.add_suffix("_mean_pct"), std_table.add_suffix("_std_pct")], axis=1).to_csv("data/eda/05_vix_regimes.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
present = mean_table.index.tolist()
ax1.boxplot([clean.loc[vix_regime == name, "SPY"] * 100 for name in present])
ax1.set_xticklabels(present)
ax1.set_title("SPY daily return (%) by previous-day VIX range")
mean_table[four].plot(kind="bar", ax=ax2)
ax2.set_title("Average daily return (%) by previous-day VIX range")
ax2.tick_params(axis="x", rotation=0)
save(fig, "10_vix_regimes.png")

inverted = (data["CURVE_10Y_2Y"].shift(1) < 0).loc[clean.index]
curve_table = clean.groupby(inverted).mean() * 100
curve_table.index = ["inverted curve" if value else "normal curve" for value in curve_table.index]
print()
print("average daily return (%) after a normal or inverted yield curve:")
print(curve_table.round(3).to_string())
curve_table.to_csv("data/eda/06_curve_regimes.csv")

yearly = (1 + clean[four]).groupby(clean.index.year).prod() - 1
print()
print("return by calendar year (%):")
print((yearly * 100).round(1).to_string())
yearly.to_csv("data/eda/07_yearly_returns.csv")
draw_heatmap(yearly, "Return by calendar year", "11_yearly_returns.png", True)

print()
print("=== 6. STATISTICAL TESTS ===")
test_rows = []

jb, jb_p = stats.jarque_bera(clean["SPY"])
test_rows.append({"test": "SPY returns are normal (Jarque-Bera)", "statistic": jb, "p_value": jb_p})

high = clean.loc[vix_regime == labels[2], "SPY"]
low = clean.loc[vix_regime == labels[0], "SPY"]
if len(high) > 1 and len(low) > 1:
    t, t_p = stats.ttest_ind(high, low, equal_var=False)
    test_rows.append({"test": "SPY mean return differs after high vs low VIX (Welch t-test)", "statistic": t, "p_value": t_p})

before = clean.loc[:"2021-12-31"]
after = clean.loc["2022-01-01":]
r_before, p_before = stats.pearsonr(before["SPY"], before["TLT"])
r_after, p_after = stats.pearsonr(after["SPY"], after["TLT"])
r_all, p_all = stats.pearsonr(clean["SPY"], clean["TLT"])
test_rows.append({"test": "SPY-TLT correlation, whole period", "statistic": r_all, "p_value": p_all})
test_rows.append({"test": "SPY-TLT correlation, before 2022", "statistic": r_before, "p_value": p_before})
test_rows.append({"test": "SPY-TLT correlation, 2022 onward", "statistic": r_after, "p_value": p_after})

tests = pd.DataFrame(test_rows)
print(tests.round(4).to_string(index=False))
tests.to_csv("data/eda/08_tests.csv", index=False)

print()
print("=== KEY NUMBERS ===")
print("SPY annual volatility (%):", round(shape_table.loc["SPY", "annual_volatility_pct"], 1))
print("SPY excess kurtosis:", round(shape_table.loc["SPY", "excess_kurtosis"], 1))
print("best SPY year:", yearly["SPY"].idxmax(), round(yearly["SPY"].max() * 100, 1), "%")
print("worst SPY year:", yearly["SPY"].idxmin(), round(yearly["SPY"].min() * 100, 1), "%")
print("SPY-VIX change correlation:", round(corr.loc["SPY", "VIX_chg"], 2))
print("SPY-TLT correlation before 2022:", round(r_before, 2), " from 2022:", round(r_after, 2))
print("share of days with VIX above 30 (%):", round(vix_high_share, 1))
print("share of days with inverted curve (%):", round(inverted_share, 1))
print()
print("charts and tables saved in data/eda")
