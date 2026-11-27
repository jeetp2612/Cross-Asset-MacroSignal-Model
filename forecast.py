import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GAP = 21
FIRST_YEAR = 2015
WILD_SHARE = 0.75
SHIFT = 0.1
ALERT_LEVEL = 0.25
HAC_LAGS = 20
BLOCK = 21
BOOTSTRAPS = 500

ASSETS = ["SPY", "TLT", "GLD", "UUP"]
MODEL_NAMES = ["naive", "har", "harx", "boosted", "blend"]
RISK_EXTRA = ["VIX", "VIX_chg5", "VIX_vs_avg63", "VIX_minus_vol", "OIL_vol21", "HYG_vol21", "CREDIT_STRESS", "FIN_COND", "FIN_COND_chg21"]

if not os.path.exists("data/risk_features.csv"):
    sys.exit("Could not find data/risk_features.csv. Run risk_features.py first.")

data = pd.read_csv("data/risk_features.csv", index_col=0, parse_dates=True)
target_columns = [asset + "_fwdvol" for asset in ASSETS]
signal_columns = [c for c in data.columns if c not in target_columns]
risk_columns = [c for c in signal_columns if c.endswith(("_vol5", "_vol21", "_vol63")) or c in RISK_EXTRA]
log_columns = [c for c in signal_columns if c.endswith(("_vol5", "_vol21", "_vol63")) or c == "VIX"]


def prepare(frame):
    table = frame[signal_columns].copy()
    for column in log_columns:
        table[column] = np.log(table[column] + SHIFT)
    return table


def back(values):
    return np.maximum(np.exp(values) - SHIFT, 0)


def fit_predict(train, test, asset):
    # Fit on log volatility so positive risk values are easier for models to predict.
    y = np.log(train[asset + "_fwdvol"] + SHIFT)
    x_train = prepare(train)
    x_test = prepare(test)
    har_columns = [asset + "_vol5", asset + "_vol21", asset + "_vol63"]

    har = LinearRegression().fit(x_train[har_columns], y)
    harx = make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100, 1000])).fit(x_train[risk_columns], y)
    boosted = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=150, min_samples_leaf=100, l2_regularization=5.0, random_state=0).fit(x_train, y)

    cutoff = train[asset + "_fwdvol"].quantile(WILD_SHARE)
    wild_labels = (train[asset + "_fwdvol"] > cutoff).astype(int)
    alert = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(x_train[risk_columns], wild_labels)

    har_log = har.predict(x_test[har_columns])
    harx_log = harx.predict(x_test[risk_columns])
    boosted_log = boosted.predict(x_test)

    # Convert the learned log predictions back to annualized volatility percentages.
    return {
        "naive": test[asset + "_vol21"].values,
        "har": back(har_log),
        "harx": back(harx_log),
        "boosted": back(boosted_log),
        "blend": back((har_log + harx_log + boosted_log) / 3),
        "wildprob": alert.predict_proba(x_test[risk_columns])[:, 1],
    }


test_years = [y for y in sorted(set(data.index.year)) if y >= FIRST_YEAR]
if not test_years:
    sys.exit("Not enough history for the year-by-year test.")

pieces = []
yearly_rows = []
wild_checks = []
alert_checks = []

for year in test_years:
    train = data[data.index < str(year) + "-01-01"].iloc[:-GAP]
    test = data[data.index.year == year]
    block = pd.DataFrame(index=test.index)

    for asset in ASSETS:
        target = asset + "_fwdvol"
        train_asset = train.dropna(subset=[target])
        predictions = fit_predict(train_asset, test, asset)
        block[asset + "_actual"] = test[target]
        block[asset + "_wildprob"] = predictions["wildprob"]
        for name in MODEL_NAMES:
            block[asset + "_" + name] = predictions[name]

        known = test[target].notna().values
        if known.sum() == 0:
            continue
        cutoff = train_asset[target].quantile(WILD_SHARE)
        actual = test[target].values[known]
        naive_mse = np.mean((predictions["naive"][known] - actual) ** 2)
        alert_checks.append({"asset": asset, "wild_actual": actual > cutoff, "alert": predictions["wildprob"][known] > ALERT_LEVEL, "naive": predictions["naive"][known] > cutoff})
        for name in MODEL_NAMES:
            guess = predictions[name][known]
            mse = np.mean((guess - actual) ** 2)
            yearly_rows.append({"year": year, "asset": asset, "model": name, "rmse": np.sqrt(mse), "naive_rmse": np.sqrt(naive_mse), "beat_naive": int(mse < naive_mse)})
            wild_checks.append({"asset": asset, "model": name, "wild_actual": actual > cutoff, "wild_guess": guess > cutoff})

    pieces.append(block)
    print("finished year", year)

predictions = pd.concat(pieces)
yearly = pd.DataFrame(yearly_rows)

score_rows = []
for asset in ASSETS:
    known = predictions[asset + "_actual"].notna()
    actual = predictions.loc[known, asset + "_actual"]
    naive_mse = ((predictions.loc[known, asset + "_naive"] - actual) ** 2).mean()
    for name in MODEL_NAMES:
        guess = predictions.loc[known, asset + "_" + name]
        mse = ((guess - actual) ** 2).mean()
        by_year = yearly[(yearly["asset"] == asset) & (yearly["model"] == name)]
        checks = [c for c in wild_checks if c["asset"] == asset and c["model"] == name]
        wild_actual = np.concatenate([c["wild_actual"] for c in checks])
        wild_guess = np.concatenate([c["wild_guess"] for c in checks])
        caught = (wild_guess & wild_actual).sum() / max(wild_actual.sum(), 1)
        score_rows.append({
            "asset": asset,
            "model": name,
            "rmse": np.sqrt(mse),
            "mae": (guess - actual).abs().mean(),
            "improvement_pct": (1 - mse / naive_mse) * 100,
            "years_beating_naive": int(by_year["beat_naive"].sum()),
            "years_total": len(by_year),
            "regime_accuracy": (wild_guess == wild_actual).mean() * 100,
            "wild_caught": caught * 100,
        })

scores = pd.DataFrame(score_rows)


def newey_west_variance(values, lags):
    centred = values - values.mean()
    count = len(centred)
    total = np.sum(centred * centred) / count
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        total += 2 * weight * np.sum(centred[lag:] * centred[:-lag]) / count
    return total


def block_bootstrap_cut(naive_error, model_error, random):
    count = len(naive_error)
    starts = count - BLOCK + 1
    pieces = int(np.ceil(count / BLOCK))
    cuts = []
    for _ in range(BOOTSTRAPS):
        picks = random.integers(0, starts, pieces)
        index = np.concatenate([np.arange(p, p + BLOCK) for p in picks])[:count]
        cuts.append((1 - np.mean(model_error[index]) / np.mean(naive_error[index])) * 100)
    return np.percentile(cuts, 5), np.percentile(cuts, 95)


random = np.random.default_rng(0)
significance_rows = []
for asset in ASSETS:
    known = predictions[asset + "_actual"].notna()
    actual = predictions.loc[known, asset + "_actual"].values
    naive_error = (predictions.loc[known, asset + "_naive"].values - actual) ** 2
    for name in MODEL_NAMES:
        if name == "naive":
            continue
        model_error = (predictions.loc[known, asset + "_" + name].values - actual) ** 2
        gap = naive_error - model_error
        variance = max(newey_west_variance(gap, HAC_LAGS), 1e-12)
        statistic = gap.mean() / np.sqrt(variance / len(gap))
        low, high = block_bootstrap_cut(naive_error, model_error, random)
        significance_rows.append({
            "asset": asset,
            "model": name,
            "dm_stat": statistic,
            "p_value": 1 - stats.norm.cdf(statistic),
            "p_adjusted": min(1.0, (1 - stats.norm.cdf(statistic)) * (len(MODEL_NAMES) - 1)),
            "cut_low": low,
            "cut_high": high,
        })
significance = pd.DataFrame(significance_rows)

alert_rows = []
for asset in ASSETS:
    checks = [c for c in alert_checks if c["asset"] == asset]
    wild_actual = np.concatenate([c["wild_actual"] for c in checks])
    for method in ["naive", "alert"]:
        guess = np.concatenate([c[method] for c in checks])
        hits = (guess & wild_actual).sum()
        alert_rows.append({
            "asset": asset,
            "method": method,
            "caught": hits / max(wild_actual.sum(), 1) * 100,
            "false_alarm": (guess & ~wild_actual).sum() / max((~wild_actual).sum(), 1) * 100,
            "precision": hits / max(guess.sum(), 1) * 100,
            "months_flagged": guess.mean() * 100,
        })
alerts = pd.DataFrame(alert_rows)

latest_rows = []
for asset in ASSETS:
    target = asset + "_fwdvol"
    known_history = data[target].dropna()
    train_full = data.dropna(subset=[target])
    latest = data.iloc[[-1]]
    predictions_now = fit_predict(train_full, latest, asset)
    asset_scores = scores[scores["asset"] == asset].set_index("model")
    best = asset_scores["rmse"].idxmin()
    forecast = float(predictions_now[best][0])
    low_cut = known_history.quantile(1 / 3)
    high_cut = known_history.quantile(2 / 3)
    if forecast < low_cut:
        label = "Calmer than usual"
    elif forecast > high_cut:
        label = "Wilder than usual"
    else:
        label = "Usual"
    row = {
        "asset": asset,
        "date": latest.index[0].date(),
        "current_vol": float(latest[asset + "_vol21"].iloc[0]),
        "best_model": best,
        "forecast_vol": forecast,
        "low_cut": low_cut,
        "high_cut": high_cut,
        "label": label,
        "history_rank": float((known_history <= forecast).mean() * 100),
        "wild_prob": float(predictions_now["wildprob"][0] * 100),
    }
    for name in MODEL_NAMES:
        row[name] = float(predictions_now[name][0])
    latest_rows.append(row)

latest_table = pd.DataFrame(latest_rows)

scores.to_csv("data/forecast_scores.csv", index=False)
significance.to_csv("data/forecast_significance.csv", index=False)
alerts.to_csv("data/forecast_alerts.csv", index=False)
yearly.to_csv("data/forecast_yearly.csv", index=False)
predictions.to_csv("data/forecast_predictions.csv")
latest_table.to_csv("data/forecast_latest.csv", index=False)

print()
print("test years:", test_years[0], "to", test_years[-1], "(", len(test_years), "years )")
print("each year is forecast by models trained only on earlier years")
print("score = how much smaller the squared error is than the naive rule 'next month = this month'")
print()
print(scores.round(2).to_string(index=False))
print()
print("average improvement over naive by model (%):")
print(scores.groupby("model")["improvement_pct"].mean().reindex(MODEL_NAMES).round(1).to_string())
print()
print("is the improvement real? (Diebold-Mariano test, one-sided; bootstrap range of the error cut, 90%)")
print(significance.round(3).to_string(index=False))
print()
print("wild-month alert versus the naive rule (wild = above the 75th percentile of past risk)")
print(alerts.round(1).to_string(index=False))
print()
print("latest forecast:")
print(latest_table.round(2).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, asset in zip(axes.flatten(), ASSETS):
    best = latest_table.set_index("asset").loc[asset, "best_model"]
    ax.plot(predictions[asset + "_actual"], label="actual next-month volatility", color="black", linewidth=1.5)
    ax.plot(predictions[asset + "_" + best], label="forecast (" + best + ")", color="tab:red")
    ax.plot(predictions[asset + "_naive"], label="naive", color="tab:gray", alpha=0.6)
    ax.set_title(asset)
    ax.set_ylabel("annual volatility (%)")
    ax.legend()
plt.tight_layout()
plt.savefig("data/forecast.png")
