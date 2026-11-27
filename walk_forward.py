import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COST = 0.001
GAP = 21
FIRST_YEAR = 2015

data = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)
prices = pd.read_csv("data/dataset.csv", index_col=0, parse_dates=True)

assets = ["SPY", "TLT", "GLD", "UUP"]
model_names = ["logistic", "random_forest", "gradient_boost"]
label_columns = [a + "_up" for a in assets]
signal_columns = [c for c in data.columns if c not in label_columns]


def make_models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=0),
        "gradient_boost": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0),
    }


test_years = [y for y in sorted(set(data.index.year)) if y >= FIRST_YEAR]

records = []
predictions = pd.DataFrame()

for year in test_years:
    train = data[data.index < str(year) + "-01-01"].iloc[:-GAP]
    test = data[data.index.year == year]
    block = pd.DataFrame(index=test.index)

    for asset in assets:
        y_train = train[asset + "_up"]
        y_test = test[asset + "_up"]

        majority = 1 if y_train.mean() >= 0.5 else 0
        baseline = (y_test == majority).mean()

        models = make_models()
        for name in model_names:
            model = models[name]
            model.fit(train[signal_columns], y_train)
            prob = model.predict_proba(test[signal_columns])[:, 1]
            guess = (prob >= 0.5).astype(int)

            if y_test.nunique() < 2:
                auc = float("nan")
            else:
                auc = roc_auc_score(y_test, prob)

            records.append({
                "year": year,
                "asset": asset,
                "model": name,
                "accuracy": (guess == y_test.values).mean(),
                "baseline": baseline,
                "auc": auc,
            })
            block[asset + "_" + name] = prob

    predictions = pd.concat([predictions, block])
    print("finished year", year)

yearly_scores = pd.DataFrame(records)

daily_returns = prices[assets].pct_change().loc[predictions.index]
by_year = daily_returns.index.year


def total_return(r):
    return (1 + r).prod() - 1


def sharpe(r):
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * (252 ** 0.5)


summary_rows = []
yearly_tables = {}
curves = {}

for asset in assets:
    hold = daily_returns[asset]
    hold_by_year = (1 + hold).groupby(by_year).prod() - 1
    curves[(asset, "buy and hold")] = (1 + hold).cumprod()

    for name in model_names:
        position = (predictions[asset + "_" + name] >= 0.5).astype(int).shift(1).fillna(0)
        trades = position.diff().abs().fillna(0)
        strategy = position * hold - trades * COST
        curves[(asset, name)] = (1 + strategy).cumprod()

        strategy_by_year = (1 + strategy).groupby(by_year).prod() - 1

        scores = yearly_scores[(yearly_scores["asset"] == asset) & (yearly_scores["model"] == name)].set_index("year")

        table = pd.DataFrame({
            "auc": scores["auc"],
            "accuracy": scores["accuracy"],
            "baseline": scores["baseline"],
            "strategy_return": strategy_by_year,
            "buy_hold_return": hold_by_year,
        })
        yearly_tables[(asset, name)] = table

        summary_rows.append({
            "asset": asset,
            "model": name,
            "avg_auc": scores["auc"].mean(),
            "years_auc_above_0.5": int((scores["auc"] > 0.5).sum()),
            "years_acc_above_baseline": int((scores["accuracy"] > scores["baseline"]).sum()),
            "years_beating_buy_hold": int((strategy_by_year > hold_by_year).sum()),
            "strategy_return": total_return(strategy),
            "buy_hold_return": total_return(hold),
            "strategy_sharpe": sharpe(strategy),
            "buy_hold_sharpe": sharpe(hold),
        })

summary = pd.DataFrame(summary_rows)

print()
print("test years:", test_years[0], "to", test_years[-1], "(", len(test_years), "years )")
print("each year is predicted by a model trained only on earlier years")
print()
print(summary.round(3).to_string(index=False))

for asset, name in [("GLD", "logistic"), ("SPY", "gradient_boost")]:
    print()
    print("year by year:", asset, name)
    print(yearly_tables[(asset, name)].round(3).to_string())

summary.to_csv("data/walk_forward_summary.csv", index=False)
yearly_scores.to_csv("data/walk_forward_yearly_scores.csv", index=False)
predictions.to_csv("data/walk_forward_predictions.csv")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, asset in zip(axes.flatten(), assets):
    ax.plot(curves[(asset, "buy and hold")], label="buy and hold", color="black", linewidth=2)
    for name in model_names:
        ax.plot(curves[(asset, name)], label=name)
    ax.set_title(asset)
    ax.legend()
plt.tight_layout()
plt.savefig("data/walk_forward.png")
