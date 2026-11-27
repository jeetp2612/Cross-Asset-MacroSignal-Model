import matplotlib.pyplot as plt
import pandas as pd

COST = 0.001

prices = pd.read_csv("data/dataset.csv", index_col=0, parse_dates=True)
predictions = pd.read_csv("data/predictions.csv", index_col=0, parse_dates=True)

assets = ["SPY", "TLT", "GLD", "UUP"]
models = ["logistic", "random_forest", "gradient_boost"]

daily_returns = prices[assets].pct_change().loc[predictions.index]


def total_return(r):
    return (1 + r).prod() - 1


def sharpe(r):
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * (252 ** 0.5)


def max_drawdown(r):
    curve = (1 + r).cumprod()
    return (curve / curve.cummax() - 1).min()


rows = []
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, asset in zip(axes.flatten(), assets):
    hold = daily_returns[asset]
    ax.plot((1 + hold).cumprod(), label="buy and hold", color="black", linewidth=2)

    rows.append({
        "asset": asset,
        "strategy": "buy and hold",
        "total_return": total_return(hold),
        "sharpe": sharpe(hold),
        "max_drawdown": max_drawdown(hold),
        "time_invested": 1.0,
    })

    for model in models:
        position = (predictions[asset + "_" + model] >= 0.5).astype(int).shift(1).fillna(0)
        trades = position.diff().abs().fillna(0)
        strategy = position * hold - trades * COST

        ax.plot((1 + strategy).cumprod(), label=model)

        rows.append({
            "asset": asset,
            "strategy": model,
            "total_return": total_return(strategy),
            "sharpe": sharpe(strategy),
            "max_drawdown": max_drawdown(strategy),
            "time_invested": position.mean(),
        })

    ax.set_title(asset)
    ax.legend()

plt.tight_layout()
plt.savefig("data/backtest.png")

results = pd.DataFrame(rows)
results.to_csv("data/backtest_results.csv", index=False)

print("trading cost per trade:", COST)
print("test period:", predictions.index.min().date(), "to", predictions.index.max().date())
print()
print(results.round(3).to_string(index=False))
