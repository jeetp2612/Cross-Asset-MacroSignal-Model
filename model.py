import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/features.csv", index_col=0, parse_dates=True)

assets = ["SPY", "TLT", "GLD", "UUP"]
label_columns = [a + "_up" for a in assets]
signal_columns = [c for c in data.columns if c not in label_columns]

GAP = 21

train = data[data.index < "2021-01-01"].iloc[:-GAP]
test = data[data.index >= "2021-01-01"]

models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20, random_state=0),
    "gradient_boost": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0),
}

rows = []
importances = {}
predictions = pd.DataFrame(index=test.index)

for asset in assets:
    y_train = train[asset + "_up"]
    y_test = test[asset + "_up"]

    majority = 1 if y_train.mean() >= 0.5 else 0
    baseline = (y_test == majority).mean()

    for name, model in models.items():
        model.fit(train[signal_columns], y_train)
        prob = model.predict_proba(test[signal_columns])[:, 1]
        guess = (prob >= 0.5).astype(int)

        rows.append({
            "asset": asset,
            "model": name,
            "accuracy": accuracy_score(y_test, guess),
            "baseline": baseline,
            "auc": roc_auc_score(y_test, prob),
        })

        predictions[asset + "_" + name] = prob

        if name == "random_forest":
            importances[asset] = pd.Series(model.feature_importances_, index=signal_columns)

results = pd.DataFrame(rows)

print("train rows:", len(train), " test rows:", len(test))
print("train until", train.index.max().date(), " test from", test.index.min().date())
print()
print(results.round(3).to_string(index=False))

for asset in assets:
    print()
    print("top 5 signals for", asset, "(random forest):")
    print(importances[asset].sort_values(ascending=False).head(5).round(3).to_string())

results.to_csv("data/results.csv", index=False)
predictions.to_csv("data/predictions.csv")
