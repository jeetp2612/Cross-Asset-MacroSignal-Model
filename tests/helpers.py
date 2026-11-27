import numpy as np
import pandas as pd


def simulate(rows=4250, seed=3):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2010-01-04", periods=rows)
    frame = pd.DataFrame(index=index)
    shocks = rng.normal(size=(rows, 4))
    sigmas = {}
    for k, (name, base, drift) in enumerate([("SPY", 0.009, 0.0005), ("TLT", 0.008, 0.0001), ("GLD", 0.009, 0.0003), ("UUP", 0.004, 0.0)]):
        sigma = np.zeros(rows)
        returns = np.zeros(rows)
        sigma[0] = base
        for t in range(1, rows):
            variance = (0.02 * base**2) + 0.08 * returns[t - 1] ** 2 + 0.9 * sigma[t - 1] ** 2
            sigma[t] = np.sqrt(variance)
            returns[t] = drift + sigma[t] * shocks[t, k]
        sigmas[name] = sigma
        frame[name] = 100 * np.cumprod(1 + returns)
    frame["OIL"] = 60 * np.exp(np.cumsum(rng.normal(0, 0.02, rows)))
    frame["VIX"] = np.maximum(sigmas["SPY"] * np.sqrt(252) * 100 * 1.15 + rng.normal(0, 1.5, rows), 9)
    frame["US10Y"] = 2.5 + np.cumsum(rng.normal(0, 0.03, rows)) * 0.3
    frame["US2Y"] = frame["US10Y"] - 0.8 + np.cumsum(rng.normal(0, 0.02, rows)) * 0.2
    frame["CURVE_10Y_2Y"] = frame["US10Y"] - frame["US2Y"]
    frame["FED_FUNDS"] = np.clip(frame["US2Y"] - 0.2, 0.05, None)
    frame["HYG"] = 80 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, rows)))
    frame["INFL_EXPECT"] = 2 + np.cumsum(rng.normal(0, 0.01, rows)) * 0.5
    frame["FIN_COND"] = -0.3 + sigmas["SPY"] * 40 + np.cumsum(rng.normal(0, 0.01, rows)) * 0.1
    frame["CLAIMS"] = 250000 + np.cumsum(rng.normal(0, 2000, rows))
    frame["UNRATE"] = np.clip(5 + np.cumsum(rng.normal(0, 0.01, rows)), 3, 10)
    frame["CPI"] = 220 * np.cumprod(1 + np.full(rows, 0.0001) + rng.normal(0, 0.00005, rows))
    frame.index.name = "date"
    return frame
