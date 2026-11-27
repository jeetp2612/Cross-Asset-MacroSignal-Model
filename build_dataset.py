import os
import sys

import pandas as pd
import yfinance as yf

from macro import align_release

START = "2010-01-01"
MACRO_START = "2009-01-01"
HORIZON = 21

os.makedirs("data", exist_ok=True)

tickers = {
    "SPY": "SPY",
    "TLT": "TLT",
    "GLD": "GLD",
    "UUP": "UUP",
    "VIX": "^VIX",
    "OIL": "CL=F",
}
optional_tickers = {"HYG": "HYG"}

macro_codes = {
    "INFL_EXPECT": ("T10YIE", 0),
    "FIN_COND": ("NFCI", 5),
    "CLAIMS": ("ICSA", 5),
    "UNRATE": ("UNRATE", 40),
    "CPI": ("CPIAUCSL", 45),
}


def save(frame, name):
    path = "data/" + name
    temporary = path + ".tmp"
    frame.to_csv(temporary)
    os.replace(temporary, path)


try:
    prices = yf.download(list(tickers.values()) + list(optional_tickers.values()), start=START, auto_adjust=True, progress=False)["Close"]
except (OSError, ValueError, KeyError, RuntimeError) as error:
    sys.exit("Could not download prices from Yahoo Finance: " + str(error))

prices = prices.rename(columns={v: k for k, v in {**tickers, **optional_tickers}.items()})
for name in optional_tickers:
    if name not in prices.columns or prices[name].isna().all():
        print("Skipping optional series", name)
        prices = prices.drop(columns=[name], errors="ignore")
missing = [name for name in tickers if name not in prices.columns]
if prices.empty or missing:
    sys.exit("Yahoo Finance returned no data for: " + ", ".join(missing or list(tickers)) + ". Check your internet connection and try again.")
prices.index = pd.to_datetime(prices.index).tz_localize(None)

fred_codes = {
    "US10Y": "DGS10",
    "US2Y": "DGS2",
    "CURVE_10Y_2Y": "T10Y2Y",
    "FED_FUNDS": "DFF",
}

fred = pd.DataFrame()
for name, code in fred_codes.items():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + code + "&cosd=" + START
    try:
        series = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        sys.exit("Could not download " + code + " from FRED: " + str(error))
    fred[name] = pd.to_numeric(series.iloc[:, 0], errors="coerce")

data = prices.join(fred, how="left")
data = data.sort_index()
data = data.ffill(limit=5)
data = data.dropna()
data.index.name = "date"

for name, (code, lag_days) in macro_codes.items():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + code + "&cosd=" + MACRO_START
    try:
        series = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Skipping optional series", name, "(" + code + "):", error)
        continue
    data[name] = align_release(series.iloc[:, 0], lag_days, data.index)

data = data.dropna()

if len(data) < 1000:
    sys.exit("The downloaded data has only " + str(len(data)) + " rows, which looks incomplete. Your old files were kept. Try again later.")

save(data, "market.csv")

for asset in ["SPY", "TLT", "GLD", "UUP"]:
    future_price = data[asset].shift(-HORIZON)
    data[asset + "_up"] = (future_price > data[asset]).astype(int)

data = data.iloc[:-HORIZON]

save(data, "dataset.csv")

print("rows:", len(data))
print("columns:", list(data.columns))
print("from", data.index.min().date(), "to", data.index.max().date())
print(data.tail())
print(data.isna().sum())
