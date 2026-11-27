import os

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

__version__ = "2.0.0"
FRESH_DAYS = 5

FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(FOLDER, "data")

ASSETS = ["SPY", "TLT", "GLD", "UUP"]
SHORT = {"SPY": "Stocks", "TLT": "Bonds", "GLD": "Gold", "UUP": "Dollar"}
FULL = {"SPY": "Stocks (SPY)", "TLT": "Bonds (TLT)", "GLD": "Gold (GLD)", "UUP": "US dollar (UUP)"}
ASSET_COLORS = {"SPY": "#2f6fb0", "TLT": "#2e8b57", "GLD": "#c99a06", "UUP": "#7a5ea8"}
MIX_COLOR = "#c0392b"

MODEL_LABELS = {
    "naive": "Naive: next month = this month",
    "har": "HAR: memory of recent risk",
    "harx": "HAR + fear and credit signals",
    "boosted": "Boosted trees on all signals",
    "blend": "Blend of three models",
}
MODEL_SHORT = {
    "naive": "naive",
    "har": "HAR",
    "harx": "HAR plus signals",
    "boosted": "boosted-trees",
    "blend": "blended",
}
DIRECTION_LABELS = {
    "logistic": "Simple model",
    "random_forest": "Random forest",
    "gradient_boost": "Boosted trees",
}
VIEWS = ["Growth", "Drawdown", "Risk"]
VIEW_LABELS = {
    "Growth": "Value of 100 invested",
    "Drawdown": "Fall from peak (%)",
    "Risk": "Yearly swing (%)",
}

PERIODS = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "All": None}
DEFAULT_COMPARE = ["SPY", "TLT", "GLD"]

PRESETS = {
    "All stocks": [100, 0, 0, 0],
    "Classic 60/40": [60, 40, 0, 0],
    "Stocks + gold": [70, 0, 30, 0],
    "Equal mix": [25, 25, 25, 25],
}

PAGE_BG = "#f3f5f8"
CARD_BG = "#ffffff"
BORDER = "#d5dbe3"
DARK = "#17324d"
ACCENT = "#2f6fb0"
GREY = "#5b6673"
INSIGHT_BG = "#fff7e0"
GREEN = "#2e8b57"
BLUE = "#2f6fb0"
AMBER = "#d98200"
RED = "#c0392b"


def load_file(name, has_index):
    path = os.path.join(DATA_FOLDER, name)
    if not os.path.exists(path):
        return None
    if has_index:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return pd.read_csv(path)


def load_market():
    market = load_file("market.csv", True)
    if market is not None:
        return market
    return load_file("dataset.csv", True)


def data_age_text(frame, today=None):
    if today is None:
        today = pd.Timestamp.today().normalize()
    last = frame.index[-1]
    days = (today - last).days
    if days <= FRESH_DAYS:
        return f"Data up to {last.date()} (up to date)", True
    return f"Data up to {last.date()} ({days} days old)", False


def mood_of(vix):
    if vix < 15:
        return "Calm", GREEN
    if vix < 25:
        return "Normal", BLUE
    if vix < 30:
        return "Nervous", AMBER
    return "Panic", RED


def link_of(value):
    if value >= 0.2:
        return "Moving together", AMBER
    if value <= -0.2:
        return "Moving opposite", GREEN
    return "Little link", BLUE


def risk_of_swing(swing):
    if swing < 10:
        return "Low", GREEN
    if swing < 20:
        return "Medium", AMBER
    return "High", RED


def market_snapshot(frame):
    vix = frame["VIX"].iloc[-1]
    rank = (frame["VIX"] <= vix).mean() * 100
    curve = frame["CURVE_10Y_2Y"].iloc[-1]
    moves = frame[["SPY", "TLT"]].pct_change()
    link = moves["SPY"].rolling(252).corr(moves["TLT"]).iloc[-1]
    if pd.isna(link):
        link = moves["SPY"].corr(moves["TLT"])
    back = min(252, len(frame) - 1)
    changes = (frame[ASSETS].iloc[-1] / frame[ASSETS].iloc[-1 - back] - 1) * 100
    return {
        "date": frame.index[-1].date(),
        "first_year": frame.index[0].year,
        "vix": vix,
        "rank": rank,
        "curve": curve,
        "link": link,
        "changes": changes,
        "best": changes.idxmax(),
    }


def dashboard_message(shot):
    mood, color = mood_of(shot["vix"])
    if mood == "Calm":
        headline = "Markets are calm"
        text = "Fear is low, so daily price swings are small right now."
    elif mood == "Normal":
        headline = "Markets look normal"
        text = "Fear is at a typical level. Expect ordinary ups and downs."
    elif mood == "Nervous":
        headline = "Investors are nervous"
        text = "Fear is high, so expect bigger daily swings. Mixing assets that move differently can cushion them."
    else:
        headline = "Markets are in panic"
        text = "Fear is extreme. In past years, days like this brought the biggest daily swings."
    if shot["curve"] < 0:
        text += " The yield curve is inverted, an old warning light for slowdowns (not a prediction)."
    return headline, text, color


def period_slice(frame, label):
    years = PERIODS[label]
    if years is None:
        return frame
    start = frame.index[-1] - pd.DateOffset(years=years)
    return frame.loc[start:]


def rebased(frame, codes):
    part = frame[codes]
    return part / part.iloc[0] * 100


def market_facts(frame, code):
    series = frame[code]
    daily = series.pct_change().dropna()
    return {
        "total": (series.iloc[-1] / series.iloc[0] - 1) * 100,
        "worst": (series / series.cummax() - 1).min() * 100,
        "swing": daily.std() * np.sqrt(252) * 100,
    }


def nearest_row(frame, x_value):
    stamp = pd.Timestamp(mdates.num2date(x_value)).tz_localize(None)
    position = frame.index.get_indexer([stamp], method="nearest")[0]
    return frame.index[position], frame.iloc[position]


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e3e7ec", linewidth=0.8)
    ax.set_axisbelow(True)


def chart_frame(frame, codes, view):
    part = frame[codes]
    if view == "Drawdown":
        return (part / part.cummax() - 1) * 100
    if view == "Risk":
        return (part.pct_change().rolling(21).std() * np.sqrt(252) * 100).dropna()
    return part / part.iloc[0] * 100


def hover_values(row, view):
    if view == "Growth":
        return "    ".join(f"{SHORT[code]} {row[code]:.0f}" for code in row.index)
    return "    ".join(f"{SHORT[code]} {row[code]:.1f}%" for code in row.index)


def draw_compare(fig, series_frame, codes, view="Growth"):
    fig.clear()
    ax = fig.add_subplot(111)
    style_axes(ax)
    if not codes:
        ax.text(0.5, 0.5, "Tick at least one market above", ha="center", va="center", transform=ax.transAxes, color=GREY)
        return ax
    for code in codes:
        ax.plot(series_frame[code], color=ASSET_COLORS[code], label=SHORT[code], linewidth=1.8)
    if view == "Growth":
        ax.axhline(100, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_ylabel(VIEW_LABELS[view])
    ax.legend(loc="best", frameon=False)
    return ax


def parse_amount(text):
    try:
        amount = float(text.replace("$", "").replace(",", "").strip())
    except ValueError:
        raise ValueError("Please type the amount as a number, like 10000") from None
    if not 0 < amount < 1e12:
        raise ValueError("The amount must be above zero")
    return amount


def normalise(values):
    total = sum(values)
    if total <= 0:
        raise ValueError("Move at least one slider above zero")
    return [value / total for value in values]


def replay_history(frame, start_year, weights):
    returns = frame.loc[str(start_year) + "-01-01":, ASSETS].pct_change().dropna()
    if len(returns) < 60:
        raise ValueError("Choose an earlier start year")
    # Each day's mix return is the four asset returns multiplied by their weights.
    mix = pd.Series(returns.values @ np.array(weights), index=returns.index)
    mix_curve = (1 + mix).cumprod()
    stock_curve = (1 + returns["SPY"]).cumprod()
    return mix_curve, stock_curve


def curve_stats(curve):
    final = curve.iloc[-1]
    years = len(curve) / 252
    daily = curve.pct_change().dropna()
    return {
        "final": final,
        "total": (final - 1) * 100,
        "yearly": (final ** (1 / years) - 1) * 100,
        "worst": (curve / curve.cummax() - 1).min() * 100,
        "swing": daily.std() * np.sqrt(252) * 100,
    }


def draw_whatif(fig, mix_curve, stock_curve, amount, year):
    fig.clear()
    ax = fig.add_subplot(111)
    style_axes(ax)
    ax.plot(mix_curve * amount, label="Your mix", linewidth=2, color=MIX_COLOR)
    ax.plot(stock_curve * amount, label="100% stocks", color="#7f8c99")
    ax.axhline(amount, color="black", linestyle="--", linewidth=0.8, label="money you started with")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"${value:,.0f}"))
    ax.set_title(f"If you had put ${amount:,.0f} in at the start of {year}")
    ax.legend(loc="upper left", frameon=False)


def whatif_text(mix_curve, stock_curve, weights):
    mix = curve_stats(mix_curve)
    stocks = curve_stats(stock_curve)
    if weights[0] > 0.999:
        return "Your mix is the same as 100% stocks. Move the sliders or pick another quick mix to compare."
    if mix["worst"] > stocks["worst"] + 3:
        if mix["final"] < stocks["final"]:
            return "Your mix cushioned the falls but gave up some growth. That is the usual trade-off."
        return "Your mix cushioned the falls and also grew as much. That was lucky in this period, so do not expect it every time."
    if mix["worst"] < stocks["worst"] - 3:
        return "Your mix fell harder than stocks in this period."
    return "Your mix behaved a lot like stocks."


def load_forecast():
    scores = load_file("forecast_scores.csv", False)
    latest = load_file("forecast_latest.csv", False)
    predictions = load_file("forecast_predictions.csv", True)
    if scores is None or latest is None or predictions is None:
        return None
    return {
        "scores": scores,
        "latest": latest.set_index("asset"),
        "predictions": predictions,
        "significance": load_file("forecast_significance.csv", False),
        "alerts": load_file("forecast_alerts.csv", False),
    }


def load_backtest():
    results = load_file("vol_target_results.csv", False)
    curves = load_file("vol_target_curves.csv", True)
    if results is None or curves is None:
        return None
    return {"results": results, "curves": curves}


def proof_of(p_value):
    if p_value < 0.01:
        return "Yes, strong", GREEN
    if p_value < 0.05:
        return "Yes", GREEN
    if p_value < 0.10:
        return "Maybe", AMBER
    return "Not proven", RED


def label_color(label):
    if label.startswith("Calmer"):
        return GREEN
    if label.startswith("Wilder"):
        return RED
    return BLUE


def percent_change(series, days):
    back = min(days, len(series) - 1)
    return (series.iloc[-1] / series.iloc[-1 - back] - 1) * 100


def asset_table(frame, forecast=None):
    rows = []
    for code in ASSETS:
        series = frame[code]
        last = series.iloc[-1]

        daily = series.pct_change().dropna()
        row = {
            "asset": code,
            "price": last,
            "change_1m": percent_change(series, 21),
            "change_3m": percent_change(series, 63),
            "change_12m": percent_change(series, 252),
            "vol_now": daily.tail(21).std() * np.sqrt(252) * 100,
            "from_peak": (last / series.tail(252).max() - 1) * 100,
            "forecast": None,
            "label": None,
        }
        if forecast is not None and code in forecast["latest"].index:
            row["forecast"] = forecast["latest"].loc[code, "forecast_vol"]
            row["label"] = forecast["latest"].loc[code, "label"]
        rows.append(row)
    return rows


def corr_table(frame):
    moves = frame[ASSETS].pct_change().dropna()
    table = moves.corr()
    table.index = [SHORT[c] for c in table.index]
    table.columns = [SHORT[c] for c in table.columns]
    return table


def markets_text(frame):
    table = corr_table(frame)
    names = list(table.columns)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], table.iloc[i, j]))
    strongest = max(pairs, key=lambda item: abs(item[2]))
    stock_pairs = [pair for pair in pairs if pair[0] == "Stocks"]
    partner = min(stock_pairs, key=lambda item: item[2])
    if abs(strongest[2]) < 0.2:
        text = f"No pair moved closely together here (the strongest link was {strongest[0]} and {strongest[1]}, {strongest[2]:+.2f}). "
    else:
        text = f"The strongest link is {strongest[0]} and {strongest[1]} ({strongest[2]:+.2f}). "
    if partner[2] <= -0.2:
        text += f"The best cushion for Stocks was {partner[1]} (link {partner[2]:+.2f}): it often rose when Stocks fell."
    elif partner[2] < 0.2:
        text += f"The most independent partner for Stocks was {partner[1]} (link {partner[2]:+.2f}), so it was not pulled along by Stocks."
    else:
        text += f"No asset gave much cushion for Stocks here (the lowest link was {partner[1]}, {partner[2]:+.2f})."
    return text


def draw_markets(fig, frame):
    fig.clear()
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.7]})
    table = corr_table(frame)
    names = list(table.columns)
    image = left.imshow(table.values, cmap="coolwarm", vmin=-1, vmax=1)
    left.set_xticks(range(len(names)))
    left.set_xticklabels(names, rotation=30, ha="right")
    left.set_yticks(range(len(names)))
    left.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            left.text(j, i, f"{table.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=left, shrink=0.8)
    left.set_title("Who moves with whom (daily moves)")
    style_axes(right)
    moves = frame[ASSETS].pct_change().dropna()
    window = 252 if len(moves) >= 756 else 63
    label = "1-year" if window == 252 else "3-month"
    drawn = False
    for code in ASSETS[1:]:
        rolling = moves["SPY"].rolling(window).corr(moves[code]).dropna()
        if rolling.empty:
            continue
        right.plot(rolling, color=ASSET_COLORS[code], label=f"Stocks and {SHORT[code]}", linewidth=1.6)
        drawn = True
    if drawn:
        right.axhline(0, color="black", linewidth=0.8)
        right.set_ylim(-1, 1)
        right.legend(loc="best", frameon=False)
    else:
        right.text(0.5, 0.5, "Choose a longer period", ha="center", va="center", transform=right.transAxes, color=GREY)
    right.set_title(f"How the link with Stocks changed ({label} window)")
    right.set_ylabel("link (+1 together, -1 opposite)")


def forecast_summary(forecast, asset):
    latest = forecast["latest"].loc[asset]
    scores = forecast["scores"]
    part = scores[scores["asset"] == asset].set_index("model")
    best = latest["best_model"]
    extra = {"p_adjusted": None, "p_value": None, "cut_low": None, "cut_high": None, "wild_prob": None, "alert_caught": None, "naive_caught": None, "false_alarm": None, "naive_false_alarm": None}
    significance = forecast.get("significance")
    if significance is not None:
        found = significance[(significance["asset"] == asset) & (significance["model"] == best)]
        if not found.empty:
            extra["p_value"] = float(found.iloc[0]["p_value"])
            extra["p_adjusted"] = float(found.iloc[0]["p_adjusted"]) if "p_adjusted" in found.columns else extra["p_value"]
            extra["cut_low"] = float(found.iloc[0]["cut_low"])
            extra["cut_high"] = float(found.iloc[0]["cut_high"])
    if "wild_prob" in forecast["latest"].columns:
        extra["wild_prob"] = float(latest["wild_prob"])
    alerts = forecast.get("alerts")
    if alerts is not None:
        mine = alerts[alerts["asset"] == asset].set_index("method")
        if "alert" in mine.index and "naive" in mine.index:
            extra["alert_caught"] = float(mine.loc["alert", "caught"])
            extra["naive_caught"] = float(mine.loc["naive", "caught"])
            extra["false_alarm"] = float(mine.loc["alert", "false_alarm"])
            extra["naive_false_alarm"] = float(mine.loc["naive", "false_alarm"])
    return {
        **extra,
        "current": latest["current_vol"],
        "forecast": latest["forecast_vol"],
        "low_cut": latest["low_cut"],
        "high_cut": latest["high_cut"],
        "label": latest["label"],
        "rank": latest["history_rank"],
        "best": best,
        "improvement": part.loc[best, "improvement_pct"],
        "years_beating": int(part.loc[best, "years_beating_naive"]),
        "years_total": int(part.loc[best, "years_total"]),
        "wild_caught": part.loc[best, "wild_caught"],
        "accuracy": part.loc[best, "regime_accuracy"],
        "naive_accuracy": part.loc["naive", "regime_accuracy"],
        "date": latest["date"],
    }


def forecast_text(forecast, asset):
    info = forecast_summary(forecast, asset)
    name = SHORT[asset]
    text = (
        f"For {name}, the {MODEL_SHORT[info['best']]} model expects a yearly swing of about {info['forecast']:.1f}% over the next month, "
        f"against {info['current']:.1f}% over the last month. The usual range is {info['low_cut']:.1f}% to {info['high_cut']:.1f}%, so risk looks "
        f"{info['label'].lower()}."
    )
    if info["best"] == "naive":
        return text + " No model beat the simple rule 'next month will look like this month' in the test."
    text += (
        f" In the year-by-year test it cut the forecast error by {info['improvement']:.0f}% compared with the simple rule, "
        f"and beat it in {info['years_beating']} of {info['years_total']} years."
    )
    if info["p_value"] is not None:
        word = proof_of(info["p_value"])[0]
        if word in ("Yes, strong", "Yes"):
            text += f" A statistical test says this is unlikely to be luck (p = {info['p_value']:.3f})."
            if info["p_adjusted"] >= 0.05:
                text += f" Allowing for the fact that four models were tried, the evidence is weaker (p = {info['p_adjusted']:.2f})."
        elif word == "Maybe":
            text += f" The statistical test is borderline (p = {info['p_value']:.3f}), so treat the gain with care."
        else:
            text += f" The statistical test could not rule out luck (p = {info['p_value']:.2f}), so treat the gain with care."
    return text


def proof_for(forecast, asset, model):
    significance = forecast.get("significance")
    if significance is None:
        return None
    found = significance[(significance["asset"] == asset) & (significance["model"] == model)]
    if found.empty:
        return None
    return float(found.iloc[0]["p_value"])


def scoreboard_rows(forecast, asset):
    scores = forecast["scores"]
    part = scores[scores["asset"] == asset].sort_values("rmse")
    rows = []
    for _, row in part.iterrows():
        if row["model"] == "naive":
            cut, years, proof = "-", "-", "-"
        else:
            cut = f"{row['improvement_pct']:+.0f}%"
            years = f"{int(row['years_beating_naive'])} of {int(row['years_total'])}"
            p_value = proof_for(forecast, asset, row["model"])
            proof = "n/a" if p_value is None else proof_of(p_value)[0]
        rows.append((MODEL_LABELS[row["model"]], cut, years, proof))
    return rows


def alert_text(forecast, asset):
    info = forecast_summary(forecast, asset)
    if info["alert_caught"] is None:
        return ""
    gain = info["alert_caught"] - info["naive_caught"]
    text = (
        f"Wild-month alert: it caught {info['alert_caught']:.0f}% of the wildest months, against {info['naive_caught']:.0f}% for the naive rule, "
        f"at the price of {info['false_alarm']:.0f}% false alarms (naive: {info['naive_false_alarm']:.0f}%)."
    )
    if gain <= 0:
        text += " Here it did not catch more wild months than the naive rule."
    return text


def draw_forecast(fig, forecast, asset, model):
    fig.clear()
    ax = fig.add_subplot(111)
    style_axes(ax)
    if forecast is None:
        ax.text(0.5, 0.5, "Press Update data to build the risk forecast", ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()
        return
    if model == "best":
        model = forecast["latest"].loc[asset, "best_model"]
    predictions = forecast["predictions"]
    ax.plot(predictions[asset + "_actual"], color="black", linewidth=1.6, label="What actually happened")
    ax.plot(predictions[asset + "_naive"], color="#9aa5b1", linewidth=1.2, label="Naive rule")
    if model != "naive":
        ax.plot(predictions[asset + "_" + model], color=MIX_COLOR, linewidth=1.6, label=MODEL_LABELS[model])
    ax.set_ylabel("Yearly swing (%)")
    ax.set_title(f"{FULL[asset]}: forecast made each day vs what followed")
    ax.legend(loc="upper left", frameon=False)


def mix_forecast_vol(frame, weights, forecast):
    if forecast is None:
        return None
    vols = np.array([forecast["latest"].loc[code, "forecast_vol"] for code in ASSETS])
    moves = frame[ASSETS].pct_change().dropna().tail(252)
    corr = moves.corr().values
    # Portfolio variance is weights.T @ covariance @ weights.
    cov = np.outer(vols, vols) * corr
    weights = np.array(weights)
    return float(np.sqrt(weights @ cov @ weights))


def reality_rows(summary):
    rows = []
    if summary is None:
        return rows
    for asset in ASSETS:
        part = summary[summary["asset"] == asset]
        if part.empty:
            continue
        best = part.loc[part["avg_auc"].idxmax()]
        rows.append({
            "asset": asset,
            "model": best["model"],
            "auc": best["avg_auc"],
            "strategy": best["strategy_return"] * 100,
            "hold": best["buy_hold_return"] * 100,
        })
    return rows


def risk_rows(forecast):
    rows = []
    if forecast is None:
        return rows
    scores = forecast["scores"]
    for asset in ASSETS:
        part = scores[scores["asset"] == asset]
        if part.empty:
            continue
        best = part.loc[part["rmse"].idxmin()]
        rows.append({
            "asset": asset,
            "model": best["model"],
            "improvement": best["improvement_pct"],
            "years_beating": int(best["years_beating_naive"]),
            "years_total": int(best["years_total"]),
        })
    return rows


def reality_text(risk, direction):
    parts = []
    if risk:
        better = sum(1 for row in risk if row["improvement"] > 0)
        average = sum(row["improvement"] for row in risk) / len(risk)
        if better == 0:
            parts.append("Risk was not forecast better than the naive rule in this test, so treat the risk forecasts with care.")
        else:
            parts.append(
                f"Risk can be forecast: in {better} of {len(risk)} markets the best model cut the error of the naive rule, by {average:.0f}% on average."
            )
    if direction:
        count = len(direction)
        trailed = sum(1 for row in direction if row["strategy"] < row["hold"])
        if trailed == count:
            parts.append(f"Direction cannot be timed: in all {count} markets, trading on the best direction model earned less than simply buying and holding.")
        elif trailed == 0:
            parts.append(f"Direction models beat buying and holding in all {count} markets in this test, so check them over more years before trusting them.")
        else:
            parts.append(f"Direction is hard to time: in {trailed} of {count} markets, trading on the best direction model earned less than buying and holding.")
    if not parts:
        return "Press Update data to run the tests that fill this screen."
    parts.append("So use this dashboard to understand and plan for risk, not to time the market.")
    return " ".join(parts)


def draw_reality(fig, risk, direction):
    fig.clear()
    if not risk and not direction:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Press Update data to see the results here", ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()
        return
    count = 2 if risk and direction else 1
    axes = fig.subplots(1, count)
    axes = list(axes) if count == 2 else [axes]
    if risk:
        ax = axes.pop(0)
        style_axes(ax)
        positions = np.arange(len(risk))
        values = [row["improvement"] for row in risk]
        bars = ax.bar(positions, values, color=[GREEN if value > 0 else RED for value in values])
        ax.bar_label(bars, fmt="%+.0f%%", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{SHORT[row['asset']]}\n({MODEL_SHORT[row['model']]})" for row in risk], fontsize=8)
        ax.set_ylabel("Error cut (%)")
        ax.set_title("Forecasting RISK vs the naive rule")
    if direction:
        ax = axes.pop(0)
        style_axes(ax)
        positions = np.arange(len(direction))
        width = 0.38
        strategy = ax.bar(positions - width / 2, [row["strategy"] for row in direction], width, label="Best direction model", color=MIX_COLOR)
        hold = ax.bar(positions + width / 2, [row["hold"] for row in direction], width, label="Buy and hold", color="#7f8c99")
        ax.bar_label(strategy, fmt="%.0f%%", fontsize=8)
        ax.bar_label(hold, fmt="%.0f%%", fontsize=8)
        ax.set_xticks(positions)
        ax.set_xticklabels([SHORT[row["asset"]] for row in direction])
        ax.set_ylabel("Total return over the test years")
        ax.set_title("Predicting DIRECTION vs buy and hold")
        ax.legend(loc="upper right", frameon=False)


def backtest_row(backtest, asset, strategy):
    results = backtest["results"]
    found = results[(results["asset"] == asset) & (results["strategy"] == strategy)]
    return found.iloc[0]


def backtest_summary(backtest, asset):
    hold = backtest_row(backtest, asset, "buy_and_hold")
    used = backtest_row(backtest, asset, "risk_forecast")
    return {
        "hold": hold,
        "used": used,
        "fall_saved": used["worst_fall"] - hold["worst_fall"],
        "return_gap": used["yearly_return"] - hold["yearly_return"],
        "sharpe_gap": used["sharpe"] - hold["sharpe"],
        "swing_cut": (1 - used["yearly_swing"] / hold["yearly_swing"]) * 100,
    }


def backtest_text(backtest, asset):
    info = backtest_summary(backtest, asset)
    name = SHORT[asset]
    calmer = info["swing_cut"] > 0
    parts = []
    if calmer:
        parts.append(f"Using the risk forecast made the ride about {info['swing_cut']:.0f}% calmer for {name}")
    else:
        parts.append(f"Using the risk forecast did not make the ride calmer for {name}")
    if info["fall_saved"] > 0.5:
        parts.append(f"and the worst fall was {info['fall_saved']:.1f} points shallower ({info['used']['worst_fall']:.0f}% instead of {info['hold']['worst_fall']:.0f}%).")
    else:
        parts.append(f"and the worst fall was about the same ({info['used']['worst_fall']:.0f}% against {info['hold']['worst_fall']:.0f}%).")
    if info["return_gap"] < -0.1:
        parts.append(f"The price: {abs(info['return_gap']):.1f} points less return per year.")
    elif info["return_gap"] > 0.1:
        parts.append(f"It also earned {info['return_gap']:.1f} points more per year.")
    else:
        parts.append("Yearly return was about the same.")
    if info["sharpe_gap"] > 0.02:
        parts.append("Return per unit of risk (Sharpe) improved.")
    elif info["sharpe_gap"] < -0.02:
        parts.append("Return per unit of risk (Sharpe) got worse.")
    else:
        parts.append("Return per unit of risk (Sharpe) hardly changed.")
    return " ".join(parts)


def backtest_rows(backtest):
    rows = []
    for asset in ASSETS:
        info = backtest_summary(backtest, asset)
        hold = info["hold"]
        used = info["used"]
        rows.append((
            SHORT[asset],
            f"{hold['yearly_return']:+.1f}% > {used['yearly_return']:+.1f}%",
            f"{hold['worst_fall']:.0f}% > {used['worst_fall']:.0f}%",
            f"{hold['sharpe']:.2f} > {used['sharpe']:.2f}",
        ))
    return rows


def draw_backtest(fig, backtest, asset):
    fig.clear()
    ax = fig.add_subplot(111)
    style_axes(ax)
    if backtest is None:
        ax.text(0.5, 0.5, "Press Update data to run the risk-managed test", ha="center", va="center", transform=ax.transAxes, color=GREY)
        ax.set_axis_off()
        return
    curves = backtest["curves"]
    ax.plot(curves[asset + "_buy_and_hold"] * 100, color="#7f8c99", linewidth=1.6, label="Buy and hold")
    ax.plot(curves[asset + "_risk_forecast"] * 100, color=MIX_COLOR, linewidth=1.8, label="Hold less when forecast risk is high")
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Value of 100 invested")
    ax.set_title(f"{FULL[asset]}: what 100 would have become")
    ax.legend(loc="upper left", frameon=False)
