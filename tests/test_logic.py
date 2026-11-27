import numpy as np
import pandas as pd
import pytest

import logic


def make_prices(rows=1500, seed=0):
    random = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=rows)
    frame = pd.DataFrame(index=index)
    for name in ["SPY", "TLT", "GLD", "UUP", "OIL"]:
        frame[name] = 100 * np.exp(np.cumsum(random.normal(0.0003, 0.01, rows)))
    frame["VIX"] = 18 + 6 * np.sin(np.arange(rows) / 60) + random.normal(0, 1, rows)
    frame["US10Y"] = 2 + np.cumsum(random.normal(0, 0.02, rows))
    frame["US2Y"] = 1.5 + np.cumsum(random.normal(0, 0.02, rows))
    frame["CURVE_10Y_2Y"] = frame["US10Y"] - frame["US2Y"]
    frame["FED_FUNDS"] = 1.0
    return frame


def make_summary(strategy_wins=False):
    rows = []
    for asset in logic.ASSETS:
        for model, auc in [("logistic", 0.50), ("random_forest", 0.55), ("gradient_boost", 0.60)]:
            rows.append({
                "asset": asset,
                "model": model,
                "avg_auc": auc,
                "strategy_return": 2.0 if strategy_wins else 0.5,
                "buy_hold_return": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("vix,word", [(10, "Calm"), (20, "Normal"), (27, "Nervous"), (40, "Panic")])
def test_mood_of(vix, word):
    assert logic.mood_of(vix)[0] == word


@pytest.mark.parametrize("value,word", [(0.5, "Moving together"), (-0.5, "Moving opposite"), (0.0, "Little link")])
def test_link_of(value, word):
    assert logic.link_of(value)[0] == word


@pytest.mark.parametrize("swing,word", [(5, "Low"), (15, "Medium"), (40, "High")])
def test_risk_of_swing(swing, word):
    assert logic.risk_of_swing(swing)[0] == word


def test_market_snapshot_fields():
    frame = make_prices()
    shot = logic.market_snapshot(frame)
    assert shot["date"] == frame.index[-1].date()
    assert -1 <= shot["link"] <= 1
    assert shot["best"] in logic.ASSETS
    assert 0 <= shot["rank"] <= 100


def test_dashboard_message_follows_fear_level():
    base = {"vix": 12.0, "curve": 0.5}
    assert logic.dashboard_message(base)[0] == "Markets are calm"
    assert logic.dashboard_message({"vix": 20.0, "curve": 0.5})[0] == "Markets look normal"
    assert logic.dashboard_message({"vix": 27.0, "curve": 0.5})[0] == "Investors are nervous"
    assert logic.dashboard_message({"vix": 45.0, "curve": 0.5})[0] == "Markets are in panic"


def test_dashboard_message_mentions_inverted_curve():
    plain = logic.dashboard_message({"vix": 12.0, "curve": 0.5})[1]
    inverted = logic.dashboard_message({"vix": 12.0, "curve": -0.5})[1]
    assert "inverted" not in plain
    assert "inverted" in inverted


def test_period_slice_lengths():
    frame = make_prices(rows=3000)
    assert len(logic.period_slice(frame, "All")) == 3000
    one_year = logic.period_slice(frame, "1Y")
    assert 250 <= len(one_year) <= 262
    assert len(logic.period_slice(frame, "5Y")) > len(one_year)


def test_rebased_starts_at_100():
    frame = make_prices()
    series = logic.rebased(frame, ["SPY", "GLD"])
    assert series.iloc[0].tolist() == pytest.approx([100.0, 100.0])


def test_market_facts_known_values():
    frame = pd.DataFrame({"SPY": [100.0, 120.0, 90.0, 110.0]})
    facts = logic.market_facts(frame, "SPY")
    assert facts["total"] == pytest.approx(10.0)
    assert facts["worst"] == pytest.approx(-25.0)
    assert facts["swing"] > 0


def test_nearest_row_finds_closest_day():
    import matplotlib.dates as mdates

    frame = make_prices(rows=50)
    target = frame.index[10]
    date, row = logic.nearest_row(frame, mdates.date2num(target) + 0.2)
    assert date == target
    assert row["SPY"] == frame["SPY"].iloc[10]


def test_data_age_text_fresh_and_old():
    frame = make_prices()
    last = frame.index[-1]
    fresh_text, fresh = logic.data_age_text(frame, today=last + pd.Timedelta(days=2))
    old_text, old = logic.data_age_text(frame, today=last + pd.Timedelta(days=30))
    assert fresh and "up to date" in fresh_text
    assert not old and "30 days old" in old_text


def test_load_file_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "DATA_FOLDER", str(tmp_path))
    assert logic.load_file("nothing.csv", True) is None
    assert logic.load_market() is None


def test_load_market_prefers_market_file(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "DATA_FOLDER", str(tmp_path))
    make_prices(rows=10).to_csv(tmp_path / "dataset.csv")
    make_prices(rows=20).to_csv(tmp_path / "market.csv")
    assert len(logic.load_market()) == 20
    (tmp_path / "market.csv").unlink()
    assert len(logic.load_market()) == 10


def test_parse_amount_accepts_money_formats():
    assert logic.parse_amount("$25,000") == 25000
    assert logic.parse_amount(" 100 ") == 100


@pytest.mark.parametrize("text", ["abc", "", "0", "-5", "nan", "1e99"])
def test_parse_amount_rejects_bad_input(text):
    with pytest.raises(ValueError):
        logic.parse_amount(text)


def test_normalise_sums_to_one():
    assert sum(logic.normalise([60, 40, 0, 0])) == pytest.approx(1.0)
    assert logic.normalise([1, 1, 0, 0]) == [0.5, 0.5, 0, 0]


def test_normalise_rejects_all_zero():
    with pytest.raises(ValueError):
        logic.normalise([0, 0, 0, 0])


def test_replay_history_all_stocks_matches_stock_curve():
    frame = make_prices()
    mix, stocks = logic.replay_history(frame, 2015, [1, 0, 0, 0])
    assert mix.iloc[-1] == pytest.approx(stocks.iloc[-1])


def test_replay_history_needs_enough_days():
    with pytest.raises(ValueError):
        logic.replay_history(make_prices(), 2090, [1, 0, 0, 0])


def test_curve_stats_doubling_in_one_year():
    index = pd.bdate_range("2020-01-01", periods=252)
    curve = pd.Series(np.linspace(1.0, 2.0, 252), index=index)
    stats = logic.curve_stats(curve)
    assert stats["total"] == pytest.approx(100.0)
    assert stats["yearly"] == pytest.approx(100.0)
    assert stats["worst"] == pytest.approx(0.0)


def test_curve_stats_drawdown_is_negative():
    curve = pd.Series([1.0, 1.5, 0.75, 1.0])
    assert logic.curve_stats(curve)["worst"] == pytest.approx(-50.0)


def test_whatif_text_mentions_tradeoff_for_a_calmer_mix():
    index = pd.bdate_range("2020-01-01", periods=300)
    stocks = pd.Series(np.concatenate([np.linspace(1, 2, 150), np.linspace(2, 1.2, 150)]), index=index)
    mix = pd.Series(np.linspace(1, 1.3, 300), index=index)
    assert "cushioned" in logic.whatif_text(mix, stocks, [0.6, 0.4, 0, 0])


def test_whatif_text_all_stocks():
    curve = pd.Series(np.linspace(1, 2, 300))
    assert "same as 100% stocks" in logic.whatif_text(curve, curve, [1, 0, 0, 0])


def test_reality_rows_pick_best_auc_model():
    rows = logic.reality_rows(make_summary())
    assert len(rows) == 4
    assert all(row["model"] == "gradient_boost" for row in rows)
    assert rows[0]["strategy"] == pytest.approx(50.0)
    assert rows[0]["hold"] == pytest.approx(100.0)


def make_forecast(improvement=20.0, best="blend"):
    scores = []
    latest = []
    for asset in logic.ASSETS:
        for model in ["naive", "har", "harx", "boosted", "blend"]:
            gain = 0.0 if model == "naive" else improvement
            scores.append({
                "asset": asset, "model": model,
                "rmse": 5.0 - gain / 10 if model != "naive" else 5.0,
                "mae": 4.0, "improvement_pct": gain,
                "years_beating_naive": 0 if model == "naive" else 9,
                "years_total": 12, "regime_accuracy": 80.0, "wild_caught": 60.0,
            })
        latest.append({
            "asset": asset, "date": "2026-01-02", "current_vol": 12.0, "best_model": best,
            "forecast_vol": 15.0, "low_cut": 10.0, "high_cut": 18.0, "label": "Usual", "history_rank": 55.0, "wild_prob": 40.0,
        })
    index = pd.bdate_range("2015-01-01", periods=400)
    predictions = pd.DataFrame(index=index)
    for asset in logic.ASSETS:
        for name in ["actual", "naive", "har", "harx", "boosted", "blend"]:
            predictions[asset + "_" + name] = 12 + np.sin(np.arange(400) / 20)
    significance = pd.DataFrame([
        {"asset": a, "model": m, "dm_stat": 3.0, "p_value": p_value, "p_adjusted": min(1.0, p_value * 3), "cut_low": 5.0, "cut_high": 30.0}
        for a in logic.ASSETS for m, p_value in [("har", 0.2), ("harx", 0.08), ("boosted", 0.03), ("blend", 0.004)]
    ])
    alerts = pd.DataFrame([
        {"asset": a, "method": m, "caught": c, "false_alarm": f, "precision": 40.0, "months_flagged": 25.0}
        for a in logic.ASSETS for m, c, f in [("naive", 40.0, 10.0), ("alert", 65.0, 25.0)]
    ])
    return {"scores": pd.DataFrame(scores), "latest": pd.DataFrame(latest).set_index("asset"), "predictions": predictions, "significance": significance, "alerts": alerts}


def make_backtest():
    rows = []
    for asset in logic.ASSETS:
        for strategy, ret, fall, sharpe, exposure in [("buy_and_hold", 10.0, -30.0, 0.6, 100.0), ("naive_rule", 8.0, -22.0, 0.65, 90.0), ("risk_forecast", 9.0, -20.0, 0.7, 88.0)]:
            rows.append({"asset": asset, "strategy": strategy, "average_exposure": exposure, "total_return": 100.0, "yearly_return": ret, "yearly_swing": 15.0 if strategy == "buy_and_hold" else 12.0, "sharpe": sharpe, "worst_fall": fall})
    index = pd.bdate_range("2015-01-01", periods=300)
    curves = pd.DataFrame(index=index)
    for asset in logic.ASSETS:
        for strategy in ["buy_and_hold", "naive_rule", "risk_forecast"]:
            curves[asset + "_" + strategy] = np.linspace(1, 1.5, 300)
    return {"results": pd.DataFrame(rows), "curves": curves}


@pytest.mark.parametrize("view", logic.VIEWS)
def test_chart_frame_and_hover_for_every_view(view):
    frame = make_prices()
    part = logic.chart_frame(frame, ["SPY", "TLT"], view)
    assert list(part.columns) == ["SPY", "TLT"]
    assert part.notna().all().all()
    assert isinstance(logic.hover_values(part.iloc[-1], view), str)
    if view == "Growth":
        assert part.iloc[0].tolist() == [100, 100]
    if view == "Drawdown":
        assert part.max().max() <= 1e-9


def test_asset_table_has_all_markets_with_and_without_forecast():
    frame = make_prices()
    rows = logic.asset_table(frame)
    assert [row["asset"] for row in rows] == logic.ASSETS
    assert all(row["forecast"] is None for row in rows)
    assert all(row["from_peak"] <= 1e-9 for row in rows)
    rows = logic.asset_table(frame, make_forecast())
    assert rows[0]["forecast"] == 15.0 and rows[0]["label"] == "Usual"


def test_percent_change_handles_short_series():
    series = pd.Series([100.0, 110.0])
    assert logic.percent_change(series, 252) == pytest.approx(10.0)


def test_corr_table_and_markets_text():
    frame = make_prices()
    table = logic.corr_table(frame)
    assert table.shape == (4, 4)
    assert table.iloc[0, 0] == pytest.approx(1.0)
    assert "Stocks" in logic.markets_text(frame)


def test_forecast_summary_and_text():
    forecast = make_forecast()
    info = logic.forecast_summary(forecast, "SPY")
    assert info["best"] == "blend" and info["years_total"] == 12
    text = logic.forecast_text(forecast, "SPY")
    assert "15.0%" in text and "9 of 12" in text
    naive_text = logic.forecast_text(make_forecast(0.0, "naive"), "SPY")
    assert "No model beat" in naive_text


def test_proof_of_levels():
    assert logic.proof_of(0.004)[0] == "Yes, strong"
    assert logic.proof_of(0.03)[0] == "Yes"
    assert logic.proof_of(0.08)[0] == "Maybe"
    assert logic.proof_of(0.3)[0] == "Not proven"


def test_forecast_summary_carries_proof_and_alert():
    forecast = make_forecast()
    info = logic.forecast_summary(forecast, "SPY")
    assert info["p_value"] == 0.004 and info["wild_prob"] == 40.0
    assert info["alert_caught"] == 65.0 and info["naive_caught"] == 40.0
    assert "unlikely to be luck" in logic.forecast_text(forecast, "SPY")
    assert "four models were tried" not in logic.forecast_text(forecast, "SPY")
    forecast_two = make_forecast(best="boosted")
    assert "four models were tried" in logic.forecast_text(forecast_two, "SPY")
    assert "65%" in logic.alert_text(forecast, "SPY")
    forecast["significance"] = pd.DataFrame([{"asset": a, "model": "blend", "dm_stat": 0.1, "p_value": 0.4, "cut_low": -5.0, "cut_high": 9.0} for a in logic.ASSETS])
    assert "could not rule out luck" in logic.forecast_text(forecast, "SPY")


def test_forecast_summary_works_without_new_files():
    forecast = make_forecast()
    forecast["significance"] = None
    forecast["alerts"] = None
    forecast["latest"] = forecast["latest"].drop(columns="wild_prob")
    info = logic.forecast_summary(forecast, "SPY")
    assert info["p_value"] is None and info["wild_prob"] is None
    assert logic.alert_text(forecast, "SPY") == ""
    assert logic.scoreboard_rows(forecast, "SPY")[0][3] == "n/a" or logic.scoreboard_rows(forecast, "SPY")[0][3] == "-"


def test_backtest_text_and_rows():
    backtest = make_backtest()
    text = logic.backtest_text(backtest, "SPY")
    assert "calmer" in text and "shallower" in text and "less return" in text and "improved" in text
    rows = logic.backtest_rows(backtest)
    assert len(rows) == 4 and rows[0][0] == "Stocks" and rows[0][1] == "+10.0% > +9.0%"
    flat = make_backtest()
    flat["results"]["worst_fall"] = -30.0
    flat["results"]["yearly_return"] = 10.0
    flat["results"]["sharpe"] = 0.6
    flat["results"]["yearly_swing"] = 15.0
    plain = logic.backtest_text(flat, "SPY")
    assert "did not make the ride calmer" in plain and "about the same" in plain and "hardly changed" in plain


def test_load_backtest(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "DATA_FOLDER", str(tmp_path))
    assert logic.load_backtest() is None
    backtest = make_backtest()
    backtest["results"].to_csv(tmp_path / "vol_target_results.csv", index=False)
    backtest["curves"].to_csv(tmp_path / "vol_target_curves.csv")
    assert logic.load_backtest()["results"].shape[0] == 12


def test_scoreboard_rows_put_best_first_and_dash_for_naive():
    rows = logic.scoreboard_rows(make_forecast(), "GLD")
    assert len(rows) == 5
    naive = next(row for row in rows if row[0] == logic.MODEL_LABELS["naive"])
    assert naive[1] == "-" and naive[2] == "-"
    assert rows[-1] == naive


def test_load_forecast_missing_and_present(monkeypatch, tmp_path):
    monkeypatch.setattr(logic, "DATA_FOLDER", str(tmp_path))
    assert logic.load_forecast() is None
    forecast = make_forecast()
    forecast["scores"].to_csv(tmp_path / "forecast_scores.csv", index=False)
    forecast["latest"].reset_index().to_csv(tmp_path / "forecast_latest.csv", index=False)
    forecast["predictions"].to_csv(tmp_path / "forecast_predictions.csv")
    loaded = logic.load_forecast()
    assert set(loaded) == {"scores", "latest", "predictions", "significance", "alerts"}
    assert loaded["significance"] is None and loaded["alerts"] is None
    assert loaded["latest"].loc["SPY", "label"] == "Usual"


def test_mix_forecast_vol():
    frame = make_prices()
    forecast = make_forecast()
    assert logic.mix_forecast_vol(frame, [1, 0, 0, 0], forecast) == pytest.approx(15.0)
    assert logic.mix_forecast_vol(frame, [0.25] * 4, forecast) < 15.0
    assert logic.mix_forecast_vol(frame, [1, 0, 0, 0], None) is None


def test_risk_rows_and_reality_text():
    risk = logic.risk_rows(make_forecast())
    assert len(risk) == 4 and risk[0]["improvement"] == 20.0
    direction = logic.reality_rows(make_summary())
    both = logic.reality_text(risk, direction)
    assert "Risk can be forecast" in both and "Direction" in both
    assert "not forecast better" in logic.reality_text(logic.risk_rows(make_forecast(-5.0)), [])
    assert "Press Update data" in logic.reality_text([], [])
    won = logic.reality_text([], logic.reality_rows(make_summary(True)))
    assert "beat buying and holding" in won


def test_drawing_functions_run_without_a_window():
    from matplotlib.figure import Figure

    frame = make_prices()
    fig = Figure()
    for view in logic.VIEWS:
        part = logic.chart_frame(frame, ["SPY", "TLT"], view)
        logic.draw_compare(fig, part, ["SPY", "TLT"], view)
    logic.draw_compare(fig, None, [])
    mix, stocks = logic.replay_history(frame, 2016, [0.6, 0.4, 0, 0])
    logic.draw_whatif(fig, mix, stocks, 10000, 2016)
    logic.draw_markets(fig, frame)
    logic.draw_markets(fig, frame.tail(100))
    forecast = make_forecast()
    for model in ["best", "naive", "har", "blend"]:
        logic.draw_forecast(fig, forecast, "SPY", model)
    logic.draw_forecast(fig, None, "SPY", "best")
    logic.draw_reality(fig, logic.risk_rows(forecast), logic.reality_rows(make_summary()))
    logic.draw_reality(fig, logic.risk_rows(forecast), [])
    logic.draw_reality(fig, [], [])
    for asset in logic.ASSETS:
        logic.draw_backtest(fig, make_backtest(), asset)
    logic.draw_backtest(fig, None, "SPY")
