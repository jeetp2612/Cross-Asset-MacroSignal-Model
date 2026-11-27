import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
from helpers import simulate

FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["risk_features.py", "forecast.py", "eda_risk.py", "vol_target.py", "build_dataset.py", "macro.py"]


def make_workspace(tmp_path, rows=1800):
    for name in SCRIPTS:
        shutil.copy(os.path.join(FOLDER, name), tmp_path / name)
    (tmp_path / "data").mkdir()
    simulate(rows=rows).to_csv(tmp_path / "data" / "market.csv")
    return tmp_path


def run(workspace, script, extra_path=None):
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    if extra_path:
        environment["PYTHONPATH"] = extra_path
    return subprocess.run([sys.executable, script], cwd=workspace, capture_output=True, text=True, env=environment, check=False)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    workspace = make_workspace(tmp_path_factory.mktemp("pipeline"))
    first = run(workspace, "risk_features.py")
    second = run(workspace, "forecast.py")
    third = run(workspace, "eda_risk.py")
    fourth = run(workspace, "vol_target.py")
    return workspace, first, second, third, fourth


def test_scripts_run_cleanly(pipeline):
    _, first, second, third, fourth = pipeline
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert third.returncode == 0, third.stderr
    assert fourth.returncode == 0, fourth.stderr


def test_risk_features_have_targets_only_at_the_end_missing(pipeline):
    workspace = pipeline[0]
    table = pd.read_csv(workspace / "data" / "risk_features.csv", index_col=0, parse_dates=True)
    target = table["SPY_fwdvol"]
    assert target.iloc[:-21].notna().all()
    assert target.iloc[-21:].isna().all()
    features = table.drop(columns=[c for c in table.columns if c.endswith("_fwdvol")])
    assert features.notna().all().all()


def test_risk_features_use_no_future_information(tmp_path):
    workspace = make_workspace(tmp_path)
    assert run(workspace, "risk_features.py").returncode == 0
    full = pd.read_csv(workspace / "data" / "risk_features.csv", index_col=0, parse_dates=True)
    market = pd.read_csv(workspace / "data" / "market.csv", index_col=0, parse_dates=True)
    market.iloc[:1500].to_csv(workspace / "data" / "market.csv")
    assert run(workspace, "risk_features.py").returncode == 0
    short = pd.read_csv(workspace / "data" / "risk_features.csv", index_col=0, parse_dates=True)
    columns = [c for c in short.columns if not c.endswith("_fwdvol")]
    shared = short.index.intersection(full.index)
    assert len(shared) > 1000
    difference = (short.loc[shared, columns] - full.loc[shared, columns]).abs().max().max()
    assert difference < 1e-9


def test_forecast_outputs_are_complete(pipeline):
    workspace = pipeline[0]
    scores = pd.read_csv(workspace / "data" / "forecast_scores.csv")
    assert set(scores["model"]) == {"naive", "har", "harx", "boosted", "blend"}
    assert set(scores["asset"]) == {"SPY", "TLT", "GLD", "UUP"}
    naive = scores[scores["model"] == "naive"]["improvement_pct"]
    assert naive.abs().max() < 1e-9
    latest = pd.read_csv(workspace / "data" / "forecast_latest.csv")
    assert len(latest) == 4
    assert latest["label"].isin(["Calmer than usual", "Usual", "Wilder than usual"]).all()
    assert (latest["forecast_vol"] > 0).all()
    predictions = pd.read_csv(workspace / "data" / "forecast_predictions.csv", index_col=0, parse_dates=True)
    assert predictions.index.min().year == 2015
    assert (workspace / "data" / "forecast.png").exists()


def test_forecast_models_beat_naive_on_clustered_risk(pipeline):
    workspace = pipeline[0]
    scores = pd.read_csv(workspace / "data" / "forecast_scores.csv")
    best = scores[scores["model"] != "naive"].groupby("asset")["improvement_pct"].max()
    assert (best > 0).sum() >= 3
    assert best.mean() > 5


def test_significance_and_alerts_are_complete(pipeline):
    workspace = pipeline[0]
    significance = pd.read_csv(workspace / "data" / "forecast_significance.csv")
    assert len(significance) == 16
    assert significance["p_value"].between(0, 1).all()
    assert (significance["p_adjusted"] >= significance["p_value"] - 1e-12).all()
    assert (significance["cut_low"] <= significance["cut_high"]).all()
    best = significance.sort_values("p_value").iloc[0]
    assert best["p_value"] < 0.05 and best["cut_low"] > 0
    assert (significance[significance["p_value"] < 0.05]["cut_high"] > 0).all()
    alerts = pd.read_csv(workspace / "data" / "forecast_alerts.csv")
    assert set(alerts["method"]) == {"naive", "alert"}
    for column in ["caught", "false_alarm", "precision", "months_flagged"]:
        assert alerts[column].between(0, 100).all()
    caught = alerts.pivot(index="asset", columns="method", values="caught")
    assert (caught["alert"] > caught["naive"]).sum() >= 3
    latest = pd.read_csv(workspace / "data" / "forecast_latest.csv")
    assert latest["wild_prob"].between(0, 100).all()


def test_vol_target_results_are_sensible(pipeline):
    workspace = pipeline[0]
    results = pd.read_csv(workspace / "data" / "vol_target_results.csv")
    assert set(results["strategy"]) == {"buy_and_hold", "naive_rule", "risk_forecast"}
    assert len(results) == 12
    hold = results[results["strategy"] == "buy_and_hold"]
    assert (hold["average_exposure"] == 100).all()
    used = results[results["strategy"] == "risk_forecast"]
    assert (used["average_exposure"] <= 100).all() and (used["average_exposure"] > 20).all()
    curves = pd.read_csv(workspace / "data" / "vol_target_curves.csv", index_col=0)
    assert curves.notna().all().all()
    assert (workspace / "data" / "vol_target.png").exists()


def test_vol_target_positions_use_no_future_information(tmp_path):
    workspace = make_workspace(tmp_path)
    assert run(workspace, "risk_features.py").returncode == 0
    assert run(workspace, "forecast.py").returncode == 0
    assert run(workspace, "vol_target.py").returncode == 0
    first = pd.read_csv(workspace / "data" / "vol_target_curves.csv", index_col=0, parse_dates=True)
    predictions = pd.read_csv(workspace / "data" / "forecast_predictions.csv", index_col=0, parse_dates=True)
    changed = predictions.copy()
    changed.loc[changed.index[-30:], [c for c in changed.columns if c.endswith(("_blend", "_har", "_harx", "_boosted"))]] *= 5
    changed.to_csv(workspace / "data" / "forecast_predictions.csv")
    assert run(workspace, "vol_target.py").returncode == 0
    second = pd.read_csv(workspace / "data" / "vol_target_curves.csv", index_col=0, parse_dates=True)
    early = first.index[:-31]
    assert (first.loc[early] - second.loc[early]).abs().max().max() < 1e-12


def test_eda_risk_writes_its_files(pipeline):
    workspace = pipeline[0]
    for name in ["risk_acf.png", "risk_vix_vs_fwdvol.png", "risk_signal_screen.png", "risk_regimes.png", "risk_signal_screen.csv", "risk_regime_tests.csv"]:
        assert (workspace / "data" / "eda" / name).exists(), name
    acf = pd.read_csv(workspace / "data" / "eda" / "risk_acf.csv")
    assert acf["SPY_abs_returns"].iloc[0] > acf["SPY_returns"].iloc[0].__abs__()


def test_scripts_stop_with_a_clear_message_when_inputs_are_missing(tmp_path):
    for name in SCRIPTS:
        shutil.copy(os.path.join(FOLDER, name), tmp_path / name)
    for script, words in [("risk_features.py", "build_dataset.py"), ("forecast.py", "risk_features.py"), ("eda_risk.py", "risk_features.py")]:
        result = run(tmp_path, script)
        assert result.returncode != 0
        assert words in result.stderr


def write_fake_sources(folder, prices_rows=1500, drop_nfci=False, empty_prices=False):
    folder.mkdir(exist_ok=True)
    market = simulate(rows=prices_rows)
    prices = market[["SPY", "TLT", "GLD", "UUP", "VIX", "OIL", "HYG"]].rename(columns={"VIX": "^VIX", "OIL": "CL=F"})
    prices.to_csv(folder / "prices.csv")
    (folder / "yfinance.py").write_text(
        "import os\n"
        "import pandas as pd\n\n\n"
        "def download(tickers, start=None, auto_adjust=True, progress=False):\n"
        "    if os.environ.get('FAKE_EMPTY') == '1':\n"
        "        return pd.DataFrame()\n"
        "    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), 'prices.csv'), index_col=0, parse_dates=True)\n"
        "    return pd.concat({'Close': prices[tickers]}, axis=1)\n"
    )
    index = market.index
    month_starts = pd.date_range(index[0] - pd.DateOffset(months=14), index[-1], freq="MS")
    weekly = pd.date_range(index[0] - pd.Timedelta(days=400), index[-1], freq="W-SAT")
    sources = {
        "DGS10": market["US10Y"],
        "DGS2": market["US2Y"],
        "T10Y2Y": market["CURVE_10Y_2Y"],
        "DFF": market["FED_FUNDS"],
        "T10YIE": market["INFL_EXPECT"],
        "NFCI": pd.Series(range(len(weekly)), index=weekly, dtype=float),
        "ICSA": pd.Series(range(len(weekly)), index=weekly, dtype=float) + 200000,
        "UNRATE": pd.Series([float(i + 1) for i in range(len(month_starts))], index=month_starts),
        "CPIAUCSL": pd.Series([200.0 + i for i in range(len(month_starts))], index=month_starts),
    }
    if drop_nfci:
        del sources["NFCI"]
    for code, series in sources.items():
        frame = series.rename(code).to_frame()
        frame.index.name = "DATE"
        frame.to_csv(folder / ("fred_" + code + ".csv"))
    (folder / "sitecustomize.py").write_text(
        "import os\n"
        "import pandas as pd\n\n"
        "original = pd.read_csv\n\n\n"
        "def fake_read_csv(path, *args, **kwargs):\n"
        "    text = str(path)\n"
        "    if text.startswith('https://fred'):\n"
        "        code = text.split('id=')[1].split('&')[0]\n"
        "        target = os.path.join(os.path.dirname(__file__), 'fred_' + code + '.csv')\n"
        "        if not os.path.exists(target):\n"
        "            raise OSError('no such series')\n"
        "        path = target\n"
        "    return original(path, *args, **kwargs)\n\n\n"
        "pd.read_csv = fake_read_csv\n"
    )
    return market


def test_build_dataset_end_to_end_with_release_lags(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    fakes = tmp_path / "fakes"
    write_fake_sources(fakes)
    for name in ["build_dataset.py", "macro.py"]:
        shutil.copy(os.path.join(FOLDER, name), workspace / name)
    result = run(workspace, "build_dataset.py", extra_path=str(fakes))
    assert result.returncode == 0, result.stderr
    market = pd.read_csv(workspace / "data" / "market.csv", index_col=0, parse_dates=True)
    dataset = pd.read_csv(workspace / "data" / "dataset.csv", index_col=0, parse_dates=True)
    for column in ["HYG", "INFL_EXPECT", "FIN_COND", "CLAIMS", "UNRATE", "CPI"]:
        assert column in market.columns
    assert market.notna().all().all()
    assert len(market) - len(dataset) == 21
    assert "SPY_up" in dataset.columns and "SPY_up" not in market.columns
    source = pd.read_csv(fakes / "fred_UNRATE.csv", index_col=0, parse_dates=True)["UNRATE"]
    for day in ["2012-03-05", "2012-03-20", "2013-07-10"]:
        visible = source[source.index + pd.Timedelta(days=40) <= pd.Timestamp(day)]
        actual = market.loc[pd.Timestamp(day):, "UNRATE"].iloc[0]
        first_day = market.loc[pd.Timestamp(day):].index[0]
        visible = source[source.index + pd.Timedelta(days=40) <= first_day]
        assert actual == visible.iloc[-1]


def test_build_dataset_skips_optional_series_but_keeps_going(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    fakes = tmp_path / "fakes"
    write_fake_sources(fakes, drop_nfci=True)
    for name in ["build_dataset.py", "macro.py"]:
        shutil.copy(os.path.join(FOLDER, name), workspace / name)
    result = run(workspace, "build_dataset.py", extra_path=str(fakes))
    assert result.returncode == 0, result.stderr
    assert "Skipping optional series FIN_COND" in result.stdout
    market = pd.read_csv(workspace / "data" / "market.csv", index_col=0, parse_dates=True)
    assert "FIN_COND" not in market.columns
    assert "UNRATE" in market.columns


def test_build_dataset_keeps_old_files_when_download_is_empty(tmp_path):
    workspace = tmp_path / "work"
    (workspace / "data").mkdir(parents=True)
    fakes = tmp_path / "fakes"
    write_fake_sources(fakes)
    for name in ["build_dataset.py", "macro.py"]:
        shutil.copy(os.path.join(FOLDER, name), workspace / name)
    old = workspace / "data" / "dataset.csv"
    old.write_text("keep,me\n1,2\n")
    environment = dict(os.environ)
    environment.update({"PYTHONPATH": str(fakes), "FAKE_EMPTY": "1", "MPLBACKEND": "Agg"})
    result = subprocess.run([sys.executable, "build_dataset.py"], cwd=workspace, capture_output=True, text=True, env=environment, check=False)
    assert result.returncode != 0
    assert "Could not download prices from Yahoo Finance" in result.stderr
    assert old.read_text() == "keep,me\n1,2\n"
