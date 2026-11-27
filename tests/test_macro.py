import pandas as pd

from macro import align_release


def test_release_lag_hides_values_until_published():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    calendar = pd.bdate_range("2020-01-02", "2020-04-01")
    aligned = align_release(series, 40, calendar)
    assert aligned.loc["2020-02-05"] != aligned.loc["2020-02-05"]
    assert aligned.loc["2020-02-12"] == 1.0
    assert aligned.loc["2020-03-10"] == 1.0
    assert aligned.loc["2020-03-12"] == 2.0


def test_zero_lag_keeps_daily_values_and_handles_gaps():
    series = pd.Series([1.0, None, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    calendar = pd.bdate_range("2020-01-02", "2020-01-07")
    aligned = align_release(series, 0, calendar)
    assert aligned.tolist() == [1.0, 1.0, 3.0, 3.0]
