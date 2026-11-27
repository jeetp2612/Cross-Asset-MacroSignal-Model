import pandas as pd


def align_release(series, lag_days, calendar):
    """Make each economic value visible only after its release delay."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    all_dates = shifted.index.union(calendar)
    return shifted.reindex(all_dates).ffill().reindex(calendar)
