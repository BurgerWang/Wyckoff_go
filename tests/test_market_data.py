import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_data import fetch, normalize_download, read_csv, validate_ohlcv


def bars(n=30):
    return pd.DataFrame({"Open": 101.0, "High": 102.0, "Low": 100.0,
                         "Close": 101.0, "Volume": 100.0},
                        index=pd.date_range("2025-01-01", periods=n))


@pytest.mark.parametrize("problem", ["duplicate", "unsorted", "nan", "infinity", "bad_ohlc", "negative_volume", "zero_price"])
def test_invalid_market_bars_are_rejected(problem):
    df = bars()
    if problem == "duplicate":
        df.index = pd.DatetimeIndex([df.index[0], *df.index[:-1]])
    elif problem == "unsorted":
        df = df.iloc[::-1]
    elif problem == "nan":
        df.iloc[0, df.columns.get_loc("Close")] = np.nan
    elif problem == "infinity":
        df.iloc[0, df.columns.get_loc("High")] = np.inf
    elif problem == "bad_ohlc":
        df.iloc[0, df.columns.get_loc("Close")] = 103
    elif problem == "negative_volume":
        df.iloc[0, df.columns.get_loc("Volume")] = -1
    else:
        df.iloc[0, df.columns.get_loc("Low")] = 0
    with pytest.raises(ValueError):
        validate_ohlcv(df)


def test_validator_preserves_input_and_attrs():
    df = bars()
    df.attrs["symbol"] = "TEST"
    result = validate_ohlcv(df)
    result.iloc[0, 0] = 100
    assert df.iloc[0, 0] == 101
    assert result.attrs == df.attrs


@pytest.mark.parametrize("order", ["fields_first", "ticker_first"])
def test_yahoo_column_layouts_select_one_instrument(order):
    df = bars()
    df.columns = pd.MultiIndex.from_tuples([(c, "TEST") if order == "fields_first" else ("TEST", c) for c in df])
    result = normalize_download(df, "TEST")
    assert result["Close"].iloc[0] == 101
    assert result.attrs["symbol"] == "TEST"
    with pytest.raises(ValueError, match="exactly"):
        normalize_download(df, "WRONG")


def test_multiple_tickers_cannot_be_collapsed_into_duplicate_ohlc():
    raw = pd.concat({"TEST": bars(), "OTHER": bars()}, axis=1)
    with pytest.raises(ValueError, match="exactly"):
        normalize_download(raw, "TEST")


def test_fractional_year_fetch_and_invalid_timeframe(monkeypatch):
    calls = []
    def download(symbol, **kwargs):
        calls.append((symbol, kwargs))
        return bars()
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    assert len(fetch("TEST", 0.5)) == 30
    request = calls[0][1]
    assert (pd.Timestamp(request["end"]) - pd.Timestamp(request["start"])).days == 183
    with pytest.raises(ValueError, match="timeframe"):
        fetch("TEST", tf="5m")
    assert len(calls) == 1


def test_four_hour_bins_have_utc_completion_times(monkeypatch):
    df = bars(8)
    df.index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: df))
    result = fetch("TEST", 1, "4h")
    assert result.index.equals(pd.date_range("2025-01-01 04:00", periods=2, freq="4h", tz="UTC"))
    assert result["Volume"].tolist() == [400, 400]
    assert result.attrs["aggregation"] == "UTC calendar bins"


def test_csv_dates_do_not_silently_sort_or_drop(tmp_path):
    df = bars().iloc[::-1]
    path = tmp_path / "input.csv"
    df.to_csv(path, index_label="Date")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_csv(path)


def test_csv_roundtrip_preserves_original_float_evidence(tmp_path):
    frame = bars(3)
    frame["Close"] = [101.12345678901235, 101.23456789012346, 101.34567890123456]
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index_label="Date")
    restored = read_csv(path)
    assert np.array_equal(restored["Close"].to_numpy(), frame["Close"].to_numpy())


def test_hourly_fetch_preserves_current_day_end_and_half_hour_completion(monkeypatch):
    df = bars(4)
    df.index = pd.date_range("2025-01-01 12:30", periods=4, freq="h", tz="UTC")
    calls = []
    def download(*args, **kwargs):
        calls.append(kwargs)
        return df
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=download))
    result = fetch("TEST", 1, "4h")
    assert isinstance(calls[0]["end"], pd.Timestamp)
    assert result.index[-1] == pd.Timestamp("2025-01-01 16:30", tz="UTC")
    assert result["Volume"].iloc[-1] == 400


def test_bin_with_unfinished_hour_is_not_published_then_rewritten(monkeypatch):
    import market_data
    actual_timestamp = pd.Timestamp
    class ClockTimestamp(actual_timestamp):
        @classmethod
        def now(cls, tz=None):
            return actual_timestamp("2025-01-01 16:10", tz="UTC")
    class PandasProxy:
        def __getattr__(self, name):
            return ClockTimestamp if name == "Timestamp" else getattr(pd, name)
    df = bars(7)
    df.index = pd.date_range("2025-01-01 09:30", periods=7, freq="h", tz="UTC")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: df))
    monkeypatch.setattr(market_data, "pd", PandasProxy())
    result = market_data.fetch("TEST", 1, "4h")
    assert result.index[-1] == actual_timestamp("2025-01-01 12:30", tz="UTC")
    assert result["Volume"].iloc[-1] == 300


def test_yahoo_empty_tail_is_explicitly_recorded_but_other_missing_data_fails():
    df = bars()
    df.iloc[-1] = [np.nan, np.nan, np.nan, np.nan, 0]
    normalized = normalize_download(df, "TEST")
    assert len(normalized) == len(df) - 1
    assert normalized.attrs["excluded_empty_dates"] == [str(df.index[-1])]
    assert normalized.attrs["warnings"]
    df.iloc[-2, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(ValueError, match="finite"):
        normalize_download(df, "TEST")
    df = bars()
    df.iloc[1] = [np.nan, np.nan, np.nan, np.nan, 0]
    with pytest.raises(ValueError, match="finite"):
        normalize_download(df, "TEST")
