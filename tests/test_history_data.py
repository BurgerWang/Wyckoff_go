"""History boundaries and gaps are evidence, not silently repaired data."""
import json

import pandas as pd
import pytest

from research import history_data as hd
from research.universe import CRYPTO, EQUITIES, Instrument


def frame(dates, symbol="AAPL", crypto=False):
    f = pd.DataFrame({"Open": 10., "High": 12., "Low": 9., "Close": 11., "Volume": 100.}, index=pd.DatetimeIndex(dates, name="Date"))
    f.attrs.update(symbol=symbol, source="binance_spot" if crypto else "yahoo", timeframe="1d",
                   market_type="spot" if crypto else "cash", price_currency="USDT" if crypto else "USD",
                   volume_unit=symbol[:-4] if crypto else "shares", adjusted=not crypto)
    if crypto:
        f["QuoteVolume"] = 1100.
        f.attrs["quote_volume_unit"] = "USDT"
        f.attrs["first_kline"] = {"date": hd.PAIR_BOUNDARIES[symbol][0], "announcement": hd.PAIR_BOUNDARIES[symbol][1]}
    return f


def test_ipo_boundary_excludes_only_documented_head():
    f = frame(["2020-09-30", "2020-10-02"], "PLTR")
    c = hd.coverage(f, EQUITIES, Instrument("PLTR", "Palantir", ""), "2006-09-30", "2020-10-03")
    assert c["effective_start"] == "2020-09-30"
    assert c["missing_dates"] == ["2020-10-01"]
    assert c["status"] == "partial_missing_sessions"
    assert [s["bars"] for s in c["continuous_segments"]] == [1, 1]


def test_unexplained_missing_head_not_treated_as_ipo():
    c = hd.coverage(frame(["2020-10-02"]), EQUITIES, Instrument("AAPL", "", ""), "2020-09-30", "2020-10-03")
    assert c["missing_dates"] == ["2020-09-30", "2020-10-01"]


@pytest.mark.parametrize("symbol,date", [("DELL", "2018-12-28"), ("VRT", "2020-02-10"), ("AVGO", "2009-08-06")])
def test_security_boundary_refuses_predecessor(symbol, date):
    assert hd.equity_start(symbol, "2006-09-30") == pd.Timestamp(date)
    with pytest.raises(ValueError, match="outside documented"):
        hd.coverage(frame(["2008-01-02"], symbol), EQUITIES, Instrument(symbol, "", ""), "2006-09-30", "2021-01-01")


def test_crypto_pair_gaps_remain_disclosed_and_units_checked():
    f = frame(["2017-08-17", "2017-08-19"], "BTCUSDT", True)
    c = hd.coverage(f, CRYPTO, CRYPTO.instruments[0], "2006-09-30", "2017-08-20")
    assert c["missing_dates"] == ["2017-08-18"]
    assert c["expected_sessions"] == 3
    f.attrs["volume_unit"] = "USDT"
    with pytest.raises(ValueError, match="volume_unit"):
        hd.coverage(f, CRYPTO, CRYPTO.instruments[0], "2006-09-30", "2017-08-20")


def test_bad_interior_bar_is_not_skipped():
    f = frame(["2020-09-30", "2020-10-01", "2020-10-02"])
    f.loc["2020-10-01", "Close"] = 20
    with pytest.raises(ValueError, match="OHLC values violate"):
        hd.coverage(f, EQUITIES, Instrument("AAPL", "", ""), "2020-09-30", "2020-10-03")


def test_first_kline_must_match_official_boundary(monkeypatch):
    def payload(url):
        ts = int(pd.Timestamp("2017-08-18", tz="UTC").timestamp() * 1000)
        return json.dumps([[ts, "10", "12", "9", "11", "100", ts + 86400000 - 1, "1100", 1, "1", "1", "0"]]).encode()
    monkeypatch.setattr(hd.bd, "_download", payload)
    with pytest.raises(ValueError, match="disagrees with official"):
        hd.first_pair_kline("BTCUSDT")


def test_crypto_cache_without_first_pair_evidence_rejected():
    f = frame(["2017-08-17"], "BTCUSDT", True)
    del f.attrs["first_kline"]
    with pytest.raises(ValueError, match="first-kline verification"):
        hd.coverage(f, CRYPTO, CRYPTO.instruments[0], "2017-08-17", "2017-08-18")


def test_roundoff_requires_equal_unadjusted_fields_and_does_not_mask_price_error():
    f = frame(["2020-09-30"])
    f.loc[:, "Close"] = 12.
    f.loc[:, "High"] = __import__("numpy").nextafter(12., 0.)
    original = frame(["2020-09-30"])
    original.loc[:, ["High", "Close"]] = 24.
    untouched, fixes = hd.normalize_adjustment_roundoff(f.copy(), "AAPL")
    assert not fixes and untouched.High.iloc[0] < untouched.Close.iloc[0]
    fixed, fixes = hd.normalize_adjustment_roundoff(f.copy(), "AAPL", original)
    assert fixed.High.iloc[0] == fixed.Close.iloc[0]
    assert fixes[0]["original_equal_endpoint"] == "Close"
    f.loc[:, "High"] = 11.
    bad, fixes = hd.normalize_adjustment_roundoff(f, "AAPL", original)
    assert not fixes
    with pytest.raises(ValueError, match="OHLC values violate"):
        hd.validate_ohlcv(bad)


def test_shortened_rest_interval_requires_exact_valid_archive_day(monkeypatch):
    day = pd.Timestamp("2018-02-08", tz="UTC")
    ts = day.value // 1_000_000
    row = [ts, "10", "12", "9", "11", "100", ts + 12345, "1100", 1, "1", "1", "0"]
    monkeypatch.setattr(hd.bd, "_archive_file", lambda *a: [(day, [10., 12., 9., 11., 100., 1100.])])
    repairs = []
    result = hd.resolve_kline(row, "BTCUSDT", {}, repairs, [])
    assert result[0] == day
    assert repairs[0]["ohlcv_equal"]
    assert repairs[0]["rest_raw_row"] == row
    monkeypatch.setattr(hd.bd, "_archive_file", lambda *a: [(day + hd.bd.DAY, [10., 12., 9., 11., 100., 1100.])])
    with pytest.raises(ValueError, match="cannot resolve incomplete"):
        hd.resolve_kline(row, "BTCUSDT", {}, [], [])
