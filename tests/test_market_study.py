from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from analysis import analyze_frame, _serializable
from research.market_study import build_replay, check_equity_sessions, load_data, markdown_summary, validate_profile_data
from research.universe import CRYPTO, EQUITIES, Instrument, verify_identity


def bars(n=85):
    close = 100 + np.sin(np.arange(n) / 3) * 6
    frame = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                          "Close": close, "Volume": 100 + np.arange(n) % 10},
                         index=pd.date_range("2025-01-01", periods=n))
    frame.attrs.update(timeframe="1d", source="test_fixture")
    return frame


def test_replay_matches_standalone_prefix_and_ignores_later_price_shock():
    frame = bars()
    instrument = Instrument("BTCUSDT", "Bitcoin", "fixture")
    original = frame.copy(deep=True)
    replay = build_replay(frame, instrument, CRYPTO, replay_bars=10, benchmark=bars(100))
    first = replay["snapshots"][0]
    prefix = frame.iloc[:first["bar_index"] + 1]
    standalone = analyze_frame(prefix, "BTCUSDT", config=CRYPTO.config,
                               benchmark=bars(100).loc[:prefix.index[-1]])["analyses"]["atr"]["wyckoff"]
    assert first["phase"] == _serializable(standalone["phase"])
    assert first["count"] == _serializable(standalone["count"])
    assert first["context"] == _serializable(standalone["context"])
    shocked = frame.copy()
    shocked.iloc[-1, :4] *= 8
    again = build_replay(shocked, instrument, CRYPTO, replay_bars=10, benchmark=bars(100))
    assert again["snapshots"][:-1] == replay["snapshots"][:-1]
    future_benchmark = bars(100)
    future_benchmark.loc[future_benchmark.index > frame.index[-1], ["Open", "High", "Low", "Close"]] *= 9
    assert build_replay(frame, instrument, CRYPTO, replay_bars=10, benchmark=future_benchmark)["snapshots"] == replay["snapshots"]
    pd.testing.assert_frame_equal(frame, original)
    assert all(e["date"] <= e["available_date"] <= s["as_of"] for s in replay["snapshots"] for e in s["events"])


def test_spcx_rejects_old_etf_wrong_class_and_cusip():
    spcx = next(s for s in EQUITIES.instruments if s.symbol == "SPCX")
    for info in ({"symbol": "SPCX", "quoteType": "ETF", "longName": "Tuttle SPAC and New Issue ETF"},
                 {"symbol": "SPCX", "quoteType": "EQUITY", "longName": "SpaceX ETF"},
                 {"symbol": "SPCX", "quoteType": "EQUITY", "longName": "SpaceX", "cusip": "wrong"}):
        with pytest.raises(ValueError, match="identity"):
            verify_identity(spcx, info)
    accepted = verify_identity(spcx, {"symbol": "SPCX", "quoteType": "EQUITY", "longName": "Space Exploration Technologies Corp."})
    assert accepted["listed_since"] == "2026-06-12"
    assert accepted["provider_cusip"] is None  # do not claim a provider identifier was verified


def test_equity_gaps_use_observed_sessions_and_listing_boundary():
    benchmark = bars(12).iloc[[0, 1, 4, 5, 6, 7, 8]]
    complete = benchmark.copy()
    check_equity_sessions(complete, benchmark, Instrument("TEST", "Test", "fixture"), "2025-01-01")
    with pytest.raises(ValueError, match="missing 1"):
        check_equity_sessions(complete.drop(complete.index[2]), benchmark, Instrument("TEST", "Test", "fixture"), "2025-01-01")
    check_equity_sessions(complete.iloc[3:], benchmark, Instrument("NEW", "New", "fixture", "2025-01-06"), "2025-01-01")


def test_exact_cache_request_and_metadata_survive_without_downloading(tmp_path, monkeypatch):
    import exchange_calendars as xcals
    dates = xcals.get_calendar("XNYS").sessions_in_range("2025-01-01", "2025-03-31")
    frame = bars(len(dates))
    frame.index = dates
    frame.attrs.update(source="yahoo", symbol="AAPL", timeframe="1d", market_type="cash", price_currency="USD", volume_unit="shares", adjusted=True)
    monkeypatch.setattr("research.market_study.fetch_equity", lambda *args: frame.copy())
    item = EQUITIES.instruments[0]
    cached = load_data(EQUITIES, item, "2025-01-01", "2025-04-01", tmp_path)
    monkeypatch.setattr("research.market_study.fetch_equity", lambda *args: pytest.fail("cache unexpectedly downloads"))
    reread = load_data(EQUITIES, item, "2025-01-01", "2025-04-01", tmp_path)
    pd.testing.assert_frame_equal(cached, reread, check_freq=False)
    assert reread.attrs["source"] == "yahoo"
    meta = json.loads((tmp_path / "AAPL.json").read_text())
    assert meta["request"]["profile"] == "us_equities"


def test_crypto_cache_refuses_missing_days_and_wrong_market_metadata(tmp_path):
    frame = bars(4)
    instrument = CRYPTO.instruments[0]
    attrs = {"symbol": "BTCUSDT", "source": "binance_spot", "timeframe": "1d", "market_type": "spot",
             "price_currency": "USDT", "volume_unit": "BTC", "adjusted": False}
    request = {"profile": CRYPTO.id, "symbol": "BTCUSDT", "start": "2025-01-01", "end": "2025-01-05"}
    frame.drop(frame.index[1]).to_csv(tmp_path / "BTCUSDT.csv", index_label="Date")
    (tmp_path / "BTCUSDT.json").write_text(json.dumps({"request": request, "attrs": attrs}))
    with pytest.raises(ValueError, match="incomplete Binance"):
        load_data(CRYPTO, instrument, "2025-01-01", "2025-01-05", tmp_path)
    frame.to_csv(tmp_path / "BTCUSDT.csv", index_label="Date")
    attrs["market_type"] = "futures"
    (tmp_path / "BTCUSDT.json").write_text(json.dumps({"request": request, "attrs": attrs}))
    with pytest.raises(ValueError, match="market_type"):
        load_data(CRYPTO, instrument, "2025-01-01", "2025-01-05", tmp_path)


def test_calendar_rejects_common_missing_session_and_weekend_bar_and_adjustment():
    frame = bars(2)
    frame.index = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
    frame.attrs.update(source="yahoo", symbol="AAPL", timeframe="1d", market_type="cash",
                       price_currency="USD", volume_unit="shares", adjusted=True)
    instrument = EQUITIES.instruments[0]
    # Holiday start and weekend end are legitimate request boundaries.
    validate_profile_data(frame, EQUITIES, instrument, "2025-01-01", "2025-01-05")
    with pytest.raises(ValueError, match="session coverage"):
        validate_profile_data(frame.iloc[:1], EQUITIES, instrument, "2025-01-01", "2025-01-05")
    extra = pd.concat([frame, frame.iloc[-1:].rename(index={pd.Timestamp("2025-01-03"): pd.Timestamp("2025-01-04")})])
    extra.attrs = frame.attrs.copy()
    with pytest.raises(ValueError, match="session coverage"):
        validate_profile_data(extra, EQUITIES, instrument, "2025-01-01", "2025-01-05")
    frame.attrs["adjusted"] = False
    with pytest.raises(ValueError, match="adjusted"):
        validate_profile_data(frame, EQUITIES, instrument, "2025-01-01", "2025-01-05")


def test_separate_profiles_failed_and_short_history_are_visible():
    assert len(EQUITIES.instruments) <= 20
    assert {s.symbol for s in CRYPTO.instruments} == {"BTCUSDT", "ETHUSDT", "DOGEUSDT"}
    assert EQUITIES.config is not CRYPTO.config
    tiny = replace(CRYPTO, instruments=CRYPTO.instruments[:1])
    item = build_replay(bars(30), tiny.instruments[0], tiny, replay_bars=126)
    assert len(item["snapshots"]) == 10
    assert item["data"]["history_coverage"] == "short"
    report = {"profile": {"label": "fixture"}, "window": {"start": "a", "end": "b"},
              "summary": {"success": 1, "failed": 1, "benchmark_status": "failed"},
              "instruments": [item, {"symbol": "FAIL", "status": "failed", "error": "bad data"}], "limitations": []}
    text = markdown_summary(report)
    assert "FAILED: bad data" in text and "规则案例" in text
