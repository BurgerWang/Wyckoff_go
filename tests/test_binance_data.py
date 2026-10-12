"""Behavior checks with synthetic responses; not live market-data evidence."""
import csv
from io import BytesIO, StringIO
import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

import pandas as pd
import pytest

from research import binance_data as bd


def kline(day, *, microseconds=False):
    stamp = pd.Timestamp(day, tz="UTC")
    divisor = 1000 if microseconds else 1_000_000
    return [stamp.value // divisor, "10", "12", "8", "11", "100",
            (stamp + bd.DAY).value // divisor - 1, "1100", 4, "50", "550", "0"]


def zipped(name, rows):
    csv_text = StringIO()
    csv.writer(csv_text).writerows(rows)
    out = BytesIO()
    with ZipFile(out, "w") as archive:
        archive.writestr(name + ".csv", csv_text.getvalue())
    return out.getvalue()


def test_rest_pagination_bounds_and_provenance(monkeypatch):
    start, end = "2021-01-01", "2023-10-01"
    days = pd.date_range(start, end, inclusive="left")
    calls = []

    def download(url):
        params = parse_qs(urlparse(url).query)
        calls.append(params)
        assert params["symbol"] == ["BTCUSDT"]
        assert params["limit"] == ["1000"]
        assert params["timeZone"] == ["0"]
        cursor = pd.Timestamp(int(params["startTime"][0]), unit="ms")
        selected = days[days >= cursor][:1000]
        return json.dumps([kline(day.isoformat()) for day in selected]).encode()

    monkeypatch.setattr(bd, "_download", download)
    frame = bd.fetch_spot_daily("BTCUSDT", start, end, now="2024-01-01")
    assert len(frame) == len(days) == 1003
    assert len(calls) == 2
    assert int(calls[1]["startTime"][0]) == pd.Timestamp(days[1000]).value // 1_000_000
    assert int(calls[0]["endTime"][0]) == pd.Timestamp(end).value // 1_000_000 - 1
    assert frame.index.tz is None
    assert frame.attrs["source"] == "binance_spot"
    assert frame.attrs["volume_unit"] == "BTC"
    assert frame.attrs["start"] == start and frame.attrs["end"] == end
    assert frame["Volume"].iloc[0] == 100
    assert frame["QuoteVolume"].iloc[0] == 1100


def test_unfinished_day_is_not_requested(monkeypatch):
    calls = []

    def download(url):
        calls.append(parse_qs(urlparse(url).query))
        return json.dumps([kline("2025-01-01")]).encode()

    monkeypatch.setattr(bd, "_download", download)
    frame = bd.fetch_spot_daily("ETHUSDT", "2025-01-01", "2025-01-05",
                                now="2025-01-02T17:00:00+07:00")
    assert list(frame.index) == [pd.Timestamp("2025-01-01")]
    assert frame.attrs["completed_end"] == "2025-01-02"
    assert int(calls[0]["endTime"][0]) == pd.Timestamp("2025-01-02").value // 1_000_000 - 1
    empty = bd.fetch_spot_daily("ETHUSDT", "2025-01-02", "2025-01-05",
                               now="2025-01-02T17:00Z")
    assert empty.empty and len(calls) == 1


@pytest.mark.parametrize("code", [403, 451])
def test_archive_microsecond_fallback_and_identity(monkeypatch, code):
    def download(url):
        if "/api/v3/" in url:
            raise HTTPError(url, code, "restricted", {}, None)
        assert "/spot/monthly/klines/DOGEUSDT/1d/" in url
        return zipped("DOGEUSDT-1d-2025-01", [kline("2025-01-01", microseconds=True),
                                               kline("2025-01-02", microseconds=True)])

    monkeypatch.setattr(bd, "_download", download)
    frame = bd.fetch_spot_daily("DOGEUSDT", "2025-01-01", "2025-01-03",
                                now="2025-02-01")
    assert frame.index[-1] == pd.Timestamp("2025-01-02")
    assert frame.attrs["transport"] == "binance_vision_archive"
    assert frame.attrs["fallback_reason"] == f"REST HTTP {code}"
    assert len(frame.attrs["source_urls"]) == 1


def test_daily_archive_when_monthly_unpublished(monkeypatch):
    def download(url):
        if "/api/v3/" in url:
            raise HTTPError(url, 451, "restricted", {}, None)
        if "/monthly/" in url:
            raise HTTPError(url, 404, "unpublished", {}, None)
        name = url.rsplit("/", 1)[-1][:-4]
        day = name.removeprefix("BTCUSDT-1d-")
        return zipped(name, [kline(day, microseconds=True)])

    monkeypatch.setattr(bd, "_download", download)
    frame = bd.fetch_spot_daily("BTCUSDT", "2025-01-01", "2025-01-03", now="2025-01-03")
    assert len(frame) == 2
    assert all("/daily/" in url for url in frame.attrs["source_urls"])


@pytest.mark.parametrize("kind", ["duplicate", "missing", "bad_high", "nan", "negative_volume",
                                  "close_time", "outside_bounds", "out_of_order"])
def test_bad_rows_fail_without_fallback(monkeypatch, kind):
    rows = [kline("2025-01-01"), kline("2025-01-02")]
    if kind == "duplicate":
        rows[1] = rows[0]
    elif kind == "missing":
        rows = rows[1:]
    elif kind == "bad_high":
        rows[0][2] = "9"
    elif kind == "nan":
        rows[0][4] = "NaN"
    elif kind == "negative_volume":
        rows[0][5] = "-1"
    elif kind == "close_time":
        rows[0][6] -= 1000
    elif kind == "outside_bounds":
        rows.append(kline("2025-01-03"))
    else:
        rows.reverse()
    calls = []

    def download(url):
        calls.append(url)
        return json.dumps(rows).encode()

    monkeypatch.setattr(bd, "_download", download)
    with pytest.raises(ValueError):
        bd.fetch_spot_daily("BTCUSDT", "2025-01-01", "2025-01-03", now="2025-02-01")
    assert all("/api/v3/" in url for url in calls)


def test_nonrequested_archive_pair_is_rejected(monkeypatch):
    def download(url):
        if "/api/v3/" in url:
            raise HTTPError(url, 403, "restricted", {}, None)
        return zipped("ETHUSDT-1d-2025-01", [kline("2025-01-01", microseconds=True)])

    monkeypatch.setattr(bd, "_download", download)
    with pytest.raises(ValueError, match="requested pair"):
        bd.fetch_spot_daily("BTCUSDT", "2025-01-01", "2025-01-02", now="2025-02-01")


@pytest.mark.parametrize("symbol,start,end", [("BTCBUSD", "2025-01-01", "2025-01-02"),
                                            ("BTCUSDT", "20250101", "2025-01-02"),
                                            ("BTCUSDT", "2025-01-02", "2025-01-01")])
def test_inputs_rejected_without_network(monkeypatch, symbol, start, end):
    monkeypatch.setattr(bd, "_download", lambda url: pytest.fail("unexpected network request"))
    with pytest.raises(ValueError):
        bd.fetch_spot_daily(symbol, start, end)


def test_rate_limit_does_not_trigger_archive_fallback(monkeypatch):
    def download(url):
        raise HTTPError(url, 429, "rate limit", {}, None)

    monkeypatch.setattr(bd, "_download", download)
    with pytest.raises(HTTPError) as error:
        bd.fetch_spot_daily("BTCUSDT", "2025-01-01", "2025-01-02", now="2025-02-01")
    assert error.value.code == 429
