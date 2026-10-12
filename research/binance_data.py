"""Public Binance spot UTC daily bars, with explicit coverage and provenance.

REST contract: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
Archive contract: https://github.com/binance/binance-public-data
No credentials, trading endpoints, imputation, or silent data repairs.
"""
import csv
from datetime import date
from io import BytesIO, StringIO
import json
import math
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from market_data import validate_ohlcv

REST_HOST = "https://data-api.binance.vision"
ARCHIVE_HOST = "https://data.binance.vision/data/spot"
REQUEST_TIMEOUT = 25
SUPPORTED_PAIRS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT")
DAY = pd.Timedelta(days=1)
COLUMNS = ["Open", "High", "Low", "Close", "Volume", "QuoteVolume"]


def _download(url):
    with urlopen(Request(url, headers={"User-Agent": "wyckoff-research/0.3"}),
                 timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def _date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("start/end must be ISO dates YYYY-MM-DD")
    return pd.Timestamp(value, tz="UTC")


def _timestamp(value):
    # Official spot archives switched to microseconds on 2025-01-01.
    # Parse integer strings directly: float conversion can lose microseconds.
    number = int(value)
    if str(number) != str(value):
        raise ValueError("kline timestamp must be an integer")
    return pd.Timestamp(number, unit="us" if number >= 100_000_000_000_000 else "ms",
                        tz="UTC")


def _parse_row(row):
    if not isinstance(row, (list, tuple)) or len(row) != 12:
        raise ValueError("expected Binance 12-field kline row")
    opened, closed = _timestamp(row[0]), _timestamp(row[6])
    if opened != opened.normalize():
        raise ValueError("kline is not aligned to UTC midnight")
    unit = pd.Timedelta(microseconds=1) if int(row[0]) >= 100_000_000_000_000 else pd.Timedelta(milliseconds=1)
    if closed != opened + DAY - unit:
        raise ValueError("kline close time is not a complete UTC 1d interval")
    values = [float(row[i]) for i in (1, 2, 3, 4, 5, 7)]
    if not all(math.isfinite(v) for v in values) or values[-1] < 0:
        raise ValueError("invalid non-finite or negative quote volume/OHLCV")
    return opened, values


def _rest(symbol, start, end, urls):
    rows = []
    cursor = start
    while cursor < end:
        url = REST_HOST + "/api/v3/klines?" + urlencode({
            "symbol": symbol, "interval": "1d", "timeZone": "0", "limit": 1000,
            "startTime": cursor.value // 1_000_000,
            "endTime": end.value // 1_000_000 - 1,
        })
        page = json.loads(_download(url))
        urls.append(url)
        if not isinstance(page, list) or len(page) > 1000:
            raise ValueError("invalid Binance kline response")
        if not page:
            break
        parsed = [_parse_row(row) for row in page]
        times = [item[0] for item in parsed]
        if any(t < cursor or t >= end for t in times):
            raise ValueError("Binance returned bars outside requested page bounds")
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("duplicate or unordered Binance UTC days")
        rows.extend(parsed)
        cursor = times[-1] + DAY
    return rows


def _archive_file(symbol, frequency, period, urls):
    name = f"{symbol}-1d-{period}"
    url = f"{ARCHIVE_HOST}/{frequency}/klines/{symbol}/1d/{name}.zip"
    raw = _download(url)
    with ZipFile(BytesIO(raw)) as archive:
        if archive.namelist() != [name + ".csv"]:
            raise ValueError("Binance archive does not contain exactly the requested pair/file")
        content = archive.read(name + ".csv").decode("utf-8")
    urls.append(url)
    parsed = [_parse_row(row) for row in csv.reader(StringIO(content))]
    times = [item[0] for item in parsed]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("duplicate or unordered Binance archive UTC days")
    for timestamp in times:
        if timestamp.strftime("%Y-%m" if frequency == "monthly" else "%Y-%m-%d") != period:
            raise ValueError("Binance archive contains dates outside its period")
    return parsed


def _archives(symbol, start, end, urls):
    rows = []
    month = start.replace(day=1)
    while month < end:
        next_month = month + pd.offsets.MonthBegin(1)
        left, right = max(start, month), min(end, next_month)
        try:
            parsed = _archive_file(symbol, "monthly", month.strftime("%Y-%m"), urls)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            # Current/recent month may not have been published yet.
            parsed = []
            for day in pd.date_range(left, right, inclusive="left", freq="D"):
                parsed.extend(_archive_file(symbol, "daily", day.strftime("%Y-%m-%d"), urls))
        rows.extend(item for item in parsed if left <= item[0] < right)
        month = next_month
    return rows


def fetch_spot_daily(symbol, start, end, *, now=None) -> pd.DataFrame:
    """Fetch [start, end) UTC spot days; exclude today's unfinished candle.

    Restricts pairs to the authorized study. Missing days (including a request
    before listing) raise rather than implying complete coverage. HTTP 403/451
    on REST uses official monthly archives, or daily archives when monthly
    files are unpublished. Other network errors propagate for diagnosis.
    ``now`` is an optional clock for reproducible historical cutoffs.
    """
    if symbol not in SUPPORTED_PAIRS:
        raise ValueError(f"unsupported spot pair: {symbol}; expected {SUPPORTED_PAIRS}")
    beginning, requested_end = _date(start), _date(end)
    if beginning >= requested_end:
        raise ValueError("start must precede end")
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if pd.isna(clock):
        raise ValueError("now must be a valid timestamp")
    clock = clock.tz_localize("UTC") if clock.tzinfo is None else clock.tz_convert("UTC")
    completed_end = min(requested_end, clock.normalize())
    urls, rows = [], []
    transport, fallback_reason = "rest", None
    if beginning < completed_end:
        try:
            rows = _rest(symbol, beginning, completed_end, urls)
        except HTTPError as exc:
            if exc.code not in (403, 451):
                raise
            fallback_reason = f"REST HTTP {exc.code}"
            transport = "binance_vision_archive"
            # Do not blend partial REST rows with archive duplicates.
            urls = []
            rows = _archives(symbol, beginning, completed_end, urls)
    index = pd.DatetimeIndex([item[0] for item in rows], tz="UTC").tz_localize(None)
    index.name = "Date"
    result = pd.DataFrame([item[1] for item in rows], index=index, columns=COLUMNS, dtype=float)
    expected = pd.date_range(beginning.tz_localize(None), completed_end.tz_localize(None),
                             inclusive="left", freq="D") if beginning < completed_end else pd.DatetimeIndex([])
    if not result.index.is_unique or not result.index.is_monotonic_increasing:
        raise ValueError("duplicate or unordered Binance UTC days")
    missing = expected.difference(result.index)
    if len(missing) or len(result.index.difference(expected)):
        raise ValueError(f"incomplete Binance UTC daily coverage for {symbol}: missing "
                         f"{len(missing)} days ({', '.join(missing.strftime('%Y-%m-%d')[:5])})")
    if not result.empty:
        result = validate_ohlcv(result)
    result.attrs.update(
        symbol=symbol, source="binance_spot", timeframe="1d", market_type="spot",
        price_currency="USDT", volume_unit=symbol[:-4], quote_volume_unit="USDT",
        start=start, end=end, completed_end=completed_end.strftime("%Y-%m-%d"),
        transport=transport, source_urls=urls, fallback_reason=fallback_reason,
        retrieved_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        as_of_utc=clock.isoformat(), adjusted=False,
        excluded_unfinished_from=clock.normalize().strftime("%Y-%m-%d"),
        identity_verification="REST request symbol / archive path and exact member name",
    )
    return result
