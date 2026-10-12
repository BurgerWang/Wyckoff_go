"""OHLCV contracts and explicit I/O, shared by analysis and research.

All analysis uses chronological, unique bars. Invalid observations are
rejected, rather than silently dropped or reordered into a plausible chart.
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("Open", "High", "Low", "Close")


def validate_ohlcv(df: pd.DataFrame, require_volume: bool = True) -> pd.DataFrame:
    """Validate and copy a single-instrument OHLC[V] frame; preserve attrs."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("OHLCV data must be a non-empty DataFrame")
    if isinstance(df.columns, pd.MultiIndex) or not df.columns.is_unique:
        raise ValueError("OHLCV columns must be unique and flat")
    required = (*PRICE_COLUMNS, "Volume") if require_volume else PRICE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("OHLCV index must be a valid DatetimeIndex")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV dates must be unique and strictly increasing")
    result = df.copy()
    numeric = [c for c in (*PRICE_COLUMNS, "Volume") if c in result.columns]
    try:
        result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("OHLCV values must be numeric") from exc
    finite_rows = np.isfinite(result[numeric].to_numpy()).all(axis=1)
    if not finite_rows.all():
        dates = ", ".join(str(date) for date in result.index[~finite_rows][:5])
        raise ValueError(f"OHLCV values must be finite; missing bars require explicit repair (dates: {dates})")
    if (result[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if "Volume" in result and (result["Volume"] < 0).any():
        raise ValueError("Volume must be non-negative")
    if ((result["High"] < result[list(PRICE_COLUMNS)].max(axis=1)) |
            (result["Low"] > result[list(PRICE_COLUMNS)].min(axis=1))).any():
        raise ValueError("OHLC values violate Low <= Open/Close <= High")
    return result


def normalize_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Select exactly one Yahoo instrument, even with either MultiIndex order."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"no market data returned for {symbol}")
    result = raw.copy()
    if isinstance(result.columns, pd.MultiIndex):
        if result.columns.nlevels != 2:
            raise ValueError("unsupported market-data column layout")
        price_level = next((i for i in range(2)
                            if set(PRICE_COLUMNS).issubset(result.columns.get_level_values(i))), None)
        if price_level is None:
            raise ValueError("market-data response has no OHLC fields")
        ticker_level = 1 - price_level
        tickers = result.columns.get_level_values(ticker_level).unique()
        if len(tickers) != 1 or str(tickers[0]).upper() != symbol.upper():
            raise ValueError("market-data response does not contain exactly the requested instrument")
        result = result.xs(tickers[0], axis=1, level=ticker_level)
    result.attrs.update(symbol=symbol, source="yahoo", adjusted=True)
    # Yahoo sometimes appends an empty daily placeholder: all prices absent
    # and zero trades. It is not a price observation. Remove only a trailing
    # run, and expose its dates; interior gaps or partially missing bars fail.
    if set((*PRICE_COLUMNS, "Volume")).issubset(result.columns):
        empty = result[list(PRICE_COLUMNS)].isna().all(axis=1) & result["Volume"].eq(0)
        cutoff = len(result)
        while cutoff and empty.iloc[cutoff - 1]:
            cutoff -= 1
        if cutoff < len(result):
            excluded = [str(date) for date in result.index[cutoff:]]
            result = result.iloc[:cutoff].copy()
            result.attrs.update(excluded_empty_dates=excluded,
                                warnings=[f"Yahoo 尾部有 {len(excluded)} 个无价格且零成交量的占位 bar，已排除: {', '.join(excluded)}"])
    return validate_ohlcv(result)


def fetch(symbol: str, years: float = 2, tf: str = "1d") -> pd.DataFrame:
    """Explicit adjusted Yahoo download. 4h bars are UTC calendar bins.

    For exchange-session-specific intraday bars, provide a preaggregated
    CSV instead. Yahoo hourly history is capped at 730 days.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be non-empty")
    if not isinstance(years, (int, float)) or not math.isfinite(years) or years <= 0:
        raise ValueError("years must be a positive finite number")
    if tf not in ("1d", "4h"):
        raise ValueError("timeframe must be '1d' or '4h'")
    symbol = symbol.strip()
    import yfinance as yf
    end = pd.Timestamp.now(tz="UTC")
    days = max(1, math.ceil(years * 365.25))
    if tf == "4h":
        start = end - pd.Timedelta(days=min(days, 730))
        raw = yf.download(symbol, start=start, end=end, interval="1h",
                          progress=False, auto_adjust=True)
        result = normalize_download(raw, symbol)
        if result.index.tz is None:
            raise ValueError("hourly source must include timezone information")
        result = result.tz_convert("UTC")
        source_completion = pd.Series(result.index + pd.Timedelta(hours=1), index=result.index).resample(
            "4h", closed="left", label="right").max()
        offsets = pd.Series(result.index - result.index.floor("h"), index=result.index).resample(
            "4h", closed="left", label="right").max()
        completion_times = {label: max(label + offsets.loc[label], completed)
                            for label, completed in source_completion.items() if pd.notna(completed)}
        # Hourly timestamps denote bar STARTS, including half-hour session
        # openings. Drop the still-open source bar before aggregating.
        result = result.loc[result.index + pd.Timedelta(hours=1) <= end]
        attrs = dict(result.attrs)
        # Label on the right: a bar is available only when its bin closes.
        result = result.resample("4h", closed="left", label="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        result = result.dropna(subset=list(PRICE_COLUMNS))
        # Calendar bins classify hourly starts; their label must never
        # precede completion of the last included source bar (e.g. 15:30).
        # Compute availability from the RAW bin, before removing its open
        # hour: otherwise a half-hour source can publish a partial bin early.
        result.index = pd.DatetimeIndex([completion_times[label] for label in result.index])
        result = result.loc[result.index <= end]
        result.attrs.update(attrs, timeframe="4h", aggregation="UTC calendar bins")
        return validate_ohlcv(result)
    start = end - pd.Timedelta(days=days)
    raw = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), interval="1d",
                      progress=False, auto_adjust=True)
    result = normalize_download(raw, symbol)
    result.attrs["timeframe"] = "1d"
    return result


def read_csv(path: str | Path, symbol: str = "") -> pd.DataFrame:
    """Read a CSV containing Date (or Datetime) and canonical OHLCV columns."""
    # Preserve the saved IEEE float values when replaying a downloaded CSV.
    # The fast default parser can round a last bit differently, changing
    # boundary comparisons or the numerical evidence in a reproduced snapshot.
    raw = pd.read_csv(path, float_precision="round_trip")
    date_column = next((c for c in ("Date", "Datetime") if c in raw), None)
    if date_column is None:
        raise ValueError("CSV requires a Date or Datetime column")
    try:
        dates = pd.to_datetime(raw.pop(date_column), errors="raise")
        raw.index = pd.DatetimeIndex(dates)
    except (ValueError, TypeError) as exc:
        raise ValueError("CSV dates must have a consistent timezone and valid format") from exc
    raw.attrs.update(symbol=symbol, source=str(Path(path)), adjusted="unspecified")
    return validate_ohlcv(raw)
