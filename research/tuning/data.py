"""Reuse verified source history, preserving market and segment boundaries."""
import json
from pathlib import Path

from market_data import read_csv
from research.history_data import coverage
from research.universe import PROFILES
from research.tuning.profiles import instrument_for
from wyckoff_core.timeline import stage_timeline


def load_history(history, market, symbol):
    instrument = instrument_for(market, symbol)
    history = Path(history)
    manifest = json.loads((history / "manifest.json").read_text())
    item = manifest.get("profiles", {}).get(market, {}).get(symbol, {})
    if item.get("status") not in ("complete", "short_history", "partial_missing_sessions") or not item.get("csv"):
        raise ValueError("missing or invalid source history")
    csv = Path(item["csv"])
    if not csv.is_absolute():
        # Existing manifests contain project-root-relative paths. Prefer the
        # relocated history's market/data location, independent of caller cwd.
        candidate = history / market / "data" / csv.name
        csv = candidate if candidate.exists() else history / csv
    frame = read_csv(csv, symbol)
    frame.attrs = item["attrs"]
    frame.index = frame.index.as_unit("ns")
    profile = PROFILES[market]
    expected = {"symbol": symbol, "timeframe": "1d", "market_type": profile.market_type,
                "price_currency": profile.price_currency, "volume_unit": "shares" if market == "us_equities" else symbol.removesuffix("USDT")}
    for key, value in expected.items():
        if frame.attrs.get(key) != value:
            raise ValueError(f"source identity mismatch: {key}")
    verified = coverage(frame, profile, instrument, **{k: manifest["request"][k] for k in ("start", "end")})
    if (verified["continuous_segments"] != item["continuous_segments"] or len(frame) != item["bars"]
            or verified["missing_dates"] != item["missing_dates"]):
        raise ValueError("history changed since verified manifest")
    return frame, item


def replay(frame, item, config, end=None):
    rows = []
    for segment in item["continuous_segments"]:
        observed = frame.loc[segment["start"]:segment["end"]]
        if end is not None:
            observed = observed.loc[:end]
        if len(observed):
            rows.extend(stage_timeline(observed, config)["rows"])
    return rows
