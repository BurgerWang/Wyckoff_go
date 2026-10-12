"""Compatibility entrypoint for the modular, descriptive Wyckoff core.

Production stages use OHLCV lifecycle evidence. ``detect_tr`` remains a
P&F-only congestion helper for callers, not the core range model.
"""
from market_data import validate_ohlcv
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.engine import analyze, report
from wyckoff_core.counts import wyckoff_count
from wyckoff_core.stages import classify_phase, sequence_state
from wyckoff_core.primitives import detect_tr, _lps_lpsy


def detect_events(df, tr=None, lookback=None, *, config=None):
    """Events known at this snapshot; optional lookback only trims output.

    Indicator warmup and range formation always see the full supplied prefix.
    An explicit range is caller evidence; normal analyze discovers it.
    """
    from wyckoff_core.features import compute_features, confirmed_swings
    from wyckoff_core.structure import replay_structure
    df = validate_ohlcv(df)
    cfg = config or CoreConfig()
    result = replay_structure(df, compute_features(df, cfg), confirmed_swings(df, cfg), cfg, tr)
    events = result["events"]
    if lookback is not None:
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        events = [e for e in events if e["date"] >= df.index[max(0, len(df) - lookback)]]
    return events


__all__ = ["CoreConfig", "TR", "analyze", "report", "detect_tr", "detect_events",
           "classify_phase", "sequence_state", "wyckoff_count"]
