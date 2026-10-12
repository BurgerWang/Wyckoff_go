"""Causal OHLCV features and confirmed price pivots.

These are observations, not measurements of signed order flow or ownership.
The tested bar never contributes to its own volume/spread baseline.
"""
import numpy as np
import pandas as pd

from market_data import validate_ohlcv
from .models import CoreConfig


def compute_features(df, config=None):
    """Return features on the input index without mutating the input."""
    config = config or CoreConfig()
    df = validate_ohlcv(df)
    spread = df["High"] - df["Low"]
    previous_close = df["Close"].shift(1)
    true_range = pd.concat([spread, (df["High"] - previous_close).abs(),
                            (df["Low"] - previous_close).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / config.core_atr_period, adjust=False).mean()
    volume_base = df["Volume"].rolling(config.baseline_window,
                                      min_periods=config.baseline_window).mean().shift(1)
    spread_base = spread.rolling(config.baseline_window,
                                 min_periods=config.baseline_window).mean().shift(1)
    out = pd.DataFrame(index=df.index)
    out["true_range"] = true_range
    out["true_range_baseline"] = true_range.rolling(config.baseline_window,
        min_periods=config.baseline_window).mean().shift(1)
    out["true_range_ratio"] = true_range / out["true_range_baseline"].where(out["true_range_baseline"] > 0)
    out["atr"] = atr
    out["volume_baseline"] = volume_base
    out["spread_baseline"] = spread_base
    out["volume_ratio"] = df["Volume"] / volume_base.where(volume_base > 0)
    out["spread_ratio"] = spread / spread_base.where(spread_base > 0)
    out["close_position"] = ((df["Close"] - df["Low"]) / spread.where(spread > 0)).fillna(0.5)
    out["price_change_atr"] = df["Close"].diff() / atr.shift(1).where(atr.shift(1) > 0)
    # Absence of a detected range is a normal negative only when the input
    # could have supplied its evidence. Zero-volume bars remain unavailable,
    # including within the baseline used to judge today's observation.
    volume_observed = (df["Volume"] > 0).rolling(config.baseline_window + 1,
        min_periods=config.baseline_window + 1).min().eq(1)
    observation_fields = ["atr", "volume_baseline", "spread_baseline", "true_range_baseline",
                          "volume_ratio", "spread_ratio", "true_range_ratio", "close_position"]
    out["observation_eligible"] = (volume_observed & np.isfinite(out[observation_fields]).all(axis=1)
        & out[["atr", "volume_baseline", "spread_baseline", "true_range_baseline"]].gt(0).all(axis=1))
    out["effort_result"] = "unknown"
    valid = out[["volume_ratio", "spread_ratio", "price_change_atr"]].notna().all(axis=1)
    high_effort = out["volume_ratio"] >= config.volume_high
    large_result = out["price_change_atr"].abs() >= 0.75
    little_result = (out["price_change_atr"].abs() <= 0.25) & (out["spread_ratio"] <= 1)
    out.loc[valid & high_effort & large_result, "effort_result"] = "effort_result_harmony"
    out.loc[valid & high_effort & little_result, "effort_result"] = "high_effort_little_result"
    out.loc[valid & (out["volume_ratio"] <= config.volume_low), "effort_result"] = "reduced_effort"
    out.attrs["threshold_origin"] = "engineering_heuristics"
    return out


def confirmed_swings(df, config=None):
    """Pivots observable only after ``swing_confirmation`` later bars.

    A contiguous flat extreme has one pivot at its last bar; confirmation
    waits for strictly lower highs/higher lows. Already confirmed pivots are
    retained even when another same-kind extreme is subsequently observed.
    """
    config = config or CoreConfig()
    df = validate_ohlcv(df, require_volume=False)
    span = config.swing_confirmation
    result = []
    for kind, column in (("high", "High"), ("low", "Low")):
        values = df[column].to_numpy()
        start = 0
        while start < len(values):
            end = start
            while end + 1 < len(values) and values[end + 1] == values[start]:
                end += 1
            if start >= span and end + span < len(values):
                before = values[start - span:start]
                after = values[end + 1:end + span + 1]
                price = values[end]
                is_extreme = ((price > before.max() and price > after.max()) if kind == "high"
                              else (price < before.min() and price < after.min()))
                if is_extreme:
                    result.append({"kind": kind, "date": df.index[end],
                                   "available_date": df.index[end + span], "price": float(price)})
            start = end + 1
    return sorted(result, key=lambda pivot: (pivot["available_date"], pivot["date"], pivot["kind"]))
