"""Causal equivalence to the real snapshot API, including pivot plateaus."""
import numpy as np
import pandas as pd
import pytest

from analysis import analyze_frame
from wyckoff_core.models import CoreConfig
from wyckoff_core.timeline import stage_timeline


def frame(seed=2, n=110):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.5, n))
    spread = rng.uniform(0.5, 2, n)
    result = pd.DataFrame({"Open": close, "High": close + spread, "Low": close - spread,
                           "Close": close, "Volume": rng.integers(50, 500, n)},
                          index=pd.date_range("2020-01-01", periods=n))
    result.iloc[25:29, result.columns.get_loc("High")] = result.High.iloc[25:29].max() + 3
    result.iloc[60, result.columns.get_loc("Volume")] = 0
    return result


@pytest.mark.parametrize("seed", [2, 12])
def test_every_phase_matches_independent_prefix_snapshot(seed):
    df = frame(seed)
    cfg = CoreConfig(baseline_window=5, min_range_bars=5)
    timeline = stage_timeline(df, cfg)
    assert len(timeline["rows"]) == len(df)
    for i, row in enumerate(timeline["rows"]):
        snapshot = analyze_frame(df.iloc[:i + 1], config=cfg)["analyses"]["atr"]["wyckoff"]
        assert row["phase"] == snapshot["phase"], (i, row, snapshot["phase"])


def test_future_extreme_cannot_rewrite_prior_rows_or_features():
    df = frame()
    cfg = CoreConfig(baseline_window=5, min_range_bars=5)
    first = stage_timeline(df.iloc[:70], cfg)
    later = df.copy()
    later.iloc[70:, :4] *= 4
    later.iloc[70:, 4] *= 100
    assert stage_timeline(later, cfg)["rows"][:70] == first["rows"]


def test_config_validation():
    with pytest.raises(ValueError, match="CoreConfig"):
        stage_timeline(frame(), {})
