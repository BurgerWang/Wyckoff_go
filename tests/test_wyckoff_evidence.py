"""Causality, missing-input and interpretation contracts for evidence layers."""
import numpy as np
import pandas as pd
import pytest

from wyckoff_core.context import analyze_context, analyze_laws
from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.readiness import evaluate_readiness


def bars(closes=None, n=50):
    closes = np.array([100.] * n if closes is None else closes, dtype=float)
    return pd.DataFrame({"Open": closes, "High": closes + 1, "Low": closes - 1,
                         "Close": closes, "Volume": 100.},
                        index=pd.date_range("2025-01-01", periods=len(closes)))


def pivot(df, i, kind, available=None):
    return {"kind": kind, "date": df.index[i], "available_date": df.index[i if available is None else available],
            "price": float(df["High" if kind == "high" else "Low"].iloc[i])}


def tr(df, **extra):
    return TR(0, 4, 110, 90, 5, 100, start_date=df.index[20],
              confirmed_date=df.index[25], range_id="test-range", **extra)


def test_features_true_range_gap_and_preceding_baselines_are_causal():
    df = bars()
    df.iloc[25] = [110, 111, 109, 110, 1000]
    original = df.copy()
    early = compute_features(df.iloc[:26])
    full = compute_features(df)
    pd.testing.assert_frame_equal(early, full.iloc[:26])
    pd.testing.assert_frame_equal(df, original)
    assert early.iloc[-1]["true_range"] == 11
    assert early.iloc[-1]["volume_ratio"] == 10
    assert early.iloc[-1]["spread_ratio"] == 1
    assert early.iloc[-1]["price_change_atr"] == 5


def test_flat_pivot_is_unique_delayed_and_existing_confirmation_persists():
    df = bars([100, 101, 102, 105, 105, 105, 102, 101, 100, 103, 104, 102, 100])
    config = CoreConfig(swing_confirmation=2)
    assert confirmed_swings(df.iloc[:7], config) == []
    early = confirmed_swings(df.iloc[:8], config)
    top = next(p for p in early if p["kind"] == "high")
    assert top["date"] == df.index[5] and top["available_date"] == df.index[7]
    assert sum(p["kind"] == "high" for p in early) == 1
    assert top in confirmed_swings(df, config)


def test_relative_strength_uses_both_corresponding_waves_and_ignores_future():
    df = bars([100, 101, 102, 104, 108, 110, 105, 101, 100, 101, 103, 107, 111, 115, 112, 109, 105, 104, 105, 107])
    features = compute_features(df)
    swings = [pivot(df, 2, "low", 3), pivot(df, 5, "high", 6), pivot(df, 8, "low", 9),
              pivot(df, 13, "high", 14), pivot(df, 17, "low", 18)]
    benchmark = pd.Series(np.linspace(100, 102, len(df)), index=df.index)
    # A flat benchmark makes stock rallies stronger and stock reactions weaker:
    # cumulative outperformance alone cannot satisfy both sides of the test.
    early = analyze_context(df, features, swings, benchmark)
    assert early["relative_strength"]["direction"] == "mixed"
    future = pd.Series([np.nan], index=[df.index[-1] + pd.Timedelta(days=1)])
    extended = analyze_context(df, features, swings, pd.concat([benchmark, future]))
    assert early == extended
    no_benchmark = analyze_context(df, features, swings)
    assert no_benchmark["relative_strength"]["status"] == "unavailable"


def test_no_forward_fill_and_no_implicit_intraday_or_timezone_alignment():
    df = bars([100, 102, 105, 102, 100, 104, 108, 104, 101, 104])
    config = CoreConfig(trend_window=3)
    swings = [pivot(df, 0, "low", 1), pivot(df, 2, "high", 3), pivot(df, 4, "low", 5), pivot(df, 6, "high", 7)]
    benchmark = pd.Series(np.arange(len(df)) + 100., index=df.index + pd.Timedelta(hours=1))
    context = analyze_context(df, compute_features(df), swings, benchmark, config=config)
    assert context["relative_strength"]["status"] == "unavailable"
    benchmark = pd.Series(np.arange(len(df)) + 100., index=df.index).drop(df.index[1])
    context = analyze_context(df, compute_features(df), swings, benchmark, config=config)
    assert len(context["relative_strength"]["rally_comparisons"]) == 1
    aware = pd.Series([100.] * len(df), index=df.index.tz_localize("UTC"))
    context = analyze_context(df, compute_features(df), swings, aware, config=config)
    assert context["relative_strength"]["status"] == "unavailable"
    df.attrs["timeframe"] = "1d"
    benchmark = pd.Series([100.] * len(df), index=df.index)
    benchmark.attrs["timeframe"] = "4h"
    context = analyze_context(df, compute_features(df), swings, benchmark, config=config)
    assert context["relative_strength"]["alignment"] == "incompatible_timeframes"


def test_explicit_daily_trading_dates_can_align_distinct_label_timezones():
    df = bars()
    df.attrs["timeframe"] = "1d"
    benchmark = pd.Series(np.linspace(100, 105, len(df)), index=df.index.tz_localize("America/New_York"))
    benchmark.attrs["timeframe"] = "1d"
    context = analyze_context(df, compute_features(df), [], benchmark)
    assert context["relative_strength"]["alignment"] == "same_explicit_daily_trading_date"
    assert context["relative_strength"]["overlap_bars"] == len(df)
    assert context["market_alignment"]["market_trend"] == "up"


def test_market_trend_needs_the_benchmark_on_every_bar_of_the_current_window():
    df = bars(np.linspace(100, 120, 50))
    features, config = compute_features(df), CoreConfig(trend_window=5)
    benchmark = pd.Series(np.linspace(100, 110, len(df)), index=df.index)

    def market(series, **kwargs):
        return analyze_context(df, features, [], series, config=config, **kwargs)["market_alignment"]

    current = market(benchmark)
    assert current["status"] == "available" and current["market_trend"] == "up"
    assert current["basis"] == "aligned_benchmark_window"
    assert current["window"] == {"start": df.index[-5], "end": df.index[-1], "bars": 5}
    # A gap older than the window does not matter.
    assert market(benchmark.drop(df.index[10]))["window"] == current["window"]
    # A benchmark that stops one bar early, or misses a bar inside the window, is not
    # stretched or stitched into a "recent" trend.
    for series, last_seen in ((benchmark.iloc[:-1], df.index[-2]), (benchmark.iloc[:-20], df.index[-21]),
                              (benchmark.drop(df.index[-3]), df.index[-1])):
        stale = market(series)
        assert stale["status"] == "unavailable" and stale["market_trend"] == "unknown"
        assert stale["aligned"] is None and "window" not in stale and "basis" not in stale
        assert any(str(last_seen) in text for text in stale["limitations"])
    # A timely caller description still applies, and is attributed to the caller.
    supplied = market(benchmark.iloc[:-20], market_context={"as_of": df.index[-1], "trend": "down"})
    assert (supplied["status"], supplied["market_trend"], supplied["basis"]) == (
        "available", "down", "caller_market_context")
    assert "window" not in supplied
    assert market(benchmark, market_context={"as_of": df.index[-1], "trend": "down"})["basis"] == "caller_market_context"


def test_relative_strength_is_not_judged_from_waves_the_benchmark_no_longer_covers():
    df = bars([100, 101, 102, 104, 108, 110, 105, 101, 100, 101, 103, 107, 111, 115, 112, 109, 105, 104, 105, 107])
    features = compute_features(df)
    swings = [pivot(df, 2, "low", 3), pivot(df, 5, "high", 6), pivot(df, 8, "low", 9),
              pivot(df, 13, "high", 14), pivot(df, 17, "low", 18)]
    # Rises less than the stock on both rallies and falls more on both reactions.
    benchmark = pd.Series(np.interp(range(len(df)), [0, 2, 5, 8, 13, 17, 19], [100, 100, 104, 92, 100, 88, 89]),
                          index=df.index)
    current = analyze_context(df, features, swings, benchmark)
    assert current["relative_strength"]["status"] == "available"
    assert current["relative_strength"]["direction"] == "stronger"
    assert evaluate_readiness(current, {}, [], None, None)["nine_buying"][6]["status"] == "met"
    # Stopping after the second rally leaves the latest reaction without a benchmark.
    stale = analyze_context(df, features, swings, benchmark.iloc[:15])
    relative = stale["relative_strength"]
    assert relative["status"] == "unavailable" and relative["direction"] == "unknown"
    assert relative["rally_outperformance"] is None and relative["reaction_resilience"] is None
    assert len(relative["rally_comparisons"]) == 2 and len(relative["reaction_comparisons"]) == 1
    assert any("回落" in text for text in relative["limitations"])
    readiness = evaluate_readiness(stale, {}, [], None, None)
    assert readiness["nine_buying"][6]["status"] == readiness["nine_selling"][3]["status"] == "unavailable"
    # An uncovered older wave is skipped as before when the latest of its kind is covered.
    older_gap = analyze_context(df, features, swings, benchmark.drop(df.index[3]))["relative_strength"]
    assert older_gap["status"] == "available" and len(older_gap["rally_comparisons"]) == 1


def test_future_market_context_does_not_claim_a_market_turn():
    df = bars()
    context = analyze_context(df, compute_features(df), [], market_context={
        "as_of": df.index[-1] + pd.Timedelta(days=1), "trend": "up", "turn_confirmed": True})
    assert context["market_alignment"]["status"] == "unavailable"
    assert context["market_alignment"]["turn_confirmed"] is None


def test_laws_require_real_volume_and_absorption_is_a_contextual_proxy():
    df = bars()
    df.iloc[-1] = [100, 100.5, 99.5, 100, 1000]
    range_ = TR(0, 4, 110, 99, 5, 104.5, confirmed_date=df.index[25])
    laws = analyze_laws(df, compute_features(df), [], range_)
    assert laws["effort_result"]["observations"][-1]["type"] == "supply_absorption_candidate"
    assert "CKS2014" in laws["supply_demand"]["source_ids"]
    assert any("无法测量" in value for value in laws["supply_demand"]["limitations"])
    no_context = analyze_laws(df, compute_features(df), [], None)
    assert no_context["effort_result"]["observations"][-1]["type"] == "high_effort_little_result"
    df["Volume"] = 0
    no_volume = analyze_laws(df, compute_features(df), [], range_)
    assert no_volume["supply_demand"]["direction"] == "unknown"
    assert no_volume["effort_result"]["status"] == "unavailable"


def test_effort_result_compares_same_direction_waves_as_well_as_individual_bars():
    closes = [110.] * 40
    closes[22:27] = [110, 107.5, 105, 102.5, 100]
    closes[27:32] = [103, 106, 109, 112, 114]
    closes[32:36] = [113, 112, 111, 110]
    df = bars(closes)
    df.loc[df.index[32:36], "Volume"] = 300
    config = CoreConfig(baseline_window=5)
    swings = [pivot(df, 22, "high", 23), pivot(df, 26, "low", 27),
              pivot(df, 31, "high", 32), pivot(df, 35, "low", 36)]
    laws = analyze_laws(df, compute_features(df, config), swings, None, config=config)
    comparison = laws["effort_result"]["wave_comparisons"][0]
    assert comparison["kind"] == "reaction"
    assert comparison["relationship"] == "increased_effort_reduced_result"
    assert comparison["interpretation"] == "possible_supply_absorption"
    assert comparison["available_date"] == df.index[36]


def test_readiness_preserves_official_order_and_missing_requirements():
    df = bars()
    features = compute_features(df)
    context = analyze_context(df, features, [])
    laws = analyze_laws(df, features, [], tr(df))
    result = evaluate_readiness(context, laws, [], tr(df), None)
    assert [p["id"] for p in result["nine_buying"]] == list(range(1, 10))
    assert [p["id"] for p in result["nine_selling"]] == list(range(1, 10))
    assert len(result["five_steps"]) == 5
    assert result["nine_buying"][6]["label"] == "强于市场"
    assert result["nine_selling"][3]["label"] == "弱于市场"
    for group, i in (("nine_buying", 0), ("nine_selling", 0), ("nine_buying", 3),
                     ("nine_selling", 4), ("nine_buying", 6), ("nine_selling", 3),
                     ("nine_buying", 8), ("nine_selling", 8)):
        assert result[group][i]["status"] == "unavailable"
        assert result[group][i]["required_data"]
    assert result["nine_selling"][7]["status"] == "unavailable"  # Bar range is not P&F crown proof.


def test_forming_range_does_not_automatically_pass_base_and_foreign_cycle_is_excluded():
    df = bars()
    context = analyze_context(df, compute_features(df), [])
    range_ = tr(df, lifecycle="forming")
    laws = analyze_laws(df, compute_features(df), [], range_)
    events = [{"event": name, "date": df.index[26 + i], "side": "accumulation", "range_id": "old-range"}
              for i, name in enumerate(("PS", "SC", "ST"))]
    result = evaluate_readiness(context, laws, events, range_, None)
    assert result["nine_buying"][1]["status"] == "not_met"
    assert result["nine_buying"][7]["status"] == "not_met"


def test_reward_risk_needs_explicit_plan_and_cannot_lower_classic_three_r_threshold():
    df = bars()
    context = analyze_context(df, compute_features(df), [])
    range_ = tr(df)
    targets = {"bullish": {"min": 110., "mid": 115., "max": 120.}}
    laws = analyze_laws(df, compute_features(df), [], range_, targets)
    absent = evaluate_readiness(context, laws, [], range_, targets)
    assert absent["nine_buying"][8]["status"] == "unavailable"
    result = evaluate_readiness(context, laws, [], range_, targets,
                                {"entry": 100, "stop": 95, "min_reward_risk": 2})
    assert result["nine_buying"][8]["status"] == "not_met"
    result = evaluate_readiness(context, laws, [], range_, targets, {"entry": 100, "stop": 98})
    assert result["nine_buying"][8]["status"] == "met"
    assert result["nine_buying"][0]["status"] == "unavailable"  # Current target is not prior objective history.
    with pytest.raises(ValueError, match="positive and finite"):
        evaluate_readiness(context, laws, [], range_, targets, {"entry": 100, "stop": float("nan")})
