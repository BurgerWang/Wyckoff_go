"""External references and the distinction between minimum target and R/R."""
import pandas as pd

from wyckoff_core.context import analyze_context, analyze_laws
from wyckoff_core.features import compute_features
from wyckoff_core.readiness import evaluate_readiness


def frame():
    return pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 100.},
                        index=pd.date_range("2025-01-01", periods=50))


def references(df):
    return {"as_of": df.index[-1], "trend": "up", "historical_objectives": {
        "buy": {"target": 100., "issued_at": df.index[20], "achieved_at": df.index[30]}},
        "stride": {"downward": {"broken": True, "confirmed_at": df.index[40],
            "anchors": [{"date": df.index[20], "price": 120.}, {"date": df.index[30], "price": 110.}]}}}


def test_references_need_known_times_and_observed_target_achievement():
    df = frame()
    f = compute_features(df)
    supplied = references(df)
    context = analyze_context(df, f, [], market_context=supplied)
    assert context["stride"]["downward_broken"] is True
    assert context["historical_objectives"]["buy"]["met"] is True
    supplied["historical_objectives"]["buy"]["target"] = 90.
    context = analyze_context(df, f, [], market_context=supplied)
    assert context["historical_objectives"]["buy"]["met"] is False
    supplied["as_of"] = df.index[-1] + pd.Timedelta(days=1)
    context = analyze_context(df, f, [], market_context=supplied)
    assert context["stride"]["downward_broken"] is None
    assert context["historical_objectives"]["buy"]["met"] is None


def test_five_step_minimum_target_is_not_nine_test_reward_risk():
    df = frame()
    f = compute_features(df)
    context = analyze_context(df, f, [], market_context=references(df))
    targets = {"bullish": {"min": 110., "mid": 115., "max": 120.}}
    laws = analyze_laws(df, f, [], None, targets)
    plan = {"entry": 100., "stop": 98., "minimum_target": 112.}
    result = evaluate_readiness(context, laws, [], None, targets, plan)
    assert result["nine_buying"][8]["status"] == "met"
    assert result["five_steps"][2]["status"] == "not_met"
    assert result["nine_buying"][0]["status"] == "met"
    assert result["nine_buying"][3]["status"] == "met"
    del plan["minimum_target"]
    assert evaluate_readiness(context, laws, [], None, targets, plan)["five_steps"][2]["status"] == "unavailable"


def test_old_bars_are_not_interpreted_using_a_future_range_boundary():
    from wyckoff_core.models import TR
    df = frame()
    df.iloc[25] = [100., 100.5, 99.5, 100., 1000.]
    tr = TR(0, 4, 110., 99., 5, 104.5, confirmed_date=df.index[40])
    laws = analyze_laws(df, compute_features(df), [], tr)
    # Increase observation window so the earlier high-effort bar is visible.
    from wyckoff_core.models import CoreConfig
    laws = analyze_laws(df, compute_features(df), [], tr, config=CoreConfig(trend_window=30))
    observation = next(p for p in laws["effort_result"]["observations"] if p["date"] == df.index[25])
    assert observation["type"] == "high_effort_little_result"
