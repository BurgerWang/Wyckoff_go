"""Independent behavioral examples for range, stage and count integration."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import pnf
import wyckoff
from wyckoff_core.counts import count_scenarios
from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import TR
from wyckoff_core.stages import classify_phase, sequence_state
from wyckoff_core.structure import available, replay_structure


def bars(closes=None, n=50):
    closes = np.array([105.] * n if closes is None else closes, dtype=float)
    return pd.DataFrame({"Open": closes, "High": closes + 1,
                         "Low": closes - 1, "Close": closes, "Volume": 100.},
                        index=pd.date_range("2025-01-01", periods=len(closes)))


def put(df, i, high, low, close, volume=100):
    df.iloc[i] = [close, high, low, close, volume]


def mature_range(df):
    return TR(0, 4, 110, 100, 5, 105, start_date=df.index[20],
              confirmed_date=df.index[25], established_date=df.index[25],
              range_id="review-range", origin="ohlcv_swings")


def replay(df):
    return replay_structure(df, compute_features(df), confirmed_swings(df),
                            initial_range=mature_range(df))


def event(df, i, name, price=105, **extra):
    return {"event": name, "date": df.index[i], "available_date": df.index[i],
            "price": price, "vol": 1., "range_id": "review-range", **extra}


def test_ordinary_ohlc_oscillation_has_real_boundaries_and_no_spring():
    closes = list(np.linspace(130, 112, 22)) + [c for c in (100, 110) * 4 for _ in range(3)]
    df = bars(closes)
    df["High"] = df["Close"] + .5
    df["Low"] = df["Close"] - .5
    chart = pnf.build(df, 1)
    result = wyckoff.analyze(chart.columns, chart.scale, df)
    assert result["tr"] is not None
    assert (result["tr"].bottom, result["tr"].top) == (99.5, 110.5)
    assert not any(e["event"] in ("SPRING", "UT", "UTAD") for e in result["events"])
    assert result["phase"]["phase"] == "B"


def test_confirmed_event_ledger_is_prefix_stable_through_a_later_reaction():
    df = bars()
    put(df, 30, 114, 109, 113, 300)
    put(df, 31, 114, 111, 112)
    put(df, 32, 109.5, 108, 109, 50)
    put(df, 33, 114, 110, 113)
    put(df, 34, 115, 111, 114)
    early = replay(df.iloc[:35])
    full = replay(df)
    assert any(e["event"] == "LPS" for e in early["events"])
    assert early["events"] == [e for e in full["events"] if available(e) <= df.index[34]]
    assert early["tr"].range_id == full["tr"].range_id
    assert early["tr"].avg_volume == full["tr"].avg_volume


def test_boundary_probe_and_internal_initiative_do_not_mechanically_set_c_or_d():
    df = bars()
    tr = mature_range(df)
    spring = event(df, 30, "SPRING", 99, side="accumulation")
    ut = event(df, 30, "UT", 111, side="distribution")
    for probe in (spring, ut):
        assert classify_phase([probe], tr, [], 105)["phase"] == "B"
    for name, side, price in (("SOS", "accumulation", 108), ("SOW", "distribution", 102)):
        move = event(df, 31, name, price, side=side, scope="internal")
        assert classify_phase([move], tr, [], price)["phase"] == "B"


def test_preserving_preliminary_support_cannot_hide_latest_two_boundary_ambiguity():
    df = bars()
    events = [event(df, 19, "PS", 101, range_id=None, side="accumulation"),
              event(df, 20, "SC", 100, range_id=None, side="accumulation"),
              event(df, 22, "AR", 110, side="accumulation"),
              event(df, 30, "AMBIGUOUS_TEST", 105)]
    result = classify_phase(events, mature_range(df), [], 105)
    assert result["status"] == "ambiguous"


def test_breakout_reentry_keeps_frozen_range_identity_and_boundaries():
    df = bars()
    put(df, 30, 113, 110, 112)
    put(df, 31, 109, 105, 108)
    result = replay(df)
    assert result["tr"].range_id == "review-range"
    assert (result["tr"].bottom, result["tr"].top) == (100, 110)
    assert [t["state"] for t in result["transitions"]] == ["breakout_up", "reentered"]


def test_detached_confirmed_congestion_retires_previous_broken_range():
    closes = [105.] * 30 + [c for c in (120, 130) * 4 for _ in range(3)]
    result = replay(bars(closes))
    assert len(result["ranges"]) == 2
    assert result["ranges"][0].lifecycle == "retired"
    assert result["tr"].lifecycle == "active"
    assert (result["tr"].bottom, result["tr"].top) == (119, 131)
    assert result["tr"].range_id != "review-range"


def test_abandoned_forming_range_does_not_block_later_detached_congestion():
    closes = list(np.linspace(120, 101, 20)) + [104, 109, 101]
    closes += [c for c in (120, 130) * 5 for _ in range(3)]
    df = bars(closes)
    put(df, 20, 105, 95, 104, 1000)
    put(df, 21, 110, 102, 109)
    put(df, 22, 108, 100, 101)
    early = replay_structure(df.iloc[:23], compute_features(df.iloc[:23]),
                             confirmed_swings(df.iloc[:23]))
    assert early["tr"].lifecycle == "forming"
    result = replay_structure(df, compute_features(df), confirmed_swings(df))
    assert result["tr"].lifecycle == "active"
    assert (result["tr"].bottom, result["tr"].top) == (119, 131)
    assert result["ranges"][0].lifecycle == "retired"


@pytest.mark.parametrize("violation_volume", [100, 0])
def test_lps_undercut_invalidates_completed_support_without_rewriting_history(violation_volume):
    df = bars()
    for i, high, low, close, vol in (
            (30, 114, 109, 113, 300), (31, 114, 111, 112, 100),
            (32, 109.5, 108, 109, 50), (33, 114, 110, 113, 100),
            (34, 115, 111, 114, 100), (35, 106, 103, 105, violation_volume),
            (36, 109, 106, 108, 100), (37, 112, 107, 111, 100),
            (38, 114, 109, 113, 100)):
        put(df, i, high, low, close, vol)
    for i in range(39, 50):
        put(df, i, 115 + i - 39, 112 + i - 39, 114 + i - 39)
    before = replay(df.iloc[:35])
    after = replay(df)
    assert sequence_state(before["events"])["acc_complete"]
    assert before["events"] == [e for e in after["events"] if available(e) <= df.index[34]]
    assert not sequence_state(after["events"])["acc_complete"]
    phase = classify_phase(after["events"], after["tr"], [], df["Close"].iloc[-1],
                           outside_bars=12, context={"price_structure": {"direction": "up"}})
    assert phase["phase"] != "E"


def test_failed_spring_cannot_be_confirmed_by_an_unrelated_later_higher_pivot():
    df = bars()
    put(df, 30, 102, 99, 101, 50)
    put(df, 31, 104, 101, 103)
    put(df, 32, 100, 98, 99)  # Violates the Spring extreme, without a high-volume SOW.
    put(df, 33, 105, 102, 104)
    put(df, 34, 106, 103, 105)
    put(df, 35, 107, 104, 106)
    put(df, 36, 102, 100.5, 101, 30)
    put(df, 37, 105, 103, 104)
    put(df, 38, 106, 104, 105)
    result = replay(df)
    assert any(e["event"] == "SPRING" and e["date"] == df.index[30] for e in result["events"])
    assert not any(e["event"] == "TEST" and e.get("parent_date") == df.index[30]
                   for e in result["events"])


@pytest.mark.parametrize("accumulation", [True, False])
@pytest.mark.parametrize("violation_volume", [100, 0])
def test_failed_initiative_cannot_revive_after_opposite_boundary_break_and_reentry(
        accumulation, violation_volume):
    df = bars()
    put(df, 30, 114, 109, 113, 300)
    put(df, 31, 100, 98, 99, violation_volume)
    put(df, 32, 106, 104, 105)
    if not accumulation:
        original = df.copy()
        df["Open"], df["Close"] = 210 - original["Open"], 210 - original["Close"]
        df["High"], df["Low"] = 210 - original["Low"], 210 - original["High"]
    early = replay(df.iloc[:31])
    after = replay(df)
    expected = "SOS" if accumulation else "SOW"
    assert any(e["event"] == expected for e in early["events"])
    assert classify_phase(early["events"], early["tr"], [], df["Close"].iloc[30])["phase"] == "D"
    assert after["tr"].lifecycle == "reentered"
    assert classify_phase(after["events"], after["tr"], [], 105)["phase"] == "B"


def count_fixture():
    df = bars()
    columns = [["X", 100, 110, 1, df.index[20]], ["O", 100, 109, 1, df.index[21]],
               ["X", 106, 110, 1, df.index[22]], ["O", 106, 109, 1, df.index[23]],
               ["X", 100, 110, 1, df.index[24]]]
    events = [event(df, 22, "SOS", 111, scope="breakout"),
              event(df, 24, "LPS", 105, status="confirmed", parent_date=df.index[22])]
    events[-1]["available_date"] = df.index[26]
    return df, columns, mature_range(df), events


def test_horizontal_span_counts_blank_columns_between_valid_endpoints():
    _, columns, tr, events = count_fixture()
    result = count_scenarios(columns, np.arange(90., 121.), tr, events)
    assert result["status"] == "scenario"
    target = result["targets"]
    assert target["width_bull"] == 5
    assert target["occupied_columns_bull"] == 3
    assert target["extension_bull"] == 15
    assert target["bullish"] == {"min": 115., "mid": 117.5, "max": 120.}


def test_new_spring_does_not_resurrect_superseded_lps_count_via_compatibility():
    df, columns, tr, events = count_fixture()
    assert count_scenarios(columns, np.arange(90., 121.), tr, events)["status"] == "scenario"
    events.append(event(df, 30, "SPRING", 99, side="accumulation"))
    assert sequence_state(events)["acc_lps"] is None
    result = count_scenarios(columns, np.arange(90., 121.), tr, events)
    assert result["targets"] is None or result["targets"]["bullish"] is None


def test_count_rebuilds_at_confirmation_when_future_columns_extend_and_reverse():
    closes = [105.] * 20 + [c for c in (100, 110) * 5 for _ in range(3)] + [120, 140, 160]
    df = bars(closes)
    tr = mature_range(df)
    events = [event(df, 35, "SOS", 110, scope="internal"),
              event(df, 38, "LPS", 100, status="confirmed", parent_date=df.index[35])]
    events[-1]["available_date"] = df.index[40]
    scale = np.arange(50., 181.)
    early_df = df.iloc[:41]
    early_chart, full_chart = pnf.build(early_df, scale), pnf.build(df, scale)
    early = count_scenarios(early_chart.columns, scale, tr, events, df=early_df, use_close=True)
    full = count_scenarios(full_chart.columns, scale, tr, events, df=df, use_close=True)
    assert early["status"] == "scenario"
    assert early["price_method"] == "close"
    assert early == full


def test_count_uses_the_displayed_high_low_chart_not_an_implicit_close_chart():
    df = bars(n=40)
    df["High"], df["Low"] = 110., 100.
    grid = np.arange(90., 121.)
    chart = pnf.build(df, grid, use_close=False)
    tr = replace(mature_range(df), end_col=len(chart.columns) - 1)
    events = [event(df, 25, "SOS", 109, scope="internal"),
              event(df, 30, "LPS", 100, status="confirmed", parent_date=df.index[25])]
    events[-1]["available_date"] = df.index[32]
    # The high/low chart has eleven columns in the anchored span, including
    # five blanks at the 100 row. All closes are 105: a close chart has none.
    supplied = count_scenarios(chart.columns, grid, tr, events)
    assert supplied["targets"]["width_bull"] == 11
    rebuilt = count_scenarios(chart.columns, grid, tr, events, df=df)
    assert rebuilt["status"] == "scenario"
    assert rebuilt["price_method"] == "high_low"
    assert rebuilt["targets"]["width_bull"] == 11
