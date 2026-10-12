"""Observable cycle changes and their negative cases; not accuracy labels."""
from copy import deepcopy
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from analysis import analyze_frame
from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.context import analyze_context
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.stages import classify_phase, range_events, sequence_state
from wyckoff_core.structure import available, replay_structure
from wyckoff_core.timeline import stage_timeline


def bars(n=65):
    close = np.full(n, 105.)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": 100.},
                        index=pd.date_range("2025-01-01", periods=n))


def put(df, i, high, low, close, volume=100):
    df.iloc[i] = [close, high, low, close, volume]


def mirror(df):
    out = df.copy()
    out["Open"], out["Close"] = 210 - df.Open, 210 - df.Close
    out["High"], out["Low"] = 210 - df.Low, 210 - df.High
    return out


def prior_range(df):
    return TR(0, 0, 110, 100, 0, 105, start_date=df.index[20],
              confirmed_date=df.index[25], established_date=df.index[25],
              range_id="previous", origin="ohlcv_swings")


def replay(df, cfg=None, **kwargs):
    return replay_structure(df, compute_features(df, cfg), confirmed_swings(df, cfg), cfg,
                            initial_range=prior_range(df), **kwargs)


def phase(df, result):
    return classify_phase(result["events"], result["tr"], [], df.Close.iloc[-1],
                          context={"stopping_candidate": result["stopping_candidate"]})


def new_cycle(bull=True):
    df = bars()
    for i, price in enumerate([98, 95, 92, 89, 86, 82], start=30):
        put(df, i, price + 1, price - 1, price)
    put(df, 36, 85, 75, 82, 1000)
    put(df, 37, 90, 81, 89)
    put(df, 38, 88, 81, 84)
    for i in range(39, len(df)):
        put(df, i, 86, 81, 84)
    put(df, 46, 80, 75.5, 77, 50)
    return df if bull else mirror(df)


@pytest.mark.parametrize("bull", [True, False])
def test_mature_range_can_restart_a_with_separate_candidate_then_ar_then_signed_b(bull):
    df = new_cycle(bull)
    pending = replay(df.iloc[:37])
    candidate = pending["stopping_candidate"]
    assert pending["tr"].range_id == "previous"
    assert candidate["parent_range_id"] == "previous"
    assert phase(df.iloc[:37], pending)["phase"] == "A"
    assert phase(df.iloc[:37], pending)["prior_range_phase"]["range_id"] == "previous"
    formed = replay(df.iloc[:39])
    assert formed["tr"].lifecycle == "forming"
    assert formed["tr"].range_id == candidate["range_id"]
    assert formed["tr"].parent_range_id == "previous"
    assert formed["ranges"][0].lifecycle == "retired"
    assert phase(df.iloc[:39], formed)["phase"] == "A"
    mature = replay(df.iloc[:47])
    assert mature["tr"].lifecycle == "active"
    selected = phase(df.iloc[:47], mature)
    assert (selected["phase"], selected["regime"]) == ("B", "accumulation" if bull else "distribution")
    assert [e["event"] for e in range_events(mature["events"], mature["tr"])] == ["SC" if bull else "BC", "AR", "ST"]
    assert not any(e.get("range_id") == "previous" for e in range_events(mature["events"], mature["tr"]))


def test_local_climax_is_not_a_new_cycle_or_sequence_reset():
    df = bars()
    for i, price in enumerate([109, 107, 106, 104, 102], start=30):
        put(df, i, price + .5, price - .5, price)
    put(df, 35, 105, 99, 104, 1000)
    result = replay(df.iloc[:36])
    local = next(e for e in result["events"] if e["event"] == "SC")
    assert local["subtype"] == "local_stopping_action"
    assert result["stopping_candidate"] is None
    assert result["tr"].range_id == "previous"
    assert phase(df.iloc[:36], result)["phase"] == "B"
    move = {"event": "SOS", "date": df.index[29], "range_id": "previous", "scope": "internal"}
    assert sequence_state([move, local])["sos_confirmed"]


def test_failed_candidate_preserves_old_range_and_does_not_later_create_ar():
    df = new_cycle()
    put(df, 37, 75, 72, 73)
    result = replay(df.iloc[:44])
    assert result["tr"].range_id == "previous"
    assert result["stopping_candidate"] is None
    assert not any(e["event"] == "AR" for e in result["events"])
    assert any(e.get("reason") == "stopping_extreme_closed_through" for e in result["events"])


def test_stopping_candidate_without_reversal_expires_and_does_not_erase_prior_range():
    df = new_cycle()
    for i in range(37, len(df)):
        put(df, i, 84 + (i - 37) * .2, 80 + (i - 37) * .2, 83 + (i - 37) * .2)
    result = replay(df.iloc[:54])
    assert result["tr"].range_id == "previous"
    assert result["stopping_candidate"] is None
    assert any(e.get("reason") == "stopping_sequence_expired" for e in result["events"])
    assert phase(df.iloc[:54], result)["phase"] != "A"


def test_retired_range_events_cannot_resurrect_a_or_become_ambiguous_negative():
    df = bars(115)
    for i in range(30, len(df)):
        put(df, i, 112 + i - 30, 110 + i - 30, 111 + i - 30)
    result = replay(df)
    assert result["tr"] is None
    assert result["ranges"][-1].lifecycle == "retired"
    assert result["transitions"][-1]["reason"] == "sustained_price_separation"
    old = {"event": "SC", "date": df.index[-1], "side": "accumulation", "range_id": "previous"}
    assert classify_phase([old], None, [], df.Close.iloc[-1])["phase"] == "?"
    # Insufficient evidence around an existing range remains abstention.
    assert classify_phase([], prior_range(df), [], 112)["hypotheses"]


def test_supported_departure_is_not_retired_just_because_it_is_old():
    df = bars(115)
    for i, h, l, c, v in ((30, 114, 109, 113, 300), (31, 114, 111, 112, 100),
                          (32, 109.5, 108, 109, 50), (33, 113.8, 110, 113, 100)):
        put(df, i, h, l, c, v)
    for i in range(34, len(df)):
        put(df, i, 115 + i - 34, 111 + i - 34, 114 + i - 34)
    for i, h, l, c in ((48, 140, 135, 138), (49, 138, 132, 135), (50, 135, 130, 132),
                        (51, 140, 136, 139), (52, 143, 139, 141)):
        put(df, i, h, l, c)
    for i in range(53, len(df)):
        put(df, i, 145 + i - 53, 142 + i - 53, 144 + i - 53)
    result = replay(df)
    assert any(e["event"] == "LPS" for e in result["events"])
    assert result["tr"].range_id == "previous"
    context = analyze_context(df, compute_features(df), confirmed_swings(df))
    assert classify_phase(result["events"], result["tr"], [], df.Close.iloc[-1],
                          outside_bars=80, context=context)["phase"] == "E"


def test_old_break_then_reentry_cannot_retire_on_the_next_single_large_departure():
    df = bars(115)
    put(df, 30, 114, 111, 113)
    put(df, 100, 132, 129, 131)
    result = replay(df.iloc[:101])
    assert result["tr"].range_id == "previous"
    assert result["tr"].lifecycle == "breakout_up"
    assert not any(t["state"] == "retired" for t in result["transitions"])


def test_forming_cycle_without_secondary_test_expires_even_inside_its_boundaries():
    df = new_cycle()
    for i in range(39, len(df)):
        put(df, i, 86, 81, 84)
    result = replay(df, CoreConfig(climax_sequence_max_bars=20))
    assert result["tr"] is None
    assert result["stopping_candidate"] is None
    assert any(t["reason"] == "forming_range_expired" for t in result["transitions"])


@pytest.mark.parametrize("bull", [True, False])
def test_every_prefix_and_future_perturbation_preserve_cycles_and_ledgers(bull):
    df = new_cycle(bull)
    snapshots = {}
    full = replay(df, on_bar=lambda i, state: snapshots.setdefault(i, deepcopy(state)))
    for end in range(26, len(df) + 1):
        prefix = replay(df.iloc[:end])
        assert prefix["tr"] == snapshots[end - 1]["tr"]
        assert prefix["stopping_candidate"] == snapshots[end - 1]["stopping_candidate"]
        assert prefix["events"] == [e for e in full["events"] if available(e) <= df.index[end - 1]]
        assert prefix["transitions"] == snapshots[end - 1]["transitions"]
    later = df.copy()
    later.iloc[47:, :4] *= 3
    later.iloc[47:, 4] *= 50
    altered = replay(later)
    assert [e for e in altered["events"] if available(e) <= df.index[46]] == replay(df.iloc[:47])["events"]
    assert len(fields(CoreConfig)) == 31


def test_signed_b_needs_unique_evidence_and_opposite_initiative_revokes_it():
    df = bars()
    tr = prior_range(df)
    ev = [{"event": name, "date": df.index[20 + i], "range_id": "previous", "side": "accumulation"}
          for i, name in enumerate(("SC", "AR", "ST"))]
    assert classify_phase(ev, tr, [], 105)["regime"] == "accumulation"
    assert classify_phase([], tr, [], 105)["regime"] == "trading range"
    assert classify_phase(ev[:2], tr, [], 105)["regime"] == "trading range"
    ev.append({"event": "SOW", "date": df.index[30], "range_id": "previous", "side": "distribution", "scope": "internal"})
    assert classify_phase(ev, tr, [], 105)["regime"] == "trading range"
    assert classify_phase(ev, replace(tr, lifecycle="forming", established_date=None), [], 105)["phase"] == "A"


def test_no_range_does_not_imply_observable_negative_without_explicit_eligibility():
    df = bars()
    assert classify_phase([], None, [], 105)["normal_no_observable"] is False
    assert classify_phase([], None, [], 105, context={"observation_eligible": False})["normal_no_observable"] is False
    assert classify_phase([], None, [], 105, context={"observation_eligible": True})["normal_no_observable"] is True
    # Normal absence is distinct from uncertainty around an existing range.
    assert classify_phase([], prior_range(df), [], 115,
                          context={"observation_eligible": True})["normal_no_observable"] is False
    candidate = {"range_id": "new-stop", "side": "accumulation", "parent_range_id": None}
    phase = classify_phase([], None, [], 105, context={"observation_eligible": True, "stopping_candidate": candidate})
    assert phase["phase"] == "A" and phase["normal_no_observable"] is False


@pytest.mark.parametrize("case,eligible", [("observed", True), ("all_zero_volume", False),
    ("current_zero_volume", False), ("baseline_zero_volume", False), ("unavailable_spread_baseline", False),
    ("warmup", False)])
def test_normal_absence_requires_observed_inputs_in_both_snapshot_and_timeline(case, eligible):
    df = bars()
    if case == "all_zero_volume":
        df["Volume"] = 0
    elif case == "current_zero_volume":
        df.iloc[-1, df.columns.get_loc("Volume")] = 0
    elif case == "baseline_zero_volume":
        df.iloc[-5, df.columns.get_loc("Volume")] = 0
    elif case == "unavailable_spread_baseline":
        # Keep historical ATR usable for the public P&F snapshot while the
        # entire current observation baseline has no measurable spread.
        df.loc[df.index[20:], "High"] = df.loc[df.index[20:], "Close"]
        df.loc[df.index[20:], "Low"] = df.loc[df.index[20:], "Close"]
    elif case == "warmup":
        df = df.iloc[:CoreConfig().baseline_window]
    timeline = stage_timeline(df)
    row = timeline["rows"][-1]
    snapshot = analyze_frame(df)["analyses"]["atr"]["wyckoff"]
    assert row["range_id"] is None and row["phase"]["phase"] == "?"
    assert row["phase"] == snapshot["phase"]
    assert row["phase"]["normal_no_observable"] is eligible
    assert snapshot["context"]["observation_eligible"] is eligible


def test_missing_volume_eligibility_recovers_only_after_complete_observed_baseline():
    df = bars()
    df.iloc[30, df.columns.get_loc("Volume")] = 0
    timeline = stage_timeline(df)
    for i, row in enumerate(timeline["rows"]):
        expected = i >= CoreConfig().baseline_window and not 30 <= i <= 50
        assert row["phase"]["normal_no_observable"] is expected
    # Eligibility, like stage evidence, cannot be repaired using future input.
    early = stage_timeline(df.iloc[:40])
    assert early["rows"] == timeline["rows"][:40]
    for end in (20, 21, 30, 31, 40, 51, 52, len(df)):
        snapshot = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert snapshot["phase"] == timeline["rows"][end - 1]["phase"]
