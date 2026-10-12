"""Phase C is causal evidence, including conservative no-boundary-probe tests."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import TR
from wyckoff_core.stages import classify_phase, sequence_state
from wyckoff_core.structure import available, replay_structure


def interior_bars(accumulation=True):
    closes = np.full(52, 105.)
    closes[26:43] = [105, 106, 107, 106, 104.5, 103.5, 103, 104, 106,
                     107, 108.5, 107, 106, 105, 104.5, 106, 107]
    df = pd.DataFrame({"Open": closes, "High": closes + 1, "Low": closes - 1,
                       "Close": closes, "Volume": 100.},
                      index=pd.date_range("2025-01-01", periods=len(closes)))
    df.iloc[32] = [103, 104, 101.5, 103, 100]
    df.iloc[40] = [104.5, 105, 103.5, 104.5, 40]
    return df if accumulation else mirror(df)


def mirror(df):
    mirrored = df.copy()
    mirrored["Open"], mirrored["Close"] = 210 - df["Open"], 210 - df["Close"]
    mirrored["High"], mirrored["Low"] = 210 - df["Low"], 210 - df["High"]
    return mirrored


def replay(df, **kwargs):
    tr = TR(0, 0, 110, 100, 0, 105, start_date=df.index[20],
            confirmed_date=df.index[25], established_date=df.index[25],
            range_id="phase-c-audit", origin="ohlcv_swings")
    return replay_structure(df, compute_features(df), confirmed_swings(df), initial_range=tr, **kwargs)


def phase(df, result=None):
    result = replay(df) if result is None else result
    return classify_phase(result["events"], result["tr"], [], df["Close"].iloc[-1])


@pytest.mark.parametrize("accumulation", [True, False])
def test_interior_four_pivot_test_confirms_c_only_when_observable(accumulation):
    df = interior_bars(accumulation)
    before, confirmed = replay(df.iloc[:42]), replay(df.iloc[:43])
    assert phase(df.iloc[:42], before)["phase"] == "B"
    p = phase(df.iloc[:43], confirmed)
    assert (p["phase"], p["regime"]) == ("C", "accumulation" if accumulation else "distribution")
    tests = [e for e in confirmed["events"] if e["event"] in ("TEST", "TEST_DEMAND")]
    assert len(tests) == 1
    test = tests[0]
    assert (test["date"], available(test)) == (df.index[40], df.index[42])
    assert (test["prior_turn_date"], test["parent_date"], test["turn_date"]) == (
        df.index[28], df.index[32], df.index[36])
    assert test["range_id"] == "phase-c-audit"
    assert not any(e["event"] in ("SPRING", "UT", "UTAD") for e in confirmed["events"])
    full = replay(df)
    assert confirmed["events"] == [e for e in full["events"] if available(e) <= df.index[42]]


@pytest.mark.parametrize("accumulation", [True, False])
@pytest.mark.parametrize("counterexample", ["no_directional_turn", "high_volume", "wide_spread",
                                             "no_response", "outside_range", "unknown_volume"])
def test_a_higher_low_alone_or_insufficient_supply_demand_evidence_stays_b(accumulation, counterexample):
    df = interior_bars()
    if counterexample == "no_directional_turn":
        df.iloc[28, df.columns.get_loc("High")] = 109.8  # Next rally is a lower high.
    elif counterexample == "high_volume":
        df.iloc[40, df.columns.get_loc("Volume")] = 120
    elif counterexample == "wide_spread":
        df.iloc[40, df.columns.get_loc("Low")] = 102  # HL, but no spread contraction.
    elif counterexample == "no_response":
        df.iloc[41] = [104.8, 105, 104.2, 104.8, 100]
        df.iloc[42] = [104.9, 105.2, 104.3, 104.9, 100]
    elif counterexample == "outside_range":
        df.iloc[38] = [111, 112, 106, 111, 100]
    else:
        df.iloc[38, df.columns.get_loc("Volume")] = 0
    if not accumulation:
        df = mirror(df)
    result = replay(df.iloc[:43])
    assert not any(e.get("parent_type") == "confirmed_swing" for e in result["events"])
    assert phase(df.iloc[:43], result)["phase"] == "B"


@pytest.mark.parametrize("accumulation", [True, False])
def test_failed_interior_test_is_recorded_and_does_not_revive_on_reentry(accumulation):
    df = interior_bars()
    df.iloc[43] = [103.5, 104.5, 102.5, 103.5, 0]  # Price invalidation needs no volume.
    df.iloc[44:48] = [[105, 106, 104, 105, 100], [104, 105, 103, 104, 30],
                      [105, 106, 104, 105, 100], [106, 107, 105, 106, 100]]
    if not accumulation:
        df = mirror(df)
    result = replay(df.iloc[:48])
    invalidations = [e for e in result["events"] if e["event"] == "INVALIDATION"]
    assert len(invalidations) == 1
    assert invalidations[0]["invalidated_event_date"] == df.index[40]
    assert invalidations[0]["date"] == df.index[43]
    assert phase(df.iloc[:48], result)["phase"] == "B"
    # A delayed record cannot reuse the invalidated four-pivot sequence.
    old_test = next(e for e in result["events"] if e.get("parent_type") == "confirmed_swing")
    late = {**old_test, "date": df.index[45], "available_date": df.index[47]}
    state = sequence_state([*result["events"], late])
    assert state["acc_test" if accumulation else "dist_test"] is None


def test_uncontextualized_standalone_test_is_not_phase_c():
    df = interior_bars()
    result = replay(df.iloc[:43])
    test = next(e for e in result["events"] if e.get("parent_type") == "confirmed_swing")
    for field in ("range_id", "range_confirmed_at", "turn_date", "parent_type", "status"):
        unsupported = {k: v for k, v in test.items() if k != field}
        assert classify_phase([unsupported], result["tr"], [], 107)["phase"] == "B"


def test_existing_initiative_does_not_regress_to_c_on_interior_test():
    df = interior_bars()
    result = replay(df.iloc[:43])
    test = next(e for e in result["events"] if e.get("parent_type") == "confirmed_swing")
    sos = {"event": "SOS", "date": df.index[38], "price": 111,
           "range_id": "phase-c-audit", "scope": "breakout", "side": "accumulation"}
    assert classify_phase([sos, test], result["tr"], [], 107)["phase"] == "D"


def test_bar_observer_includes_warmup_and_zero_volume_and_matches_prefix_replay():
    df = interior_bars()
    df.iloc[43, df.columns.get_loc("Volume")] = 0
    snapshots = {}
    full = replay(df, on_bar=lambda i, state: snapshots.setdefault(i, deepcopy(state)))
    assert list(snapshots) == list(range(len(df)))
    for end in (1, 20, 25, 26, 41, 42, 43, 44, len(df)):
        # Preserve the supplied range dates, including before its confirmation.
        prefix = df.iloc[:end]
        tr = snapshots[0]["tr"]
        expected = replay_structure(prefix, compute_features(prefix), confirmed_swings(prefix), initial_range=tr)
        observed = snapshots[end - 1]
        assert sorted(observed["events"], key=lambda e: (available(e), e["date"])) == expected["events"]
        assert observed["tr"] == expected["tr"]
        assert observed["transitions"] == expected["transitions"]
    assert snapshots[len(df) - 1]["tr"] == full["tr"]


def test_two_boundary_ambiguity_cancels_pending_probe_confirmation():
    df = interior_bars()
    df.iloc[26:] = [105, 106, 104, 105, 100]
    df.iloc[30] = [101, 102, 97, 101, 70]
    df.iloc[31] = [104, 105, 103, 104, 100]
    df.iloc[32] = [105, 111, 99, 105, 100]
    df.iloc[36] = [101, 102, 100.5, 101, 30]
    df.iloc[37] = [104, 105, 103, 104, 100]
    df.iloc[38] = [105, 106, 104, 105, 100]
    result = replay(df.iloc[:39])
    assert any(e["event"] == "SPRING" and e["date"] == df.index[30] for e in result["events"])
    assert any(e["event"] == "AMBIGUOUS_TEST" and e["date"] == df.index[32] for e in result["events"])
    assert not any(e["event"] == "TEST" for e in result["events"])
    assert phase(df.iloc[:39], result)["phase"] == "?"
