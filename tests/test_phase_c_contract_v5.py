"""Terminal support invalidation and inspectable relative-volume comparisons."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wyckoff_core.context import analyze_context
from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.stages import classify_phase, range_events, sequence_state
from wyckoff_core.structure import available, replay_structure


def bars(n=65):
    x = np.full(n, 105.)
    return pd.DataFrame({"Open": x, "High": x + 1, "Low": x - 1,
                         "Close": x, "Volume": 100.},
                        index=pd.date_range("2025-01-01", periods=n))


def put(df, i, high, low, close, volume=100):
    df.iloc[i] = [close, high, low, close, volume]


def mirror(df):
    out = df.copy()
    out["Open"], out["Close"] = 210 - df.Open, 210 - df.Close
    out["High"], out["Low"] = 210 - df.Low, 210 - df.High
    return out


def replay(df, **kwargs):
    tr = TR(0, 0, 110, 100, 0, 105, start_date=df.index[20],
            confirmed_date=df.index[25], established_date=df.index[25],
            range_id="contract-v5", origin="ohlcv_swings")
    return replay_structure(df, compute_features(df), confirmed_swings(df),
                            initial_range=tr, **kwargs)


def phase(df, result, bull=True, outside_bars=0):
    return classify_phase(result["events"], result["tr"], [], df.Close.iloc[-1],
                          outside_bars=outside_bars,
                          context={"price_structure": {"direction": "up" if bull else "down"}})


def boundary_test(completed_lps=False):
    df = bars()
    if completed_lps:
        put(df, 30, 110, 104, 109, 300)
        put(df, 31, 110, 107, 108)
    else:
        put(df, 30, 102, 97, 101, 70)
    put(df, 36, 101.5, 100.1, 101, 30)
    put(df, 37, 105, 103, 104)
    put(df, 38, 106, 104, 105)
    return df


def reentry(completed_lps=False, violation_volume=100):
    df = boundary_test(completed_lps)
    put(df, 39, 100.2, 99.9, 99.95, violation_volume)
    put(df, 40, 106, 104, 105)
    return df


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("confirmation_close", [99.98, 100., 100.02])
def test_probe_test_requires_range_recovery_on_its_confirmation_bar(bull, confirmation_close):
    df = boundary_test()
    put(df, 36, 100.1, 99.9, 99.95, 30)
    put(df, 37, 100.2, 99.92, 99.97)
    put(df, 38, 100.3, 99.94, confirmation_close)
    if not bull:
        df = mirror(df)
    confirmed = replay(df.iloc[:39])
    returned = replay(df.iloc[:40])
    test_name = "TEST" if bull else "TEST_DEMAND"
    tests = [e for e in returned["events"] if e["event"] == test_name]
    if confirmation_close < 100:
        assert not tests
        assert phase(df.iloc[:39], confirmed, bull)["phase"] == "?"
        assert phase(df.iloc[:40], returned, bull)["phase"] == "B"
    else:
        assert len(tests) == 1
        assert (tests[0]["date"], available(tests[0])) == (df.index[36], df.index[38])
        assert phase(df.iloc[:39], confirmed, bull)["phase"] == "C"
        assert phase(df.iloc[:40], returned, bull)["phase"] == "C"


@pytest.mark.parametrize("bull", [True, False])
def test_rejected_confirmation_does_not_block_a_new_valid_probe_test_chain(bull):
    df = boundary_test()
    put(df, 36, 100.1, 99.9, 99.95, 30)
    put(df, 37, 100.2, 99.92, 99.97)
    put(df, 38, 100.3, 99.94, 99.98)
    put(df, 44, 102, 97, 101, 70)
    put(df, 50, 102, 100.5, 101, 30)
    put(df, 51, 105, 103, 104)
    put(df, 52, 106, 104, 105)
    if not bull:
        df = mirror(df)
    full = replay(df)
    tests = [e for e in full["events"] if e["event"] == ("TEST" if bull else "TEST_DEMAND")]
    assert [(e["date"], e["parent_date"]) for e in tests] == [(df.index[50], df.index[44])]
    assert phase(df.iloc[:53], replay(df.iloc[:53]), bull)["phase"] == "C"
    for end in (37, 38, 39, 40, 51, 52, 53):
        assert replay(df.iloc[:end])["events"] == [
            e for e in full["events"] if available(e) <= df.index[end - 1]]
    later = df.copy()
    later.iloc[40:, :4] *= .3
    later.iloc[40:, 4] *= 20
    altered = replay(later)
    assert [e for e in altered["events"] if available(e) <= df.index[39]] == replay(df.iloc[:40])["events"]


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("violation_volume", [0, 100])
@pytest.mark.parametrize("completed_lps", [False, True])
def test_opposite_close_terminates_confirmed_support_even_within_pivot_buffer(
        bull, violation_volume, completed_lps):
    df = reentry(completed_lps, violation_volume)
    if not bull:
        df = mirror(df)
    support_name = ("LPS" if bull else "LPSY") if completed_lps else ("TEST" if bull else "TEST_DEMAND")
    before = replay(df.iloc[:39])
    support = next(e for e in before["events"] if e["event"] == support_name)
    assert phase(df.iloc[:39], before, bull)["phase"] == ("D" if completed_lps else "C")
    state = sequence_state(range_events(before["events"], before["tr"]))
    assert state["acc_complete" if bull else "dist_complete"] is completed_lps
    # The old extreme-only check still holds, so this isolates the range-close
    # contract rather than duplicating the established deep-undercut checks.
    buffer = CoreConfig().boundary_buffer_atr * compute_features(df).atr.iloc[39]
    assert (df.Low.iloc[39] > support["price"] - buffer if bull
            else df.High.iloc[39] < support["price"] + buffer)
    outside = replay(df.iloc[:40])
    invalidations = [e for e in outside["events"] if e["event"] == "INVALIDATION"]
    assert len(invalidations) == 1
    assert invalidations[0]["invalidated_event"] == support_name
    assert invalidations[0]["invalidated_event_date"] == support["date"]
    assert invalidations[0]["reason"] == "opposite_range_boundary_closed_through"
    assert available(invalidations[0]) == df.index[39]
    # Completed LPS consumes its runtime initiative. Its own invalidation must
    # clear the sequence even when there is no active initiative left to watch.
    after = replay(df.iloc[:43])
    state = sequence_state(range_events(after["events"], after["tr"]))
    assert not state["acc_complete" if bull else "dist_complete"]
    assert not state["sos_confirmed" if bull else "sow_confirmed"]
    assert state["acc_test" if bull else "dist_test"] is None
    assert phase(df.iloc[:43], after, bull)["phase"] == "B"
    assert before["events"] == [e for e in after["events"] if available(e) <= df.index[38]]


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("violation_volume", [0, 100])
def test_a_new_probe_and_later_test_can_establish_c_after_terminal_invalidation(bull, violation_volume):
    df = reentry(violation_volume=violation_volume)
    put(df, 44, 102, 97, 101, 70)
    put(df, 50, 102, 100.5, 101, 30)
    put(df, 51, 105, 103, 104)
    put(df, 52, 106, 104, 105)
    if not bull:
        df = mirror(df)
    intermediate = replay(df.iloc[:52])
    assert phase(df.iloc[:52], intermediate, bull)["phase"] == "B"
    full = replay(df.iloc[:53])
    assert phase(df.iloc[:53], full, bull)["phase"] == "C"
    tests = [e for e in full["events"] if e["event"] == ("TEST" if bull else "TEST_DEMAND")]
    assert [(e["date"], e["parent_date"]) for e in tests] == [
        (df.index[36], df.index[30]), (df.index[50], df.index[44])]
    state = sequence_state(full["events"])
    assert state["acc_test" if bull else "dist_test"]["date"] == df.index[50]


@pytest.mark.parametrize("bull", [True, False])
def test_active_initiative_invalidation_does_not_duplicate_its_watched_test(bull):
    df = boundary_test()
    put(df, 39, 110, 104, 109, 300)
    put(df, 40, 100.2, 99.9, 99.95, 0)
    if not bull:
        df = mirror(df)
    result = replay(df.iloc[:43])
    invalidations = [e for e in result["events"] if e["event"] == "INVALIDATION"]
    assert len(invalidations) == 1
    assert invalidations[0]["invalidated_event"] == ("SOS" if bull else "SOW")
    assert invalidations[0]["invalidated_event_date"] == df.index[39]
    assert phase(df.iloc[:43], result, bull)["phase"] == "B"


@pytest.mark.parametrize("bull", [True, False])
def test_support_invalidation_cancels_old_excursion_but_accepts_a_fresh_one(bull):
    df = boundary_test()
    put(df, 34, 99, 96.9, 98, 50)  # Below the range, still within the Spring buffer.
    put(df, 35, 99, 96, 98, 50)  # Invalidates the old Spring and its pending recovery.
    put(df, 36, 103, 98, 102, 80)
    put(df, 40, 99, 97, 98, 100)  # An independently started excursion.
    put(df, 41, 103, 98, 102, 80)
    put(df, 45, 102, 100.5, 101, 30)
    put(df, 46, 105, 103, 104)
    put(df, 47, 106, 104, 105)
    if not bull:
        df = mirror(df)
    failed = replay(df.iloc[:38])
    canceled = [e for e in failed["events"] if e.get("invalidated_event") == "BOUNDARY_EXCURSION"]
    assert len(canceled) == 1
    assert canceled[0]["invalidated_event_date"] == df.index[34]
    assert available(canceled[0]) == df.index[35]
    assert canceled[0]["reason"] == "support_chain_invalidated_during_recovery"
    assert not any(e.get("subtype") == "multi_bar_recovery" for e in failed["events"])
    full = replay(df.iloc[:48])
    recovered = [e for e in full["events"] if e.get("subtype") == "multi_bar_recovery"]
    assert len(recovered) == 1
    assert (recovered[0]["date"], available(recovered[0])) == (df.index[40], df.index[41])
    assert phase(df.iloc[:48], full, bull)["phase"] == "C"


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("continuation_volume", [0, 100])
def test_same_side_outside_closes_preserve_supported_e(bull, continuation_volume):
    df = bars()
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
    df.iloc[55, df.columns.get_loc("Volume")] = continuation_volume
    if not bull:
        df = mirror(df)
    result = replay(df)
    support_name = "LPS" if bull else "LPSY"
    assert any(e["event"] == support_name for e in result["events"])
    assert not any(e.get("invalidated_event") == support_name for e in result["events"])
    context = analyze_context(df, compute_features(df), confirmed_swings(df))
    assert classify_phase(result["events"], result["tr"], [], df.Close.iloc[-1],
                          outside_bars=30, context=context)["phase"] == "E"


@pytest.mark.parametrize("bull", [True, False])
def test_relative_effort_records_both_baselines_without_claiming_raw_volume_contracts(bull):
    df = boundary_test()
    df.iloc[32:36, df.columns.get_loc("Volume")] = 10000
    df.iloc[36, df.columns.get_loc("Volume")] = 100
    if not bull:
        df = mirror(df)
    result = replay(df.iloc[:39])
    test = next(e for e in result["events"] if e["event"] == ("TEST" if bull else "TEST_DEMAND"))
    assert phase(df.iloc[:39], result, bull)["phase"] == "C"
    assert test["volume"] == 100 > test["parent_volume"] == 70
    assert test["volume_ratio"] < test["parent_volume_ratio"] == .7
    assert test["parent_volume_baseline"] == 100
    assert test["test_volume_baseline"] == pytest.approx(2078.5)
    assert test["effort_comparison"] == "prior_baseline_relative_ratio"
    assert "relative_effort_below_probe" in test["evidence"]
    assert "reduced_effort" not in test["evidence"]
    control = df.iloc[:39].copy()
    control.iloc[32:36, control.columns.get_loc("Volume")] = 100
    assert phase(control, replay(control), bull)["phase"] == "B"


@pytest.mark.parametrize("bull", [True, False])
def test_multi_bar_probe_effort_uses_extreme_date_baseline_not_recovery_date(bull):
    df = bars()
    put(df, 30, 99, 97, 98, 150)
    put(df, 31, 98, 96, 97, 120)
    put(df, 32, 103, 98, 102, 80)
    put(df, 36, 102, 100.5, 101, 30)
    put(df, 37, 104, 102, 103)
    put(df, 38, 106, 104, 105)
    if not bull:
        df = mirror(df)
    result = replay(df.iloc[:39])
    test = next(e for e in result["events"] if e["event"] == ("TEST" if bull else "TEST_DEMAND"))
    parent = next(e for e in result["events"] if e.get("subtype") == "multi_bar_recovery")
    features = compute_features(df)
    assert test["parent_date"] == parent["date"] == df.index[31]
    assert available(parent) == df.index[32]
    assert test["parent_volume"] == 120
    assert test["parent_volume_baseline"] == features.volume_baseline.iloc[31]
    assert test["parent_volume_baseline"] != features.volume_baseline.iloc[32]
    assert test["test_volume_baseline"] == features.volume_baseline.iloc[36]


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("completed_lps", [False, True])
def test_invalidation_prefixes_and_new_evidence_are_unchanged_by_future_prices(bull, completed_lps):
    df = reentry(completed_lps, violation_volume=0)
    put(df, 44, 102, 97, 101, 70)
    put(df, 50, 102, 100.5, 101, 30)
    put(df, 51, 105, 103, 104)
    put(df, 52, 106, 104, 105)
    if not bull:
        df = mirror(df)
    snapshots = {}
    full = replay(df, on_bar=lambda i, state: snapshots.setdefault(i, deepcopy(state)))
    for end in range(26, 54):
        prefix = replay(df.iloc[:end])
        assert prefix["events"] == [e for e in full["events"] if available(e) <= df.index[end - 1]]
        assert prefix["tr"] == snapshots[end - 1]["tr"]
        assert prefix["transitions"] == snapshots[end - 1]["transitions"]
    changed = df.copy()
    changed.iloc[53:, :4] *= .3
    changed.iloc[53:, 4] *= 20
    altered = replay(changed)
    assert [e for e in altered["events"] if available(e) <= df.index[52]] == replay(df.iloc[:53])["events"]
