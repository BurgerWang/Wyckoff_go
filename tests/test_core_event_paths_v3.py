"""Gap, finite recovery and multi-bar initiative paths with causal negatives."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.stages import classify_phase, range_events, sequence_state
from wyckoff_core.structure import available, replay_structure


def bars(n=65):
    x = np.full(n, 105.)
    return pd.DataFrame({"Open": x, "High": x + 1, "Low": x - 1,
                         "Close": x, "Volume": 100.}, index=pd.date_range("2025-01-01", periods=n))


def put(df, i, h, l, c, v=100):
    df.iloc[i] = [c, h, l, c, v]


def mirror(df, total=210):
    x = df.copy()
    x["Open"], x["Close"] = total - df.Open, total - df.Close
    x["High"], x["Low"] = total - df.Low, total - df.High
    return x


def replay(df, cfg=None, top=110, **kwargs):
    tr = TR(0, 0, top, 100, 0, (100 + top) / 2, start_date=df.index[20],
            confirmed_date=df.index[25], established_date=df.index[25], range_id="events-v3", origin="ohlcv_swings")
    return replay_structure(df, compute_features(df, cfg), confirmed_swings(df, cfg), cfg, initial_range=tr, **kwargs)


def phase(df, result, direction="unknown"):
    return classify_phase(result["events"], result["tr"], [], df.Close.iloc[-1],
                          context={"price_structure": {"direction": direction},
                                   "stopping_candidate": result["stopping_candidate"]})


def gap_climax():
    df = bars()
    for i, c in enumerate((112, 114, 116, 118, 120), start=30):
        put(df, i, c + 1, c - 1, c)
    df.iloc[35] = [126, 126.5, 124.5, 125, 1000]
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_gap_climax_uses_preceding_true_range_baseline_and_keeps_old_cycle(bull):
    df = mirror(gap_climax()) if bull else gap_climax()
    f = compute_features(df)
    assert f.spread_ratio.iloc[35] < CoreConfig().spread_high < f.true_range_ratio.iloc[35]
    result = replay(df.iloc[:36])
    e = next(e for e in result["events"] if e["event"] == ("SC" if bull else "BC"))
    assert e["climax_path"] == "gap_true_range"
    assert available(e) == df.index[35]
    assert phase(df.iloc[:36], result)["phase"] == "A"
    assert result["tr"].range_id == "events-v3"
    assert e["range_id"] != result["tr"].range_id


@pytest.mark.parametrize("counterexample", ["no_gap", "low_volume", "no_rejection"])
def test_gap_alone_or_quiet_or_strong_close_does_not_make_climax(counterexample):
    df = gap_climax()
    if counterexample == "no_gap":
        df.iloc[35] = [120, 121, 119, 119.5, 1000]
    elif counterexample == "low_volume":
        df.iloc[35, df.columns.get_loc("Volume")] = 100
    else:
        df.iloc[35, df.columns.get_loc("Close")] = 126.4
    assert not any(e["event"] == "BC" for e in replay(df.iloc[:36])["events"])


@pytest.mark.parametrize("bull", [True, False])
def test_gap_climax_ar_and_secondary_test_compare_the_same_spread_measure(bull):
    df = gap_climax()
    put(df, 36, 121, 116, 118)
    put(df, 37, 122, 117, 121)
    for i in range(38, 45):
        put(df, i, 123, 121, 122)
    put(df, 45, 126, 123, 125, 50)  # Intrabar 3 > climax 2, true range 4 < climax 6.5.
    if bull:
        df = mirror(df)
    result = replay(df.iloc[:46])
    assert result["tr"].lifecycle == "active"
    assert any(e["event"] == "ST" and e["date"] == df.index[45] for e in result["events"])
    assert (phase(df.iloc[:46], result)["phase"], phase(df.iloc[:46], result)["regime"]) == (
        "B", "accumulation" if bull else "distribution")


def recovery():
    df = bars()
    put(df, 30, 99, 97, 98, 150)
    put(df, 31, 98, 96, 97)
    put(df, 32, 103, 98, 102, 80)
    put(df, 36, 102, 100.5, 101, 30)
    put(df, 37, 104, 102, 103)
    put(df, 38, 106, 104, 105)
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_multi_day_recovery_is_known_on_reclaim_not_extreme_and_c_waits_for_test(bull):
    df = recovery() if bull else mirror(recovery())
    early = replay(df.iloc[:32])
    assert not any(e["event"] in ("SPRING", "UT", "UTAD") for e in early["events"])
    recovered = replay(df.iloc[:33])
    probe = next(e for e in recovered["events"] if e.get("subtype") == "multi_bar_recovery")
    assert (probe["date"], available(probe)) == (df.index[31], df.index[32])
    assert phase(df.iloc[:33], recovered)["phase"] == "B"
    full = replay(df.iloc[:39])
    test = next(e for e in full["events"] if e["event"] == ("TEST" if bull else "TEST_DEMAND"))
    assert (test["date"], available(test)) == (df.index[36], df.index[38])
    assert test["parent_date"] == probe["date"]
    assert phase(df.iloc[:39], full)["phase"] == "C"


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("bad", ["expired", "zero_volume", "high_recovery_effort", "weak_recovery", "opposite_boundary"])
def test_unconfirmed_or_failed_recovery_cannot_make_spring_or_c(bull, bad):
    df = recovery()
    end = 33
    if bad == "expired":
        for i in range(31, 37):
            put(df, i, 99, 96.5, 98)
        put(df, 37, 104, 98, 103, 80)
        end = 38
    elif bad == "zero_volume":
        df.iloc[31, df.columns.get_loc("Volume")] = 0
    elif bad == "high_recovery_effort":
        df.iloc[32, df.columns.get_loc("Volume")] = 500
    elif bad == "weak_recovery":
        put(df, 32, 103, 99, 100.1, 80)
    else:
        put(df, 32, 112, 98, 105, 80)
    if not bull:
        df = mirror(df)
    result = replay(df.iloc[:end])
    assert not any(e.get("subtype") == "multi_bar_recovery" for e in result["events"])
    assert phase(df.iloc[:end], result)["phase"] != "C"
    assert any(e["event"] == "INVALIDATION" for e in result["events"])


def wave():
    df = bars()
    for i, c in enumerate([105, 106, 107, 106, 104.5, 103.5, 103, 105, 107, 109, 111, 113], 26):
        put(df, i, c + 1, c - 1, c, 110)
    put(df, 32, 104, 101.5, 103, 100)
    put(df, 38, 113, 109, 110)
    put(df, 39, 110, 107, 108)
    put(df, 40, 108, 106, 107, 40)
    put(df, 41, 112, 108, 111)
    put(df, 42, 115, 112, 114)
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_multi_bar_push_can_be_below_mid_and_requires_later_support_for_d(bull):
    df = wave() if bull else mirror(wave(), 230)
    f = compute_features(df)
    assert (f.volume_ratio.iloc[33:38] < CoreConfig().volume_high).all()
    result = replay(df.iloc[:38], top=130)
    event = next(e for e in result["events"] if e.get("subtype") == "multi_bar_initiative")
    assert event["event"] == ("SOS" if bull else "SOW")
    assert event["scope"] == "internal"
    assert (event["price"] < result["tr"].mid) == bull
    assert event["reference_available_date"] <= event["wave_start_date"] < available(event)
    assert phase(df.iloc[:38], result, "up" if bull else "down")["phase"] == "B"
    full = replay(df.iloc[:43], top=130)
    assert any(e["event"] == ("LPS" if bull else "LPSY") for e in full["events"])
    assert phase(df.iloc[:43], full, "up" if bull else "down")["phase"] == "D"


@pytest.mark.parametrize("bad", ["low_effort", "zero_volume", "ordinary_swing", "no_reference", "too_slow"])
def test_drift_or_unsupported_wave_does_not_make_initiative(bad):
    df = wave()
    if bad == "low_effort":
        df.iloc[33:38, df.columns.get_loc("Volume")] = 40
    elif bad == "zero_volume":
        df.iloc[34, df.columns.get_loc("Volume")] = 0
    elif bad == "ordinary_swing":
        for i, c in enumerate((104, 106, 107, 108.5, 108), 33):
            put(df, i, c + 1, c - 1, c)
    elif bad == "no_reference":
        for i in range(26, 31):
            put(df, i, 106, 104, 105)
    cfg = CoreConfig(local_trend_bars=3) if bad == "too_slow" else None
    result = replay(df.iloc[:38], cfg, top=130)
    assert not any(e.get("subtype") == "multi_bar_initiative" for e in result["events"])


@pytest.mark.parametrize("bull", [True, False])
def test_broken_support_invalidates_newer_initiative_in_runtime_and_ledger(bull):
    df = wave()
    put(df, 43, 126, 113, 124, 300)
    put(df, 44, 110, 104, 106, 0)
    if not bull:
        df = mirror(df, 230)
    before = replay(df.iloc[:44], top=130)
    full = replay(df.iloc[:45], top=130)
    name = "SOS" if bull else "SOW"
    assert any(e["event"] == name and e["date"] == df.index[43] for e in before["events"])
    assert any(e.get("invalidated_event") == name and e.get("invalidated_event_date") == df.index[43]
               and e.get("reason") == "support_for_active_initiative_violated" for e in full["events"])
    assert not sequence_state(range_events(full["events"], full["tr"]))["sos_confirmed" if bull else "sow_confirmed"]
    assert phase(df.iloc[:45], full, "up" if bull else "down")["phase"] == "B"


@pytest.mark.parametrize("path", ["recovery", "wave"])
def test_unfinished_new_evidence_has_a_finite_confirmation_window(path):
    df = recovery() if path == "recovery" else wave()
    start = 33 if path == "recovery" else 38
    value = 105 if path == "recovery" else 114
    for i in range(start, len(df)):
        put(df, i, value + 1, value - 1, value)
    result = replay(df, CoreConfig(climax_sequence_max_bars=10), top=110 if path == "recovery" else 130)
    reason = "probe_test_window_expired" if path == "recovery" else "initiative_expired"
    assert any(e.get("reason") == reason for e in result["events"])
    state = sequence_state(range_events(result["events"], result["tr"]))
    assert state["acc_spring"] is None and not state["sos_confirmed"]


@pytest.mark.parametrize("source,top", [(recovery, 110), (wave, 130), (gap_climax, 110)])
def test_all_prefixes_and_future_changes_preserve_event_observation_dates(source, top):
    df = source()
    observed = {}
    full = replay(df, top=top, on_bar=lambda i, state: observed.setdefault(i, deepcopy(state)))
    for end in range(26, len(df) + 1):
        prefix = replay(df.iloc[:end], top=top)
        assert prefix["tr"] == observed[end - 1]["tr"]
        assert prefix["events"] == [e for e in full["events"] if available(e) <= df.index[end - 1]]
        assert prefix["transitions"] == observed[end - 1]["transitions"]
    later = df.copy()
    later.iloc[43:, :4] *= .3
    later.iloc[43:, 4] *= 100
    altered = replay(later, top=top)
    assert [e for e in altered["events"] if available(e) <= df.index[42]] == replay(df.iloc[:43], top=top)["events"]
