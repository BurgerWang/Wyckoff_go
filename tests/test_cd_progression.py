"""C/D progression and count revocation through the automatic production path."""
import numpy as np
import pandas as pd
import pytest

import pnf
from analysis import analyze_frame, report
from test_core_usability_v8 import establish_range
from test_phase_c_contract_v5 import bars, boundary_test, mirror, put
from wyckoff_core.engine import analyze
from wyckoff_core.features import confirmed_swings
from wyckoff_core.structure import available
from wyckoff_core.timeline import stage_timeline


def internal_chain():
    df = establish_range(boundary_test())
    for row in ((34, 108, 104, 107, 100), (39, 110, 104, 109, 300),
                (40, 109, 106, 107, 100), (41, 104.5, 103, 104, 40),
                (42, 108.5, 105, 108, 100), (43, 109.5, 107, 109, 100),
                (44, 110, 108, 109, 100), (45, 107, 101, 104, 0),
                (46, 108, 105, 107, 100), (47, 109, 106, 108, 100)):
        put(df, *row)
    return df


def fixed_snapshot(df):
    grid = np.arange(80., 130., .5)
    chart = pnf.build(df, grid)
    return analyze(chart.columns, grid, df)


def new_stopping_range():
    df = bars(57)
    for i in range(30):
        put(df, i, 91, 89, 90)
    for i, high, low, close in ((20, 92, 88, 90), (22, 90, 84, 87), (26, 95, 89, 93)):
        put(df, i, high, low, close)
    for i, close in enumerate([90, 88, 86, 84, 82, 80], 30):
        put(df, i, close + 1, close - 1, close)
    put(df, 36, 85, 75, 82, 1000)
    put(df, 37, 100, 81, 99)
    put(df, 38, 99, 85, 90)
    for i in range(39, 47):
        put(df, i, 93, 88, 90)
    for row in ((46, 80, 75.5, 78, 50), (47, 93, 87, 91, 100), (48, 94, 89, 92, 100),
                (49, 98, 80, 97, 300), (50, 97.8, 93, 96, 100), (51, 97.5, 94, 95, 100),
                (52, 97, 94.5, 96, 100), (53, 96.5, 94, 95, 100), (54, 96, 92, 94, 40),
                (55, 98, 94, 97, 100), (56, 99, 95, 98, 100)):
        put(df, *row)
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_internal_d_cannot_borrow_a_directional_comparison_from_before_its_cycle(bull):
    df = new_stopping_range()
    if not bull:
        df = mirror(df)
    timeline = stage_timeline(df)
    result = analyze_frame(df)
    core = result["analyses"]["atr"]["wyckoff"]
    assert core["tr"].start_date == df.index[36]
    assert (core["tr"].bottom, core["tr"].top) == ((75, 100) if bull else (110, 135))
    global_structure = core["context"]["price_structure"]
    local = core["context"]["range_price_structure"]
    assert global_structure["direction"] == ("up" if bull else "down")
    assert local["reference"] == "confirmed_pivots_since_range_start"
    assert local["range_id"] == core["analysis_range_id"] and local["start_date"] == df.index[36]
    assert local["direction"] == "unknown" and local["confirmed_pivots"] == 3
    # The global HH/LH or LL/HL comparison uses a previous-cycle pivot; the
    # new cycle has only one continuation high/low after its stopping action.
    assert local["last_high" if bull else "last_low"]["date"] == df.index[37]
    side = "accumulation" if bull else "distribution"
    assert core["phase"]["phase"] == "B"
    h = core["phase"]["hypotheses"][side]
    assert "initiative_with_directional_reaction_structure" not in h["evidence"]
    assert "directional_swings" not in h["evidence"]
    text = report(result)
    assert "当前区间确认摆动: 方向证据不足" in text
    assert f"起点 {df.index[36]} | 交替支点 3" in text
    support = core["sequence"]["acc_lps" if bull else "dist_lpsy"]
    move = core["sequence"]["acc_sos" if bull else "dist_sow"]
    assert support["status"] == "confirmed" and move["scope"] == "internal"
    assert (support["date"], available(support), support["parent_date"]) == (df.index[54], df.index[56], df.index[49])
    assert [e["event"] for e in core["events"]] == (
        ["SC", "AR", "ST", "SOS", "LPS"] if bull else ["BC", "AR", "ST", "SOW", "LPSY"])
    for end in (39, 49, 50, 55, 56, 57):
        snap = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert snap["phase"] == timeline["rows"][end - 1]["phase"]
        assert snap["events"] == [e for e in timeline["events"] if available(e) <= df.index[end - 1]]
    future = pd.DataFrame([[p, p + 3, p - 3, p, 10000] for p in (55, 180, 60, 175, 65)],
                          columns=df.columns, index=pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5))
    extended = stage_timeline(pd.concat([df, future]))
    assert extended["rows"][:len(df)] == timeline["rows"]
    assert [e for e in extended["events"] if available(e) <= df.index[-1]] == timeline["events"]


@pytest.mark.parametrize("bull", [True, False])
def test_c_waits_for_internal_reaction_confirmation_then_d_revokes_on_failure(bull):
    df = internal_chain()
    if not bull:
        df = mirror(df)
    timeline = stage_timeline(df)
    key = "acc_lps" if bull else "dist_lpsy"
    move_key = "acc_sos" if bull else "dist_sow"
    for end, expected in ((38, "B"), (39, "C"), (40, "C"), (42, "C"), (43, "C"),
                          (44, "D"), (46, "B"), (48, "B")):
        core = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert core["phase"]["phase"] == expected
        assert core["phase"] == timeline["rows"][end - 1]["phase"]
        assert core["events"] == [e for e in timeline["events"] if available(e) <= df.index[end - 1]]
        if 40 <= end <= 44:
            assert core["sequence"][move_key]["scope"] == "internal"
            assert (core["sequence"][key] is not None) is (end == 44)
    supported = fixed_snapshot(df.iloc[:44])
    pivot = supported["sequence"][key]
    move = supported["sequence"][move_key]
    assert (move["date"], available(move)) == (df.index[39], df.index[39])
    assert (pivot["date"], available(pivot), pivot["parent_date"]) == (df.index[41], df.index[43], move["date"])
    assert supported["count"]["status"] == "scenario"
    assert supported["targets"]["bullish" if bull else "bearish"] is not None
    for end in (46, 48):
        failed = fixed_snapshot(df.iloc[:end])
        assert failed["sequence"][key] is None and failed["sequence"][move_key] is None
        assert failed["sequence"]["acc_test" if bull else "dist_test"] is None
        assert failed["targets"] is None
        loss = next(e for e in failed["events"] if e.get("invalidated_event") == pivot["event"])
        assert loss["invalidated_event_date"] == pivot["date"]
        assert loss["available_date"] == df.index[45]
    later = df.copy()
    later.iloc[48:, :4] *= 2
    later.iloc[48:, 4] *= 50
    assert stage_timeline(later)["rows"][:48] == timeline["rows"][:48]


@pytest.mark.parametrize("bull", [True, False])
def test_late_confirmation_cannot_attach_a_pre_initiative_pivot_as_lps(bull):
    df = internal_chain()
    put(df, 39, 110, 104, 109, 100)  # No initiative here.
    put(df, 42, 109, 104, 108, 300)  # Initiative occurs after the pivot's action.
    if not bull:
        df = mirror(df)
    prefix = df.iloc[:44]
    core = fixed_snapshot(prefix)
    move = core["sequence"]["acc_sos" if bull else "dist_sow"]
    assert (move["date"], available(move)) == (df.index[42], df.index[42])
    pivot = next(p for p in confirmed_swings(prefix)
                 if p["date"] == df.index[41] and p["kind"] == ("low" if bull else "high"))
    assert pivot["date"] < move["date"] < pivot["available_date"]
    assert core["sequence"]["acc_lps" if bull else "dist_lpsy"] is None
    assert not any(e["event"] == ("LPS" if bull else "LPSY") for e in core["events"])
    assert core["phase"]["phase"] == "C" and core["targets"] is None
    assert core["phase"] == stage_timeline(prefix)["rows"][-1]["phase"]
