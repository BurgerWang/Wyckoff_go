"""Superseded tests stay historical when a newer same-cycle chain fails."""
import pytest

from analysis import analyze_frame
from test_core_usability_v8 import establish_range
from test_phase_c_contract_v5 import bars, boundary_test, mirror, put
from wyckoff_core.features import compute_features
from wyckoff_core.models import CoreConfig
from wyckoff_core.structure import available
from wyckoff_core.timeline import stage_timeline


def replacement_chain():
    df = establish_range(boundary_test())
    put(df, 36, 100.8, 99.9, 100.2, 30)
    for row in ((41, 102, 99.63, 101, 70), (47, 102, 100.3, 101, 30),
                (48, 105, 103, 104, 100), (49, 106, 104, 105, 100),
                (50, 100.7, 99.7, 100, 0), (51, 105, 103, 104, 100),
                (52, 106, 104, 105, 100)):
        put(df, *row)
    return df.iloc[:55]


@pytest.mark.parametrize("bull", [True, False])
def test_new_probe_replaces_c_and_new_test_failure_cannot_revive_old_test(bull):
    df = replacement_chain()
    if not bull:
        df = mirror(df)
    key = "acc_test" if bull else "dist_test"
    name = "TEST" if bull else "TEST_DEMAND"
    timeline = stage_timeline(df)
    phases = {38: "C", 41: "B", 47: "B", 48: "B", 49: "C", 50: "B", 52: "B", 54: "B"}
    snapshots = {}
    for i, expected in phases.items():
        result = analyze_frame(df.iloc[:i + 1], kinds=["atr", "trad"])
        core = snapshots[i] = result["analyses"]["atr"]["wyckoff"]
        assert core["phase"] == result["analyses"]["trad"]["wyckoff"]["phase"]
        assert core["phase"]["phase"] == expected
        assert core["phase"] == timeline["rows"][i]["phase"]
        assert core["events"] == [e for e in timeline["events"] if available(e) <= df.index[i]]
        assert core["tr"].range_id == snapshots[38]["tr"].range_id
    old = snapshots[38]["sequence"][key]
    new = snapshots[49]["sequence"][key]
    assert (old["date"], available(old), old["parent_date"]) == (df.index[36], df.index[38], df.index[30])
    assert (new["date"], available(new), new["parent_date"]) == (df.index[47], df.index[49], df.index[41])
    # The second probe supersedes the old chain without crossing its buffered
    # extreme. This isolates replacement from the older failure/new-chain tests.
    assert not [e for e in snapshots[49]["events"] if e["event"] == "INVALIDATION"]
    for i in (41, 47, 48, 50, 52, 54):
        assert snapshots[i]["sequence"][key] is None
    failure = snapshots[50]
    loss = [e for e in failure["events"] if e["event"] == "INVALIDATION"]
    assert len(loss) == 1
    assert (loss[0]["invalidated_event"], loss[0]["invalidated_event_date"], available(loss[0])) == (
        name, new["date"], df.index[50])
    assert loss[0]["reason"] == "confirmed_support_extreme_violated"
    buffer = CoreConfig().boundary_buffer_atr * compute_features(df).atr.iloc[50]
    assert df.Low.iloc[50] >= old["price"] - buffer if bull else df.High.iloc[50] <= old["price"] + buffer
    assert df.Volume.iloc[50] == 0 and failure["last_price"] == (100 if bull else 110)
    for i in (50, 52, 54):
        # Historical confirmed records remain; a valid-looking old record is
        # not selected again after the latest chain's terminal invalidation.
        assert [e for e in snapshots[i]["events"] if e["event"] == name] == [old, new]
    future = df.copy()
    future.iloc[51:, :4] *= 0.1 if bull else 3
    future.iloc[51:, 4] *= 30
    altered = stage_timeline(future)
    assert altered["rows"][:51] == timeline["rows"][:51]
    assert [e for e in altered["events"] if available(e) <= df.index[50]] == failure["events"]


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("kind", ["probe", "initiative"])
def test_expiry_on_new_evidence_bar_targets_only_the_old_chain(bull, kind):
    df = establish_range(bars(101))
    if kind == "probe":
        put(df, 30, 102, 97, 101, 70)
        rows = ((91, 102, 98, 101, 70), (97, 102, 100.5, 101, 30),
                (98, 105, 103, 104, 100), (99, 106, 104, 105, 100))
        ends = (91, 92, 98, 99, 100)
        parent_key = "acc_spring" if bull else "dist_ut"
        child_key = "acc_test" if bull else "dist_test"
        parent_name = "SPRING" if bull else "UT"
        reason = "probe_test_window_expired"
        action, confirmation = 97, 99
    else:
        put(df, 30, 110, 104, 109, 300)
        rows = ((90, 102, 100.5, 101, 40), (91, 110, 104, 109, 300),
                (92, 109, 106, 108, 100), (93, 105, 103, 104, 40),
                (94, 108, 105, 107, 100), (95, 109, 106, 108, 100))
        ends = (91, 92, 93, 94, 95, 96)
        parent_key = "acc_sos" if bull else "dist_sow"
        child_key = "acc_lps" if bull else "dist_lpsy"
        parent_name = "SOS" if bull else "SOW"
        reason = "initiative_expired"
        action, confirmation = 93, 95
    for row in rows:
        put(df, *row)
    if not bull:
        df = mirror(df)
    timeline = stage_timeline(df)
    snapshots = {}
    for end in ends:
        core = snapshots[end] = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert core["phase"] == timeline["rows"][end - 1]["phase"]
        assert core["events"] == [e for e in timeline["events"] if available(e) <= df.index[end - 1]]
        assert core["tr"].range_id == snapshots[91]["tr"].range_id
        if end <= confirmation:
            assert core["sequence"][child_key] is None
        else:
            child = core["sequence"][child_key]
            assert (child["date"], available(child), child["parent_date"]) == (
                df.index[action], df.index[confirmation], df.index[91])
    # The default 60-bar deadline is inclusive; expiry is on bar 91, when
    # the fresh same-side probe/initiative is also installed.
    assert snapshots[91]["sequence"][parent_key]["date"] == df.index[30]
    assert not [e for e in snapshots[91]["events"] if e["event"] == "INVALIDATION"]
    expiry = [e for e in snapshots[92]["events"] if e["event"] == "INVALIDATION"]
    assert len(expiry) == 1
    assert (expiry[0]["invalidated_event"], expiry[0]["invalidated_event_date"], available(expiry[0])) == (
        parent_name, df.index[30], df.index[91])
    assert expiry[0]["reason"] == reason
    assert snapshots[92]["sequence"][parent_key]["date"] == df.index[91]
    if kind == "probe":
        assert snapshots[99]["phase"]["phase"] == "B"
        assert snapshots[100]["phase"]["phase"] == "C"
    else:
        # Action bar 90 belongs before the new initiative even though its
        # pivot confirmation is on 92; only the later action at 93 may attach.
        pivot = next(p for p in snapshots[93]["swings"] if p["date"] == df.index[90]
                     and p["kind"] == ("low" if bull else "high"))
        assert pivot["available_date"] == df.index[92]
        assert snapshots[93]["sequence"][child_key] is None
        assert snapshots[96]["sequence"]["acc_complete" if bull else "dist_complete"]
    future = df.copy()
    future.iloc[confirmation + 1:, :4] *= 3 if bull else 0.1
    future.iloc[confirmation + 1:, 4] *= 30
    altered = stage_timeline(future)
    assert altered["rows"][:confirmation + 1] == timeline["rows"][:confirmation + 1]
