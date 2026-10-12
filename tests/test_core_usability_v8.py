"""Production-path counterexamples for evidence ownership, not truth labels."""
from copy import deepcopy

import pytest

from analysis import analyze_frame
from test_core_cycles_v3 import new_cycle
from test_phase_c_contract_v5 import bars, boundary_test, mirror, put
from wyckoff_core.context import analyze_context
from wyckoff_core.engine import report
from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.stages import range_events, sequence_state
from wyckoff_core.structure import available, replay_structure
from wyckoff_core.timeline import stage_timeline


def establish_range(df):
    for i in range(28):
        close = [105, 107, 109, 107, 105, 103, 101, 103][i % 8]
        put(df, i, close + 1, close - 1, close)
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_excursion_invalidation_does_not_invalidate_a_same_date_confirmed_test(bull):
    df = establish_range(boundary_test())
    put(df, 36, 100, 99.4, 99.7, 30)
    put(df, 37, 100.1, 99.5, 99.8)
    put(df, 38, 100.2, 99.6, 100.02)
    # An actual later loss must still clear the test, unlike the excursion's
    # independent expiry on the test's confirmation bar.
    put(df, 39, 100.2, 99.1, 99.3)
    if not bull:
        df = mirror(df)
    timeline = stage_timeline(df)
    name, key = ("TEST", "acc_test") if bull else ("TEST_DEMAND", "dist_test")
    before = analyze_frame(df.iloc[:39])["analyses"]["atr"]["wyckoff"]
    test = before["sequence"][key]
    assert test is not None and test["event"] == name
    assert (test["date"], available(test)) == (df.index[36], df.index[38])
    expiry = next(e for e in before["events"] if e["event"] == "INVALIDATION")
    assert expiry["invalidated_event"] == "BOUNDARY_EXCURSION"
    assert expiry["invalidated_event_date"] == test["date"]
    assert before["phase"]["phase"] == "C"
    assert before["phase"] == timeline["rows"][38]["phase"]
    loss_bar = analyze_frame(df.iloc[:40])["analyses"]["atr"]["wyckoff"]
    assert loss_bar["sequence"][key] is None and loss_bar["phase"]["phase"] != "C"
    assert loss_bar["phase"] == timeline["rows"][39]["phase"]
    after = analyze_frame(df.iloc[:41])["analyses"]["atr"]["wyckoff"]
    assert after["sequence"][key] is None
    assert after["phase"]["phase"] == "B"
    assert after["phase"] == timeline["rows"][40]["phase"]
    loss = [e for e in after["events"] if e["event"] == "INVALIDATION"][-1]
    assert (loss["invalidated_event"], loss["invalidated_event_date"]) == (name, test["date"])
    assert before["events"] == [e for e in after["events"] if available(e) <= df.index[38]]
    for end in (36, 37, 38, 39, 40, 43):
        snap = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert snap["events"] == [e for e in timeline["events"] if available(e) <= df.index[end - 1]]
        assert snap["phase"] == timeline["rows"][end - 1]["phase"]
    changed = df.copy()
    changed.iloc[40:, :4] *= 3
    changed.iloc[40:, 4] *= 20
    assert stage_timeline(changed)["rows"][:40] == timeline["rows"][:40]
    text = report(after)
    assert f"失效对象 {name} @ {test['date']}" in text
    assert f"原因 {loss['reason']}" in text


@pytest.mark.parametrize("bull", [True, False])
def test_outside_bar_preserves_previous_ordered_structure_and_raw_pivot_ledger(bull):
    df = bars(50)
    df.loc[:, ["Open", "Close"]] = 100.
    df.loc[:, "High"] = 101.
    df.loc[:, "Low"] = 99.
    for i, high, low, close in ((22, 95, 90, 93), (26, 108, 103, 107),
                               (30, 99, 95, 98), (34, 112, 105, 110),
                               (38, 118, 85, 108)):
        put(df, i, high, low, close)
    if not bull:
        df = mirror(df)
    prefix = df.iloc[:41]
    pivots = confirmed_swings(prefix)
    dual = [p for p in pivots if p["date"] == df.index[38]]
    assert {p["kind"] for p in dual} == {"high", "low"}
    assert {p["available_date"] for p in dual} == {df.index[40]}
    saved = deepcopy(pivots)
    context = analyze_context(prefix, compute_features(prefix), pivots)
    structure = context["price_structure"]
    assert structure["direction"] == ("up" if bull else "down")
    pivot = structure["last_high" if bull else "last_low"]
    assert pivot["date"] == df.index[34]
    assert pivot["price"] == (112 if bull else 98)
    assert pivots == saved
    # Compare independent prefixes around both confirmations, rather than
    # trusting the shared replay to certify its own historical output.
    timeline = stage_timeline(df)
    for end in range(34, 43):
        snap = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert snap["phase"] == timeline["rows"][end - 1]["phase"]
    changed = df.copy()
    changed.iloc[41:, :4] *= .2
    changed.iloc[41:, 4] *= 100
    assert stage_timeline(changed)["rows"][:41] == timeline["rows"][:41]


@pytest.mark.parametrize("bull", [True, False])
def test_new_stopping_cycle_owns_sequence_counts_checklist_and_report(bull):
    df = establish_range(new_cycle())
    put(df, 30, 103, 96, 98, 300)
    if not bull:
        df = mirror(df)
    before = analyze_frame(df.iloc[:36])["analyses"]["atr"]["wyckoff"]
    assert before["tr"] is not None and before["tr"].lifecycle in ("active", "breakout_up", "breakout_down")
    assert before["readiness"]["nine_buying"][7]["status"] == "met"
    assert before["sequence"]["sow_confirmed" if bull else "sos_confirmed"]
    result = analyze_frame(df.iloc[:37])["analyses"]["atr"]["wyckoff"]
    candidate = result["stopping_candidate"]
    assert candidate is not None
    assert result["phase"]["phase"] == "A"
    assert result["analysis_cycle_id"] == candidate["range_id"]
    assert result["analysis_range_id"] is None
    assert result["tr"].range_id == candidate["parent_range_id"]
    assert result["phase"]["prior_range_phase"]["range_id"] == result["tr"].range_id
    assert result["targets"] is None
    assert result["count"]["reason"] == "current_stopping_cycle_has_no_established_range"
    assert result["readiness"]["nine_buying"][7]["status"] == "not_met"
    assert not result["sequence"]["sow_confirmed"] and not result["sequence"]["sos_confirmed"]
    current = [e for e in result["events"] if e.get("range_id") == candidate["range_id"]]
    assert result["sequence"] == sequence_state(current)
    assert before["events"] == [e for e in result["events"] if available(e) <= df.index[35]]
    text = report(result)
    assert "父周期保留 区间" in text and "父周期阶段:" in text
    assert f"当前停止周期: {candidate['range_id']}" in text
    assert f"{'SC' if bull else 'BC'} @" in text.split("当前停止周期事件")[1]
    formed = analyze_frame(df.iloc[:39])["analyses"]["atr"]["wyckoff"]
    assert formed["stopping_candidate"] is None
    assert formed["analysis_cycle_id"] == formed["analysis_range_id"] == candidate["range_id"]
    assert formed["sequence"] == sequence_state(range_events(formed["events"], formed["tr"]))
    # If the candidate fails instead, the retained parent resumes ownership.
    failed = df.copy()
    if bull:
        put(failed, 37, 75, 72, 73)
    else:
        put(failed, 37, 138, 135, 137)
    resumed = analyze_frame(failed.iloc[:38])["analyses"]["atr"]["wyckoff"]
    assert resumed["stopping_candidate"] is None
    assert resumed["analysis_cycle_id"] == resumed["analysis_range_id"] == result["tr"].range_id
    assert resumed["sequence"] == sequence_state(range_events(resumed["events"], resumed["tr"]))
    assert resumed["readiness"]["nine_buying"][7]["status"] == "met"
    features = compute_features(df)
    assert replay_structure(df, features, confirmed_swings(df))["events"] == stage_timeline(df)["events"]
