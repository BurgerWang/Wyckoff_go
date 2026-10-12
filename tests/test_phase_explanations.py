"""Production explanations distinguish observed tests/structure from finality."""
import json

import pandas as pd
import pytest

from analysis import analyze_frame, report, to_json
from test_core_usability_v8 import establish_range
from test_phase_c_contract_v5 import bars, boundary_test, mirror, put
from wyckoff_core.structure import available
from wyckoff_core.timeline import stage_timeline


def supported_e():
    df = establish_range(bars(58))
    for i, high, low, close, volume in ((30, 114, 109, 113, 300), (31, 114, 111, 112, 100),
                                       (32, 109.5, 108, 109, 50), (33, 113.8, 110, 113, 100)):
        put(df, i, high, low, close, volume)
    for i in range(34, len(df)):
        put(df, i, 115 + i - 34, 111 + i - 34, 114 + i - 34)
    for i, high, low, close in ((48, 140, 135, 138), (49, 138, 132, 135), (50, 135, 130, 132),
                               (51, 140, 136, 139), (52, 143, 139, 141)):
        put(df, i, high, low, close)
    for i in range(53, len(df)):
        put(df, i, 145 + i - 53, 142 + i - 53, 144 + i - 53)
    return df


@pytest.mark.parametrize("bull", [True, False])
def test_confirmed_boundary_test_does_not_claim_a_verified_terminal_c(bull):
    df = establish_range(boundary_test())
    if not bull:
        df = mirror(df)
    before = analyze_frame(df.iloc[:38])
    assert before["analyses"]["atr"]["wyckoff"]["phase"]["phase"] == "B"
    assert "最后供需测试" not in report(before)
    timeline = stage_timeline(df)
    for end in (39, 48):
        result = analyze_frame(df.iloc[:end])
        core = result["analyses"]["atr"]["wyckoff"]
        side, key = ("accumulation", "acc_test") if bull else ("distribution", "dist_test")
        test = core["sequence"][key]
        assert test["status"] == "confirmed"
        assert (test["date"], available(test)) == (df.index[36], df.index[38])
        assert (core["phase"]["phase"], core["phase"]["status"]) == ("C", "candidate")
        assert core["phase"]["hypotheses"][side]["phase"] == "C"
        assert core["phase"] == timeline["rows"][end - 1]["phase"]
        assert "是否属于结束 B 的最后供需测试尚未独立验证" in report(result)
    # The bearish parent really is an ordinary UT, without later weakness.
    names = {e["event"] for e in core["events"]}
    assert ("SPRING" if bull else "UT") in names
    assert "UTAD" not in names and "SOS" not in names and "SOW" not in names


@pytest.mark.parametrize("bull", [True, False])
@pytest.mark.parametrize("close,status", [(125., "challenged"), (130., "at_pivot"), (135., "holding")])
def test_e_explains_current_close_against_historical_hl_or_lh(bull, close, status):
    df = supported_e()
    put(df, 57, close + 4, close - 1, close)
    if not bull:
        df = mirror(df)
    result = analyze_frame(df, kinds=["atr", "trad"])
    side = "accumulation" if bull else "distribution"
    for item in result["analyses"].values():
        core = item["wyckoff"]
        assert (core["tr"].bottom, core["tr"].top) == (100, 110)
        assert (core["phase"]["phase"], core["phase"]["status"]) == ("E", "candidate")
        h = core["phase"]["hypotheses"][side]
        assert "continuing_directional_structure" in h["evidence"]
        check = h["structure_check"]
        assert check["reference"] == "confirmed_pivots" and check["status"] == status
        assert check["last_price"] == (close if bull else 210 - close)
        assert check["reaction_pivot"] == {
            "kind": "low" if bull else "high", "price": 130 if bull else 80,
            "date": df.index[50], "available_date": df.index[52]}
        support = core["sequence"]["acc_lps" if bull else "dist_lpsy"]
        assert support["price"] == (108 if bull else 102)
        assert not any(e.get("invalidated_event") == support["event"]
                       and e.get("invalidated_event_date") == support["date"] for e in core["events"])
        assert core["phase"] == stage_timeline(df)["rows"][-1]["phase"]
    text = report(result)
    assert "E 的延续结构证据来自已确认的历史摆动" in text
    assert f"当前收盘 {df.Close.iloc[-1]:.4g}" in text
    assert "动作 2025-02-20" in text and "确认 2025-02-22" in text
    relation = "低于" if bull else "高于"
    if status == "at_pivot":
        relation = "等于"
    elif status == "holding":
        relation = "高于" if bull else "低于"
    assert f"{relation}最近确认{'低点 130' if bull else '高点 80'}" in text
    assert ("当前收盘与已确认方向结构存在冲突" in text) is (status == "challenged")
    serialized = json.loads(to_json(result))
    check = serialized["analyses"]["atr"]["wyckoff"]["phase"]["hypotheses"][side]["structure_check"]
    assert check["reaction_pivot"]["available_date"] == df.index[52].isoformat()


@pytest.mark.parametrize("bull", [True, False])
def test_e_close_explanation_uses_only_known_pivots_and_cannot_rewrite_history(bull):
    df = supported_e()
    put(df, 57, 129, 124, 125)
    if not bull:
        df = mirror(df)
    timeline = stage_timeline(df)
    side = "accumulation" if bull else "distribution"
    for end in (51, 52, 53, 57, 58):
        core = analyze_frame(df.iloc[:end])["analyses"]["atr"]["wyckoff"]
        assert core["phase"] == timeline["rows"][end - 1]["phase"]
        h = core["phase"]["hypotheses"].get(side, {})
        pivot = h.get("structure_check", {}).get("reaction_pivot")
        if pivot:
            assert pivot["available_date"] <= df.index[end - 1]
            assert (pivot["date"] == df.index[50]) is (end >= 53)
    future = pd.DataFrame([[30, 40, 20, 30, 10000]] * 5, columns=df.columns,
                          index=pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5))
    extended = stage_timeline(pd.concat([df, future]))
    assert extended["rows"][:len(df)] == timeline["rows"]
    assert [e for e in extended["events"] if available(e) <= df.index[-1]] == timeline["events"]
