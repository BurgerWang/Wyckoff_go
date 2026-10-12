"""V3 E facts are recomputed from the bars. Passing is geometry, never phase truth."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from research.llm_api import request_body
from research.llm_contract import (VERSION_V2, VERSION_V3, FactConstraintError, make_packet,
                                   review_e_facts, validate_response)
from research.llm_pilot import plan_asof_revision, plan_facts_revision, plan_pilot, run_pilot
from research.llm_prompt import (PROMPT_VERSION_V2_ASOF, PROMPT_VERSION_V3_FACTS, REVISION_V3, build_prompt,
                                 prepare_revision)
from test_llm_api import FakeTransport, providers, response

# open, high, low, close. Range s0002-s0012 (high 105.5 at s0006, low 94.5 at s0009),
# first close above it at s0014, rally to 110 at s0015, pullback low 106.5 at s0017,
# close back above 110 at s0019, three plainly rising bars s0020-s0022, then a dip
# that pierces 106.5 intrabar at s0024 while every close stays above it.
BARS = [
    (100, 101, 99, 100), (100, 102, 98, 99), (99, 103, 96, 102), (102, 105, 100, 104),
    (104, 104.5, 99, 100), (100, 103, 97, 102), (102, 105.5, 101, 103), (103, 104, 98, 99),
    (99, 100, 95.5, 96), (96, 99, 94.5, 98), (98, 102, 97, 101), (101, 104, 100, 103),
    (103, 105, 101, 104), (104, 105.2, 103, 104.5), (104.5, 108, 104, 107), (107, 110, 106.6, 109),
    (109, 109.5, 107, 107.5), (107.5, 108.5, 106.5, 108), (108, 109.8, 107.5, 109.5), (109.5, 112, 109, 111),
    (111, 113, 109, 112), (112, 115, 110, 114), (114, 117, 111.5, 116), (116, 116.5, 110, 111),
    (111, 111.5, 106, 108), (108, 110, 107, 109), (109, 110.5, 108, 109.5),
]
FACTS = {"side": "up",
         "range": {"start": "s0002", "end": "s0012", "upper": "s0006", "lower": "s0009"},
         "leave": "s0014",
         "pivot": {"comparison": "s0013", "reaction": "s0015", "action": "s0017", "confirmed": "s0019"},
         "support": {"reaction": "s0015", "test": "s0017", "confirmed": "s0019", "invalidated": None}}
LAST = f"s{len(BARS) - 1:04d}"


def bundle(bars=BARS, *, down=False):
    """`down` reflects every price about 100, turning the same sessions into the mirror case."""
    rows = [(200 - o, 200 - low, 200 - high, 200 - c) if down else (o, high, low, c) for o, high, low, c in bars]
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], dtype=float,
                      index=pd.date_range("2025-01-01", periods=len(rows)))
    df["Volume"] = 1000.
    made = make_packet(df, as_of=df.index[-1].date().isoformat(), information_start="2025-01-01",
                       case_ref="case-001", analysis_scale="registered_daily_window")
    made["packet"].update(version=VERSION_V3, claim_time_scope="asof_assessment")
    return made


def cite(made, component, kind="support"):
    last = len(made["packet"]["bars"]) - 1
    return {"kind": kind, "e_component": component, "statement": "工程引用夹具，不构成阶段真值",
            "happened_at": made["packet"]["as_of"], "known_at": made["packet"]["as_of"],
            "observations": [{"pointer": f"/bars/{last}/close", "value": made["packet"]["bars"][last]["close"]}]}


def answer(made, *, direction="accumulation", facts=FACTS, current="holding", lifecycle="valid"):
    p = made["packet"]
    negative = {"challenged", "invalidated"}
    claims = [cite(made, "historical_structure")]
    if current != "unavailable":
        claims.append(cite(made, "current_structure", "counterevidence" if current in negative else "support"))
    if lifecycle != "unavailable":
        claims.append(cite(made, "support_lifecycle", "counterevidence" if lifecycle in negative else "support"))
    return {"version": p["version"], "origin": "ai_reference", "case_ref": p["case_ref"], "mode": p["mode"],
            "as_of": p["as_of"], "status": "judged", "phase": "E", "direction": direction, "reason": "supported",
            "range_start": facts["range"]["start"] if facts["range"] else None, "bc_terminal_verified": False,
            "e_assessment": {"historical_structure": "supported", "current_structure": current,
                             "support_lifecycle": lifecycle},
            "claims": claims, "needs": [], "e_facts": deepcopy(facts)}


def validate(value, made):
    return validate_response(json.dumps(value), made, finish_reason="stop")


def findings(value, made):
    with pytest.raises(FactConstraintError) as caught:
        validate(value, made)
    assert caught.value.answer == value
    return {(f["class"], f["component"]) for f in caught.value.review["findings"]}


def test_true_geometry_passes_and_keeps_the_unbroken_boundary_apart_from_support():
    made = bundle()
    value = answer(made)
    assert validate(value, made) == value
    derived = review_e_facts(value, made["packet"])["derived"]
    assert derived["range"] == {"upper": 105.5, "lower": 94.5, "upper_sessions": ["s0006"],
                                "lower_sessions": ["s0009"]}
    assert derived["leave"] == {"boundary": 105.5, "first_close_beyond": "s0014"}
    assert derived["boundary_hold"] == {"from": "s0014", "through": LAST, "unbroken": True}
    assert derived["pivot"] == {"level": 106.5, "asof_close": 109.5, "relation": "holding",
                                "pierced_after_confirmation": True}
    assert derived["support"] == {"level": 106.5, "first_close_through": None}


def test_misread_range_extreme_is_a_fact_error_while_the_leave_is_judged_on_the_true_boundary():
    made = bundle()
    value = answer(made)
    value["e_facts"]["range"]["upper"] = "s0003"  # high 105 < the real 105.5 at s0006
    assert findings(value, made) == {("fact_error", "range")}
    # The close at s0014 still clears the real boundary; only the boundary citation was wrong.
    assert review_e_facts(value, made["packet"])["derived"]["leave"]["first_close_beyond"] == "s0014"
    value["e_facts"]["range"].update(upper="s0006", lower="s0008")
    assert findings(value, made) == {("fact_error", "range")}
    value["e_facts"]["range"]["lower"] = "s0009"
    value["e_facts"]["leave"] = "s0015"  # a later close beyond, not the first
    assert findings(value, made) == {("fact_error", "leave")}
    value["e_facts"]["leave"] = "s0014"
    value["range_start"] = "s0003"
    assert findings(value, made) == {("fact_error", "range")}


def test_plainly_rising_bars_are_not_a_confirmed_pivot():
    made = bundle()
    value = answer(made)
    # s0020-s0022 only step upward: s0021's low exceeds s0020's and s0022 makes a new high.
    value["e_facts"]["pivot"] = {"comparison": "s0019", "reaction": "s0020", "action": "s0021",
                                 "confirmed": "s0022"}
    assert findings(value, made) == {("fact_error", "pivot")}
    value["e_facts"]["pivot"] = {"comparison": "s0020", "reaction": "s0020", "action": "s0021",
                                 "confirmed": "s0022"}
    assert findings(value, made) == {("fact_error", "pivot")}
    # A real pullback whose comparison is not a lower prior low is rejected too.
    value["e_facts"]["pivot"] = dict(FACTS["pivot"], comparison="s0014")
    assert validate(value, made)["e_facts"]["pivot"]["comparison"] == "s0014"  # 104 < 106.5 and bounds the swing
    value["e_facts"]["pivot"] = dict(FACTS["pivot"], confirmed="s0020")
    assert findings(value, made) == {("fact_error", "pivot")}
    # s0016's high is the top of its own pullback, yet s0015 stood higher since the comparison.
    value["e_facts"]["pivot"] = dict(FACTS["pivot"], reaction="s0016")
    with pytest.raises(FactConstraintError) as caught:
        validate(value, made)
    assert [f["message"] for f in caught.value.review["findings"]] == [
        "reaction is not the extreme between comparison and action"]
    # s0016 is inside the pullback but s0017 goes lower before the confirmation.
    value["e_facts"]["pivot"] = dict(FACTS["pivot"], action="s0016")
    with pytest.raises(FactConstraintError) as caught:
        validate(value, made)
    assert [f["message"] for f in caught.value.review["findings"]] == [
        "action is not the pullback extreme between reaction and confirmation"]
    # The same pullback reaching, or only matching, the comparison low is not a higher low.
    for low in (102.5, 103.):
        not_higher = bundle(BARS[:17] + [(107.5, 108.5, low, 108)] + BARS[18:])
        with pytest.raises(FactConstraintError) as caught:
            validate(answer(not_higher), not_higher)
        assert [f["message"] for f in caught.value.review["findings"]] == [
            "action does not hold beyond a comparison pivot that bounds the swing"]


def test_an_intrabar_pierce_is_not_challenged_while_the_asof_close_holds():
    made = bundle()
    assert findings(answer(made, current="challenged"), made) == {("fact_error", "current_structure")}
    assert findings(answer(made, current="at_pivot"), made) == {("fact_error", "current_structure")}


def test_a_close_through_challenges_the_pivot_and_invalidates_the_support():
    made = bundle(BARS[:-1] + [(109, 110.5, 105.8, 106)])
    facts = deepcopy(FACTS)
    facts["support"]["invalidated"] = LAST
    value = answer(made, facts=facts, current="challenged", lifecycle="invalidated")
    assert validate(value, made) == value
    derived = review_e_facts(value, made["packet"])["derived"]
    assert derived["pivot"]["relation"] == "challenged" and derived["support"]["first_close_through"] == LAST
    assert findings(answer(made, facts=facts, current="holding", lifecycle="invalidated"), made) == {
        ("fact_error", "current_structure")}
    assert findings(answer(made, facts=facts, current="challenged", lifecycle="valid"), made) == {
        ("fact_error", "support_lifecycle")}
    # Declaring no invalidation contradicts the close; the lifecycle is then not judged at all.
    assert findings(answer(made, current="challenged", lifecycle="valid"), made) == {("fact_error", "support")}
    level = bundle(BARS[:-1] + [(109, 110.5, 105.8, 106.5)])
    assert validate(answer(level, current="at_pivot"), level)["e_assessment"]["current_structure"] == "at_pivot"


def test_an_unbroken_boundary_cannot_stand_in_for_a_confirmed_support():
    made = bundle()
    facts = dict(FACTS, support=None)
    value = answer(made, facts=facts)
    assert findings(value, made) == {("evidence_gap", "support_lifecycle")}
    assert review_e_facts(value, made["packet"])["derived"]["boundary_hold"]["unbroken"] is True
    # Declining to assess is legal and still lets a supported phase stand.
    abstained = answer(made, facts=dict(facts, pivot=None), current="unavailable", lifecycle="unavailable")
    assert validate(abstained, made)["phase"] == "E"
    # The reaction must be the extreme the pullback came off, not an earlier lesser high.
    # s0014's high 108 is exceeded at s0015 before the pullback; every other condition holds.
    off_peak = deepcopy(FACTS)
    off_peak["support"].update(reaction="s0014", confirmed="s0018")
    with pytest.raises(FactConstraintError) as caught:
        validate(answer(made, facts=off_peak), made)
    assert [f["message"] for f in caught.value.review["findings"]] == [
        "reaction is not the extreme before the pullback"]
    # A support that predates its target range cannot be attributed to it.
    early = deepcopy(FACTS)
    early["range"].update(start="s0016", end="s0018", upper="s0016", lower="s0017")
    early["leave"] = "s0019"
    assert ("fact_error", "support") in findings(answer(made, facts=early), made)


def test_missing_declarations_are_gaps_not_fact_errors():
    made = bundle()
    assert findings(answer(made, facts=dict(FACTS, pivot=None)), made) == {("evidence_gap", "current_structure")}
    bare = {"side": None, "range": None, "leave": None, "pivot": None, "support": None}
    assert findings(answer(made, facts=bare, current="unavailable", lifecycle="unavailable"), made) == {
        ("evidence_gap", "historical_structure")}
    sideless = dict(FACTS, side=None)
    assert findings(answer(made, facts=sideless), made) == {
        ("evidence_gap", "leave"), ("evidence_gap", "pivot"), ("evidence_gap", "support"),
        ("evidence_gap", "historical_structure")}
    # An abstention with nothing declared has nothing to check.
    p = made["packet"]
    unresolved = answer(made, facts=bare, current="unavailable", lifecycle="unavailable")
    unresolved.update(status="unresolved", phase=None, direction=None, reason="missing_context", claims=[],
                      needs=["需要左侧停止及区间背景"])
    unresolved["e_assessment"]["historical_structure"] = "unassessed"
    assert validate(unresolved, made)["as_of"] == p["as_of"]


def test_the_down_side_applies_the_mirrored_conditions():
    made = bundle(down=True)
    facts = deepcopy(FACTS)
    facts["side"] = "down"
    facts["range"].update(upper="s0009", lower="s0006")
    value = answer(made, direction="distribution", facts=facts)
    assert validate(value, made) == value
    derived = review_e_facts(value, made["packet"])["derived"]
    assert derived["range"]["lower"] == 94.5 and derived["leave"]["boundary"] == 94.5
    assert derived["pivot"] == {"level": 93.5, "asof_close": 90.5, "relation": "holding",
                                "pierced_after_confirmation": True}
    facts["range"]["lower"] = "s0003"
    assert findings(answer(made, direction="distribution", facts=facts), made) == {("fact_error", "range")}
    assert findings(answer(made, direction="distribution", current="challenged",
                           facts=dict(deepcopy(FACTS), side="down",
                                      range={"start": "s0002", "end": "s0012", "upper": "s0009",
                                             "lower": "s0006"})), made) == {("fact_error", "current_structure")}


def test_malformed_facts_fail_the_contract_before_any_fact_review():
    made = bundle()
    value = answer(made)
    value["e_facts"]["pivot"]["action"] = "s0099"
    with pytest.raises(ValueError, match="outside input") as caught:
        validate(value, made)
    assert not isinstance(caught.value, FactConstraintError)
    with pytest.raises(ValueError, match="contradicts the judged direction"):
        validate(answer(made, facts=dict(FACTS, side="down")), made)
    value = answer(made)
    del value["e_facts"]
    with pytest.raises(ValueError, match="missing or extra"):
        validate(value, made)
    value = answer(made)
    value["e_facts"]["range"]["middle"] = "s0005"
    with pytest.raises(ValueError, match="missing or extra"):
        validate(value, made)
    unscoped = bundle()
    del unscoped["packet"]["claim_time_scope"]
    with pytest.raises(ValueError, match="as-of claim scope"):
        validate(answer(unscoped), unscoped)
    # Earlier contracts neither require nor accept the new field.
    older = bundle()
    older["packet"]["version"] = VERSION_V2
    value = answer(older)
    with pytest.raises(ValueError, match="missing or extra"):
        validate(value, older)
    del value["e_facts"]
    assert validate(value, older)["version"] == VERSION_V2


def source_items(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    originals = {}
    for ref, market in (("case-001", "us_equities"), ("case-025", "binance_spot")):
        made = bundle()
        made["packet"].update(version="wyckoff-ai-v1", case_ref=ref)
        del made["packet"]["claim_time_scope"]
        made["local"].update(role="development", source_case={
            "role": "development", "label_mode": "asof", "market": market, "as_of": made["local"]["as_of"]})
        data = {"bundle": made, "prompts": {v: build_prompt(made, variant=v) for v in ("baseline", "contrast")}}
        path = source / (ref + ".json")
        originals[path] = json.dumps(data)
        path.write_text(originals[path])
    return source, originals


def test_the_facts_prompt_is_a_new_version_and_earlier_constructions_do_not_move(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    source, originals = source_items(tmp_path)
    older = tmp_path / "asof"
    prepare_revision(source, older, asof_assessment=True)
    before = {p.name: p.read_text() for p in older.glob("case-*.json")}
    out = tmp_path / "facts"
    summary = prepare_revision(source, out, e_facts=True)
    assert (summary["prompt_version"], summary["contract_version"], summary["api_calls"]) == (
        PROMPT_VERSION_V3_FACTS, VERSION_V3, 0)
    assert all(path.read_text() == original for path, original in originals.items())
    for name, text in before.items():
        scoped = json.loads(text)
        assert build_prompt(scoped["bundle"], version=PROMPT_VERSION_V2_ASOF) == scoped["prompts"]["baseline"]
        assert "e_facts" not in text and REVISION_V3 not in scoped["prompts"]["baseline"]["messages"][0]["content"]
        new = json.loads((out / name).read_text())
        assert new["bundle"]["packet"]["version"] == VERSION_V3
        assert new["bundle"]["packet"]["claim_time_scope"] == "asof_assessment"
        assert new["bundle"]["packet"]["bars"] == scoped["bundle"]["packet"]["bars"]
        prompt = new["prompts"]["baseline"]
        assert prompt["prompt_version"] == PROMPT_VERSION_V3_FACTS
        assert set(prompt["response_schema"]["properties"]["e_facts"]["properties"]) == set(FACTS)
        system = prompt["messages"][0]["content"]
        assert REVISION_V3 in system
        example, _ = json.JSONDecoder().raw_decode(system.split("FORMAT_EXAMPLE:\n", 1)[1])
        assert validate(example, new["bundle"])["e_facts"]["range"] is None
    with pytest.raises(ValueError, match="versions differ"):
        build_prompt(json.loads(next(iter(before.values())))["bundle"], version=PROMPT_VERSION_V3_FACTS)


def test_the_facts_runner_is_bounded_and_files_findings_apart_from_contract_failures(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    source, _ = source_items(tmp_path)
    packets, asof, output = tmp_path / "packets", tmp_path / "asof", tmp_path / "run"
    prepare_revision(source, packets, e_facts=True)
    prepare_revision(source, asof, asof_assessment=True)
    facts_prompt = json.loads((packets / "case-001.json").read_text())["prompts"]["baseline"]
    asof_prompt = json.loads((asof / "case-001.json").read_text())["prompts"]["baseline"]
    for prompt, profile in ((facts_prompt, "pilot"), (facts_prompt, "asof_revision_pilot"),
                            (asof_prompt, "facts_revision_pilot")):
        with pytest.raises(ValueError):
            request_body(providers()[0], prompt, max_tokens=8192 if profile == "pilot" else 16384, profile=profile)
    for wrong in (lambda: plan_pilot(packets, tmp_path / "wrong"), lambda: plan_asof_revision(packets, tmp_path / "wrong"),
                  lambda: plan_facts_revision(asof, tmp_path / "wrong")):
        with pytest.raises(ValueError):
            wrong()
    assert not (tmp_path / "wrong").exists()
    plan = plan_facts_revision(packets, output)
    assert plan["experiment_kind"] == "facts_revision_pilot" and plan["api_calls"] == 0
    env = tmp_path / ".env"
    env.write_text("B.ai_API_KEY=private-test-bai-key\n")
    actions = []
    for n, job in enumerate(plan["schedule"]):
        made = json.loads((output / (job["case_ref"] + ".json")).read_text())["bundle"]
        value = answer(made)
        if n == 1:
            value["e_facts"]["range"]["upper"] = "s0003"
        elif n == 2:
            value = answer(made, facts=dict(FACTS, support=None))
        elif n == 3:
            value["claims"][0]["observations"][0]["value"] = -1.
        actions.append(lambda p, a=value: response(p, a))
    transport = FakeTransport(actions)
    summary = run_pilot(output, env, transport=transport)
    assert len(transport.calls) == summary["api_calls"] == 4  # a finding never hunts another provider
    assert summary["valid_contract_outputs"] == 1 and summary["invalid_contract_outputs"] == 1
    assert summary["fact_finding_outputs"] == {"fact_error": 1, "evidence_gap": 1}
    saved = [json.loads((output / f"{j['case_ref']}-baseline-{j['repeat']}.json").read_text())
             for j in plan["schedule"]]
    assert [r["state"] for r in saved] == ["valid", "fact_findings", "fact_findings", "invalid_contract"]
    assert saved[0]["attempts"][-1]["fact_review"]["findings"] == []
    for result, kind in zip(saved[1:3], ("fact_error", "evidence_gap")):
        attempt = result["attempts"][-1]
        assert attempt["answer"]["phase"] == "E" and "contract_error" not in attempt
        assert {f["class"] for f in attempt["fact_review"]["findings"]} == {kind}
    assert "fact_review" not in saved[3]["attempts"][-1]
    blocked = tmp_path / "blocked"
    plan_facts_revision(packets, blocked)
    changed = json.loads((blocked / "plan.json").read_text())
    changed["max_reserved_usd"] = .61
    (blocked / "plan.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="facts pilot plan bounds"):
        run_pilot(blocked, env, transport=FakeTransport([]))
    assert not (blocked / "attempts.jsonl").exists()
