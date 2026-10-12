"""V2 evidence grouping and a separate fixed v1 budget experiment."""
from copy import deepcopy
import json

import pytest

from research.llm_api import Budget, evaluate_request, request_body
from research.llm_contract import VERSION_V2, validate_response
from research.llm_pilot import plan_asof_revision, plan_budget_diagnostic, plan_pilot, run_pilot
from research.llm_prompt import PROMPT_VERSION_V2, PROMPT_VERSION_V2_ASOF, build_prompt, prepare_revision
from test_llm_api import FakeTransport, item, providers, response
from test_llm_prompt import claim, packet, unresolved


def revised():
    bundle = packet()
    bundle["packet"]["version"] = VERSION_V2
    return bundle


def validate(answer, bundle):
    return validate_response(json.dumps(answer), bundle, finish_reason="stop")


def test_v2_groups_e_evidence_without_forcing_unavailable_components_into_phase_abstention():
    bundle = revised()
    answer = unresolved(bundle)
    answer.update(status="judged", phase="E", direction="accumulation", reason="supported", needs=[])
    c = claim(bundle)
    c["e_component"] = None
    answer["claims"] = [c]
    answer["e_assessment"]["historical_structure"] = "supported"
    with pytest.raises(ValueError, match="historical_structure requires"):
        validate(answer, bundle)
    c["e_component"] = "historical_structure"
    assert validate(answer, bundle)["e_assessment"]["support_lifecycle"] == "unavailable"
    answer["e_assessment"].update(current_structure="challenged", support_lifecycle="valid")
    with pytest.raises(ValueError, match="current_structure requires"):
        validate(answer, bundle)
    current = deepcopy(c)
    current.update(kind="counterevidence", e_component="current_structure")
    answer["claims"].append(current)
    with pytest.raises(ValueError, match="support_lifecycle requires"):
        validate(answer, bundle)
    lifecycle = deepcopy(c)
    lifecycle["e_component"] = "support_lifecycle"
    answer["claims"].append(lifecycle)
    assert validate(answer, bundle)["phase"] == "E"
    # Grouping proves no meaning: these synthetic claims are not phase truth.


@pytest.mark.parametrize("failure", ["session_pointer", "blind_core_issue", "future_claim", "missing_component"])
def test_v2_rejects_observed_contract_failures(failure):
    bundle = revised()
    answer = unresolved(bundle)
    c = claim(bundle)
    c["e_component"] = None
    answer["claims"] = [c]
    if failure == "session_pointer":
        c["observations"][0]["pointer"] = "/bars/s0009/close"
    elif failure == "blind_core_issue":
        c["kind"] = "core_issue"
    elif failure == "future_claim":
        c.update(happened_at="s0000", known_at="s0000")
    else:
        del c["e_component"]
    with pytest.raises(ValueError):
        validate(answer, bundle)


def test_offline_revision_keeps_v1_packets_and_ab_evidence_intact(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    source = tmp_path / "source"
    source.mkdir()
    originals = {}
    for ref in ("case-001", "case-025"):
        data = item()
        data["bundle"]["packet"]["case_ref"] = ref
        data["bundle"]["local"]["role"] = "development"
        data["prompts"] = {v: build_prompt(data["bundle"], variant=v) for v in ("baseline", "contrast")}
        path = source / (ref + ".json")
        originals[path] = json.dumps(data)
        path.write_text(originals[path])
    out = tmp_path / "v2"
    assert prepare_revision(source, out)["api_calls"] == 0
    for path, original in originals.items():
        assert path.read_text() == original
        old = json.loads(original)
        new = json.loads((out / path.name).read_text())
        expected = deepcopy(old["bundle"])
        expected["packet"]["version"] = VERSION_V2
        assert new["bundle"] == expected
        a, b = new["prompts"]["baseline"], new["prompts"]["contrast"]
        assert a["prompt_version"] == PROMPT_VERSION_V2
        assert a["messages"][1] == b["messages"][1]
        assert a["response_schema"]["properties"]["claims"]["items"]["properties"]["kind"]["enum"] == ["support", "counterevidence"]
        assert build_prompt(old["bundle"]) == old["prompts"]["baseline"]
    with pytest.raises(FileExistsError):
        prepare_revision(source, out)
    with pytest.raises(ValueError, match="versions differ"):
        build_prompt(packet(), version=PROMPT_VERSION_V2)


def test_diagnostic_changes_only_output_budget_and_uses_it_in_bounded_runner(tmp_path):
    source = tmp_path / "v1"
    source.mkdir()
    data = item()
    data["bundle"]["local"]["role"] = "development"
    (source / "case-001.json").write_text(json.dumps(data))
    (source / "plan.json").write_text(json.dumps({"status": "completed"}))
    for repeat in (1, 2):
        result = evaluate_request(data, "baseline", providers(), FakeTransport([response]), Budget())
        (source / f"case-001-baseline-{repeat}.json").write_text(json.dumps(result))
    out = tmp_path / "diagnostic"
    plan = plan_budget_diagnostic(source, out)
    assert len(plan["schedule"]) == 2 and plan["max_model_requests"] == 6
    env = tmp_path / ".env"
    env.write_text("B.ai_API_KEY=private-test-bai-key\n")
    transport = FakeTransport([response, response])
    summary = run_pilot(out, env, transport=transport)
    assert summary["api_calls"] == 2 and summary["valid_contract_outputs"] == 2
    assert summary["variant_agreement"] == []
    assert summary["repeat_stability"][0]["comparable"] is True
    original = request_body(providers()[0], data["prompts"]["baseline"])
    for _, body in transport.calls:
        assert body["max_tokens"] == 16384
        restored = {**body, "max_tokens": 8192}
        assert restored == original
    with pytest.raises(ValueError, match="bounds"):
        request_body(providers()[0], data["prompts"]["baseline"], max_tokens=16384)
    with pytest.raises(ValueError, match="bounds"):
        request_body(providers()[0], data["prompts"]["baseline"], max_tokens=32768, profile="output_budget_diagnostic")
    with pytest.raises(ValueError, match="already started"):
        run_pilot(out, env, transport=transport)
    contaminated = json.loads((source / "case-001-baseline-1.json").read_text())
    contaminated["attempts"][0]["request"]["reasoning_effort"] = "high"
    (source / "case-001-baseline-1.json").write_text(json.dumps(contaminated))
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="differs"):
        plan_budget_diagnostic(source, rejected)
    assert not rejected.exists()


def test_snapshot_scope_is_explicit_and_does_not_rewrite_historical_claim_times():
    legacy = revised()
    bundle = deepcopy(legacy)
    bundle["packet"]["claim_time_scope"] = "asof_assessment"
    answer = unresolved(bundle)
    c = claim(bundle, i=1)
    c.update(happened_at="s0001", e_component=None)
    answer["claims"] = [c]
    assert validate(answer, bundle)["claims"][0]["known_at"] == "s0009"
    c["known_at"] = "s0008"
    assert validate(answer, legacy)["claims"][0]["known_at"] == "s0008"
    with pytest.raises(ValueError, match="known_at"):
        validate(answer, bundle)
    assert c["known_at"] == "s0008"  # rejected, never silently patched
    c["known_at"] = "s0009"
    c["observations"][0]["pointer"] = "/bars/10/close"
    with pytest.raises(ValueError, match="citation"):
        validate(answer, bundle)


def test_snapshot_prompt_binds_identity_and_uses_a_matching_format_example():
    bundle = revised()
    bundle["packet"].update(claim_time_scope="asof_assessment", case_ref="case-025")
    prompt = build_prompt(bundle, version=PROMPT_VERSION_V2_ASOF)
    schema = prompt["response_schema"]["properties"]
    assert schema["case_ref"]["enum"] == ["case-025"]
    assert schema["mode"]["enum"] == ["blind_label"]
    assert schema["claims"]["items"]["properties"]["known_at"]["enum"] == ["s0009"]
    example, _ = json.JSONDecoder().raw_decode(prompt["messages"][0]["content"].split("FORMAT_EXAMPLE:\n", 1)[1])
    assert validate(example, bundle)["case_ref"] == "case-025"
    contrast = build_prompt(bundle, version=PROMPT_VERSION_V2_ASOF, variant="contrast")
    assert "工程合成" in contrast["messages"][0]["content"]
    assert contrast["messages"][1] == prompt["messages"][1]
    with pytest.raises(ValueError, match="scopes"):
        build_prompt(bundle, version=PROMPT_VERSION_V2)


def test_scoped_live_runner_is_bounded_and_keeps_legal_abstentions(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    source = tmp_path / "source"
    source.mkdir()
    for ref, market in (("case-001", "us_equities"), ("case-025", "binance_spot")):
        data = item()
        data["bundle"]["packet"]["case_ref"] = ref
        data["bundle"]["local"].update(role="development", source_case={
            "role": "development", "label_mode": "asof", "market": market, "as_of": "2025-01-10"})
        data["prompts"] = {v: build_prompt(data["bundle"], variant=v) for v in ("baseline", "contrast")}
        (source / (ref + ".json")).write_text(json.dumps(data))
    packets, output = tmp_path / "packets", tmp_path / "run"
    prepare_revision(source, packets, asof_assessment=True)
    scoped = json.loads((packets / "case-001.json").read_text())["prompts"]["baseline"]
    with pytest.raises(ValueError, match="scoped"):
        request_body(providers()[0], scoped)
    with pytest.raises(ValueError, match="scoped"):
        plan_pilot(packets, tmp_path / "wrong-entry")
    assert not (tmp_path / "wrong-entry").exists()
    plan_asof_revision(packets, output)
    env = tmp_path / ".env"
    env.write_text("B.ai_API_KEY=private-test-bai-key\n")
    actions = []
    for job in json.loads((output / "plan.json").read_text())["schedule"]:
        bundle = json.loads((output / (job["case_ref"] + ".json")).read_text())["bundle"]
        actions.append(lambda p, a=unresolved(bundle): response(p, a))
    transport = FakeTransport(actions)
    summary = run_pilot(output, env, transport=transport)
    assert summary["valid_contract_outputs"] == summary["api_calls"] == 4
    assert summary["coverage"]["unresolved"] == 4
    assert summary["variant_agreement"] == [] and len(summary["repeat_stability"]) == 2
    assert all(x[1]["max_tokens"] == 16384 for x in transport.calls)
    # A changed plan must be rejected before a charge, not silently expanded.
    blocked = tmp_path / "blocked"
    plan_asof_revision(packets, blocked)
    plan = json.loads((blocked / "plan.json").read_text())
    plan["max_model_requests"] = 9
    (blocked / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="bounds"):
        run_pilot(blocked, env, transport=FakeTransport([]))
    assert not (blocked / "attempts.jsonl").exists()
