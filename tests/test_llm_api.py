"""Provider fallback/budget/persistence diagnostics with no real credentials."""
from copy import deepcopy
import json

import pytest

from research.llm_api import (Budget, BudgetExceeded, MODEL_IDS, Provider, ProviderFailure,
                             evaluate_request, load_providers, request_body, safe_response)
from research.llm_pilot import run_pilot, summarize
from research.llm_prompt import build_prompt
from test_llm_prompt import packet, unresolved


def providers():
    return [Provider(name, model, "private-test-key-" + name) for name, model in MODEL_IDS.items()]


def item():
    bundle = packet()
    return {"bundle": bundle, "prompts": {v: build_prompt(bundle, variant=v) for v in ("baseline", "contrast")}}


def response(provider, answer=None, finish="stop"):
    answer = unresolved(item()["bundle"]) if answer is None else answer
    return {"id": "test-request", "model": provider.model,
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(answer), "reasoning_content": "private hidden reasoning"}}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 800}}


class FakeTransport:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def request(self, provider, *, path, payload):
        self.calls.append((provider.name, deepcopy(payload)))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action(provider) if callable(action) else action


def test_credentials_aliases_are_parsed_without_execution_or_secret_repr(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# deepseek-v4.1-flash\nB.ai_API_KEY="private-bai-key"\nOPENROUTER_API_KEY=private-or-key\nDEEPSEEK_API_KEY=private-ds-key\n')
    configured = load_providers(env)
    assert [p.name for p in configured] == list(MODEL_IDS)
    assert "private-" not in repr(configured)
    env.write_text('B.ai_API_KEY=private-one-key\nBAI_API_KEY=private-other-key\n')
    with pytest.raises(ValueError, match="conflicting"):
        load_providers(env)
    env.write_text('B.ai_API_KEY="private-key with\ttab"\n')
    with pytest.raises(ValueError, match="shape"):
        load_providers(env)


def test_requests_have_channel_specific_reasoning_and_fixed_destination_semantics():
    prompt = item()["prompts"]["baseline"]
    for p in providers():
        body = request_body(p, prompt)
        assert body["max_tokens"] == 8192 and body["response_format"] == {"type": "json_object"}
        assert "tools" not in body and "temperature" not in body and "plugins" not in body
        if p.name == "openrouter":
            assert body["reasoning"]["effort"] == "low"
            assert body["provider"]["only"] == ["deepseek"]
            assert body["provider"]["allow_fallbacks"] is False
        else:
            assert body["reasoning_effort"] == "low"
    with pytest.raises(ValueError, match="mapping"):
        request_body(Provider("b.ai", "some-other-model", "private-key"), prompt)


def test_valid_abstention_does_not_retry_or_fallback_and_omits_reasoning():
    transport = FakeTransport([response])
    result = evaluate_request(item(), "baseline", providers(), transport, Budget())
    assert [x[0] for x in transport.calls] == ["b.ai"]
    assert result["state"] == "valid"
    assert result["attempts"][0]["answer"]["status"] == "unresolved"
    assert "private hidden reasoning" not in json.dumps(result)


def test_only_availability_failure_moves_in_order_and_consumes_failed_reservations():
    transport = FakeTransport([ProviderFailure("unavailable_http", 429), ProviderFailure("network_unavailable"), response])
    budget, events = Budget(), []
    result = evaluate_request(item(), "baseline", providers(), transport, budget, record=events.append)
    assert [x[0] for x in transport.calls] == ["b.ai", "openrouter", "deepseek_native"]
    assert budget.requests == 3 and budget.reserved_usd == sum(a["reserved_usd"] for a in result["attempts"])
    assert [e["state"] for e in events] == ["reserved", "availability_failure", "reserved", "availability_failure", "reserved", "valid"]
    assert result["state"] == "valid"


@pytest.mark.parametrize("kind", ["bad_request", "citation_error", "truncation", "wrong_model", "refusal"])
def test_bad_request_or_returned_judgment_failure_does_not_provider_hop(kind):
    p = providers()[0]
    raw = response(p)
    if kind == "bad_request":
        action = ProviderFailure("bad_request", 400)
    else:
        if kind == "citation_error":
            answer = unresolved(item()["bundle"])
            answer["claims"] = [{"kind": "support", "statement": "wrong evidence", "happened_at": "s0001", "known_at": "s0009",
                                 "observations": [{"pointer": "/bars/900/close", "value": 100}]}]
            raw["choices"][0]["message"]["content"] = json.dumps(answer)
        elif kind == "truncation":
            raw["choices"][0]["finish_reason"] = "length"
        elif kind == "wrong_model":
            raw["model"] = "other-family"
        elif kind == "refusal":
            raw["choices"][0]["message"]["refusal"] = "refused"
        action = raw
    transport = FakeTransport([action])
    result = evaluate_request(item(), "baseline", providers(), transport, Budget())
    assert len(transport.calls) == 1
    assert result["state"] == ("request_failure" if kind == "bad_request" else "invalid_contract")


def test_budget_caps_attempts_before_network_and_stops_unknown_cost_retry():
    budget = Budget(max_requests=1)
    transport = FakeTransport([ProviderFailure("network_unavailable")])
    result = evaluate_request(item(), "baseline", providers(), transport, budget)
    assert result["budget_exhausted"] is True
    assert len(transport.calls) == 1
    with pytest.raises(BudgetExceeded):
        evaluate_request(item(), "baseline", providers(), transport, budget)
    assert len(transport.calls) == 1
    for kwargs in ({"max_requests": 100}, {"max_usd": 2}):
        with pytest.raises(ValueError):
            Budget(**kwargs)


def test_cost_mismatch_stops_budget_and_key_redaction_covers_nested_response():
    p = providers()[0]
    raw = response(p)
    raw["usage"]["cost"] = .5
    raw["note"] = p.api_key
    result = evaluate_request(item(), "baseline", providers(), FakeTransport([raw]), Budget())
    assert result["attempts"][0]["budget_mismatch"] is True
    assert p.api_key not in json.dumps(result)
    assert p.api_key not in json.dumps(safe_response({"nested": [p.api_key]}, providers()))


def test_charged_empty_answer_stops_fallback_and_metadata_is_also_redacted():
    p = providers()[0]
    raw = response(p)
    raw["model"] = p.api_key
    raw["choices"][0]["native_finish_reason"] = p.api_key
    raw["choices"][0]["message"]["content"] = ""
    raw["usage"]["cost"] = 2.
    budget, events = Budget(), []
    transport = FakeTransport([raw])
    result = evaluate_request(item(), "baseline", providers(), transport, budget, record=events.append)
    assert result["attempts"][0]["budget_mismatch"] is True
    assert budget.reserved_usd == budget.max_usd and len(transport.calls) == 1
    assert p.api_key not in json.dumps(result) and p.api_key not in json.dumps(events)


@pytest.mark.parametrize("failure", ["envelope", "upstream", "usage"])
def test_other_charged_failures_cannot_skip_budget_check(failure):
    raw = response(providers()[0])
    raw["usage"]["cost"] = 2.
    if failure == "envelope":
        raw["choices"] = []
    elif failure == "upstream":
        raw["error"] = {"code": "upstream_failure"}
    else:
        raw["usage"]["cost"] = 10**100
    transport, budget = FakeTransport([raw]), Budget()
    result = evaluate_request(item(), "baseline", providers(), transport, budget)
    assert len(transport.calls) == 1 and budget.reserved_usd == budget.max_usd
    assert result["attempts"][0]["budget_mismatch"] is True


def test_persisted_run_refuses_replay_and_retains_partial_budget_boundary(tmp_path):
    env = tmp_path / ".env"
    env.write_text("B.ai_API_KEY=private-test-bai-key\n")
    data = item()
    (tmp_path / "case-001.json").write_text(json.dumps(data))
    plan = {"status": "prepared", "max_model_requests": 1, "max_reserved_usd": 1.,
            "provider_order": list(MODEL_IDS), "schedule": [{"case_ref": "case-001", "variant": "baseline", "repeat": 1},
                                                          {"case_ref": "case-001", "variant": "contrast", "repeat": 1}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    transport = FakeTransport([response])
    summary = run_pilot(tmp_path, env, transport=transport)
    assert summary["status"] == "partial" and summary["api_calls"] == 1
    assert summary["stop_reason"] == "budget_exhausted"
    assert (tmp_path / "attempts.jsonl").exists()
    with pytest.raises(ValueError, match="already started"):
        run_pilot(tmp_path, env, transport=transport)
    assert len(transport.calls) == 1
    assert "private-test-bai-key" not in (tmp_path / "attempts.jsonl").read_text()


def test_cross_provider_results_cannot_be_counted_as_ab_agreement():
    r = evaluate_request(item(), "baseline", providers(), FakeTransport([response]), Budget())
    r["repeat"] = 1
    s = evaluate_request(item(), "contrast", providers()[1:], FakeTransport([response]), Budget())
    s["repeat"] = 1
    plan = {"schedule": [{"case_ref": "case-001"}] * 2, "provider_order": list(MODEL_IDS)}
    summary = summarize(plan, [r, s], Budget())
    assert summary["variant_agreement"][0]["comparable"] is False
    assert summary["variant_agreement"][0]["decision_equal"] is None
    assert summary["phase_accuracy"] is None
