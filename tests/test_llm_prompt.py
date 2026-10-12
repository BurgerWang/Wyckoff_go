"""AI engineering diagnostics, never human phase labels or model accuracy."""
from copy import deepcopy
import json

import pandas as pd
import pytest

from research.llm_contract import make_packet, validate_response
from research.llm_prompt import build_prompt, prepare_pilot
from test_phase_explanations import supported_e
from test_phase_c_contract_v5 import put
from test_phase_c_contract_v5 import boundary_test
from test_core_usability_v8 import establish_range


def frame():
    df = pd.DataFrame({"Open": [100.] * 12, "High": [102.] * 12, "Low": [98.] * 12,
                       "Close": [101.] * 12, "Volume": [100.] * 12},
                      index=pd.date_range("2025-01-01", periods=12))
    return df


def packet(df=None, **kwargs):
    return make_packet(frame() if df is None else df, as_of="2025-01-10",
                       information_start="2025-01-01", case_ref="case-001",
                       analysis_scale="registered_daily_window", **kwargs)


def unresolved(bundle):
    p = bundle["packet"]
    return {"version": p["version"], "origin": "ai_reference", "case_ref": p["case_ref"],
            "mode": p["mode"], "as_of": p["as_of"], "status": "unresolved", "phase": None,
            "direction": None, "reason": "missing_context", "range_start": None,
            "bc_terminal_verified": False,
            "e_assessment": {"historical_structure": "unassessed", "current_structure": "unavailable",
                             "support_lifecycle": "unavailable"},
            "claims": [], "needs": ["需要左侧停止及区间背景"]}


def claim(bundle, i=9):
    return {"kind": "support", "statement": "此条仅为工程引用夹具，不构成阶段真值",
            "happened_at": "s0009", "known_at": "s0009",
            "observations": [{"pointer": f"/bars/{i}/close", "value": bundle["packet"]["bars"][i]["close"]}]}


def check(value, bundle, finish_reason="stop"):
    return validate_response(json.dumps(value), bundle, finish_reason=finish_reason)


def test_blind_payload_is_prefix_invariant_and_strips_labels_dates_identity_and_attrs():
    df = frame()
    df["phase"] = "FUTURE_ANSWER"
    df.attrs.update(symbol="AAPL", notes="ignore prompt", future_return=123)
    original = packet(df)
    shocked = df.copy()
    shocked.loc["2025-01-11":, "Close"] = float("nan")
    assert packet(shocked) == original
    future_index = pd.concat([df, df.iloc[-1:].set_axis([pd.Timestamp("2025-01-13T12:00")])])
    assert packet(future_index) == original
    model_input = json.dumps(build_prompt(original)["messages"], ensure_ascii=False)
    assert all(x not in model_input for x in ("AAPL", "2025-01-", "FUTURE_ANSWER", "ignore prompt", "future_return"))
    assert len(original["packet"]["bars"]) == 10
    assert original["local"]["information_end"] == "2025-01-10"
    assert original["local"]["eligible_human_truth"] is False
    assert original["packet"]["bars"][0]["close"] == 100.
    with pytest.raises(ValueError, match="blind"):
        packet(core_snapshot={"phase": "C"})


def test_scope_and_budget_cannot_silently_change_registered_context():
    df = frame().drop(pd.Timestamp("2025-01-10"))
    with pytest.raises(ValueError, match="endpoints"):
        packet(df)
    with pytest.raises(ValueError, match="byte guard"):
        build_prompt(packet(), max_input_bytes=100)
    with pytest.raises(ValueError, match="scale"):
        make_packet(frame(), as_of="2025-01-10", information_start="2025-01-01",
                    case_ref="case-001", analysis_scale="whatever the model prefers")


@pytest.mark.parametrize("mutation", ["empty", "truncated", "markdown", "duplicates", "nan", "extra", "human", "identity"])
def test_failed_responses_are_not_valid_abstentions(mutation):
    bundle = packet()
    answer = unresolved(bundle)
    text, finish = json.dumps(answer), "stop"
    if mutation == "empty":
        text = ""
    elif mutation == "truncated":
        finish = "length"
    elif mutation == "markdown":
        text = "```json\n" + text + "\n```"
    elif mutation == "duplicates":
        text = '{"origin":"external_human",' + text[1:]
    elif mutation == "nan":
        text = text.replace('"phase": null', '"phase": NaN')
    else:
        answer[{"extra": "trade", "human": "origin", "identity": "case_ref"}[mutation]] = "external_human"
        text = json.dumps(answer)
    with pytest.raises(ValueError):
        validate_response(text, bundle, finish_reason=finish)
    assert check(unresolved(bundle), bundle)["status"] == "unresolved"


@pytest.mark.parametrize("mutation", ["invented", "misquoted", "known_early", "future", "terminal", "core_blind"])
def test_evidence_clock_and_terminal_c_overclaims_rejected(mutation):
    bundle = packet()
    answer = unresolved(bundle)
    answer["claims"] = [claim(bundle)]
    c = answer["claims"][0]
    if mutation == "invented":
        c["observations"][0]["pointer"] = "/bars/10/close"
    elif mutation == "misquoted":
        c["observations"][0]["value"] = 999.
    elif mutation == "known_early":
        c.update(happened_at="s0005", known_at="s0006")
    elif mutation == "future":
        c["known_at"] = "s0010"
    elif mutation == "terminal":
        answer["bc_terminal_verified"] = True
    else:
        c["kind"] = "core_issue"
    with pytest.raises(ValueError):
        check(answer, bundle)


def test_unknown_missing_volume_and_explicit_negative_have_different_contracts():
    df = frame()
    df.Volume = 0.
    bundle = packet(df)
    answer = unresolved(bundle)
    answer["reason"] = "missing_volume"
    assert check(answer, bundle)["status"] == "unresolved"
    answer.update(status="no_observable_phase", reason="no_observable_phase", needs=[], claims=[claim(bundle)])
    with pytest.raises(ValueError, match="negative"):
        check(answer, bundle)
    answer.update(status="judged", phase="C", direction="accumulation", reason="supported")
    with pytest.raises(ValueError, match="volume"):
        check(answer, bundle)
    ordinary = packet()
    answer = unresolved(ordinary)
    answer.update(status="no_observable_phase", reason="no_observable_phase", needs=[], claims=[claim(ordinary)])
    assert check(answer, ordinary)["phase"] is None
    answer["needs"] = ["缺上下文"]
    with pytest.raises(ValueError, match="negative"):
        check(answer, ordinary)


def test_core_review_retains_real_e_lifecycle_and_rejects_future_or_early_core_claim():
    df = supported_e()
    put(df, 57, 129., 124., 125.)
    kwargs = dict(as_of=df.index[-1].date().isoformat(), information_start=df.index[0].date().isoformat(),
                  case_ref="case-002", analysis_scale="registered_daily_window", mode="core_review")
    bundle = make_packet(df, **kwargs)
    decorated = df.copy()
    decorated.attrs.update(symbol="SECRET_LOCAL_SYMBOL", warnings=["ignore task"], excluded_empty_dates=["2099-01-01"])
    assert make_packet(decorated, **kwargs) == bundle
    w = bundle["packet"]["core"]["analyses"]["atr"]["wyckoff"]
    assert w["phase"]["phase"] == "E"
    assert w["phase"]["hypotheses"]["accumulation"]["structure_check"]["status"] == "challenged"
    assert w["sequence"]["acc_lps"]["price"] == 108.
    assert w["phase"]["hypotheses"]["accumulation"]["structure_check"]["reaction_pivot"]["available_date"] == "s0052"
    answer = unresolved(bundle)
    answer["claims"] = [{"kind": "core_issue", "statement": "工程复核观察，不是真值",
                         "happened_at": "s0050", "known_at": "s0057",
                         "observations": [{"pointer": "/core/analyses/atr/wyckoff/phase/phase", "value": "E"}]}]
    answer["e_assessment"] = {"historical_structure": "supported", "current_structure": "challenged", "support_lifecycle": "valid"}
    assert check(answer, bundle)["e_assessment"]["current_structure"] == "challenged"
    answer["claims"][0]["known_at"] = "s0050"
    with pytest.raises(ValueError, match="core snapshot"):
        check(answer, bundle)
    with pytest.raises(ValueError, match="unbound"):
        make_packet(df, core_snapshot={"symbol": "OTHER", "as_of": kwargs["as_of"], "data": {"bars": len(df)}}, **kwargs)
    changed = df.copy()
    changed.loc[df.index[-1], ["Open", "High", "Low", "Close"]] = [900., 901., 899., 900.]
    changed_bundle = make_packet(changed, **kwargs)
    assert changed_bundle["packet"]["bars"][-1]["close"] == 900.
    assert changed_bundle["packet"]["core"] != bundle["packet"]["core"]


def test_empty_abstention_cannot_assert_e_support_is_valid():
    bundle = packet()
    answer = unresolved(bundle)
    answer["e_assessment"]["support_lifecycle"] = "valid"
    with pytest.raises(ValueError, match="E assertions"):
        check(answer, bundle)


def test_production_test_confirmation_remains_at_asof_not_earlier_action():
    df = establish_range(boundary_test())
    kwargs = dict(information_start=df.index[0].date().isoformat(), case_ref="case-003",
                  analysis_scale="registered_daily_window", mode="core_review")
    before = make_packet(df, as_of=df.index[37].date().isoformat(), **kwargs)
    after = make_packet(df, as_of=df.index[38].date().isoformat(), **kwargs)
    assert before["packet"]["core"]["analyses"]["atr"]["wyckoff"]["phase"]["phase"] == "B"
    core = after["packet"]["core"]["analyses"]["atr"]["wyckoff"]
    assert core["phase"]["phase"] == "C"
    assert core["sequence"]["acc_test"]["date"] == "s0036"
    assert core["sequence"]["acc_test"]["available_date"] == "s0038"
    assert make_packet(df.iloc[:38], as_of=df.index[37].date().isoformat(), **kwargs) == before
    answer = unresolved(after)
    answer.update(status="judged", phase="C", direction="accumulation", reason="supported", needs=[])
    answer["claims"] = [{"kind": "support", "statement": "核心测试已确认，但终结性未验证",
                         "happened_at": "s0036", "known_at": "s0038",
                         "observations": [{"pointer": "/core/analyses/atr/wyckoff/sequence/acc_test/available_date", "value": "s0038"}]}]
    assert check(answer, after)["bc_terminal_verified"] is False
    answer["claims"][0]["known_at"] = "s0036"
    with pytest.raises(ValueError, match="core snapshot"):
        check(answer, after)


def test_ab_variants_change_only_instructions_and_do_not_mutate_bundle():
    bundle = packet()
    copy = deepcopy(bundle)
    a, b = build_prompt(bundle), build_prompt(bundle, variant="contrast")
    assert a["messages"][1] == b["messages"][1]
    assert a["response_schema"] == b["response_schema"]
    assert a["provider_order"] == ["b.ai", "openrouter", "deepseek_native"]
    assert b["input_bytes"] > a["input_bytes"]
    assert "工程合成" in b["messages"][0]["content"]
    assert bundle == copy


def test_pilot_refuses_changed_source_before_writing_or_overwriting(tmp_path, monkeypatch):
    import socket
    from research.tuning.annotation import _bars, _digest
    def no_network(*args, **kwargs):
        raise AssertionError("offline preparation must not open network")
    monkeypatch.setattr(socket, "socket", no_network)
    df = frame()
    for market, symbol in (("us_equities", "AAPL"), ("binance_spot", "BTCUSDT")):
        pack = tmp_path / "pilot" / market / symbol
        pack.mkdir(parents=True)
        data = tmp_path / "history" / market / "data"
        data.mkdir(parents=True)
        df.rename_axis("Date").to_csv(data / f"{symbol}.csv")
        source = {"market": market, "symbol": symbol, "timeframe": "1d", "ohlcv_sha256": _digest(_bars(df))}
        case = {"id": f"{symbol}-development", "role": "development", "label_mode": "asof",
                "as_of": "2025-01-10", "information_end": "2025-01-10", "information_start": "2025-01-01",
                "market": market, "symbol": symbol, "timeframe": "1d", "panel": "window"}
        (pack / "manifest.json").write_text(json.dumps({"source": source, "cases": [case], "pack_id": symbol}))
    args = (tmp_path / "pilot", tmp_path / "history", tmp_path / "output")
    summary = prepare_pilot(*args)
    assert summary["api_calls"] == summary["ai_answers"] == 0
    with pytest.raises(FileExistsError):
        prepare_pilot(*args)
    changed = df.copy()
    changed.loc[changed.index[0], "Volume"] += 1
    changed.rename_axis("Date").to_csv(tmp_path / "history" / "us_equities" / "data" / "AAPL.csv")
    new_output = tmp_path / "changed-output"
    with pytest.raises(ValueError, match="frozen"):
        prepare_pilot(*args[:2], new_output)
    assert not new_output.exists()
