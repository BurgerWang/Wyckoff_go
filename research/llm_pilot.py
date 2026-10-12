"""Small development A/B; explicit --run enables paid model requests."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from research.llm_api import Budget, BudgetExceeded, MODEL_IDS, Provider, Transport, evaluate_request, load_providers, request_body
from research.llm_prompt import build_prompt


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_cases(packets):
    """First registered development date per market, without core/AI screening."""
    selected = {}
    for path in sorted(Path(packets).glob("case-*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        bundle = item["bundle"]
        case = bundle["local"]["source_case"]
        if case["role"] != "development" or case["label_mode"] != "asof" or bundle["packet"]["mode"] != "blind_label":
            raise ValueError("pilot requires registered asof development blind inputs")
        if case["market"] not in ("us_equities", "binance_spot"):
            continue
        for variant in ("baseline", "contrast"):
            if item["prompts"][variant] != build_prompt(bundle, variant=variant, version=item["prompts"][variant]["prompt_version"]):
                raise ValueError("saved prompt does not match current frozen prompt construction")
        if case["market"] not in selected or case["as_of"] < selected[case["market"]]["bundle"]["local"]["as_of"]:
            selected[case["market"]] = item
    if set(selected) != {"us_equities", "binance_spot"}:
        raise ValueError("need one registered development case from each market")
    return [selected[k] for k in ("us_equities", "binance_spot")]


def plan_pilot(packets, output, *, max_requests=12, max_usd=1.):
    budget = Budget(max_requests=max_requests, max_usd=max_usd)
    items = select_cases(packets)
    if any(x["bundle"]["packet"].get("claim_time_scope") is not None for x in items):
        raise ValueError("scoped packets require their dedicated bounded pilot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Balanced instruction order across repetitions; all inputs remain stateless.
    schedule = [{"case_ref": item["bundle"]["packet"]["case_ref"], "variant": variant, "repeat": repeat}
                for item in items for repeat in (1, 2)
                for variant in (("baseline", "contrast") if repeat == 1 else ("contrast", "baseline"))]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
            "selection": "earliest_registered_development_asof_per_market; no answer screening",
            "schedule": schedule, "max_model_requests": budget.max_requests, "max_reserved_usd": budget.max_usd,
            "max_tokens_per_request": 8192, "reasoning_effort": "low", "response_format": "json_object",
            "provider_order": ["b.ai", "openrouter", "deepseek_native"],
            "retry_count": 0, "api_calls": 0, "phase_accuracy": None, "calibration": None, "trading_results": None}
    _write(output / "plan.json", plan)
    for item in items:
        _write(output / (item["bundle"]["packet"]["case_ref"] + ".json"), item)
    return plan


def plan_budget_diagnostic(source, output):
    """Same exposed v1 baseline, two fixed outputs; change only max_tokens."""
    source, output = Path(source), Path(output)
    prior_plan = json.loads((source / "plan.json").read_text())
    item = json.loads((source / "case-001.json").read_text())
    if prior_plan["status"] != "completed" or item["bundle"]["local"]["role"] != "development":
        raise ValueError("diagnostic requires the completed exposed development pilot")
    prompt = item["prompts"]["baseline"]
    if prompt != build_prompt(item["bundle"]):
        raise ValueError("diagnostic must keep the frozen v1 baseline")
    expected = request_body(Provider("b.ai", MODEL_IDS["b.ai"], "unused-offline-key"), prompt)
    references = []
    for repeat in (1, 2):
        name = f"case-001-baseline-{repeat}.json"
        result = json.loads((source / name).read_text())
        if len(result["attempts"]) != 1:
            raise ValueError("budget reference changed channels")
        attempt = result["attempts"][0]
        if attempt["provider"] != "b.ai" or attempt["request"] != expected:
            raise ValueError("budget reference differs from the fixed baseline")
        references.append({"file": str(source / name), "provider": attempt["provider"],
                           "requested_model": attempt["requested_model"], "returned_model": attempt["returned_model"],
                           "finish_reason": attempt["finish_reason"], "state": result["state"]})
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
            "experiment_kind": "output_budget_diagnostic", "selection": "same preselected case-001 v1 baseline",
            "schedule": [{"case_ref": "case-001", "variant": "baseline", "repeat": r} for r in (1, 2)],
            "max_model_requests": 6, "max_reserved_usd": .30, "max_tokens_per_request": 16384,
            "reasoning_effort": "low", "response_format": "json_object", "timeout_seconds": 180,
            "provider_order": ["b.ai", "openrouter", "deepseek_native"], "references": references,
            "changed_model_parameter": "max_tokens: 8192 -> 16384; all other same-channel body fields equal",
            "timeout_note": "Client read timeout allows larger output; not a model parameter or retry.",
            "retry_count": 0, "api_calls": 0, "phase_accuracy": None, "calibration": None, "trading_results": None}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    _write(output / "case-001.json", item)
    return plan


SCOPED_PILOTS = {"asof_revision_pilot": ("asof", "wyckoff-prompt-v2-asof"),
                 "facts_revision_pilot": ("facts", "wyckoff-prompt-v3-facts")}


def plan_asof_revision(packets, output):
    """Two exposed windows, two fixed repeats; no automatic answer retries."""
    return _plan_scoped(packets, output, "asof_revision_pilot")


def plan_facts_revision(packets, output):
    """Same two windows and bounds for the v3 facts contract. Planning sends nothing."""
    return _plan_scoped(packets, output, "facts_revision_pilot")


def _plan_scoped(packets, output, kind):
    label, prompt_version = SCOPED_PILOTS[kind]
    items = select_cases(packets)
    if any(x["bundle"]["packet"].get("claim_time_scope") != "asof_assessment"
           or x["prompts"]["baseline"]["prompt_version"] != prompt_version for x in items):
        raise ValueError(f"{label} pilot requires two explicitly scoped packets")
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
            "experiment_kind": kind, "selection": "same two exposed preselected development windows",
            "schedule": [{"case_ref": x["bundle"]["packet"]["case_ref"], "variant": "baseline", "repeat": r}
                         for r in (1, 2) for x in items],
            "max_model_requests": 8, "max_reserved_usd": .60, "max_tokens_per_request": 16384,
            "reasoning_effort": "low", "response_format": "json_object", "timeout_seconds": 180,
            "provider_order": ["b.ai", "openrouter", "deepseek_native"], "retry_count": 0, "api_calls": 0,
            "phase_accuracy": None, "calibration": None, "trading_results": None}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "plan.json", plan)
    for item in items:
        _write(output / (item["bundle"]["packet"]["case_ref"] + ".json"), item)
    return plan


def summarize(plan, results, budget):
    valid = [r for r in results if r["state"] == "valid"]
    attempts = [a for r in results for a in r["attempts"]]
    coverage = {s: sum(r["attempts"][-1].get("answer", {}).get("status") == s for r in valid)
                for s in ("judged", "unresolved", "no_observable_phase")}
    stability = []
    pairs = []
    def decision(r):
        a = r["attempts"][-1]["answer"]
        return {k: a[k] for k in ("status", "phase", "direction", "reason")}
    def provider(r):
        a = r["attempts"][-1]
        return (a["provider"], a["requested_model"], a.get("returned_model"))
    for case_ref in sorted({x["case_ref"] for x in plan["schedule"]}):
        single_variant = plan.get("experiment_kind") in ("output_budget_diagnostic", *SCOPED_PILOTS)
        for variant in (("baseline",) if single_variant else ("baseline", "contrast")):
            group = [r for r in valid if r["case_ref"] == case_ref and r["variant"] == variant]
            comparable = len(group) == 2 and provider(group[0]) == provider(group[1])
            stability.append({"case_ref": case_ref, "variant": variant, "comparable": comparable,
                              "decision_equal": decision(group[0]) == decision(group[1]) if comparable else None})
        for repeat in (() if single_variant else (1, 2)):
            group = [r for r in valid if r["case_ref"] == case_ref and r["repeat"] == repeat]
            comparable = len(group) == 2 and provider(group[0]) == provider(group[1])
            pairs.append({"case_ref": case_ref, "repeat": repeat, "comparable": comparable,
                          "decision_equal": decision(group[0]) == decision(group[1]) if comparable else None})
    reported = [a.get("reported_cost_usd") for a in attempts]
    estimates = [a.get("peak_estimate_usd") for a in attempts]
    return {"status": "completed" if len(results) == len(plan["schedule"]) else "partial",
            "scheduled_outputs": len(plan["schedule"]), "finished_outputs": len(results),
            "api_calls": budget.requests, "valid_contract_outputs": len(valid), "coverage": coverage,
            "invalid_contract_outputs": sum(r["state"] == "invalid_contract" for r in results),
            "fact_finding_outputs": {kind: sum(
                r["state"] == "fact_findings"
                and any(f["class"] == kind for f in r["attempts"][-1]["fact_review"]["findings"]) for r in results)
                for kind in ("fact_error", "evidence_gap")},
            "availability_failures": sum(a["state"] == "availability_failure" for a in attempts),
            "provider_calls": {p: sum(a["provider"] == p for a in attempts) for p in plan["provider_order"]},
            "reserved_usd": budget.reserved_usd,
            "reported_cost_usd": sum(reported) if reported and all(x is not None for x in reported) else None,
            "peak_token_estimate_usd": sum(x for x in estimates if x is not None),
            "cost_estimate_missing_attempts": sum(x is None for x in estimates),
            "repeat_stability": stability, "variant_agreement": pairs,
            "phase_accuracy": None, "calibration": None, "trading_results": None,
            "experiment_kind": plan.get("experiment_kind", "prompt_ab_pilot"),
            "max_tokens_per_request": plan.get("max_tokens_per_request", 8192),
            "limitations": ["Exposed development windows only; no independent truth.",
                            "Contract/decision agreement is not semantic accuracy.",
                            "Reserved byte-based allowance is not exact tokenization or an invoice.",
                            "Price estimate uses full Peak rates without cache discounts.",
                            "Fallback providers are only live-tested if primary availability fails."]}


def run_pilot(output, env_path, *, transport=None, progress=lambda event: None):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan["status"] != "prepared" or (output / "attempts.jsonl").exists():
        raise ValueError("run already started; reattach to its process/output, do not replay charged requests")
    providers = load_providers(env_path)
    budget = Budget(max_requests=plan["max_model_requests"], max_usd=plan["max_reserved_usd"])
    profile = plan.get("experiment_kind", "pilot")
    max_tokens, effort = plan.get("max_tokens_per_request", 8192), plan.get("reasoning_effort", "low")
    timeout = plan.get("timeout_seconds", 60)
    if profile == "output_budget_diagnostic" and (
            plan["max_model_requests"] > 6 or plan["max_reserved_usd"] > .30
            or plan["schedule"] != [{"case_ref": "case-001", "variant": "baseline", "repeat": r} for r in (1, 2)]):
        raise ValueError("diagnostic plan bounds exceeded")
    if profile in SCOPED_PILOTS and (
            plan["max_model_requests"] > 8 or plan["max_reserved_usd"] > .60
            or plan["schedule"] != [{"case_ref": ref, "variant": "baseline", "repeat": r}
                                    for r in (1, 2) for ref in ("case-001", "case-025")]):
        raise ValueError(f"{SCOPED_PILOTS[profile][0]} pilot plan bounds exceeded")
    if type(timeout) is not int or not 1 <= timeout <= 180:
        raise ValueError("bounded request timeout required")
    for job in plan["schedule"]:
        item = json.loads((output / (job["case_ref"] + ".json")).read_text())
        request_body(providers[0], item["prompts"][job["variant"]], max_tokens=max_tokens, effort=effort, profile=profile)
    plan.update(status="running", providers=[{"name": p.name, "model": p.model} for p in providers])
    _write(output / "plan.json", plan)
    transport = transport if transport is not None else Transport(timeout=timeout)
    results = []
    stop_reason = None
    def record(event, job):
        event = {"at": datetime.now(timezone.utc).isoformat(), "job": job, **event}
        with (output / "attempts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
    try:
        for job in plan["schedule"]:
            item = json.loads((output / (job["case_ref"] + ".json")).read_text(encoding="utf-8"))
            try:
                result = evaluate_request(item, job["variant"], providers, transport, budget,
                                          record=lambda event: record(event, deepcopy(job)),
                                          max_tokens=max_tokens, effort=effort, profile=profile)
            except BudgetExceeded:
                stop_reason = "budget_exhausted"
                break
            result["repeat"] = job["repeat"]
            results.append(result)
            _write(output / f"{job['case_ref']}-{job['variant']}-{job['repeat']}.json", result)
            _write(output / "summary.json", summarize(plan, results, budget))
            progress({"finished": len(results), "state": result["state"],
                      "provider": result["attempts"][-1]["provider"], "requests": budget.requests})
            if result.get("budget_exhausted") or any(a.get("budget_mismatch") for a in result["attempts"]):
                stop_reason = "budget_exhausted_or_mismatch"
                break
    finally:
        summary = summarize(plan, results, budget)
        if stop_reason:
            summary["stop_reason"] = stop_reason
        _write(output / "summary.json", summary)
        plan.update(status=summary["status"], api_calls=budget.requests)
        _write(output / "plan.json", plan)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, default=Path("output/llm/20261001-prompt-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan_pilot(args.packets, args.output)
    if args.run:
        summary = run_pilot(args.output, args.env,
                            progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "prepared", "output": str(args.output), "api_calls": 0}))


if __name__ == "__main__":
    main()
