"""Frozen-candidate validation and explicit profile export; no test-set selection."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .data import load_history, replay
from .holdout import check_owner, claim_holdout
from .labels import group_information_bounds, iso_date, load_labels
from .metrics import score
from .policy import VALIDATION_POLICY
from .preflight import evaluate_reference_capacity
from .profiles import resolve_config
from .runner import digest, run_lock, write_json
from .space import validate_candidate
from .splits import build_split_plan
from .validation_stats import legal_neighbors, paired_cluster_bootstrap, score_summaries, summarize_stability
from wyckoff_core.models import CoreConfig


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verified_run(directory):
    # Imported lazily because the CLI calls this module only after parsing.
    from .cli import _qualified_reference, implementation_id
    manifest = _read(directory / "run.json")
    context = manifest["context"]
    if digest(context) != manifest["signature"]:
        raise ValueError("run manifest identity changed")
    if context.get("validation_policy") != VALIDATION_POLICY:
        raise ValueError("validation policy was not registered before search")
    if context["implementation"] != implementation_id():
        raise ValueError("implementation changed since search; restore it or start a new study")
    frame, item = load_history(context["history"], context["market"], context["symbol"])
    data_identity = digest({"attrs": item["attrs"], "dates": frame.index.tolist(), "values": frame.to_numpy().tolist()})
    reference = _qualified_reference(load_labels(context["labels_path"], market=context["market"],
        symbol=context["symbol"], label_mode=context["label_mode"]), frame, item, context["market"])
    plan = build_split_plan(reference, holdout_start=context["holdout_start"], holdout_end=context["holdout_end"])
    if data_identity != context["data"] or digest(reference) != context["labels"] or digest(plan) != context["splits"]:
        raise ValueError("source data, labels or split changed since search")
    if plan["status"] != "READY" or plan != _read(directory / "split-plan.json"):
        raise ValueError("registered split is not a qualified unchanged holdout")
    result = _read(directory / "results.json")
    candidate = _read(directory / "candidate.json")
    selection = _read(directory / "selection.json")
    if (result["status"] != "SEARCH_COMPLETED" or result["candidate"] != candidate
            or selection["candidate_identity"] != digest(candidate)
            or selection["results_identity"] != digest(result)
            or selection["run_signature"] != manifest["signature"]
            or candidate["run_signature"] != manifest["signature"]
            or candidate.get("status") != "CANDIDATE_ONLY" or candidate.get("holdout_evaluated") is not False):
        raise ValueError("candidate is not the unchanged frozen search result")
    for field in ("market", "symbol", "label_mode"):
        if candidate[field] != context[field]:
            raise ValueError("candidate target identity changed")
    config = validate_candidate(candidate["params"], context["market"])
    chosen = next(record for record in result["studies"] if record["fold"] == "final-development"
                  and record["sampler"] == selection["sampler"])
    if chosen["params"] != candidate["params"]:
        raise ValueError("candidate differs from development selection")
    if candidate["training_end"] != max(plan["development_dates"]):
        raise ValueError("candidate training cutoff changed")
    information_end = max(group_information_bounds(label)[1] for label in reference["labels"]
                          if label["as_of"] in set(plan["development_dates"]))
    if candidate["label_information_end"] != information_end:
        raise ValueError("candidate label-information cutoff changed")
    return manifest, frame, item, reference, plan, candidate, config, chosen


def _compact(metric):
    return {key: value for key, value in metric.items() if key != "cases"}


def validate_run(run):
    """Re-evaluation returns the cached identical result; changed candidates fail."""
    from .cli import _subset, implementation_id
    directory = Path(run).resolve()
    with run_lock(directory):
        manifest, frame, item, reference, plan, candidate, config, chosen = _verified_run(directory)
        heldout = _subset(reference, plan["holdout_dates"])
        capacity = evaluate_reference_capacity(heldout)
        if not capacity["eligible"]:
            return {"status": "INSUFFICIENT_REFERENCE_CAPACITY", "reference_capacity": capacity,
                    "reasons": capacity["diagnostic_reasons"], "holdout_evaluated": False,
                    "profile_export_allowed": False}
        identity = {"run_signature": manifest["signature"], "candidate_identity": digest(candidate),
                    "validation_policy": VALIDATION_POLICY}
        if (directory / "validation.json").exists():
            check_owner(_subset(reference, plan["holdout_dates"]), identity)
            result = _read(directory / "validation.json")
            if result.get("identity") != identity:
                raise ValueError("holdout already consumed by a different candidate or policy")
            ledger = _read(directory / "holdout-use.json")
            if ledger.get("identity") != identity:
                raise ValueError("holdout ledger differs from frozen candidate")
            if ledger.get("status") != "COMPLETED" or ledger.get("validation_identity") != digest(result):
                raise ValueError("cached validation report is changed or lacks a committed holdout digest")
            return result
        development_dates = sorted({day for fold in plan["final_inner"] for day in fold["test_dates"]})
        development = _subset(reference, development_dates)
        end = max(plan["development_dates"])
        default_rows = replay(frame, item, CoreConfig(), end)
        candidate_rows = replay(frame, item, config, end)
        default_metric = score(development, default_rows)
        candidate_metric = score(development, candidate_rows)
        if candidate_metric != chosen["development_metrics"] or default_metric != chosen["default_metrics"]:
            raise ValueError("development metrics do not reproduce the frozen search")
        neighbors = legal_neighbors(config, manifest["context"]["market"])
        neighbor_metrics = []
        folds = [{"default_metric": score(_subset(reference, fold["test_dates"]), default_rows),
                  "candidate_metric": score(_subset(reference, fold["test_dates"]), candidate_rows),
                  "neighbor_metrics": []} for fold in plan["final_inner"]]
        perturbations = []
        for neighbor in neighbors:
            rows = replay(frame, item, CoreConfig(**neighbor["config"]), end)
            metric = score(development, rows)
            neighbor_metrics.append(metric)
            perturbations.append({"parameter": neighbor["parameter"], "direction": neighbor["direction"],
                                  "params": neighbor["config"], "metrics": _compact(metric)})
            for fold, registered in zip(folds, plan["final_inner"]):
                fold["neighbor_metrics"].append(score(_subset(reference, registered["test_dates"]), rows))
        stability = summarize_stability(default_metric, candidate_metric, neighbor_metrics, folds=folds)
        outer_checks = []
        for record in _read(directory / "results.json")["studies"]:
            if not record["fold"].startswith("outer-") or record["sampler"] != chosen["sampler"]:
                continue
            baseline, selected = record["outer_default_metrics"], record["outer_metrics"]
            ce_regressions = [state for state, values in baseline["per_class"].items()
                              if state.startswith(("C/", "E/")) and values["recall"] is not None
                              and selected["per_class"][state]["recall"] < values["recall"] - 0.05 - 1e-12]
            passed = selected["joint_macro_f1"] >= baseline["joint_macro_f1"] - 0.05 - 1e-12 and not ce_regressions
            outer_checks.append({"fold": record["fold"], "sampler": record["sampler"],
                                 "passed": passed, "ce_recall_regressions": ce_regressions})
        stability["outer_workflow_checks"] = outer_checks
        stability["passed"] = stability["passed"] and len(outer_checks) == len(plan["outer"]) and all(
            check["passed"] for check in outer_checks)
        stability_record = {"identity": identity, "scope": "development labels only",
                            "summary": stability, "extra_replays": len(neighbors), "neighbors": perturbations}
        write_json(directory / "stability.json", stability_record)
        # No holdout label/prediction is scored until the development gates pass.
        if not stability["passed"]:
            return {"status": "CANDIDATE_ONLY", "reasons": ["development_stability_failed"],
                    "identity": identity, "holdout_evaluated": False, "stability": stability,
                    "profile_export_allowed": False}
        heldout = _subset(reference, plan["holdout_dates"])
        with claim_holdout(heldout, identity):
            ledger_path = directory / "holdout-use.json"
            if ledger_path.exists():
                ledger = _read(ledger_path)
                if ledger["identity"] != identity:
                    raise ValueError("holdout already opened by a different candidate")
                if ledger.get("status") not in ("OPENED_FOR_FROZEN_CANDIDATE", "COMPLETED"):
                    raise ValueError("invalid holdout-use state")
            else:
                ledger = {"identity": identity, "status": "OPENED_FOR_FROZEN_CANDIDATE",
                          "opened_at": datetime.now(timezone.utc).isoformat()}
                write_json(ledger_path, ledger)
            heldout = _subset(reference, plan["holdout_dates"])
            end = max(plan["holdout_dates"])
            default_rows = replay(frame, item, CoreConfig(), end)
            candidate_rows = replay(frame, item, config, end)
            causal_checks = []
            checkpoints = sorted({plan["holdout_dates"][i] for i in (0, len(plan["holdout_dates"]) // 2, -1)})
            for day in checkpoints:
                complete = next(row for row in candidate_rows if str(row["as_of"])[:10] == day)
                prefix = replay(frame, item, config, day)[-1]
                passed = digest(complete) == digest(prefix)
                causal_checks.append({"as_of": day, "prefix_equal": passed})
                if not passed:
                    raise ValueError("candidate failed causal prefix check; holdout remains consumed")
            default_metric = score(heldout, default_rows)
            candidate_metric = score(heldout, candidate_rows)
            uncertainty = paired_cluster_bootstrap(default_metric, candidate_metric, reference=heldout,
                seed=VALIDATION_POLICY["bootstrap_seed"], repeats=VALIDATION_POLICY["bootstrap_repeats"])
            reasons = []
            if candidate_metric["status"] != "complete_scope" or default_metric["status"] != "complete_scope":
                reasons.append("incomplete_holdout_label_support")
            if candidate_metric["joint_macro_f1"] <= default_metric["joint_macro_f1"]:
                reasons.append("holdout_f1_did_not_improve")
            if candidate_metric["joint_agreement"] < default_metric["joint_agreement"]:
                reasons.append("holdout_agreement_regressed")
            if not uncertainty["inference_eligible"]:
                reasons.append("insufficient_effective_test_clusters")
            intervals = uncertainty["intervals"]
            if (intervals["joint_macro_f1"] is None or intervals["joint_agreement"] is None
                    or intervals["joint_macro_f1"]["lower"] <= 0 or intervals["joint_agreement"]["lower"] < 0):
                reasons.append("paired_improvement_interval_does_not_support_promotion")
            if implementation_id() != manifest["context"]["implementation"]:
                raise ValueError("implementation changed during validation; frozen holdout remains consumed")
            status = ("VALIDATED_ON_LABELED_HOLDOUT" if not reasons else
                      "INSUFFICIENT_TEST_POWER" if not uncertainty["inference_eligible"] else "NO_ROBUST_IMPROVEMENT")
            result = {"schema_version": 1, "status": status, "identity": identity,
                      "market": candidate["market"], "symbol": candidate["symbol"], "label_mode": candidate["label_mode"],
                      "reasons": reasons, "holdout_evaluated": True, "profile_export_allowed": not reasons,
                      "validated_at": ledger.get("validated_at", datetime.now(timezone.utc).isoformat()),
                      "default_metrics": default_metric, "candidate_metrics": candidate_metric,
                      "default_summaries": score_summaries(default_metric),
                      "candidate_summaries": score_summaries(candidate_metric),
                      "uncertainty": uncertainty, "stability": stability, "causal_checks": causal_checks,
                      "label_information_end": max(group_information_bounds(label)[1] for label in heldout["labels"]),
                      "scope": "Offline historical out-of-sample evaluation of this security's labeled samples. "
                          "Does not establish whole-history accuracy, profitability or contemporaneous deployment."}
            if ledger.get("status") == "COMPLETED" and ledger["validation_identity"] != digest(result):
                raise ValueError("recovered result differs from committed validation")
            # Commit the expected result digest first. A crash before report writing
            # may recover the identical candidate/result, never seal an unsigned report.
            write_json(ledger_path, {"identity": identity, "status": "COMPLETED",
                                    "validated_at": result["validated_at"], "validation_identity": digest(result)})
            write_json(directory / "validation.json", result)
            return result


def export_profile(run, output, *, effective_from=None):
    """Export one validated symbol profile. Caller must opt in to applying it."""
    directory, output = Path(run).resolve(), Path(output).resolve()
    with run_lock(directory):
        manifest, _, _, reference, plan, candidate, config, _ = _verified_run(directory)
        validation = _read(directory / "validation.json")
        ledger = _read(directory / "holdout-use.json")
        expected_identity = {"run_signature": manifest["signature"], "candidate_identity": digest(candidate),
                             "validation_policy": VALIDATION_POLICY}
        from .cli import _subset
        check_owner(_subset(reference, plan["holdout_dates"]), expected_identity)
        if (validation.get("status") != "VALIDATED_ON_LABELED_HOLDOUT"
                or validation.get("profile_export_allowed") is not True
                or validation.get("reasons") != [] or validation.get("identity") != expected_identity
                or ledger.get("status") != "COMPLETED" or ledger.get("identity") != expected_identity
                or ledger.get("validation_identity") != digest(validation)):
            raise ValueError("candidate lacks an unchanged successful labeled-holdout validation")
        if output.exists():
            payload = _read(output)
            records = payload.get("profiles", [])
            if (len(records) == 1 and records[0].get("validation_identity") == digest(validation)
                    and (effective_from is None or records[0].get("effective_from") == effective_from)):
                fixed = {"market": candidate["market"], "symbol": candidate["symbol"], "timeframe": "1d",
                         "label_mode": candidate["label_mode"], "status": "VALIDATED_ON_LABELED_HOLDOUT",
                         "params": asdict(config), "training_end": candidate["training_end"],
                         "label_information_end": validation["label_information_end"],
                         "run_signature": manifest["signature"], "validation_scope": validation["scope"],
                         "source_identity": manifest["context"]["data"]}
                if any(records[0].get(key) != value for key, value in fixed.items()):
                    raise ValueError("existing exported profile changed")
                resolve_config(candidate["market"], candidate["symbol"], profiles=payload,
                               as_of=records[0]["effective_from"], label_mode=candidate["label_mode"])
                return payload
            raise ValueError("profile destination already exists with different content")
        now = datetime.now(timezone.utc)
        earliest = max(now.date(), datetime.fromisoformat(validation["label_information_end"]).date(),
                       datetime.fromisoformat(candidate["training_end"]).date()) + timedelta(days=1)
        effective_from = iso_date(effective_from) if effective_from is not None else earliest.isoformat()
        if effective_from < earliest.isoformat():
            raise ValueError("effective date must follow creation and all price/label information")
        record = {"market": candidate["market"], "symbol": candidate["symbol"], "timeframe": "1d",
                  "label_mode": candidate["label_mode"], "status": "VALIDATED_ON_LABELED_HOLDOUT",
                  "params": asdict(config), "training_end": candidate["training_end"],
                  "label_information_end": validation["label_information_end"], "created_at": now.isoformat(),
                  "effective_from": effective_from, "run_signature": manifest["signature"],
                  "validation_identity": digest(validation), "validation_scope": validation["scope"],
                  "source_identity": manifest["context"]["data"],
                  "application_note": "Explicit opt-in. Earlier charts require later-calibrated reconstruction mode."}
        payload = {"schema_version": 1, "profiles": [record]}
        resolve_config(record["market"], record["symbol"], profiles=payload, as_of=effective_from,
                       label_mode=record["label_mode"])
        write_json(output, payload)
        return payload
