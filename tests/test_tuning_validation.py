"""Engineering fixtures exercise promotion; they are never human market labels."""
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from research.tuning import cli, holdout, validation
from research.tuning.labels import joint_classes, load_labels
from research.tuning.metrics import score
from research.tuning.policy import VALIDATION_POLICY
from research.tuning.profiles import resolve_config
from research.tuning.runner import digest, write_json
from research.tuning.splits import build_split_plan
from wyckoff_core.models import CoreConfig
from test_tuning_evaluation import complete_reference, label


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "holdout-registry")
    raw = complete_reference("asof")
    for i in range(4, 12):
        for panel_index, panel in enumerate(("cycle", "window")):
            base = date(2025, 1, 1) + timedelta(days=(i * 2 + panel_index) * 25)
            for j, state in enumerate(joint_classes("asof")):
                phase, regime = state.split("/") if "/" in state else ("no_observable_phase", "none")
                day = (base + timedelta(days=j)).isoformat()
                raw["labels"].append(label(day, phase=phase, regime=regime, panel=panel,
                    group=f"holdout_{panel}_{i}", role="holdout", mode="asof", information_start=base.isoformat()))
    raw["labels"].sort(key=lambda x: x["as_of"])
    label_path = tmp_path / "labels.json"
    write_json(label_path, raw)
    dates = pd.date_range("2009-01-01", "2026-09-29")
    frame = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.}, index=dates)
    item = {"continuous_segments": [{"start": "2009-01-01", "end": "2026-09-29"}], "attrs": {"fixture": True}}
    monkeypatch.setattr(validation, "load_history", lambda *args: (frame, item))
    reference = cli._qualified_reference(load_labels(label_path, market="us_equities", symbol="MSFT", label_mode="asof"),
                                         frame, item, "us_equities")
    plan = build_split_plan(reference)
    assert plan["status"] == "READY"
    # Deterministic synthetic predictions with different, nondegenerate group
    # improvements. Real production replay is tested in test_tuning_tool.py.
    def replay(frame, item, config, end=None):
        result = []
        for record in reference["labels"]:
            if end and record["as_of"] > end:
                continue
            group = int(record.get("cycle_id", record.get("block_id")).rsplit("_", 1)[1])
            j = joint_classes("asof").index(record["joint_state"])
            correct = (j + group) % 13 < (10 - group % 3 if config.phase_e_closes == 5 else 2 + group % 4)
            if correct:
                phase, regime = record["phase"], record["regime"]
            else:
                wrong = joint_classes("asof")[(j + 3) % 13]
                phase, regime = wrong.split("/") if "/" in wrong else ("no_observable_phase", "none")
            result.append({"as_of": record["as_of"], "phase": phase, "regime": regime, "status": "candidate"})
        return result
    monkeypatch.setattr(validation, "replay", replay)
    context = {"market": "us_equities", "symbol": "MSFT", "label_mode": "asof", "history": str(tmp_path),
        "labels_path": str(label_path), "holdout_start": "2024-10-01", "holdout_end": "2026-09-30",
        "data": digest({"attrs": item["attrs"], "dates": frame.index.tolist(), "values": frame.to_numpy().tolist()}),
        "labels": digest(reference), "splits": digest(plan), "implementation": cli.implementation_id(),
        "validation_policy": VALIDATION_POLICY}
    signature = digest(context)
    candidate_config = CoreConfig(phase_e_closes=5)
    candidate = {"schema_version": 1, "status": "CANDIDATE_ONLY", "market": "us_equities", "symbol": "MSFT",
        "timeframe": "1d", "label_mode": "asof", "params": asdict(candidate_config), "run_signature": signature,
        "training_end": max(plan["development_dates"]), "label_information_end": max(plan["development_dates"]),
        "created_at": "2026-09-30T08:00:00+00:00", "accuracy_claim": "not_validated_on_holdout", "holdout_evaluated": False}
    studies = []
    for i, fold in enumerate(plan["outer"]):
        ref = cli._subset(reference, fold["test_dates"])
        studies.append({"fold": f"outer-{i}", "sampler": "tpe", "params": candidate["params"],
            "outer_metrics": score(ref, replay(frame, item, candidate_config)),
            "outer_default_metrics": score(ref, replay(frame, item, CoreConfig()))})
    development = cli._subset(reference, sorted({d for fold in plan["final_inner"] for d in fold["test_dates"]}))
    studies.append({"fold": "final-development", "sampler": "tpe", "params": candidate["params"],
        "development_metrics": score(development, replay(frame, item, candidate_config)),
        "default_metrics": score(development, replay(frame, item, CoreConfig()))})
    result = {"status": "SEARCH_COMPLETED", "candidate": candidate, "studies": studies}
    run = tmp_path / "run"
    write_json(run / "run.json", {"signature": signature, "context": context})
    write_json(run / "candidate.json", candidate)
    write_json(run / "results.json", result)
    write_json(run / "split-plan.json", plan)
    write_json(run / "selection.json", {"run_signature": signature, "candidate_identity": digest(candidate),
                                       "sampler": "tpe", "results_identity": digest(result)})
    return run


def test_frozen_validation_profile_and_cached_result(frozen_run, tmp_path, monkeypatch):
    result = validation.validate_run(frozen_run)
    assert result["status"] == "VALIDATED_ON_LABELED_HOLDOUT", result["reasons"]
    assert result["uncertainty"]["effective_clusters"] == {"cycle": 12, "window": 12}
    assert all(check["prefix_equal"] for check in result["causal_checks"])
    monkeypatch.setattr(validation, "replay", lambda *args: pytest.fail("cached validation must not rescore holdout"))
    assert validation.validate_run(frozen_run) == result
    profile = validation.export_profile(frozen_run, tmp_path / "profile.json")
    item = profile["profiles"][0]
    assert resolve_config("us_equities", "MSFT", profiles=profile, as_of=item["effective_from"]).phase_e_closes == 5
    with pytest.raises(ValueError, match="later-calibrated"):
        resolve_config("us_equities", "MSFT", profiles=profile, as_of="2025-01-01")
    assert validation.export_profile(frozen_run, tmp_path / "profile.json") == profile
    assert cli.main(["export", "--run", str(frozen_run), "--output", str(tmp_path / "second.json")]) == 0


@pytest.mark.parametrize("changed", ["candidate", "labels", "policy", "implementation", "results"])
def test_changed_inputs_rejected_before_holdout(frozen_run, changed):
    if changed == "labels":
        path = frozen_run.parent / "labels.json"
        payload = json.loads(path.read_text()); payload["labels"][0]["phase"] = "B"
    else:
        path = frozen_run / ("candidate.json" if changed == "candidate" else "results.json" if changed == "results" else "run.json")
        payload = json.loads(path.read_text())
        if changed == "candidate": payload["params"]["phase_e_closes"] = 4
        elif changed == "results": payload["studies"][0]["outer_metrics"]["joint_macro_f1"] = 1
        elif changed == "policy": payload["context"]["validation_policy"]["bootstrap_repeats"] = 2
        else: payload["context"]["implementation"] = "changed"
    write_json(path, payload)
    with pytest.raises(ValueError): validation.validate_run(frozen_run)
    assert not (frozen_run / "holdout-use.json").exists()


def test_changed_report_and_exported_profile_rejected(frozen_run, tmp_path):
    validation.validate_run(frozen_run)
    profile_path = tmp_path / "profile.json"
    validation.export_profile(frozen_run, profile_path)
    payload = json.loads(profile_path.read_text()); payload["profiles"][0]["params"]["phase_e_closes"] = 4
    write_json(profile_path, payload)
    with pytest.raises(ValueError, match="profile changed"): validation.export_profile(frozen_run, profile_path)
    path = frozen_run / "validation.json"
    payload = json.loads(path.read_text()); payload["candidate_metrics"]["joint_macro_f1"] = 1
    write_json(path, payload)
    with pytest.raises(ValueError, match="digest"): validation.validate_run(frozen_run)
    with pytest.raises(ValueError): validation.export_profile(frozen_run, tmp_path / "other.json")


def test_unsigned_report_under_opened_ledger_never_sealed(frozen_run):
    validation.validate_run(frozen_run)
    path = frozen_run / "holdout-use.json"
    ledger = json.loads(path.read_text()); ledger["status"] = "OPENED_FOR_FROZEN_CANDIDATE"
    ledger.pop("validation_identity"); write_json(path, ledger)
    with pytest.raises(ValueError, match="committed holdout digest"): validation.validate_run(frozen_run)
    assert json.loads(path.read_text())["status"] == "OPENED_FOR_FROZEN_CANDIDATE"


def test_committed_ledger_recovers_missing_report_identically(frozen_run, monkeypatch):
    original = validation.write_json
    def crash_before_report(path, value):
        if Path(path).name == "validation.json": raise OSError("simulated crash after digest commit")
        original(path, value)
    monkeypatch.setattr(validation, "write_json", crash_before_report)
    with pytest.raises(OSError): validation.validate_run(frozen_run)
    assert not (frozen_run / "validation.json").exists()
    before = json.loads((frozen_run / "holdout-use.json").read_text())
    monkeypatch.setattr(validation, "write_json", original)
    result = validation.validate_run(frozen_run)
    assert before["validation_identity"] == digest(result)
    assert before == json.loads((frozen_run / "holdout-use.json").read_text())


def test_export_refuses_past_effective_date(frozen_run, tmp_path):
    validation.validate_run(frozen_run)
    with pytest.raises(ValueError, match="effective date"):
        validation.export_profile(frozen_run, tmp_path / "profile.json", effective_from="2025-01-01")
    assert not (tmp_path / "profile.json").exists()


def test_other_run_cannot_reuse_same_holdout_after_seed_or_path_change(frozen_run, tmp_path):
    other = tmp_path / "other-run"
    shutil.copytree(frozen_run, other)
    manifest = json.loads((other / "run.json").read_text())
    manifest["context"]["seed"] = 999
    manifest["signature"] = digest(manifest["context"])
    result = json.loads((other / "results.json").read_text())
    result["candidate"]["run_signature"] = manifest["signature"]
    write_json(other / "run.json", manifest)
    write_json(other / "candidate.json", result["candidate"])
    write_json(other / "results.json", result)
    write_json(other / "selection.json", {"run_signature": manifest["signature"],
        "candidate_identity": digest(result["candidate"]), "results_identity": digest(result), "sampler": "tpe"})
    assert validation.validate_run(frozen_run)["profile_export_allowed"]
    with pytest.raises(ValueError, match="another frozen candidate"):
        validation.validate_run(other)
    assert not (other / "holdout-use.json").exists()


def test_deleted_shared_ownership_blocks_cache_and_export(frozen_run, tmp_path):
    validation.validate_run(frozen_run)
    (holdout.REGISTRY / "us_equities/MSFT/uses.json").unlink()
    with pytest.raises(ValueError, match="shared holdout ownership"):
        validation.validate_run(frozen_run)
    with pytest.raises(ValueError, match="shared holdout ownership"):
        validation.export_profile(frozen_run, tmp_path / "profile.json")


def test_unstable_development_never_opens_holdout(frozen_run, monkeypatch):
    monkeypatch.setattr(validation, "legal_neighbors", lambda *args: [])
    result = validation.validate_run(frozen_run)
    assert result["status"] == "CANDIDATE_ONLY"
    assert not result["holdout_evaluated"]
    assert not (frozen_run / "holdout-use.json").exists()
    assert not (holdout.REGISTRY / "us_equities/MSFT/uses.json").exists()
