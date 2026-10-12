"""Engineering fixtures only: never real human cycle boundaries or market truth."""
from copy import deepcopy
import json

import pytest

from test_tuning_annotation import history, payload  # noqa: F401
from research.tuning import annotation, cli
from research.tuning.cycles import (VERSION, prepare_cycle_annotation_pack,
                                    reference_case, validate_cycle_reviewer)
from research.tuning.holdout import _intervals
from research.tuning.labels import load_labels, group_information_bounds, group_intervals
from research.tuning.preflight import evaluate_annotation_capacity, reference_interval_clusters
from research.tuning.splits import build_split_plan


def registration():
    return {"version": "human-cycle-registration-v1", "market": "binance_spot", "symbol": "BTCUSDT",
            "timeframe": "1d", "label_mode": "asof",
            "registered_by": {"name": "Fixture selector", "identity_id": "selector-1", "type": "human",
                              "registered_at": "2026-10-01T00:00:00+07:00", "independent_human": True,
                              "no_strategy_outputs": True},
            "partition": {"development_start": "2019-01-01", "holdout_start": "2021-01-01", "holdout_end": "2026-09-30"},
            "cycles": [{"cycle_id": "fixture-human-development", "role": "development", "analysis_scale": "日线目标结构",
                        "parent_cycle": None, "information_start": "2020-01-01", "information_end": "2020-12-31",
                        "case_dates": ["2020-03-01", "2020-05-01"]},
                       {"cycle_id": "fixture-human-holdout", "role": "holdout", "analysis_scale": "日线目标结构",
                        "parent_cycle": None, "information_start": "2022-01-01", "information_end": "2022-12-31",
                        "case_dates": ["2022-03-01", "2022-05-01"]}]}


def prepare(history, tmp_path, raw=None):
    path, rubric, pack = tmp_path / "registration.json", tmp_path / "rubric.md", tmp_path / "cycle-pack"
    path.write_text(json.dumps(registration() if raw is None else raw))
    rubric.write_text("Independent human rubric fixture, not a human reference.\n")
    manifest = prepare_cycle_annotation_pack(history[0], path, pack, rubric_path=rubric)
    return pack, manifest


def originals(pack, tmp_path):
    result = []
    for key in ("a", "b"):
        raw = json.loads((pack / f"review-{key}.json").read_text())
        raw["reviewer"] = {"name": f"Fixture human {key}", "identity_id": key,
                           "reviewed_at": "2026-10-01T01:00:00+07:00", "independent_human": True,
                           "no_strategy_outputs": True, "no_future_prices_for_asof": True,
                           "blind_review": True, "no_other_review_seen": True, "independent_of_other_review": True}
        for row in raw["labels"]:
            row.update(status="completed", phase="B", regime="undetermined", notes="Fixture personally reviewed.")
        path = tmp_path / f"original-{key}.json"
        path.write_text(json.dumps(raw))
        result.append((raw, path))
    return result


def test_registered_cycle_physical_prefix_and_frozen_replay(history, tmp_path):
    pack, manifest = prepare(history, tmp_path)
    assert manifest["version"] == VERSION
    assert annotation._load_manifest(pack) == manifest
    for case in manifest["cases"]:
        data = payload(pack / case["chart"])
        assert data["bars"][0]["date"] == case["information_start"]
        assert data["bars"][-1]["date"] == case["as_of"] == case["information_end"]
        assert "group_information_end" not in data["case"]
    for name in ("annotation.json", "review-a.json", "review-b.json"):
        raw = json.loads((pack / name).read_text())
        assert raw["rubric_identity"] == manifest["rubric_identity"]
        assert not raw["reviewer"]["name"] and all(x["status"] == "blank" and x["phase"] is None for x in raw["labels"])
    cap = evaluate_annotation_capacity(manifest, role="holdout", holdout_bar_count=999)
    assert cap["effective_clusters"] == {"cycle": 1, "window": 0}
    assert cap["eligible"] is False
    assert cap["cluster_information_intervals"]["cycle"][0]["end"] == "2022-12-31"
    assert cap["information_capacity_upper_bound"]["status"] == "not_applicable_variable_cycle_information"


def test_retrospective_cycle_contains_declared_future_only_in_separate_mode(history, tmp_path):
    raw = registration()
    raw["label_mode"] = "retrospective"
    pack, manifest = prepare(history, tmp_path, raw)
    case = manifest["cases"][0]
    assert payload(pack / case["chart"])["bars"][-1]["date"] == "2020-12-31" > case["as_of"]
    assert annotation._load_manifest(pack) == manifest


@pytest.mark.parametrize("mutation, message", [
    ("empty", "no human cycles"), ("person", "named independent human"),
    ("scale", "analysis_scale"), ("future", "registered partition"),
    ("target", "unique actual session"), ("duplicate", "unique actual session"),
    ("diagnostic", "diagnostic or consumed"),
])
def test_bad_registration_fails_before_pack_generation(history, tmp_path, mutation, message):
    raw = registration()
    if mutation == "empty": raw["cycles"] = []
    elif mutation == "person": raw["registered_by"]["independent_human"] = False
    elif mutation == "scale": raw["cycles"][0]["analysis_scale"] = ""
    elif mutation == "future": raw["cycles"][0]["information_end"] = "2021-01-01"
    elif mutation == "target": raw["cycles"][0]["case_dates"] = ["2020-02-30"]
    elif mutation == "duplicate": raw["cycles"][0]["case_dates"] *= 2
    else:
        raw["cycles"][0].update(information_start="2019-01-01", information_end="2019-05-01", case_dates=["2019-02-01"])
    with pytest.raises(ValueError, match=message if mutation != "target" else "ISO date"):
        prepare(history, tmp_path, raw)
    assert not (tmp_path / "cycle-pack/manifest.json").exists()


def test_cycle_single_review_rejected_and_consensus_reloads_original_files(history, tmp_path):
    pack, manifest = prepare(history, tmp_path)
    a, b = originals(pack, tmp_path)
    with pytest.raises(ValueError, match="paired consensus"):
        annotation.import_annotations(pack, a[1])
    output = tmp_path / "reference.json"
    result = annotation.import_consensus_annotations(pack, a[1], b[1], output)
    loaded = load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")
    assert len(loaded["labels"]) == len(manifest["cases"])
    assert _intervals(loaded) == [["2020-01-01", "2020-12-31"], ["2022-01-01", "2022-12-31"]]
    a[0]["labels"][0]["phase"] = "A"
    a[1].write_text(json.dumps(a[0]))
    with pytest.raises(ValueError, match="artifact changed"):
        load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")
    assert result["sources"][0]["consensus_summary"]["merged_status_counts"]["completed"] == 4


@pytest.mark.parametrize("mutation", ["drop", "group", "scale", "remove_consensus"])
def test_consensus_reference_cannot_hide_completion_or_shorten_group(history, tmp_path, mutation):
    pack, _ = prepare(history, tmp_path)
    a, b = originals(pack, tmp_path)
    output = tmp_path / "reference.json"
    result = annotation.import_consensus_annotations(pack, a[1], b[1])
    if mutation == "drop": result["labels"].pop()
    elif mutation == "group": result["labels"][0]["group_information_end"] = result["labels"][0]["as_of"]
    elif mutation == "scale": result["labels"][0]["analysis_scale"] = "another scale"
    else: result["sources"][0].pop("consensus_reviews")
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")


def test_selector_later_cycle_exposure_rejects_earlier_cycle_endpoint(history, tmp_path):
    pack, manifest = prepare(history, tmp_path)
    raw = json.loads((pack / "review-a.json").read_text())
    raw["reviewer"] = {"name": "Renamed fixture", "identity_id": "SELECTOR-1"}
    raw["labels"][0].update(status="completed", as_of="2020-12-31")
    with pytest.raises(ValueError, match="already saw later"):
        validate_cycle_reviewer(manifest, raw)


def test_only_early_completed_case_retains_full_group_purge_and_holdout_bounds(history, tmp_path):
    pack, manifest = prepare(history, tmp_path)
    case = reference_case(manifest, manifest["cases"][0])
    label = {**case, "phase": "B", "regime": "undetermined"}
    reference = {"labels": [label], "market": "binance_spot", "symbol": "BTCUSDT", "label_mode": "asof"}
    assert reference_interval_clusters(reference)["bounds"][("cycle", case["cycle_id"])][1] == "2020-12-31"
    assert _intervals(reference) == [["2020-01-01", "2020-12-31"]]
    plan = build_split_plan(reference, holdout_start="2020-07-01")
    assert plan["development_dates"] == []
    assert any(x["reason"] == "group_or_information_crosses_cutoff_or_outside_scope" for x in plan["excluded"])
    shortened = deepcopy(label)
    shortened["group_information_end"] = "2020-04-01"
    later = {**label, "as_of": "2020-03-02", "information_end": "2020-03-02"}
    with pytest.raises(ValueError, match="inconsistent"):
        group_intervals([shortened, later])
    incomplete = deepcopy(label)
    incomplete.pop("group_information_end")
    with pytest.raises(ValueError, match="complete group"):
        group_information_bounds(incomplete)


def test_cycle_cli_and_empty_real_registration_do_not_create_truth(history, tmp_path, capsys):
    pack, _ = prepare(history, tmp_path)
    a, b = originals(pack, tmp_path)
    output = tmp_path / "cli-reference.json"
    assert cli.main(["import-consensus", "--pack", str(pack), "--review-a", str(a[1]), "--review-b", str(b[1]), "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["labels"]) == 4


def test_deleted_whole_consensus_source_cannot_skip_original_file_verification(history, tmp_path):
    pack, _ = prepare(history, tmp_path)
    a, b = originals(pack, tmp_path)
    result = annotation.import_consensus_annotations(pack, a[1], b[1])
    result["labels"] = []
    a[1].unlink()
    output = tmp_path / "deleted-source-labels.json"
    output.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="artifact changed or missing"):
        load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")


def test_explicit_internal_selector_and_review_before_registration_fail(history, tmp_path):
    raw = registration()
    raw["registered_by"]["independence"] = "strategy_internal"
    with pytest.raises(ValueError, match="named independent human"):
        prepare(history, tmp_path, raw)
    pack, manifest = prepare(history, tmp_path)
    a, _ = originals(pack, tmp_path)
    a[0]["reviewer"]["reviewed_at"] = "2026-09-30T23:59:59+07:00"
    with pytest.raises(ValueError, match="after registration"):
        validate_cycle_reviewer(manifest, a[0])


def test_real_file_adjudication_chain_cannot_be_replaced_by_a_resolver_name(history, tmp_path):
    pack, _ = prepare(history, tmp_path)
    a, b = originals(pack, tmp_path)
    b[0]["labels"][0].update(phase="C", regime="distribution")
    b[1].write_text(json.dumps(b[0]))
    partial = annotation.import_consensus_annotations(pack, a[1], b[1])
    assert len(partial["labels"]) == 3
    judge = deepcopy(a[0])
    judge["reviewer"] = {"name": "Fixture adjudicator", "reviewed_at": "2026-10-01T02:00:00+07:00",
                         "independent_human": True, "no_strategy_outputs": True, "no_future_prices_for_asof": True}
    for row in judge["labels"]:
        row.update(status="blank", phase=None, regime=None)
    judge["labels"][0].update(status="completed", phase="D", regime="distribution", ambiguity_resolved=True,
                              resolver="Fixture adjudicator", resolution_rationale="Fixture independent adjudication.")
    path, output = tmp_path / "judge.json", tmp_path / "adjudicated.json"
    path.write_text(json.dumps(judge))
    result = annotation.import_consensus_annotations(pack, a[1], b[1], output, adjudication=path)
    assert len(load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")["labels"]) == 4
    assert result["labels"][0]["phase"] == "D" and result["labels"][0]["resolver"] == "Fixture adjudicator"
    judge["labels"][0]["phase"] = "E"
    path.write_text(json.dumps(judge))
    with pytest.raises(ValueError, match="artifact changed"):
        load_labels(output, market="binance_spot", symbol="BTCUSDT", label_mode="asof")


def test_registry_and_exported_cycle_bounds_do_not_shrink_to_target_dates(history, tmp_path, monkeypatch):
    from research.tuning import holdout
    raw = registration()
    path, rubric, pack = tmp_path / "registration.json", tmp_path / "rubric.md", tmp_path / "cycle-pack"
    path.write_text(json.dumps(raw))
    rubric.write_text("Human rubric fixture")
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "registry")
    prepare_cycle_annotation_pack(history[0], path, pack, rubric_path=rubric)
    bounds = json.loads((pack / "cycle-bounds.json").read_text())
    assert bounds["human_cycle_bounds"][0]["information_end"] == "2020-12-31"
    bounds["human_cycle_bounds"][0]["information_end"] = "2020-05-01"
    (pack / "cycle-bounds.json").write_text(json.dumps(bounds))
    with pytest.raises(ValueError, match="exported bounds changed"):
        annotation._load_manifest(pack)
