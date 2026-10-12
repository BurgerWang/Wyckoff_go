"""Synthetic fixtures check software only; they are never market reference truth."""
import json
from datetime import date, timedelta

import pytest

from research.tuning.labels import joint_classes, load_labels
from research.tuning.metrics import score
from research.tuning.splits import build_split_plan


def label(day, *, phase="A", regime="accumulation", panel="cycle", group="c1",
          role="development", mode="retrospective", information_start=None, information_end=None):
    return {"as_of": day, "phase": phase, "regime": regime, "market": "us_equities", "symbol": "MSFT",
            "timeframe": "1d", "label_mode": mode, "independence": "external_human",
            "source_id": "fixture_human", "adjudication": "explicit", "panel": panel,
            "cycle_id" if panel == "cycle" else "block_id": group,
            "information_start": information_start or day, "information_end": information_end or day,
            "role": role}


def reference(labels, mode="retrospective", symbol="MSFT"):
    return {"labels": labels, "market": "us_equities", "symbol": symbol, "label_mode": mode,
            "sources": [{"id": "fixture_human", "original_symbol": symbol, "original_market": "US equity",
                         "original_timeframe": "1d", "author": "Synthetic test-only fixture",
                         "source_url": "https://example.invalid/software-fixture", "hindsight": mode == "retrospective"}]}


def load(tmp_path, ref, mode="retrospective"):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(ref))
    return load_labels(path, market="us_equities", symbol="MSFT", label_mode=mode)


def complete_reference(mode="retrospective"):
    """Both panels, whole independent groups, all fixed states in each group."""
    result = []
    for role, count, origin in (("development", 12, date(2010, 1, 1)), ("holdout", 4, date(2025, 1, 1))):
        for i in range(count):
            for panel_index, panel in enumerate(("cycle", "window")):
                base = origin + timedelta(days=(i * 2 + panel_index) * 25)
                states = joint_classes(mode)
                end = (base + timedelta(days=len(states) - 1)).isoformat()
                for j, state in enumerate(states):
                    phase, regime = state.split("/") if "/" in state else ("no_observable_phase", "none")
                    day = (base + timedelta(days=j)).isoformat()
                    result.append(label(day, phase=phase, regime=regime, panel=panel,
                                        group=f"{role}_{panel}_{i}", role=role, mode=mode,
                                        information_start=base.isoformat(),
                                        information_end=day if mode == "asof" else end))
    return reference(result, mode)


def test_loader_normalizes_source_and_preserves_provenance(tmp_path):
    ref = load(tmp_path, reference([label("2012-01-01")]))
    item = ref["labels"][0]
    assert item["joint_state"] == "A/accumulation"
    assert item["source"]["author"] == "Synthetic test-only fixture"
    assert item["information_start"] == "2012-01-01"
    assert item["role"] == "development"
    assert not item["diagnostic_only"]


@pytest.mark.parametrize("changes", [{"ambiguity_resolved": False}, {"resolver": ""},
    {"resolution_rationale": ""}, {"resolver": "Synthetic test-only fixture"}])
def test_https_source_resolved_ambiguity_still_requires_independent_resolution(tmp_path, changes):
    item = label("2012-01-01")
    item.update(ambiguity="resolved", ambiguity_resolved=True, resolver="Independent fixture reviewer",
                resolution_rationale="Fixture review only")
    item.update(changes)
    with pytest.raises(ValueError, match="resol"):
        load(tmp_path, reference([item]))


@pytest.mark.parametrize("field,value", [("as_of", "2012-02-30"), ("phase", "F"),
    ("phase", "?"), ("regime", "imagined"), ("regime", "undetermined"),
    ("timeframe", "1h"), ("label_mode", "asof"), ("independence", "rule_based"),
    ("independence", "llm"), ("independence", "synthetic"), ("eligible", False),
    ("role", "whatever"), ("source_id", "missing"), ("adjudication", "ambiguous"), ("ambiguity", True),
    ("information_start", "2012-01-02"), ("information_end", "2011-12-31")])
def test_loader_rejects_malformed_or_nonindependent_truth(tmp_path, field, value):
    item = label("2012-01-01")
    item[field] = value
    with pytest.raises(ValueError):
        load(tmp_path, reference([item]))


@pytest.mark.parametrize("field,value", [("original_symbol", "AAPL"), ("original_market", "CME futures"),
    ("original_timeframe", "1h"), ("author", ""), ("source_url", "http://example.invalid"),
    ("independence", "synthetic")])
def test_loader_rejects_unattributable_or_transplanted_source(tmp_path, field, value):
    ref = reference([label("2012-01-01")])
    ref["sources"][0][field] = value
    with pytest.raises(ValueError):
        load(tmp_path, ref)


def test_loader_rejects_duplicates_and_mode_mixing(tmp_path):
    item = label("2012-01-01")
    with pytest.raises(ValueError, match="duplicate"):
        load(tmp_path, reference([item, dict(item)]))
    with pytest.raises(ValueError, match="label_mode"):
        load(tmp_path, reference([item]), "asof")
    ref = reference([label("2012-01-01", mode="asof", information_end="2012-01-02")], "asof")
    with pytest.raises(ValueError, match="future"):
        load(tmp_path, ref, "asof")


def test_legacy_nine_points_remain_diagnostic_and_cannot_tune():
    ref = load_labels("research/labels/independent_phase_labels.json", market="us_equities",
                      symbol="AAPL", label_mode="retrospective")
    assert len(ref["labels"]) == 9
    assert ref["diagnostic_only"]
    assert all("missing_information_end" in item["provenance_issues"] for item in ref["labels"])
    plan = build_split_plan(ref)
    assert plan["status"] == "LABELS_INSUFFICIENT"
    assert not plan["holdout_dates"]
    assert len(plan["excluded"]) == 2
    metrics = score(ref, [])
    assert metrics["status"] == "partial_scope_diagnostic"
    assert metrics["joint_agreement"] == metrics["coverage"] == 0
    assert len(metrics["classes"]) == 10
    assert "E/distribution" in metrics["missing_classes"]


def test_wrong_direction_and_unknown_predictions_are_errors():
    ref = reference([label("2012-01-01", phase="C"), label("2012-01-02", phase="E")])
    result = score(ref, [{"as_of": "2012-01-01", "phase": {"phase": "C", "regime": "distribution", "status": "candidate"}},
                         {"as_of": "2012-01-02", "phase": {"phase": "?", "regime": "unknown", "status": "candidate"}}])
    assert result["joint_macro_f1"] == result["joint_agreement"] == 0
    assert result["coverage"] == 0.5
    assert result["confusion"]["C/accumulation"]["C/distribution"] == 0.25
    assert result["confusion"]["E/accumulation"]["abstention"] == 0.25


@pytest.mark.parametrize("regime,expected", [("markup", "accumulation"), ("reaccumulation", "accumulation"),
    ("markdown", "distribution"), ("redistribution", "distribution")])
def test_e_direction_mapping(regime, expected):
    ref = reference([label("2012-01-01", phase="E", regime=expected)])
    result = score(ref, [{"as_of": "2012-01-01", "phase": "E", "regime": regime, "status": "candidate"}])
    assert result["joint_agreement"] == 1
    assert result["joint_macro_f1"] == pytest.approx(0.1)  # full 10, never observed-class mean


def test_asof_negative_distinguishes_normal_unknown_from_abstention():
    labels = [label(f"2012-01-0{i}", phase="no_observable_phase", regime="none", mode="asof") for i in range(1, 6)]
    predictions = [{"as_of": f"2012-01-0{i}", "phase": "?", "regime": "unknown", "status": status}
                   for i, status in enumerate(("candidate", "insufficient_data", "ambiguous", "insufficient_evidence"), 1)]
    predictions[0].update(range_id=None, hypotheses={}, normal_no_observable=True)
    predictions.append({"as_of": "2012-01-05", "phase": "A", "regime": "undetermined", "status": "candidate"})
    result = score(reference(labels, "asof"), predictions)
    assert len(result["classes"]) == 13
    assert result["joint_agreement"] == pytest.approx(0.2)
    assert result["coverage"] == pytest.approx(0.4)
    assert result["confusion"]["no_observable_phase"]["A/undetermined"] > 0


def test_real_production_no_range_is_negative_but_warmup_is_abstention():
    import pandas as pd
    from wyckoff_core.timeline import stage_timeline

    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 100},
                         index=pd.date_range("2012-01-01", periods=60))
    timeline = stage_timeline(frame)
    first, last = timeline["rows"][0], timeline["rows"][-1]
    assert first["phase"]["status"] == "insufficient_data"
    assert last["range_id"] is None
    assert last["phase"]["status"] == "insufficient_evidence"
    items = [label(row["as_of"].date().isoformat(), phase="no_observable_phase", regime="none", mode="asof")
             for row in (first, last)]
    metrics = score(reference(items, "asof"), timeline)
    assert metrics["cases"][0]["predicted"] == "abstention"
    assert metrics["cases"][1]["predicted"] == "no_observable_phase"
    assert metrics["joint_agreement"] == 0.5
    # Ambiguous references and uncontextualized '?' predictions cannot be negatives.
    assert score(reference(items[1:], "asof"), [{"as_of": items[1]["as_of"], "phase": "?",
                                                    "regime": "unknown", "status": "candidate"}])["coverage"] == 0


@pytest.mark.parametrize("missing_volume", ["all", "current", "background"])
def test_no_range_with_unavailable_volume_is_abstention(missing_volume):
    import pandas as pd
    from wyckoff_core.timeline import stage_timeline

    frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 100.0},
                         index=pd.date_range("2012-01-01", periods=65))
    if missing_volume == "all":
        frame["Volume"] = 0.0
    else:
        frame.loc[frame.index[-1 if missing_volume == "current" else -10], "Volume"] = 0.0
    row = stage_timeline(frame)["rows"][-1]
    assert row["range_id"] is None
    item = label(row["as_of"].date().isoformat(), phase="no_observable_phase", regime="none", mode="asof")
    result = score(reference([item], "asof"), [row])
    assert result["cases"][0]["predicted"] == "abstention"
    assert result["coverage"] == 0


def test_uncertain_asof_question_mark_cannot_be_reference_negative(tmp_path):
    ref = reference([label("2012-01-01", phase="?", regime="unknown", mode="asof")], "asof")
    with pytest.raises(ValueError, match="reference phase"):
        load(tmp_path, ref, "asof")


def test_asof_undetermined_ab_supported_but_retrospective_b_abstains():
    items = [label("2012-01-01", phase="A", regime="undetermined", mode="asof"),
             label("2012-01-02", phase="B", regime="undetermined", mode="asof")]
    predictions = [{"as_of": item["as_of"], "phase": item["phase"], "regime": "trading range", "status": "candidate"} for item in items]
    assert score(reference(items, "asof"), predictions)["joint_agreement"] == 1
    result = score(reference([label("2012-01-02", phase="B")]), predictions)
    assert result["coverage"] == result["joint_agreement"] == 0
    assert result["per_class"]["B/accumulation"]["recall"] == 0


def test_panel_and_group_weights_ignore_group_size():
    items = [label(f"2012-01-{i:02}", group="many") for i in range(1, 11)]
    items += [label("2012-02-01", group="one"), label("2012-03-01", panel="window", group="w1")]
    predictions = [{"as_of": item["as_of"], "phase": "A", "regime": "distribution" if item["cycle_id"] == "many" else "accumulation"}
                   for item in items if item["panel"] == "cycle"]
    predictions.append({"as_of": "2012-03-01", "phase": "A", "regime": "accumulation"})
    result = score(reference(items), predictions)
    assert result["joint_agreement"] == pytest.approx(0.75)
    assert result["panels"]["cycle"]["joint_agreement"] == pytest.approx(0.5)
    assert result["panels"]["window"]["weight"] == pytest.approx(0.5)
    assert [group["weight"] for group in result["groups"]] == pytest.approx([0.25, 0.25, 0.5])


def test_missing_prediction_counts_as_abstention_and_duplicates_rejected():
    ref = reference([label("2012-01-01")])
    assert score(ref, [])["missing_prediction_dates"] == ["2012-01-01"]
    prediction = {"as_of": "2012-01-01", "phase": "A", "regime": "accumulation"}
    with pytest.raises(ValueError, match="duplicate prediction"):
        score(ref, [prediction, prediction])


@pytest.mark.parametrize("mode", ["asof", "retrospective"])
def test_complete_fixture_constructs_grouped_nested_and_final_folds(tmp_path, mode):
    ref = load(tmp_path, complete_reference(mode), mode)
    plan = build_split_plan(ref)
    assert plan["status"] == "READY", plan["reasons"]
    assert len(plan["outer"]) == len(plan["final_inner"]) == 2
    assert plan["support"]["development"]["cycles"] == 12
    holdout = set(plan["holdout_dates"])
    pooled = []
    by_date = {item["as_of"]: item for item in ref["labels"]}
    for fold in plan["outer"]:
        assert len(fold["inner"]) == 2
        pooled.extend(fold["test_dates"])
        for item in [fold, *fold["inner"]]:
            assert max(item["train_dates"]) < min(item["test_dates"])
            assert not holdout.intersection(item["train_dates"] + item["test_dates"])
            train_groups = {by_date[day].get("cycle_id", by_date[day].get("block_id")) for day in item["train_dates"]}
            test_groups = {by_date[day].get("cycle_id", by_date[day].get("block_id")) for day in item["test_dates"]}
            assert not train_groups.intersection(test_groups)
            assert max(by_date[day]["information_end"] for day in item["train_dates"]) < min(by_date[day]["information_start"] for day in item["test_dates"])
    assert len(pooled) == len(set(pooled))
    json.dumps(plan)  # durable, serializable split plan


def test_information_overlap_purges_whole_training_group(tmp_path):
    raw = complete_reference()
    baseline = build_split_plan(load(tmp_path, raw))
    first_outer = baseline["outer"][0]
    train_day = first_outer["train_dates"][0]
    test_day = first_outer["test_dates"][0]
    item = next(item for item in raw["labels"] if item["as_of"] == train_day)
    cycle = item["cycle_id"]
    for entry in raw["labels"]:
        if entry.get("cycle_id") == cycle:
            entry["information_end"] = test_day
    plan = build_split_plan(load(tmp_path, raw))
    affected_dates = {entry["as_of"] for entry in raw["labels"] if entry.get("cycle_id") == cycle}
    assert not affected_dates.intersection(plan["outer"][0]["train_dates"])
    assert ["cycle", cycle] in plan["outer"][0]["purged_groups"]


def test_crossing_group_is_excluded_not_split(tmp_path):
    raw = complete_reference()
    raw["labels"] += [label("2024-09-30", group="crossing", information_end="2024-10-02"),
                       label("2024-10-02", group="crossing", role="holdout", information_start="2024-09-30")]
    plan = build_split_plan(load(tmp_path, raw))
    assert "2024-09-30" not in plan["development_dates"]
    assert "2024-10-02" not in plan["holdout_dates"]
    assert any(item["group"] == ["cycle", "crossing"] and "crosses_cutoff" in item["reason"] for item in plan["excluded"])


def test_shared_qualification_never_deletes_a_cross_cutoff_cycle_boundary(tmp_path):
    raw = complete_reference()
    crossing = [label("2024-09-30", group="qualified_crossing", information_end="2024-10-02"),
                label("2024-10-02", group="qualified_crossing", role="holdout", information_start="2024-09-30")]
    crossing[0]["qualification_issue"] = "common_warmup_or_zero_volume"
    raw["labels"] += crossing
    plan = build_split_plan(load(tmp_path, raw))
    assert "2024-10-02" not in plan["holdout_dates"]
    assert "2024-09-30" not in plan["development_dates"]
    excluded = next(item for item in plan["excluded"] if item["group"] == ["cycle", "qualified_crossing"])
    assert excluded["reason"] == "common_data_qualification_failed"
    assert excluded["dates"] == ["2024-09-30", "2024-10-02"]


def test_failed_diagnostic_qualification_still_taints_other_holdout_groups(tmp_path):
    raw = complete_reference()
    diagnostic = label("2025-12-01", role="diagnostic", group="bad_diagnostic", information_end="2025-12-10")
    diagnostic["qualification_issue"] = "common_warmup_or_zero_volume"
    raw["labels"] += [diagnostic, label("2025-12-05", role="holdout", group="relabelled")]
    plan = build_split_plan(load(tmp_path, raw))
    assert "2025-12-05" not in plan["holdout_dates"]
    assert any(item["group"] == ["cycle", "relabelled"] and item["reason"] == "known_diagnostic_information_overlap"
               for item in plan["excluded"])


def test_diagnostic_role_taints_entire_group_and_information_overlap(tmp_path):
    raw = complete_reference()
    raw["labels"] += [label("2025-12-01", role="diagnostic", group="tainted", information_end="2025-12-10"),
                       label("2025-12-02", role="holdout", group="tainted"),
                       label("2025-12-05", role="holdout", group="renamed")]
    plan = build_split_plan(load(tmp_path, raw))
    assert not {"2025-12-01", "2025-12-02", "2025-12-05"}.intersection(plan["holdout_dates"])
    assert any(item["group"] == ["cycle", "renamed"] and item["reason"] == "known_diagnostic_information_overlap" for item in plan["excluded"])


def test_known_doge_case_cannot_become_holdout_with_new_group_identity():
    item = label("2026-04-05", role="holdout", group="independent_new_name")
    item.update(market="binance_spot", symbol="DOGEUSDT")
    ref = {"labels": [item], "market": "binance_spot", "symbol": "DOGEUSDT", "label_mode": "retrospective"}
    plan = build_split_plan(ref)
    assert not plan["holdout_dates"]
    assert plan["excluded"][0]["reason"] == "known_diagnostic_information_overlap"


def test_missing_classes_or_panel_prevent_supervised_eligibility(tmp_path):
    raw = complete_reference()
    raw["labels"] = [item for item in raw["labels"] if item["phase"] != "E"]
    plan = build_split_plan(load(tmp_path, raw))
    assert plan["status"] == "LABELS_INSUFFICIENT"
    assert "E/distribution" in plan["support"]["holdout"]["missing_classes"]
    raw = complete_reference()
    raw["labels"] = [item for item in raw["labels"] if item["panel"] == "cycle"]
    plan = build_split_plan(load(tmp_path, raw))
    assert plan["status"] == "LABELS_INSUFFICIENT"
    assert plan["support"]["development"]["missing_panels"] == ["window"]


@pytest.mark.parametrize("mode", ["asof", "retrospective"])
def test_full_fixed_class_perfect_predictions_score_one(mode):
    ref = complete_reference(mode)
    predictions = [{"as_of": item["as_of"], "phase": item["phase"], "regime": item["regime"], "status": "candidate"}
                   for item in ref["labels"]]
    result = score(ref, predictions)
    assert result["status"] == "complete_scope"
    assert result["missing_classes"] == result["missing_panels"] == []
    assert result["joint_macro_f1"] == result["joint_agreement"] == result["coverage"] == 1


def test_holdout_end_is_exclusive_and_roles_cannot_move_dates(tmp_path):
    raw = complete_reference()
    raw["labels"] += [label("2026-09-30", role="holdout", group="exclusive_end"),
                       label("2025-12-31", role="development", group="false_development")]
    plan = build_split_plan(load(tmp_path, raw))
    assert "2026-09-30" not in plan["holdout_dates"]
    assert "2025-12-31" not in plan["development_dates"]
    assert any(item["reason"] == "role_does_not_match_holdout_dates" for item in plan["excluded"])
