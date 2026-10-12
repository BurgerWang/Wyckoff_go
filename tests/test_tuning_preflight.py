"""Reference metadata checks; synthetic labels are never market truth."""
from copy import deepcopy
from datetime import date, timedelta
import json

import pytest

from research.tuning.labels import joint_classes
from research.tuning.preflight import (
    evaluate_annotation_capacity, evaluate_reference_capacity,
    information_capacity_upper_bound,
)


def feasible_reference(mode="asof"):
    labels = []
    for i in range(12):
        for p, panel in enumerate(("cycle", "window")):
            start = date(2000, 1, 1) + timedelta(days=(2 * i + p) * 200)
            for offset, state in enumerate(joint_classes(mode)):
                day = (start + timedelta(days=offset)).isoformat()
                phase, regime = state.split("/") if "/" in state else (state, "none")
                labels.append({"as_of": day, "phase": phase, "regime": regime, "panel": panel,
                               "cycle_id" if panel == "cycle" else "block_id": f"{panel}{i}",
                               "role": "holdout", "information_start": start.isoformat(),
                               "information_end": day})
    return {"label_mode": mode, "labels": labels, "symbol": "FIXTURE", "market": "us_equities"}


def manifest(intervals, *, symbol="FIXTURE", market="us_equities"):
    return {"pack_id": "metadata-fixture", "source": {"market": market, "symbol": symbol},
            "settings": {"label_mode": "asof", "context_bars": 100,
                         "holdout_start": "2024-10-01", "holdout_end": "2026-09-30"},
            "cases": [{"as_of": end, "information_start": start, "information_end": end,
                       "panel": "window", "block_id": f"window{i}", "role": "holdout"}
                      for i, (start, end) in enumerate(intervals)]}


@pytest.mark.parametrize("mode", ["asof", "retrospective"])
def test_feasible_labelled_metadata_no_scores_no_mutation(mode):
    reference = feasible_reference(mode)
    reference["predictions"] = reference["prices"] = object()  # neither is read or serialized
    labels = deepcopy(reference["labels"])
    result = evaluate_reference_capacity(reference)
    assert result["eligible"] and result["geometry_eligible"]
    assert result["status"] == "REFERENCE_CAPACITY_READY"
    assert result["effective_clusters"] == result["raw_groups"] == {"cycle": 12, "window": 12}
    assert not result["missing_classes"] and not result["missing_panels"]
    assert result["policy_version"] == "labeled-holdout-v1"
    assert not result["metadata_only"]
    assert all(panels == {"cycle": 12, "window": 12} for panels in result["class_unit_support"].values())
    assert reference["labels"] == labels
    json.dumps(result)
    reference["labels"].reverse()
    assert result == evaluate_reference_capacity(reference)


def test_transitive_connected_exposures_merge_and_support_counts_effective_units():
    reference = {"label_mode": "retrospective", "labels": []}
    for i, (start, end) in enumerate((("2020-01-01", "2020-01-03"),
                                    ("2020-01-03", "2020-01-05"),
                                    ("2020-01-05", "2020-01-07"))):
        reference["labels"].append({"as_of": end, "information_start": start, "information_end": end,
                                    "role": "holdout", "panel": "cycle", "cycle_id": f"c{i}",
                                    "phase": "A", "regime": "accumulation"})
    result = evaluate_reference_capacity(reference)
    assert result["overlap_clusters"]["cycle"] == [["c0", "c1", "c2"]]
    assert result["effective_clusters"] == {"cycle": 1, "window": 0}
    assert result["raw_class_unit_support"]["A/accumulation"]["cycle"] == 3
    assert result["class_unit_support"]["A/accumulation"]["cycle"] == 1
    assert "missing_sampling_panels" in result["diagnostic_reasons"]


def test_closed_cross_panel_overlap_rejected_with_sufficient_per_panel_units():
    reference = feasible_reference()
    cycle = next(label for label in reversed(reference["labels"]) if label.get("cycle_id") == "cycle0")
    for label in reference["labels"]:
        if label.get("block_id") == "window0":
            label["information_start"] = cycle["information_end"]
    result = evaluate_reference_capacity(reference)
    assert result["unit_count_eligible"]
    assert result["cross_panel_information_overlap"]
    assert "cross_panel_information_overlap" in result["diagnostic_reasons"]
    assert not result["eligible"]


@pytest.mark.parametrize("issue,value,reason", [
    ("diagnostic_only", True, "diagnostic_group"),
    ("qualification_issue", "common_warmup_or_zero_volume", "common_data_qualification_failed"),
    ("provenance_issues", ["missing_source"], "missing_information_or_role_provenance"),
    ("role", None, "missing_or_invalid_reference_role"),
])
def test_contamination_is_not_cleared_by_counts_or_class_support(issue, value, reason):
    reference = feasible_reference()
    reference["labels"][0][issue] = value
    result = evaluate_reference_capacity(reference)
    assert result["unit_count_eligible"]
    assert not result["eligible"]
    assert reason in result["contaminated_groups"][0]["reasons"]


def test_diagnostic_information_taints_new_identity_and_known_market_cases():
    reference = feasible_reference()
    label = reference["labels"][0]
    reference["diagnostic_intervals"] = [(label["information_start"], label["information_end"])]
    result = evaluate_reference_capacity(reference)
    assert any("known_diagnostic_information_overlap" in group["reasons"] for group in result["contaminated_groups"])
    reference = {"label_mode": "asof", "market": "binance_spot", "symbol": "DOGEUSDT",
                 "labels": [{"as_of": "2026-04-01", "information_start": "2026-03-01",
                             "information_end": "2026-04-01", "role": "holdout", "panel": "window",
                             "block_id": "renamed", "phase": "A", "regime": "accumulation"}]}
    result = evaluate_reference_capacity(reference)
    assert result["contaminated_groups"][0]["reasons"] == ["known_diagnostic_information_overlap"]


def test_empty_and_missing_information_reference_have_explicit_reasons():
    empty = evaluate_reference_capacity({"label_mode": "asof", "labels": []})
    assert empty["missing_panels"] == ["cycle", "window"]
    assert empty["effective_clusters"] == {"cycle": 0, "window": 0}
    assert not empty["eligible"]
    reference = feasible_reference()
    reference["labels"][0].pop("information_start")
    result = evaluate_reference_capacity(reference)
    assert not result["reference_intervals_complete"]
    assert result["cross_panel_information_overlap"] is None
    assert not result["unit_count_eligible"]
    assert "missing_reference_information_intervals" in result["diagnostic_reasons"]


def test_input_changes_recompute_geometry_and_support_without_cache():
    reference = feasible_reference()
    assert evaluate_reference_capacity(reference)["eligible"]
    reference["labels"] = [label for label in reference["labels"] if label["phase"] != "E"]
    result = evaluate_reference_capacity(reference)
    assert set(result["missing_classes"]) == {"E/accumulation", "E/distribution"}
    assert not result["eligible"]
    reference["labels"] = [label for label in reference["labels"] if label.get("block_id") != "window11"]
    result = evaluate_reference_capacity(reference)
    assert result["effective_clusters"]["window"] == 11
    assert "fewer_than_12_effective_clusters_per_panel" in result["diagnostic_reasons"]


@pytest.mark.parametrize("mutation", ["duplicate", "future", "outside", "blank_label"])
def test_invalid_labelled_reference_is_rejected(mutation):
    reference = feasible_reference()
    if mutation == "duplicate":
        reference["labels"].append(dict(reference["labels"][0]))
    elif mutation == "future":
        reference["labels"][0]["information_end"] = reference["labels"][1]["as_of"]
    elif mutation == "outside":
        reference["labels"][1]["information_start"] = reference["labels"][2]["as_of"]
    else:
        reference["labels"][0]["phase"] = ""
    with pytest.raises(ValueError):
        evaluate_reference_capacity(reference)


# Group exposure envelopes copied from the actual blank v1 manifests on
# 2026-09-30. Only timestamps are retained; these contain no phase truth.
REAL_V1_ENVELOPES = {
    "AAPL": [("2024-10-09", "2025-04-01"), ("2024-11-20", "2025-05-14"),
             ("2025-01-06", "2025-06-30"), ("2025-02-20", "2025-08-08"),
             ("2025-05-12", "2025-10-31"), ("2025-07-01", "2025-12-16"),
             ("2025-08-13", "2026-02-02"), ("2025-09-23", "2026-03-13"),
             ("2025-11-04", "2026-04-28"), ("2025-12-19", "2026-06-09"),
             ("2026-02-05", "2026-07-28"), ("2026-03-19", "2026-09-04")],
    "BTCUSDT": [("2024-10-07", "2025-02-02"), ("2024-12-05", "2025-04-03"),
                ("2025-01-05", "2025-05-03"), ("2025-02-04", "2025-06-04"),
                ("2025-03-06", "2025-07-03"), ("2025-04-05", "2025-07-31"),
                ("2025-05-05", "2025-08-31"), ("2025-06-05", "2025-09-27"),
                ("2025-07-01", "2025-11-01"), ("2025-08-01", "2025-11-26"),
                ("2025-10-02", "2026-01-27"), ("2026-05-27", "2026-09-25")],
    "ETHUSDT": [("2024-10-07", "2025-02-02"), ("2024-11-04", "2025-03-06"),
                ("2024-12-05", "2025-04-03"), ("2025-01-05", "2025-05-03"),
                ("2025-09-02", "2025-12-26"), ("2025-10-02", "2026-01-27"),
                ("2026-05-27", "2026-09-25")],
    "DOGEUSDT": [("2024-10-07", "2025-02-02"), ("2025-07-01", "2025-11-01"),
                 ("2025-08-01", "2025-11-26"), ("2025-09-02", "2025-12-26"),
                 ("2025-10-02", "2026-01-27")],
}


@pytest.mark.parametrize("symbol,bars,raw,clusters", [("AAPL", 500, 12, 1),
    ("BTCUSDT", 729, 12, 2), ("ETHUSDT", 729, 7, 3), ("DOGEUSDT", 729, 5, 2)])
def test_real_two_year_v1_envelope_capacity_is_insufficient(symbol, bars, raw, clusters):
    pack = manifest(REAL_V1_ENVELOPES[symbol], symbol=symbol,
                    market="us_equities" if symbol == "AAPL" else "binance_spot")
    before = deepcopy(pack)
    result = evaluate_annotation_capacity(pack, holdout_bar_count=bars)
    assert pack == before
    assert result["raw_groups"]["window"] == raw
    assert result["effective_clusters"] == {"cycle": 0, "window": clusters}
    assert result["metadata_only"] and result["window_only"]
    assert not result["class_support_known"] and not result["eligible"]
    assert len(result["missing_classes"]) == 13
    assert result["information_capacity_upper_bound"]["provably_insufficient"]
    assert result["information_capacity_upper_bound"]["maximum_disjoint_information_clusters"] == bars // 100


def test_blank_manifest_never_fabricates_truth_and_reports_cutoff_crossing():
    pack = manifest([("2024-09-30", "2024-10-20")])
    pack["cases"][0].update(phase="A", regime="accumulation")
    result = evaluate_annotation_capacity(pack)
    assert result["class_unit_support"]["A/accumulation"] == {"cycle": 0, "window": 0}
    assert "information_crosses_partition_cutoff" in result["diagnostic_reasons"]
    assert result["cutoff_crossing_groups"] == [["window", "window0"]]
    assert not result["eligible"]
    result = evaluate_annotation_capacity(manifest([]))
    assert result["missing_panels"] == ["cycle", "window"]


def test_frozen_minimum_and_bar_bound_validation():
    assert information_capacity_upper_bound(1200, 100)["provably_insufficient"] is False
    assert information_capacity_upper_bound(1199, 100)["provably_insufficient"] is True
    assert not evaluate_reference_capacity(feasible_reference(), minimum=13)["eligible"]
    for minimum in (1, 11, 12.0, True):
        with pytest.raises(ValueError, match="frozen policy"):
            evaluate_reference_capacity(feasible_reference(), minimum=minimum)
    for available, context in ((-1, 100), (True, 100), (500.0, 100), (500, 0)):
        with pytest.raises(ValueError):
            information_capacity_upper_bound(available, context)
