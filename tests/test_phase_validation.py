import json

import pandas as pd
import pytest

from research.phase_validation import evaluate_labels


def inputs(tmp_path, phases=("A", "?", "C")):
    csv = tmp_path / "AAPL.csv"
    dates = ["2009-01-05", "2009-01-06", "2009-01-07"]
    pd.DataFrame({"Date": dates, "phase": phases, "regime": "trading range", "status": "candidate"}).to_csv(csv, index=False)
    (tmp_path / "report.json").write_text(json.dumps({"profiles": {
        "us_equities": {"AAPL": {"status": "complete", "csv": str(csv)}}, "binance_spot": {}}}))
    source = {"id": "human", "author": "Expert", "original_symbol": "AAPL", "original_market": "US equity",
              "original_timeframe": "1d", "source_url": "https://example.com/original-chart"}
    labels = [{"id": str(i), "symbol": "AAPL", "market": "us_equities", "timeframe": "1d",
               "phase": p, "as_of": d, "source_id": "human", "published_at": "2015-01-01",
               "cycle_id": "one_cycle", "independence": "external_human", "adjudication": "explicit_chart_interior",
               "eligible": True, "retrospective": True} for i, (d, p) in enumerate(zip(dates, "ABC"))]
    reference = {"sources": [source], "labels": labels, "sampling": "Test fixture; not real labels"}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(reference))
    return path, reference


def test_abstentions_count_as_errors_and_clustered_cases_have_no_population_inference(tmp_path):
    path, _ = inputs(tmp_path)
    result = evaluate_labels(tmp_path, path, tmp_path / "result")
    m = result["metrics"]["us_equities"]
    assert m["agreement_including_abstentions"] == pytest.approx(2 / 3)
    assert m["selective_agreement"] == 1
    assert m["abstentions"] == 1 and m["cycles"] == 1
    assert m["confusion"]["B"]["?"] == 1
    assert m["full_history_accuracy"] is None and m["confidence_interval"] is None
    crypto = result["metrics"]["binance_spot"]
    assert crypto["agreement_including_abstentions"] is None
    assert crypto["status"] == "unavailable_no_eligible_labels"


def test_matching_phase_with_opposite_direction_is_not_jointly_correct(tmp_path):
    path, reference = inputs(tmp_path, ("A", "B", "C"))
    for label in reference["labels"]:
        label["regime"] = "distribution"
    path.write_text(json.dumps(reference))
    csv = tmp_path / "AAPL.csv"
    table = pd.read_csv(csv, keep_default_na=False)
    table["regime"] = ["distribution", "trading range", "reaccumulation"]
    table.to_csv(csv, index=False)
    m = evaluate_labels(tmp_path, path)["metrics"]["us_equities"]
    assert m["agreement_including_abstentions"] == 1
    assert m["joint_phase_side_agreement"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("invalid", ["self_label", "interval", "wrong_market", "missing_date", "duplicate"])
def test_reference_quality_cannot_be_silently_relaxed(tmp_path, invalid):
    path, reference = inputs(tmp_path)
    if invalid == "self_label":
        reference["labels"][0]["independence"] = "same_core_rules"
    elif invalid == "interval":
        reference["labels"][0].pop("as_of")
        reference["labels"][0].update(start="2009-01-05", end="2009-01-07")
    elif invalid == "wrong_market":
        reference["sources"][0]["original_market"] = "CME futures"
    elif invalid == "missing_date":
        reference["labels"][0]["as_of"] = "2009-01-04"
    else:
        reference["labels"].append(dict(reference["labels"][0], id="duplicate"))
    path.write_text(json.dumps(reference))
    result = evaluate_labels(tmp_path, path)
    assert result["excluded"]
    assert result["metrics"]["us_equities"]["labeled_points"] == (0 if invalid == "wrong_market" else 3 if invalid == "duplicate" else 2)
