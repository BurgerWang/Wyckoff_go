"""Synthetic human identities and states below test software, never market truth."""
from copy import deepcopy
import json

import pytest

from research.tuning.label_quality import agreement_report


@pytest.fixture
def reviews():
    source = {"market": "binance_spot", "symbol": "BTCUSDT", "timeframe": "1d", "ohlcv_sha256": "fixture"}
    rubric = {"version": "phase-rubric-v1", "sha256": "synthetic-software-fixture"}
    cases = [{"id": f"case-{i}", "market": "binance_spot", "symbol": "BTCUSDT", "timeframe": "1d",
              "as_of": f"2025-01-0{i + 1}", "label_mode": "asof", "role": "development", "panel": "window",
              "block_id": "fixture-window", "information_start": "2024-12-01", "information_end": f"2025-01-0{i + 1}",
              "chart": f"charts/{i}.html"} for i in range(6)]
    manifest = {"pack_id": "fixture-pack", "source": source, "settings": {"label_mode": "asof"},
                "rubric_identity": rubric, "sampling": {"method": "synthetic-software-fixture"}, "cases": cases}
    review_a = {"pack_id": "fixture-pack", "source_identity": deepcopy(source), "label_mode": "asof",
                "rubric_identity": deepcopy(rubric), "reviewer": {
                    "name": "Fixture human A", "identity_id": "fixture-a", "reviewed_at": "2026-09-30T12:00:00+07:00",
                    "independent_human": True, "no_strategy_outputs": True, "no_future_prices_for_asof": True,
                    "blind_review": True, "no_other_review_seen": True, "independent_of_other_review": True},
                "labels": [{**row, "status": "blank", "phase": None, "regime": None} for row in deepcopy(cases)]}
    review_b = deepcopy(review_a)
    review_b["reviewer"].update(name="Fixture human B", identity_id="fixture-b")
    return review_a, review_b, manifest


def complete(review, index, phase, regime):
    review["labels"][index].update(status="completed", phase=phase, regime=regime)


def test_agreement_denominator_confusion_coverage_and_purity(reviews):
    a, b, manifest = reviews
    for review in (a, b):
        complete(review, 0, "A", "accumulation")
        complete(review, 1, "B", "accumulation")
        complete(review, 2, "C", "accumulation")
        complete(review, 3, "no_observable_phase", "undetermined")
    complete(b, 1, "B", "distribution")
    complete(b, 2, "D", "accumulation")
    complete(a, 4, "A", "undetermined")
    b["labels"][4].update(status="unresolved", phase="unknown", request_context=True)
    a["labels"][5].update(status="excluded", notes="prior exposure")
    before = deepcopy((a, b, manifest))
    report = agreement_report(a, b, manifest=manifest)
    assert (a, b, manifest) == before
    assert report["registered_cases"] == 6 and report["common_completed"] == 4
    assert report["letter"]["rate"] == .75
    assert report["direction"]["rate"] == .75
    assert report["joint"]["rate"] == .5
    assert report["joint"]["confusion"]["B/accumulation"]["B/distribution"] == 1
    assert len(report["joint"]["confusion"]) == 13
    assert [row["id"] for row in report["disagreements"]] == ["case-1", "case-2"]
    assert report["common_completed_by_role"]["development"] == 4
    assert report["coverage"]["review_a"]["direction_undetermined_completed"] == 1
    assert report["coverage"]["review_b"]["unknown_or_unresolved"] == 1
    assert report["coverage"]["review_b"]["request_context"] == 1
    assert report["coverage"]["review_a"]["no_observable_phase_completed"] == 1
    assert report["status_confusion"]["excluded"]["blank"] == 1
    assert len(report["not_common_completed"]) == 2
    json.dumps(report, allow_nan=False)


def test_zero_common_completed_and_constant_kappa_are_unavailable(reviews):
    a, b, manifest = reviews
    report = agreement_report(a, b, manifest=manifest)
    assert report["status"] == "NO_COMMON_COMPLETED"
    assert report["joint"]["rate"] is None and report["joint"]["cohens_kappa"] is None
    for review in (a, b):
        complete(review, 0, "A", "accumulation")
    report = agreement_report(a, b, manifest=manifest)
    assert report["joint"]["rate"] == 1
    assert report["joint"]["cohens_kappa"] is None
    assert report["joint"]["kappa_status"] == "undefined_constant_marginals"
    assert "cohens_kappa" not in agreement_report(a, b, manifest=manifest, include_kappa=False)["joint"]


def test_kappa_uses_both_human_marginals(reviews):
    a, b, manifest = reviews
    for index, state in enumerate(("A", "A", "B", "B")):
        complete(a, index, state, "accumulation")
    for index, state in enumerate(("A", "A", "A", "B")):
        complete(b, index, state, "accumulation")
    report = agreement_report(a, b, manifest=manifest)
    assert report["joint"]["chance_agreement"] == .5
    assert report["joint"]["cohens_kappa"] == .5


@pytest.mark.parametrize("field", ["independent_human", "no_strategy_outputs", "blind_review", "no_other_review_seen",
                                   "independent_of_other_review", "no_future_prices_for_asof"])
def test_missing_blind_human_attestations_rejected(reviews, field):
    a, b, manifest = reviews
    b["reviewer"].pop(field)
    with pytest.raises(ValueError, match=field):
        agreement_report(a, b, manifest=manifest)


@pytest.mark.parametrize("mutation", ["name", "identity", "llm", "timestamp", "timestamp_zone"])
def test_self_review_llm_and_invalid_identity_rejected(reviews, mutation):
    a, b, manifest = reviews
    if mutation == "name":
        b["reviewer"]["name"] = "  FIXTURE   HUMAN A  "
    elif mutation == "identity":
        b["reviewer"]["identity_id"] = a["reviewer"]["identity_id"]
    elif mutation == "llm":
        b["reviewer"]["type"] = "LLM"
    elif mutation == "timestamp":
        b["reviewer"]["reviewed_at"] = ""
    else:
        b["reviewer"]["reviewed_at"] = "2026-09-30T12:00:00"
    with pytest.raises(ValueError):
        agreement_report(a, b, manifest=manifest)


@pytest.mark.parametrize("mutation", ["mode", "source", "pack", "sampling", "rubric", "date", "bounds", "case_subset",
                                      "both_case_subset", "duplicate", "role", "cycle", "nan"])
def test_registration_mismatch_and_corrupt_json_rejected(reviews, mutation):
    a, b, manifest = reviews
    if mutation == "mode":
        b["label_mode"] = "retrospective"
    elif mutation == "source":
        b["source_identity"]["symbol"] = "ETHUSDT"
    elif mutation == "pack":
        b["pack_id"] = "other"
    elif mutation == "sampling":
        b["sampling"] = {"method": "changed"}
    elif mutation == "rubric":
        a["rubric_identity"]["sha256"] = b["rubric_identity"]["sha256"] = "changed"
    elif mutation == "date":
        b["labels"][0]["as_of"] = "2024-12-31"
    elif mutation == "bounds":
        for review in (a, b):
            review["labels"][0]["information_start"] = "2024-11-01"
    elif mutation == "case_subset":
        b["labels"].pop()
    elif mutation == "both_case_subset":
        a["labels"].pop()
        b["labels"].pop()
    elif mutation == "duplicate":
        b["labels"].append(deepcopy(b["labels"][0]))
    elif mutation == "role":
        b["labels"][0]["role"] = "holdout"
    elif mutation == "cycle":
        b["labels"][0]["cycle_id"] = "fabricated-cycle"
    else:
        b["labels"][0]["notes"] = float("nan")
    with pytest.raises(ValueError):
        agreement_report(a, b, manifest=manifest)


def test_explicit_sampling_comparison_without_manifest(reviews):
    a, b, manifest = reviews
    with pytest.raises(ValueError, match="sampling metadata"):
        agreement_report(a, b)
    for review in (a, b):
        review["sampling"] = deepcopy(manifest["sampling"])
    assert agreement_report(a, b)["manifest_supplied"] is False


def test_protocol_identity_is_checked_when_declared_but_not_required_on_old_reviews(reviews):
    a, b, manifest = reviews
    manifest["protocol_sha256"] = "fixture-existing-manifest-protocol"
    assert agreement_report(a, b, manifest=manifest)["common_completed"] == 0
    for review in (a, b):
        review["protocol_sha256"] = "different-protocol"
    with pytest.raises(ValueError, match="protocol_sha256"):
        agreement_report(a, b, manifest=manifest)


def test_registered_rubric_cannot_be_omitted_by_both_reviewers(reviews):
    a, b, manifest = reviews
    a.pop("rubric_identity")
    b.pop("rubric_identity")
    with pytest.raises(ValueError, match="rubric_identity"):
        agreement_report(a, b, manifest=manifest)


def test_resolved_ambiguity_requires_a_third_person_and_preserves_original_counts(reviews):
    a, b, manifest = reviews
    complete(a, 0, "C", "accumulation")
    complete(b, 0, "D", "accumulation")
    a["labels"][0].update(ambiguity=True, ambiguity_resolved=True,
                         resolver="Fixture human C", resolution_rationale="Synthetic software check")
    report = agreement_report(a, b, manifest=manifest)
    assert report["joint"]["agreements"] == 0
    assert len(report["disagreements"]) == 1


@pytest.mark.parametrize("mutation", ["unknown", "undetermined_C", "future", "context_completed", "context_exposure", "self_resolver"])
def test_completed_truth_contract_and_exposure_rejected(reviews, mutation):
    a, b, manifest = reviews
    complete(a, 0, "A", "accumulation")
    row = a["labels"][0]
    if mutation == "unknown":
        row["phase"] = "unknown"
    elif mutation == "undetermined_C":
        row.update(phase="C", regime="undetermined")
    elif mutation == "future":
        row["information_end"] = "2025-01-02"
    elif mutation == "context_completed":
        row["request_context"] = True
    elif mutation == "context_exposure":
        row.update(status="unresolved", additional_exposure=[{"information_start": "2024-11-01", "information_end": "2025-01-01"}])
    else:
        row.update(ambiguity=True, ambiguity_resolved=True, resolver=b["reviewer"]["name"], resolution_rationale="fixture")
    with pytest.raises(ValueError):
        agreement_report(a, b, manifest=manifest)


def test_retrospective_ten_classes_and_no_negative(reviews):
    a, b, manifest = reviews
    for review in (a, b):
        review["label_mode"] = "retrospective"
        review["reviewer"].pop("no_future_prices_for_asof")
        for row in review["labels"]:
            row.update(label_mode="retrospective", information_end="2025-01-07")
        complete(review, 0, "E", "markup")
    manifest["settings"]["label_mode"] = "retrospective"
    for row in manifest["cases"]:
        row.update(label_mode="retrospective", information_end="2025-01-07")
    report = agreement_report(a, b, manifest=manifest)
    assert len(report["joint"]["confusion"]) == 10
    assert report["joint"]["rate"] == 1
    complete(a, 0, "no_observable_phase", "undetermined")
    with pytest.raises(ValueError, match="only an explicit asof"):
        agreement_report(a, b, manifest=manifest)


def test_reordered_case_rows_do_not_change_report(reviews):
    a, b, manifest = reviews
    complete(a, 0, "A", "undetermined")
    complete(b, 0, "A", "undetermined")
    before = agreement_report(a, b, manifest=manifest)
    a["labels"].reverse()
    b["labels"] = b["labels"][3:] + b["labels"][:3]
    assert agreement_report(a, b, manifest=manifest) == before
