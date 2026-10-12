"""Synthetic software fixtures below are not human market annotations."""
from copy import deepcopy

import pytest

from research.tuning.consensus import VERSION, build_consensus


@pytest.fixture
def inputs():
    source = {"market": "binance_spot", "symbol": "BTCUSDT", "timeframe": "1d", "ohlcv_sha256": "software-fixture"}
    rubric = {"version": "phase-rubric-v1", "sha256": "software-fixture"}
    cases = [{"id": f"case-{i}", **{key: source[key] for key in ("market", "symbol", "timeframe")},
              "as_of": f"2025-01-0{i + 1}", "label_mode": "asof", "role": "development", "panel": "cycle",
              "cycle_id": "synthetic-contract-only", "information_start": "2024-12-01",
              "information_end": f"2025-01-0{i + 1}", "chart": f"charts/{i}.html"} for i in range(5)]
    manifest = {"pack_id": "software-fixture", "source": source, "settings": {"label_mode": "asof"},
                "rubric_identity": rubric, "sampling": {"method": "software-fixture"}, "cases": cases}
    a = {"pack_id": manifest["pack_id"], "source_identity": deepcopy(source), "label_mode": "asof",
         "rubric_identity": deepcopy(rubric), "reviewer": {
             "name": "Fixture A", "identity_id": "fixture-a", "reviewed_at": "2026-10-01T12:00:00+07:00",
             "independent_human": True, "no_strategy_outputs": True, "no_future_prices_for_asof": True,
             "blind_review": True, "no_other_review_seen": True, "independent_of_other_review": True},
         "labels": [{**deepcopy(case), "status": "blank", "phase": None, "regime": None,
                     "request_context": False, "additional_exposure": [], "notes": ""} for case in cases]}
    b = deepcopy(a)
    b["reviewer"].update(name="Fixture B", identity_id="fixture-b")
    for review in (a, b):
        review["labels"][0].update(status="completed", phase="A", regime="accumulation")
        review["labels"][1].update(status="completed", phase="B", regime="accumulation")
    b["labels"][0]["regime"] = "reaccumulation"
    b["labels"][1]["phase"] = "C"
    a["labels"][2].update(status="completed", phase="D", regime="accumulation")
    b["labels"][4].update(status="excluded", notes="fixture prior exposure")
    return manifest, a, b


def adjudication(a):
    result = deepcopy(a)
    result["reviewer"] = {"name": "Fixture C", "identity_id": "fixture-c", "reviewed_at": "2026-10-01T13:00:00+07:00",
                          "independent_human": True, "no_strategy_outputs": True, "no_future_prices_for_asof": True}
    for row in result["labels"]:
        row.update(status="blank", phase=None, regime=None)
    return result


def resolve(review, index=1, phase="C", regime="accumulation"):
    review["labels"][index].update(status="completed", phase=phase, regime=regime,
        ambiguity_resolved=True, resolver=review["reviewer"]["name"], resolution_rationale="Synthetic evidence check")


def test_joint_mapping_only_no_single_review_selection_and_preserved_coverage(inputs):
    manifest, a, b = inputs
    b["labels"][1]["notes"] = "fixture alternate interpretation"
    exposure = {"information_start": "2024-12-15", "information_end": "2024-12-20", "notes": "fixture supplemental view"}
    b["labels"][2].update(request_context=True, additional_exposure=[exposure])
    before = deepcopy(inputs)
    result = build_consensus(*inputs)
    assert inputs == before
    assert result["reviewer"] == a["reviewer"] and result["reviewer"] is not a["reviewer"]
    assert [row["status"] for row in result["labels"]] == ["completed", "unresolved", "unresolved", "blank", "excluded"]
    assert result["labels"][0] == a["labels"][0]
    assert all(row["phase"] is row["regime"] is None for row in result["labels"][1:])
    assert result["labels"][2]["request_context"] is True
    assert result["labels"][2]["additional_exposure"] == [exposure]
    metadata = result["consensus"]
    assert metadata["version"] == VERSION
    assert metadata["merged_status_counts"] == {"blank": 1, "completed": 1, "unresolved": 2, "excluded": 1}
    assert metadata["original_coverage"]["review_b"]["statuses"]["excluded"] == 1
    assert metadata["case_decisions"][1]["original_notes"]["review_b"] == "fixture alternate interpretation"
    assert metadata["original_common_completed"] == 2
    assert metadata["reviewers"]["adjudicator"] is None


def test_explicit_third_human_resolves_disagreement_without_claiming_blindness(inputs):
    manifest, a, b = inputs
    judge = adjudication(a)
    resolve(judge, phase="E", regime="distribution")
    judge["labels"][2].update(status="unresolved", notes="fixture insufficient evidence")
    before = deepcopy((manifest, a, b, judge))
    result = build_consensus(manifest, a, b, judge)
    assert (manifest, a, b, judge) == before
    assert result["reviewer"] == a["reviewer"]
    row = result["labels"][1]
    assert (row["status"], row["phase"], row["regime"]) == ("completed", "E", "distribution")
    assert row["ambiguity_resolved"] is True and row["resolver"] == "Fixture C"
    assert row["resolution_rationale"] == "Synthetic evidence check"
    assert row["ambiguity"] == "independent_review_adjudication"
    assert result["labels"][0] == a["labels"][0]  # blank adjudication does not override agreement
    assert result["labels"][2]["status"] == "unresolved"
    assert result["consensus"]["reviewers"]["adjudicator"] == judge["reviewer"]
    assert "blind_review" not in judge["reviewer"]
    assert len(result["labels"]) == len(manifest["cases"])


def test_adjudication_cannot_precede_original_review(inputs):
    manifest, a, b = inputs
    judge = adjudication(a)
    judge["reviewer"]["reviewed_at"] = "2026-10-01T11:59:59+07:00"
    with pytest.raises(ValueError, match="cannot precede"):
        build_consensus(manifest, a, b, judge)


@pytest.mark.parametrize("index", [1, 2])
def test_adjudicator_can_accept_an_existing_answer(inputs, index):
    manifest, a, b = inputs
    judge = adjudication(a)
    resolve(judge, index=index, phase=a["labels"][index]["phase"])
    result = build_consensus(manifest, a, b, judge)
    assert result["labels"][index]["status"] == "completed"
    assert result["labels"][index]["phase"] == a["labels"][index]["phase"]


@pytest.mark.parametrize("mutation", ["same_name", "same_id", "nonhuman", "strategy", "future_attestation", "timezone", "missing_id"])
def test_adjudicator_identity_and_attestations_rejected(inputs, mutation):
    manifest, a, b = inputs
    judge = adjudication(a)
    person = judge["reviewer"]
    if mutation == "same_name":
        person["name"] = "  FIXTURE   A  "
    elif mutation == "same_id":
        person["identity_id"] = "  FIXTURE-B  "  # different name does not establish independence
    elif mutation == "nonhuman":
        person["type"] = "LLM"
    elif mutation == "strategy":
        person["no_strategy_outputs"] = False
    elif mutation == "future_attestation":
        person["no_future_prices_for_asof"] = False
    elif mutation == "timezone":
        person["reviewed_at"] = "2026-10-01T12:00:00"
    else:
        person["identity_id"] = ""
    with pytest.raises(ValueError):
        build_consensus(manifest, a, b, judge)


@pytest.mark.parametrize("mutation", ["same_author", "same_id", "partial_a", "partial_b", "nonblind", "case_bounds", "rubric", "future_exposure", "derived_review"])
def test_original_blind_reviews_and_complete_registration_are_required(inputs, mutation):
    manifest, a, b = inputs
    if mutation == "same_author":
        b["reviewer"]["name"] = "fixture a"
    elif mutation == "same_id":
        b["reviewer"]["identity_id"] = a["reviewer"]["identity_id"]
    elif mutation.startswith("partial_"):
        (a if mutation == "partial_a" else b)["labels"].pop()
    elif mutation == "nonblind":
        b["reviewer"]["blind_review"] = False
    elif mutation == "case_bounds":
        b["labels"][3]["information_start"] = "2024-12-10"
    elif mutation == "rubric":
        b["rubric_identity"]["sha256"] = "changed"
    elif mutation == "derived_review":
        a["consensus"] = {}
    else:
        b["labels"][3]["additional_exposure"] = [{"information_start": "2025-01-01", "information_end": "2025-01-05"}]
    with pytest.raises(ValueError):
        build_consensus(manifest, a, b)


@pytest.mark.parametrize("mutation", ["pack", "source", "mode", "rubric", "partial", "short_bounds", "future_bounds", "future_exposure",
                                      "missing_resolution", "wrong_resolver", "missing_rationale", "invalid_class", "unreviewed", "context"])
def test_adjudication_registration_and_completion_fail_closed(inputs, mutation):
    manifest, a, b = inputs
    judge = adjudication(a)
    resolve(judge)
    row = judge["labels"][1]
    if mutation == "pack":
        judge["pack_id"] = "other"
    elif mutation == "source":
        judge["source_identity"]["symbol"] = "ETHUSDT"
    elif mutation == "mode":
        judge["label_mode"] = "retrospective"
    elif mutation == "rubric":
        judge["rubric_identity"]["sha256"] = "other"
    elif mutation == "partial":
        judge["labels"].pop()
    elif mutation == "short_bounds":
        row["information_start"] = "2024-12-20"
    elif mutation == "future_bounds":
        row["information_end"] = "2025-01-03"
    elif mutation == "future_exposure":
        row["additional_exposure"] = [{"information_start": "2025-01-01", "information_end": "2025-01-03"}]
    elif mutation == "missing_resolution":
        row.pop("ambiguity_resolved")
    elif mutation == "wrong_resolver":
        row["resolver"] = a["reviewer"]["name"]
    elif mutation == "missing_rationale":
        row["resolution_rationale"] = " "
    elif mutation == "invalid_class":
        row["regime"] = "undetermined"
    elif mutation == "unreviewed":
        resolve(judge, index=3)
    else:
        row["request_context"] = True
    with pytest.raises(ValueError):
        build_consensus(manifest, a, b, judge)


def test_adjudicator_cannot_clear_an_original_context_request(inputs):
    manifest, a, b = inputs
    b["labels"][2]["request_context"] = True
    judge = adjudication(a)
    resolve(judge, index=2)
    with pytest.raises(ValueError, match="request_context"):
        build_consensus(manifest, a, b, judge)


def test_actual_exposure_union_keeps_both_humans_and_adjudicator(inputs):
    manifest, a, b = inputs
    judge = adjudication(a)
    resolve(judge)
    exposures = [{"information_start": f"2024-12-{day}", "information_end": f"2024-12-{day}"} for day in (10, 11, 12)]
    a["labels"][1]["additional_exposure"] = exposures[:1]
    b["labels"][1]["additional_exposure"] = exposures[:2]
    judge["labels"][1]["additional_exposure"] = exposures[2:]
    result = build_consensus(manifest, a, b, judge)
    assert result["labels"][1]["additional_exposure"] == exposures
    result["labels"][1]["additional_exposure"][0]["information_start"] = "changed-output"
    assert a["labels"][1]["additional_exposure"][0]["information_start"] == "2024-12-10"


def test_adjudicator_can_explicitly_exclude_a_previous_agreement(inputs):
    manifest, a, b = inputs
    judge = adjudication(a)
    judge["labels"][0].update(status="excluded", phase="D", regime="distribution", notes="fixture scale mismatch")
    result = build_consensus(manifest, a, b, judge)
    assert result["labels"][0]["status"] == "excluded"
    assert result["labels"][0]["phase"] is result["labels"][0]["regime"] is None
    assert result["consensus"]["original_common_completed"] == 2


def test_retrospective_adjudication_does_not_require_asof_attestation(inputs):
    manifest, a, b = inputs
    manifest["settings"]["label_mode"] = "retrospective"
    for case in manifest["cases"]:
        case.update(label_mode="retrospective", information_end="2025-01-06")
    for review in (a, b):
        review["label_mode"] = "retrospective"
        review["reviewer"].pop("no_future_prices_for_asof")
        for row in review["labels"]:
            row.update(label_mode="retrospective", information_end="2025-01-06")
    judge = adjudication(a)
    judge["reviewer"].pop("no_future_prices_for_asof")
    resolve(judge)
    result = build_consensus(manifest, a, b, judge)
    assert result["labels"][1]["status"] == "completed"


@pytest.mark.parametrize("manifest,a,b,judge", [(None, {}, {}, None), ({}, None, {}, None), ({}, {}, None, None)])
def test_nonobjects_fail_with_value_error(manifest, a, b, judge):
    with pytest.raises(ValueError):
        build_consensus(manifest, a, b, judge)
