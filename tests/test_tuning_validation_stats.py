"""Synthetic software checks only; no fixture is independent market truth."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
import random

import pytest

from research.tuning.labels import joint_classes
from research.tuning.metrics import ABSTENTION, score
from research.tuning.preflight import evaluate_reference_capacity
from research.tuning.space import validate_candidate
from research.tuning.validation_stats import (
    _group_confusions, _overlap_clusters, _sample_score, legal_neighbors,
    paired_cluster_bootstrap, score_summaries, summarize_stability,
)
from wyckoff_core.models import CoreConfig


def fixture(groups, *, mode="retrospective"):
    """Group entries: (panel, id, [(expected, default, candidate), ...])."""
    labels, defaults, candidates = [], [], []
    cursor = date(2010, 1, 1)
    for panel, ident, cases in groups:
        start = cursor.isoformat()
        end = (cursor + timedelta(days=len(cases) - 1)).isoformat()
        for expected, default, candidate in cases:
            day = cursor.isoformat()
            phase, regime = expected.split("/") if "/" in expected else (expected, "none")
            labels.append({"as_of": day, "phase": phase, "regime": regime, "panel": panel,
                           "cycle_id" if panel == "cycle" else "block_id": ident,
                           "information_start": start, "information_end": end, "role": "holdout"})
            for output, predicted in ((defaults, default), (candidates, candidate)):
                if predicted == ABSTENTION:
                    row = {"status": "missing"}
                else:
                    p, r = predicted.split("/") if "/" in predicted else (predicted, "none")
                    row = {"phase": p, "regime": r, "status": "normal"}
                output.append({"as_of": day, **row})
            cursor += timedelta(days=1)
        cursor += timedelta(days=2)
    reference = {"label_mode": mode, "labels": labels}
    return reference, score(reference, defaults), score(reference, candidates)


def full_fixture(count=12, mode="retrospective"):
    states = joint_classes(mode)
    return fixture([(panel, f"{panel}{i}", [(state, ABSTENTION, state) for state in states])
                    for panel in ("cycle", "window") for i in range(count)], mode=mode)


def test_equal_cluster_mass_unequal_point_counts_and_duplicate_draws():
    a, b = "A/accumulation", "B/accumulation"
    _, default, candidate = fixture([
        ("cycle", "small", [(a, a, a)]),
        ("cycle", "large", [(b, b, ABSTENTION)] * 9),
        ("window", "window", [(a, a, a)] * 3),
    ])
    assert candidate["joint_agreement"] == pytest.approx(.75)
    assert score_summaries(candidate)["phase_letter_agreement"] == pytest.approx(.75)
    groups = {}
    for case in candidate["cases"]:
        groups.setdefault((case["panel"], case["group_id"]), []).append(case)
    tables = _group_confusions(groups, candidate["classes"])
    # Sorted cycle tables: large index 0, small index 1. Drawing small twice
    # gives its panel total 0.5; it is not collapsed back into one unique group.
    actual = _sample_score(tables, {"cycle": [1, 1], "window": [0]}, 10)
    assert actual == pytest.approx({"joint_macro_f1": .1, "joint_agreement": 1.0})
    actual = _sample_score(tables, {"cycle": [0, 0], "window": [0]}, 10)
    assert actual == pytest.approx({"joint_macro_f1": .1, "joint_agreement": .5})
    assert default["joint_macro_f1"] == pytest.approx(.2)


@pytest.mark.parametrize("mode,count", [("retrospective", 10), ("asof", 13)])
def test_bootstrap_retains_fixed_missing_classes_and_sparse_scope(mode, count):
    a = "A/accumulation"
    reference, default, candidate = fixture([
        (panel, str(i), [(a, ABSTENTION, a)]) for panel in ("cycle", "window") for i in range(4)
    ], mode=mode)
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=20)
    assert len(result["classes"]) == count
    assert len(result["missing_classes"]) == count - 1
    assert result["observed_differences"]["joint_macro_f1"] == pytest.approx(1 / count)
    assert result["status"] == "INSUFFICIENT_TEST_POWER"
    assert result["intervals"]["joint_macro_f1"]["lower"] == pytest.approx(1 / count)
    assert not result["intervals"]["joint_macro_f1"]["inferential"]


def test_full_support_power_eligibility_and_reproducibility():
    reference, default, _ = full_fixture()
    rows = [{"as_of": label["as_of"], "phase": label["phase"], "regime": label["regime"],
             "status": "missing" if i % 7 == 0 else "normal"} for i, label in enumerate(reference["labels"])]
    candidate = score(reference, rows)
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=25, seed=123)
    assert result == paired_cluster_bootstrap(default, candidate, reference=reference, repeats=25, seed=123)
    assert result["inference_eligible"]
    assert result["effective_clusters"] == {"cycle": 12, "window": 12}
    assert result["observed_differences"]["joint_macro_f1"] > .5
    assert result["observed_differences"]["joint_agreement"] > .5
    assert result["intervals"]["joint_agreement"]["inferential"]


def test_degenerate_bootstrap_is_not_confidence_from_zero_empirical_spread():
    reference, default, candidate = full_fixture()
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=25)
    assert result["unit_count_eligible"]
    assert "degenerate_resampling_distribution" in result["diagnostic_reasons"]
    assert not result["inference_eligible"]
    assert result["resampling_distribution"]["joint_agreement"]["degenerate"]
    assert result["resampling_distribution"]["joint_agreement"]["std"] == 0
    assert not result["intervals"]["joint_agreement"]["inferential"]


def test_cross_panel_overlap_blocks_independent_panel_inference():
    states = joint_classes("retrospective")
    reference, default, candidate = fixture([
        (panel, f"{panel}{i}", [(state, ABSTENTION, state) for state in states])
        for i in range(12) for panel in ("cycle", "window")
    ])
    for label in reference["labels"]:
        if label["panel"] == "window":
            cycle_id = label["block_id"].replace("window", "cycle")
            cycle_end = next(other["information_end"] for other in reference["labels"]
                             if other.get("cycle_id") == cycle_id)
            label["information_start"] = cycle_end
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=5)
    assert result["effective_clusters"] == {"cycle": 12, "window": 12}
    assert result["cross_panel_information_overlap"]
    assert "cross_panel_information_overlap" in result["diagnostic_reasons"]
    assert not result["inference_eligible"]
    assert result["status"] == "INSUFFICIENT_TEST_POWER"


def test_prediction_free_capacity_and_bootstrap_share_dependency_geometry():
    reference, default, candidate = full_fixture()
    for label in reference["labels"]:
        if label.get("block_id") in ("window1", "window2"):
            label["information_start"] = next(item["information_start"] for item in reference["labels"]
                                               if item.get("block_id") == "window0")
    capacity = evaluate_reference_capacity(reference)
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=3)
    assert capacity["effective_clusters"] == result["effective_clusters"] == {"cycle": 12, "window": 10}
    assert capacity["overlap_clusters"] == result["overlap_clusters"]
    assert capacity["cross_panel_information_overlap"] == result["cross_panel_information_overlap"]
    assert set(capacity["diagnostic_reasons"]) <= set(result["diagnostic_reasons"])
    assert not capacity["eligible"] and not result["inference_eligible"]


def test_missing_reference_or_information_interval_prohibits_inference():
    reference, default, candidate = full_fixture()
    for source in (None, {**reference, "labels": [{key: value for key, value in label.items()
                                                  if key != "information_start"} for label in reference["labels"]]}):
        result = paired_cluster_bootstrap(default, candidate, reference=source, repeats=3)
        assert result["status"] == "INSUFFICIENT_TEST_POWER"
        assert not result["inference_eligible"]
        assert not result["reference_intervals_complete"]


def test_overlapping_context_transitively_merges_adjacent_windows():
    reference, default, candidate = full_fixture()
    windows = [label for label in reference["labels"] if label["panel"] == "window"]
    for label in windows:
        label["information_start"] = "2010-01-01"
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=3)
    assert result["raw_groups"] == {"cycle": 12, "window": 12}
    assert result["effective_clusters"] == {"cycle": 12, "window": 1}
    assert len(result["overlap_clusters"]["window"][0]) == 12
    assert result["status"] == "INSUFFICIENT_TEST_POWER"


def test_unequal_merged_clusters_preserve_original_group_weights():
    a, b = "A/accumulation", "B/accumulation"
    reference, _, candidate = fixture([
        ("cycle", "c1", [(a, a, a)]),
        ("cycle", "c2", [(a, a, a)] * 7),
        ("cycle", "c3", [(b, b, ABSTENTION)]),
        ("window", "w", [(a, a, a)]),
    ])
    # c1/c2 form one overlap cluster, c3 remains separate. Weighting the merged
    # clusters equally would wrongly give agreement .75 rather than 5/6.
    for label in reference["labels"]:
        if label.get("cycle_id") == "c2":
            label["information_start"] = reference["labels"][0]["as_of"]
    groups = {}
    for case in candidate["cases"]:
        groups.setdefault((case["panel"], case["group_id"]), []).append(case)
    tables, membership, complete = _overlap_clusters(groups, candidate["classes"], reference)
    assert complete and membership["cycle"] == [["c1", "c2"], ["c3"]]
    observed = _sample_score(tables, {"cycle": [0, 1], "window": [0]}, 10)
    assert observed["joint_agreement"] == pytest.approx(5 / 6)
    assert observed["joint_macro_f1"] == pytest.approx(candidate["joint_macro_f1"])
    repeated = _sample_score(tables, {"cycle": [0, 0], "window": [0]}, 10)
    assert repeated["joint_agreement"] == pytest.approx(1)


@pytest.mark.parametrize("field,value", [("as_of", "2020-01-01"), ("expected", "C/distribution"),
                                        ("panel", "window"), ("group_id", "different")])
def test_pair_rejects_misaligned_dates_references_panels_and_groups(field, value):
    _, default, candidate = full_fixture(count=1)
    candidate = deepcopy(candidate)
    candidate["cases"][0][field] = value
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(default, candidate, repeats=3)


def test_pair_accepts_reordered_cases_but_rejects_removed_fixed_class_or_bad_weight():
    reference, default, candidate = full_fixture(count=1)
    candidate["cases"].reverse()
    paired_cluster_bootstrap(default, candidate, reference=reference, repeats=3)
    candidate["classes"].pop()
    with pytest.raises(ValueError, match="fixed"):
        paired_cluster_bootstrap(default, candidate, repeats=3)
    _, default, candidate = full_fixture(count=1)
    candidate["cases"][0]["weight"] = 1
    with pytest.raises(ValueError, match="weights"):
        paired_cluster_bootstrap(default, candidate, repeats=3)


def test_missing_panel_does_not_redistribute_mass_or_emit_interval():
    a = "A/accumulation"
    reference, default, candidate = fixture([("cycle", "c", [(a, ABSTENTION, a)])])
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=3)
    assert result["intervals"] == {"joint_macro_f1": None, "joint_agreement": None}
    assert result["independent_units"] == {"cycle": 1, "window": 0}
    assert default["represented_panel_weight"] == .5


def test_weighted_phase_direction_summary_keeps_abstention_denominator():
    a, opposite = "C/accumulation", "C/distribution"
    _, _, metric = fixture([("cycle", "c", [(a, a, opposite)]),
                             ("window", "w", [(a, a, ABSTENTION)])])
    summary = score_summaries(metric)
    assert summary["phase_letter_agreement"] == .5
    assert summary["direction_error_rate"] == .5
    assert summary["direction_unknown_or_abstention_rate"] == .5
    assert summary["direction_denominator_weight"] == 1


@pytest.mark.parametrize("market", ["us_equities", "binance_spot"])
def test_neighbors_exact_adjacent_all_dimensions_legal_and_no_mutation(market):
    original = asdict(CoreConfig())
    neighbors = legal_neighbors(original, market)
    assert original == asdict(CoreConfig())
    assert {neighbor["parameter"] for neighbor in neighbors} == set(original)
    for neighbor in neighbors:
        params = neighbor["config"]
        validate_candidate(params, market)
        changed = [name for name in params if params[name] != original[name]]
        assert changed == [neighbor["parameter"]]
    assert len({tuple(sorted(neighbor["config"].items())) for neighbor in neighbors}) == len(neighbors)


def test_neighbors_exclude_invalid_constraint_and_do_not_round_offgrid():
    config = {**asdict(CoreConfig()), "swing_confirmation": 4, "min_range_bars": 10,
              "volume_high": 1.2, "preliminary_volume_min": 1.2}
    neighbors = legal_neighbors(config, "us_equities")
    assert not any(neighbor["parameter"] == "min_range_bars" and neighbor["direction"] == -1 for neighbor in neighbors)
    assert not any(neighbor["parameter"] == "preliminary_volume_min" and neighbor["direction"] == 1 for neighbor in neighbors)
    with pytest.raises(ValueError, match="outside"):
        legal_neighbors({**config, "volume_high": 1.234}, "us_equities")


def test_stability_90_percent_threshold_and_no_best_neighbor_selection():
    reference, default, candidate = full_fixture(count=1)
    # The candidate is perfect. Abstaining all A/B/D hurts macro F1 by .6,
    # while supported C/E recall remains identical to the original default.
    rows = [{"as_of": label["as_of"], "phase": label["phase"], "regime": label["regime"],
             "status": "normal" if label["phase"] in "CE" else "missing"} for label in reference["labels"]]
    bad = score(reference, rows)
    assert summarize_stability(default, candidate, [candidate] * 9 + [bad])["passed"]
    result = summarize_stability(default, candidate, [candidate] * 8 + [bad] * 2)
    assert not result["passed"] and result["stable_fraction"] == .8
    assert not summarize_stability(default, candidate, [])["passed"]


def test_stability_ce_and_fold_checks_use_original_default_not_candidate():
    reference, _, perfect = full_fixture(count=1)
    rows = [{"as_of": label["as_of"], "phase": label["phase"], "regime": label["regime"],
             "status": "missing" if label["phase"] == "C" else "normal"} for label in reference["labels"]]
    bad_ce = score(reference, rows)
    result = summarize_stability(perfect, perfect, [perfect], folds=[
        {"default_metric": perfect, "candidate_metric": bad_ce, "neighbor_metrics": [perfect]}])
    assert not result["passed"]
    assert result["folds"][0]["checks"][0]["ce_recall_regressions"]
    result = summarize_stability(perfect, perfect, [bad_ce])
    assert not result["ce_recall_passed"]


def test_varying_draws_reproducible_and_panel_stratified():
    a, b = "A/accumulation", "B/accumulation"
    reference, default, candidate = fixture([
        ("cycle", "c1", [(a, a, ABSTENTION)]), ("cycle", "c2", [(b, ABSTENTION, b)] * 4),
        ("window", "w1", [(a, a, a)]), ("window", "w2", [(a, ABSTENTION, a)] * 2),
    ])
    result = paired_cluster_bootstrap(default, candidate, reference=reference, repeats=20, seed=9)
    assert result == paired_cluster_bootstrap(default, candidate, reference=reference, repeats=20, seed=9)
    # Agreement delta = .25*(-#c1 + #c2 + #w2). Compute all exact paired
    # panel draws using the public PRNG rule; cluster point sizes cancel.
    rng = random.Random(9)
    differences = []
    for _ in range(20):
        cycle = [rng.randrange(2) for _ in range(2)]
        window = [rng.randrange(2) for _ in range(2)]
        differences.append(.25 * (cycle.count(1) - cycle.count(0) + window.count(1)))
    ordered = sorted(differences)
    q = .025 * 19
    lower = ordered[int(q)] + (ordered[int(q) + 1] - ordered[int(q)]) * (q - int(q))
    assert result["intervals"]["joint_agreement"]["lower"] == pytest.approx(lower)
