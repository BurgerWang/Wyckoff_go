"""Pure date-metadata fixtures, never human reference truth or accuracy scores."""
import copy
from datetime import date, timedelta
from itertools import combinations

import pytest

from research.tuning.sampling import sample_sparse_windows


def history(count=120):
    dates = [(date(2020, 1, 1) + timedelta(days=index)).isoformat() for index in range(count)]
    return dates, [{"start": dates[0], "end": dates[-1], "bars": count}]


def settings(**overrides):
    return {"label_mode": "asof", "seed": 20260930, "context_bars": 10, "block_bars": 5,
            "dates_per_block": 3, "development_blocks": 3, "holdout_blocks": 3,
            "development_start": "2020-01-01", "development_end": "2020-03-01",
            "holdout_start": "2020-03-01", "holdout_end": "2020-05-01", **overrides}


def sample(**kwargs):
    dates, segments = history()
    return sample_sparse_windows(dates, segments, settings(), **kwargs)


def disjoint(blocks):
    for left, right in combinations(blocks, 2):
        assert left["information_end"] < right["information_start"] or right["information_end"] < left["information_start"]


def test_exact_context_and_full_exposure_with_two_frozen_partitions():
    dates, segments = history()
    registered = settings()
    result = sample_sparse_windows(dates, segments, registered)
    assert result["status"] == "COMPLETE"
    assert result["version"] == "sparse-window-v2"
    assert result["identity"]["settings"]["development_start"] == registered["development_start"]
    assert result["identity"]["settings"]["holdout_end"] == registered["holdout_end"]
    disjoint(result["blocks"])
    for block in result["blocks"]:
        assert block["exposure_bars"] == 14  # 10-bar contexts plus the 5-target-day span.
        assert block["information_start"] == block["exposure_start"]
        assert registered[f"{block['role']}_start"] <= block["information_start"]
        assert block["information_end"] < registered[f"{block['role']}_end"]
        assert block["dates"] == sorted(set(block["dates"]))
        assert "cycle_id" not in block and block["panel"] == "window"
        for case in block["cases"]:
            assert dates.index(case["information_end"]) - dates.index(case["information_start"]) + 1 == 10
            assert case["information_end"] == case["as_of"]
    assert result["promotion_eligible"] is False
    assert result["class_support_known"] is False
    assert result["cycle_support_qualified"] is False


def test_future_metadata_cannot_change_earlier_asof_selection():
    dates, segments = history(122)  # The complete frozen partition is present.
    before = sample_sparse_windows(dates, segments, settings())
    future_dates, future_segments = history(400)
    after = sample_sparse_windows(future_dates, future_segments, settings())
    assert before["blocks"] == after["blocks"]
    assert before["support"] == after["support"]
    assert before["sampling_id"] != after["sampling_id"]  # The true source metadata did change.
    assert max(case["information_end"] for block in before["blocks"] for case in block["cases"]) < "2020-05-01"


def test_retrospective_cases_show_exact_declared_full_block_context():
    dates, segments = history()
    result = sample_sparse_windows(dates, segments, settings(label_mode="retrospective"))
    for block in result["blocks"]:
        for case in block["cases"]:
            assert case["information_start"] == block["exposure_start"]
            assert case["information_end"] == block["end"]
            assert case["as_of"] <= case["information_end"]


def test_segment_breaks_are_not_bridged_or_context_shortened():
    dates, _ = history()
    segments = [{"start": dates[0], "end": dates[7], "bars": 8},
                {"start": dates[8], "end": dates[66], "bars": 59},
                {"start": dates[67], "end": dates[-1], "bars": 53}]
    result = sample_sparse_windows(dates, segments, settings(development_blocks=20, holdout_blocks=20))
    for block in result["blocks"]:
        assert any(segment["start"] <= block["information_start"] <= block["information_end"] <= segment["end"] for segment in segments)
        assert block["exposure_bars"] == 14
    assert sum(role["rejection_counts"].get("target_crosses_continuous_segment", 0) for role in result["support"].values()) > 0
    assert result["support"]["development"]["rejection_counts"]["insufficient_context_or_segment_warmup"] > 0


def test_all_roles_exclude_complete_diagnostic_consumed_and_human_cycle_bounds():
    forbidden = [("2020-01-20", "2020-01-25"), ("2020-03-20", "2020-03-25")]
    consumed = [("2020-02-05", "2020-02-10"), ("2020-04-05", "2020-04-10")]
    cycles = [{"cycle_id": "human-dev", "role": "development", "information_start": "2020-02-20", "information_end": "2020-02-25"},
              {"cycle_id": "human-holdout", "role": "holdout", "information_start": "2020-04-20", "information_end": "2020-04-25"}]
    result = sample(diagnostic_intervals=forbidden, consumed_intervals=consumed, human_cycle_bounds=cycles)
    bounds = forbidden + consumed + [(cycle["information_start"], cycle["information_end"]) for cycle in cycles]
    for block in result["blocks"]:
        assert all(block["information_end"] < start or end < block["information_start"] for start, end in bounds)
    for support in result["support"].values():
        assert support["rejection_counts"]["known_diagnostic_information_overlap"] > 0
        assert support["rejection_counts"]["consumed_holdout_information_overlap"] > 0
        assert support["rejection_counts"]["human_cycle_information_overlap"] > 0
    assert result["identity"]["exclusions"]["human_cycle_information_overlap"] == cycles


def test_inclusive_exclusion_endpoints_block_context_even_when_target_is_clean():
    dates, segments = history(14)
    result = sample_sparse_windows(dates, segments, settings(development_blocks=1, holdout_blocks=0),
                                   diagnostic_intervals=[(dates[0], dates[0])])
    assert result["support"]["development"]["available_blocks"] == 0
    assert result["support"]["development"]["rejection_counts"]["known_diagnostic_information_overlap"] == 1
    assert result["status"] == "INSUFFICIENT"


def test_capacity_counts_fragmented_clean_runs_instead_of_total_clean_bars():
    dates, segments = history(60)
    # Four clean 13-bar runs cannot contain a complete 14-bar exposure.
    holes = [(dates[index], dates[index]) for index in (13, 27, 41, 55)]
    result = sample_sparse_windows(dates, segments, settings(development_blocks=1, holdout_blocks=0),
                                   consumed_intervals=holes)
    assert result["support"]["development"]["max_disjoint_capacity"] == 0
    assert result["status"] == "INSUFFICIENT"
    # Exact capacity in a 60-bar unbroken partition is floor(60 / 14) == 4.
    full = sample_sparse_windows(dates, segments, settings(development_blocks=20, holdout_blocks=0))
    assert full["support"]["development"]["max_disjoint_capacity"] == 4
    assert full["support"]["development"]["selected_blocks"] == 4
    assert full["support"]["development"]["status"] == "PARTIAL"
    assert full["support"]["development"]["reasons"] == ["fewer_than_requested_disjoint_windows"]
    assert full["status"] == "PARTIAL"


def test_capacity_exact_for_each_small_fragment_layout():
    dates, segments = history(12)
    registered = settings(context_bars=2, block_bars=2, dates_per_block=2,
                          development_blocks=20, holdout_blocks=0)
    for holes in ((), (3,), (1, 6), (2, 5, 8)):
        candidates = [(index, index + 2) for index in range(10)
                      if not any(index <= hole <= index + 2 for hole in holes)]
        maximum = max((len(subset) for size in range(len(candidates) + 1)
                       for subset in combinations(candidates, size)
                       if all(a[1] < b[0] or b[1] < a[0] for a, b in combinations(subset, 2))), default=0)
        result = sample_sparse_windows(dates, segments, registered,
                                       diagnostic_intervals=[(dates[index], dates[index]) for index in holes])
        assert result["support"]["development"]["max_disjoint_capacity"] == maximum
        disjoint(result["blocks"])


def test_registered_seed_is_deterministic_without_scores_and_never_reduces_capacity():
    dates, segments = history()
    first = sample_sparse_windows(dates, segments, settings())
    assert first == sample_sparse_windows(dates, segments, copy.deepcopy(settings()))
    manifests = [sample_sparse_windows(dates, segments, settings(seed=seed)) for seed in range(6)]
    assert len({result["sampling_id"] for result in manifests}) == 6
    assert len({tuple(block["start"] for block in result["blocks"]) for result in manifests}) > 1
    assert all(result["support"]["development"]["max_disjoint_capacity"] == 4 for result in manifests)
    with pytest.raises(ValueError, match="exact registered metadata"):
        sample_sparse_windows(dates, segments, settings(strategy_score=99))


def test_requested_counts_preserve_registered_schedule_prefix():
    dates, segments = history()
    small = sample_sparse_windows(dates, segments, settings(development_blocks=1, holdout_blocks=1))
    large = sample_sparse_windows(dates, segments, settings(development_blocks=4, holdout_blocks=4))
    for role in ("development", "holdout"):
        left = [block for block in small["blocks"] if block["role"] == role]
        right = [block for block in large["blocks"] if block["role"] == role]
        assert left == right[:len(left)]


@pytest.mark.parametrize("change, message", [
    ({"information_start": "2019-12-31"}, "actual ordered source dates"),
    ({"information_end": "2020-03-01"}, "registered partition"),
    ({"information_start": "2020-02-10", "information_end": "2020-02-01"}, "actual ordered source dates"),
    ({"role": "diagnostic"}, "registered role"),
    ({"information_end": None}, "ISO date"),
])
def test_human_cycle_wrong_bounds_are_rejected_without_clipping(change, message):
    bound = {"cycle_id": "external-cycle", "role": "development",
             "information_start": "2020-01-10", "information_end": "2020-02-10", **change}
    with pytest.raises(ValueError, match=message):
        sample(human_cycle_bounds=[bound])


def test_human_cycle_cannot_bridge_segments_and_missing_bounds_fail():
    dates, _ = history()
    segments = [{"start": dates[0], "end": dates[19]}, {"start": dates[20], "end": dates[-1]}]
    bound = {"cycle_id": "external", "role": "development", "information_start": dates[15], "information_end": dates[25]}
    with pytest.raises(ValueError, match="continuous segment"):
        sample_sparse_windows(dates, segments, settings(), human_cycle_bounds=[bound])
    del bound["information_end"]
    with pytest.raises(ValueError, match="complete information bounds"):
        sample(human_cycle_bounds=[bound])


@pytest.mark.parametrize("bound", [None, 1, "bad", []])
def test_malformed_human_cycle_element_is_a_controlled_validation_error(bound):
    with pytest.raises(ValueError, match="complete metadata object"):
        sample(human_cycle_bounds=[bound])


@pytest.mark.parametrize("bounds", [None, 1, "bad", {}])
def test_malformed_human_cycle_collection_is_a_controlled_validation_error(bounds):
    with pytest.raises(ValueError, match="must be a list"):
        sample(human_cycle_bounds=bounds)


def test_short_history_is_not_padded_and_one_target_reports_only_real_exposure():
    dates, segments = history(9)
    result = sample_sparse_windows(dates, segments, settings())
    assert result["blocks"] == [] and result["status"] == "INSUFFICIENT"
    dates, segments = history(30)
    one = sample_sparse_windows(dates, segments, settings(dates_per_block=1, development_blocks=20, holdout_blocks=0))
    assert one["support"]["development"]["max_disjoint_capacity"] == 3
    assert all(block["exposure_bars"] == 10 and len(block["dates"]) == 1 for block in one["blocks"])


def test_registered_chronological_order_and_input_safety():
    dates, segments = history()
    result = sample_sparse_windows(dates, segments, settings(order="chronological"))
    assert [block["start"] for block in result["blocks"]] == sorted(block["start"] for block in result["blocks"])
    with pytest.raises(ValueError, match="overlap"):
        sample_sparse_windows(dates, segments, settings(development_end="2020-03-02"))
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        sample_sparse_windows([dates[0], dates[0]], segments, settings())
    with pytest.raises(ValueError, match="cover all source dates"):
        sample_sparse_windows(dates, [{"start": dates[1], "end": dates[-1]}], settings())
    with pytest.raises(ValueError, match="integer"):
        sample_sparse_windows(dates, segments, settings(seed=True))
    with pytest.raises(ValueError, match="version"):
        sample_sparse_windows(dates, segments, settings(version="labeled-holdout-v1"))


def test_empty_source_has_zero_capacity_and_frozen_exclusions_change_identity():
    empty = sample_sparse_windows([], [], settings())
    assert empty["status"] == "INSUFFICIENT" and empty["blocks"] == []
    assert all(role["max_disjoint_capacity"] == 0 for role in empty["support"].values())
    first = sample()
    changed = sample(diagnostic_intervals=[("2030-01-01", "2030-01-02")])
    assert first["blocks"] == changed["blocks"]
    assert first["sampling_id"] != changed["sampling_id"]
