"""Engineering geometry and entrypoints only; no human truth or real holdout."""
from copy import deepcopy
from datetime import date, timedelta
import json

import pytest

from research.tuning import annotation, cli, holdout, preflight
from research.tuning.cycles import VERSION, prepare_cycle_annotation_pack
from test_cycle_annotation import registration
from test_tuning_annotation import history  # noqa: F401


def manifests():
    source = {"market": "us_equities", "symbol": "FIXTURE", "timeframe": "1d",
              "ohlcv_sha256": "fixture-source", "attrs": {"price_currency": "USD"}}
    settings = {"label_mode": "asof", "development_start": "2019-01-01",
                "holdout_start": "2020-01-01", "holdout_end": "2026-09-30"}
    packs = []
    for p, panel in enumerate(("cycle", "window")):
        pack = {"pack_id": f"fixture-{panel}", "source": deepcopy(source), "settings": dict(settings),
                "protocol_sha256": "fixture-protocol", "rubric_identity": {"sha256": "fixture-rubric"},
                "sampling": {"panel": panel}, "cases": [], "diagnostic_intervals": []}
        cycles = []
        for i in range(12):
            start = date(2020, 1, 1) + timedelta(days=(2 * i + p) * 60)
            end = start + timedelta(days=29)
            ident = f"{panel}{i}"
            cycles.append({"cycle_id": ident, "information_start": start.isoformat(),
                           "information_end": end.isoformat()})
            for offset in (10, 20):
                day = (start + timedelta(days=offset)).isoformat()
                pack["cases"].append({"id": f"fixture-{panel}-{i}-{offset}", **source,
                    "as_of": day, "label_mode": "asof", "role": "holdout", "panel": panel,
                    "cycle_id" if panel == "cycle" else "block_id": ident,
                    "information_start": start.isoformat(), "information_end": day})
        if panel == "cycle":
            pack.update(version=VERSION, cycle_registration={"cycles": cycles},
                        exclusion_snapshot={"diagnostic_intervals": [], "consumed_intervals": []})
        else:
            pack["sampling_registration"] = {"diagnostic_intervals": [], "consumed_intervals": []}
        packs.append(pack)
    return packs


def test_joint_blank_geometry_can_pass_without_states_scores_or_mutation(monkeypatch):
    packs = manifests()
    before = deepcopy(packs)
    monkeypatch.setattr(preflight, "reference_state", lambda *args: pytest.fail("cannot read class states"))
    result = preflight.evaluate_annotation_joint_capacity(packs)
    assert result["geometry_eligible"] and not result["eligible"]
    assert result["metadata_only"] and not result["class_support_known"]
    assert result["raw_groups"] == result["effective_clusters"] == {"cycle": 12, "window": 12}
    assert len(result["missing_classes"]) == 13
    assert all(value == {"cycle": 0, "window": 0} for value in result["class_unit_support"].values())
    assert not result["predictions_evaluated"] and not result["holdout_consumed"]
    assert packs == before
    assert preflight.evaluate_annotation_joint_capacity(list(reversed(packs))) == result
    json.dumps(result)


@pytest.mark.parametrize("closed_endpoint", [False, True])
def test_full_cycle_selection_end_rejects_distinct_window_target(closed_endpoint):
    packs = manifests()
    cycle = packs[0]["cycle_registration"]["cycles"][0]
    window = packs[1]["cases"][0]
    cycle["information_end"] = window["information_start"] if closed_endpoint else window["information_end"]
    result = preflight.evaluate_annotation_joint_capacity(packs)
    assert result["unit_count_eligible"]
    assert result["cross_panel_information_overlap"] and not result["geometry_eligible"]
    assert result["cluster_information_intervals"]["cycle"][0]["end"] == cycle["information_end"]


def test_cycle_overlaps_merge_transitively_in_joint_geometry():
    packs = manifests()
    cycles = packs[0]["cycle_registration"]["cycles"]
    cycles[0]["information_end"] = cycles[1]["information_start"]
    cycles[1]["information_end"] = cycles[2]["information_start"]
    result = preflight.evaluate_annotation_joint_capacity(packs)
    assert result["effective_clusters"] == {"cycle": 10, "window": 12}
    assert result["overlap_clusters"]["cycle"][0] == ["cycle0", "cycle1", "cycle2"]
    assert not result["geometry_eligible"]


@pytest.mark.parametrize("mutation, message", [
    ("source", "same exact source"), ("market", "same exact source"),
    ("mode", "same exact source"), ("rubric", "same frozen rubric"),
    ("protocol", "same frozen rubric"), ("partition", "same explicit partition"),
    ("missing_partition", "ISO date"), ("duplicate_pack", "distinct registered pack"),
    ("duplicate_group", "duplicate group"), ("same_date", "duplicate reference session"),
    ("case_identity", "registered identity"), ("case_role", "registered identity"),
    ("case_panel", "registered sampling panel"),
])
def test_joint_identity_and_all_case_conflicts_fail_before_filtering(mutation, message):
    packs = manifests()
    if mutation == "source": packs[1]["source"]["ohlcv_sha256"] = "changed"
    elif mutation == "market": packs[1]["source"]["market"] = "binance_spot"
    elif mutation == "mode": packs[1]["settings"]["label_mode"] = "retrospective"
    elif mutation == "rubric": packs[1]["rubric_identity"]["sha256"] = "changed"
    elif mutation == "protocol": packs[1]["protocol_sha256"] = "changed"
    elif mutation == "partition": packs[1]["settings"]["holdout_start"] = "2021-01-01"
    elif mutation == "missing_partition":
        for pack in packs: pack["settings"].pop("development_start")
    elif mutation == "duplicate_pack": packs[1]["pack_id"] = packs[0]["pack_id"]
    elif mutation == "duplicate_group":
        packs[0] = deepcopy(packs[1]); packs[0]["pack_id"] = "another-window"
    elif mutation == "same_date":
        packs[1]["cases"][0].update(as_of=packs[0]["cases"][0]["as_of"],
            information_start=packs[0]["cases"][0]["information_start"],
            information_end=packs[0]["cases"][0]["information_end"], role="development")
    elif mutation == "case_identity": packs[1]["cases"][0]["symbol"] = "OTHER"
    elif mutation == "case_role": packs[1]["cases"][0]["role"] = "diagnostic"
    elif mutation == "case_panel": packs[1]["cases"][0]["panel"] = "cycle"
    with pytest.raises(ValueError, match=message):
        preflight.evaluate_annotation_joint_capacity(packs)


def test_joint_mixed_group_roles_and_invalid_partitions_do_not_depend_on_input_order():
    packs = manifests()
    packs[0]["cases"][0]["role"] = "development"
    for case in packs[1]["cases"][:2]: case["role"] = "development"
    result = preflight.evaluate_annotation_joint_capacity(packs)
    assert result["cutoff_crossing_groups"] == [["cycle", "cycle0"], ["window", "window0"]]
    assert not result["geometry_eligible"]
    assert result == preflight.evaluate_annotation_joint_capacity(list(reversed(packs)))


@pytest.mark.parametrize("boundary", ["development_start", "holdout_start", "holdout_end"])
def test_joint_frozen_partition_uses_full_group_bounds(boundary):
    packs = manifests()
    if boundary == "development_start":
        for case in packs[0]["cases"][:2]: case["role"] = "development"
        for pack in packs: pack["settings"]["development_start"] = "2020-01-02"
        for pack in packs: pack["settings"]["holdout_start"] = "2020-03-01"
    elif boundary == "holdout_start":
        for case in packs[0]["cases"][:2]: case["role"] = "development"
        for pack in packs: pack["settings"]["holdout_start"] = "2020-01-25"
    else:
        packs[0]["cycle_registration"]["cycles"][-1]["information_end"] = "2026-09-30"
    result = preflight.evaluate_annotation_joint_capacity(packs)
    assert not result["geometry_eligible"]
    assert result["cutoff_crossing_groups"]
    assert "information_crosses_partition_cutoff" in result["diagnostic_reasons"]


@pytest.mark.parametrize("origin", ["cycle_frozen", "window_frozen", "current", "diagnostic"])
def test_joint_union_frozen_and_new_exposure_blocks_complete_cycle(origin):
    packs = manifests()
    # Both actual ASOF charts end earlier; selector's complete cycle still overlaps.
    interval = ["2020-01-26", "2020-01-30"]
    current = []
    if origin == "cycle_frozen": packs[0]["exclusion_snapshot"]["consumed_intervals"] = [interval]
    elif origin == "window_frozen": packs[1]["sampling_registration"]["consumed_intervals"] = [interval]
    elif origin == "current": current = [interval]
    else: packs[1]["sampling_registration"]["diagnostic_intervals"] = [interval]
    result = preflight.evaluate_annotation_joint_capacity(packs, consumed_intervals=current)
    assert not result["geometry_eligible"]
    if origin == "diagnostic": assert result["contaminated_groups"]
    else: assert result["consumed_overlap_groups"] == [["cycle", "cycle0"]]
    packs[0]["pack_id"] = "renamed-cycle"
    assert not preflight.evaluate_annotation_joint_capacity(packs, consumed_intervals=current)["geometry_eligible"]


def real_fixture_packs(history, tmp_path):
    path, rubric = tmp_path / "registration.json", tmp_path / "rubric.md"
    raw = registration()
    path.write_text(json.dumps(raw)); rubric.write_text("Engineering human rubric fixture, not real truth.\n")
    cycle, window = tmp_path / "cycle", tmp_path / "window"
    manifest = prepare_cycle_annotation_pack(history[0], path, cycle, rubric_path=rubric)
    annotation.prepare_annotation_pack(history[0], "binance_spot", "BTCUSDT", window,
        sampling_version="sparse-window-v2", development_start=raw["partition"]["development_start"],
        holdout_start=raw["partition"]["holdout_start"], holdout_end=raw["partition"]["holdout_end"],
        development_blocks=2, holdout_blocks=2, rubric_path=rubric,
        human_cycle_bounds=manifest["sampling"]["human_cycle_bounds"])
    return cycle, window


def test_joint_cli_verifies_frozen_packs_and_new_shared_consumption_read_only(history, tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    monkeypatch.setattr(holdout, "REGISTRY", registry)
    cycle, window = real_fixture_packs(history, tmp_path)
    for owner, name in ((cli, "replay"), (cli, "optimize_batch"), (holdout, "check_unused"),
                        (holdout, "claim_holdout")):
        monkeypatch.setattr(owner, name, lambda *a, **k: pytest.fail("preflight must not score or claim"))
    output = tmp_path / "capacity.json"
    args = ["preflight", "--joint-packs", str(cycle), str(window), "--output", str(output)]
    assert cli.main(args) == 2
    assert not registry.exists()
    report = json.loads(output.read_text())
    assert report["metadata_only"] and not report["eligible"]
    assert not report["predictions_evaluated"] and not report["holdout_consumed"]
    assert not report["reports"][1]["cross_panel_information_overlap"]
    uses = registry / "binance_spot/BTCUSDT/uses.json"
    uses.parent.mkdir(parents=True)
    contents = json.dumps({"schema_version": 1, "market": "binance_spot", "symbol": "BTCUSDT",
        "timeframe": "1d", "uses": [{"information_intervals": [["2022-12-01", "2022-12-31"]]}]})
    uses.write_text(contents)
    assert cli.main(args) == 2
    assert uses.read_text() == contents
    report = json.loads(output.read_text())
    assert report["reports"][1]["consumed_overlap_groups"] == [["cycle", "fixture-human-holdout"]]
    assert not list(tmp_path.rglob(".lock")) and not list(tmp_path.rglob("*.sqlite3"))
    assert not list(tmp_path.rglob("run.json"))
    chart = next((cycle / "charts").glob("*.html"))
    chart.write_text(chart.read_text() + "tampered")
    with pytest.raises(ValueError, match="chart"):
        annotation._load_manifest(cycle)
