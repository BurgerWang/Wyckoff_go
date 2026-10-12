"""Versioned blind sampling and context honesty; no market reference labels."""
import copy
import json

import pytest

from research.tuning import annotation, holdout
from research.tuning.labels import load_labels
from research.tuning.preflight import evaluate_annotation_capacity
from test_tuning_annotation import history, prepare, reviewed  # noqa: F401


def sparse(history, output, **kwargs):
    return prepare(history, output, sampling_version="sparse-window-v2", holdout_start="2020-01-01", **kwargs)


def test_sparse_pack_preserves_v1_and_chart_bounds(history, tmp_path, monkeypatch):
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "registry")
    old = tmp_path / "old"
    old_manifest = prepare(history, old)
    before = (old / "manifest.json").read_bytes()
    pack = tmp_path / "sparse"
    manifest = sparse(history, pack)
    assert manifest["version"] == annotation.SPARSE_PACK_VERSION
    assert manifest["pack_id"] != old_manifest["pack_id"]
    assert annotation._load_manifest(pack) == manifest
    assert annotation._load_manifest(old) == old_manifest
    assert (old / "manifest.json").read_bytes() == before
    assert all(value["overlapping_block_pairs"] == 0 for value in annotation.annotation_overlap_support(manifest).values())
    capacity = evaluate_annotation_capacity(manifest)
    assert capacity["effective_clusters"]["window"] == 2 and not capacity["eligible"]
    assert manifest["sampling"]["support"]["holdout"]["max_disjoint_capacity"] > 2
    for block in manifest["sampling"]["blocks"]:
        assert block["exposure_bars"] == 129
        cases = [case for case in manifest["cases"] if case["block_id"] == block["block_id"]]
        assert min(case["information_start"] for case in cases) == block["information_start"]
        assert max(case["information_end"] for case in cases) == block["information_end"]
        assert all(case["information_end"] == case["as_of"] for case in cases)


def test_sparse_excludes_human_cycles_and_freezes_consumption_snapshot(history, tmp_path, monkeypatch):
    registry = tmp_path / "registry"
    monkeypatch.setattr(holdout, "REGISTRY", registry)
    bounds = [{"cycle_id": "human-boundary-metadata-only", "role": "holdout",
               "information_start": "2021-01-01", "information_end": "2022-12-31"}]
    pack = tmp_path / "pack"
    manifest = sparse(history, pack, human_cycle_bounds=bounds)
    for block in manifest["sampling"]["blocks"]:
        assert block["information_end"] < "2021-01-01" or block["information_start"] > "2022-12-31"
    assert manifest["sampling_registration"]["human_cycle_bounds"] == bounds
    uses = registry / "binance_spot/BTCUSDT/uses.json"
    uses.parent.mkdir(parents=True)
    uses.write_text(json.dumps({"schema_version": 1, "market": "binance_spot", "symbol": "BTCUSDT",
        "timeframe": "1d", "uses": [{"information_intervals": [["2020-01-01", "2026-09-29"]]}]}))
    assert annotation._load_manifest(pack) == manifest
    with pytest.raises(ValueError, match="different sampling"):
        sparse(history, pack, human_cycle_bounds=bounds)


@pytest.mark.parametrize("field,value", [("request_context", True), ("request_context", "true"),
    ("additional_exposure", [{"information_start": "2010-01-01", "information_end": "2026-09-29"}])])
def test_import_cannot_ignore_pending_or_unregistered_context(history, tmp_path, field, value):
    pack = tmp_path / "pack"
    prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    raw["labels"][0][field] = value
    path.write_text(json.dumps(raw))
    output = tmp_path / "labels.json"
    with pytest.raises(ValueError, match="context|exposure"):
        annotation.import_annotations(pack, path, output)
    assert not output.exists()


def test_import_preserves_checked_exposure_and_original_review_is_rechecked(history, tmp_path):
    pack = tmp_path / "pack"
    prepare(history, pack)
    raw, path = reviewed(pack, tmp_path)
    record = raw["labels"][0]
    record.update(request_context=False, additional_exposure=[{
        "information_start": record["information_start"], "information_end": record["information_end"]}])
    path.write_text(json.dumps(raw))
    result = annotation.import_annotations(pack, path, tmp_path / "labels.json")
    assert load_labels(tmp_path / "labels.json", market="binance_spot", symbol="BTCUSDT", label_mode="asof")["labels"]
    assert result["labels"][0]["additional_exposure"] == record["additional_exposure"]
    original = copy.deepcopy(record)
    source = result["sources"][0]
    annotation.validate_annotation_label(result["labels"][0], source, tmp_path / "labels.json")
    raw["labels"][0]["request_context"] = True
    path.write_text(json.dumps(raw))
    source["review_sha256"] = annotation._bytes_digest(path)
    with pytest.raises(ValueError, match="pending request_context"):
        annotation.validate_annotation_label(result["labels"][0], source, tmp_path / "labels.json")
    assert original["request_context"] is False
