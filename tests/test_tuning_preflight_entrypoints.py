"""Impossible reference geometry must stop before optimizer or test exposure."""
import json

import pandas as pd
import pytest

from research.tuning import cli, holdout, validation
from research.tuning.labels import load_labels
from research.tuning.preflight import evaluate_reference_capacity
from research.tuning.splits import build_split_plan
from test_tuning_evaluation import complete_reference
from test_tuning_validation import frozen_run  # noqa: F401 - pytest fixture


def test_search_rejects_raw_ready_but_four_cluster_reference(tmp_path, monkeypatch):
    raw = complete_reference("asof")
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(raw))
    dates = pd.date_range("2009-01-01", "2026-09-29")
    frame = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.}, index=dates)
    item = {"continuous_segments": [{"start": "2009-01-01", "end": "2026-09-29"}], "attrs": {}}
    monkeypatch.setattr(cli, "load_history", lambda *args: (frame, item))
    monkeypatch.setattr(cli, "optimize_batch", lambda *args, **kwargs: pytest.fail("optimizer must not start"))
    monkeypatch.setattr(cli, "replay", lambda *args, **kwargs: pytest.fail("predictions must not be evaluated"))
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "registry")
    reference = load_labels(path, market="us_equities", symbol="MSFT", label_mode="asof")
    assert build_split_plan(reference)["status"] == "READY"
    result = cli.supervised(tmp_path, "us_equities", "MSFT", path, tmp_path / "run", trials=2)
    assert result["status"] == "INSUFFICIENT_REFERENCE_CAPACITY"
    assert not result["optimization_started"]
    assert result["reference_capacity"]["effective_clusters"] == {"cycle": 4, "window": 4}
    assert (tmp_path / "run/qualification.json").exists()
    assert not list((tmp_path / "run").rglob("study.sqlite3"))
    assert not (tmp_path / "run/run.json").exists()
    assert not list((tmp_path / "registry").rglob("uses.json"))


def test_validation_metadata_rejection_precedes_any_replay_or_claim(frozen_run, monkeypatch):
    def insufficient(reference):
        subset = [row for row in reference["labels"] if
                  int(row.get("cycle_id", row.get("block_id")).rsplit("_", 1)[1]) < 4]
        return evaluate_reference_capacity({**reference, "labels": subset})
    monkeypatch.setattr(validation, "evaluate_reference_capacity", insufficient)
    monkeypatch.setattr(validation, "replay", lambda *args, **kwargs: pytest.fail("no score or prediction allowed"))
    monkeypatch.setattr(validation, "claim_holdout", lambda *args, **kwargs: pytest.fail("no test consumption allowed"))
    result = validation.validate_run(frozen_run)
    assert result["status"] == "INSUFFICIENT_REFERENCE_CAPACITY"
    assert not result["holdout_evaluated"] and not result["profile_export_allowed"]
    assert not (frozen_run / "holdout-use.json").exists()
    assert not list(holdout.REGISTRY.rglob("uses.json"))
