import json

import numpy as np
import pandas as pd

from research import long_study
from research.universe import Instrument, ResearchProfile
from wyckoff_core.models import CORE_VERSION


def test_session_gap_resets_warmup_and_keeps_all_observed_rows(tmp_path, monkeypatch):
    profile = ResearchProfile("us_equities", "Fixture", "equity", "cash", "USD",
                              (Instrument("AAPL", "Fixture", ""),), "SPY")
    monkeypatch.setattr(long_study, "PROFILES", {profile.id: profile})
    dates = pd.date_range("2020-01-01", periods=30).append(pd.date_range("2020-02-01", periods=30))
    close = 100 + 3 * np.sin(np.arange(60))
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close,
                       "Volume": 100 + np.arange(60)}, index=dates)
    csv = tmp_path / "AAPL.csv"
    df.to_csv(csv, index_label="Date")
    segments = [{"start": "2020-01-01", "end": "2020-01-30", "bars": 30},
                {"start": "2020-02-01", "end": "2020-03-01", "bars": 30}]
    item = {"status": "partial_missing_sessions", "csv": str(csv), "attrs": {},
            "continuous_segments": segments, "first": "2020-01-01", "last": "2020-03-01",
            "span_years": 0.2, "missing_count": 1}
    manifest = {"request": {"start": "2020-01-01", "end": "2020-03-02"}, "limitations": [],
                "profiles": {profile.id: {"AAPL": item}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(long_study, "coverage", lambda *a, **kw: {"continuous_segments": segments})
    old_report = tmp_path / "study" / "report.json"
    old_report.parent.mkdir()
    old_report.write_text("preserved core2 history")
    result = long_study.replay_history(tmp_path)
    assert old_report.read_text() == "preserved core2 history"
    assert (tmp_path / f"core-v{CORE_VERSION}-study" / "report.json").is_file()
    instrument = result["profiles"][profile.id]["AAPL"]
    assert instrument["status"] == "complete" and instrument["source_status"] == "partial_missing_sessions"
    assert len(instrument["prefix_checks"]) == 6
    stages = pd.read_csv(instrument["csv"], keep_default_na=False)
    assert len(stages) == 60 and stages.Date.is_unique
    assert result["core_version"] == CORE_VERSION
    assert {"phase_cycle_id", "parent_range_id", "prior_range_phase", "stopping_candidate"} <= set(stages)
    assert (stages.core_version == CORE_VERSION).all()
    assert (stages.loc[:19, "status"] == "insufficient_data").all()
    assert (stages.loc[30:49, "status"] == "insufficient_data").all()


def test_failed_source_is_visible_and_cannot_be_promoted_from_old_csv(tmp_path, monkeypatch):
    profile = ResearchProfile("us_equities", "Fixture", "equity", "cash", "USD",
                              (Instrument("AAPL", "Fixture", ""),), "SPY")
    monkeypatch.setattr(long_study, "PROFILES", {profile.id: profile})
    (tmp_path / "manifest.json").write_text(json.dumps({"request": {}, "limitations": [],
        "profiles": {profile.id: {"AAPL": {"status": "failed", "error": "unresolved source error"}}}}))
    output = tmp_path / "study" / profile.id
    output.mkdir(parents=True)
    (output / "AAPL.stages.csv").write_text("stale successful file")
    result = long_study.replay_history(tmp_path)
    assert result["profiles"][profile.id]["AAPL"] == {
        "status": "failed", "error": "ValueError: unresolved source error"}
