"""Research contract checks, not market-accuracy evidence."""
from dataclasses import asdict
import json

import optuna
import pandas as pd
import pytest

from research.tuning.cli import _qualified_reference, main
from research.tuning.profiles import resolve_config
from research.tuning.runner import optimize_batch, run_lock
from research.tuning.space import BATCHES, sample_configs, search_space, validate_candidate
from wyckoff_core.models import CoreConfig


@pytest.mark.parametrize("market", ["us_equities", "binance_spot"])
def test_spaces_and_samples_obey_declared_constraints(market):
    space = search_space(market)
    assert set(space) == set(asdict(CoreConfig())) == {n for group in BATCHES.values() for n in group}
    assert validate_candidate(asdict(CoreConfig()), market) == CoreConfig()
    first = sample_configs(market, 100, 42)
    assert first == sample_configs(market, 100, 42)
    assert len(set(first)) > 90
    for cfg in first:
        assert cfg.min_range_bars >= 2 * cfg.swing_confirmation + 1
        assert cfg.preliminary_volume_min <= cfg.volume_high
        assert cfg.local_trend_bars <= cfg.trend_window


def _profile():
    return {"schema_version": 1, "profiles": [{"market": "us_equities", "symbol": "AAPL",
        "timeframe": "1d", "label_mode": "asof", "status": "VALIDATED_ON_LABELED_HOLDOUT",
        "training_end": "2023-12-31", "label_information_end": "2024-12-31",
        "created_at": "2025-01-01T12:00:00+00:00", "effective_from": "2025-01-02",
        "params": {"phase_e_closes": 5}}]}


def test_profile_identity_time_modes_and_explicit_precedence():
    profile = _profile()
    assert resolve_config("us_equities", "AAPL", profiles=profile, as_of="2025-01-02").phase_e_closes == 5
    assert resolve_config("us_equities", "MSFT", profiles=profile) == CoreConfig()
    with pytest.raises(ValueError):
        resolve_config("us_equities", "AAPL", profiles=profile, as_of="2024-01-01")
    with pytest.raises(ValueError):
        resolve_config("us_equities", "AAPL", profiles=profile, as_of="2025-01-02", label_mode="retrospective")
    assert resolve_config("us_equities", "AAPL", explicit=CoreConfig(), profiles=profile) == CoreConfig()
    assert resolve_config("us_equities", "AAPL", profiles=profile, mode="reconstruction").phase_e_closes == 5
    with pytest.raises(ValueError):
        resolve_config("binance_spot", "AAPL", explicit=CoreConfig())


@pytest.mark.parametrize("field", ["training_end", "label_information_end", "created_at", "effective_from"])
def test_profile_nat_cannot_bypass_temporal_check(field):
    profile = _profile()
    profile["profiles"][0][field] = "NaT"
    with pytest.raises(ValueError):
        resolve_config("us_equities", "AAPL", profiles=profile, as_of="2000-01-01")


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_exact_resume_total_budget_and_input_freeze(tmp_path, sampler):
    def objective(cfg):
        return {"score": 1 - abs(cfg.phase_e_closes - 5) / 10, "constraints": {"allowed": 0}}
    kwargs = {"market": "us_equities", "base": CoreConfig(), "batch": "C", "sampler_name": sampler,
        "total_trials": 36, "seed": 23, "identity": "engineering-fixture", "objective": objective}
    with run_lock(tmp_path / "split"):
        first = optimize_batch(tmp_path / "split", max_new_trials=17, **kwargs)
        assert first["proposals"] == 17
        completed = optimize_batch(tmp_path / "split", **kwargs)
        assert completed["proposals"] == 36
        repeated = optimize_batch(tmp_path / "split", **kwargs)
        assert repeated["proposals"] == 36
    with run_lock(tmp_path / "whole"):
        optimize_batch(tmp_path / "whole", **kwargs)
    def params(directory):
        study = optuna.load_study(study_name="parameters", storage=f"sqlite:///{directory / 'study.sqlite3'}")
        return [(t.params, t.value, t.constraints) for t in study.trials]
    assert params(tmp_path / "whole") == params(tmp_path / "split")
    with pytest.raises(ValueError, match="changed"):
        optimize_batch(tmp_path / "split", **{**kwargs, "identity": "changed"})


def test_failed_objective_consumes_budget_and_soft_constraint_not_selected(tmp_path):
    kwargs = {"market": "us_equities", "base": CoreConfig(), "batch": "C", "sampler_name": "random",
              "total_trials": 3, "seed": 11, "identity": "engineering-fixture"}
    def broken(cfg):
        raise ValueError("bad input must abort")
    with pytest.raises(ValueError, match="abort"):
        optimize_batch(tmp_path, objective=broken, **kwargs)
    calls = []
    def invalid(cfg):
        calls.append(cfg)
        return {"score": 1, "constraints": {"violation": 1}}
    result = optimize_batch(tmp_path, objective=invalid, **kwargs)
    assert len(calls) == 2
    assert result["proposals"] == 3 and result["status"] == "NO_FEASIBLE_CANDIDATE"


def test_divergent_checkpoint_and_concurrent_writer_rejected(tmp_path):
    kwargs = {"market": "us_equities", "base": CoreConfig(), "batch": "C", "sampler_name": "random",
        "total_trials": 2, "seed": 1, "identity": "engineering-fixture", "objective": lambda cfg: {"score": 0.5}}
    with run_lock(tmp_path):
        with pytest.raises(ValueError, match="already"):
            with run_lock(tmp_path):
                pass
        optimize_batch(tmp_path, max_new_trials=1, **kwargs)
    checkpoint = tmp_path / "sampler.json"
    state = json.loads(checkpoint.read_text()); state["trials"] = 0
    checkpoint.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="diverged"):
        optimize_batch(tmp_path, **kwargs)


def test_qualification_preserves_cycle_boundaries():
    dates = pd.date_range("2024-08-01", periods=65)
    frame = pd.DataFrame({"Volume": [1.] * 65}, index=dates)
    frame.loc["2024-09-30", "Volume"] = 0
    reference = {"labels": [{"as_of": "2024-09-30", "cycle_id": "cross"},
                             {"as_of": "2024-10-02", "cycle_id": "cross"}]}
    item = {"continuous_segments": [{"start": str(dates[0].date()), "end": str(dates[-1].date())}]}
    qualified = _qualified_reference(reference, frame, item, "us_equities")
    assert len(qualified["labels"]) == 2
    assert qualified["labels"][0]["qualification_issue"]
    assert "qualification_issue" not in qualified["labels"][1]


def test_inspect_cli_declared_universe(tmp_path, capsys):
    path = tmp_path / "inspect.json"
    assert main(["inspect", "--market", "binance_spot", "--symbol", "ETHUSDT", "--output", str(path)]) == 0
    assert len(json.loads(path.read_text())["parameters"]) == 31
    assert main(["inspect", "--market", "binance_spot", "--symbol", "AAPL"]) == 2


def test_supervised_nested_pipeline_only_exports_unvalidated_candidate(tmp_path, monkeypatch):
    from test_tuning_evaluation import complete_reference, label
    from datetime import date, timedelta
    from research.tuning.labels import joint_classes
    from research.tuning import cli
    from research.tuning import holdout
    monkeypatch.setattr(holdout, "REGISTRY", tmp_path / "holdout-registry")
    reference = complete_reference("asof")
    # The original four holdout groups are enough for split construction, but
    # cannot pass the new prediction-free12-cluster promotion preflight.
    for i in range(4, 12):
        for panel_index, panel in enumerate(("cycle", "window")):
            base = date(2025, 1, 1) + timedelta(days=(i * 2 + panel_index) * 25)
            for j, state in enumerate(joint_classes("asof")):
                phase, regime = state.split("/") if "/" in state else ("no_observable_phase", "none")
                reference["labels"].append(label((base + timedelta(days=j)).isoformat(),
                    phase=phase, regime=regime, panel=panel, group=f"holdout_{panel}_{i}",
                    role="holdout", mode="asof", information_start=base.isoformat()))
    dates = pd.date_range("2009-01-01", "2026-09-29")
    price = pd.Series(100 + 0.01 * pd.RangeIndex(len(dates)), index=dates)
    frame = pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1,
                          "Close": price, "Volume": 1000}, index=dates)
    item = {"continuous_segments": [{"start": "2009-01-01", "end": "2026-09-29"}], "attrs": {"fixture": True}}
    monkeypatch.setattr(cli, "load_history", lambda *args: (frame, item))
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(reference))
    monkeypatch.chdir(tmp_path)
    result = cli.supervised(tmp_path, "us_equities", "MSFT", "labels.json", tmp_path / "run",
                            trials=2, batches=("C",), joint_batches=("J2",), joint_trials=1)
    assert result["status"] == "SEARCH_COMPLETED"
    candidate = result["candidate"]
    assert candidate["status"] == "CANDIDATE_ONLY" and not candidate["holdout_evaluated"]
    assert candidate["params"] == asdict(CoreConfig())  # flat response retains original default
    assert candidate["label_information_end"] < "2024-10-01"
    results = json.loads((tmp_path / "run/results.json").read_text())
    assert len(results["studies"]) == 6  # two outer folds and final, each TPE/Random
    assert {r["sampler"] for r in results["studies"]} == {"tpe", "random"}
    assert all(r["batches"][0]["proposals"] == 2 for r in results["studies"])
    assert all(r["joint_refinement"][0]["proposals"] == 1 for r in results["studies"])
    context = json.loads((tmp_path / "run/run.json").read_text())["context"]
    assert context["joint_batches"] == ["J2"] and context["joint_trials_per_sampler"] == 1
    assert not (tmp_path / "run/holdout.json").exists()
    assert cli.main(["resume", "--run", str(tmp_path / "run")]) == 0
    resumed = json.loads((tmp_path / "run/results.json").read_text())
    assert all(r["batches"][0]["proposals"] == 2 for r in resumed["studies"])
    assert all(r["joint_refinement"][0]["proposals"] == 1 for r in resumed["studies"])
    from research.tuning import validation
    monkeypatch.setattr(validation, "load_history", lambda *args: (frame, item))
    verified = validation.validate_run(tmp_path / "run")
    assert verified["status"] == "INSUFFICIENT_TEST_POWER"
    assert not verified["profile_export_allowed"]
    assert all(check["prefix_equal"] for check in verified["causal_checks"])
    with pytest.raises(ValueError, match="successful"):
        validation.export_profile(tmp_path / "run", tmp_path / "profile.json")
    assert cli.main(["resume", "--run", str(tmp_path / "run")]) == 2
