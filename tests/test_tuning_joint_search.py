"""Synthetic joint-search engineering; never a market accuracy claim."""
from dataclasses import asdict, replace
import json
import math

import optuna
import pytest

from research.tuning.runner import optimize_batch, run_lock
from research.tuning.space import BATCHES, JOINT_BATCHES, JOINT_SEARCH_VERSION, sample_configs, validate_candidate
from wyckoff_core.models import CoreConfig


def study_trials(directory):
    return optuna.load_study(study_name="parameters", storage=f"sqlite:///{directory / 'study.sqlite3'}").trials


def kwargs(batch="J1", sampler="random", trials=36, base=None):
    return {"market": "us_equities", "base": base or CoreConfig(), "batch": batch,
            "sampler_name": sampler, "total_trials": trials, "seed": 23,
            "identity": "synthetic-joint-engineering"}


def interaction(config):
    # A structural field from original A and a trend field from original B
    # must move together. Either isolated move makes the score worse.
    original = CoreConfig()
    if config.swing_confirmation == 1 and config.trend_window == 10:
        value = .8
    elif (config.swing_confirmation, config.trend_window) == (original.swing_confirmation, original.trend_window):
        value = .5
    else:
        value = .45
    return {"score": value, "metrics": {"synthetic": True, "selected_score": value}}


def test_predeclared_joint_groups_do_not_change_original_batches_or_space():
    assert JOINT_SEARCH_VERSION == "daily-joint-v1"
    assert tuple(JOINT_BATCHES) == ("J1", "J2", "J3")
    assert not set(JOINT_BATCHES) & set(BATCHES)
    assert "swing_confirmation" in BATCHES["A"] and "trend_window" in BATCHES["B"]
    for market in ("us_equities", "binance_spot"):
        first = sample_configs(market, 12, 23, batches=tuple(JOINT_BATCHES))
        assert first == sample_configs(market, 12, 23, batches=tuple(JOINT_BATCHES))
        for config in first:
            assert validate_candidate(asdict(config), market) == config


def test_cross_original_batch_interaction_requires_joint_proposal(tmp_path):
    for name in ("A", "B"):
        result = optimize_batch(tmp_path / name, objective=interaction, **kwargs(batch=name))
        assert result["best_value"] <= .5
    joint = optimize_batch(tmp_path / "joint", objective=interaction,
                           minimum_value=.51, selection_reference=CoreConfig(), **kwargs())
    assert joint["status"] == "SEARCH_COMPLETED"
    assert joint["config"]["swing_confirmation"] == 1
    assert joint["config"]["trend_window"] == 10
    assert joint["value"] == joint["metrics"]["selected_score"] == .8
    assert joint["metrics"]["synthetic"]
    assert joint["diagnostics"]["complete"] == joint["diagnostics"]["feasible"] == 36
    assert all(set(trial.params) == set(JOINT_BATCHES["J1"]) for trial in study_trials(tmp_path / "joint"))


@pytest.mark.parametrize("sampler", ["tpe", "random"])
def test_joint_exact_resume_crosses_tpe_startup_and_preserves_budget(tmp_path, sampler):
    options = {**kwargs(sampler=sampler), "objective": interaction,
               "selection_reference": CoreConfig(), "minimum_value": .51}
    with run_lock(tmp_path / "split"):
        first = optimize_batch(tmp_path / "split", max_new_trials=17, **options)
        assert first["proposals"] == 17
        completed = optimize_batch(tmp_path / "split", **options)
        assert completed["proposals"] == 36
        assert completed == optimize_batch(tmp_path / "split", **options)
    whole = optimize_batch(tmp_path / "whole", **options)
    assert completed == whole
    def trace(directory):
        return [(trial.params, trial.value, trial.state, trial.constraints, trial.user_attrs)
                for trial in study_trials(directory)]
    assert trace(tmp_path / "split") == trace(tmp_path / "whole")
    assert json.loads((tmp_path / "split/sampler.json").read_text())["trials"] == 36
    for changes in ({"minimum_value": .52}, {"selection_reference": replace(CoreConfig(), trend_window=20)}):
        with pytest.raises(ValueError, match="changed"):
            optimize_batch(tmp_path / "split", **{**options, **changes})


def test_original_default_distance_and_actual_selected_minimum(tmp_path):
    values = {2: .5, 3: .715, 5: .7, 8: .722}
    objective = lambda config: {"score": values[config.phase_e_closes],
                                "metrics": {"actual": values[config.phase_e_closes], "preserved": [1, 2]}}
    options = {**kwargs(batch="C", base=replace(CoreConfig(), phase_e_closes=8)), "objective": objective}
    base_nearest = optimize_batch(tmp_path / "base", **options)
    original_nearest = optimize_batch(tmp_path / "original", selection_reference=CoreConfig(), **options)
    assert base_nearest["config"]["phase_e_closes"] == 8
    assert original_nearest["config"]["phase_e_closes"] == 3
    assert original_nearest["value"] == .715 and original_nearest["best_value"] == .722
    floor = optimize_batch(tmp_path / "floor", selection_reference=CoreConfig(), minimum_value=.72, **options)
    assert floor["config"]["phase_e_closes"] == 8
    assert floor["value"] == floor["metrics"]["actual"] == .722
    assert floor["metrics"]["preserved"] == [1, 2]
    rejected = optimize_batch(tmp_path / "too_high", minimum_value=.73, **options)
    assert rejected["status"] == "NO_QUALIFYING_CANDIDATE"
    assert rejected["config"] == asdict(options["base"])
    assert rejected["best_value"] == .722 and "value" not in rejected


def test_invalid_joint_configs_consume_budget_and_report_reasons(tmp_path):
    calls = []
    base = replace(CoreConfig(), min_range_bars=8)
    def objective(config):
        calls.append(config)
        return {"score": .5}
    result = optimize_batch(tmp_path, objective=objective, **kwargs(base=base))
    diagnostic = result["diagnostics"]
    assert diagnostic["invalid_config"] > 0
    assert diagnostic["invalid_config"] + diagnostic["complete"] == result["proposals"] == 36
    assert len(calls) == diagnostic["complete"]
    assert diagnostic["reason_counts"]["invalid_config"]["range length cannot be below confirmation span"] > 0
    assert diagnostic["objective_failure"] == 0


def test_duplicates_still_score_and_resume_failure_and_constraints(tmp_path):
    options = kwargs(batch="C", trials=12)
    def broken(config):
        raise ValueError("synthetic objective failure")
    with pytest.raises(ValueError, match="synthetic objective failure"):
        optimize_batch(tmp_path, objective=broken, **options)
    calls = []
    def rejected(config):
        calls.append(config)
        return {"score": .9, "constraints": {"original_default": 1, "ce_recall": 2},
                "metrics": {"fixture": True}}
    result = optimize_batch(tmp_path, objective=rejected, **options)
    diagnostic = result["diagnostics"]
    assert result["status"] == "NO_FEASIBLE_CANDIDATE"
    assert result["proposals"] == 12 and len(calls) == 11
    assert diagnostic["complete"] == diagnostic["constraint_rejected"] == 11
    assert diagnostic["feasible"] == 0 and diagnostic["objective_failure"] == 1
    assert diagnostic["duplicate"] >= 8  # only four configs; no retry bypasses budget
    assert diagnostic["reason_counts"]["constraint_rejected"] == {"ce_recall": 11, "original_default": 11}
    assert diagnostic["reason_counts"]["objective_failure"] == {"ValueError: synthetic objective failure": 1}
    tagged = [trial for trial in study_trials(tmp_path) if "duplicate_of_trial" in trial.user_attrs]
    assert len(tagged) == diagnostic["duplicate"]
    assert result == optimize_batch(tmp_path, objective=rejected, **options)
    assert len(calls) == 11


@pytest.mark.parametrize("minimum", [True, math.nan, math.inf, "0.5"])
def test_minimum_is_validated_before_creating_study(tmp_path, minimum):
    with pytest.raises(ValueError, match="minimum_value"):
        optimize_batch(tmp_path, minimum_value=minimum, objective=interaction, **kwargs())
    assert not (tmp_path / "study.sqlite3").exists()


def test_nonfinite_constraint_is_failure_not_feasible_candidate(tmp_path):
    with pytest.raises(ValueError, match="constraints must be finite"):
        optimize_batch(tmp_path, objective=lambda config: {"score": .5, "constraints": {"bad": math.nan}},
                       **kwargs(trials=1))
    trials = study_trials(tmp_path)
    assert len(trials) == 1 and trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[0].user_attrs["objective_failure"]["type"] == "ValueError"
