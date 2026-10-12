"""Serial, bounded Optuna studies with exact versioned JSON RNG checkpoints."""
from contextlib import contextmanager
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from research.tuning.space import (
    BATCHES, JOINT_BATCHES, JOINT_SEARCH_VERSION, SPACE_VERSION, search_space, validate_candidate,
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is already writing this run") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def _rng_state(sampler):
    def serialize(rng):
        kind, keys, pos, gaussian, cached = rng.get_state()
        return [kind, keys.tolist(), pos, gaussian, cached]
    state = {"rng": serialize(sampler._rng.rng)}
    if hasattr(sampler, "_random_sampler"):
        state["random_rng"] = serialize(sampler._random_sampler._rng.rng)
    return state


def _restore_rng(sampler, state):
    def restore(rng, value):
        rng.set_state((value[0], np.array(value[1], dtype=np.uint32), value[2], value[3], value[4]))
    restore(sampler._rng.rng, state["rng"])
    if hasattr(sampler, "_random_sampler"):
        restore(sampler._random_sampler._rng.rng, state["random_rng"])


def _study_diagnostics(trials, base):
    """All proposal outcomes, including prior failures and repeated configs."""
    import optuna
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
    feasible = [trial for trial in complete if all(v <= 0 for v in (trial.constraints or {}).values())]
    reasons = {name: Counter() for name in ("invalid_config", "objective_failure", "duplicate", "constraint_rejected")}
    seen = set()
    for trial in trials:
        fingerprint = digest({**asdict(base), **trial.params})
        if fingerprint in seen:
            reasons["duplicate"]["repeated_configuration"] += 1
        seen.add(fingerprint)
        if trial.user_attrs.get("invalid_config"):
            reasons["invalid_config"][trial.user_attrs["invalid_config"]] += 1
        if trial.user_attrs.get("objective_failure"):
            failure = trial.user_attrs["objective_failure"]
            reasons["objective_failure"][f"{failure['type']}: {failure['reason']}"] += 1
        for name, value in (trial.constraints or {}).items():
            if trial.state == optuna.trial.TrialState.COMPLETE and value > 0:
                reasons["constraint_rejected"][name] += 1
    return {"complete": len(complete), "feasible": len(feasible),
            "invalid_config": sum(reasons["invalid_config"].values()),
            "objective_failure": sum(reasons["objective_failure"].values()),
            "duplicate": sum(reasons["duplicate"].values()),
            "constraint_rejected": len(complete) - len(feasible),
            "reason_counts": {name: dict(sorted(counts.items())) for name, counts in reasons.items()},
            "duplicate_policy": "count_and_evaluate; every proposal consumes budget"}


def optimize_batch(directory, *, market, base, batch, sampler_name, total_trials, seed, identity, objective,
                   max_new_trials=None, selection_reference=None, minimum_value=None):
    """objective(config) -> score/constraints/metrics; cap includes failed proposals.

    The pinned5.0 univariate samplers use NumPy RandomState. Only those RNGs
    need restoring; JSON avoids executing pickle from a user-provided run.
    Joint batches propose all their fields together, retaining the same bounded
    grid and cross-field validation. Repeated proposals are marked and scored;
    neither retries nor failures increase the requested cumulative budget.
    Selection may use the original default as its distance reference and an
    explicit floor on the actual selected value, before the near-best tie rule.
    All callers must hold the enclosing run_lock.
    """
    import optuna
    if optuna.__version__ != "5.0.0":
        raise ValueError("this checkpoint format requires optuna==5.0.0")
    declared = {**BATCHES, **JOINT_BATCHES}
    if batch not in declared or sampler_name not in ("tpe", "random"):
        raise ValueError("unknown batch or sampler")
    if isinstance(total_trials, bool) or not isinstance(total_trials, int) or not 1 <= total_trials <= 4096:
        raise ValueError("total trial budget must be 1..4096")
    base = validate_candidate(asdict(base), market)
    original_selection_reference = selection_reference
    selection_reference = base if selection_reference is None else validate_candidate(asdict(selection_reference), market)
    if minimum_value is not None:
        if isinstance(minimum_value, bool) or not isinstance(minimum_value, (int, float)) or not np.isfinite(minimum_value):
            raise ValueError("minimum_value must be finite or None")
        minimum_value = float(minimum_value)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature_input = {"input": identity, "market": market, "base": asdict(base), "batch": batch,
                       "sampler": sampler_name, "seed": seed, "budget": total_trials, "space": SPACE_VERSION}
    if batch in JOINT_BATCHES:
        signature_input["joint_search"] = JOINT_SEARCH_VERSION
    if original_selection_reference is not None:
        signature_input["selection_reference"] = asdict(selection_reference)
    if minimum_value is not None:
        signature_input["minimum_value"] = minimum_value
    signature = digest(signature_input)
    sampler = (optuna.samplers.TPESampler(seed=seed, multivariate=False, group=False, n_startup_trials=32)
               if sampler_name == "tpe" else optuna.samplers.RandomSampler(seed=seed))
    database = directory / "study.sqlite3"
    study = optuna.create_study(storage=f"sqlite:///{database.resolve()}", study_name="parameters",
        sampler=sampler, pruner=optuna.pruners.NopPruner(), direction="maximize", load_if_exists=True)
    previous = study.user_attrs.get("signature")
    if previous is not None and previous != signature:
        raise ValueError("run inputs/implementation/budget changed; use a new run directory")
    if previous is None and study.trials:
        raise ValueError("unidentified existing study cannot be resumed")
    study.set_user_attr("signature", signature)
    checkpoint = directory / "sampler.json"
    if study.trials:
        if not checkpoint.exists():
            raise ValueError("missing RNG checkpoint; cannot claim exact resume")
        state = json.loads(checkpoint.read_text())
        if (state.get("optuna_version") != optuna.__version__ or state.get("signature") != signature
                or state.get("trials") != len(study.trials)):
            raise ValueError("DB/checkpoint diverged; exact resume unavailable, start a separate study")
        _restore_rng(sampler, state["state"])
        if any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials):
            raise ValueError("unfinished trial requires explicit interrupted-run recovery")

    def checkpoint_now():
        write_json(checkpoint, {"optuna_version": optuna.__version__, "signature": signature,
                               "trials": len(study.trials), "state": _rng_state(sampler)})

    dimensions = search_space(market)
    # Reconstruct from DB rather than a side file, so old studies and failed
    # proposals participate in duplicate diagnostics after an exact resume.
    proposed = {}
    for previous_trial in study.trials:
        proposed.setdefault(digest({**asdict(base), **previous_trial.params}), previous_trial.number)
    remaining = max(0, total_trials - len(study.trials))
    if max_new_trials is not None:
        if isinstance(max_new_trials, bool) or not isinstance(max_new_trials, int) or max_new_trials < 0:
            raise ValueError("max_new_trials must be a nonnegative integer")
        remaining = min(remaining, max_new_trials)
    for _ in range(remaining):
        trial = study.ask()
        params = asdict(base)
        params.update({name: dimensions[name].suggest(trial, name) for name in declared[batch]})
        fingerprint = digest(params)
        if fingerprint in proposed:
            trial.set_user_attr("duplicate_of_trial", proposed[fingerprint])
        proposed.setdefault(fingerprint, trial.number)
        try:
            config = validate_candidate(params, market)
        except ValueError as exc:
            trial.set_user_attr("invalid_config", str(exc))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            checkpoint_now()
            continue
        try:
            result = objective(config)
            value = float(result["score"])
            if not np.isfinite(value):
                raise ValueError("objective must be finite")
            for name, constraint in result.get("constraints", {}).items():
                constraint = float(constraint)
                if not np.isfinite(constraint):
                    raise ValueError("objective constraints must be finite")
                trial.set_constraint(name, constraint)
            trial.set_user_attr("config", asdict(config))
            trial.set_user_attr("metrics", result.get("metrics", {}))
            study.tell(trial, value)
        except BaseException as exc:
            trial.set_user_attr("objective_failure", {"type": type(exc).__name__, "reason": str(exc)})
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            checkpoint_now()
            raise
        checkpoint_now()
    trials = study.trials
    diagnostics = _study_diagnostics(trials, base)
    feasible = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE
                and all(v <= 0 for v in (t.constraints or {}).values())]
    if not feasible:
        result = {"status": "NO_FEASIBLE_CANDIDATE", "config": asdict(base), "proposals": len(trials),
                  "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in trials), "diagnostics": diagnostics}
        write_json(directory / "candidate.json", result)
        return result
    best_value = max(t.value for t in feasible)
    selectable = [t for t in feasible if minimum_value is None or t.value >= minimum_value]
    if not selectable:
        result = {"status": "NO_QUALIFYING_CANDIDATE", "config": asdict(base), "proposals": len(trials),
                  "best_value": best_value, "minimum_value": minimum_value,
                  "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in trials), "diagnostics": diagnostics}
        write_json(directory / "candidate.json", result)
        return result
    def distance(trial):
        defaults = asdict(selection_reference)
        return sum(((trial.user_attrs["config"][n] - defaults[n]) /
                    (max(dimensions[n].values()) - min(dimensions[n].values()) or 1)) ** 2 for n in declared[batch])
    chosen = min((t for t in selectable if t.value >= best_value - 0.01), key=lambda t: (distance(t), -t.value, t.number))
    result = {"status": "SEARCH_COMPLETED", "config": chosen.user_attrs["config"], "value": chosen.value,
              "best_value": best_value,
              "selected_trial": chosen.number, "proposals": len(study.trials),
              "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials),
              "metrics": chosen.user_attrs["metrics"], "diagnostics": diagnostics,
              "accuracy_claim": "none_without_independent_validation"}
    write_json(directory / "candidate.json", result)
    return result
