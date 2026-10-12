"""Offline per-symbol tuning: inspect, diagnose and qualified nested search."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from research.tuning.data import load_history, replay
from research.tuning.labels import group_information_bounds, load_labels
from research.tuning.metrics import score
from research.tuning.policy import VALIDATION_POLICY
from research.tuning.preflight import evaluate_reference_capacity
from research.tuning.profiles import instrument_for, resolve_config
from research.tuning.runner import digest, optimize_batch, run_lock, write_json
from research.tuning.space import (BATCHES, JOINT_BATCHES, JOINT_SEARCH_VERSION, SPACE_VERSION,
                                   common_warmup, sample_configs, search_space)
from research.tuning.splits import build_split_plan
from research.universe import PROFILES
from wyckoff_core.models import CoreConfig

ROOT = Path(__file__).resolve().parents[2]


def implementation_id():
    files = sorted((ROOT / "wyckoff_core").glob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    files += [ROOT / "market_data.py", ROOT / "research/history_data.py", ROOT / "research/universe.py"]
    import numpy
    import pandas
    import optuna
    return digest({"files": {str(path.relative_to(ROOT)): path.read_text() for path in files},
                   "versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "optuna": optuna.__version__}})


def _subset(reference, dates):
    allowed = set(dates)
    labels = [label for label in reference["labels"] if label["as_of"] in allowed]
    return {**reference, "labels": labels, "diagnostic_only": any(label.get("diagnostic_only") for label in labels)}


def _qualified_reference(reference, frame, item, market):
    observed = {d.strftime("%Y-%m-%d") for d in frame.index}
    if any(label["as_of"] not in observed for label in reference["labels"]):
        raise ValueError("label session is absent from exact source history")
    for label in reference["labels"]:
        if label.get("group_information_start") is not None or label.get("group_information_end") is not None:
            a, b = group_information_bounds(label)
            if a not in observed or b not in observed or not any(s["start"] <= a <= b <= s["end"] for s in item["continuous_segments"]):
                raise ValueError("complete group information is outside verified source or continuous segment")
    eligible = set()
    for segment in item["continuous_segments"]:
        block = frame.loc[segment["start"]:segment["end"]]
        eligible.update(d.strftime("%Y-%m-%d") for d in block.iloc[common_warmup(market) - 1:].index
                        if block.loc[d, "Volume"] > 0)
    labels = [{**label, **({"qualification_issue": "common_warmup_or_zero_volume"}
                if label["as_of"] not in eligible else {})} for label in reference["labels"]]
    return {**reference, "labels": labels}


def diagnose(history, market, symbol, output, *, samples=32, seed=20260930):
    implementation = implementation_id()
    frame, item = load_history(history, market, symbol)
    output = Path(output)
    if (output / "diagnostic.json").exists():
        raise ValueError("diagnostic output exists; choose a new directory")
    started = time.perf_counter()
    baseline = replay(frame, item, resolve_config(market, symbol))
    baseline_states = [(r["phase"]["phase"], r["phase"]["regime"]) for r in baseline]
    trials = []
    for number, config in enumerate(sample_configs(market, samples, seed)):
        clock = time.perf_counter()
        rows = replay(frame, item, config)
        states = [(r["phase"]["phase"], r["phase"]["regime"]) for r in rows]
        trials.append({"sample": number, "params": asdict(config), "seconds": time.perf_counter() - clock,
            "phase_counts": dict(Counter(p for p, _ in states)),
            "joint_disagreements_with_default": sum(a != b for a, b in zip(states, baseline_states)),
            "bars": len(rows)})
    result = {"market": market, "symbol": symbol, "timeframe": "1d", "bars": len(frame),
        "first": item["first"], "last": item["last"], "span_years": item["span_years"],
        "status": "SHORT_HISTORY" if symbol == "SPCX" else "LABELS_INSUFFICIENT",
        "mode": "unlabeled_sensitivity", "accuracy": None, "best_params": None,
        "space_version": SPACE_VERSION, "seed": seed, "implementation": implementation,
        "history": str(Path(history).resolve()), "source_attrs": item["attrs"],
        "continuous_segments": item["continuous_segments"],
        "data_identity": digest({"attrs": item["attrs"], "dates": frame.index.tolist(), "values": frame.to_numpy().tolist()}),
        "default_params": asdict(CoreConfig()), "default_phase_counts": dict(Counter(p for p, _ in baseline_states)),
        "samples": trials, "seconds": time.perf_counter() - started,
        "limitation": "Parameter response and default disagreement are not accuracy; no configuration selected."}
    if implementation_id() != implementation:
        raise ValueError("implementation changed during diagnosis; results not published")
    write_json(output / "diagnostic.json", result)
    (output / "summary.md").write_text(f"# {market}/{symbol} parameter sensitivity\n\n"
        f"{len(frame)} real daily bars, {samples} samples plus original default.\n\n"
        "No independent accuracy labels, no best parameters and no production replacement.\n")
    return result


def supervised(history, market, symbol, labels, output, *, label_mode="asof", trials=128,
               batches=tuple(BATCHES), joint_batches=tuple(JOINT_BATCHES), joint_trials=None,
               seed=20260930, holdout_start="2024-10-01", holdout_end="2026-09-30"):
    instrument_for(market, symbol)
    if not batches or len(set(batches)) != len(batches) or any(b not in BATCHES for b in batches):
        raise ValueError("batches must be a nonempty, ordered list without duplicates")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 4096:
        raise ValueError("trials must be 1..4096")
    if len(set(joint_batches)) != len(joint_batches) or any(b not in JOINT_BATCHES for b in joint_batches):
        raise ValueError("joint batches must be an ordered list without duplicates")
    joint_trials = trials if joint_trials is None else joint_trials
    if isinstance(joint_trials, bool) or not isinstance(joint_trials, int) or not 1 <= joint_trials <= 4096:
        raise ValueError("joint trials must be 1..4096")
    frame, item = load_history(history, market, symbol)
    reference = _qualified_reference(load_labels(labels, market=market, symbol=symbol, label_mode=label_mode),
                                     frame, item, market)
    plan = build_split_plan(reference, holdout_start=holdout_start, holdout_end=holdout_end)
    capacity = evaluate_reference_capacity(_subset(reference, plan["holdout_dates"]))
    output = Path(output)
    if symbol == "SPCX" or plan["status"] != "READY" or not capacity["eligible"]:
        result = {"status": "SHORT_HISTORY" if symbol == "SPCX" else
                  "LABELS_INSUFFICIENT" if plan["status"] != "READY" else "INSUFFICIENT_REFERENCE_CAPACITY",
                  "market": market, "symbol": symbol, "split_plan": plan,
                  "reference_capacity": capacity,
                  "accuracy": None, "best_params": None, "optimization_started": False}
        if not (output / "run.json").exists():
            write_json(output / "qualification.json", result)
        return result
    from research.tuning.holdout import check_unused
    check_unused(_subset(reference, plan["holdout_dates"]))
    context = {"market": market, "symbol": symbol, "label_mode": label_mode, "batches": list(batches),
        "joint_batches": list(joint_batches), "joint_trials_per_sampler": joint_trials,
        "joint_search_version": JOINT_SEARCH_VERSION,
        "trials_per_batch_per_sampler": trials, "seed": seed, "space_version": SPACE_VERSION,
        "implementation": implementation_id(), "data": digest({"attrs": item["attrs"],
           "dates": frame.index.tolist(), "values": frame.to_numpy().tolist()}),
        "labels": digest(reference), "splits": digest(plan), "history": str(Path(history).resolve()),
        "labels_path": str(Path(labels).resolve()), "holdout_start": holdout_start, "holdout_end": holdout_end,
        "validation_policy": VALIDATION_POLICY}
    signature = digest(context)
    with run_lock(output):
        manifest = output / "run.json"
        if manifest.exists() and json.loads(manifest.read_text())["signature"] != signature:
            raise ValueError("run input changed; resume requires identical data/code/labels/budget")
        if (output / "holdout-use.json").exists():
            raise ValueError("holdout has been opened; search cannot be resumed or candidate reselected")
        write_json(manifest, {"signature": signature, "context": context})
        write_json(output / "split-plan.json", plan)
        completed = []
        schedules = [(f"outer-{i}", fold["inner"], fold["train_dates"], fold["test_dates"])
                     for i, fold in enumerate(plan["outer"])]
        schedules.append(("final-development", plan["final_inner"], plan["development_dates"], []))
        for fold_name, inner, training_dates, test_dates in schedules:
            validation_dates = sorted({d for fold in inner for d in fold["test_dates"]})
            validation_reference = _subset(reference, validation_dates)
            end = max(training_dates)
            default_rows = replay(frame, item, CoreConfig(), end)
            default_metric = score(validation_reference, default_rows)
            default_folds = [score(_subset(reference, fold["test_dates"]), default_rows) for fold in inner]
            for sampler_name in ("tpe", "random"):
                selected = CoreConfig()
                selected_metric = default_metric
                chain = []
                joint_chain = []
                for batch in (*batches, *joint_batches):
                    def objective(config):
                        rows = replay(frame, item, config, end)
                        metric = score(validation_reference, rows)
                        constraints = {"pooled_default_regression": default_metric["joint_macro_f1"] - metric["joint_macro_f1"]}
                        if batch in BATCHES:
                            constraints["pooled_previous_batch_regression"] = selected_metric["joint_macro_f1"] - metric["joint_macro_f1"]
                        for i, fold in enumerate(inner):
                            part = score(_subset(reference, fold["test_dates"]), rows)
                            constraints[f"fold_{i}_regression"] = default_folds[i]["joint_macro_f1"] - part["joint_macro_f1"] - 0.05
                            for state, values in default_folds[i]["per_class"].items():
                                if state.startswith(("C/", "E/")) and values["recall"] is not None:
                                    constraints[f"fold_{i}_{state}_recall"] = values["recall"] - part["per_class"][state]["recall"] - 0.05
                        return {"score": metric["joint_macro_f1"], "constraints": constraints, "metrics": metric}
                    result = optimize_batch(output / fold_name / sampler_name / batch, market=market,
                        base=selected, batch=batch, sampler_name=sampler_name,
                        total_trials=joint_trials if batch in JOINT_BATCHES else trials,
                        seed=seed, identity={"run": signature, "fold": fold_name}, objective=objective,
                        selection_reference=CoreConfig(),
                        minimum_value=selected_metric["joint_macro_f1"] + 0.01 + 1e-12)
                    # Acceptance uses the chosen configuration's score. Joint
                    # proposals change interacting fields atomically; their
                    # individual intermediate steps are never evaluated.
                    if (result["status"] == "SEARCH_COMPLETED"
                            and result["value"] > selected_metric["joint_macro_f1"] + 0.01):
                        selected = CoreConfig(**result["config"])
                        selected_metric = result["metrics"]
                    (joint_chain if batch in JOINT_BATCHES else chain).append(result)
                summary = {"fold": fold_name, "sampler": sampler_name, "params": asdict(selected),
                    "development_metrics": selected_metric, "default_metrics": default_metric,
                    "batches": chain, "joint_refinement": joint_chain}
                if test_dates:
                    summary["outer_metrics"] = score(_subset(reference, test_dates), replay(frame, item, selected, max(test_dates)))
                    summary["outer_default_metrics"] = score(_subset(reference, test_dates), replay(frame, item, CoreConfig(), max(test_dates)))
                completed.append(summary)
                write_json(output / "results.json", {"status": "IN_PROGRESS", "studies": completed})
        finals = [record for record in completed if record["fold"] == "final-development"]
        chosen = max(finals, key=lambda r: r["development_metrics"]["joint_macro_f1"])
        candidate = {"schema_version": 1, "status": "CANDIDATE_ONLY", "market": market, "symbol": symbol,
            "timeframe": "1d", "label_mode": label_mode, "params": chosen["params"], "run_signature": signature,
            "training_end": max(plan["development_dates"]),
            "label_information_end": max(group_information_bounds(label)[1] for label in reference["labels"]
                                         if label["as_of"] in set(plan["development_dates"])),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "accuracy_claim": "not_validated_on_holdout", "holdout_evaluated": False}
        write_json(output / "candidate.json", candidate)
        results = {"status": "SEARCH_COMPLETED", "studies": completed, "candidate": candidate}
        write_json(output / "selection.json", {"run_signature": signature, "candidate_identity": digest(candidate),
                                               "sampler": chosen["sampler"], "results_identity": digest(results)})
        write_json(output / "results.json", results)
        return {"status": "SEARCH_COMPLETED", "candidate": candidate, "optimization_started": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="show all31 parameter ranges, defaults and batches")
    inspect.add_argument("--market", choices=tuple(PROFILES), default="us_equities")
    inspect.add_argument("--symbol")
    inspect.add_argument("--output", type=Path)
    preflight = commands.add_parser("preflight", help="check evaluation metadata before search or holdout consumption")
    source = preflight.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack", type=Path)
    source.add_argument("--packs-root", type=Path)
    source.add_argument("--labels", type=Path)
    source.add_argument("--joint-packs", type=Path, nargs=2,
                        help="check one human-cycle and one window pack together, without scoring")
    preflight.add_argument("--market", choices=tuple(PROFILES), default="us_equities")
    preflight.add_argument("--symbol")
    preflight.add_argument("--label-mode", choices=("asof", "retrospective"), default="asof")
    preflight.add_argument("--history", type=Path, default=ROOT / "output/long-history/20260930")
    preflight.add_argument("--holdout-start", default="2024-10-01")
    preflight.add_argument("--holdout-end", default="2026-09-30")
    preflight.add_argument("--output", type=Path)
    quality = commands.add_parser("label-quality", help="compare two independent blind human reviews")
    quality.add_argument("--pack", type=Path, required=True)
    quality.add_argument("--review-a", type=Path, required=True)
    quality.add_argument("--review-b", type=Path, required=True)
    quality.add_argument("--output", type=Path)
    resume = commands.add_parser("resume", help="continue completed-trial checkpoints with original cumulative budget")
    resume.add_argument("--run", type=Path, required=True)
    validate = commands.add_parser("validate", help="validate frozen candidate once on registered holdout")
    validate.add_argument("--run", type=Path, required=True)
    export = commands.add_parser("export", help="export only an unchanged validated labeled-holdout profile")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--effective-from")
    annotate = commands.add_parser("prepare-labels", help="prepare independent OHLCV-only blank human annotation packs")
    annotate.add_argument("--market", choices=tuple(PROFILES), default="us_equities")
    annotate.add_argument("--symbol")
    annotate.add_argument("--all", action="store_true")
    annotate.add_argument("--history", type=Path, default=ROOT / "output/long-history/20260930")
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--label-mode", choices=("asof", "retrospective"), default="asof")
    annotate.add_argument("--rubric", type=Path, help="freeze this human annotation rubric in a new pack")
    annotate.add_argument("--sampling-version", choices=("window-v1", "sparse-window-v2"), default="window-v1")
    annotate.add_argument("--development-start", help="earliest development exposure for sparse-window-v2")
    annotate.add_argument("--cycle-bounds", type=Path, help="human-defined cycle metadata to exclude from sparse windows")
    annotate.add_argument("--seed", type=int, default=20260930)
    annotate.add_argument("--development-blocks", type=int, default=24)
    annotate.add_argument("--holdout-blocks", type=int, default=12)
    annotate.add_argument("--context-bars", type=int, default=100)
    annotate.add_argument("--holdout-start", default="2024-10-01")
    annotate.add_argument("--holdout-end", default="2026-09-30")
    review = commands.add_parser("import-labels", help="import completed independent human reviews without inventing labels")
    review.add_argument("--pack", type=Path, required=True)
    review.add_argument("--reviewed", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    cycle = commands.add_parser("prepare-cycles", help="prepare blind charts from actual human cycle registration")
    cycle.add_argument("--history", type=Path, default=ROOT / "output/long-history/20260930")
    cycle.add_argument("--registration", type=Path, required=True)
    cycle.add_argument("--rubric", type=Path, required=True)
    cycle.add_argument("--output", type=Path, required=True)
    consensus = commands.add_parser("import-consensus", help="import both original blind reviews and optional independent adjudication")
    consensus.add_argument("--pack", type=Path, required=True)
    consensus.add_argument("--review-a", type=Path, required=True)
    consensus.add_argument("--review-b", type=Path, required=True)
    consensus.add_argument("--adjudication", type=Path)
    consensus.add_argument("--output", type=Path, required=True)
    for name in ("diagnose", "run"):
        command = commands.add_parser(name)
        command.add_argument("--market", choices=tuple(PROFILES), default="us_equities")
        command.add_argument("--symbol")
        command.add_argument("--history", type=Path, default=ROOT / "output/long-history/20260930")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=20260930)
        if name == "diagnose":
            command.add_argument("--all", action="store_true")
            command.add_argument("--samples", type=int, default=32)
        else:
            command.add_argument("--labels", type=Path, required=True)
            command.add_argument("--label-mode", choices=("asof", "retrospective"), default="asof")
            command.add_argument("--trials", type=int, default=128)
            command.add_argument("--batches", nargs="+", choices=tuple(BATCHES), default=list(BATCHES))
            command.add_argument("--joint-batches", nargs="*", choices=tuple(JOINT_BATCHES),
                                 default=list(JOINT_BATCHES), help="joint groups; empty list disables refinement")
            command.add_argument("--joint-trials", type=int, help="budget per joint group and sampler; defaults to --trials")
            command.add_argument("--holdout-start", default="2024-10-01")
            command.add_argument("--holdout-end", default="2026-09-30")
    args = parser.parse_args(argv)
    try:
        if args.command == "label-quality":
            from research.tuning.annotation import _load_manifest
            from research.tuning.label_quality import agreement_report
            manifest = _load_manifest(args.pack)
            result = agreement_report(json.loads(args.review_a.read_text()),
                json.loads(args.review_b.read_text()), manifest=manifest)
            if args.output:
                if args.output.exists() and json.loads(args.output.read_text()) != result:
                    raise ValueError("refusing overwrite of a different human agreement report")
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight":
            if args.joint_packs:
                from research.tuning.annotation import _load_manifest
                from research.tuning.preflight import evaluate_annotation_joint_capacity
                from research.tuning import holdout
                manifests = [_load_manifest(pack) for pack in args.joint_packs]
                source = manifests[0]["source"]
                shared = holdout._load(holdout.REGISTRY / source["market"] / source["symbol"] / "uses.json",
                                      source["market"], source["symbol"])
                consumed = [interval for use in shared["uses"] for interval in use["information_intervals"]]
                result = {"eligible": False, "metadata_only": True,
                    "reports": [evaluate_annotation_joint_capacity(manifests, role=role,
                                consumed_intervals=consumed) for role in ("development", "holdout")],
                    "predictions_evaluated": False, "holdout_consumed": False}
            elif args.labels:
                if not args.symbol:
                    raise ValueError("--symbol is required with --labels")
                frame, item = load_history(args.history, args.market, args.symbol)
                reference = _qualified_reference(load_labels(args.labels, market=args.market,
                    symbol=args.symbol, label_mode=args.label_mode), frame, item, args.market)
                plan = build_split_plan(reference, holdout_start=args.holdout_start, holdout_end=args.holdout_end)
                capacity = evaluate_reference_capacity(_subset(reference, plan["holdout_dates"]))
                result = {"split_plan": plan, "reference_capacity": capacity,
                          "eligible": plan["status"] == "READY" and capacity["eligible"],
                          "predictions_evaluated": False, "holdout_consumed": False}
            else:
                from research.tuning.annotation import _load_manifest
                from research.tuning.preflight import evaluate_annotation_capacity
                packs = ([args.pack] if args.pack else [args.packs_root / market / i.symbol
                    for market, profile in PROFILES.items() for i in profile.instruments])
                reports = []
                for pack in packs:
                    manifest = _load_manifest(pack)
                    settings, source = manifest["settings"], manifest["source"]
                    frame, _ = load_history(source["history"], source["market"], source["symbol"])
                    dates = frame.index.strftime("%Y-%m-%d")
                    bars = int(sum((dates >= settings["holdout_start"]) & (dates < settings["holdout_end"])))
                    reports.append(evaluate_annotation_capacity(manifest, holdout_bar_count=bars))
                result = {"eligible": False, "metadata_only": True, "reports": reports,
                          "predictions_evaluated": False, "holdout_consumed": False}
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["eligible"] else 2
        if args.command == "prepare-cycles":
            from .cycles import prepare_cycle_annotation_pack
            from .annotation import write_annotation_index
            with run_lock(args.output):
                result = prepare_cycle_annotation_pack(args.history, args.registration, args.output, rubric_path=args.rubric)
                write_annotation_index(args.output)
            print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "accuracy": None}))
            return 0
        if args.command == "import-consensus":
            from .annotation import import_consensus_annotations
            result = import_consensus_annotations(args.pack, args.review_a, args.review_b, args.output, adjudication=args.adjudication)
            print(json.dumps(result["import_summary"], ensure_ascii=False))
            return 0
        if args.command == "prepare-labels":
            from research.tuning.annotation import prepare_annotation_pack, write_annotation_index, write_annotation_overview
            if args.all == bool(args.symbol):
                raise ValueError("provide exactly one of --all or --symbol")
            cycle_bounds = []
            if args.cycle_bounds:
                if args.all or args.sampling_version != "sparse-window-v2":
                    raise ValueError("cycle bounds require a single-symbol sparse-window-v2 pack")
                cycle_metadata = json.loads(args.cycle_bounds.read_text())
                if not isinstance(cycle_metadata, dict):
                    raise ValueError("cycle-bound metadata must be a source-identified object")
                if (cycle_metadata.get("market"), cycle_metadata.get("symbol"), cycle_metadata.get("timeframe")) != (args.market, args.symbol, "1d"):
                    raise ValueError("cycle-bound metadata source identity mismatch")
                cycle_bounds = cycle_metadata["human_cycle_bounds"]
                if not isinstance(cycle_bounds, list):
                    raise ValueError("human_cycle_bounds must be a list of complete intervals")
            targets = ([(market, i.symbol) for market, profile in PROFILES.items() for i in profile.instruments]
                       if args.all else [(args.market, args.symbol)])
            summary = []
            for market, symbol in targets:
                destination = args.output / market / symbol if args.all else args.output
                with run_lock(destination):
                    result = prepare_annotation_pack(args.history, market, symbol, destination,
                        label_mode=args.label_mode, seed=args.seed, development_blocks=args.development_blocks,
                        holdout_blocks=args.holdout_blocks, context_bars=args.context_bars,
                        holdout_start=args.holdout_start, holdout_end=args.holdout_end, rubric_path=args.rubric,
                        sampling_version=args.sampling_version, development_start=args.development_start,
                        human_cycle_bounds=cycle_bounds)
                    write_annotation_index(destination)
                summary.append({"market": market, "symbol": symbol, "status": result["status"],
                                "cases": len(result["cases"]), "sampling": result["sampling"]})
                print(json.dumps({"market": market, "symbol": symbol, "status": result["status"],
                                  "cases": len(result["cases"]),
                                  "support": result["sampling"]["support"]}, ensure_ascii=False), flush=True)
            if args.all:
                write_json(args.output / "summary.json", {"label_mode": args.label_mode,
                    "status": "UNLABELED_ANNOTATION_PACKS", "completed": summary, "accuracy": None})
                write_annotation_overview(args.output, summary)
            return 0
        if args.command == "import-labels":
            from research.tuning.annotation import import_annotations
            result = import_annotations(args.pack, args.reviewed, args.output)
            print(json.dumps(result["import_summary"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate":
            from research.tuning.validation import validate_run
            result = validate_run(args.run)
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ("default_metrics", "candidate_metrics")}, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "VALIDATED_ON_LABELED_HOLDOUT" else 2
        if args.command == "export":
            from research.tuning.validation import export_profile
            result = export_profile(args.run, args.output, effective_from=args.effective_from)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "resume":
            context = json.loads((args.run / "run.json").read_text())["context"]
            result = supervised(context["history"], context["market"], context["symbol"], context["labels_path"],
                args.run, label_mode=context["label_mode"], trials=context["trials_per_batch_per_sampler"],
                batches=context["batches"], seed=context["seed"], holdout_start=context["holdout_start"],
                holdout_end=context["holdout_end"], joint_batches=context.get("joint_batches", ()),
                joint_trials=context.get("joint_trials_per_sampler"))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "SEARCH_COMPLETED" else 2
        if args.command == "inspect":
            if args.symbol:
                instrument_for(args.market, args.symbol)
            space = search_space(args.market)
            result = {"market": args.market, "symbol": args.symbol, "space_version": SPACE_VERSION,
                "joint_search_version": JOINT_SEARCH_VERSION,
                "joint_batches": {name: list(fields) for name, fields in JOINT_BATCHES.items()},
                "parameters": {n: {"default": asdict(CoreConfig())[n], "values": list(d.values()),
                    "step": d.step, "batch": next(b for b, names in BATCHES.items() if n in names)} for n, d in space.items()}}
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "diagnose":
            if args.all == bool(args.symbol):
                raise ValueError("provide exactly one of --all or --symbol")
            targets = [(market, i.symbol) for market, profile in PROFILES.items() for i in profile.instruments] if args.all else [(args.market, args.symbol)]
            summary = []
            for market, symbol in targets:
                destination = args.output / market / symbol if args.all else args.output
                with run_lock(destination):
                    item = diagnose(args.history, market, symbol, destination, samples=args.samples, seed=args.seed)
                summary.append({key: item[key] for key in ("market", "symbol", "status", "bars", "seconds")})
                print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
                if args.all:
                    write_json(args.output / "summary.json", {"accuracy": None, "completed": summary})
            return 0
        if not args.symbol:
            raise ValueError("--symbol is required for an independent supervised study")
        result = supervised(args.history, args.market, args.symbol, args.labels, args.output,
            label_mode=args.label_mode, trials=args.trials, batches=args.batches, seed=args.seed,
            holdout_start=args.holdout_start, holdout_end=args.holdout_end,
            joint_batches=args.joint_batches, joint_trials=args.joint_trials)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "SEARCH_COMPLETED" else 2
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"tuning failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
