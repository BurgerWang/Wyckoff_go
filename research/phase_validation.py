"""Independent expert-case agreement, with abstentions counted as mismatches.

Curated retrospective charts are not a representative accuracy sample. No
confidence interval or population accuracy is inferred from clustered points.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from analysis import to_json

PHASES = tuple("ABCDE")


def directional_side(regime):
    if regime in ("accumulation", "reaccumulation", "markup"):
        return "accumulation"
    if regime in ("distribution", "redistribution", "markdown"):
        return "distribution"
    return "unknown"


def evaluate_labels(study, labels_path, output=None):
    study, labels_path = Path(study), Path(labels_path)
    reference = json.loads(labels_path.read_text(encoding="utf-8"))
    replay = json.loads((study / "report.json").read_text(encoding="utf-8"))
    sources = {s["id"]: s for s in reference["sources"]}
    tables, cases, excluded, seen = {}, [], list(reference.get("excluded_candidates", [])), set()
    for label in reference["labels"]:
        reason = None
        source = sources.get(label.get("source_id"), {})
        date = label.get("as_of") or (label.get("start") if label.get("start") == label.get("end") else None)
        market, symbol = label.get("market"), label.get("symbol")
        allowed_market = {"us_equities": "US equity", "binance_spot": "Binance spot USDT"}.get(market)
        if (not label.get("eligible") or label.get("independence") != "external_human"
                or label.get("adjudication") not in ("explicit", "explicit_chart_interior")
                or label.get("phase") not in PHASES or not date):
            reason = "not an explicit independently authored single-session phase label"
        elif (source.get("original_symbol") != symbol or source.get("original_market") != allowed_market
              or source.get("original_timeframe") != "1d" or label.get("timeframe") != "1d"
              or not source.get("author") or not str(source.get("source_url", "")).startswith("https://")
              or not label.get("published_at") or not (label.get("cycle_id") or label.get("case_id"))):
            reason = "source security/market/timeframe/provenance does not support this target label"
        key = (market, symbol, date)
        if key in seen:
            reason = "duplicate target session; cannot inflate denominator"
        item = replay["profiles"].get(market, {}).get(symbol, {})
        if not reason and item.get("status") != "complete":
            reason = "target stage replay unavailable"
        if not reason:
            table_key = (market, symbol)
            if table_key not in tables:
                table = pd.read_csv(item["csv"], dtype={"Date": str}, keep_default_na=False)
                if table.Date.duplicated().any():
                    raise ValueError("duplicate prediction sessions")
                tables[table_key] = table.set_index("Date")
            table = tables[table_key]
            if date not in table.index:
                reason = "expert session not present; no nearest-date substitution"
        if reason:
            excluded.append({"id": label["id"], "reason": reason})
            continue
        seen.add(key)
        prediction = table.loc[date]
        phase = prediction["phase"] if prediction["phase"] in PHASES else "?"
        if prediction["status"] in ("insufficient_data", "ambiguous", "insufficient_evidence"):
            phase = "?"
        expected_side = directional_side(label.get("regime"))
        predicted_side = directional_side(prediction["regime"])
        cases.append({"id": label["id"], "market": market, "symbol": symbol, "as_of": date,
            "expected": label["phase"], "predicted": phase, "correct": phase == label["phase"],
            "expected_side": expected_side, "predicted_side": predicted_side,
            "joint_correct": (expected_side != "unknown" and predicted_side == expected_side
                              and phase == label["phase"]),
            "regime": prediction["regime"], "status": prediction["status"],
            "cycle_id": label.get("cycle_id") or label["case_id"], "source_id": label["source_id"],
            "source_url": source["source_url"], "retrospective": label.get("retrospective", label.get("hindsight")),
            "notes": prediction.get("notes", "")})
    metrics = {}
    for market in replay["profiles"]:
        subset = [c for c in cases if c["market"] == market]
        n, correct = len(subset), sum(c["correct"] for c in subset)
        classified = sum(c["predicted"] in PHASES for c in subset)
        directional = [c for c in subset if c["expected_side"] != "unknown"]
        joint_correct = sum(c["joint_correct"] for c in directional)
        confusion = {actual: {pred: 0 for pred in (*PHASES, "?")} for actual in PHASES}
        for case in subset:
            confusion[case["expected"]][case["predicted"]] += 1
        per_phase = {}
        for phase in PHASES:
            tp = confusion[phase][phase]
            support = sum(confusion[phase].values())
            predicted = sum(confusion[p][phase] for p in PHASES)
            per_phase[phase] = {"support": support, "precision": tp / predicted if predicted else None,
                "recall": tp / support if support else None,
                "f1": 2 * tp / (support + predicted) if support else None}
        f1s = [v["f1"] for v in per_phase.values() if v["f1"] is not None]
        metrics[market] = {"status": "measured_expert_case_agreement" if n else "unavailable_no_eligible_labels",
            "labeled_points": n, "cycles": len({c["cycle_id"] for c in subset}),
            "symbols": sorted({c["symbol"] for c in subset}), "correct": correct,
            "agreement_including_abstentions": correct / n if n else None,
            "classified_points": classified, "abstentions": n - classified,
            "selective_agreement": correct / classified if classified else None,
            "directional_reference_points": len(directional), "joint_correct": joint_correct,
            "joint_phase_side_agreement": joint_correct / len(directional) if directional else None,
            "macro_f1_observed_phases": sum(f1s) / len(f1s) if f1s else None,
            "confusion": confusion, "per_phase": per_phase,
            "full_history_accuracy": None,
            "confidence_interval": None,
            "inference_limit": "Purposive, retrospective, clustered chart points; not independent trials or a representative population sample."}
    result = {"label_file": str(labels_path), "study_report": str(study / "report.json"),
        "sampling": reference["sampling"], "metrics": metrics, "cases": cases, "excluded": excluded,
        "limitations": ["No parameter selection against these labels; comparison is a retrospective development diagnostic.",
            "Human phases use later chart context; daily causal rules need not identify the same phase before its evidence confirms.",
            "Current Yahoo adjustment vintage differs from original historical charts; no point-in-time price/volume archive is claimed.",
            "Teaching cases and current stock selection have selection bias; unlabelled bars have unknown correctness.",
            "Macro F1 here describes only supported reference phases; unavailable phase precision is not zero."]}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation.json").write_text(to_json(result) + "\n", encoding="utf-8")
        lines = ["# Independent expert-case agreement", "", "This is not full-history stage accuracy.", ""]
        for market, m in metrics.items():
            value = f"{m['agreement_including_abstentions']:.1%}" if m["labeled_points"] else "unavailable"
            lines += [f"- {market}: {m['correct']}/{m['labeled_points']} = {value}; {m['cycles']} cycles, "
                      f"{len(m['symbols'])} symbols, {m['abstentions']} abstentions."]
            joint = f"{m['joint_phase_side_agreement']:.1%}" if m["directional_reference_points"] else "unavailable"
            lines += [f"  Phase + directional side: {m['joint_correct']}/{m['directional_reference_points']} = {joint}."]
        lines += ["", "| Symbol | As of | Expert phase / side | Core phase / side | Phase match | Joint match | Source |",
                  "|---|---|---|---|---|---|---|"]
        for c in cases:
            lines.append(f"| {c['symbol']} | {c['as_of']} | {c['expected']} / {c['expected_side']} | "
                         f"{c['predicted']} / {c['predicted_side']} | {c['correct']} | {c['joint_correct']} | "
                         f"[{c['source_id']}]({c['source_url']}) |")
        lines += ["", *[f"- {x}" for x in result["limitations"]],
                  "- Confidence intervals and full-history accuracy are unavailable for this purposive clustered sample."]
        (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("output/long-history/20260930/study"))
    parser.add_argument("--labels", type=Path, default=Path("research/labels/independent_phase_labels.json"))
    parser.add_argument("--output", type=Path, default=Path("output/long-history/20260930/validation"))
    args = parser.parse_args(argv)
    result = evaluate_labels(args.study, args.labels, args.output)
    print(to_json(result["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
