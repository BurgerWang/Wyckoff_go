"""Offline all-history causal stage replay; no downloads or rule tuning."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from analysis import analyze_frame, to_json
from market_data import read_csv
from research.history_data import coverage
from research.universe import PROFILES
from wyckoff_core.timeline import stage_timeline
from wyckoff_core.models import CORE_VERSION


def replay_history(history, output=None, *, verify_prefixes=True):
    history = Path(history)
    output = Path(output) if output else history / f"core-v{CORE_VERSION}-study"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((history / "manifest.json").read_text(encoding="utf-8"))
    report = {"created_at": pd.Timestamp.now(tz="UTC").isoformat(),
              "source_manifest": str(history / "manifest.json"), "request": manifest["request"],
              "scope": "all_observed_daily_bars_stage_only", "profiles": {},
              "core_version": None,
              "limitations": manifest["limitations"] + [
                  "Stage frequency is model output, not accuracy or profitable signal frequency.",
                  "Missing expected sessions reset the state machine; each continuous segment warms up independently.",
                  "No P&F targets, benchmark strength, three-law/readiness snapshots or trading backtest in this stage-only export."]}
    for market, profile in PROFILES.items():
        destination = output / market
        destination.mkdir(parents=True, exist_ok=True)
        results = report["profiles"][market] = {}
        for instrument in profile.instruments:
            symbol = instrument.symbol
            item = manifest["profiles"].get(market, {}).get(symbol, {})
            try:
                if item.get("status") == "failed" or not item.get("csv"):
                    raise ValueError(item.get("error", "missing history"))
                df = read_csv(item["csv"], symbol)
                df.attrs = item["attrs"]
                df.index = df.index.as_unit("ns")
                verified_coverage = coverage(df, profile, instrument, **{
                    k: manifest["request"][k] for k in ("start", "end")})
                if verified_coverage["continuous_segments"] != item["continuous_segments"]:
                    raise ValueError("history segments changed since manifest")
                rows, ledger, ranges, transitions, checks = [], [], [], [], []
                for number, segment in enumerate(item["continuous_segments"]):
                    observed = df.loc[segment["start"]:segment["end"]]
                    result = stage_timeline(observed, profile.config)
                    report["core_version"] = result["core_version"]
                    for row in result["rows"]:
                        phase = row["phase"]
                        hypotheses = phase.get("hypotheses", {})
                        rows.append({"Date": row["as_of"].strftime("%Y-%m-%d"), "segment": number,
                            "close": row["close"], "phase": phase["phase"], "regime": phase["regime"],
                            "status": phase["status"], "confidence": phase["confidence"],
                            "acc_phase": hypotheses.get("accumulation", {}).get("phase"),
                            "dist_phase": hypotheses.get("distribution", {}).get("phase"),
                            "range_id": row["range_id"], "range_lifecycle": row["range_lifecycle"],
                            "core_version": result["core_version"],
                            "phase_cycle_id": phase.get("cycle_id", row["range_id"]),
                            "parent_range_id": phase.get("parent_range_id", row["range_parent_id"]),
                            "prior_range_phase": json.dumps(phase.get("prior_range_phase"), ensure_ascii=False),
                            "stopping_candidate": to_json(row["stopping_candidate"]),
                            "range_top": row["range_top"], "range_bottom": row["range_bottom"],
                            "outside_closes": row["outside_closes"],
                            "notes": json.dumps(phase["notes"], ensure_ascii=False)})
                    ledger.extend(result["events"])
                    ranges.extend(result["ranges"])
                    transitions.extend(result["transitions"])
                    if verify_prefixes:
                        for i in sorted({min(20, len(observed) - 1), len(observed) // 2, len(observed) - 1}):
                            snapshot = analyze_frame(observed.iloc[:i + 1], config=profile.config)["analyses"]["atr"]["wyckoff"]
                            if snapshot["phase"] != result["rows"][i]["phase"]:
                                raise AssertionError(f"{symbol} segment {number} phase mismatch on {observed.index[i]}")
                            checks.append({"as_of": observed.index[i].strftime("%Y-%m-%d"), "segment": number, "status": "passed"})
                table = pd.DataFrame(rows)
                if len(table) != len(df):
                    raise AssertionError("each source bar must have exactly one prediction")
                csv = destination / f"{symbol}.stages.csv"
                table.to_csv(csv, index=False)
                (destination / f"{symbol}.evidence.json").write_text(to_json({"core_version": report["core_version"],
                    "events": ledger, "ranges": ranges,
                    "transitions": transitions, "config": asdict(profile.config)}) + "\n", encoding="utf-8")
                results[symbol] = {"status": "complete", "bars": len(table), "first": item["first"],
                    "last": item["last"], "span_years": item["span_years"], "csv": str(csv),
                    "source_status": item["status"], "missing_count": item["missing_count"],
                    "phase_counts": dict(Counter(table.phase)), "regime_counts": dict(Counter(table.regime)),
                    "expert_accuracy": "not_measured_by_this_runner", "prefix_checks": checks}
                print(f"{market} {symbol}: {len(table)} stages; prefixes={len(checks)}", flush=True)
            except Exception as exc:
                results[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                print(f"{market} {symbol} FAILED: {results[symbol]['error']}", flush=True)
            (output / "report.json").write_text(to_json(report) + "\n", encoding="utf-8")
    lines = ["# Long-history causal stage replay", "", "Daily model candidates; frequencies do not measure accuracy.", "",
             "| Market | Symbol | Bars | Years | A | B | C | D | E | ? | Status |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for market, items in report["profiles"].items():
        for symbol, item in items.items():
            counts = item.get("phase_counts", {})
            lines.append(f"| {market} | {symbol} | {item.get('bars', 0)} | {item.get('span_years', '—')} | "
                         + " | ".join(str(counts.get(p, 0)) for p in "ABCDE?") + f" | {item['status']} |")
    lines += ["", *[f"- {text}" for text in report["limitations"]]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("output/long-history/20260930"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = replay_history(args.history, args.output)
    return 2 if any(i["status"] == "failed" for items in report["profiles"].values() for i in items.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
