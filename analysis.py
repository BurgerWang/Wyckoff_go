"""Explicit application entrypoint: validated bars -> structure -> report.

The library path never downloads or writes; the CLI owns data and output I/O.
"""
import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from market_data import fetch, read_csv, validate_ohlcv
import pnf
import wyckoff


def analyze_frame(df: pd.DataFrame, symbol: str = "", kinds=None,
                  reversal: int = 3, *, config=None, benchmark=None,
                  market_context=None, trade_plan=None) -> dict:
    """Analyze one immutable snapshot without network, files, or plotting."""
    df = validate_ohlcv(df)
    symbol = symbol or df.attrs.get("symbol", "")
    df.attrs["symbol"] = symbol
    kinds = list(dict.fromkeys(["atr"] if kinds is None else kinds))
    if not kinds or any(kind not in ("atr", "pct", "trad") for kind in kinds):
        raise ValueError("scales must be one or more of atr, pct, trad")
    analyses = {}
    for kind in kinds:
        chart, label = pnf.build_config(symbol, df, kind, reversal=reversal)
        analyses[kind] = {
            "pnf": {"direction": chart.direction, "top": chart.top, "bottom": chart.bottom,
                    "columns": len(chart.columns), "scale": label, "reversal": chart.reversal},
            "wyckoff": wyckoff.analyze(chart.columns, chart.scale, df, chart.reversal,
                                      config=config, benchmark=benchmark,
                                      market_context=market_context, trade_plan=trade_plan),
        }
    return {"schema_version": 2, "symbol": symbol, "as_of": df.index[-1],
            "data": {"bars": len(df), "source": df.attrs.get("source", "provided"),
                     "adjusted": df.attrs.get("adjusted", "unspecified"),
                     "timeframe": df.attrs.get("timeframe", "provided"),
                     "excluded_empty_dates": df.attrs.get("excluded_empty_dates", []),
                     "warnings": df.attrs.get("warnings", [])}, "analyses": analyses}


def _serializable(value):
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_json(result: dict) -> str:
    return json.dumps(_serializable(result), ensure_ascii=False, indent=2, allow_nan=False)


def report(result: dict) -> str:
    lines = [f"标的: {result['symbol'] or '(未指定)'} | {result['data']['bars']} bars | {result['as_of']}"]
    lines.extend(f"行情提示: {warning}" for warning in result["data"].get("warnings", []))
    for kind, analysis in result["analyses"].items():
        p = analysis["pnf"]
        lines.extend(["", f"P&F 刻度: {kind} | {p['scale']}",
                      f"P&F 当前列: {p['direction'] or '尚无方向'} | {p['columns']} 列",
                      wyckoff.report(analysis["wyckoff"])])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wyckoff 量价阶段与 P&F 结构分析")
    parser.add_argument("symbol", nargs="?", default="", help="Yahoo ticker, or label for local CSV")
    parser.add_argument("--csv", type=Path, help="local Date/Datetime,Open,High,Low,Close,Volume CSV")
    parser.add_argument("--benchmark-csv", type=Path, help="completed same-timeframe market benchmark OHLCV CSV")
    parser.add_argument("--years", type=float, default=2)
    parser.add_argument("--timeframe", choices=("1d", "4h"), default="1d")
    parser.add_argument("--scale", choices=("atr", "pct", "trad"), action="append", dest="kinds")
    parser.add_argument("--reversal", type=int, default=3)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="save text/JSON report")
    parser.add_argument("--chart-dir", type=Path, help="optionally save P&F charts (requires plot extra)")
    args = parser.parse_args(argv)
    if not args.csv and not args.symbol:
        parser.error("provide a symbol or --csv")
    try:
        df = read_csv(args.csv, args.symbol) if args.csv else fetch(args.symbol, args.years, args.timeframe)
        benchmark = read_csv(args.benchmark_csv) if args.benchmark_csv else None
        result = analyze_frame(df, args.symbol, args.kinds, args.reversal, benchmark=benchmark)
        text = to_json(result) if args.format == "json" else report(result)
        if args.chart_dir:
            args.chart_dir.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", args.symbol).strip(".") or "instrument"
            for kind in result["analyses"]:
                chart, _ = pnf.build_config(args.symbol, df, kind, reversal=args.reversal)
                pnf.plot(chart, df, str(args.chart_dir / f"{stem}_pnf_{kind}.png"))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
