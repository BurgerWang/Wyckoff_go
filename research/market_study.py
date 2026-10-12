"""Separate-market real-data corpus and causal snapshot replay.

No parameter search, trades, labels manufactured from these same rules, or
pooled cross-market accuracy. Files checkpoint each symbol for resumption.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import pandas as pd

from analysis import analyze_frame, to_json, _serializable
from market_data import normalize_download, read_csv, validate_ohlcv
from wyckoff_core.stages import range_events
from research.universe import PROFILES, Instrument, ResearchProfile, verify_identity


LIMITATIONS = [
    "阶段、事件、案例由规则生成，没有独立专家标签；本报告不计算阶段准确率或交易收益。",
    "两个市场采用分别保存的默认 core 配置，尚未分别校准；不得合并 USD/USDT 价格或股票/币成交量。",
    "股票是固定的当前研究池，存在选择与存活偏差；不能代表全部历史股票。",
    "股票行情为下载时的 Yahoo 复权版本，并非历史时点公司行动数据库；前缀回放保证计算时序，不证明数据本身是 point-in-time。",
    "日线 bar 完成后才可使用：股票按交易会话日期，币按 UTC 日界。相同 bar 数对应不同日历长度。",
    "SPCX 只接受 SpaceX 上市起点后的数据；短历史标的覆盖不足会单独显示。",
]


def date_window(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tz is not None or end.tz is not None or start != start.normalize() or end != end.normalize() or start >= end:
        raise ValueError("start/end must be naive ISO dates, start < end; end is exclusive")
    return start, end


def fetch_equity(instrument: Instrument, start, end) -> pd.DataFrame:
    import yfinance as yf
    start, end = date_window(start, end)
    identity = verify_identity(instrument, yf.Ticker(instrument.symbol).get_info()) if instrument.identity_tokens else {"status": "not_required"}
    effective_start = max(start, pd.Timestamp(instrument.listed_since)) if instrument.listed_since else start
    if effective_start >= end:
        raise ValueError(f"{instrument.symbol}: requested interval precedes verified listing")
    raw = yf.download(instrument.symbol, start=effective_start.strftime("%Y-%m-%d"),
                      end=end.strftime("%Y-%m-%d"), interval="1d", auto_adjust=True,
                      progress=False, threads=False, timeout=20)
    result = normalize_download(raw, instrument.symbol)
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    if (result.index < effective_start).any() or (result.index >= end).any():
        raise ValueError("provider returned bars outside the authorized security/date interval")
    result.attrs.update(timeframe="1d", market_type="cash", price_currency="USD",
                        volume_unit="shares", session_timezone="America/New_York",
                        identity=identity, requested_start=start.isoformat(), requested_end=end.isoformat(),
                        effective_start=effective_start.isoformat(), fetched_at=pd.Timestamp.now(tz="UTC").isoformat())
    return validate_ohlcv(result)


def load_data(profile: ResearchProfile, instrument: Instrument, start, end, directory: Path, *, refresh=False):
    """Cache only exact requests, retaining attrs in a readable sidecar."""
    directory.mkdir(parents=True, exist_ok=True)
    csv, metadata = directory / f"{instrument.symbol}.csv", directory / f"{instrument.symbol}.json"
    request = {"profile": profile.id, "symbol": instrument.symbol, "start": str(start), "end": str(end)}
    if not refresh and csv.exists() and metadata.exists():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved.get("request") == request:
            frame = read_csv(csv, instrument.symbol)
            frame.attrs = saved["attrs"]
            if instrument.identity_tokens:
                identity = frame.attrs.get("identity", {})
                verify_identity(instrument, {"symbol": identity.get("symbol"), "longName": identity.get("name"),
                                             "quoteType": identity.get("quote_type"), "cusip": identity.get("provider_cusip")})
                if frame.index[0] < pd.Timestamp(instrument.listed_since):
                    raise ValueError("cached SPCX includes pre-SpaceX history")
            return validate_profile_data(frame, profile, instrument, start, end)
    if profile.asset_class == "crypto":
        from research.binance_data import fetch_spot_daily
        frame = fetch_spot_daily(instrument.symbol, start, end)
    else:
        frame = fetch_equity(instrument, start, end)
    frame = validate_profile_data(frame, profile, instrument, start, end)
    frame.index.name = "Date"
    frame.to_csv(csv, index_label="Date")
    metadata.write_text(to_json({"request": request, "attrs": frame.attrs}) + "\n", encoding="utf-8")
    return frame


def validate_profile_data(frame, profile, instrument, start, end):
    frame = validate_ohlcv(frame)
    begin, finish = date_window(start, end)
    expected = {"symbol": instrument.symbol, "timeframe": "1d", "market_type": profile.market_type,
                "price_currency": profile.price_currency,
                "volume_unit": instrument.symbol[:-4] if profile.asset_class == "crypto" else "shares",
                "adjusted": profile.asset_class == "equity",
                "source": "binance_spot" if profile.asset_class == "crypto" else "yahoo"}
    for key, value in expected.items():
        if frame.attrs.get(key) != value:
            raise ValueError(f"{instrument.symbol}: source metadata mismatch for {key}")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError("daily research bars must use naive session/UTC dates")
    frame.index = frame.index.as_unit("ns")
    if instrument.listed_since:
        begin = max(begin, pd.Timestamp(instrument.listed_since))
    if (frame.index < begin).any() or (frame.index >= finish).any():
        raise ValueError("history outside requested security/date interval")
    if profile.asset_class == "crypto":
        expected_days = pd.date_range(begin, finish, inclusive="left", freq="D")
        if not frame.index.equals(expected_days):
            raise ValueError("incomplete Binance UTC daily coverage in research data/cache")
    else:
        import exchange_calendars as xcals
        # Include neighboring sessions: constructing a calendar on a holiday
        # makes that date earlier than its first session and query raises OOB.
        calendar = xcals.get_calendar("XNYS", start=begin - pd.Timedelta(days=31),
                                      end=finish + pd.Timedelta(days=31))
        sessions = calendar.sessions_in_range(begin, finish - pd.Timedelta(days=1))
        if sessions.tz is not None:
            sessions = sessions.tz_localize(None)
        missing, extra = sessions.difference(frame.index), frame.index.difference(sessions)
        if len(missing) or len(extra):
            raise ValueError(f"{instrument.symbol}: exchange session coverage mismatch; missing {len(missing)}, unexpected {len(extra)}")
        frame.attrs["session_validation"] = "XNYS exchange_calendars (US common cash-market sessions)"
    return frame


def build_replay(df, instrument: Instrument, profile: ResearchProfile, *, replay_bars=126, benchmark=None):
    df = validate_ohlcv(df)
    if not isinstance(replay_bars, int) or isinstance(replay_bars, bool) or replay_bars < 1:
        raise ValueError("replay_bars must be a positive integer")
    warmup = profile.config.baseline_window + 1
    if len(df) < warmup:
        raise ValueError(f"need at least {warmup} bars for replay; received {len(df)}")
    benchmark = validate_ohlcv(benchmark) if benchmark is not None else None
    first = max(warmup - 1, len(df) - replay_bars)
    snapshots, cases = [], []
    phases, events_seen, event_counts = Counter(), set(), Counter()
    previous_phase = None
    for i in range(first, len(df)):
        prefix = df.iloc[:i + 1].copy()
        # The benchmark is cut before calling core; no final-market snapshot.
        aligned = benchmark.loc[benchmark.index <= prefix.index[-1]] if benchmark is not None else None
        if aligned is not None and aligned.empty:
            aligned = None
        result = analyze_frame(prefix, instrument.symbol, config=profile.config, benchmark=aligned)
        core = result["analyses"]["atr"]["wyckoff"]
        for event in core["events"]:
            if not event["date"] <= event["available_date"] <= prefix.index[-1]:
                raise ValueError("future event in prefix snapshot")
        label = f"{core['phase']['regime']} / {core['phase']['phase']}"
        phases[label] += 1
        new_events = []
        for e in core["events"]:
            key = (e["event"], str(e["date"]), str(e["available_date"]), e.get("range_id"), e.get("side"), str(e.get("invalidated_event_date")))
            if key not in events_seen:
                events_seen.add(key)
                # The first snapshot may already contain years of old events.
                if e["available_date"] >= df.index[first]:
                    event_counts[e["event"]] += 1
                    new_events.append(e["event"])
        if new_events or (previous_phase is not None and label != previous_phase):
            cases.append({"snapshot_index": len(snapshots), "date": prefix.index[-1],
                          "label": ", ".join(dict.fromkeys(new_events)) or f"阶段变化: {previous_phase} → {label}"})
        snapshot = {"bar_index": i, "as_of": core["as_of"], "phase": core["phase"], "tr": core["tr"],
                    "events": range_events(core["events"], core["tr"])[-80:],
                    "context": core["context"], "laws": core["laws"], "readiness": core["readiness"],
                    "count": core["count"], "warnings": core["warnings"], "pnf": result["analyses"]["atr"]["pnf"]}
        snapshots.append(snapshot)
        previous_phase = label
    warnings = list(df.attrs.get("warnings", []))
    if len(df) < 252:
        warnings.append(f"短历史: {len(df)} bars，不能代表完整跨周期表现")
    if benchmark is None:
        warnings.append("无独立可对齐基准；相对强弱保持 unavailable")
    data = {**df.attrs, "bars": len(df), "start": df.index[0], "end": df.index[-1],
            "replay_start": df.index[first], "replay_snapshots": len(snapshots), "warnings": warnings,
            "benchmark": profile.benchmark if benchmark is not None else None,
            "history_coverage": "short" if len(df) < 252 else "multi_month"}
    bars = [{"date": date, **{c.lower(): float(row[c]) for c in ("Open", "High", "Low", "Close", "Volume")}}
            for date, row in df.iterrows()]
    return _serializable({"symbol": instrument.symbol, "name": instrument.name, "group": instrument.group,
                          "status": "ok", "data": data, "bars": bars, "snapshots": snapshots,
                          "summary": {"phase_counts": dict(phases), "event_counts": dict(event_counts), "cases": cases,
                                      "accuracy": "not_evaluated_without_independent_labels"}})


def check_equity_sessions(frame, benchmark, instrument, start):
    """Use the observed benchmark sessions to detect gaps, not weekday filling."""
    effective = max(pd.Timestamp(start), pd.Timestamp(instrument.listed_since)) if instrument.listed_since else pd.Timestamp(start)
    expected = benchmark.index[benchmark.index >= effective]
    missing = expected.difference(frame.index)
    if len(missing):
        raise ValueError(f"missing {len(missing)} benchmark trading sessions: {', '.join(d.strftime('%Y-%m-%d') for d in missing[:5])}")
    extra = frame.index.difference(expected)
    if len(extra):
        raise ValueError(f"unexpected {len(extra)} sessions absent from benchmark")


def markdown_summary(report):
    lines = [f"# {report['profile']['label']}研究", "", f"请求历史: {report['window']['start']} 至 {report['window']['end']}（结束日不含）",
             f"成功 {report['summary']['success']} / 失败 {report['summary']['failed']}；基准: {report['summary']['benchmark_status']}",
             "", "每个历史快照独立使用该日及以前的 bars；ATR/P&F 箱格也在各前缀重新计算。", "",
             "| 标的 | 数据 bars | 回放 bars | 规则案例 | 最后阶段候选 | 状态 |", "| --- | ---: | ---: | ---: | --- | --- |"]
    for item in report["instruments"]:
        if item["status"] == "ok":
            last = item["snapshots"][-1]["phase"]
            lines.append(f"| {item['symbol']} | {item['data']['bars']} | {len(item['snapshots'])} | {len(item['summary']['cases'])} | {last['regime']} / {last['phase']} | OK |")
        else:
            lines.append(f"| {item['symbol']} | — | — | — | — | FAILED: {str(item['error']).replace('|', '/')} |")
    lines.extend(["", "规则案例用于人工复核，不是独立真值。", "", *[f"- {limit}" for limit in report["limitations"]]])
    if report["summary"].get("benchmark_error"):
        lines.extend(["", "基准获取错误: " + report["summary"]["benchmark_error"]])
    return "\n".join(lines) + "\n"


def run_study(profile, start, end, output, *, replay_bars=126, refresh=False):
    from research.replay_view import write_replay_html
    start_date, end_date = date_window(start, end)
    if end_date > pd.Timestamp.now(tz="UTC").tz_localize(None).normalize():
        raise ValueError("end cannot include an unfinished UTC day")
    directory = Path(output) / profile.id
    directory.mkdir(parents=True, exist_ok=True)
    raw_dir = directory / "data"
    benchmark = None
    benchmark_error = None
    try:
        benchmark_instrument = next((s for s in profile.instruments if s.symbol == profile.benchmark),
                                    Instrument(profile.benchmark, "Market benchmark", "基准"))
        benchmark = load_data(profile, benchmark_instrument, start, end, raw_dir, refresh=refresh)
    except Exception as exc:
        benchmark_error = f"{type(exc).__name__}: {exc}"
        print(f"{profile.id} benchmark FAILED: {benchmark_error}", flush=True)
    profile_data = {k: getattr(profile, k) for k in ("id", "label", "asset_class", "market_type", "price_currency", "timeframe")}
    report = {"schema_version": 1, "profile": profile_data, "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
              "window": {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d"), "replay_bars": replay_bars},
              "config": asdict(profile.config), "instruments": [], "summary": {}, "limitations": LIMITATIONS}
    for instrument in profile.instruments:
        then = time.monotonic()
        try:
            frame = load_data(profile, instrument, start, end, raw_dir, refresh=refresh)
            if profile.asset_class == "equity" and benchmark is not None:
                check_equity_sessions(frame, benchmark, instrument, start)
            independent = None if instrument.symbol == profile.benchmark else benchmark
            item = build_replay(frame, instrument, profile, replay_bars=replay_bars, benchmark=independent)
        except Exception as exc:
            item = {"symbol": instrument.symbol, "name": instrument.name, "group": instrument.group,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        # Checkpoint successful and failed instruments for resumption/inspection.
        (directory / f"{instrument.symbol}.replay.json").write_text(to_json(item) + "\n", encoding="utf-8")
        report["instruments"].append(item)
        report["summary"] = {"success": sum(x["status"] == "ok" for x in report["instruments"]),
                             "failed": sum(x["status"] == "failed" for x in report["instruments"]),
                             "benchmark_status": "ok" if benchmark is not None else "failed",
                             "benchmark_error": benchmark_error, "independent_labels": "unavailable"}
        (directory / "study.json").write_text(to_json(report) + "\n", encoding="utf-8")
        print(f"{profile.id} {instrument.symbol}: {item['status']} ({time.monotonic() - then:.1f}s)" +
              (f" {item['error']}" if item["status"] == "failed" else ""), flush=True)
    (directory / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
    write_replay_html(report, directory / "replay.html")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="分市场真实案例与逐根回放，不进行参数调优")
    parser.add_argument("--market", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"), help="exclusive UTC date")
    parser.add_argument("--replay-bars", type=int, default=126)
    parser.add_argument("--output", type=Path, default=Path("output/market-study"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    if args.replay_bars < 1:
        parser.error("--replay-bars must be positive")
    try:
        reports = [run_study(profile, args.start, args.end, args.output, replay_bars=args.replay_bars, refresh=args.refresh)
                   for key, profile in PROFILES.items() if args.market in (key, "all")]
        return 2 if any(r["summary"]["failed"] or r["summary"]["benchmark_status"] != "ok" for r in reports) else 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"study failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
