"""Pure snapshot orchestration: evidence before interpretation."""
from dataclasses import asdict, replace

import numpy as np

from market_data import validate_ohlcv
from .context import analyze_context, analyze_laws
from .counts import count_scenarios
from .features import compute_features, confirmed_swings
from .models import CORE_VERSION, CoreConfig
from .primitives import _validate_columns
from .readiness import evaluate_readiness
from .stages import classify_phase, range_events, sequence_state
from .structure import replay_structure


def _range_columns(tr, columns):
    if tr is None:
        return None
    # These are chart presentation coordinates, not the price-range model.
    indexes = [i for i, c in enumerate(columns) if (tr.start_date is None or c[4] >= tr.start_date)
               and (tr.cause_end_date is None or c[4] <= tr.cause_end_date)]
    return replace(tr, start_col=indexes[0] if indexes else 0,
                   end_col=indexes[-1] if indexes else -1, width=len(indexes))


def analyze(columns, scale, df, reversal=3, *, config=None, benchmark=None,
            market_context=None, trade_plan=None, pnf_use_close=False):
    """All events use confirmation timestamps; absent evidence stays absent.

    Supply/demand and absorption are OHLCV proxies. Callers supply suitable,
    completed benchmark/market bars. No download, plotting or orders here.
    """
    cfg = config or CoreConfig()
    if not isinstance(cfg, CoreConfig):
        raise ValueError("config must be CoreConfig")
    if not isinstance(reversal, (int, np.integer)) or isinstance(reversal, bool) or reversal < 1:
        raise ValueError("reversal must be a positive integer")
    if df is None:
        return {"tr": None, "events": [], "phase": {"regime": "unknown", "phase": "?",
                "notes": ["未提供行情"], "confidence": "low", "status": "insufficient_data", "events": [],
                "normal_no_observable": False},
                "targets": None, "last_price": None, "as_of": None, "warnings": ["未提供行情数据"],
                "laws": {}, "context": {}, "readiness": {}, "ranges": [], "transitions": [],
                "sequence": sequence_state([]), "core_version": CORE_VERSION, "stopping_candidate": None,
                "analysis_cycle_id": None, "analysis_range_id": None}
    df = validate_ohlcv(df)
    _validate_columns(columns, df)
    s = np.asarray(scale, dtype=float)
    if scale is None or s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all() or (np.diff(s) <= 0).any():
        raise ValueError("scale must contain at least two increasing finite prices")
    features = compute_features(df, cfg)
    swings = confirmed_swings(df, cfg)
    structure = replay_structure(df, features, swings, cfg)
    tr = _range_columns(structure["tr"], columns)
    events = structure["events"]
    context = analyze_context(df, features, swings, benchmark, market_context, cfg, tr=tr)
    context["stop_window_start"] = df.index[max(0, len(df) - cfg.trend_window)]
    context["stopping_candidate"] = structure["stopping_candidate"]
    context["observation_eligible"] = bool(features["observation_eligible"].iloc[-1])
    if tr and tr.start_date in df.index:
        start = df.index.get_loc(tr.start_date)
        before = df.iloc[max(0, start - cfg.trend_window):start]
        a = float(features["atr"].iloc[max(0, start - 1)])
        change = float(before["Close"].iloc[-1] - before["Close"].iloc[0]) / a if len(before) >= cfg.trend_window and a > 0 else None
        direction = "unknown" if change is None else "up" if change > cfg.trend_min_atr else "down" if change < -cfg.trend_min_atr else "flat"
        context["range_prior_trend"] = {"direction": direction, "change_atr": change,
            "reference": "completed_bars_before_range_start", "end_date": before.index[-1] if len(before) else None}
    # A new stopping candidate has no established range of its own. Keep the
    # parent geometry/ledger, but do not publish its projections or checklist
    # as evidence for the new cycle.
    candidate = structure["stopping_candidate"]
    analysis_tr = None if candidate else tr
    current_events = ([e for e in events if e.get("range_id") == candidate["range_id"]]
                      if candidate else range_events(events, tr))
    count = count_scenarios(columns, s, analysis_tr, current_events, reversal, df, use_close=pnf_use_close)
    if candidate:
        count["reason"] = "current_stopping_cycle_has_no_established_range"
    targets = count["targets"]
    laws = analyze_laws(df, features, swings, analysis_tr, targets, cfg)
    outside = 0
    last_price = float(df["Close"].iloc[-1])
    if tr and (last_price > tr.top or last_price < tr.bottom):
        for date, close in df["Close"].iloc[::-1].items():
            if date < tr.confirmed_date or not (close > tr.top if last_price > tr.top else close < tr.bottom):
                break
            outside += 1
    phase = classify_phase(events, tr, columns, last_price, outside, context, cfg)
    if len(df) <= cfg.baseline_window:
        phase.update(phase="?", regime="unknown", status="insufficient_data", normal_no_observable=False,
                     notes=["量价基准尚未预热完成"])
    readiness = evaluate_readiness(context, laws, current_events, analysis_tr, targets, trade_plan)
    warnings = []
    if (df["Volume"].iloc[-cfg.baseline_window:] == 0).any():
        warnings.append("近期存在零成交量 bar，量价证据可能不足")
    if count["reason"] == "nonuniform_grid_requires_separate_count_method":
        warnings.append("非等宽 P&F 刻度：固定箱水平计数不可用")
    if context["relative_strength"]["status"] == "unavailable":
        warnings.append("缺少可对齐市场基准，相对强弱不可用")
    return {"tr": tr, "events": events, "phase": phase, "targets": targets,
            "last_price": last_price, "as_of": df.index[-1], "warnings": warnings,
            "core_version": CORE_VERSION, "config": asdict(cfg), "context": context, "laws": laws,
            "readiness": readiness, "count": count, "swings": swings,
            "sequence": sequence_state(current_events),
            "analysis_cycle_id": candidate["range_id"] if candidate else tr.range_id if tr else None,
            "analysis_range_id": analysis_tr.range_id if analysis_tr else None,
            "ranges": [_range_columns(r, columns) for r in structure["ranges"]],
            "transitions": structure["transitions"],
            "stopping_candidate": structure["stopping_candidate"],
            "coverage": {"event_model": "conservative_ohlcv_candidates",
                "phase_model": "competing_lifecycle_hypotheses",
                "order_flow": "unavailable_without_order_book",
                "limitations": ["阈值为工程启发式；非典型结构和主观趋势线需要独立验证。",
                                "阶段和计数不是胜率、收益或机构行为的证明。"]}}


def report(res):
    lines = ["== Wyckoff 量价结构分析 ==", f"截至: {res.get('as_of')}"]
    if res.get("last_price") is not None:
        lines.append(f"现价: {res['last_price']:.2f}")
    tr = res["tr"]
    candidate = res.get("stopping_candidate")
    if candidate:
        lines.append(f"当前停止周期: {candidate['range_id']} | {candidate['side']} | 等待自动反向运动形成区间")
    if tr:
        lines.append(f"{'父周期保留' if candidate else 'OHLCV'} 区间: {tr.bottom:.2f}–{tr.top:.2f} | {tr.lifecycle} | 起点 {tr.start_date}")
    else:
        lines.append("区间: 尚无有效区间")
    phase = res["phase"]
    lines.append(f"阶段候选: {phase['regime']} / Phase {phase['phase']}，证据={phase['confidence']}")
    lines.extend(f"  - {note}" for note in phase["notes"])
    prior = phase.get("prior_range_phase")
    if prior:
        lines.append(f"父周期阶段: {prior['range_id']} | {prior['regime']} / Phase {prior['phase']} ({prior['status']})")
    for side, h in phase.get("hypotheses", {}).items():
        lines.append(f"  {side}: Phase {h['phase']} ({h['status']})")
    local = res.get("context", {}).get("range_price_structure")
    if local:
        direction = {"up": "HH/HL", "down": "LH/LL", "mixed": "方向混合", "unknown": "方向证据不足"}[local["direction"]]
        lines.append(f"{'父区间' if candidate else '当前区间'}确认摆动: {direction} | 起点 {local['start_date']}"
                     f" | 交替支点 {local['confirmed_pivots']}")
        lines.extend(f"  - {item}" for item in local["evidence"])
    laws = res.get("laws", {})
    if laws:
        lines.append(f"供需代理: {laws['supply_demand']['direction']} | 努力/结果: {laws['effort_result']['status']} | 因果计数: {laws['cause_effect']['status']}")
    shown = ([e for e in res["events"] if e.get("range_id") == candidate["range_id"]] if candidate else
             range_events(res["events"], tr) if tr else res["events"][-20:])
    if shown:
        lines.append("当前停止周期事件（date / available_date）：" if candidate else
                     "当前区间事件候选（date / available_date）：" if tr else "量价事件候选（date / available_date）：")
        for e in shown[-20:]:
            detail = (f" | 失效对象 {e.get('invalidated_event')} @ {e.get('invalidated_event_date')}"
                      f" | 原因 {e.get('reason')}" if e["event"] == "INVALIDATION" else "")
            lines.append(f"  {e['event']} @ {e['price']:.2f} | {e['date']} / {e['available_date']} | {e.get('scope', e['status'])}{detail}")
    targets = res.get("targets")
    if targets:
        for side, key in (("bull", "bullish"), ("bear", "bearish")):
            if targets[key]:
                lines.append(f"{key} 水平跨度计数: {targets[f'width_{side}']} 列 | {targets[key]}")
    else:
        lines.append(f"水平计数不可用: {res.get('count', {}).get('reason', 'missing_data')}")
    for key, items in res.get("readiness", {}).items():
        if key in ("nine_buying", "nine_selling", "five_steps"):
            counts = {state: sum(x["status"] == state for x in items) for state in ("met", "not_met", "unavailable")}
            lines.append(f"{key}: 满足 {counts['met']} / 未满足 {counts['not_met']} / 缺数据 {counts['unavailable']}")
    lines.extend(f"提示: {w}" for w in res.get("warnings", []))
    lines.append("规则候选供走势分析辅助；OHLCV 代理不测量真实订单流或机构持仓。")
    return "\n".join(lines)
