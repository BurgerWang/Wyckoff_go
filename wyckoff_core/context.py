"""Market context and the three Wyckoff laws as inspectable evidence.

Bar volume is unsigned. Directional activity and absorption below are
OHLCV inferences; they are not order-flow imbalance or institutional trades.
"""
import numpy as np
import pandas as pd
import math
from itertools import groupby

from .models import CoreConfig

_STANDARD = ["WA_TUTORIAL"]
_OHLCV_LIMITATION = "OHLCV 为间接量价证据，无法测量真实供需、订单流失衡或机构持仓。"


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _extend_alternating_swings(result, swings):
    """Append known pivot groups; an unordered double extreme replaces neither.

    Detect both extrema before compression, so replacing a same-kind pivot
    cannot erase an earlier valid pivot when the other extreme follows.
    """
    ordered = sorted(swings, key=lambda p: (p["available_date"], p["date"], p["kind"]))
    for _, group in groupby(ordered, key=lambda p: (p["available_date"], p["date"])):
        pivots = list(group)
        if len({p["kind"] for p in pivots}) > 1:
            # OHLC does not establish the order of same-bar high and low.
            continue
        pivot = pivots[-1]
        if result and result[-1]["kind"] == pivot["kind"]:
            better = pivot["price"] >= result[-1]["price"] if pivot["kind"] == "high" else pivot["price"] <= result[-1]["price"]
            if better:
                result[-1] = pivot
        else:
            result.append(pivot)
    return result


def _alternating_swings(swings, as_of):
    """Compress only already-known pivots, preserving their original ledger."""
    return _extend_alternating_swings([], [p for p in swings if p["available_date"] <= as_of])


def _trend(df, features, config):
    # This recent pre-bar trend is not automatically the trend before a TR.
    recent = df.iloc[max(0, len(df) - config.trend_window - 1):-1]
    atr = _number(features["atr"].iloc[-2]) if len(features) > 1 else None
    change = float(recent["Close"].iloc[-1] - recent["Close"].iloc[0]) if len(recent) >= config.trend_window else None
    normalized = change / atr if change is not None and atr and atr > 0 else None
    direction = "unknown" if normalized is None else "up" if normalized > config.trend_min_atr else "down" if normalized < -config.trend_min_atr else "flat"
    return {"direction": direction, "change_atr": normalized,
            "start_date": recent.index[0] if len(recent) else None,
            "end_date": recent.index[-1] if len(recent) else None,
            "reference": "recent_completed_bars_before_current_bar",
            "evidence": [] if normalized is None else [f"最近先行窗口净变动 {normalized:.2f} ATR"],
            "limitations": ["先行窗口是工程描述，不能单独判定吸筹、派发或范围的原始趋势。"]}


def _structure(swings):
    highs = [p for p in swings if p["kind"] == "high"]
    lows = [p for p in swings if p["kind"] == "low"]
    hh = highs[-1]["price"] > highs[-2]["price"] if len(highs) >= 2 else None
    lh = highs[-1]["price"] < highs[-2]["price"] if len(highs) >= 2 else None
    hl = lows[-1]["price"] > lows[-2]["price"] if len(lows) >= 2 else None
    ll = lows[-1]["price"] < lows[-2]["price"] if len(lows) >= 2 else None
    direction = "up" if hh and hl else "down" if lh and ll else "unknown" if hh is None or hl is None else "mixed"
    evidence = []
    if len(highs) >= 2:
        evidence.append(f"确认高点 {highs[-2]['price']:.4g} → {highs[-1]['price']:.4g}")
    if len(lows) >= 2:
        evidence.append(f"确认低点 {lows[-2]['price']:.4g} → {lows[-1]['price']:.4g}")
    return {"direction": direction, "higher_highs": hh, "higher_lows": hl,
            "lower_highs": lh, "lower_lows": ll, "evidence": evidence,
            "last_high": highs[-1] if highs else None,
            "last_low": lows[-1] if lows else None,
            "confirmed_pivots": len(swings), "source_ids": _STANDARD.copy()}


def _range_swings(swings, tr):
    """Filter raw pivots before compression; prior cycles cannot replace them."""
    return [p for p in swings if tr.start_date is not None and p["date"] >= tr.start_date]


def _range_structure(pivots, tr):
    return {**_structure(pivots), "range_id": tr.range_id, "start_date": tr.start_date,
            "reference": "confirmed_pivots_since_range_start"}


def _benchmark_series(df, benchmark):
    if not isinstance(benchmark, (pd.DataFrame, pd.Series)):
        raise ValueError("benchmark must be a chronological Close Series or DataFrame")
    subject_tf, benchmark_tf = df.attrs.get("timeframe"), getattr(benchmark, "attrs", {}).get("timeframe")
    if subject_tf and benchmark_tf and subject_tf != "provided" and benchmark_tf != "provided" and subject_tf != benchmark_tf:
        return None, "incompatible_timeframes"
    if isinstance(benchmark, pd.DataFrame):
        if "Close" not in benchmark.columns:
            raise ValueError("benchmark requires Close")
        series = benchmark["Close"].copy()
        series.attrs = benchmark.attrs.copy()
    elif isinstance(benchmark, pd.Series):
        series = benchmark.copy()
    else:
        raise ValueError("benchmark must be a chronological Close Series or DataFrame")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.hasnans or not series.index.is_unique or not series.index.is_monotonic_increasing:
        raise ValueError("benchmark dates must be unique and strictly increasing")
    daily = df.attrs.get("timeframe") == "1d" and benchmark.attrs.get("timeframe") == "1d"
    if daily:
        subject_keys = pd.Index(df.index.date)
        benchmark_keys = pd.Index(series.index.date)
        if not subject_keys.is_unique or not benchmark_keys.is_unique:
            raise ValueError("daily benchmark alignment requires one bar per trading date")
        series.index = benchmark_keys
        series = series.loc[series.index <= subject_keys[-1]]
        series = pd.to_numeric(series, errors="raise").astype(float)
        if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
            raise ValueError("benchmark Close must be positive and finite")
        aligned = series.reindex(subject_keys)
        aligned.index = df.index
        return aligned, "same_explicit_daily_trading_date"
    if (df.index.tz is None) != (series.index.tz is None):
        return None, "incompatible_timestamp_timezones"
    series = series.loc[series.index <= df.index[-1]]
    series = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(series.to_numpy()).all() or (series <= 0).any():
        raise ValueError("benchmark Close must be positive and finite")
    return series.reindex(df.index), "exact_timestamp"


def _relative_strength(df, swings, benchmark):
    result = {"status": "unavailable", "direction": "unknown", "rally_outperformance": None,
              "reaction_resilience": None, "rally_comparisons": [], "reaction_comparisons": [],
              "evidence": [], "limitations": [], "source_ids": _STANDARD.copy()}
    if benchmark is None:
        result["limitations"] = ["未提供适当的市场基准，不能判断相对强弱。"]
        return result, None
    aligned, alignment = _benchmark_series(df, benchmark)
    result["alignment"] = alignment
    if aligned is None or aligned.notna().sum() < 3:
        result["limitations"] = ["同时间/显式交易日的基准重叠不足；不前向填充或拼接不同周期。"]
        return result, aligned
    result["status"] = "available"
    result["overlap_bars"] = int(aligned.notna().sum())
    uncovered = set()  # kinds whose latest wave the benchmark does not span
    for first, last in zip(swings, swings[1:]):
        start, end = first["date"], last["date"]
        if start not in aligned.index or end not in aligned.index:
            continue
        stock_return = float(df.loc[end, "Close"] / df.loc[start, "Close"] - 1)
        if not (stock_return > 0 if first["kind"] == "low" else stock_return < 0):
            continue
        key = "rally_comparisons" if first["kind"] == "low" else "reaction_comparisons"
        # Entire corresponding wave must be observed on the same timestamps.
        wave = aligned.loc[start:end]
        if len(wave) < 2 or wave.isna().any():
            uncovered.add(key)
            continue
        uncovered.discard(key)
        market_return = float(aligned.loc[end] / aligned.loc[start] - 1)
        comparison = {"start_date": start, "end_date": end, "available_date": last["available_date"],
                      "stock_return": stock_return, "benchmark_return": market_return,
                      "return_difference": stock_return - market_return}
        result[key].append(comparison)
    if uncovered:
        # Older comparisons stay listed, but they are not the current relative strength.
        names = "、".join(name for key, name in (("rally_comparisons", "上涨"), ("reaction_comparisons", "回落"))
                         if key in uncovered)
        result["status"] = "unavailable"
        result["limitations"] = [f"最近的确认{names}波段缺少同时间基准；不用更早的波段代替当前相对强弱。",
                                 "基准适用性及其 bar 完成状态由调用者确定；比较只用当前已知数据。"]
        return result, aligned
    rallies, reactions = result["rally_comparisons"], result["reaction_comparisons"]
    if rallies:
        result["rally_outperformance"] = all(p["return_difference"] > 0 for p in rallies[-2:])
    if reactions:
        result["reaction_resilience"] = all(p["return_difference"] > 0 for p in reactions[-2:])
    if rallies and reactions:
        positive = all(p["return_difference"] > 0 for p in rallies[-2:] + reactions[-2:])
        negative = all(p["return_difference"] < 0 for p in rallies[-2:] + reactions[-2:])
        result["direction"] = "stronger" if positive else "weaker" if negative else "mixed"
        result["evidence"] = [f"对应确认波段：上涨 {len(rallies)} 段、回落 {len(reactions)} 段；相对方向 {result['direction']}"]
    else:
        result["limitations"].append("缺少可对齐的上涨和回落确认波段，不能完成 Wyckoff 相对强弱测试。")
    result["limitations"].append("基准适用性及其 bar 完成状态由调用者确定；比较只用当前已知数据。")
    return result, aligned


def analyze_context(df, features, swings, benchmark=None, market_context=None, config=None, *, tr=None):
    config = config or CoreConfig()
    pivots = _alternating_swings(swings, df.index[-1])
    structure = _structure(pivots)
    relative, aligned = _relative_strength(df, pivots, benchmark)
    market = {"status": "unavailable", "direction": "unknown", "market_trend": "unknown",
              "aligned": None, "turn_confirmed": None, "evidence": [],
              "limitations": [], "source_ids": _STANDARD.copy()}
    if aligned is not None and aligned.notna().any():
        # The window is the subject's own latest bars. A benchmark that ends
        # earlier or has a gap there gives no current market trend.
        window = aligned.iloc[-config.trend_window:]
        if len(window) == config.trend_window and window.notna().all():
            movement = float(window.iloc[-1] / window.iloc[0] - 1)
            direction = "up" if movement > 0 else "down" if movement < 0 else "flat"
            market.update(status="available", market_trend=direction, direction=direction,
                          basis="aligned_benchmark_window",
                          window={"start": window.index[0], "end": window.index[-1], "bars": len(window)},
                          evidence=[f"同时间基准最近窗口收益 {movement:.2%}"])
        else:
            market["limitations"].append(
                f"基准未覆盖标的最近 {config.trend_window} 根 bar（最后重叠 {aligned.dropna().index[-1]}），"
                "不用过期或拼接的窗口判断市场趋势。")
    if market_context is not None:
        if not isinstance(market_context, dict):
            raise ValueError("market_context must be a dict with as_of and trend")
        supplied_as_of = market_context.get("as_of")
        timely = False
        if supplied_as_of is not None:
            try:
                timely = pd.Timestamp(supplied_as_of) <= df.index[-1]
            except (TypeError, ValueError):
                timely = False
        if timely:
            direction = market_context.get("trend", market_context.get("direction", "unknown"))
            direction = {"markup": "up", "markdown": "down"}.get(direction, direction)
            if direction in ("up", "down", "flat"):
                market.update(status="available", market_trend=direction, direction=direction,
                              basis="caller_market_context")
                market.pop("window", None)
                market["evidence"].append(f"调用者提供的大盘描述截至 {supplied_as_of}")
            turn = market_context.get("turn_confirmed")
            if isinstance(turn, bool):
                market["turn_confirmed"] = turn
        else:
            market["limitations"].append("外部市场情境缺少合法 as_of 或来自未来，未用于判断。")
    if market["market_trend"] in ("up", "down") and structure["direction"] in ("up", "down"):
        market["aligned"] = market["market_trend"] == structure["direction"]
    if market["status"] == "unavailable":
        market["limitations"].append("缺少可用基准或截至当前的市场情境。")
    stride, historical = _external_references(df, market_context)
    result = {"prior_trend": _trend(df, features, config), "price_structure": structure,
              "relative_strength": relative, "market_alignment": market,
              "stride": stride, "historical_objectives": historical,
              "source_ids": _STANDARD.copy()}
    if tr is not None:
        local_pivots = _alternating_swings(_range_swings(swings, tr), df.index[-1])
        result["range_price_structure"] = _range_structure(local_pivots, tr)
    return result


def _external_references(df, supplied):
    """Typed optional references; never substitute today's count for a prior one.

    Stride is an explicit caller-verified line interpretation, not an inferred
    universal trend-line algorithm. Objective completion is checked against
    the supplied completed OHLCV bar at the asserted achievement date.
    """
    stride = {"downward_broken": None, "upward_broken": None, "references": {},
              "limitations": ["自动 stride/趋势线切换未实现；仅接受有锚点及确认时间的外部解释。"]}
    historical = {side: {"met": None, "evidence": [], "required_data": ["历史目标、发布与达到时间及对应行情"]}
                  for side in ("buy", "sell")}
    if not isinstance(supplied, dict) or supplied.get("as_of") is None:
        return stride, historical
    try:
        known = pd.Timestamp(supplied["as_of"])
        if pd.isna(known) or known > df.index[-1]:
            return stride, historical
    except (TypeError, ValueError):
        return stride, historical
    lines = supplied.get("stride", {})
    objectives = supplied.get("historical_objectives", {})
    if not isinstance(lines, dict) or not isinstance(objectives, dict):
        raise ValueError("stride and historical_objectives must be dictionaries")
    for name in ("downward", "upward"):
        ref = lines.get(name)
        if not isinstance(ref, dict) or not isinstance(ref.get("broken"), bool):
            continue
        anchors = ref.get("anchors", [])
        try:
            confirmed = pd.Timestamp(ref["confirmed_at"])
            dates = [pd.Timestamp(a["date"]) for a in anchors]
            prices = [float(a["price"]) for a in anchors]
            valid = (len(anchors) >= 2 and not pd.isna(confirmed) and confirmed <= known
                     and all(not pd.isna(d) and d <= confirmed for d in dates)
                     and all(d1 < d2 for d1, d2 in zip(dates, dates[1:]))
                     and all(math.isfinite(p) and p > 0 for p in prices))
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            stride[f"{name}_broken"] = ref["broken"]
            stride["references"][name] = {**ref, "evidence_type": "caller_verified_interpretation"}
    for side, column in (("buy", "Low"), ("sell", "High")):
        ref = objectives.get(side)
        if not isinstance(ref, dict):
            continue
        try:
            target = float(ref["target"])
            issued, reached = pd.Timestamp(ref["issued_at"]), pd.Timestamp(ref["achieved_at"])
            valid = (math.isfinite(target) and target > 0 and not pd.isna(issued) and not pd.isna(reached)
                     and issued < reached <= known and reached in df.index)
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            price = float(df.loc[reached, column])
            met = price <= target if side == "buy" else price >= target
            historical[side] = {"met": bool(met), "target": target, "issued_at": issued,
                "achieved_at": reached, "evidence": [f"外部历史目标 {target:g}，发布 {issued}，检查 {reached} 的 {column}={price:g}。"],
                "required_data": [], "evidence_type": "caller_record_with_observed_price_check"}
    return stride, historical


def _wave_activity(df, features, swings):
    activities = {"rallies": [], "reactions": []}
    for first, last in zip(swings, swings[1:]):
        first_pos, last_pos = df.index.get_loc(first["date"]), df.index.get_loc(last["date"])
        if last_pos <= first_pos:
            continue
        segment = features.iloc[first_pos + 1:last_pos + 1]
        ratios = segment["volume_ratio"].dropna()
        if len(ratios) != len(segment) or not len(ratios):
            continue
        key = "rallies" if first["kind"] == "low" else "reactions"
        price_change = float(df["Close"].iloc[last_pos] - df["Close"].iloc[first_pos])
        if not (price_change > 0 if key == "rallies" else price_change < 0):
            continue
        atr = _number(features["atr"].iloc[first_pos])
        activities[key].append({"start_date": first["date"], "end_date": last["date"],
                                "available_date": last["available_date"], "bars": len(segment),
                                "mean_volume_ratio": float(ratios.mean()),
                                "total_volume": float(df["Volume"].iloc[first_pos + 1:last_pos + 1].sum()),
                                "price_change": price_change,
                                "result_atr": abs(price_change) / atr if atr and atr > 0 else None})
    rally_mean = float(np.mean([p["mean_volume_ratio"] for p in activities["rallies"][-2:]])) if activities["rallies"] else None
    reaction_mean = float(np.mean([p["mean_volume_ratio"] for p in activities["reactions"][-2:]])) if activities["reactions"] else None
    available = rally_mean is not None and reaction_mean is not None
    activities.update(rally_mean_volume_ratio=rally_mean, reaction_mean_volume_ratio=reaction_mean,
                      bullish=rally_mean > reaction_mean if available else None,
                      bearish=reaction_mean > rally_mean if available else None)
    return activities


def _wave_effort_result(activity):
    comparisons = []
    for kind, key in (("rally", "rallies"), ("reaction", "reactions")):
        waves = activity[key]
        for previous, current in zip(waves[-3:], waves[-3:][1:]):
            if previous["result_atr"] is None or current["result_atr"] is None:
                continue
            effort_change = current["total_volume"] - previous["total_volume"]
            result_change = current["result_atr"] - previous["result_atr"]
            if effort_change > 0 and result_change < 0:
                relation = "increased_effort_reduced_result"
                interpretation = "possible_supply_absorption" if kind == "reaction" else "possible_supply_opposition"
            elif effort_change < 0 and result_change < 0:
                relation = "reduced_effort_reduced_result"
                interpretation = "possible_reduced_supply" if kind == "reaction" else "possible_reduced_demand"
            elif effort_change * result_change > 0:
                relation, interpretation = "effort_result_harmony", "directional_activity_candidate"
            else:
                relation, interpretation = "divergence_or_unchanged_effort_result", "context_required"
            comparisons.append({"kind": kind, "previous": previous, "current": current,
                                "available_date": current["available_date"], "relationship": relation,
                                "interpretation": interpretation})
    return comparisons


def analyze_laws(df, features, swings, tr, targets=None, config=None):
    config = config or CoreConfig()
    pivots = _alternating_swings(swings, df.index[-1])
    activity = _wave_activity(df, features, pivots)
    usable = features["volume_ratio"].notna() & (df["Volume"] > 0)
    recent = features.loc[usable].iloc[-config.trend_window:]
    evidence = []
    direction = "unknown"
    if len(recent):
        bar = features.iloc[-1]
        result = _number(bar["price_change_atr"])
        volume = _number(bar["volume_ratio"])
        spread = _number(bar["spread_ratio"])
        if result is not None and volume is not None and spread is not None and df["Volume"].iloc[-1] > 0:
            if result >= 0.75 and volume >= config.volume_high and spread >= 1:
                direction = "demand"
                evidence.append("当前 bar 净上移、相对量及价差支持需求占优的候选解释。")
            elif result <= -0.75 and volume >= config.volume_high and spread >= 1:
                direction = "supply"
                evidence.append("当前 bar 净下移、相对量及价差支持供应占优的候选解释。")
            elif activity["bullish"] and _structure(pivots)["direction"] == "up":
                direction = "demand"
                evidence.append("确认结构抬高，最近上涨波段相对量高于回落波段。")
            elif activity["bearish"] and _structure(pivots)["direction"] == "down":
                direction = "supply"
                evidence.append("确认结构下移，最近回落波段相对量高于上涨波段。")
            else:
                direction = "balanced"
                evidence.append("可用量价观察尚未形成一致方向优势；balanced 表示代理证据无优势。")
    supply = {"status": "candidate" if direction != "unknown" else "unavailable",
              "direction": direction, "activity": activity, "evidence": evidence,
              "limitations": [_OHLCV_LIMITATION, "价量相对阈值和波段窗口是工程启发式。"],
              "source_ids": ["WA_TUTORIAL", "CKS2014"]}
    observations = []
    start = max(0, len(df) - config.trend_window)
    for i in range(start, len(df)):
        bar = features.iloc[i]
        if bar["effort_result"] == "unknown" or df["Volume"].iloc[i] <= 0:
            continue
        raw_type = bar["effort_result"]
        interpretation = raw_type
        if raw_type == "high_effort_little_result":
            if tr is not None and tr.confirmed_date is not None and df.index[i] >= tr.confirmed_date:
                atr = _number(bar["atr"]) or 0
                support = abs(float(df["Low"].iloc[i]) - tr.bottom) <= config.test_tolerance_atr * atr
                resistance = abs(float(df["High"].iloc[i]) - tr.top) <= config.test_tolerance_atr * atr
                if support and not resistance:
                    interpretation = "supply_absorption_candidate"
                elif resistance and not support:
                    interpretation = "demand_absorption_candidate"
            # Never identify institutions from this unsigned price/volume proxy.
        elif raw_type == "reduced_effort":
            change = _number(bar["price_change_atr"])
            if change is not None and change < 0:
                interpretation = "reduced_supply_candidate"
            elif change is not None and change > 0:
                interpretation = "reduced_demand_candidate"
        observations.append({"date": df.index[i], "available_date": df.index[i], "type": interpretation,
                             "volume_ratio": _number(bar["volume_ratio"]),
                             "spread_ratio": _number(bar["spread_ratio"]),
                             "price_change_atr": _number(bar["price_change_atr"])})
    wave_comparisons = _wave_effort_result(activity)
    effort = {"status": "candidate" if usable.iloc[-config.trend_window:].any() else "unavailable",
              "observations": observations, "evidence": [p["type"] for p in observations],
              "wave_comparisons": wave_comparisons,
              "limitations": [_OHLCV_LIMITATION, "高努力低结果可能来自不同机制；吸收仅为结合位置的候选解释。"],
              "source_ids": _STANDARD.copy()}
    effort["evidence"].extend(p["interpretation"] for p in wave_comparisons)
    valid_targets = targets is not None and bool(targets.get("bullish") or targets.get("bearish"))
    cause = {"status": "scenario" if valid_targets else "unavailable", "targets": targets,
             "evidence": ["已形成范围提供 cause 上下文。"] if tr is not None else [],
             "conditions": ["水平计数需要明确计数线、范围和方向，且满足相应尺度的计数规则。"],
             "limitations": ["计数是条件性价格情景，不是概率预测；不能据此认定已完成吸筹/派发。"],
             "source_ids": _STANDARD.copy()}
    if not valid_targets:
        cause["limitations"].append("当前没有可用计数情景，原因须与计数模块的具体条件一并查看。")
    return {"supply_demand": supply, "effort_result": effort, "cause_effect": cause,
            "source_ids": _STANDARD.copy()}
