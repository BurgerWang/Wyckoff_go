"""Wyckoff's five steps and nine tests, with explicit unavailable inputs.

This is an analytical checklist, not authorization or a trade recommendation.
The official buying and selling tests have different ordering.
"""
import math

_SOURCE = ["WA_TUTORIAL"]


def _item(number, label, status, evidence=(), required_data=()):
    return {"id": number, "label": label, "status": status, "evidence": list(evidence),
            "required_data": list(required_data), "source_ids": _SOURCE.copy()}


def _boolean_test(number, label, value, evidence, required):
    return _item(number, label, "unavailable" if value is None else "met" if value else "not_met",
                 evidence, required if value is None else ())


def _event_combination(events, names, side, tr):
    candidates = []
    range_id = getattr(tr, "range_id", "") if tr is not None else ""
    for e in events:
        if e.get("side") not in (None, side):
            continue
        if range_id and e.get("range_id") not in (None, "", range_id):
            continue
        if e["event"] in names:
            candidates.append(e)
    candidates.sort(key=lambda e: (e.get("available_date", e["date"]), e["date"]))
    progress = 0
    for e in candidates:
        if e["event"] == names[0]:
            progress = 1
        elif progress < len(names) and progress > 0 and e["event"] == names[progress]:
            progress += 1
    return progress == len(names)


def _reward_risk(side, targets, plan):
    if plan is None:
        return None, [], ["明确的入场价、初始止损及同方向有效计数情景"]
    try:
        entry, stop = float(plan["entry"]), float(plan["stop"])
        threshold = float(plan.get("min_reward_risk", 3))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("trade_plan requires numeric entry, stop and min_reward_risk") from exc
    if not all(math.isfinite(v) and v > 0 for v in (entry, stop, threshold)):
        raise ValueError("trade_plan prices and min_reward_risk must be positive and finite")
    risk = entry - stop if side == "buy" else stop - entry
    if risk <= 0:
        return None, ["所给止损不能形成该方向的初始风险计划。"], ["该方向有效的初始止损"]
    target = (targets or {}).get("bullish" if side == "buy" else "bearish")
    if not target:
        return None, [], ["同方向有效计数情景"]
    values = [float(v) for v in target.values() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    if not values:
        return None, [], ["有限数值的同方向价格目标"]
    conservative = min(values) if side == "buy" else max(values)
    reward = conservative - entry if side == "buy" else entry - conservative
    ratio = reward / risk
    minimum = max(3.0, threshold)
    return ratio >= minimum, [f"条件情景：入场 {entry:g}、止损 {stop:g}、保守目标 {conservative:g}，收益/风险 {ratio:.2f}，要求至少 {minimum:g}。"], []


def _minimum_objective(direction, targets, plan):
    """Five-step cause requirement is distinct from nine-test reward/risk."""
    required = ["明确的最低价格目标 minimum_target、方向及有效同向计数"]
    if direction not in ("up", "down") or plan is None or "minimum_target" not in plan:
        return None, [], required
    try:
        threshold = float(plan["minimum_target"])
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_target must be positive and finite") from exc
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("minimum_target must be positive and finite")
    target = (targets or {}).get("bullish" if direction == "up" else "bearish")
    if not target:
        return None, [], required
    conservative = target["min"] if direction == "up" else target["max"]
    met = conservative >= threshold if direction == "up" else conservative <= threshold
    return met, [f"{direction} 条件计数保守目标 {conservative:g}，所要求最低目标 {threshold:g}。"], []


def evaluate_readiness(context, laws, events, tr, targets, trade_plan=None):
    structure = context.get("price_structure", {})
    relative = context.get("relative_strength", {})
    market = context.get("market_alignment", {})
    activity = laws.get("supply_demand", {}).get("activity", {})
    stride = context.get("stride", {})
    objectives = context.get("historical_objectives", {})
    base = (tr is not None and tr.top > tr.bottom and tr.confirmed_date is not None
            and getattr(tr, "lifecycle", "active") != "forming")
    base_evidence = ["已确认的交易区间提供横向 base/crown 候选；不能单独确定吸筹或派发。"] if base else ["未观察到已确认的横向范围。"]
    relative_available = relative.get("status") == "available" and bool(relative.get("rally_comparisons")) and bool(relative.get("reaction_comparisons"))
    stronger = relative.get("direction") == "stronger" if relative_available else None
    weaker = relative.get("direction") == "weaker" if relative_available else None
    buy_rr, buy_evidence, buy_required = _reward_risk("buy", targets, trade_plan)
    sell_rr, sell_evidence, sell_required = _reward_risk("sell", targets, trade_plan)
    crown = True if base and (targets or {}).get("bearish") and (targets or {}).get("width_bear", 0) >= 2 else None
    objective_required = ["先前已知的同方向 P&F 目标及其完成历史"]
    buying = [
        _boolean_test(1, "先前下跌目标完成", objectives.get("buy", {}).get("met"), objectives.get("buy", {}).get("evidence", []), objective_required),
        _boolean_test(2, "PS、SC、ST 停止趋势组合", _event_combination(events, ("PS", "SC", "ST"), "accumulation", tr), ["按已知时间检查 PS→SC→ST 候选组合。"], ["同一结构的 PS/SC/ST"]),
        _boolean_test(3, "上涨增量、回落缩量", activity.get("bullish"), ["比较对应确认上涨/回落波段的相对量。"], ["具有有效成交量基准的上涨及回落确认波段"]),
        _boolean_test(4, "下降 stride 被打破", stride.get("downward_broken"), ["外部有锚点及确认时间的解释。"] if stride.get("downward_broken") is not None else [], ["有明确锚点与确认时间的下降趋势线/stride"]),
        _boolean_test(5, "更高低点", structure.get("higher_lows"), structure.get("evidence", []), ["至少两个确认低点"]),
        _boolean_test(6, "更高高点", structure.get("higher_highs"), structure.get("evidence", []), ["至少两个确认高点"]),
        _boolean_test(7, "强于市场", stronger, relative.get("evidence", []), ["同时间/交易日适用基准及对应上涨和回落波段"]),
        _boolean_test(8, "横向 base 形成", base, base_evidence, ["已确认交易区间"]),
        _boolean_test(9, "条件上涨收益至少三倍初始风险", buy_rr, buy_evidence, buy_required),
    ]
    selling = [
        _boolean_test(1, "先前上涨目标完成", objectives.get("sell", {}).get("met"), objectives.get("sell", {}).get("evidence", []), objective_required),
        _boolean_test(2, "上涨缩量、回落增量", activity.get("bearish"), ["比较对应确认上涨/回落波段的相对量。"], ["具有有效成交量基准的上涨及回落确认波段"]),
        _boolean_test(3, "PSY、BC 停止趋势组合", _event_combination(events, ("PSY", "BC"), "distribution", tr), ["按已知时间检查 PSY→BC 候选组合。"], ["同一结构的 PSY/BC"]),
        _boolean_test(4, "弱于市场", weaker, relative.get("evidence", []), ["同时间/交易日适用基准及对应上涨和回落波段"]),
        _boolean_test(5, "上涨 stride 被打破", stride.get("upward_broken"), [], ["有明确锚点与确认时间的上涨趋势线/stride"]),
        _boolean_test(6, "更低高点", structure.get("lower_highs"), structure.get("evidence", []), ["至少两个确认高点"]),
        _boolean_test(7, "更低低点", structure.get("lower_lows"), structure.get("evidence", []), ["至少两个确认低点"]),
        _boolean_test(8, "P&F 横向 crown 形成", crown, base_evidence if crown else [], ["同一结构的有效横向 P&F crown/计数证据"]),
        _boolean_test(9, "条件下跌收益至少三倍初始风险", sell_rr, sell_evidence, sell_required),
    ]
    market_direction = market.get("market_trend", "unknown")
    market_known = market.get("status") == "available" and market_direction in ("up", "down", "flat")
    alignment = None
    if market_known and relative_available:
        alignment = (market.get("aligned") is True and (stronger if market_direction == "up" else weaker if market_direction == "down" else False))
    applicable = buying if market_direction == "up" else selling if market_direction == "down" else None
    readiness = None
    if applicable:
        statuses = [p["status"] for p in applicable]
        readiness = False if "not_met" in statuses else True if all(s == "met" for s in statuses) else None
    cause_met, cause_evidence, cause_required = _minimum_objective(market_direction, targets, trade_plan)
    steps = [
        _boolean_test(1, "确定市场位置及趋势", True if market_known else None, market.get("evidence", []), ["当前市场基准或有 as_of 的市场情境"]),
        _boolean_test(2, "选取顺应市场且相对强弱合适的标的", alignment, relative.get("evidence", []), ["市场方向、标的结构与对应基准波段"]),
        _boolean_test(3, "cause 满足明确最低目标", cause_met, cause_evidence, cause_required),
        _boolean_test(4, "逐项检查移动准备度", readiness, ["按当前市场方向检查九项；未提供的历史目标和 stride 不视为通过。"], ["对应九项测试的完整可用资料"]),
        _boolean_test(5, "结合已确认的大盘转折择时", market.get("turn_confirmed"), [], ["截至当前的大盘转折及其确认资料"]),
    ]
    return {"nine_buying": buying, "nine_selling": selling, "five_steps": steps,
            "status": "analytical_checklist", "source_ids": _SOURCE.copy(),
            "limitations": ["met 表示规则候选证据满足该条，不是预测准确率或交易许可。",
                            "缺历史目标、趋势线、基准或交易计划时保留 unavailable；不制造通过条件。"]}
