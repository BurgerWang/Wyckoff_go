"""Competing A–E hypotheses and sequential evidence, without probabilities."""
from .structure import available


def _current_structure_check(pivots, last_price, acc):
    """Describe today's close against a known reaction, without changing phase.

    The direction is a comparison of confirmed historical pivots. A close can
    challenge the latest HL/LH before a replacement pivot is confirmed.
    """
    kind = "low" if acc else "high"
    pivot = pivots.get("last_" + kind)
    check = {"reference": "confirmed_pivots", "direction": "up" if acc else "down",
             "last_price": float(last_price), "reaction_pivot": None, "status": "unavailable"}
    if pivot is None or pivot.get("price") is None:
        return check
    price = float(pivot["price"])
    check["reaction_pivot"] = {"kind": kind, "price": price, "date": pivot.get("date"),
                               "available_date": pivot.get("available_date", pivot.get("date"))}
    challenged = last_price < price if acc else last_price > price
    check["status"] = "challenged" if challenged else "at_pivot" if last_price == price else "holding"
    return check


def range_events(events, tr):
    if tr is None:
        return [e for e in events if not e.get("range_id")]
    current = [e for e in events if (not tr.range_id or not e.get("range_id") or e["range_id"] == tr.range_id)
               and (tr.start_date is None or e["date"] >= tr.start_date)]
    # Preliminary action precedes the climax that starts a range. Retain the
    # last unassigned PS/PSY before that start, not every older cycle.
    for name in ("PS", "PSY"):
        prior = [e for e in events if e["event"] == name and not e.get("range_id")
                 and tr.start_date is not None and e["date"] < tr.start_date]
        if prior and any(e["event"] == ("SC" if name == "PS" else "BC") and e["date"] == tr.start_date for e in current):
            current.append(max(prior, key=available))
    return sorted(current, key=lambda e: (available(e), e["date"]))


def sequence_state(events):
    """Ordered evidence. An isolated test never completes a Wyckoff cycle.

    Compatible keys are descriptive and do not mean actionable confirmation.
    AR/ST need a side and an ordered stopping sequence, or explicit range
    context for a climaxless base. Opposing initiative invalidates support.
    """
    state = {"accum_progress": 0, "dist_progress": 0,
             "acc_stopped": False, "dist_stopped": False,
             "acc_spring": None, "acc_sos": None, "acc_lps": None,
             "dist_ut": None, "dist_sow": None, "dist_lpsy": None,
             "acc_test": None, "dist_test": None}
    range_id = None
    interior_floor = {True: None, False: None}

    def reset(acc, cutoff=None):
        interior_floor[acc] = cutoff
        if acc:
            state.update(accum_progress=0, acc_stopped=False, acc_spring=None, acc_sos=None, acc_lps=None, acc_test=None)
        else:
            state.update(dist_progress=0, dist_stopped=False, dist_ut=None, dist_sow=None, dist_lpsy=None, dist_test=None)

    for e in sorted(events, key=lambda x: (available(x), x["date"])):
        if e.get("range_id") and range_id and e["range_id"] != range_id:
            reset(True)
            reset(False)
        if e.get("range_id"):
            range_id = e["range_id"]
        name, side = e["event"], e.get("side")
        if name == "INVALIDATION":
            if side in ("accumulation", "distribution"):
                acc = side == "accumulation"
                relevant = [state[key] for key in (("acc_spring", "acc_test", "acc_sos", "acc_lps") if acc else ("dist_ut", "dist_test", "dist_sow", "dist_lpsy"))]
                if any(p and p["event"] == e.get("invalidated_event")
                       and p["date"] == e.get("invalidated_event_date") for p in relevant):
                    reset(acc, available(e))
        elif name in ("SC", "BC"):
            if e.get("subtype") == "local_stopping_action":
                continue
            reset(True, available(e))
            reset(False, available(e))
            state["accum_progress" if name == "SC" else "dist_progress"] = 1
        elif name in ("AR", "ST"):
            if side not in ("accumulation", "distribution"):
                continue
            key = "accum_progress" if side == "accumulation" else "dist_progress"
            if name == "AR" and state[key] == 1:
                state[key] = 2
            elif name == "ST" and state[key] == 2:
                state[key] = 3
                state["acc_stopped" if side == "accumulation" else "dist_stopped"] = True
        elif name == "SPRING":
            # Preserve earlier stopping evidence; a new test clears obsolete
            # initiative in its own side rather than certifying Phase C.
            state.update(acc_spring=e, acc_test=None, acc_sos=None, acc_lps=None)
            if e.get("range_id") or e.get("range_confirmed_at") or state["accum_progress"] >= 3:
                state["accum_progress"] = 4
        elif name in ("UT", "UTAD"):
            state.update(dist_ut=e, dist_test=None, dist_sow=None, dist_lpsy=None)
            if e.get("range_id") or e.get("range_confirmed_at") or state["dist_progress"] >= 3:
                state["dist_progress"] = 4
        elif name in ("TEST", "TEST_DEMAND"):
            acc = name == "TEST"
            parent = state["acc_spring" if acc else "dist_ut"]
            if parent and e.get("parent_date") == parent["date"]:
                state["acc_test" if acc else "dist_test"] = e
            elif (e.get("subtype") == ("interior_higher_low" if acc else "interior_lower_high")
                  and e.get("parent_type") == "confirmed_swing" and e.get("range_id")
                  and e.get("status") == "confirmed"
                  and all(e.get(key) is not None for key in (
                      "range_confirmed_at", "prior_turn_date", "prior_turn_available_date", "parent_date",
                      "reference_available_date", "turn_date", "turn_available_date"))
                  and e["range_confirmed_at"] < e["prior_turn_date"]
                  <= e["prior_turn_available_date"] <= e["parent_date"]
                  <= e["reference_available_date"] <= e["turn_date"]
                  <= e["turn_available_date"] <= e["date"] < available(e)
                  and (interior_floor[acc] is None or e["prior_turn_date"] > interior_floor[acc])
                  and state["acc_sos" if acc else "dist_sow"] is None):
                state["acc_test" if acc else "dist_test"] = e
                state["accum_progress" if acc else "dist_progress"] = 4
        elif name in ("SOS", "SOW"):
            acc = name == "SOS"
            reset(not acc, available(e))
            key = "accum_progress" if acc else "dist_progress"
            if state[key] >= 3 or e.get("range_id") or e.get("range_confirmed_at"):
                state[key] = 5
                state["acc_sos" if acc else "dist_sow"] = e
                state["acc_lps" if acc else "dist_lpsy"] = None
        elif name in ("LPS", "LPSY"):
            acc = name == "LPS"
            key = "accum_progress" if acc else "dist_progress"
            move = state["acc_sos" if acc else "dist_sow"]
            if move and state[key] == 5 and available(e) > available(move):
                if e.get("parent_date", move["date"]) == move["date"]:
                    state[key] = 6
                    state["acc_lps" if acc else "dist_lpsy"] = e
        elif name == "AMBIGUOUS_TEST":
            reset(True, available(e))
            reset(False, available(e))
    state.update(acc_complete=state["accum_progress"] == 6,
                 dist_complete=state["dist_progress"] == 6,
                 sos_confirmed=state["acc_sos"] is not None,
                 sow_confirmed=state["dist_sow"] is not None)
    return state


def classify_phase(events, tr, columns, last_price, outside_bars=0,
                   context=None, config=None):
    """Infer descriptive hypotheses from lifecycle + ordered events + structure.

    Internal SOS/SOW alone remains B. C accepts confirmed tests of boundary
    probes or ordered interior directional tests. E needs post-break support
    AND directional confirmed pivots. The current close's relation to the
    latest reaction pivot is a separate explanation, not an E phase gate;
    counting outside closes alone cannot establish E.
    """
    from .models import CoreConfig
    cfg = config or CoreConfig()
    recent = range_events(events, tr)
    state = sequence_state(recent)
    context = context or {}
    # Production provides a separately filtered/compressed current-range view.
    # Retain the historical context contract for explicit legacy callers.
    pivots = context.get("range_price_structure", context.get("price_structure", {}))
    structure = pivots.get("direction", "unknown")
    prior = context.get("range_prior_trend", {}).get("direction", "unknown")
    names = sorted({e["event"] for e in recent})
    result = {"regime": "unknown", "phase": "?", "notes": [], "events": names,
              "confidence": "low", "status": "insufficient_evidence", "hypotheses": {},
              "normal_no_observable": False,
              "source_ids": ["WA_TUTORIAL", "WA_ANATOMY"], "evidence_type": "rule_based_candidate"}
    candidate = context.get("stopping_candidate")
    if candidate:
        result.update(regime=candidate["side"], phase="A", status="candidate",
                      cycle_id=candidate["range_id"], parent_range_id=candidate.get("parent_range_id"),
                      events=sorted({e["event"] for e in events if e.get("range_id") == candidate["range_id"]}))
        result["notes"] = ["已持续离开前区间后出现独立停止候选；等待自动反向运动，前周期账本保留"
                           if candidate.get("parent_range_id") else "观察到独立停止候选；等待自动反向运动，尚未冻结新交易区间"]
        if tr is not None:
            previous = classify_phase(events, tr, columns, last_price, outside_bars,
                                      {**context, "stopping_candidate": None}, cfg)
            result["prior_range_phase"] = {k: previous[k] for k in ("regime", "phase", "status")}
            result["prior_range_phase"]["range_id"] = tr.range_id
        return result
    if tr is None:
        stops = [e for e in recent if e["event"] in ("SC", "BC", "AR")
                 and (context.get("stop_window_start") is None or available(e) >= context["stop_window_start"])]
        if stops:
            last = stops[-1]
            side = last.get("side") or ("accumulation" if last["event"] == "SC" else "distribution")
            result.update(regime=side, phase="A", status="candidate")
            result["notes"] = ["观察到停止趋势/自动反向运动候选，尚无成熟区间；SC/BC 不证明最终方向"]
        else:
            result["normal_no_observable"] = context.get("observation_eligible") is True
            if structure in ("up", "down"):
                result["regime"] = "markup" if structure == "up" else "markdown"
            result["notes"] = ["没有可观察的交易区间；仅描述已确认价格结构"]
        return result
    mature = tr.established_date is not None or (tr.origin == "pnf_congestion" and tr.lifecycle != "forming")
    for side, acc in (("accumulation", True), ("distribution", False)):
        move = state["acc_sos" if acc else "dist_sow"]
        support = state["acc_lps" if acc else "dist_lpsy"]
        test = state["acc_test" if acc else "dist_test"]
        outside = last_price > tr.top if acc else last_price < tr.bottom
        adverse = last_price < tr.bottom if acc else last_price > tr.top
        evidence, invalidations = [], []
        phase = "A" if not mature else "B"
        if test and mature:
            phase = "C"
            evidence.append("confirmed_interior_directional_test" if test.get("parent_type") == "confirmed_swing"
                            else "confirmed_test_after_boundary_probe")
        breakout = move is not None and move.get("scope") == "breakout"
        supported_internal = move is not None and support is not None and structure == ("up" if acc else "down")
        continuation = pivots.get("last_high" if acc else "last_low")
        reaction = pivots.get("last_low" if acc else "last_high")
        after_break = (move is not None and support is not None and continuation is not None and reaction is not None
                       and continuation["date"] > move["date"] and reaction["date"] >= support["date"])
        if mature and not adverse and (breakout or supported_internal):
            phase = "D"
            evidence.append("breakout_initiative" if breakout else "initiative_with_directional_reaction_structure")
        if (phase == "D" and outside and support is not None
                and outside_bars >= cfg.phase_e_closes and structure == ("up" if acc else "down") and after_break):
            phase = "E"
            evidence.extend(["confirmed_post_initiative_support", "continuing_directional_structure", "sustained_outside_closes"])
        if adverse:
            phase = "?"
            invalidations.append("price_beyond_opposite_boundary")
        elif move and move.get("scope") == "internal" and support is None:
            evidence.append("internal_initiative_does_not_establish_D")
        if state["accum_progress" if acc else "dist_progress"] >= 2:
            evidence.append("stopping_sequence")
        if state["acc_stopped" if acc else "dist_stopped"]:
            evidence.append("completed_stopping_test")
        if structure == ("up" if acc else "down"):
            evidence.append("directional_swings")
        strength = "medium" if phase in ("C", "D", "E") else "low"
        result["hypotheses"][side] = {"phase": phase, "status": "invalidated" if adverse else "candidate",
            "strength": strength, "evidence": evidence, "invalidations": invalidations,
            "variant": "reaccumulation" if acc and prior == "up" else "redistribution" if not acc and prior == "down" else side}
        if phase == "E":
            result["hypotheses"][side]["structure_check"] = _current_structure_check(pivots, last_price, acc)
    acc_h, dist_h = result["hypotheses"].values()
    latest = max((available(e) for e in recent), default=None)
    if any(e["event"] == "AMBIGUOUS_TEST" and available(e) == latest for e in recent):
        result.update(status="ambiguous", regime="trading range")
        result["notes"] = ["同根测试双边界，竞争假设仍有冲突"]
        return result
    selected = None
    for side, hyp in result["hypotheses"].items():
        if hyp["phase"] in ("C", "D", "E") and hyp["status"] != "invalidated":
            if selected is not None:
                result.update(status="ambiguous", regime="trading range")
                result["notes"] = ["吸筹与派发路径均存在，无法唯一选择阶段"]
                return result
            selected = (side, hyp)
    if selected:
        side, hyp = selected
        regime = hyp["variant"]
        if hyp["phase"] == "E":
            regime = "markup" if side == "accumulation" else "markdown"
        result.update(regime=regime, phase=hyp["phase"], confidence=hyp["strength"], status="candidate")
        result["notes"] = ["候选来自生命周期、顺序事件和已确认结构；并非交易触发或统计概率", *hyp["evidence"]]
        if hyp["phase"] == "C":
            result["notes"].append("已确认的是测试支点及其可知时点；是否属于结束 B 的最后供需测试尚未独立验证。")
        elif hyp["phase"] == "E":
            result["notes"].append("E 的延续结构证据来自已确认的历史摆动；当前收盘需另行核对最近反应支点。")
            check = hyp["structure_check"]
            pivot = check["reaction_pivot"]
            if pivot:
                relation = "低于" if last_price < pivot["price"] else "高于" if last_price > pivot["price"] else "等于"
                kind = "低点" if pivot["kind"] == "low" else "高点"
                result["notes"].append(f"当前收盘 {last_price:.4g} {relation}最近确认{kind} {pivot['price']:.4g}"
                                       f"（动作 {pivot['date']} / 确认 {pivot['available_date']}）。")
                if check["status"] == "challenged":
                    result["notes"].append("当前收盘与已确认方向结构存在冲突；E 仍是历史摆动及既有有效支持下的候选。")
            else:
                result["notes"].append("最近确认反应支点缺少价格，当前收盘与该支点的关系不可评估。")
    else:
        result.update(regime="trading range", phase="A" if not mature else "B", status="candidate")
        if acc_h["status"] == "invalidated" or dist_h["status"] == "invalidated":
            result.update(regime="markup" if last_price > tr.top else "markdown", phase="?")
            result["notes"] = ["价格已在区间外，但缺少同方向阶段证据"]
        else:
            result["notes"] = ["区间尚在形成" if not mature else "建因区间；单个 Spring/UT 或内部 SOS/SOW 不足以决定 C/D"]
            supported = []
            for side, acc in (("accumulation", True), ("distribution", False)):
                move = state["acc_sos" if acc else "dist_sow"]
                stopped = state["acc_stopped" if acc else "dist_stopped"]
                forming_stop = not mature and state["accum_progress" if acc else "dist_progress"] == 2
                directed = move and move.get("scope") == "internal" and structure == ("up" if acc else "down")
                if stopped or forming_stop or directed:
                    supported.append(side)
            if len(supported) == 1:
                hyp = result["hypotheses"][supported[0]]
                result.update(regime=hyp["variant"], notes=[*result["notes"], *hyp["evidence"]])
    return result
