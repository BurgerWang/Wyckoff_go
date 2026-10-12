"""Causal OHLCV range lifecycle and contextual event candidates.

The thresholds implement a conservative, explicitly partial encoding of
Wyckoff teaching. Events describe observable price/volume, never operators.
"""
from dataclasses import replace
from decimal import Decimal

import numpy as np

from .models import CoreConfig, TR


def available(event):
    return event.get("available_date", event["date"])


def _event(name, date, price, ratio, known=None, **details):
    return {"event": name, "date": date, "available_date": known if known is not None else date,
            "price": float(price), "vol": float(ratio) if ratio is not None else None,
            "volume_ratio": float(ratio) if ratio is not None else None, "status": "candidate",
            "source_ids": ["WA_TUTORIAL", "WA_ANATOMY"], **details}


def replay_structure(df, features, swings, config=None, initial_range=None, *, on_bar=None):
    """Replay only information available on each bar; never slide away a range.

    Two formation paths: stopping action + AR (forming until a secondary
    test), or repeated confirmed OHLCV swings (climaxless congestion).
    A stopping candidate has its own identity while the prior range remains
    observable. Only its confirmed AR installs a new forming range. Sustained
    price separation retires an obsolete range; a timeout alone does not.

    ``on_bar(i, state)`` observes every processed prefix synchronously. The
    callback must not mutate or retain the shared event/range/transition lists;
    callers that need a stored snapshot must copy their selected evidence.
    """
    cfg = config or CoreConfig()
    # Decimal complements preserve the original literal 0.3 SOW boundary.
    climax_close_max = float(Decimal(1) - Decimal(str(cfg.climax_recovery_close)))
    initiative_close_max = float(Decimal(1) - Decimal(str(cfg.initiative_close_min)))
    dates = df.index
    close = df["Close"].to_numpy(dtype=float)
    opening = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    atr = features["atr"].to_numpy(dtype=float)
    vr = features["volume_ratio"].to_numpy(dtype=float)
    sr = features["spread_ratio"].to_numpy(dtype=float)
    cp = features["close_position"].to_numpy(dtype=float)
    true_ratio = features["true_range_ratio"].to_numpy(dtype=float)
    positions = {date: i for i, date in enumerate(dates)}
    arrivals = {}
    for swing in swings:
        arrivals.setdefault(swing["available_date"], []).append(swing)
    known_swings, events, ranges, transitions = [], [], [], []
    tr = replace(initial_range) if initial_range is not None else None
    start_i = positions.get(tr.start_date, 0) if tr else 0
    established_i = positions.get(tr.established_date or tr.confirmed_date, 0) if tr else None
    if tr and not tr.range_id:
        tr.range_id = "provided:" + str(dates[start_i])
    if tr:
        ranges.append(tr)
    climax = None
    last_range_end = 0
    first_break = None
    pending_test = {"accumulation": None, "distribution": None}
    initiative = {"accumulation": None, "distribution": None}
    watched = {"accumulation": None, "distribution": None}
    interior_since = {"accumulation": established_i or 0,
                      "distribution": established_i or 0}
    excursions = {"accumulation": None, "distribution": None}
    wave_floor = {"accumulation": established_i or 0, "distribution": established_i or 0}
    used_waves = set()

    def active_stop():
        if climax and climax["eligible"] and climax["ar"] is None:
            return {k: climax[k] for k in ("range_id", "side", "price", "parent_range_id")} | {
                "date": dates[climax["i"]], "available_date": dates[climax["i"]]}
        return None

    def notify(i):
        if on_bar is not None:
            on_bar(i, {"tr": tr, "events": events, "ranges": ranges,
                       "transitions": transitions, "stopping_candidate": active_stop()})

    def emit(name, i, price, side=None, pivot_i=None, **extra):
        j = i if pivot_i is None else pivot_i
        v = vr[j] if np.isfinite(vr[j]) else None
        event = _event(name, dates[j], price, v, dates[i], side=side,
                       volume=float(volume[j]), range_id=extra.pop("range_id", tr.range_id if tr else None), **extra)
        events.append(event)
        return event

    def cancel_excursion(i, side):
        excursion = excursions[side]
        if excursion is not None:
            emit("INVALIDATION", i, close[i], side, invalidated_event="BOUNDARY_EXCURSION",
                 invalidated_event_date=dates[excursion["i"]], status="confirmed",
                 reason="support_chain_invalidated_during_recovery")
            excursions[side] = None

    def transition(i, state, reason):
        nonlocal tr
        if tr.lifecycle == state:
            return
        tr = replace(tr, lifecycle=state)
        ranges[-1] = tr
        transitions.append({"range_id": tr.range_id, "available_date": dates[i],
                            "state": state, "reason": reason})

    def retire(i, reason):
        nonlocal tr, established_i, first_break, last_range_end
        nonlocal initiative, pending_test, watched, interior_since, excursions, wave_floor, used_waves
        if tr is None:
            return
        transition(i, "retired", reason)
        tr, established_i, first_break = None, None, None
        last_range_end = i
        initiative = {"accumulation": None, "distribution": None}
        pending_test = {"accumulation": None, "distribution": None}
        watched = {"accumulation": None, "distribution": None}
        interior_since = {"accumulation": i, "distribution": i}
        excursions = {"accumulation": None, "distribution": None}
        wave_floor = {"accumulation": i, "distribution": i}
        used_waves = set()

    def create(i, left, bottom, top, origin, forming=False, cycle_id=None, parent_id=None):
        nonlocal tr, start_i, established_i, first_break, initiative, pending_test, watched, interior_since
        nonlocal excursions, wave_floor, used_waves
        if bottom <= 0 or top <= bottom or (top - bottom) / bottom > cfg.range_max_pct:
            return False
        if tr:
            transitions.append({"range_id": tr.range_id, "available_date": dates[i],
                                "state": "retired", "reason": "new_stopping_cycle" if forming else "detached_new_range"})
            ranges[-1] = replace(tr, lifecycle="retired")
        start_i, established_i = left, None if forming else i
        start = dates[left]
        tr = TR(0, 0, float(top), float(bottom), 0, float((top + bottom) / 2),
                avg_volume=float(volume[left:i + 1].mean()), start_date=start,
                end_date=dates[i], confirmed_date=dates[i],
                established_date=None if forming else dates[i], cause_end_date=dates[i],
                range_id=cycle_id or origin + ":" + str(start) + ":" + str(dates[i]),
                origin=origin, lifecycle="forming" if forming else "active", parent_range_id=parent_id)
        ranges.append(tr)
        transitions.append({"range_id": tr.range_id, "available_date": dates[i],
                            "state": tr.lifecycle, "reason": origin,
                            "bottom": tr.bottom, "top": tr.top})
        first_break = None
        initiative = {"accumulation": None, "distribution": None}
        pending_test = {"accumulation": None, "distribution": None}
        watched = {"accumulation": None, "distribution": None}
        interior_since = {"accumulation": i, "distribution": i}
        excursions = {"accumulation": None, "distribution": None}
        wave_floor = {"accumulation": i, "distribution": i}
        used_waves = set()
        return True

    def wave_evidence(i, bull):
        """A finite multi-bar displacement through an already confirmed turn.

        The normalized thresholds reuse the existing movement, close-efficiency
        and local-trend settings. Mean effort must reach its preceding baseline;
        no isolated high-volume bar or retrospective diagonal line is required.
        """
        side = "accumulation" if bull else "distribution"
        same = [s for s in known_swings if s["kind"] == ("low" if bull else "high")
                and positions[s["date"]] > wave_floor[side]]
        if not same:
            return None
        anchor = same[-1]
        j = positions[anchor["date"]]
        duration = i - j
        if not 2 <= duration <= cfg.local_trend_bars or (side, j) in used_waves:
            return None
        turns = [s for s in known_swings if s["kind"] != anchor["kind"]
                 and s["available_date"] <= anchor["date"] and positions[s["date"]] > wave_floor[side]]
        if not turns:
            return None
        reference = turns[-1]
        delta = np.diff(close[j:i + 1]) * (1 if bull else -1)
        net, path = delta.sum(), np.abs(delta).sum()
        valid = (volume[j:i + 1] > 0).all() and np.isfinite(vr[j + 1:i + 1]).all()
        held = close[j:i + 1].min() >= tr.bottom if bull else close[j:i + 1].max() <= tr.top
        crossed = close[i] > reference["price"] + cfg.boundary_buffer_atr * atr[j] if bull else close[i] < reference["price"] - cfg.boundary_buffer_atr * atr[j]
        # Starting outside a range cannot invent its missing breakthrough.
        started_inside = tr.bottom <= close[j] <= tr.top
        if not (valid and held and crossed and started_inside and path > 0
                and net / path >= cfg.initiative_close_min
                and net >= max(cfg.range_min_height_atr, duration * cfg.initiative_move_min_atr) * atr[j]
                and net >= cfg.initiative_spread_min * abs(reference["price"] - anchor["price"])
                and vr[j + 1:i + 1].mean() >= 1
                and (cp[i] >= cfg.climax_recovery_close if bull else cp[i] <= climax_close_max)):
            return None
        boundary = tr.top if bull else tr.bottom
        outside = close[i] > boundary if bull else close[i] < boundary
        crossings = [k for k in range(j + 1, i + 1)
                     if (close[k] > boundary if bull else close[k] < boundary)]
        if outside and not all(close[k] > boundary if bull else close[k] < boundary
                               for k in range(crossings[0], i + 1)):
            return None
        return {"subtype": "multi_bar_initiative", "scope": "breakout" if outside else "internal",
                "wave_start_date": anchor["date"], "reference_swing_date": reference["date"],
                "reference_available_date": reference["available_date"],
                "breakout_date": dates[crossings[0]] if outside else None,
                "wave_bars": duration, "net_move_atr": float(net / atr[j]),
                "path_efficiency": float(net / path), "mean_volume_ratio": float(vr[j + 1:i + 1].mean()),
                "evidence": ["confirmed_turn_crossed", "expanded_wave_vs_prior_reaction", "multi_bar_directional_result",
                             "efficient_displacement", "baseline_wave_effort"]}

    def interior_test(i, j, swing, side, bull):
        """A conservative no-Spring/no-UTAD path, not every HL/LH in B.

        Require an ordered four-pivot directional sequence, an entire interior
        reaction, contraction of actual volume and spread, and an observed
        response through the test bar. Existing initiative belongs to the LPS
        path. Resets prevent old pivots from reviving invalidated evidence.
        """
        if (established_i is None or j - established_i < cfg.min_range_bars
                or pending_test[side] or initiative[side] or watched[side]):
            return None
        previous = [s for s in known_swings if s["date"] < swing["date"]
                    and positions[s["date"]] > interior_since[side]]
        same = [s for s in previous if s["kind"] == swing["kind"]]
        opposite = [s for s in previous if s["kind"] != swing["kind"]]
        if not same or len(opposite) < 2:
            return None
        reference, turn = same[-1], opposite[-1]
        earlier = [s for s in opposite if s["date"] < reference["date"]]
        if not earlier:
            return None
        prior_turn = earlier[-1]
        if not (prior_turn["available_date"] <= reference["date"]
                and reference["available_date"] <= turn["date"]
                and turn["available_date"] <= swing["date"]):
            return None
        k, left = positions[reference["date"]], positions[prior_turn["date"]]
        improves = (swing["price"] > reference["price"] and turn["price"] > prior_turn["price"]
                    if bull else swing["price"] < reference["price"] and turn["price"] < prior_turn["price"])
        contained = (low[left:i + 1].min() >= tr.bottom and high[left:i + 1].max() <= tr.top
                     and (volume[left:i + 1] > 0).all())
        quiet = (vr[j] <= cfg.volume_low and sr[j] <= cfg.interior_test_spread_max and volume[j] < volume[k]
                 and high[j] - low[j] < high[k] - low[k])
        response = close[i] > high[j] if bull else close[i] < low[j]
        if not (improves and contained and quiet and response):
            return None
        return emit("TEST" if bull else "TEST_DEMAND", i, swing["price"], side, j,
                    status="confirmed", subtype="interior_higher_low" if bull else "interior_lower_high",
                    parent_type="confirmed_swing", parent_date=reference["date"],
                    reference_available_date=reference["available_date"],
                    prior_turn_date=prior_turn["date"], prior_turn_available_date=prior_turn["available_date"],
                    turn_date=turn["date"], turn_available_date=turn["available_date"],
                    range_confirmed_at=tr.established_date or tr.confirmed_date,
                    evidence=["mature_range", "ordered_directional_swings", "interior_reaction",
                              "reduced_volume_and_spread", "confirmed_pivot", "response_through_test_bar"])

    def update_geometry(i):
        nonlocal tr, first_break
        if tr is None or (tr.confirmed_date is not None and dates[i] < tr.confirmed_date):
            return
        inside = tr.bottom <= close[i] <= tr.top
        if inside:
            tr = replace(tr, cause_end_date=dates[i])
            ranges[-1] = tr
            first_break = None
        up = close[i] > tr.top and close[i - 1] <= tr.top
        down = close[i] < tr.bottom and close[i - 1] >= tr.bottom
        if up or down:
            first_break = i
            if established_i is not None:
                transition(i, "breakout_up" if up else "breakout_down", "close_outside")
            else:
                transitions.append({"range_id": tr.range_id, "available_date": dates[i],
                                    "state": "forming_departure", "reason": "close_outside_unestablished_range"})
        elif inside and tr.lifecycle in ("breakout_up", "breakout_down"):
            transition(i, "reentered", "close_back_inside")

    for i in range(len(df)):
        new_swings = arrivals.get(dates[i], [])
        known_swings.extend(new_swings)
        if i == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            notify(i)
            continue
        a = atr[i]
        if tr and established_i is None and i - start_i > cfg.climax_sequence_max_bars:
            retire(i, "forming_range_expired")
        if climax:
            bull = climax["side"] == "accumulation"
            invalid = close[i] < climax["price"] - cfg.boundary_buffer_atr * a if bull else close[i] > climax["price"] + cfg.boundary_buffer_atr * a
            limit = min(cfg.trend_window, cfg.climax_sequence_max_bars) if climax["ar"] is None else cfg.climax_sequence_max_bars
            expired = i - climax["i"] > limit and not climax["st"]
            if not climax["st"] and (invalid or expired):
                reason = "stopping_extreme_closed_through" if invalid else "stopping_sequence_expired"
                emit("INVALIDATION", i, close[i], climax["side"], range_id=climax["range_id"],
                     invalidated_event="SC" if bull else "BC", invalidated_event_date=dates[climax["i"]],
                     status="confirmed", reason=reason)
                if tr and tr.range_id == climax["range_id"] and established_i is None:
                    retire(i, reason)
                climax = None
        maintained_support = (tr is not None and ((close[i] > tr.top and watched["accumulation"]
            and watched["accumulation"]["event"] == "LPS") or (close[i] < tr.bottom and watched["distribution"]
            and watched["distribution"]["event"] == "LPSY")))
        if (tr and not maintained_support and first_break is not None and i - first_break >= cfg.climax_sequence_max_bars
                and (close[i] > tr.top + (tr.top - tr.bottom) or close[i] < tr.bottom - (tr.top - tr.bottom))):
            retire(i, "sustained_price_separation")
        for side, bull in (("accumulation", True), ("distribution", False)):
            move = initiative[side]
            adverse = tr is not None and (close[i] < tr.bottom if bull else close[i] > tr.top)
            expired = move is not None and i - positions[available(move)] > cfg.climax_sequence_max_bars
            if move and (adverse or expired):
                emit("INVALIDATION", i, close[i], side, invalidated_event=move["event"],
                     invalidated_event_date=move["date"], reason="initiative_expired" if expired else "opposite_range_boundary_closed_through",
                     status="confirmed")
                initiative[side] = None
                pending_test[side] = None
                watched[side] = None
                interior_since[side] = i
                wave_floor[side] = i
                cancel_excursion(i, side)
            test = pending_test[side]
            if test and i - positions[available(test)] > cfg.climax_sequence_max_bars:
                emit("INVALIDATION", i, close[i], side, invalidated_event=test["event"],
                     invalidated_event_date=test["date"], reason="probe_test_window_expired", status="confirmed")
                pending_test[side] = None
                if watched[side] is test:
                    watched[side] = None
                interior_since[side] = i
        # Keep historical observations immutable. A later violation is a new
        # ledger entry, not deletion or silent resurrection of old evidence.
        for side, bull in (("accumulation", True), ("distribution", False)):
            support = watched[side]
            if support and dates[i] > available(support):
                violated = low[i] < support["price"] - cfg.boundary_buffer_atr * a if bull else high[i] > support["price"] + cfg.boundary_buffer_atr * a
                # A confirmed support cannot survive a close through the
                # opposite range boundary just because its pivot's ATR buffer
                # still holds. The phase layer already treats that close as
                # invalidation; keep the ledger terminal on reentry as well.
                adverse_close = (tr is not None and support.get("status") == "confirmed"
                    and support["event"] in ("TEST", "TEST_DEMAND", "LPS", "LPSY")
                    and (close[i] < tr.bottom if bull else close[i] > tr.top))
                if violated or adverse_close:
                    emit("INVALIDATION", i, close[i] if adverse_close else low[i] if bull else high[i], side,
                         invalidated_event=support["event"], invalidated_event_date=support["date"],
                         reason="opposite_range_boundary_closed_through" if adverse_close else "confirmed_support_extreme_violated",
                         status="confirmed")
                    move = initiative[side]
                    if move is not None:
                        emit("INVALIDATION", i, low[i] if bull else high[i], side,
                             invalidated_event=move["event"], invalidated_event_date=move["date"],
                             reason="support_for_active_initiative_violated", status="confirmed")
                    watched[side] = None
                    initiative[side] = None
                    pending_test[side] = None
                    interior_since[side] = i
                    wave_floor[side] = i
                    cancel_excursion(i, side)
        for side, excursion in excursions.items():
            if excursion and (i - excursion["i"] > cfg.local_trend_bars or volume[i] <= 0):
                emit("INVALIDATION", i, close[i], side, invalidated_event="BOUNDARY_EXCURSION",
                     invalidated_event_date=dates[excursion["i"]], status="confirmed",
                     reason="recovery_window_expired" if volume[i] > 0 else "recovery_volume_unavailable")
                excursions[side] = None
        if tr is not None and not tr.bottom <= close[i] <= tr.top:
            interior_since = {"accumulation": i, "distribution": i}
        if not np.isfinite([vr[i], sr[i], cp[i]]).all() or volume[i] <= 0:
            update_geometry(i)  # price invalidation remains observable without volume
            notify(i)
            continue
        trend_start = max(0, i - cfg.trend_window)
        trend = (close[i - 1] - close[trend_start]) / a
        local_trend = (close[i - 1] - close[max(0, i - cfg.local_trend_bars)]) / a
        prev_extreme = slice(max(0, i - cfg.baseline_window), i)
        selling = min(trend, local_trend) <= -cfg.trend_min_atr and low[i] <= low[prev_extreme].min()
        buying = max(trend, local_trend) >= cfg.trend_min_atr and high[i] >= high[prev_extreme].max()
        # Preliminary support/supply is a local slowing proxy, not proof of a
        # dominant participant. A climax may close against the prior trend.
        climax_name = None
        gap_down, gap_up = opening[i] < low[i - 1], opening[i] > high[i - 1]
        gap_climax = np.isfinite(true_ratio[i]) and true_ratio[i] >= cfg.spread_high
        if vr[i] >= cfg.volume_high:
            if selling and cp[i] >= cfg.climax_recovery_close and (sr[i] >= cfg.spread_high or (gap_down and gap_climax)):
                climax_name = "SC"
            elif buying and cp[i] <= climax_close_max and (sr[i] >= cfg.spread_high or (gap_up and gap_climax)):
                climax_name = "BC"
        if climax_name:
            side = "accumulation" if climax_name == "SC" else "distribution"
            price = low[i] if side == "accumulation" else high[i]
            # Local climaxes are observations, not permission to reset an
            # established cycle. A new cycle needs prior sustained separation.
            outside = (close[max(0, i - cfg.phase_e_closes):i] < tr.bottom if side == "accumulation"
                       else close[max(0, i - cfg.phase_e_closes):i] > tr.top) if tr else None
            eligible = tr is None or (len(outside) == cfg.phase_e_closes and outside.all()
                and (price < tr.bottom if side == "accumulation" else price > tr.top))
            climax_path = "intrabar_spread" if sr[i] >= cfg.spread_high else "gap_true_range"
            climax_evidence = ["prior_trend", "wide_spread" if climax_path == "intrabar_spread" else "opening_gap_and_true_range",
                              "high_volume", "off_extreme_close"]
            if eligible:
                if climax and climax["ar"] is None:
                    emit("INVALIDATION", i, price, climax["side"], range_id=climax["range_id"],
                         invalidated_event="SC" if climax["side"] == "accumulation" else "BC",
                         invalidated_event_date=dates[climax["i"]], reason="superseded_stopping_candidate", status="confirmed")
                cycle_id = "ohlcv_climax_ar:" + str(dates[i])
                parent = tr.range_id if tr else None
                emit(climax_name, i, price, side, range_id=cycle_id, parent_range_id=parent,
                     evidence=climax_evidence, climax_path=climax_path, subtype="new_stopping_candidate")
                climax = {"i": i, "side": side, "price": price, "volume": volume[i],
                          "spread": features["true_range"].iloc[i] if climax_path == "gap_true_range" else high[i] - low[i],
                          "spread_type": climax_path, "ar": None, "extreme": None, "st": False,
                          "range_id": cycle_id, "parent_range_id": parent, "eligible": True}
            else:
                emit(climax_name, i, price, side, subtype="local_stopping_action",
                     climax_path=climax_path, evidence=climax_evidence)
        elif vr[i] >= cfg.preliminary_volume_min and sr[i] >= cfg.preliminary_spread_min and (selling or buying):
            side = "accumulation" if selling else "distribution"
            name = "PS" if selling else "PSY"
            if not any(e["event"] == name and positions[e["date"]] >= i - cfg.trend_window for e in events):
                emit(name, i, low[i] if selling else high[i], side,
                     evidence=["prior_trend", "increased_activity"], subtype="preliminary_activity_proxy")
        if climax and 0 < i - climax["i"] <= cfg.climax_sequence_max_bars:
            bull = climax["side"] == "accumulation"
            if climax["ar"] is None and i - climax["i"] <= cfg.trend_window:
                extreme = climax["extreme"]
                value = high[i] if bull else low[i]
                if extreme is None or (value > extreme[1] if bull else value < extreme[1]):
                    climax["extreme"] = (i, value)
                    extreme = climax["extreme"]
                counter = close[i] < close[i - 1] if bull else close[i] > close[i - 1]
                moved = extreme[1] - climax["price"] if bull else climax["price"] - extreme[1]
                if counter and moved >= cfg.ar_min_move_atr * a and extreme[0] < i:
                    j, price = extreme
                    bottom, top = (climax["price"], price) if bull else (price, climax["price"])
                    if bottom <= close[i] <= top and create(i, climax["i"], bottom, top, "ohlcv_climax_ar", forming=True,
                            cycle_id=climax["range_id"], parent_id=climax["parent_range_id"]):
                        climax["ar"] = emit("AR", i, price, climax["side"], j,
                            parent_date=dates[climax["i"]], status="confirmed", evidence=["counter_move", "later_reversal"])
            elif climax["ar"] is not None and not climax["st"] and i - climax["i"] >= cfg.min_range_bars:
                value = low[i] if bull else high[i]
                test_spread = features["true_range"].iloc[i] if climax["spread_type"] == "gap_true_range" else high[i] - low[i]
                if (tr and tr.range_id == climax["range_id"] and tr.bottom <= close[i] <= tr.top
                        and abs(value - climax["price"]) <= cfg.test_tolerance_atr * a
                        and volume[i] < climax["volume"] and test_spread < climax["spread"]
                        and vr[i] <= cfg.st_volume_ratio_max):
                    emit("ST", i, value, climax["side"], parent_date=climax["ar"]["date"],
                         evidence=["retest", "reduced_effort"])
                    climax["st"] = True
                    if tr and tr.lifecycle == "forming" and tr.start_date == dates[climax["i"]]:
                        established_i = i
                        tr = replace(tr, established_date=dates[i])
                        ranges[-1] = tr
                        transition(i, "active", "secondary_test")
                        interior_since = {"accumulation": i, "distribution": i}
        # A non-climactic range requires repeated *confirmed* swing tests.
        search_from = last_range_end
        can_replace = (tr is None or (first_break is not None and i - first_break >= cfg.min_range_bars)) and active_stop() is None
        if can_replace and new_swings:
            left_limit = first_break + 1 if tr is not None else search_from
            ss = [s for s in known_swings if positions[s["date"]] >= left_limit]
            hs = [s for s in ss if s["kind"] == "high"][-cfg.min_range_tests:]
            ls = [s for s in ss if s["kind"] == "low"][-cfg.min_range_tests:]
            if len(hs) == cfg.min_range_tests and len(ls) == cfg.min_range_tests:
                left = min(positions[s["date"]] for s in hs + ls)
                top = max(s["price"] for s in hs)
                bottom = min(s["price"] for s in ls)
                clustered = (max(s["price"] for s in hs) - min(s["price"] for s in hs) <= cfg.test_tolerance_atr * a
                             and max(s["price"] for s in ls) - min(s["price"] for s in ls) <= cfg.test_tolerance_atr * a)
                detached = tr is None or bottom > tr.top or top < tr.bottom
                if (clustered and detached and i - left >= cfg.min_range_bars
                        and top - bottom >= cfg.range_min_height_atr * a and bottom <= close[i] <= top):
                    create(i, left, bottom, top, "ohlcv_swings")
                    last_range_end = left
        if tr is None or (tr.confirmed_date is not None and dates[i] < tr.confirmed_date):
            notify(i)
            continue
        inside = tr.bottom <= close[i] <= tr.top
        if inside:
            tr = replace(tr, cause_end_date=dates[i])
            ranges[-1] = tr
        lower = low[i] < tr.bottom - cfg.boundary_buffer_atr * a
        upper = high[i] > tr.top + cfg.boundary_buffer_atr * a
        mature = established_i is not None and i > established_i
        prior_inside = tr.bottom <= close[i - 1] <= tr.top
        if mature:
            for side, bull in (("accumulation", True), ("distribution", False)):
                excursion = excursions[side]
                breach = lower if bull else upper
                outside = close[i] < tr.bottom if bull else close[i] > tr.top
                if excursion:
                    value = low[i] if bull else high[i]
                    if value < excursion["price"] if bull else value > excursion["price"]:
                        excursion["price"], excursion["extreme_i"] = value, i
                    excursion["max_volume_ratio"] = max(excursion["max_volume_ratio"], vr[i])
                    opposite = upper if bull else lower
                    excursion["max_volume"] = max(excursion["max_volume"], volume[i])
                    if opposite:
                        emit("INVALIDATION", i, close[i], side, invalidated_event="BOUNDARY_EXCURSION",
                             invalidated_event_date=dates[excursion["i"]], status="confirmed",
                             reason="opposite_boundary_breach")
                        excursions[side] = None
                    elif inside:
                        recovered = (close[i] > tr.bottom + cfg.boundary_buffer_atr * a and cp[i] >= cfg.climax_recovery_close
                                     if bull else close[i] < tr.top - cfg.boundary_buffer_atr * a and cp[i] <= climax_close_max)
                        recovered = recovered and vr[i] <= cfg.probe_volume_ratio_max and (
                            excursion["max_volume_ratio"] <= cfg.probe_volume_ratio_max or volume[i] < excursion["max_volume"])
                        if recovered:
                            name = "SPRING" if bull else ("UTAD" if initiative[side] and i - established_i >= cfg.min_range_bars else "UT")
                            pending_test[side] = emit(name, i, excursion["price"], side, excursion["extreme_i"],
                                subtype="multi_bar_recovery", parent_date=dates[excursion["i"]],
                                recovery_date=dates[i], range_confirmed_at=tr.established_date or tr.confirmed_date,
                                evidence=["finite_boundary_excursion", "recovery_inside", "reduced_or_limited_recovery_effort"], test_confirmed=False)
                            watched[side] = pending_test[side]
                            initiative[side] = None
                        else:
                            emit("INVALIDATION", i, close[i], side, invalidated_event="BOUNDARY_EXCURSION",
                                 invalidated_event_date=dates[excursion["i"]], status="confirmed", reason="weak_boundary_recovery")
                        excursions[side] = None
                elif prior_inside and breach and outside and not (lower and upper):
                    excursions[side] = {"i": i, "extreme_i": i, "price": low[i] if bull else high[i],
                                        "max_volume_ratio": vr[i], "max_volume": volume[i]}
                    emit("BOUNDARY_EXCURSION", i, low[i] if bull else high[i], side,
                         recovery_deadline_bars=cfg.local_trend_bars, evidence=["close_outside", "recovery_not_yet_observed"])
        if mature and inside and (prior_inside or (lower and upper)):
            if lower and upper:
                emit("AMBIGUOUS_TEST", i, close[i], evidence=["both_boundaries_breached"])
                pending_test = {"accumulation": None, "distribution": None}
                initiative = {"accumulation": None, "distribution": None}
                watched = {"accumulation": None, "distribution": None}
                interior_since = {"accumulation": i, "distribution": i}
                wave_floor = {"accumulation": i, "distribution": i}
                excursions = {"accumulation": None, "distribution": None}
            elif lower and vr[i] <= cfg.probe_volume_ratio_max:
                pending_test["accumulation"] = emit("SPRING", i, low[i], "accumulation",
                    evidence=["support_breach", "recovery_inside"], test_confirmed=False)
                watched["accumulation"] = pending_test["accumulation"]
                initiative["accumulation"] = None
            elif upper and vr[i] <= cfg.probe_volume_ratio_max:
                # A UT in B is ordinary range testing. UTAD is a later *candidate*
                # after prior weakness; neither alone certifies a Phase C.
                weak = initiative["distribution"] is not None
                name = "UTAD" if weak and i - established_i >= cfg.min_range_bars else "UT"
                pending_test["distribution"] = emit(name, i, high[i], "distribution",
                    evidence=["resistance_breach", "recovery_inside"], test_confirmed=False)
                watched["distribution"] = pending_test["distribution"]
                initiative["distribution"] = None
        for swing in new_swings:
            j = positions[swing["date"]]
            if j <= start_i or not np.isfinite(vr[j]) or volume[j] <= 0:
                continue
            for side, bull in (("accumulation", True), ("distribution", False)):
                if swing["kind"] != ("low" if bull else "high"):
                    continue
                # The watch pass precedes this bar's new confirmations. Do
                # not install support that is already beyond its opposite
                # range boundary; later reentry needs a new valid pivot.
                adverse_close = close[i] < tr.bottom if bull else close[i] > tr.top
                if adverse_close:
                    continue
                test = pending_test[side]
                if test and dates[j] > available(test):
                    holds = low[j] > test["price"] if bull else high[j] < test["price"]
                    near = low[j] <= tr.bottom + cfg.test_tolerance_atr * a if bull else high[j] >= tr.top - cfg.test_tolerance_atr * a
                    if holds and near and vr[j] < test["vol"] and sr[j] <= cfg.probe_test_spread_max:
                        parent_baseline = features["volume_baseline"].iloc[positions[test["date"]]]
                        test_baseline = features["volume_baseline"].iloc[j]
                        watched[side] = emit("TEST" if bull else "TEST_DEMAND", i, swing["price"], side, j,
                             status="confirmed", parent_date=test["date"], parent_volume=test.get("volume"),
                             parent_volume_ratio=test.get("volume_ratio"),
                             parent_volume_baseline=float(parent_baseline) if np.isfinite(parent_baseline) else None,
                             test_volume_baseline=float(test_baseline) if np.isfinite(test_baseline) else None,
                             effort_comparison="prior_baseline_relative_ratio",
                             evidence=["confirmed_pivot", "relative_effort_below_probe", "holds_test_extreme"])
                        pending_test[side] = None
                interior = interior_test(i, j, swing, side, bull)
                if interior is not None:
                    watched[side] = interior
                move = initiative[side]
                if move and dates[j] > move["date"]:
                    boundary = tr.top if bull else tr.bottom
                    near = abs(swing["price"] - boundary) <= cfg.test_tolerance_atr * a
                    holds_range = low[j] > tr.bottom if bull else high[j] < tr.top
                    reaction = close[j] < close[j - 1] if bull else close[j] > close[j - 1]
                    quiet = vr[j] <= cfg.lps_volume_ratio_max and sr[j] <= cfg.lps_spread_ratio_max
                    blocked_rally = not bull and vr[j] >= cfg.volume_high and features["price_change_atr"].iloc[j] <= cfg.lpsy_poor_result_max_atr
                    # Interior HL/LH is also valid; a former boundary test is a
                    # narrower subtype. High effort/poor result LPSY is separate.
                    if holds_range and reaction and (quiet or blocked_rally):
                        watched[side] = emit("LPS" if bull else "LPSY", i, swing["price"], side, j,
                             status="confirmed", parent_date=move["date"], scope=move["scope"],
                             subtype="low_effort_reaction" if quiet else "high_effort_poor_result",
                             evidence=["confirmed_reaction_pivot", "range_holds", "reduced_effort" if quiet else "poor_upward_result"])
                        if near and move["scope"] == "breakout":
                            emit("BU" if bull else "BACKUP_ICE", i, swing["price"], side, j,
                                 status="confirmed", parent_date=move["date"], evidence=["former_boundary_retest"])
                        initiative[side] = None
        spread_move = (close[i] - close[i - 1]) / a
        strong_up = (vr[i] >= cfg.volume_high and sr[i] >= cfg.initiative_spread_min
                     and cp[i] >= cfg.initiative_close_min and spread_move >= cfg.initiative_move_min_atr)
        strong_down = (vr[i] >= cfg.volume_high and sr[i] >= cfg.initiative_spread_min
                       and cp[i] <= initiative_close_max and spread_move <= -cfg.initiative_move_min_atr)
        breakout_up = close[i] > tr.top and close[i - 1] <= tr.top
        breakout_down = close[i] < tr.bottom and close[i - 1] >= tr.bottom
        for side, name, strong, boundary_break, location in (
                ("accumulation", "SOS", strong_up, breakout_up, close[i] >= tr.mid),
                ("distribution", "SOW", strong_down, breakout_down, close[i] <= tr.mid)):
            wave = wave_evidence(i, side == "accumulation") if mature else None
            if mature and ((strong and location and (inside or boundary_break)) or wave):
                recent = initiative[side]
                wave_break = wave and wave["scope"] == "breakout" and (recent is None or recent["scope"] != "breakout")
                if boundary_break or wave_break or recent is None or i - positions[recent["date"]] >= cfg.min_range_bars:
                    if wave and not (strong and location and (inside or boundary_break)):
                        initiative[side] = emit(name, i, close[i], side, **wave)
                        used_waves.add((side, positions[wave["wave_start_date"]]))
                    else:
                        scope = "breakout" if boundary_break else "internal"
                        initiative[side] = emit(name, i, close[i], side, scope=scope,
                            evidence=["wide_spread", "high_volume", "directional_result", scope])
                    opposite = "distribution" if side == "accumulation" else "accumulation"
                    initiative[opposite] = None
                    pending_test[opposite] = None
                    watched[opposite] = None
                    interior_since[opposite] = i
                    wave_floor[opposite] = i
        update_geometry(i)
        notify(i)
    return {"tr": tr, "events": sorted(events, key=lambda e: (available(e), e["date"])),
            "ranges": ranges, "transitions": transitions, "stopping_candidate": active_stop()}
