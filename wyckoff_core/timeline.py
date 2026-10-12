"""Daily stage replay using the production state machine and classifier.

This intentionally omits P&F targets, relative strength and trade readiness.
Only confirmed pivots are exposed to each bar; future ledger mutations are
never stored in a historical row.
"""
from copy import deepcopy

from market_data import validate_ohlcv
from .context import _extend_alternating_swings, _range_structure, _range_swings, _structure
from .features import compute_features, confirmed_swings
from .models import CORE_VERSION, CoreConfig
from .stages import classify_phase
from .structure import replay_structure


def stage_timeline(df, config=None):
    """Return immutable per-bar phases and the final evidence ledger, offline."""
    cfg = CoreConfig() if config is None else config
    if not isinstance(cfg, CoreConfig):
        raise ValueError("config must be CoreConfig")
    df = validate_ohlcv(df)
    features = compute_features(df, cfg)
    swings = confirmed_swings(df, cfg)
    alternating, cursor, rows = [], 0, []
    range_alternating, structure_range = [], None
    positions = {date: i for i, date in enumerate(df.index)}
    prior_cache = {}
    previous_range, previous_side, outside = None, None, 0

    def observe(i, state):
        nonlocal cursor, previous_range, previous_side, outside, structure_range
        date = df.index[i]
        newly_known = []
        while cursor < len(swings) and swings[cursor]["available_date"] <= date:
            newly_known.append(swings[cursor])
            cursor += 1
        _extend_alternating_swings(alternating, newly_known)
        tr = state["tr"]
        context = {"price_structure": _structure(alternating),
                   "observation_eligible": bool(features["observation_eligible"].iloc[i]),
                   "stopping_candidate": state["stopping_candidate"],
                   "stop_window_start": df.index[max(0, i + 1 - cfg.trend_window)]}
        key = (tr.range_id, tr.start_date) if tr else None
        if key != structure_range:
            range_alternating.clear()
            structure_range = key
            local_swings = _range_swings(swings[:cursor], tr) if tr else []
        else:
            local_swings = _range_swings(newly_known, tr) if tr else []
        _extend_alternating_swings(range_alternating, local_swings)
        if tr is not None:
            context["range_price_structure"] = _range_structure(range_alternating, tr)
        if tr and tr.start_date in positions:
            key = (tr.range_id, tr.start_date)
            if key not in prior_cache:
                start = positions[tr.start_date]
                before = df.iloc[max(0, start - cfg.trend_window):start]
                atr = float(features["atr"].iloc[max(0, start - 1)])
                change = (float(before["Close"].iloc[-1] - before["Close"].iloc[0]) / atr
                          if len(before) >= cfg.trend_window and atr > 0 else None)
                direction = ("unknown" if change is None else "up" if change > cfg.trend_min_atr
                             else "down" if change < -cfg.trend_min_atr else "flat")
                prior_cache[key] = {"direction": direction, "change_atr": change,
                                    "reference": "completed_bars_before_range_start",
                                    "end_date": before.index[-1] if len(before) else None}
            context["range_prior_trend"] = prior_cache[key]
        close = float(df["Close"].iloc[i])
        range_key = (tr.range_id, tr.confirmed_date, tr.top, tr.bottom) if tr else None
        side = ("above" if close > tr.top else "below" if close < tr.bottom else None) if tr else None
        if side is None:
            outside = 0
        elif range_key == previous_range and side == previous_side:
            outside += 1
        else:
            outside = 0
            for t, price in df["Close"].iloc[:i + 1][::-1].items():
                if t < tr.confirmed_date or not (price > tr.top if side == "above" else price < tr.bottom):
                    break
                outside += 1
        previous_range, previous_side = range_key, side
        phase = classify_phase(state["events"], tr, [], close, outside, context, cfg)
        if i + 1 <= cfg.baseline_window:
            phase.update(phase="?", regime="unknown", status="insufficient_data", normal_no_observable=False,
                         notes=["量价基准尚未预热完成"])
        rows.append({"as_of": date, "close": close, "phase": deepcopy(phase),
                     "range_id": tr.range_id if tr else None,
                     "range_parent_id": tr.parent_range_id if tr else None,
                     "range_lifecycle": tr.lifecycle if tr else None,
                     "range_top": tr.top if tr else None, "range_bottom": tr.bottom if tr else None,
                     "stopping_candidate": deepcopy(state["stopping_candidate"]),
                     "outside_closes": outside})

    structure = replay_structure(df, features, swings, cfg, on_bar=observe)
    return {"rows": rows, "events": structure["events"], "ranges": structure["ranges"],
            "transitions": structure["transitions"], "config": cfg, "core_version": CORE_VERSION,
            "scope": "stage_only_production_classifier"}
