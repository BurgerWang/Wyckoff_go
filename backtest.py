#!/usr/bin/env python3
"""P&F direction-timing backtest + parameter grid search.

Historical research module, separate from the analysis application.
P&F construction uses the same primitives as the charting tool so research
and descriptive analysis agree on box, reversal and column semantics.

Strategy: hold long while the P&F column is X, flat while O.
Box sizes are computed from data up to each day only (rolling ATR median /
percentage of that day's price / traditional table) -> no lookahead.
Incremental P&F updates keep the grid search fast.

Usage: python3 backtest.py GOOG AAPL MSFT SPY
"""

import argparse

import numpy as np
import pandas as pd

from market_data import fetch as fetch_market, validate_ohlcv
from pnf import IncrementalPnF, make_scale, atr14, _idx_floor, _idx_ceil, _extend_scale


def fetch(symbol: str, years: float = 5, tf: str = "1d") -> pd.DataFrame:
    """Compatibility research download entry, using the shared data contract."""
    return fetch_market(symbol, years, tf)


# ---------------------------------------------------------------- incremental P&F
class IncPnF(IncrementalPnF):
    """Compatibility name for the shared high/low P&F engine."""

    def __init__(self, scale: np.ndarray, reversal: int = 3):
        super().__init__(scale, reversal=reversal, use_close=False)


# ---------------------------------------------------------------- box strategies
def box_at(df: pd.DataFrame, t: int, kind: str, param: float) -> float:
    """Legacy experimental size as of bar t, preserving its $1 ATR floor.

    The descriptive application uses pnf.build_config without this research
    floor. Retaining it here does not revalidate old research conclusions.
    """
    if kind == "atr":
        seg = df.iloc[: t + 1]
        a = atr14(seg)
        return max(1.0, round(float(a.median()) * param, 2))
    if kind == "pct":
        return max(0.05, float(df["Close"].iloc[t]) * param)
    raise ValueError(kind)


def make_scale_for(df: pd.DataFrame, t: int, kind: str, param: float) -> np.ndarray:
    if kind == "trad":
        return make_scale(
            float(df.iloc[: t + 1][["Low", "Close"]].min().min()),
            float(df.iloc[: t + 1][["High", "Close"]].max().max()),
            "traditional",
        )
    lo = float(df.iloc[: t + 1][["Low", "Close"]].min().min())
    hi = float(df.iloc[: t + 1][["High", "Close"]].max().max())
    if kind == "pct":
        # geometric grid, consistent with pnf.py (audit fix: pct was a
        # periodic fixed grid via box_at)
        return make_scale(lo, hi, "pct", pct=param)
    return make_scale(lo, hi, "fixed", box=box_at(df, t, kind, param))


# ---------------------------------------------------------------- backtest
REBAL = 21  # box/scale 月度重估（只用截至当天的数据，无未来函数）


def _check_window(warmup: int, horizons=()):
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integers")


def _check_periods(periods_per_year):
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite")


def _advance_pnf(df, t, p, kind, param, reversal):
    """Update the causal grid and distinguish a bar reversal from regridding."""
    if p is None or t % REBAL == 0:
        scale = make_scale_for(df, t, kind, param)
        if p is None or not np.array_equal(p.scale, scale):
            p = IncPnF(scale, reversal)
            for tt in range(max(0, t - REBAL * 3), t):
                row = df.iloc[tt]
                p.update(
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    df.index[tt],
                )
    previous = p.direction
    row = df.iloc[t]
    p.update(float(row["High"]), float(row["Low"]), float(row["Close"]), df.index[t])
    return p, previous is not None and p.direction != previous


def backtest(
    df: pd.DataFrame,
    kind: str,
    param: float,
    reversal: int,
    warmup: int = 60,
    periods_per_year: float = 252,
) -> dict:
    """Direction timing: long while X, flat while O.
    Signal at day-t close, executed at day-(t+1) open (no lookahead).
    Equity: compounded, three-segment returns aligned with the open fills
    (buy day: open->close, sell day: prev close->open; audit fix)."""
    _check_window(warmup)
    _check_periods(periods_per_year)
    if df is not None and len(df):
        df = validate_ohlcv(df, require_volume=False)
    if df is None or len(df) < warmup + 10:
        return {
            "strat": 0.0,
            "buyhold": 0.0,
            "excess": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "avg_hold_days": 0.0,
            "max_dd": 0.0,
            "sharpe": 0.0,
            "signals": 0,
        }
    p = None
    pos, entry_px, entry_date = 0, 0.0, None
    pending = None
    eq = np.ones(len(df))
    trades = []
    opens = df["Open"].values
    closes = df["Close"].values
    n = len(df)
    for t in range(n):
        # ---- yesterday's holding period return (up to today's open/close)
        if t > 0:
            if pos == 1:
                if pending is not None and pending == 0:
                    eq[t] = eq[t - 1] * (opens[t] / closes[t - 1])  # sold at open
                else:
                    eq[t] = eq[t - 1] * (closes[t] / closes[t - 1])  # held all day
            else:
                eq[t] = eq[t - 1]
        # ---- execute signal at today's open (from yesterday's close)
        if t > 0 and pending is not None:
            open_t = float(opens[t])
            if pending != pos:  # only record on direction change
                if pos == 1:
                    trades.append((entry_date, df.index[t], entry_px, open_t))
                pos = pending
                if pos == 1:
                    entry_px, entry_date = open_t, df.index[t]
                    eq[t] *= closes[t] / opens[t]  # bought at open, day return
            pending = None
        # ---- update P&F with today's bar
        p, _ = _advance_pnf(df, t, p, kind, param, reversal)
        # ---- signal at today's close for tomorrow (P&F includes today)
        if t >= warmup and p is not None:
            target = 1 if p.direction == "X" else 0
            if target != pos:
                pending = target
    if pos == 1:
        trades.append((entry_date, df.index[n - 1], entry_px, float(closes[n - 1])))
    eqs = pd.Series(eq[warmup:])
    dd = float(((eqs.cummax() - eqs) / eqs.cummax()).max())
    daily = eqs.pct_change().dropna()  # compound daily returns
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(periods_per_year))
        if daily.std() > 0
        else 0.0
    )
    buyhold = float(df["Close"].iloc[-1] / df["Close"].iloc[warmup] - 1)
    strat = eq[-1] - 1.0
    wins = [1 if tr[3] > tr[2] else 0 for tr in trades]
    rets = [tr[3] / tr[2] - 1 for tr in trades]
    hold_days = [(tr[1] - tr[0]).days for tr in trades]
    return {
        "strat": strat,
        "buyhold": buyhold,
        "excess": strat - buyhold,
        "n_trades": len(trades),
        "win_rate": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_ret": float(np.mean(rets)) if rets else 0.0,
        "avg_hold_days": float(np.mean(hold_days)) if hold_days else 0.0,
        "max_dd": dd,
        "sharpe": sharpe,
        "signals": len(trades),
    }


# ---------------------------------------------------------------- event study
def event_study(
    df: pd.DataFrame,
    kind: str,
    param: float,
    reversal: int,
    horizons=(5, 10, 20),
    warmup: int = 60,
) -> dict:
    """Signal accuracy: for every column reversal, check whether the next
    N-day move agreed with the new column direction (X=up, O=down).
    Baseline = direction probability of non-event days; X events are judged
    against base_up, O events against base_down (audit fix)."""
    _check_window(warmup, horizons)
    if df is not None and len(df):
        df = validate_ohlcv(df, require_volume=False)
    out = {
        "kind": f"{kind}:{param:.3g}" if param else kind,
        "rev": reversal,
        "n_events": 0,
    }
    for h in horizons:
        for metric in ("base_ok", "base_down", "X_ok", "O_ok", "X_avg", "O_avg"):
            out[f"{metric}_{h}d"] = float("nan")
    if df is None or len(df) < warmup + 10:
        return out
    p = None
    events = []  # (date_idx, new_dir, price)
    for t in range(len(df)):
        p, reversed_today = _advance_pnf(df, t, p, kind, param, reversal)
        if t >= warmup and reversed_today:
            events.append((t, p.direction, float(df["Close"].iloc[t])))
    closes = df["Close"].values
    idxs = df.index
    rows = []
    for t, d, px in events:
        for h in horizons:
            if t + h >= len(df):
                continue
            fwd = closes[t + h] / closes[t] - 1
            rows.append({"event": d, "horizon": h, "fwd": fwd, "date": idxs[t]})
    ev = pd.DataFrame(rows, columns=["event", "horizon", "fwd", "date"])
    out["n_events"] = len(events)
    ev_days = {t for t, _, _ in events}
    for h in horizons:
        sub = ev[ev["horizon"] == h]
        x = sub[sub["event"] == "X"]["fwd"]
        o = sub[sub["event"] == "O"]["fwd"]
        # baseline: non-event days only, same horizon (audit fix)
        base = [
            closes[t + h] / closes[t] - 1
            for t in range(warmup, len(df) - h)
            if t not in ev_days
        ]
        b_up = float(np.mean([b > 0 for b in base])) if base else float("nan")
        x_ok = float((x > 0).mean()) if len(x) else float("nan")
        o_ok = float((o < 0).mean()) if len(o) else float("nan")
        x_avg = float(x.mean()) if len(x) else float("nan")
        o_avg = float(o.mean()) if len(o) else float("nan")
        out[f"base_ok_{h}d"] = b_up
        out[f"base_down_{h}d"] = (
            float(np.mean([b < 0 for b in base])) if base else float("nan")
        )
        out[f"X_ok_{h}d"] = x_ok
        out[f"O_ok_{h}d"] = o_ok
        out[f"X_avg_{h}d"] = x_avg
        out[f"O_avg_{h}d"] = o_avg
    return out


def run_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, param in GRID:
        for rev in REVERSALS:
            rows.append(event_study(df, kind, param, rev))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- wyckoff
def run_wyckoff(
    df: pd.DataFrame,
    kind: str,
    param: float,
    reversal: int,
    warmup: int = 90,
    horizons=(20, 60),
) -> dict:
    """Daily: incremental P&F + Wyckoff analysis -> event log + phase state
    machine backtest. No lookahead: analysis uses data up to day t only.
    Equity: compound, three-segment returns (audit fix)."""
    import wyckoff as wk

    _check_window(warmup, horizons)
    if df is not None and len(df):
        df = validate_ohlcv(df)
    if df is None or len(df) < warmup + 10:
        return {
            "n_events": 0,
            "events": [],
            "event_details": [],
            "stats": {"h": [], "base_up": [], "base_down": [], "n": {}, "ok": {}},
            "strat": 0.0,
            "buyhold": 0.0,
            "excess": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "max_dd": 0.0,
            "phase_days": {},
        }
    p = IncPnF(make_scale_for(df, min(warmup, len(df) - 1), kind, param), reversal)
    closes = df["Close"].values
    opens = df["Open"].values
    n = len(df)
    events_log = []  # (availability bar, event, pivot price)
    event_details = []
    phases = [None] * n  # regime per day
    eq = np.ones(n)
    pos = 0
    entry_px, entry_t = None, None
    trades = []
    pending = None
    for t in range(n):
        row = df.iloc[t]
        # ---- yesterday's holding period return
        if t > 0:
            if pos == 1:
                if pending is not None and pending == 0:
                    eq[t] = eq[t - 1] * (opens[t] / closes[t - 1])
                else:
                    eq[t] = eq[t - 1] * (closes[t] / closes[t - 1])
            else:
                eq[t] = eq[t - 1]
        # ---- execute pending signal at today's open
        if t > 0 and pending is not None:
            open_t = float(opens[t])
            if pending != pos:
                if pos == 1:
                    trades.append((df.index[entry_t], df.index[t], entry_px, open_t))
                pos = pending
                if pos == 1:
                    entry_px, entry_t = open_t, t
                    eq[t] *= closes[t] / opens[t]
            pending = None
        # ---- update incremental P&F
        p.update(
            float(row["High"]), float(row["Low"]), float(row["Close"]), df.index[t]
        )
        if t < warmup:
            continue
        # ---- Wyckoff analysis on data up to t (no lookahead)
        res = wk.analyze(p.columns, p.scale, df.iloc[: t + 1], reversal)
        today = df.index[t]
        for e in res["events"]:
            if e.get("available_date", e["date"]) != today or e["event"] not in (
                "SPRING",
                "UT",
                "SOS",
                "SOW",
                "SC",
                "BC",
                "AR",
                "LPS",
                "LPSY",
            ):
                continue
            tag = e["event"]
            # AR is a rally after SC and a reaction after BC. Unknown-side
            # legacy observations are logged, but excluded from direction stats.
            if tag == "AR" and e.get("side") in ("accumulation", "distribution"):
                tag = "AR_UP" if e["side"] == "accumulation" else "AR_DOWN"
            events_log.append((t, tag, float(e["price"])))
            event_details.append(
                {
                    **e,
                    "t": t,
                    "event": tag,
                    "price": float(e["price"]),
                    "date": e["date"],
                    "available_date": e.get("available_date", e["date"]),
                }
            )
        regime = res["phase"]["regime"]
        phases[t] = regime
        target = 1 if regime in ("accumulation", "markup") else 0
        if target != pos:
            pending = target
    if pos == 1:
        trades.append(
            (df.index[entry_t], df.index[n - 1], entry_px, float(closes[n - 1]))
        )
    # ---- event study: direction accuracy vs no-signal baseline
    bull_events = ("SPRING", "SOS", "SC", "LPS", "AR_UP")
    bear_events = ("UT", "SOW", "BC", "LPSY", "AR_DOWN")
    stats = {"h": [], "base_up": [], "base_down": [], "n": {}, "ok": {}}
    ev_days = {t for t, _, _ in events_log}
    for h in horizons:
        base = [
            closes[t + h] / closes[t] - 1
            for t in range(warmup, n - h)
            if t not in ev_days
        ]
        b_up = float(np.mean([b > 0 for b in base])) if base else float("nan")
        b_down = float(np.mean([b < 0 for b in base])) if base else float("nan")
        acc = {}
        for ev in bull_events + bear_events:
            expect_up = ev in bull_events
            hit = [
                1
                if (
                    closes[t + h] > closes[t]
                    if expect_up
                    else closes[t + h] < closes[t]
                )
                else 0
                for t, e, px in events_log
                if e == ev and t + h < n
            ]
            if hit:
                acc[ev] = (len(hit), float(np.mean(hit)))
        stats["h"].append(h)
        stats["base_up"].append(b_up)
        stats["base_down"].append(b_down)
        stats["n"][h] = {k: v[0] for k, v in acc.items()}
        stats["ok"][h] = {k: v[1] for k, v in acc.items()}
    # ---- phase state machine performance
    rets = [tr[3] / tr[2] - 1 for tr in trades]
    wins = [r > 0 for r in rets]
    dd = 0.0
    peak = -1e18
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    return {
        "n_events": len(events_log),
        "events": events_log,
        "event_details": event_details,
        "stats": stats,
        "strat": eq[-1] - 1.0,
        "buyhold": float(closes[-1] / closes[warmup] - 1),
        "excess": eq[-1] - 1.0 - float(closes[-1] / closes[warmup] - 1),
        "n_trades": len(trades),
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "avg_ret": float(np.mean(rets)) if rets else 0.0,
        "max_dd": dd,
        "phase_days": {r: sum(1 for x in phases if x == r) for r in set(phases) if r},
    }


# ------------------------------------------------- Part 1 faithful trading sim
def barrier_exit(open_price, high, low, direction, stop, target):
    """OHLC exit fill: open gaps fill at open; ambiguous bars hit stop first.

    Daily bars cannot reveal which intrabar barrier was touched first. The
    stop-first convention is deliberately conservative and shared by studies.
    """
    if direction == 1:
        if open_price <= stop:
            return float(open_price), "SL"
        if open_price >= target:
            return float(open_price), "TP"
        if low <= stop:
            return float(stop), "SL"
        if high >= target:
            return float(target), "TP"
    elif direction == -1:
        if open_price >= stop:
            return float(open_price), "SL"
        if open_price <= target:
            return float(open_price), "TP"
        if high >= stop:
            return float(stop), "SL"
        if low <= target:
            return float(target), "TP"
    else:
        raise ValueError("direction must be 1 (long) or -1 (short)")
    return None


def run_wyckoff_trade(
    df: pd.DataFrame,
    kind: str,
    param: float,
    reversal: int = 3,
    warmup: int = 120,
    atr_mult: float = 1.5,
    rr: float = 2.0,
    lps_bars: int = 15,
    range_watch: int = 60,
    periods_per_year: float = 252,
) -> dict:
    """Part 1 (MQL5) faithful trade backtest.

    Real-time single-instance FSM, bias-isolated, TR locked at detection,
    spring invalidation (close < spring low -> IDLE), LPS/LPSY entry
    (lps_bars window: close pulled back inside range, non-loud volume),
    SL = entry -/+ ATR*1.5, TP = entry +/- 2R (RR=2). Longs and shorts."""
    import wyckoff as wk

    _check_window(warmup)
    _check_periods(periods_per_year)
    if not np.isfinite(atr_mult) or atr_mult <= 0 or not np.isfinite(rr) or rr <= 0:
        raise ValueError(
            "ATR multiplier and risk/reward ratio must be positive and finite"
        )
    if df is not None and len(df):
        df = validate_ohlcv(df)
    if df is None or len(df) < warmup + 20:
        return {
            "strat": 0.0,
            "buyhold": 0.0,
            "excess": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "avg_hold_days": 0.0,
            "max_dd": 0.0,
            "sharpe": 0.0,
            "signals": 0,
            "trades": [],
        }
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    vols = df["Volume"].values
    atr = atr14(df).values

    p = IncPnF(make_scale_for(df, warmup, kind, param), reversal)
    for tt in range(warmup + 1):
        rr_ = df.iloc[tt]
        p.update(
            float(rr_["High"]), float(rr_["Low"]), float(rr_["Close"]), df.index[tt]
        )

    state = "IDLE"
    tr = None  # locked trading range
    spring_low = ut_high = 0.0
    sos_high = sow_low = 0.0
    lps_cnt = lpsy_cnt = 0
    range_watch_cnt = 0
    pos = 0  # +1 long / -1 short
    entry_px = entry_t = 0.0
    entry_capital = 1.0
    sl = tp = 0.0
    pending = None  # (direction, known risk distance), next open
    trades = []
    eq = np.ones(len(df))

    for t in range(warmup + 1, len(df)):
        o, h, l, c = opens[t], highs[t], lows[t], closes[t]
        eq[t] = eq[t - 1]
        # ---- execute pending at today's open
        if pending is not None:
            direction, risk = pending
            entry_px, entry_t = o, t
            entry_capital = eq[t - 1]
            pos = 1 if direction == "LONG" else -1
            sl, tp = o - pos * risk, o + pos * risk * rr
            pending = None
        # ---- in-trade: intra-bar SL/TP
        if pos:
            exit_fill = barrier_exit(o, h, l, pos, sl, tp)
            mark = exit_fill[0] if exit_fill else c
            ret = pos * (mark / entry_px - 1)
            eq[t] = entry_capital * (1 + ret)
            if exit_fill is not None:
                exit_px, reason = exit_fill
                trades.append(
                    (df.index[entry_t], df.index[t], entry_px, exit_px, reason, ret)
                )
                pos = 0
                state = "IDLE"
                tr = None
        # ---- update P&F with today's bar (for TR detection at IDLE)
        row = df.iloc[t]
        p.update(
            float(row["High"]), float(row["Low"]), float(row["Close"]), df.index[t]
        )
        # ---- FSM transitions on today's close
        if state == "IDLE":
            tr = wk.detect_tr(p.columns, p.box, df.iloc[: t + 1])
            if tr is not None and tr.bias != "unknown" and tr.trend_bars:
                state = "RANGE"
                range_watch_cnt = 0
                lps_cnt = lpsy_cnt = 0
        elif state == "RANGE":
            range_watch_cnt += 1
            if range_watch_cnt > range_watch:
                state = "IDLE"
                tr = None
            elif tr.bias == "accumulation":
                if l < tr.bottom and c > tr.bottom and vols[t] < tr.avg_volume * 1.2:
                    state = "SPRING"
                    spring_low = l
            else:
                if h > tr.top and c < tr.top and vols[t] < tr.avg_volume * 1.2:
                    state = "UT"
                    ut_high = h
        elif state == "SPRING":
            if c < spring_low:
                state = "IDLE"
                tr = None  # invalidation
            elif c > tr.top and vols[t] > tr.avg_volume * 1.2:
                state = "SOS"
                sos_high = h
                lps_cnt = 0
        elif state == "UT":
            if c > ut_high:
                state = "IDLE"
                tr = None
            elif c < tr.bottom and vols[t] > tr.avg_volume * 1.2:
                state = "SOW"
                sow_low = l
                lpsy_cnt = 0
        elif state == "SOS":
            if c < spring_low:
                state = "IDLE"
                tr = None
            else:
                lps_cnt += 1
                if lps_cnt > lps_bars:
                    state = "IDLE"
                    tr = None
                elif c < sos_high and c > tr.bottom and vols[t] < tr.avg_volume * 1.2:
                    a = atr[t] if atr[t] > 0 else (tr.top - tr.bottom) / 3
                    pending = ("LONG", a * atr_mult)
                    state = "IN_TRADE"
        elif state == "SOW":
            if c > ut_high:
                state = "IDLE"
                tr = None
            else:
                lpsy_cnt += 1
                if lpsy_cnt > lps_bars:
                    state = "IDLE"
                    tr = None
                elif c > sow_low and c < tr.top and vols[t] < tr.avg_volume * 1.2:
                    a = atr[t] if atr[t] > 0 else (tr.top - tr.bottom) / 3
                    pending = ("SHORT", a * atr_mult)
                    state = "IN_TRADE"

    # close any open position at the end
    if pos != 0:
        ret = (closes[-1] / entry_px - 1) if pos == 1 else (1 - closes[-1] / entry_px)
        trades.append(
            (df.index[entry_t], df.index[len(df) - 1], entry_px, closes[-1], "END", ret)
        )
    eqs = pd.Series(eq[warmup:])
    dd = float(((eqs.cummax() - eqs) / eqs.cummax()).max())
    daily = eqs.pct_change().dropna()
    sharpe = (
        float(daily.mean() / daily.std() * np.sqrt(periods_per_year))
        if daily.std() > 0
        else 0.0
    )
    strat = eq[-1] - 1.0
    buyhold = float(closes[-1] / closes[warmup] - 1)
    wins = [1 if tr_[5] > 0 else 0 for tr_ in trades]
    rets = [tr_[5] for tr_ in trades]
    hold_days = [(tr_[1] - tr_[0]).days for tr_ in trades]
    return {
        "strat": strat,
        "buyhold": buyhold,
        "excess": strat - buyhold,
        "n_trades": len(trades),
        "win_rate": sum(wins) / len(wins) if wins else 0.0,
        "avg_ret": sum(rets) / len(rets) if rets else 0.0,
        "avg_hold_days": sum(hold_days) / len(hold_days) if hold_days else 0.0,
        "max_dd": dd,
        "sharpe": sharpe,
        "signals": len(trades),
        "trades": trades,
    }


# ---------------------------------------------------------------- grid search
GRID = [
    ("atr", 0.25),
    ("atr", 0.5),
    ("atr", 0.75),
    ("atr", 1.0),
    ("atr", 1.5),
    ("atr", 2.0),
    ("pct", 0.005),
    ("pct", 0.01),
    ("pct", 0.015),
    ("pct", 0.02),
    ("pct", 0.03),
    ("pct", 0.04),
    ("trad", 0.0),
]
REVERSALS = [2, 3, 4, 5]


def run_grid(
    df: pd.DataFrame, revs: list[int] = REVERSALS, periods_per_year: float = 252
) -> pd.DataFrame:
    rows = []
    for kind, param in GRID:
        for rev in revs:
            s = backtest(df, kind, param, rev, periods_per_year=periods_per_year)
            rows.append(
                {
                    "kind": f"{kind}" + (f":{param:.3g}" if param else ""),
                    "rev": rev,
                    **s,
                }
            )
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=["GOOG"])
    parser.add_argument("--years", type=float, default=10)
    parser.add_argument("--tf", choices=("1d", "4h"), default="1d")
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=252,
        help="annualization bars/year; default preserves daily-stock research convention",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--events", action="store_true")
    modes.add_argument("--wyckoff", action="store_true")
    modes.add_argument("--wk-trade", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.years) or args.years <= 0:
        parser.error("--years must be positive and finite")
    if not np.isfinite(args.periods_per_year) or args.periods_per_year <= 0:
        parser.error("--periods-per-year must be positive and finite")
    syms = args.symbols or ["GOOG"]
    years, tf = args.years, args.tf
    trade_mode, wk_mode, ev_mode = args.wk_trade, args.wyckoff, args.events
    if trade_mode:
        print(f"===== Part 1 完整交易回测 ({years}y, {tf}) =====")
        all_rows = []
        for sym in syms:
            df = fetch(sym, years, tf)
            for kind, param in (("atr", 0.5), ("pct", 0.01)):
                r = run_wyckoff_trade(
                    df, kind, param, periods_per_year=args.periods_per_year
                )
                print(
                    f"\n{sym} [{kind}:{param}] Part1: "
                    f"strat {r['strat']:.1%} vs buyhold {r['buyhold']:.1%} "
                    f"(excess {r['excess']:.1%}) | 交易 {r['n_trades']} 次, "
                    f"胜率 {r['win_rate']:.1%}, 平均 {r['avg_ret']:.2%}, "
                    f"maxDD {r['max_dd']:.1%}"
                )
                row = {
                    "symbol": sym,
                    "kind": kind,
                    "excess": r["excess"],
                    "win": r["win_rate"],
                    "trades": r["n_trades"],
                    "dd": r["max_dd"],
                }
                all_rows.append(row)
        if all_rows:
            agg = (
                pd.DataFrame(all_rows)
                .groupby("kind")[["excess", "win", "trades"]]
                .mean()
            )
            print("\n===== 汇总 =====")
            print(agg.round(2).to_string())
        return 0
    if wk_mode:
        all_rows = []
        print(f"===== Wyckoff 信号研究 ({years}y) =====")
        for sym in syms:
            df = fetch(sym, years, tf)
            for kind, param in (("atr", 0.5), ("pct", 0.01)):
                r = run_wyckoff(df, kind, param, 3)
                print(
                    f"\n{sym} [{kind}:{param}] 状态机: "
                    f"strat {r['strat']:.1%} vs buyhold {r['buyhold']:.1%} "
                    f"(excess {r['excess']:.1%}) | 交易 {r['n_trades']} 次, "
                    f"胜率 {r['win_rate']:.1%}, 平均 {r['avg_ret']:.2%}, "
                    f"maxDD {r['max_dd']:.1%} | 事件 {r['n_events']}"
                )
                print(f"  阶段分布: {r['phase_days']}")
                for i, h in enumerate(r["stats"]["h"]):
                    bu = r["stats"]["base_up"][i]
                    bd = r["stats"]["base_down"][i]
                    parts = [f"基准↑{bu:.1%}/↓{bd:.1%}"]
                    for ev in ("SPRING_C", "SOS_C", "LPS_C", "SOW_C"):
                        if ev in r["stats"]["ok"][h]:
                            n = r["stats"]["n"][h][ev]
                            ok = r["stats"]["ok"][h][ev]
                            parts.append(f"{ev}:{ok:.0%}(n={n})")
                    for ev in ("SPRING", "SOS", "UT", "SOW"):
                        if ev in r["stats"]["ok"][h]:
                            n = r["stats"]["n"][h][ev]
                            ok = r["stats"]["ok"][h][ev]
                            parts.append(f"孤{ev}:{ok:.0%}(n={n})")
                    print(f"  后{h}日: " + " | ".join(parts))
                row = {
                    "symbol": sym,
                    "kind": kind,
                    "excess": r["excess"],
                    "win": r["win_rate"],
                    "trades": r["n_trades"],
                    "dd": r["max_dd"],
                    "events": r["n_events"],
                }
                all_rows.append(row)

        agg = pd.DataFrame(all_rows)
        print("\n===== 汇总 =====")
        print(
            agg.groupby("kind")
            .agg(
                excess_mean=("excess", "mean"),
                win_mean=("win", "mean"),
                n_trades_mean=("trades", "mean"),
            )
            .to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 3 else f"{x:.2f}")
        )
        return 0
    if ev_mode:
        all_ev = []
        for sym in syms:
            df = fetch(sym, years, tf)
            e = run_events(df)
            e.insert(0, "symbol", sym)
            all_ev.append(e)
            print(f"\n===== {sym} 信号准确性事件研究 ({years}y) =====")
            cols = [
                "kind",
                "rev",
                "n_events",
                "X_ok_5d",
                "X_ok_10d",
                "X_ok_20d",
                "O_ok_5d",
                "O_ok_10d",
                "O_ok_20d",
            ]
            print(
                e[cols].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.1%}" if abs(x) <= 1 else f"{x:.2f}",
                )
            )
        agg = pd.concat(all_ev)
        print("\n===== 汇总（全部标的事件合并，含无信号基准）=====")
        ac = (
            agg.groupby(["kind", "rev"])
            .agg(
                n=("n_events", "sum"),
                B5=("base_ok_5d", "mean"),
                X5=("X_ok_5d", "mean"),
                O5=("O_ok_5d", "mean"),
                B10=("base_ok_10d", "mean"),
                X10=("X_ok_10d", "mean"),
                O10=("O_ok_10d", "mean"),
                B20=("base_ok_20d", "mean"),
                X20=("X_ok_20d", "mean"),
                O20=("O_ok_20d", "mean"),
            )
            .reset_index()
        )
        cols = [
            "kind",
            "rev",
            "n",
            "B5",
            "X5",
            "O5",
            "B10",
            "X10",
            "O10",
            "B20",
            "X20",
            "O20",
        ]
        print(
            ac[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.1%}" if abs(x) <= 1 else f"{x:.0f}",
            )
        )
        return 0
    out = []
    for sym in syms:
        df = fetch(sym, years, tf)
        df.attrs["symbol"] = sym
        g = run_grid(df, periods_per_year=args.periods_per_year)
        g.insert(0, "symbol", sym)
        out.append(g)
        g = g.copy()
        g["avg_hold_days"] = g["avg_hold_days"].map(lambda x: f"{x:.1f}d")
        print(f"\n===== {sym} ({years}y, warmup 60d) =====")
        print(
            g.sort_values("excess", ascending=False).to_string(
                index=False,
                float_format=lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}",
            )
        )
    if len(out) > 1:
        allg = pd.concat(out)
        print("\n===== 汇总（按 excess 均值排序，跨标的稳健性）=====")
        agg = (
            allg.groupby(["kind", "rev"])
            .agg(
                excess_mean=("excess", "mean"),
                excess_std=("excess", "std"),
                win_mean=("win_rate", "mean"),
                n_trades_mean=("n_trades", "mean"),
            )
            .reset_index()
        )
        print(
            agg.sort_values("excess_mean", ascending=False)
            .head(10)
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.2%}" if abs(x) < 10 else f"{x:.2f}",
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
