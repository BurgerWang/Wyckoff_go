"""P&F congestion and legacy pivot helpers retained for callers.

Production OHLCV lifecycle/events/stages live in structure/stages/engine.
The P&F congestion helper does not define production price boundaries.
"""
import math

import numpy as np
import pandas as pd

from .models import TR

def _validate_columns(columns, df=None):
    previous = None
    for c in columns:
        if len(c) != 5 or c[0] not in ("X", "O"):
            raise ValueError("P&F columns require [direction, low, high, bars, date]")
        if not np.isfinite([c[1], c[2]]).all() or c[1] > c[2]:
            raise ValueError("P&F column prices must be finite with low <= high")
        if previous is not None and c[4] <= previous:
            raise ValueError("P&F column dates must increase")
        if df is not None and c[4] not in df.index:
            raise ValueError("P&F column dates must belong to the OHLCV snapshot")
        previous = c[4]


def _congestion_bounds(segment, box, max_amp):
    tops = sorted((float(c[2]) for c in segment if c[0] == "X"), reverse=True)
    bottoms = sorted(float(c[1]) for c in segment if c[0] == "O")
    if len(tops) < 2 or len(bottoms) < 2:
        return None
    top = tops[1] if len(tops) >= 3 and tops[0] > tops[1] + 2 * box else tops[0]
    bot = bottoms[1] if len(bottoms) >= 3 and bottoms[0] < bottoms[1] - 2 * box else bottoms[0]
    if bot <= 0 or bot >= top or (top - bot) / bot > max_amp:
        return None
    if sum(abs(x - top) <= 2 * box for x in tops) < 2 or sum(abs(x - bot) <= 2 * box for x in bottoms) < 2:
        return None
    return bot, top


def detect_tr(columns: list, box: float, df=None, min_width: int = 5,
              max_amp: float = 0.35) -> TR | None:
    """Recent congestion from completed columns; exclude the active column.

    Boundaries need repeated tests, rather than a chain of overlapping trend
    legs. The 10..120-bar formation window is a bounded heuristic in
    bar units, not days. Prior trend only supplies a tentative directional bias.
    """
    if not math.isfinite(box) or box <= 0 or min_width < 2 or not 0 < max_amp < 1:
        raise ValueError("invalid trading-range parameters")
    _validate_columns(columns, df)
    if len(columns) - 1 < min_width:
        return None
    end = len(columns) - 2
    start = end
    bot, top = float(columns[end][1]), float(columns[end][2])
    for i in range(end - 1, -1, -1):
        c = columns[i]
        new_top, new_bot = max(top, c[2]), min(bot, c[1])
        if new_bot <= 0 or (new_top - new_bot) / new_bot > max_amp:
            break
        if c[2] < bot - box or c[1] > top + box:
            break
        if df is not None:
            t0 = df.index.get_loc(columns[i - 1][4]) + 1 if i else 0
            if df.index.get_loc(columns[end][4]) - t0 + 1 > 120:
                break
        start, top, bot = i, new_top, new_bot
    if end - start + 1 < min_width:
        return None
    start_pos = df.index.get_loc(columns[start - 1][4]) + 1 if df is not None and start else 0
    # Anchor at the FIRST observable congestion, not the most recent
    # reversal. Later tests must not erase their own original boundary or
    # introduce future volume into the baseline used to recognize them.
    formation_end = None
    for j in range(start + min_width - 1, end + 1):
        bounds = _congestion_bounds(columns[start:j + 1], box, max_amp)
        if bounds is None:
            continue
        bot, top = bounds
        if df is not None:
            potential_end = df.index.get_loc(columns[j][4])
            inside = df["Close"].iloc[start_pos:potential_end + 1].between(bot, top)
            if not inside.any():
                continue
            # The lead-in column may begin far outside the congestion.
            # Its earlier trend bars do not belong in the range's volume
            # baseline or the prior-trend window.
            formation_start_pos = start_pos + int(np.flatnonzero(inside.to_numpy())[0])
            if potential_end - formation_start_pos + 1 < 10:
                continue
        formation_end = j
        break
    if formation_end is None:
        return None
    # A later completed leg entirely detached from the original range
    # invalidates this structure; a pierce or breakout alone does not.
    if any(c[1] > top + 2 * box or c[2] < bot - 2 * box for c in columns[formation_end + 1:end + 1]):
        return None
    tr = TR(start, formation_end, top, bot, formation_end - start + 1, (top + bot) / 2)
    if df is not None:
        start_pos = formation_start_pos
        end_pos = df.index.get_loc(columns[formation_end][4])
        segment_df = df.iloc[start_pos:end_pos + 1]
        if not 10 <= len(segment_df) <= 120:
            return None
        tr.start_date, tr.end_date = segment_df.index[0], segment_df.index[-1]
        tr.confirmed_date = df.index[min(end_pos + 1, len(df) - 1)]
        tr.avg_volume = float(segment_df["Volume"].mean()) if "Volume" in df else 0.0
        before = df.iloc[max(0, start_pos - 15):start_pos]
        if len(before) == 15:
            movement = float(before["Close"].iloc[-1] / before["Close"].iloc[0] - 1)
            typical = float((before["High"] - before["Low"]).median() / before["Close"].median())
            if abs(movement) > typical:
                tr.bias = "accumulation" if movement < 0 else "distribution"
                tr.trend_bars = 15
    return tr


def _indicators(df):
    # Compare a bar against preceding observations; the tested bar does not
    # dilute its own volume/spread anomaly, and slicing retains warm-up data.
    volume = df["Volume"].rolling(20, min_periods=20).mean().shift(1)
    spread = (df["High"] - df["Low"]).rolling(20, min_periods=20).mean().shift(1)
    return volume, spread


def _event(name, date, price, vol, available_date=None, **context):
    return {"event": name, "date": date, "available_date": available_date if available_date is not None else date,
            "price": float(price), "vol": float(vol), **context}


def _available(e):
    return e.get("available_date", e["date"])


def _lps_lpsy(df: pd.DataFrame, tr: TR, events: list, window: int = 20) -> list:
    """Confirmed reaction pivots following SOS/SOW; exclude breakout bar.

    A reaction must test former resistance/support with reduced spread and
    volume. Confirmation on a later bar is distinct from the pivot date.
    """
    if window < 2:
        raise ValueError("LPS window must be at least two bars")
    vol_base, spread_base = _indicators(df)
    out = []
    breaks = sorted((e for e in events if e["event"] in ("SOS", "SOW")), key=_available)
    for b in breaks:
        start = df.index.get_loc(_available(b))
        bull = b["event"] == "SOS"
        name = "LPS" if bull else "LPSY"
        candidate = None
        next_break = next((_available(other) for other in breaks if _available(other) > _available(b)), None)
        for i in range(start + 1, min(start + window + 1, len(df))):
            if next_break is not None and df.index[i] >= next_break:
                break
            row, prior = df.iloc[i], df.iloc[i - 1]
            if row["Close"] < tr.bottom if bull else row["Close"] > tr.top:
                break
            if candidate is not None:
                j, price, vr = candidate
                undercut = row["Low"] < price if bull else row["High"] > price
                if undercut:
                    candidate = None
            if candidate is not None:
                j, price, vr = candidate
                recovered = (row["Low"] > price and row["Close"] > df["Close"].iloc[j]) if bull else (row["High"] < price and row["Close"] < df["Close"].iloc[j])
                if recovered:
                    out.append(_event(name, df.index[j], price, vr, df.index[i],
                                      side="accumulation" if bull else "distribution",
                                      range_confirmed_at=tr.confirmed_date))
                    break
            vb, sb = float(vol_base.iloc[i]), float(spread_base.iloc[i])
            if not np.isfinite([vb, sb]).all() or vb <= 0 or sb <= 0:
                continue
            price = float(row["Low"] if bull else row["High"])
            boundary = tr.top if bull else tr.bottom
            pullback = row["Close"] < prior["Close"] if bull else row["Close"] > prior["Close"]
            if (pullback and abs(price - boundary) <= 1.5 * sb and 0 < row["Volume"] <= 1.2 * vb
                    and row["High"] - row["Low"] <= 1.2 * sb):
                if candidate is None or (price < candidate[1] if bull else price > candidate[1]):
                    candidate = (i, price, float(row["Volume"] / vb))
    return out


