#!/usr/bin/env python3
"""Point & Figure construction, descriptive signals and chart reports.

Fixed/percentage/traditional scales share one incremental construction engine.
ATR configurations are historical snapshots: recalibrating their box size can
change earlier columns. Price objectives and reference trend lines are estimates.
"""
from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- data
def fetch(symbol: str, years: float = 2) -> pd.DataFrame:
    from market_data import fetch as fetch_market_data
    return fetch_market_data(symbol, years)


def atr14(df: pd.DataFrame) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean()


# ---------------------------------------------------------------- scales
_MAX_BOXES = 1_000_000
# StockCharts Support Center, "P&F Charts", Chart Scaling table.
_TRADITIONAL = ((0.25, 0.0625), (1.0, 0.125), (5.0, 0.25),
                (20.0, 0.5), (100.0, 1.0), (200.0, 2.0),
                (500.0, 4.0), (1000.0, 5.0), (25000.0, 50.0),
                (math.inf, 500.0))


class BoxScale(np.ndarray):
    """An ndarray retaining the rule needed to extend a price grid."""

    def __new__(cls, values, kind="custom", param=None):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.kind, obj.param = kind, param
        return obj

    def __array_finalize__(self, source):
        self.kind = getattr(source, "kind", "custom")
        self.param = getattr(source, "param", None)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _size_guard(count):
    if count < 2 or count > _MAX_BOXES:
        raise ValueError(f"scale must contain 2..{_MAX_BOXES} boxes; increase box size")


def _traditional_step(price):
    return next(step for upper, step in _TRADITIONAL if price < upper)


def _traditional_grid(lo, hi):
    chunks, lower = [], -math.inf
    for upper, step in _TRADITIONAL:
        left, right = max(lo, lower), min(hi, upper)
        if left <= right:
            start = math.floor(left / step) * step
            if math.isfinite(lower):
                start = max(start, lower)
            count = int(math.floor((right - start) / step)) + 1
            if count > 0:
                if count + sum(len(c) for c in chunks) > _MAX_BOXES:
                    raise ValueError("traditional scale exceeds box limit")
                chunks.append(start + np.arange(count) * step)
        if hi < upper:
            break
        lower = upper
    return np.unique(np.concatenate(chunks))


def make_scale(min_v: float, max_v: float, kind: str = "fixed",
               box: float = None, pct: float = 0.01) -> np.ndarray:
    """Generate a validated grid with stable anchors and price margins.

    Percentage levels are powers of (1+pct), anchored at 1, so adding a new
    low does not shift all earlier box boundaries. Traditional follows the
    complete StockCharts table, including penny-stock and high-price tiers.
    """
    min_v, max_v = _positive(min_v, "min_v"), _positive(max_v, "max_v")
    if min_v > max_v:
        raise ValueError("min_v must not exceed max_v")
    lo, hi = min_v * 0.95, max_v * 1.05
    if lo <= 0 or not math.isfinite(hi):
        raise ValueError("price range is outside representable scale bounds")
    if kind == "fixed":
        b = _positive(1.0 if box is None else box, "box")
        if not math.isfinite(hi / b) or (hi - lo) / b > _MAX_BOXES - 4:
            raise ValueError("fixed scale exceeds box limit; increase box size")
        first, last = math.floor(lo / b) - 1, math.ceil(hi / b) + 1
        _size_guard(last - first + 1)
        values = np.arange(first, last + 1, dtype=float) * b
        return _validated_scale(BoxScale(values, "fixed", b))
    if kind == "pct":
        pct = _positive(pct, "pct")
        if 1.0 + pct <= 1.0:
            raise ValueError("pct is too small for a distinct floating-point grid")
        log_step = math.log1p(pct)
        first, last = math.floor(math.log(lo) / log_step) - 1, \
                      math.ceil(math.log(hi) / log_step) + 1
        _size_guard(last - first + 1)
        values = np.exp(np.arange(first, last + 1, dtype=float) * log_step)
        return _validated_scale(BoxScale(values, "pct", pct))
    if kind == "traditional":
        return _validated_scale(BoxScale(_traditional_grid(
                                         lo - 4 * _traditional_step(lo),
                                         hi + 4 * _traditional_step(hi)),
                                         "traditional"))
    raise ValueError(f"unknown scale kind: {kind}")


def _validated_scale(scale):
    values = np.asarray(scale, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() \
            or not np.all(np.diff(values) > 0):
        raise ValueError("scale must contain at least 2 strictly increasing finite values")
    _size_guard(len(values))
    if isinstance(scale, BoxScale):
        return scale.copy()
    diffs = np.diff(values)
    if np.allclose(diffs, diffs[0], rtol=1e-10, atol=abs(diffs[0]) * 1e-10):
        return BoxScale(values.copy(), "fixed", float(diffs[0]))
    if (values > 0).all():
        ratios = values[1:] / values[:-1]
        if np.allclose(ratios, ratios[0], rtol=1e-10, atol=0):
            return BoxScale(values.copy(), "pct", float(ratios[0] - 1))
    return BoxScale(values.copy())


def _idx_floor(scale: np.ndarray, price: float) -> int:
    """index of the largest box-price <= price (floor semantics)"""
    i = int(np.searchsorted(scale, price, side="right")) - 1
    return max(0, min(i, len(scale) - 1))


def _idx_ceil(scale: np.ndarray, price: float) -> int:
    """index of the smallest box-price >= price (ceil semantics)"""
    i = int(np.searchsorted(scale, price, side="left"))
    return max(0, min(i, len(scale) - 1))


# ---------------------------------------------------------------- P&F core
@dataclass
class PnF:
    symbol: str
    box: float                       # minimum step; dynamic grids vary by price
    reversal: int = 3
    scale: np.ndarray = None         # ascending box-price grid
    columns: list = field(default_factory=list)   # (dir, lo, hi, n_bars, last_date)
    signals: list = field(default_factory=list)
    column_start_dates: list = field(default_factory=list)

    @property
    def direction(self):
        return self.columns[-1][0] if self.columns else None

    @property
    def top(self):
        return self.columns[-1][2] if self.columns else None

    @property
    def bottom(self):
        return self.columns[-1][1] if self.columns else None

    @property
    def last_date(self):
        return self.columns[-1][4] if self.columns else None

    @property
    def scale_name(self):
        if self.scale is None:
            return f"fixed {self.box:.2f}"
        kind = getattr(self.scale, "kind", "custom")
        if kind == "fixed":
            return f"fixed {self.scale.param:g}"
        if kind == "pct":
            return f"percentage {self.scale.param:.2%}"
        return f"{kind} ({len(self.scale)} boxes)"


def _extend_scale(S: np.ndarray, lo_px: float, hi_px: float,
                  reversal: int) -> np.ndarray:
    """Extend known scale rules without changing existing box levels.

    An arbitrary nonuniform grid has no extrapolation rule and must be supplied
    with adequate price/reversal margins; silently inventing one changes P&F.
    """
    S = _validated_scale(S)
    if not math.isfinite(lo_px) or not math.isfinite(hi_px) or lo_px > hi_px:
        raise ValueError("invalid scale extension bounds")
    while True:
        below = lo_px < S[0] or _idx_floor(S, lo_px) < reversal
        above = hi_px > S[-1] or _idx_ceil(S, hi_px) + reversal >= len(S)
        if not below and not above:
            return S
        if S.kind == "custom":
            raise ValueError("custom scale needs wider price and reversal margins")
        pieces = []
        for lower, needed in ((True, below), (False, above)):
            if not needed:
                continue
            edge, bound = (S[0], lo_px) if lower else (S[-1], hi_px)
            sign = -1 if lower else 1
            if S.kind == "fixed":
                count = max(0, math.ceil(abs(bound - edge) / S.param)) + reversal + 2
                _size_guard(len(S) + count)
                new = edge + sign * np.arange(1, count + 1) * S.param
            elif S.kind == "pct":
                if bound <= 0:
                    raise ValueError("percentage scale requires positive prices")
                log_step = math.log1p(S.param)
                count = max(0, math.ceil(abs(math.log(bound / edge)) / log_step)) + reversal + 2
                _size_guard(len(S) + count)
                new = edge * np.exp(sign * np.arange(1, count + 1) * log_step)
            else:
                if lower:
                    lo = min(bound, edge) - (reversal + 2) * _traditional_step(min(bound, edge))
                    new = _traditional_grid(lo, edge)
                    new = new[new < edge]
                else:
                    hi = max(bound, edge) + (reversal + 2) * _traditional_step(max(bound, edge))
                    new = _traditional_grid(edge, hi)
                    new = new[new > edge]
            pieces.append(new[::-1] if lower and S.kind != "traditional" else new)
        lower_piece = pieces.pop(0) if below else np.array([])
        upper_piece = pieces.pop(0) if above else np.array([])
        S = _validated_scale(BoxScale(np.concatenate([lower_piece, S, upper_piece]),
                                       S.kind, S.param))


class IncrementalPnF(PnF):
    """One-bar-at-a-time construction, also used by batch builds/backtests.

    The first close is a neutral anchor, not a direction signal. A subsequent
    price crossing the next box establishes direction. If the first decisive
    high/low bar crosses both sides, its close relative to the anchor selects
    direction (X on a tie); intrabar order cannot be recovered from OHLC data.
    Column extrema are filled box labels, inclusive, including one-box columns.
    """

    def __init__(self, scale, reversal=3, use_close=False, symbol=""):
        if isinstance(reversal, bool) or not isinstance(reversal, (int, np.integer)) \
                or reversal < 1:
            raise ValueError("reversal must be a positive integer")
        if not isinstance(use_close, bool):
            raise ValueError("use_close must be a boolean")
        S = _validated_scale(scale)
        super().__init__(symbol, float(np.min(np.diff(S))), int(reversal), S)
        self.use_close = use_close
        self._anchor = None
        self._anchor_date = None
        self._anchor_bars = 0
        self._last_observation_date = None

    def update(self, h, l, c, date):
        h, l, c = (_positive(v, name) for v, name in ((h, "High"), (l, "Low"), (c, "Close")))
        if l > h or not l <= c <= h:
            raise ValueError("OHLC prices are inconsistent")
        date = pd.Timestamp(date)
        if pd.isna(date) or (self._last_observation_date is not None
                            and date <= self._last_observation_date):
            raise ValueError("bar dates must be valid and strictly increasing")
        high, low = (c, c) if self.use_close else (h, l)
        lo_bound = min(low, self.bottom if self.columns else c,
                       self._anchor if self._anchor is not None else c)
        hi_bound = max(high, self.top if self.columns else c,
                       self._anchor if self._anchor is not None else c)
        S = self.scale
        if lo_bound < S[0] or hi_bound > S[-1] \
                or _idx_floor(S, lo_bound) < self.reversal \
                or _idx_ceil(S, hi_bound) + self.reversal >= len(S):
            S = _extend_scale(S, lo_bound, hi_bound, self.reversal)
            self.scale = S
            self.box = float(np.min(np.diff(S)))
        self._last_observation_date = date
        if self._anchor is None:
            self._anchor, self._anchor_date, self._anchor_bars = c, date, 1
            return
        if not self.columns:
            self._anchor_bars += 1
            up = _idx_floor(S, self._anchor) + 1
            down = _idx_ceil(S, self._anchor) - 1
            rises, falls = high >= S[up], low <= S[down]
            if rises and (not falls or c >= self._anchor):
                self.columns.append(["X", S[up], S[_idx_floor(S, high)],
                                     self._anchor_bars, date])
            elif falls:
                self.columns.append(["O", S[_idx_ceil(S, low)], S[down],
                                     self._anchor_bars, date])
            else:
                return
            self.column_start_dates.append(self._anchor_date)
            return
        d, lo, hi, n, _ = self.columns[-1]
        ilo, ihi = _idx_floor(S, lo), _idx_floor(S, hi)
        if d == "X":
            if high >= S[ihi + 1]:
                hi = S[_idx_floor(S, high)]
            elif low <= S[ihi - self.reversal]:
                self.columns.append(["O", S[_idx_ceil(S, low)], S[ihi - 1], 1, date])
                self.column_start_dates.append(date)
                self.signals.append(("REVERSAL_X->O", date, S[ihi - 1]))
                return
        else:
            # A fractional move below the current O label does not add an O.
            if low <= S[ilo - 1]:
                lo = S[_idx_ceil(S, low)]
            elif high >= S[ilo + self.reversal]:
                self.columns.append(["X", S[ilo + 1], S[_idx_floor(S, high)], 1, date])
                self.column_start_dates.append(date)
                self.signals.append(("REVERSAL_O->X", date, S[ilo + 1]))
                return
        self.columns[-1] = [d, lo, hi, n + 1, date]


def build(df: pd.DataFrame, scale: np.ndarray | float, reversal: int = 3,
          use_close: bool = True) -> PnF:
    """Batch construction through the same engine as incremental updates."""
    from market_data import validate_ohlcv
    df = validate_ohlcv(df, require_volume=False)
    if isinstance(scale, Real):
        scale = make_scale(float(df["Low"].min()), float(df["High"].max()),
                           "fixed", box=_positive(scale, "box"))
    p = IncrementalPnF(scale, reversal, use_close, df.attrs.get("symbol", ""))
    for idx, row in df.iterrows():
        p.update(float(row["High"]), float(row["Low"]), float(row["Close"]), idx)
    return p


def column_tops(p: PnF) -> list:
    """(col_idx, top_price) for every X column"""
    return [(i, c[2]) for i, c in enumerate(p.columns) if c[0] == "X"]


def column_bottoms(p: PnF) -> list:
    return [(i, c[1]) for i, c in enumerate(p.columns) if c[0] == "O"]


def double_top_breakout(p: PnF) -> dict | None:
    """Current X-column exceeds the previous X-column top."""
    tops = column_tops(p)
    if len(tops) < 2:
        return None
    prev_top = tops[-2][1]
    if p.direction == "X" and p.top > prev_top:
        return {"type": "double-top breakout", "level": prev_top,
                "current_top": p.top}
    return None


def double_bottom_breakout(p: PnF) -> dict | None:
    bottoms = column_bottoms(p)
    if len(bottoms) < 2:
        return None
    prev_bot = bottoms[-2][1]
    if p.direction == "O" and p.bottom < prev_bot:
        return {"type": "double-bottom breakdown", "level": prev_bot,
                "current_bottom": p.bottom}
    return None


def horizontal_count_target(p: PnF, bearish_frac: float = 1.0) -> dict | None:
    """Heuristic congestion count using the horizontal-count formula.
    Extension = congestion width (columns) x box size x reversal amount.
    Bullish: target = pattern low (lowest O-column bottom) + Extension.
    Bearish: target = pattern high (highest X-column top) - Extension,
             Cohen's 2/3 extension available via bearish_frac=2/3.
    Congestion must be >= 5 columns wide; negative targets ignored.
    A common price band must persist through the congestion. Chain overlap
    with a growing union incorrectly counts a whole trend as one base.
    The final column must exit the entire base, not only its previous column.
    Nonuniform scales use a local-box approximation, explicitly marked.
    """
    bearish_frac = _positive(bearish_frac, "bearish_frac")
    if len(p.columns) < 5:
        return None
    cols = p.columns
    S = p.scale
    # Width includes the lead-out column, but its expansion cannot widen base.
    base_lo, base_hi = cols[-2][1], cols[-2][2]
    base_start = len(cols) - 2
    for i in range(len(cols) - 3, -1, -1):
        c = cols[i]
        if c[2] >= base_lo and c[1] <= base_hi:
            base_start = i
            base_lo = max(base_lo, c[1])
            base_hi = min(base_hi, c[2])
        else:
            break
    width = len(cols) - base_start            # includes breakout column
    if width < 5:
        return None
    seg = cols[base_start:-1]
    pat_lo = min(c[1] for c in seg if c[0] == "O")
    pat_hi = max(c[2] for c in seg if c[0] == "X")
    # effective box size at the middle of the congestion (dynamic grids)
    mid = (pat_lo + pat_hi) / 2
    k = min(len(S) - 2, _idx_floor(S, mid)) if S is not None else 0
    box_eff = float(S[k + 1] - S[k]) if S is not None else p.box
    approximate = S is not None and getattr(S, "kind", "custom") != "fixed"
    if p.direction == "X" and p.top > pat_hi:
        ext = width * box_eff * p.reversal
        tgt = pat_lo + ext
        if tgt <= 0:
            return None
        return {"side": "bullish", "base_cols": width, "pattern_low": pat_lo,
                "box_eff": box_eff, "extension": ext, "target": tgt,
                "heuristic": True, "approximate": approximate}
    if p.direction == "O" and p.bottom < pat_lo:
        ext = width * box_eff * p.reversal * bearish_frac
        tgt = pat_hi - ext
        if tgt <= 0:
            return None
        return {"side": "bearish", "base_cols": width, "pattern_high": pat_hi,
                "box_eff": box_eff, "extension": ext, "target": tgt,
                "heuristic": True, "approximate": approximate}
    return None


def vertical_count_target(p: PnF, bearish_frac: float = 2 / 3) -> dict | None:
    """StockCharts vertical-count price objective (docs verified).
    Count Column = most recent column that triggers a Double Top Breakout
    (X column above prior X-column top) or Double Bottom Breakdown (O column
    below prior O-column bottom). Column height is fixed only after a 3-box
    reversal; until then the target is preliminary.
    Extension = filled box count x box size x reversal (bullish),
    or x 2/3 reversal (bearish,
    Dorsey, StockCharts default). Bullish target = column low + Extension;
    bearish target = column high - Extension.
    """
    bearish_frac = _positive(bearish_frac, "bearish_frac")
    if len(p.columns) < 3:
        return None
    cols = p.columns
    last_i = len(cols) - 1
    # find the most recent Double Top/Bottom breakout column
    for i in range(last_i, -1, -1):
        d = cols[i][0]
        prev = None
        for j in range(i - 1, -1, -1):       # prior same-direction column
            if cols[j][0] == d:
                prev = cols[j]
                break
        if prev is None:
            continue
        if d == "X" and cols[i][2] > prev[2]:
            side, col, level = "X", cols[i], prev[2]
            break
        if d == "O" and cols[i][1] < prev[1]:
            side, col, level = "O", cols[i], prev[1]
            break
    else:
        return None
    fixed = i < last_i                       # breakout column already reversed
    S = p.scale
    if S is None or getattr(S, "kind", "custom") == "fixed" \
            or np.allclose(np.diff(S), p.box, rtol=1e-10, atol=p.box * 1e-10):
        box_count = int(round((col[2] - col[1]) / p.box)) + 1
        height = box_count * p.box
        approximate = False
    else:
        box_count = _idx_floor(S, col[2]) - _idx_floor(S, col[1]) + 1
        # StockCharts permits high-low as an estimate when box size changes.
        height = col[2] - col[1]
        approximate = True
    if side == "X":
        ext = height * p.reversal
        tgt = col[1] + ext
        return {"side": "bullish", "col": i, "break_level": level,
                "height": height, "extension": ext, "target": tgt,
                "fixed": fixed, "box_count": box_count,
                "approximate": approximate}
    ext = height * p.reversal * bearish_frac
    tgt = col[2] - ext
    if tgt <= 0:
        return None
    return {"side": "bearish", "col": i, "break_level": level,
            "height": height, "extension": ext, "target": tgt,
            "fixed": fixed, "bearish_frac": bearish_frac,
            "box_count": box_count, "approximate": approximate}


def _grid_level(S, index):
    """Unclamped grid projection; return None for unknown extrapolation."""
    if 0 <= index < len(S):
        return float(S[index])
    edge_index = 0 if index < 0 else len(S) - 1
    steps = index - edge_index
    kind = getattr(S, "kind", "custom")
    if kind == "fixed":
        return float(S[edge_index] + steps * S.param)
    if kind == "pct":
        return float(S[edge_index] * (1 + S.param) ** steps)
    if kind == "traditional":
        value = float(S[edge_index])
        sign = -1 if steps < 0 else 1
        for _ in range(abs(steps)):
            value += sign * _traditional_step(value if sign > 0 else np.nextafter(value, -math.inf))
        return value
    return None


def trendline_status(p: PnF) -> dict | None:
    """Historical extreme reference lines, not StockCharts active trend lines.

    Start one box outside the extreme; move one box per column. No switching
    or touch-adjustment model is implemented, so both references are marked
    heuristic and must not be treated as a current automatic trend verdict.
    """
    if len(p.columns) < 4:
        return None
    S = p.scale
    last_i = len(p.columns) - 1

    res = {}
    o_cols = [(i, c) for i, c in enumerate(p.columns) if c[0] == "O"]
    if o_cols:
        ai, ac = min(o_cols, key=lambda x: x[1][1])
        if S is not None:
            ia = _idx_floor(S, ac[1])
            line = _grid_level(S, ia - 1 + (last_i - ai))
        else:
            line = ac[1] + (last_i - ai - 1) * p.box
        if line is not None:
            res["support"] = {"anchor_col": ai, "anchor": ac[1], "line": line,
                              "price_above": p.bottom > line, "heuristic": True}
    x_cols = [(i, c) for i, c in enumerate(p.columns) if c[0] == "X"]
    if x_cols:
        ai, ac = max(x_cols, key=lambda x: x[1][2])
        if S is not None:
            ia = _idx_floor(S, ac[2])
            line = _grid_level(S, ia + 1 - (last_i - ai))
        else:
            line = ac[2] - (last_i - ai - 1) * p.box
        if line is not None:
            res["resistance"] = {"anchor_col": ai, "anchor": ac[2], "line": line,
                                 "price_below": p.top < line, "heuristic": True}
    return res if res else None


# ---------------------------------------------------------------- plotting
def plot(p: PnF, df: pd.DataFrame, path: str, show_trendline: bool = True):
    """Render equal-height boxes without changing the application's backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    S = p.scale
    fig = Figure(figsize=(15, 8))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#aaa")
    ax.grid(True, color="#222", linewidth=0.6)
    def row_index(price):
        return _idx_floor(S, price) if S is not None else price / p.box

    for i, (d, lo, hi, n, dt) in enumerate(p.columns):
        color = "#3b82f6" if d == "X" else "#ef4444"
        rows = np.arange(row_index(lo), row_index(hi) + 1)
        ax.plot(np.full(len(rows), i), rows, linestyle="none", color=color,
                marker="x" if d == "X" else "o", markersize=7,
                markerfacecolor="none", markeredgewidth=1.3)
    if show_trendline and p.columns:
        status = trendline_status(p) or {}
        for key, sign, color in (("support", 1, "#f59e0b"),
                                 ("resistance", -1, "#a855f7")):
            if key not in status:
                continue
            ref = status[key]
            start, end = ref["anchor_col"], len(p.columns) - 1
            y0 = row_index(ref["anchor"]) - sign
            ax.plot([start, end], [y0, y0 + sign * (end - start)],
                    color=color, lw=1.4, ls="--", alpha=0.9,
                    label=f"{key} reference (heuristic)")
        if status:
            ax.legend(facecolor="#0e1117", labelcolor="#aaa")
    # last column highlight
    li = len(p.columns) - 1
    ax.axvline(li - 0.45, color="#888", lw=0.8, alpha=0.5)
    last = f"{p.direction} @ {p.top if p.direction == 'X' else p.bottom:.2f}" \
           if p.columns else "undetermined (no box move)"
    ax.set_title(f"{p.symbol}  Point&Figure  {p.scale_name}  "
                 f"reversal={p.reversal}  last={last}",
                 color="#ddd", fontsize=13)
    ax.set_xlim(-1, len(p.columns) + 1)
    if p.columns:
        ylo = row_index(min(c[1] for c in p.columns))
        yhi = row_index(max(c[2] for c in p.columns))
        ax.set_ylim(ylo - 2, yhi + 2)
        ticks = np.unique(np.linspace(ylo, yhi, min(15, int(yhi - ylo) + 1)).astype(int))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{S[k] if S is not None else k * p.box:g}" for k in ticks])
    else:
        ax.text(0.5, 0.5, "No directional box move yet", color="#aaa",
                ha="center", transform=ax.transAxes)
    ax.set_ylabel("Price", color="#aaa")
    ax.set_xlabel(f"Columns (n={len(p.columns)}, data through {df.index[-1].date()})",
                  color="#aaa")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


# ---------------------------------------------------------------- report
def report(p: PnF, df: pd.DataFrame, extra: str = "") -> str:
    lines = [
        f"== {p.symbol} P&F report ==",
        f"params: {p.scale_name}  reversal={p.reversal}",
        f"data: {len(df)} bars, last close {df['Close'].iloc[-1]:.2f} "
        f"on {df.index[-1].date()}",
    ]
    if p.columns:
        lines.append(f"columns: {len(p.columns)}  last col: {p.direction} "
                     f"({'top' if p.direction == 'X' else 'bottom'} = "
                     f"{p.top if p.direction == 'X' else p.bottom:.2f})")
    else:
        lines.append("columns: 0  direction undetermined (no box move)")
    if double_top_breakout(p):
        lines.append(f"signal: DOUBLE-TOP BREAKOUT above {double_top_breakout(p)['level']:.2f}")
    if double_bottom_breakout(p):
        lines.append(f"signal: DOUBLE-BOTTOM BREAKDOWN below {double_bottom_breakout(p)['level']:.2f}")
    tgt = horizontal_count_target(p)
    if tgt:
        lines.append(f"price objective (heuristic): {'↑' if tgt['side']=='bullish' else '↓'} "
                     f"target {tgt['target']:.2f} (base width {tgt['base_cols']} cols)")
    vc = vertical_count_target(p)
    if vc:
        state = "未固定(preliminary)" if not vc["fixed"] else "已固定"
        if vc["approximate"]:
            state += ", variable-box approximation"
        conflict = ((vc["side"] == "bullish" and p.direction == "O") or
                    (vc["side"] == "bearish" and p.direction == "X"))
        warn = " ⚠与当前信号方向矛盾" if conflict else ""
        lines.append(f"vertical count: {'↑' if vc['side']=='bullish' else '↓'} "
                     f"target {vc['target']:.2f} (col {vc['col']}, 高 "
                     f"{vc['height']:.2f}, ext {vc['extension']:.2f}, {state}){warn}")
    ts = trendline_status(p)
    if ts:
        if "support" in ts:
            s = ts["support"]
            lines.append(f"support reference (heuristic): column "
                         f"{'above' if s['price_above'] else 'below'} line @ "
                         f"{s['line']:.2f} (anchor col {s['anchor_col']} @ {s['anchor']:.2f})")
        if "resistance" in ts:
            r = ts["resistance"]
            lines.append(f"resistance reference (heuristic): column "
                         f"{'below' if r['price_below'] else 'above'} line @ "
                         f"{r['line']:.2f} (anchor col {r['anchor_col']} @ {r['anchor']:.2f})")
    # recent reversals
    if p.signals:
        last = p.signals[-3:]
        lines.append("recent reversals: " + ", ".join(
            f"{s[0]} @ {s[2]:.2f} ({s[1].date()})" for s in last))
    # key levels
    tops = [c[2] for c in p.columns if c[0] == "X"][-5:]
    bots = [c[1] for c in p.columns if c[0] == "O"][-5:]
    lines.append(f"recent X-tops: {[round(t,2) for t in tops]}")
    lines.append(f"recent O-bottoms: {[round(b,2) for b in bots]}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


# ---------------------------------------------------------------- main
CONFIGS = [
    ("atr",   "ATR中位数x0.5 (历史快照)"),
    ("pct",   "百分比 1% (StockCharts默认)"),
    ("trad",  "传统 ChartCraft (Dorsey表)"),
]


def build_config(symbol: str, df: pd.DataFrame, kind: str,
                 box_mult: float = 0.5, pct: float = 0.01,
                 reversal: int = 3) -> tuple[PnF, str]:
    """Build P&F for one scale config. Returns (PnF, scale label)."""
    from market_data import validate_ohlcv
    df = validate_ohlcv(df, require_volume=False)
    df.attrs["symbol"] = symbol
    min_v = float(df[["Low", "Close"]].min().min())
    max_v = float(df[["High", "Close"]].max().max())
    a = atr14(df)
    if kind == "atr":
        box_mult = _positive(box_mult, "box_mult")
        box = float(a.median()) * box_mult
        if not math.isfinite(box) or box <= 0:
            raise ValueError("ATR box is zero or invalid; use a fixed box or percentage scale")
        sc = make_scale(min_v, max_v, "fixed", box=box)
        label = f"ATR中位数x{box_mult} = {box:g} (历史快照; ATR14尾={a.iloc[-1]:g})"
    elif kind == "pct":
        sc = make_scale(min_v, max_v, "pct", pct=pct)
        label = f"百分比 {pct*100:g}% (动态格)"
    elif kind == "trad":
        sc = make_scale(min_v, max_v, "traditional")
        label = "传统 ChartCraft/Dorsey 表"
    else:
        raise ValueError(kind)
    p = build(df, sc, reversal, use_close=False)
    return p, label


def run(symbol: str, kinds: list[str] | None = None, years: float = 2,
        outdir: str = ".") -> tuple[list[tuple[str, PnF, str, str]], pd.DataFrame]:
    """Explicit fetch/render; return ([(kind, chart, text, path)], frame)."""
    allowed = {k for k, _ in CONFIGS}
    kinds = [k for k, _ in CONFIGS] if kinds is None else kinds
    if not kinds or not set(kinds).issubset(allowed):
        raise ValueError(f"kinds must be a non-empty subset of {sorted(allowed)}")
    df = fetch(symbol, years)
    df.attrs["symbol"] = symbol
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    filename_symbol = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    results = []
    for kind, name in CONFIGS:
        if kind not in kinds:
            continue
        p, label = build_config(symbol, df, kind)
        path = str(destination / f"{filename_symbol}_pnf_{kind}.png")
        plot(p, df, path)
        txt = report(p, df, extra=f"scale: {label}")
        results.append((kind, p, txt, path))
    return results, df


def consensus(results: list, px: float) -> dict:
    """Describe majority column direction across the supplied configurations.

    Agreement is not a validated trading recommendation. At least two
    unanimous configurations receive the legacy STRONG code; a single
    configuration or non-unanimous majority receives WEAK. Neutral charts
    abstain but count in the denominator. A close beyond a high/low reversal
    level is marked critical because high/low extension can preempt reversal.
    """
    px = _positive(px, "px")
    votes: dict[str, int] = {}
    details = []
    for kind, p, txt, path in results:
        d = p.direction
        if d is None:
            details.append({"kind": kind, "dir": None, "last": None,
                            "line": None, "dist": None, "box_w": p.box,
                            "critical": False})
            continue
        votes[d] = votes.get(d, 0) + 1
        sc = p.scale
        if d == "X":
            idx = _idx_floor(sc, p.top) if sc is not None else 0
            line = _grid_level(sc, idx - p.reversal) if sc is not None \
                   else p.top - p.reversal * p.box
            if line is None:
                raise ValueError("scale does not cover the reversal level")
            dist = px - line                      # 还需跌多少触发空头反转
            box_w = sc[idx] - sc[max(0, idx - 1)] if sc is not None else p.box
            last_px = p.top
        else:
            idx = _idx_floor(sc, p.bottom) if sc is not None else 0
            line = _grid_level(sc, idx + p.reversal) if sc is not None \
                   else p.bottom + p.reversal * p.box
            if line is None:
                raise ValueError("scale does not cover the reversal level")
            dist = line - px                      # 还需涨多少触发多头反转
            box_w = sc[min(len(sc) - 1, idx + 1)] - sc[idx] if sc is not None else p.box
            last_px = p.bottom
        critical = dist <= 0.5 * box_w
        details.append({"kind": kind, "dir": d, "last": last_px,
                        "line": line, "dist": dist, "box_w": box_w,
                        "critical": critical})
    x, o = votes.get("X", 0), votes.get("O", 0)
    total = len(results)
    if x == total and total >= 2:
        verdict = ("多头方向（一致）", "STRONG_LONG")
    elif o == total and total >= 2:
        verdict = ("空头方向（一致）", "STRONG_SHORT")
    elif x > total / 2:
        verdict = ("偏多方向（部分配置）", "WEAK_LONG")
    elif o > total / 2:
        verdict = ("偏空方向（部分配置）", "WEAK_SHORT")
    else:
        verdict = ("方向未形成多数共识", "FLAT")
    # 临界降级: 胜方任一档距离反转线 < 0.5 格
    winners = [dd for dd in details if dd["dir"] == ("X" if x > o else "O")]
    critical_any = any(dd["critical"] for dd in winners)
    if critical_any and verdict[1] not in ("FLAT",):
        verdict = (verdict[0] + " ※临界", verdict[1] + "_CRITICAL")
    return {"votes": {"X": x, "O": o}, "verdict": verdict[0],
            "code": verdict[1], "details": details, "critical_any": critical_any}


def summary(results: list, df: pd.DataFrame) -> str:
    px = float(df["Close"].iloc[-1])
    c = consensus(results, px)
    lines = [
        f"== {df.attrs.get('symbol', '')} P&F 多参数对照 ==",
        f"data: {len(df)} bars, last close {px:.2f} ({df.index[-1].date()})",
        f"{'scale':<12}{'box':>7}{'列数':>6}  最后信号",
    ]
    for kind, p, txt, path in results:
        dd = next(x for x in c["details"] if x["kind"] == kind)
        if not p.columns:
            lines.append(f"{kind:<12}{dd['box_w']:>7.2f}{0:>6}  方向未确定")
            continue
        last = p.columns[-1]
        lines.append(f"{kind:<12}{dd['box_w']:>7.2f}{len(p.columns):>6}  "
                     f"{last[0]} @ {last[1]:.2f}-{last[2]:.2f}")
    lines.append(f"=> 投票: X={c['votes']['X']} O={c['votes']['O']}  "
                 f"裁决: {c['verdict']}")
    lines.append("   距反转线:")
    for dd in c["details"]:
        if dd["dir"] is None:
            lines.append(f"   {dd['kind']:<6} 方向未确定")
            continue
        flag = " ※临界" if dd["critical"] else ""
        lines.append(f"   {dd['kind']:<6} {dd['dir']} @ {dd['last']:.2f}  "
                     f"线={dd['line']:.2f} 距={dd['dist']:+.2f}{flag}")
    return "\n".join(lines)


def main(argv=None):
    """Compatibility CLI; calculations/imports alone have no external I/O."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="GOOG")
    for kind, label in CONFIGS:
        parser.add_argument(f"--{kind}", action="store_true", help=label.replace("%", "%%"))
    parser.add_argument("--years", type=float, default=2)
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args(argv)
    selected = [k for k, _ in CONFIGS if getattr(args, k)]
    try:
        results, df = run(args.symbol, selected or None, args.years, args.outdir)
        print(summary(results, df))
        for kind, p, txt, path in results:
            print(f"\n{txt}\nchart: {path}")
            if kind == "atr":
                from wyckoff import analyze, report as w_report
                res = analyze(p.columns, p.scale, df, reversal=p.reversal)
                print(f"\n{w_report(res)}")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
