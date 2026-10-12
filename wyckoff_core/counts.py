"""Explicit fixed-box horizontal-span count scenarios.

StockCharts/Wyckoff Analytics counts the full horizontal span, including
blanks. Occupied boxes at the selected row are retained as a diagnostic,
not silently substituted for the span. Nonuniform grids are unsupported.
"""
import math

import numpy as np

from .stages import range_events, sequence_state
from .structure import available


def count_scenarios(columns, scale, tr, events, reversal=3, df=None, *, use_close=False):
    if not isinstance(use_close, bool):
        raise ValueError("use_close must be bool")
    if not isinstance(reversal, (int, np.integer)) or isinstance(reversal, bool) or reversal < 1:
        raise ValueError("reversal must be a positive integer")
    output = {"status": "unavailable", "reason": "no_observable_range", "targets": None,
              "method": "horizontal_span", "price_method": "close" if use_close else "high_low",
              "source_ids": ["WA_TUTORIAL", "CMT_PRUDEN"]}
    if tr is None:
        return output
    if scale is None or len(scale) < 2:
        output["reason"] = "missing_scale"
        return output
    grid = np.asarray(scale, dtype=float)
    steps = np.diff(grid)
    if grid.ndim != 1 or not np.isfinite(grid).all() or (steps <= 0).any():
        raise ValueError("scale must be finite and strictly increasing")
    if not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-10):
        output["reason"] = "nonuniform_grid_requires_separate_count_method"
        return output
    if tr.lifecycle == "forming":
        output["reason"] = "range_not_established"
        return output
    box = float(steps[0])
    recent = range_events(events, tr)
    state = sequence_state(recent)
    targets = {"box": box, "reversal": int(reversal), "status": "scenario",
               "method": "horizontal_span", "price_method": output["price_method"], "bullish": None, "bearish": None,
               "source_ids": output["source_ids"], "range_id": tr.range_id}
    for acc, side, key, pivot_key in ((True, "bull", "bullish", "acc_lps"),
                                      (False, "bear", "bearish", "dist_lpsy")):
        pivot = state[pivot_key]
        targets.update({f"width_{side}": 0, f"extension_{side}": 0.0,
                        f"count_line_{side}": None, f"fallback_{side}": False})
        if pivot is None or pivot.get("status") != "confirmed":
            continue
        last_price = float(df["Close"].iloc[-1]) if df is not None else None
        if last_price is not None and (last_price < tr.bottom if acc else last_price > tr.top):
            continue
        # Rebuild the known P&F snapshot at count confirmation. A final chart's
        # completed/active columns may contain prices from after this event.
        cs, starts = columns, None
        if df is not None:
            import pnf
            known = pnf.build(df.loc[:available(pivot)], grid, reversal, use_close=use_close)
            cs, starts = known.columns, known.column_start_dates
        if not cs:
            continue
        raw = float(pivot["price"])
        row = np.searchsorted(grid, raw, side="right") - 1 if acc else np.searchsorted(grid, raw, side="left")
        if not 0 <= row < len(grid):
            continue
        line = float(grid[row])
        left = tr.start_col
        if tr.start_date is not None:
            left = next((j for j, c in enumerate(cs) if c[4] >= tr.start_date), len(cs))
        if starts is not None:
            right = next((j for j in range(len(cs) - 1, -1, -1) if starts[j] <= pivot["date"]), -1)
        else:
            right = min(tr.end_col, max((j for j, c in enumerate(cs) if c[4] <= pivot["date"]), default=-1))
        if left < 0 or right < left or right >= len(cs):
            continue
        segment = cs[left:right + 1]
        # Anchor at the first genuine intersection with the selected row in
        # the observed formation, not a price midpoint fallback.
        first = next((j for j, c in enumerate(segment) if c[1] <= line <= c[2]), None)
        if first is None or not segment[-1][1] <= line <= segment[-1][2]:
            continue
        left += first
        segment = cs[left:right + 1]
        width = len(segment)
        extension = width * box * reversal
        targets.update({f"width_{side}": width, f"extension_{side}": extension,
                        f"count_line_{side}": line, f"count_line_raw_{side}": raw,
                        f"occupied_columns_{side}": sum(c[1] <= line <= c[2] for c in segment),
                        f"start_col_{side}": left, f"end_col_{side}": right,
                        f"available_date_{side}": available(pivot),
                        f"anchor_date_{side}": pivot["date"]})
        if width < 2:
            continue
        lower, upper = (min(tr.bottom, line) + extension, max(tr.bottom, line) + extension) if acc else (min(tr.top, line) - extension, max(tr.top, line) - extension)
        if lower <= 0 or not all(math.isfinite(value) for value in (lower, upper)):
            continue
        targets[key] = {"min": lower, "mid": (lower + upper) / 2, "max": upper}
    if targets["bullish"] or targets["bearish"]:
        output.update(status="scenario", reason=None, targets=targets)
    else:
        output["reason"] = "no_valid_confirmed_count_anchor_or_span"
    return output


def wyckoff_count(columns, scale, tr, events, reversal=3, df=None, *, use_close=False):
    """Compatibility result; richer unavailable reasons use count_scenarios."""
    return count_scenarios(columns, scale, tr, events, reversal, df, use_close=use_close)["targets"]
