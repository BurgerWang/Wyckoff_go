"""Historical vol_management experiment; run explicitly with python -m research.vol_management.

Importing this module performs no downloads, file writes or research run.
Results are descriptive research, not an executable trading recommendation.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt
import numpy as np
from research.common import (
    SYMBOLS,
    cached_fetch,
    experiment_args,
    load_records,
    attach_returns,
    metrics,
)

syms = list(SYMBOLS)

PHASE_W = {
    "accumulation": 1.3,  # 低波动吸筹 -> 加仓
    "distribution": 0.7,  # 高波动派发 -> 减仓
    "markup": 1.2,  # 稳定上涨 -> 加仓
    "markdown": 0.5,  # 高波动下跌 -> 减仓
    "trading range": 1.0,
    "unknown": 1.0,
}


def build_portfolio(recs, weight_fn):
    """weight_fn: DataFrame 行 -> 仓位。返回每标的日仓位序列。
    前视修复: t 日状态(收盘后) -> t+1 日收益; 权重按标的 shift 1 日。"""
    recs = recs.copy()
    recs["w"] = recs.apply(weight_fn, axis=1)
    recs = recs.sort_values(["sym", "date"])
    recs["w"] = recs.groupby("sym")["w"].shift(1).fillna(1.0)
    return recs


def wA(r):
    return 1.0


def wB1(r):
    return PHASE_W.get(r["phase"], 1.0)


def wB2(r):
    w = PHASE_W.get(r["phase"], 1.0)
    return w * 0.7 if r["ev"] == "SOS" else w


def wB3(r):
    v = r["vol20"]
    if not np.isfinite(v) or v <= 0:
        return 1.0
    return float(np.clip(0.15 / v, 0.3, 1.5))


def wB4(r):
    w = PHASE_W.get(r["phase"], 1.0)
    v = r["vol20"]
    if np.isfinite(v) and v > 0:
        w *= float(np.clip(0.15 / v, 0.3, 1.5))
    return w


def main():
    args = experiment_args(syms, years=10, record_option="records")
    fetch = cached_fetch(bt.fetch)
    recs = attach_returns(load_records(args.records), fetch, args.symbols, args.years)
    full_dates = recs.groupby("date")["sym"].nunique()
    full_dates = full_dates[full_dates == len(args.symbols)].index
    recs_full = recs[recs["date"].isin(full_dates)]
    if recs_full.empty:
        raise ValueError("No dates contain the complete requested symbol universe")
    print(
        f"合并后: {len(recs)} 行; {len(args.symbols)} 标的齐全日期: {len(full_dates)} 天 ({len(full_dates) / len(recs['date'].unique()) * 100:.0f}%)"
    )
    strategies = {
        "A_恒定": wA,
        "B1_阶段映射": wB1,
        "B2_阶段+突破降杠杆": wB2,
        "B3_波动率目标": wB3,
        "B4_阶段+波动率目标": wB4,
    }
    print(f"\n{'策略':<18}{'年化':>8}{'波动':>8}{'Sharpe':>8}{'maxDD':>8}{'Calmar':>8}")
    results = {}
    for name, fn in strategies.items():
        weighted = build_portfolio(recs, fn)
        # 每日组合收益 = 等权平均 (w * ret), 仅 {len(args.symbols)} 标的齐全日期 (固定分母)
        daily = recs_full[["date", "sym"]].merge(
            weighted[["sym", "date", "w", "ret"]],
            on=["sym", "date"],
            how="left",
            validate="one_to_one",
        )
        daily = (daily["w"] * daily["ret"]).groupby(daily["date"]).mean().sort_index()
        ann, vol, sharpe, dd = metrics(daily)
        calmar = ann / dd if dd > 0 else 0
        results[name] = (ann, vol, sharpe, dd, calmar)
        print(
            f"{name:<18}{ann * 100:>7.1f}%{vol * 100:>7.1f}%{sharpe:>8.2f}{dd * 100:>7.1f}%{calmar:>8.2f}"
        )
    print(f"\n===== 每标的配对 (B4 vs A, Sharpe 提升, shift 修复后) =====")
    paired = []
    for sym in args.symbols:
        g = recs[recs["sym"] == sym].sort_values("date")
        if len(g) < 500:
            continue
        daily_a = g["ret"]  # A 恒定
        # 重新算 B4 (含 shift)
        gw = g.copy()
        gw["w"] = gw.apply(wB4, axis=1)
        gw["w"] = gw.groupby("sym")["w"].shift(1).fillna(1.0)
        daily_b = gw["w"] * gw["ret"]
        sa = (
            daily_a.mean() / daily_a.std(ddof=1) * np.sqrt(252)
            if daily_a.std(ddof=1) > 0
            else 0
        )
        sb = (
            daily_b.mean() / daily_b.std(ddof=1) * np.sqrt(252)
            if daily_b.std(ddof=1) > 0
            else 0
        )
        paired.append((sb - sa, sym))
    paired.sort(reverse=True)
    pos = sum(1 for d, _ in paired if d > 0)
    print(f"Sharpe 提升为正的标的: {pos}/{len(paired)}")
    for d, sym in paired[:5]:
        print(f"  {sym:<8}: {d:+.3f}")
    for d, sym in paired[-3:]:
        print(f"  {sym:<8}: {d:+.3f}")


if __name__ == "__main__":
    main()
