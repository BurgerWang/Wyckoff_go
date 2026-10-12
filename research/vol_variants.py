"""Historical vol_variants experiment; run explicitly with python -m research.vol_variants.

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
    lag_weights,
    volatility_weights,
    metrics,
)

syms = list(SYMBOLS)

PHASE_W0 = {
    "accumulation": 1.3,
    "distribution": 0.7,
    "markup": 1.2,
    "markdown": 0.5,
    "trading range": 1.0,
    "unknown": 1.0,
}


def weights(recs, phase_w, tv=0.15, clip_lo=0.3, clip_hi=1.5):
    w = recs["phase"].map(phase_w).fillna(1.0).values
    v = recs["vol20"].values
    return lag_weights(recs, w * volatility_weights(v, tv, clip_lo, clip_hi))


def portfolio_series(recs, w, rebalance="D"):
    """按调仓频率构建组合日收益序列 (周/月调仓: 仓位只在调仓日更新)"""
    if rebalance not in ("D", "W", "M"):
        raise ValueError("rebalance must be D, W or M")
    recs = recs.assign(w=w).sort_values(["sym", "date"])
    if rebalance == "W":
        recs["wk"] = recs["date"].dt.to_period("W")
        recs["w"] = recs.groupby(["sym", "wk"])["w"].transform("first")
    elif rebalance == "M":
        recs["mk"] = recs["date"].dt.to_period("M")
        recs["w"] = recs.groupby(["sym", "mk"])["w"].transform("first")
    daily = (recs["w"] * recs["ret"]).groupby(recs["date"]).mean().sort_index()
    return daily.values


def main():
    args = experiment_args(syms, years=10, record_option="records")
    fetch = cached_fetch(bt.fetch)
    recs = attach_returns(load_records(args.records), fetch, args.symbols, args.years)
    print(
        f"数据: {len(recs)} 行, 日期范围: {recs['date'].min().date()} ~ {recs['date'].max().date()}"
    )
    print("\n===== A. 子样本稳健性 (前5年 vs 后5年) =====")
    w_A = np.ones(len(recs))
    w_B4 = weights(recs, PHASE_W0)
    mid = recs["date"].median()
    for label, mask in [("前5年", recs["date"] < mid), ("后5年", recs["date"] >= mid)]:
        sub = recs[mask]
        if sub.empty:
            print(f"{label}: 样本不足")
            continue
        dA = portfolio_series(sub, w_A[mask.values])
        dB = portfolio_series(sub, w_B4[mask.values])
        aA = metrics(dA)
        aB = metrics(dB)
        print(
            f"{label}: A Sharpe {aA[2]:.2f} maxDD {aA[3] * 100:.1f}% | B4 Sharpe {aB[2]:.2f} maxDD {aB[3] * 100:.1f}% | ΔSharpe {aB[2] - aA[2]:+.2f}"
        )
    print("\n===== B. 调仓频率 (日/周/月) =====")
    for freq, label in [("D", "日调仓"), ("W", "周调仓"), ("M", "月调仓")]:
        dA = portfolio_series(recs, w_A, freq)
        dB = portfolio_series(recs, w_B4, freq)
        aA = metrics(dA)
        aB = metrics(dB)
        print(
            f"{label}: A Sharpe {aA[2]:.2f} | B4 Sharpe {aB[2]:.2f} maxDD {aB[3] * 100:.1f}% | ΔSharpe {aB[2] - aA[2]:+.2f}"
        )
    print("\n===== D. 市场分层 (美股 vs 加密) =====")
    crypto = {"BTC-USD", "ETH-USD"}
    for label, grp in [
        ("美股", [s for s in args.symbols if s not in crypto]),
        ("加密", [s for s in args.symbols if s in crypto]),
    ]:
        mask = recs["sym"].isin(grp).values
        sub = recs[mask]
        if sub.empty:
            print(f"{label}: 样本不足")
            continue
        dA = portfolio_series(sub, w_A[mask])
        dB = portfolio_series(sub, w_B4[mask])
        aA = metrics(dA)
        aB = metrics(dB)
        print(
            f"{label}: A Sharpe {aA[2]:.2f} | B4 Sharpe {aB[2]:.2f} | ΔSharpe {aB[2] - aA[2]:+.2f}"
        )


if __name__ == "__main__":
    main()
