"""Historical vol_robust experiment; run explicitly with python -m research.vol_robust.

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


def weights(recs, phase_w, tv, clip_lo, clip_hi):
    w = recs["phase"].map(phase_w).fillna(1.0).values
    v = recs["vol20"].values
    return lag_weights(recs, w * volatility_weights(v, tv, clip_lo, clip_hi))


def portfolio_metrics(recs, w, fee=0.0005):
    """Available-symbol daily mean and single-side exposure-change costs.

    The changing symbol calendar is a research convention, not a self-financing
    portfolio simulation. Initial exposure changes from zero and pays one fee.
    """
    recs = recs.reset_index(drop=True).assign(w=w).sort_values(["sym", "date"])
    daily = (recs["w"] * recs["ret"]).groupby(recs["date"]).mean().sort_index()
    changes = recs.groupby("sym")["w"].diff().abs().fillna(recs["w"].abs())
    turnover = changes.groupby(recs["date"]).mean().reindex(daily.index)
    cost = turnover.to_numpy() * fee
    net = daily.to_numpy() - cost
    ann, vol, sharpe, dd = metrics(net)
    return ann, vol, sharpe, dd, float(np.mean(cost) * 252)


def main():
    args = experiment_args(syms, years=10, record_option="records")
    fetch = cached_fetch(bt.fetch)
    recs = attach_returns(load_records(args.records), fetch, args.symbols, args.years)
    print(f"数据: {len(recs)} 行")
    print("\n===== 1. 参数敏感性 (B4 变体网格) =====")
    print(
        f"{'目标波动':>7}{'clip':>10}{'阶段缩放':>8}{'年化':>8}{'Sharpe':>8}{'maxDD':>8}{'年成本':>8}"
    )
    sharpe_vals = []
    base_sharpe = None
    for tv in (0.10, 0.15, 0.20):
        for clip in ((0.3, 1.5), (0.2, 2.0)):
            for sc in (0.5, 1.0, 1.5):
                phw = {k: (v - 1) * sc + 1 for k, v in PHASE_W0.items()}
                w = weights(recs, phw, tv, clip[0], clip[1])
                ann, vol, sharpe, dd, cost = portfolio_metrics(recs, w)
                sharpe_vals.append(sharpe)
                if tv == 0.15 and clip == (0.3, 1.5) and sc == 1.0:
                    base_sharpe = sharpe
                print(
                    f"{tv:>6.2f}{str(clip):>11}{sc:>8.1f}{ann * 100:>7.1f}%{sharpe:>8.2f}{dd * 100:>7.1f}%{cost * 100:>7.2f}%"
                )
    sharpe_vals = np.array(sharpe_vals)
    print(
        f"\n18 变体 Sharpe: min {sharpe_vals.min():.2f}, max {sharpe_vals.max():.2f}, "
        f"mean {sharpe_vals.mean():.2f}, 全部>1.0: {(sharpe_vals > 1.0).sum()}/{len(sharpe_vals)}, "
        f"全部>1.2: {(sharpe_vals > 1.2).sum()}/{len(sharpe_vals)}"
    )
    print("\n===== 2. 交易成本 (5bps 单边, 换手×成本) =====")
    fee = 0.0005
    for name, w in [
        ("A_恒定", np.ones(len(recs))),
        ("B3_波动率目标", weights(recs, {k: 1.0 for k in PHASE_W0}, 0.15, 0.3, 1.5)),
        ("B4_阶段+波动率", weights(recs, PHASE_W0, 0.15, 0.3, 1.5)),
    ]:
        ann, vol, sharpe, dd, cost = portfolio_metrics(recs, w, fee)
        print(
            f"{name:<12} 年化 {ann * 100:>6.1f}% | Sharpe {sharpe:>5.2f} | maxDD {dd * 100:>5.1f}% | 年均成本 {cost * 100:.2f}%"
        )
    print("\n===== 3. 每标的配对显著性 (B4 vs B3 / B4 vs A) =====")
    w_b3 = weights(recs, {k: 1.0 for k in PHASE_W0}, 0.15, 0.3, 1.5)
    w_b4 = weights(recs, PHASE_W0, 0.15, 0.3, 1.5)
    d34, d4a = [], []
    for sym in args.symbols:
        g = recs[recs["sym"] == sym].sort_values("date")
        if len(g) < 500:
            continue
        rets = g["ret"].values
        r3 = w_b3[g.index] * rets
        r4 = w_b4[g.index] * rets
        s3, s4, sA = metrics(r3)[2], metrics(r4)[2], metrics(rets)[2]
        d34.append(s4 - s3)
        d4a.append(s4 - sA)
    d34, d4a = np.array(d34), np.array(d4a)
    if len(d34) < 2:
        print("配对样本不足")
        return

    from scipy import stats

    t34, p34 = stats.ttest_rel(d34, np.zeros_like(d34))
    t4a, p4a = stats.ttest_rel(d4a, np.zeros_like(d4a))
    print(
        f"B4 vs B3 (阶段增量): ΔSharpe 均值 {d34.mean():+.3f}, t={t34:.2f}, p={p34:.4f}, 正方向 {(d34 > 0).sum()}/{len(d34)}"
    )
    print(
        f"B4 vs A:              ΔSharpe 均值 {d4a.mean():+.3f}, t={t4a:.2f}, p={p4a:.4f}, 正方向 {(d4a > 0).sum()}/{len(d4a)}"
    )


if __name__ == "__main__":
    main()
