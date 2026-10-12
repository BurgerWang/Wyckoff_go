"""Historical trade_system experiment; run explicitly with python -m research.trade_system.

Importing this module performs no downloads, file writes or research run.
Results are descriptive research, not an executable trading recommendation.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt
import numpy as np
from pnf import atr14
from research.common import SYMBOLS, cached_fetch, experiment_args, simulate_trade

syms = list(SYMBOLS)


def sim(data, struct, mode="2R"):
    sym, st, sos_t, lt, spring_low = struct
    df = data[sym]
    opens = df["Open"].values
    n = len(df)
    if lt + 1 >= n:
        return None
    entry_t = lt + 1
    entry = opens[entry_t]
    atr = float(atr14(df).iloc[lt])
    if not np.isfinite(atr) or atr <= 0:
        atr = entry * 0.02
    if mode == "2R":
        sl = entry - atr * 1.5
        tp = entry + atr * 1.5 * 2
    elif mode == "struct":  # historical variant retains its ATR-based target
        sl = min(spring_low, entry - atr * 1.5)
        tp = entry + atr * 1.5 * 2
    else:
        raise ValueError("mode must be 2R or struct")
    return simulate_trade(df, entry_t, 1, sl, tp)


def main():
    args = experiment_args(syms, years=10, record_option=None)
    fetch = cached_fetch(bt.fetch)
    rng = np.random.default_rng(7)
    data = {}
    for sym in args.symbols:
        df = fetch(sym, args.years)
        data[sym] = df
    structures = []
    for sym in args.symbols:
        df = data[sym]
        r = bt.run_wyckoff(df, "atr", 0.5, 3)
        evs = sorted(r["events"])
        # 最近 spring 表
        spring_idx = {}
        for t, ev, px in evs:
            if ev == "SPRING_C":
                spring_idx[t] = px
        used = set()
        for t, ev, px in evs:
            if ev != "SOS_C" or t in used:
                continue
            lps = [e for e in evs if e[1] == "LPS_C" and e[0] > t and e[0] not in used]
            if not lps:
                continue
            lt = lps[0][0]
            if lt - t > 30:
                continue
            # spring low: SOS 前最近的 SPRING_C 价格(即 low)
            prior_spring = [s for s in spring_idx if s < t]
            if not prior_spring:
                continue
            st = max(prior_spring)
            spring_low = spring_idx[st]
            used.add(t)
            used.add(lt)
            structures.append((sym, st, t, lt, spring_low))
    print(f"完整结构 (Spring→SOS→LPS): {len(structures)} 个")
    if not structures:
        print("样本不足，未产生有效结构")
        return
    results = {"2R": [], "struct": []}
    for mode in results:
        for s in structures:
            res = sim(data, s, mode)
            if res:
                results[mode].append(res)
    print(
        f"\n{'变体':<10}{'n':>4}{'胜率':>7}{'均值':>9}{'PF':>7}{'平均持仓':>8}{'样本乘积':>9}"
    )
    for mode, rs in results.items():
        if not rs:
            print(f"{mode}: 样本不足")
            continue
        rets = [r[0] for r in rs]
        wins = [r for r in rs if r[0] > 0]
        losses = [r for r in rs if r[0] <= 0]
        pf = (
            abs(sum(r[0] for r in wins) / sum(r[0] for r in losses))
            if losses and sum(r[0] for r in losses) != 0
            else float("inf")
        )
        total = np.prod([1 + r[0] for r in rs]) - 1
        avg_hold = np.mean([r[1] for r in rs])
        print(
            f"{mode:<10}{len(rs):>4}{len(wins) / len(rs) * 100:>6.0f}%{np.mean(rets) * 100:>8.2f}%{pf:>7.2f}{avg_hold:>7.0f}d{total * 100:>8.1f}%"
        )
        sl_cnt = sum(1 for r in rs if r[2] == "SL")
        tp_cnt = sum(1 for r in rs if r[2] == "TP")
        print(
            f"          SL:{sl_cnt} TP:{tp_cnt} END:{sum(1 for r in rs if r[2] == 'END')}"
        )
    print("\n===== 随机对照 (同交易日, 随机日入场, 2R 止损) =====")
    rand_rets = []
    for _ in range(500):
        s = structures[rng.integers(0, len(structures))]
        sym, st, sos_t, lt, spring_low = s
        df = data[sym]
        n = len(df)
        if n <= 330:
            continue
        rt = int(rng.integers(300, n - 30))
        entry = df["Open"].values[rt]
        atr = float(atr14(df).iloc[rt - 1])
        if not np.isfinite(atr) or atr <= 0:
            atr = entry * 0.02
        sl = entry - atr * 1.5
        tp = entry + atr * 1.5 * 2
        result = simulate_trade(df, rt, 1, sl, tp)
        if result is not None:
            rand_rets.append(result[0])
    if not rand_rets:
        print("随机对照样本不足")
        return
    rand_rets = np.array(rand_rets)
    print(
        f"随机入场 {len(rand_rets)} 次: 胜率 {np.mean(rand_rets > 0) * 100:.0f}%, 均值 {np.mean(rand_rets) * 100:.2f}%, 样本乘积 {(np.prod(1 + rand_rets) - 1) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
