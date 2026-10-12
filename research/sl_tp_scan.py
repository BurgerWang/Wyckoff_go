"""Historical sl_tp_scan experiment; run explicitly with python -m research.sl_tp_scan.

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


def simulate(data, struct, sl_atr, rr):
    sym, direction, ref_px, lt, atr = struct
    df = data[sym]
    opens = df["Open"].values
    n = len(df)
    if lt + 1 >= n:
        return None
    entry = opens[lt + 1]
    if not np.isfinite(atr) or atr <= 0:
        atr = entry * 0.02
    sl_dist = atr * sl_atr
    sign = 1 if direction == "LONG" else -1
    stop, target = entry - sign * sl_dist, entry + sign * sl_dist * rr
    result = simulate_trade(df, lt + 1, sign, stop, target)
    return result[0] if result is not None else None


def main():
    args = experiment_args(syms, years=10, record_option=None)
    fetch = cached_fetch(bt.fetch)
    data = {}
    structures = []
    for sym in args.symbols:
        df = fetch(sym, args.years)
        data[sym] = df
        r = bt.run_wyckoff(df, "atr", 0.5, 3)
        evs = sorted(r["events"])
        used = set()
        # 多头
        spring_idx = {t: px for t, ev, px in evs if ev == "SPRING_C"}
        for t, ev, px in evs:
            if ev != "SOS_C" or t in used:
                continue
            lps = [e for e in evs if e[1] == "LPS_C" and e[0] > t and e[0] not in used]
            if not lps:
                continue
            lt = lps[0][0]
            if lt - t > 30:
                continue
            prior = [s for s in spring_idx if s < t]
            if not prior:
                continue
            used.add(t)
            used.add(lt)
            atr = float(atr14(df).iloc[lt])
            structures.append((sym, "LONG", spring_idx[max(prior)], lt, atr))
        # 空头 (用 _C 标签: FSM 宽松推进导致孤立 UT/SOW/LPSY 几乎不存在)
        ut_idx = {t: px for t, ev, px in evs if ev in ("UT", "UT_C")}
        for t, ev, px in evs:
            if ev not in ("SOW", "SOW_C") or t in used:
                continue
            lpsy = [
                e
                for e in evs
                if e[1] in ("LPSY", "LPSY_C") and e[0] > t and e[0] not in used
            ]
            if not lpsy:
                continue
            lt = lpsy[0][0]
            if lt - t > 30:
                continue
            prior = [s for s in ut_idx if s < t]
            if not prior:
                continue
            used.add(t)
            used.add(lt)
            atr = float(atr14(df).iloc[lt])
            structures.append((sym, "SHORT", ut_idx[max(prior)], lt, atr))
    print(
        f"完整结构: {len(structures)} (多头 {sum(1 for s in structures if s[1] == 'LONG')}, 空头 {sum(1 for s in structures if s[1] == 'SHORT')})"
    )
    print(f"\n===== 止损/目标参数扫描 =====")
    print(
        f"{'SL':>5} {'RR':>4} {'n':>4} {'胜率':>7} {'均值':>8} {'PF':>6} {'样本乘积':>9}"
    )
    grid_results = {}
    best = None
    for sl_atr in (1.0, 1.5, 2.0):
        for rr in (1.5, 2.0, 3.0, 4.0):
            rets = []
            for s in structures:
                r_ = simulate(data, s, sl_atr, rr)
                if r_ is not None:
                    rets.append(r_)
            rets = np.array(rets)
            if len(rets) == 0:
                continue
            wins = rets[rets > 0]
            losses = rets[rets <= 0]
            pf = (
                abs(wins.sum() / losses.sum())
                if len(losses) and losses.sum() != 0
                else float("inf")
            )
            total = np.prod(1 + rets) - 1
            print(
                f"{sl_atr:>5.1f} {rr:>4.1f} {len(rets):>4} {np.mean(rets > 0) * 100:>6.0f}% {np.mean(rets) * 100:>7.2f}% {pf:>6.2f} {total * 100:>8.1f}%"
            )
            grid_results[(sl_atr, rr)] = (np.mean(rets > 0), np.mean(rets), pf, total)
            if best is None or total > best[1]:
                best = ((sl_atr, rr), total)
    if best is None:
        print("样本不足，未产生有效结构")
        return
    print(
        f"\n最优: SL={best[0][0]}ATR, RR={best[0][1]} → 样本乘积 {best[1] * 100:.1f}%"
    )
    sl_atr, rr = best[0]
    rng = np.random.default_rng(5)
    rand_rets = []
    for _ in range(1000):
        sym = args.symbols[rng.integers(0, len(args.symbols))]
        df = data[sym]
        n = len(df)
        if n <= 330:
            continue
        rt = int(rng.integers(300, n - 30))
        entry = df["Open"].values[rt]
        atr = float(atr14(df).iloc[rt - 1])
        if not np.isfinite(atr) or atr <= 0:
            atr = entry * 0.02
        sl, tp = entry - atr * sl_atr, entry + atr * sl_atr * rr
        result = simulate_trade(df, rt, 1, sl, tp)
        if result is not None:
            rand_rets.append(result[0])
    if not rand_rets:
        print("随机对照样本不足")
        return
    rand_rets = np.array(rand_rets)
    print(
        f"\n随机对照 (SL={sl_atr}ATR, RR={rr}): n={len(rand_rets)}, 胜率 {np.mean(rand_rets > 0) * 100:.0f}%, 均值 {np.mean(rand_rets) * 100:+.2f}%, 样本乘积 {(np.prod(1 + rand_rets) - 1) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
