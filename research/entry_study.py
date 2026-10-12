"""Historical entry_study experiment; run explicitly with python -m research.entry_study.

Importing this module performs no downloads, file writes or research run.
Results are descriptive research, not an executable trading recommendation.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt
import numpy as np
from research.common import SYMBOLS, cached_fetch, experiment_args

syms = list(SYMBOLS)


def random_wait(data, rng, sym, sos_t, lps_t, h, trials=300, return_samples=False):
    closes, _ = data[sym]
    rs = []
    for _ in range(trials):
        rt = int(rng.integers(sos_t + 1, lps_t + 1))
        if rt + h < len(closes):
            rs.append(closes[rt + h] / closes[rt] - 1)
    if return_samples:
        return np.asarray(rs)
    return float(np.mean(rs)) if rs else np.nan


def random_day(data, rng, sym, h, trials=300):
    closes, _ = data[sym]
    rs = []
    lo, hi = 200, len(closes) - h - 1
    if hi <= lo:
        return np.nan
    for _ in range(trials):
        rt = int(rng.integers(lo, hi))
        rs.append(closes[rt + h] / closes[rt] - 1)
    return float(np.mean(rs)) if rs else np.nan


def main():
    args = experiment_args(syms, years=10, record_option=None)
    fetch = cached_fetch(bt.fetch)
    rng = np.random.default_rng(42)
    data = {}
    for sym in args.symbols:
        df = fetch(sym, args.years)
        data[sym] = (df["Close"].values, df.index)
    structures = []
    for sym in args.symbols:
        closes, idx = data[sym]
        n = len(closes)
        r = bt.run_wyckoff(fetch(sym, args.years), "atr", 0.5, 3)
        evs = sorted(r["events"])
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
            used.add(t)
            used.add(lt)
            for h in (20, 60):
                if lt + h < n:
                    structures.append(
                        (
                            sym,
                            t,
                            lt,
                            h,
                            closes[lt + h] / closes[lt] - 1,
                            closes[lt],
                            closes[t],
                        )
                    )
    h20 = [s for s in structures if s[3] == 20]
    h60 = [s for s in structures if s[3] == 60]
    print(
        f"独立结构: {len(set(s[0] for s in structures))} 标的, 配对 {len({(s[0], s[1], s[2]) for s in structures})} (20d={len(h20)}, 60d={len(h60)})"
    )
    print(f"\n{'':12}{'LPS入场':>10}{'随机等待':>10}{'全局随机':>10}{'SOS追突破':>10}")
    for h, samples in ((20, h20), (60, h60)):
        if not samples:
            print(f"{h}d: 样本不足")
            continue
        lps = np.mean([s[4] for s in samples])
        wait_samples = [
            random_wait(data, rng, s[0], s[1], s[2], h, return_samples=True)
            for s in samples
        ]
        rw = np.mean([returns.mean() for returns in wait_samples])
        rd = np.mean([random_day(data, rng, s[0], h) for s in samples])
        sos = np.mean(
            [data[s[0]][0][s[1] + h] / data[s[0]][0][s[1]] - 1 for s in samples]
        )
        up_lps = np.mean([1 if s[4] > 0 else 0 for s in samples])
        up_sos = np.mean(
            [
                1 if data[s[0]][0][s[1] + h] / data[s[0]][0][s[1]] - 1 > 0 else 0
                for s in samples
            ]
        )
        up_rw = np.mean([(returns > 0).mean() for returns in wait_samples])
        print(
            f"{h}d         {lps * 100:>8.2f}% {rw * 100:>9.2f}% {rd * 100:>9.2f}% {sos * 100:>9.2f}%"
        )
        print(
            f"  上涨率:  LPS {up_lps * 100:.0f}% | 随机等待 {up_rw * 100:.0f}% | SOS {up_sos * 100:.0f}%"
        )
    lps20 = np.array([s[4] for s in h20])
    rw20 = np.array([random_wait(data, rng, s[0], s[1], s[2], 20) for s in h20])
    mask = ~np.isnan(rw20)
    if mask.sum() > 10:
        import scipy.stats as st

        t, p = st.ttest_rel(lps20[mask], rw20[mask])
        print(
            f"\nLPS vs 随机等待 (20d, n={mask.sum()}): 差 {np.mean(lps20[mask] - rw20[mask]) * 100:+.2f}pp, t={t:.2f}, p={p:.4f}"
        )


if __name__ == "__main__":
    main()
