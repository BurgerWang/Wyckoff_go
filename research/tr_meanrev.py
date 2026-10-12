"""Historical tr_meanrev experiment; run explicitly with python -m research.tr_meanrev.

Importing this module performs no downloads, file writes or research run.
Results are descriptive research, not an executable trading recommendation.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt
import wyckoff as wk
import numpy as np
import pandas as pd
from backtest import IncPnF, make_scale_for
from research.common import SYMBOLS, cached_fetch, experiment_args

syms = list(SYMBOLS)


def main():
    args = experiment_args(syms, years=10, record_option=None)
    fetch = cached_fetch(bt.fetch)
    data = {}
    rows = []
    for sym in args.symbols:
        df = fetch(sym, args.years)
        data[sym] = df
        if len(df) <= 120:
            continue
        closes = df["Close"].values
        vol_base = pd.Series(df["Volume"]).rolling(20).mean().values
        warmup = 120
        p = IncPnF(make_scale_for(df, warmup, "atr", 0.5), 3)
        for tt in range(warmup + 1):
            rr = df.iloc[tt]
            p.update(
                float(rr["High"]), float(rr["Low"]), float(rr["Close"]), df.index[tt]
            )
        for t in range(warmup + 1, len(df)):
            row = df.iloc[t]
            p.update(
                float(row["High"]), float(row["Low"]), float(row["Close"]), df.index[t]
            )
            res = wk.analyze(p.columns, p.scale, df.iloc[: t + 1], 3)
            tr = res["tr"]
            vr = float(row["Volume"] / vol_base[t]) if vol_base[t] > 0 else 1.0
            rows.append(
                (
                    sym,
                    t,
                    tr.top if tr else np.nan,
                    tr.bottom if tr else np.nan,
                    tr.mid if tr else np.nan,
                    closes[t],
                    vr,
                )
            )
            if len(rows) % 100000 == 0:
                print(f"  ...{len(rows)} 行", flush=True)
    rec = pd.DataFrame(rows, columns=["sym", "t", "top", "bot", "mid", "close", "vr"])
    rec = rec.dropna()
    print(f"TR 内记录: {len(rec)}")
    rng = np.random.default_rng(3)
    results = {"long": [], "short": [], "rand_long": [], "rand_short": []}
    for sym in rec["sym"].unique():
        g = rec[rec["sym"] == sym].reset_index(drop=True)
        closes = data[sym]["Close"].values
        n = len(closes)
        for i in range(len(g)):
            r = g.iloc[i]
            t = int(r["t"])
            if t + 21 >= n:
                continue
            entry = closes[t + 1]  # 次日收盘起算（不声称开盘成交）
            fwd20 = closes[t + 21]
            pos = (
                (r["close"] - r["bot"]) / (r["top"] - r["bot"])
                if r["top"] > r["bot"]
                else 0.5
            )
            ret = fwd20 / entry - 1
            if r["close"] <= r["bot"] and r["vr"] < 1.2:
                results["long"].append(ret)
            elif r["close"] >= r["top"] and r["vr"] < 1.2:
                results["short"].append(-ret)
            # 随机对照: 中带日 (同标的, 同量能条件)
            if 0.35 < pos < 0.65 and r["vr"] < 1.2 and rng.random() < 0.02:
                results["rand_long"].append(ret)
                results["rand_short"].append(-ret)
    print(f"\n{'策略':<12}{'n':>6}{'胜率':>8}{'均值20d':>10}{'样本乘积':>10}")
    for k, v in results.items():
        if len(v) > 10:
            arr = np.array(v)
            total = np.prod(1 + arr) - 1
            print(
                f"{k:<12}{len(arr):>6}{np.mean(arr > 0) * 100:>7.0f}%{np.mean(arr) * 100:>9.2f}%{total * 100:>9.1f}%"
            )
        else:
            print(f"{k:<12}{len(v):>6}  (样本不足)")
    longs = np.array(results["long"])
    shorts = np.array(results["short"])
    rand_l = np.array(results["rand_long"])
    rand_s = np.array(results["rand_short"])
    if len(longs) > 10 and len(rand_l) > 10:
        all_sig = np.concatenate([longs, shorts])
        all_rand = np.concatenate([rand_l, rand_s])
        print(
            f"\n合并信号: n={len(all_sig)}, 均值 {np.mean(all_sig) * 100:+.2f}%, 胜率 {np.mean(all_sig > 0) * 100:.0f}%"
        )
        print(
            f"随机对照: n={len(all_rand)}, 均值 {np.mean(all_rand) * 100:+.2f}%, 胜率 {np.mean(all_rand > 0) * 100:.0f}%"
        )
        print(f"差值: {np.mean(all_sig) * 100 - np.mean(all_rand) * 100:+.2f}pp")


if __name__ == "__main__":
    main()
