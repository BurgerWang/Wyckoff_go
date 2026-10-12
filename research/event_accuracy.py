"""Historical event_accuracy experiment; run explicitly with python -m research.event_accuracy.

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
from backtest import IncPnF, make_scale_for
from research.common import SYMBOLS, cached_fetch, experiment_args

syms = list(SYMBOLS)


def main():
    args = experiment_args(syms, years=10, record_option=None)
    fetch = cached_fetch(bt.fetch)
    evs = []
    for sym in args.symbols:
        df = fetch(sym, args.years)
        n = len(df)
        warmup = 120
        if len(df) <= warmup:
            continue
        p = IncPnF(make_scale_for(df, warmup, "atr", 0.5), 3)
        for tt in range(warmup + 1):
            rr = df.iloc[tt]
            p.update(
                float(rr["High"]), float(rr["Low"]), float(rr["Close"]), df.index[tt]
            )
        for t in range(warmup + 1, n):
            row = df.iloc[t]
            p.update(
                float(row["High"]), float(row["Low"]), float(row["Close"]), df.index[t]
            )
            res = wk.analyze(p.columns, p.scale, df.iloc[: t + 1], 3)
            tr = res["tr"]
            for e in res["events"]:
                if e.get("available_date", e["date"]) == df.index[t] and e["event"] in (
                    "SPRING",
                    "UT",
                    "SOS",
                    "SOW",
                ):
                    evs.append(
                        (
                            sym,
                            t,
                            e["event"],
                            float(e["price"]),
                            tr.top if tr else np.nan,
                            tr.bottom if tr else np.nan,
                        )
                    )
    print(f"事件总数: {len(evs)}")
    data = {sym: fetch(sym, args.years) for sym in args.symbols}
    print("数据预取完成")
    results = {"SPRING": [], "UT": [], "SOS": [], "SOW": []}
    for sym, t, event, price, top, bot in evs:
        df = data[sym]
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        if t + 20 >= n or np.isnan(top) or np.isnan(bot):
            continue
        win_low = lows[t + 1 : t + 21].min() if t + 21 <= n else lows[t + 1 :].min()
        win_high = highs[t + 1 : t + 21].max() if t + 21 <= n else highs[t + 1 :].max()
        win_close_min = (
            closes[t + 1 : t + 21].min() if t + 21 <= n else closes[t + 1 :].min()
        )
        win_close_max = (
            closes[t + 1 : t + 21].max() if t + 21 <= n else closes[t + 1 :].max()
        )
        if event == "SPRING":
            ok = win_low >= price  # spring low 守住
            results["SPRING"].append(ok)
        elif event == "UT":
            ok = win_high <= price  # UT high 守住
            results["UT"].append(ok)
        elif event == "SOS":
            ok = win_close_min > top  # 突破后 close 守在 TR top 上
            results["SOS"].append(ok)
        elif event == "SOW":
            ok = win_close_max < bot  # 破位后 close 守在 TR bottom 下
            results["SOW"].append(ok)
    print(f"\n===== 事件判定准确率 (事后 20 日定义性验证) =====")
    print(f"{'事件':<8}{'n':>6}{'定义':<28}{'成功率':>8}")
    for ev, rs in results.items():
        if rs:
            print(f"{ev:<8}{len(rs):>6}{'':<28}{np.mean(rs) * 100:>7.0f}%")
    print(f"\n===== 随机对照 (当日极值参考价, 分位数匹配) =====")
    rng = np.random.default_rng(11)
    n_ref = 3000
    cnt_s, cnt_u, n_sampled = 0, 0, 0
    for sym in args.symbols:
        df = data[sym]
        lows = df["Low"].values
        highs = df["High"].values
        n = len(df)
        if n <= 190:
            continue
        for _ in range(n_ref // len(args.symbols)):
            t = int(rng.integers(150, n - 40))
            if t + 21 <= n:
                n_sampled += 1
                if lows[t + 1 : t + 21].min() >= lows[t]:
                    cnt_s += 1
                if highs[t + 1 : t + 21].max() <= highs[t]:
                    cnt_u += 1
    print(
        f"当日 low 未来20日守住率: {cnt_s / n_sampled if n_sampled else np.nan:.0%} (对照 SPRING)"
    )
    print(
        f"当日 high 未来20日守住率: {cnt_u / n_sampled if n_sampled else np.nan:.0%} (对照 UT)"
    )


if __name__ == "__main__":
    main()
