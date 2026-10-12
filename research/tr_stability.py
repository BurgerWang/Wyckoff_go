"""Historical tr_stability experiment; run explicitly with python -m research.tr_stability.

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
from research.common import cached_fetch, experiment_args

syms = ["GOOG", "XOM", "TLT"]


def main():
    args = experiment_args(syms, years=5, record_option=None)
    fetch = cached_fetch(bt.fetch)
    for sym in args.symbols:
        df = fetch(sym, args.years)
        warmup = 120
        if len(df) <= warmup:
            print(f"{sym}: 样本不足")
            continue
        p = IncPnF(make_scale_for(df, warmup, "atr", 0.5), 3)
        for tt in range(warmup + 1):
            rr = df.iloc[tt]
            p.update(
                float(rr["High"]), float(rr["Low"]), float(rr["Close"]), df.index[tt]
            )
        # 最终全样本 TR (事后视图)
        p_full = IncPnF(make_scale_for(df, len(df) - 1, "atr", 0.5), 3)
        for tt in range(len(df)):
            rr = df.iloc[tt]
            p_full.update(
                float(rr["High"]), float(rr["Low"]), float(rr["Close"]), df.index[tt]
            )
        res_full = wk.analyze(p_full.columns, p_full.scale, df, 3)
        tr_full = res_full["tr"]
        # 滚动 TR: 每 30 天采样
        rows = []
        for t in range(warmup + 1, len(df)):
            p.update(
                float(df["High"].iloc[t]),
                float(df["Low"].iloc[t]),
                float(df["Close"].iloc[t]),
                df.index[t],
            )
            if (t - warmup - 1) % 30:
                continue
            res = wk.analyze(p.columns, p.scale, df.iloc[: t + 1], 3)
            tr = res["tr"]
            if tr:
                rows.append((df.index[t].date(), tr.top, tr.bottom, tr.width))
        if tr_full is None:
            print(f"\n{sym}: 事后全样本无可识别 TR; 滚动样本={len(rows)}")
            continue
        print(
            f"\n{sym}: 事后全样本 TR top={tr_full.top:.1f} bot={tr_full.bottom:.1f} width={tr_full.width} (bias={tr_full.bias})"
        )
        if rows:
            tops = [r[1] for r in rows]
            bots = [r[2] for r in rows]
            print(
                f"  滚动识别 (n={len(rows)}): top 范围 {min(tops):.1f}-{max(tops):.1f}, bot 范围 {min(bots):.1f}-{max(bots):.1f}"
            )
            # 最后一次滚动 vs 全样本
            lt, lb = rows[-1][1], rows[-1][2]
            drift = max(
                abs(lt - tr_full.top) / tr_full.top,
                abs(lb - tr_full.bottom) / tr_full.bottom,
            )
            print(
                f"  最终滚动 top={lt:.1f} bot={lb:.1f} | vs 全样本偏差 {drift * 100:.1f}%"
            )
            # 稳定性: 相邻滚动变化
            chg = [
                abs(rows[i + 1][1] - rows[i][1]) / rows[i][1]
                + abs(rows[i + 1][2] - rows[i][2]) / rows[i][2]
                for i in range(len(rows) - 1)
            ]
            if not chg:
                continue
            print(f"  相邻 30 日边界平均变化: {np.mean(chg) * 100:.1f}% (越小越稳定)")


if __name__ == "__main__":
    main()
