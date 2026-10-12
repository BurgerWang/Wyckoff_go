"""Historical framework_val experiment; run explicitly with python -m research.framework_val.

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
import json
from backtest import IncPnF, make_scale_for
from research.common import SYMBOLS, cached_fetch, experiment_args

syms = list(SYMBOLS)


def main():
    args = experiment_args(syms, years=10, record_option="output")
    output = args.output
    fetch = cached_fetch(bt.fetch)
    recs = []
    for sym in args.symbols:
        df = fetch(sym, args.years)
        if len(df) <= 120:
            continue
        closes = df["Close"].values
        rets = np.diff(np.log(closes))
        volatility = {
            w: np.concatenate(
                [[np.nan], pd.Series(rets).rolling(w).std().values * np.sqrt(252)]
            )
            for w in (10, 20, 60)
        }
        vol20 = volatility[20]
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
            in_tr = 1 if res["tr"] is not None else 0
            ph = res["phase"]["regime"] if res["phase"] else "?"
            ev_tag = ""
            available_events = []
            for e in res["events"]:
                if e.get("available_date", e["date"]) == df.index[t]:
                    ev_tag = e["event"]
                    available_events.append(
                        {
                            "event": e["event"],
                            "date": str(e["date"]),
                            "available_date": str(e.get("available_date", e["date"])),
                            "price": float(e["price"]),
                            "side": e.get("side"),
                            "status": e.get("status", "candidate"),
                            "range_id": e.get("range_id"),
                        }
                    )
            vr = float(row["Volume"] / vol_base[t]) if vol_base[t] > 0 else 1.0
            recs.append(
                (
                    sym,
                    df.index[t],
                    in_tr,
                    ph,
                    vol20[t],
                    vr,
                    ev_tag,
                    closes[t] / closes[t - 1] - 1,
                    volatility[10][t],
                    volatility[60][t],
                    json.dumps(available_events),
                )
            )
    rec_df = pd.DataFrame(
        recs,
        columns=[
            "sym",
            "date",
            "in_tr",
            "phase",
            "vol20",
            "vol_ratio",
            "ev",
            "ret",
            "vol10",
            "vol60",
            "events_json",
        ],
    )
    if rec_df.empty:
        raise ValueError("No symbols supplied enough bars to produce framework records")
    output.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(output, index=False)
    print(f"记录: {len(rec_df)} 日, {rec_df['sym'].nunique()} 标的")
    print(f"TR 内占比: {rec_df['in_tr'].mean() * 100:.1f}%")
    print(f"阶段分布: {rec_df['phase'].value_counts().to_dict()}")
    print("\n===== 维度 2: 结构特征区分度 =====")
    print("\n[2a] TR 内 vs TR 外 (每标的配对对比):")
    d_vol, d_vr = [], []
    for sym in rec_df["sym"].unique():
        g = rec_df[rec_df["sym"] == sym]
        tr_vol = g.loc[g["in_tr"] == 1, "vol20"].mean()
        out_vol = g.loc[g["in_tr"] == 0, "vol20"].mean()
        tr_vr = g.loc[g["in_tr"] == 1, "vol_ratio"].mean()
        out_vr = g.loc[g["in_tr"] == 0, "vol_ratio"].mean()
        if np.isfinite(tr_vol) and np.isfinite(out_vol):
            d_vol.append(tr_vol - out_vol)
            d_vr.append(tr_vr - out_vr)
    d_vol, d_vr = np.array(d_vol), np.array(d_vr)
    print(
        f"  波动率差 (TR内-外): 均值 {(d_vol.mean() if len(d_vol) else np.nan) * 100:+.2f}pp/年化, 一致方向 {(d_vol < 0).sum()}/{len(d_vol)} (理论: TR内更低)"
    )
    print(
        f"  量能比差 (TR内-外): 均值 {(d_vr.mean() if len(d_vr) else np.nan):+.3f}, 一致方向 {(d_vr < 0).sum()}/{len(d_vr)} (理论: TR内更低)"
    )
    print("\n[2b] 事件日量能特征 (vol/20日均量):")
    ev_map = {}
    for ev in ["SPRING", "UT", "SOS", "SOW", "SC", "BC", "LPS"]:
        s = rec_df.loc[rec_df["ev"] == ev, "vol_ratio"]
        all_s = rec_df["vol_ratio"]
        if len(s) > 10:
            ev_map[ev] = (len(s), s.mean(), all_s.mean())
            print(
                f"  {ev:<8}: n={len(s):>4} 均值量比 {s.mean():.2f} vs 全体 {all_s.mean():.2f}"
            )
    print("\n[2c] 阶段量能 + 波动率:")
    for ph in ["accumulation", "distribution", "markup", "markdown"]:
        s = rec_df[rec_df["phase"].str.contains(ph, na=False)]
        if len(s) > 50:
            print(
                f"  {ph:<14}: n={len(s):>6} 量比 {s['vol_ratio'].mean():.2f} | 波动率 {s['vol20'].mean() * 100:.1f}%"
            )
    print("\n===== 维度 3: 波动率状态信息 =====")
    print("\n[3a] TR 前半 vs 后半波动率 (压缩假说):")
    half_diffs = []
    for sym in rec_df["sym"].unique():
        g = rec_df[rec_df["sym"] == sym]
        # 用连续 TR 段
        in_tr = g["in_tr"].values
        segs = []
        start = None
        for i, v in enumerate(in_tr):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                segs.append((start, i))
                start = None
        if start is not None:
            segs.append((start, len(in_tr)))
        for a, b in segs:
            if b - a < 40:
                continue
            mid = (a + b) // 2
            h1 = g["vol20"].iloc[a:mid].mean()
            h2 = g["vol20"].iloc[mid:b].mean()
            if np.isfinite(h1) and np.isfinite(h2):
                half_diffs.append(h1 - h2)
    half_diffs = np.array(half_diffs)
    print(
        f"  TR 段 (≥40日): {len(half_diffs)} 个, 前半-后半波动率差: {(half_diffs.mean() if len(half_diffs) else np.nan) * 100:+.2f}pp, 压缩方向一致 {(half_diffs > 0).sum()}/{len(half_diffs)} (理论: 后半更低=正)"
    )
    print("\n[3b] SOS 后 20 日 vs 前 20 日波动率 (扩张假说):")
    sos_diffs = []
    for sym in rec_df["sym"].unique():
        g = rec_df[rec_df["sym"] == sym]
        idx = g.index[g["ev"] == "SOS"].tolist()
        for i in idx:
            pos = g.index.get_loc(i)
            if pos >= 20 and pos + 20 < len(g):
                before = g["vol20"].iloc[pos - 20 : pos].mean()
                after = g["vol20"].iloc[pos : pos + 20].mean()
                if np.isfinite(before) and np.isfinite(after):
                    sos_diffs.append(after - before)
    sos_diffs = np.array(sos_diffs)
    print(
        f"  SOS 事件: {len(sos_diffs)} 个, 后-前波动率差: {(sos_diffs.mean() if len(sos_diffs) else np.nan) * 100:+.2f}pp, 扩张方向一致 {(sos_diffs > 0).sum()}/{len(sos_diffs)} (理论: 突破后扩张=正)"
    )
    print("\n[3c] 阶段波动率排序 (理论: markdown/派发 > markup/吸筹 > TR):")
    ph_vol = {}
    for ph in ["accumulation", "distribution", "markup", "markdown", "trading range"]:
        s = rec_df[rec_df["phase"].str.contains(ph, na=False)]
        if len(s) > 50:
            ph_vol[ph] = s["vol20"].mean() * 100
    for k, v in sorted(ph_vol.items(), key=lambda x: -x[1]):
        print(f"  {k:<14}: {v:.1f}% 年化波动率")


if __name__ == "__main__":
    main()
