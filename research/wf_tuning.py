"""Historical wf_tuning experiment; run explicitly with python -m research.wf_tuning.

Importing this module performs no downloads, file writes or research run.
Results are descriptive research, not an executable trading recommendation.
"""

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt
import numpy as np
import pandas as pd
from research.common import (
    SYMBOLS,
    cached_fetch,
    experiment_args,
    load_records,
    attach_returns,
    volatility_weights,
    metrics,
    session_dates,
)

syms = list(SYMBOLS)

PHASES = (
    "accumulation",
    "distribution",
    "markup",
    "markdown",
    "trading range",
    "unknown",
)


def build_matrices(recs, fetch, symbols, years=10):
    """Align all symbols on common dates, lagging on each original calendar first."""
    recs = recs[recs["sym"].isin(symbols)].sort_values(["sym", "date"]).copy()
    recs["phase_lag"] = recs.groupby("sym")["phase"].shift(1).fillna("unknown")
    full = recs.groupby("date")["sym"].nunique()
    dates = full[full == len(symbols)].index.sort_values()
    if len(dates) < 2:
        raise ValueError("Not enough common-symbol dates for walk-forward research")
    R = (
        recs.pivot(index="date", columns="sym", values="ret")
        .reindex(index=dates, columns=symbols)
        .to_numpy()
    )
    phase_index = {phase: i for i, phase in enumerate(PHASES)}
    recs["phase_lag"] = recs["phase_lag"].map(phase_index).fillna(5).astype(int)
    PH = (
        recs.pivot(index="date", columns="sym", values="phase_lag")
        .reindex(index=dates, columns=symbols)
        .to_numpy(dtype=int)
    )
    V = {}
    for window in (10, 20, 60):
        column = f"vol{window}"
        if column in recs:
            recs[f"lag_{column}"] = recs.groupby("sym")[column].shift(1)
            V[window] = (
                recs.pivot(index="date", columns="sym", values=f"lag_{column}")
                .reindex(index=dates, columns=symbols)
                .to_numpy()
            )
        else:
            # Legacy records lack these windows; recompute on the source calendar,
            # then lag before filtering common dates (including crypto weekends).
            values = []
            for symbol in symbols:
                df = fetch(symbol, years)
                log_returns = np.log(df["Close"] / df["Close"].shift(1))
                volatility = log_returns.rolling(window).std().shift(1) * np.sqrt(252)
                values.append(
                    pd.Series(
                        volatility.to_numpy(),
                        index=session_dates(df.index),
                        name=symbol,
                    )
                )
            V[window] = pd.concat(values, axis=1).reindex(dates)[symbols].to_numpy()
    return dates, R, PH, V


def build_weights(coeff_arr, tv_scale, win, phase_matrix, volatility):
    """Use already lagged inputs and clip the actual target exactly once."""
    phase = np.asarray(coeff_arr)[phase_matrix]
    return phase * volatility_weights(volatility[win], 0.15 * tv_scale)


def port_sharpe(weights, returns, indices):
    if len(indices) < 2:
        raise ValueError("A split must contain at least two portfolio observations")
    daily = (weights[indices] * returns[indices]).mean(axis=1)
    return metrics(daily)[2]


def main():
    args = experiment_args(syms, years=10, record_option="records")
    fetch = cached_fetch(bt.fetch)
    recs = attach_returns(load_records(args.records), fetch, args.symbols, args.years)
    dates, R, PH, V = build_matrices(recs, fetch, args.symbols, args.years)
    print(
        f"样本: {len(dates)} 个共同日期 ({dates.min().date()} ~ {dates.max().date()})"
    )
    acc_vals = [1.0, 1.2, 1.4]
    dis_vals = [0.5, 0.7, 0.9]
    markup_vals = [1.0, 1.2, 1.4]
    markdown_vals = [0.3, 0.5, 0.7]
    tv_vals = [0.10, 0.15, 0.20]
    win_vals = [10, 20, 60]
    cutoff = pd.Timestamp("2023-01-01")
    train_idx = np.where(dates < cutoff)[0]
    test_idx = np.where(dates >= cutoff)[0]
    if len(train_idx) < 2 or len(test_idx) < 2:
        raise ValueError(
            "Both training and validation splits need at least two common dates"
        )
    print(f"训练期: {len(train_idx)} 天, 验证期: {len(test_idx)} 天")
    grid = []
    for acc in acc_vals:
        for dis in dis_vals:
            for mu in markup_vals:
                for md in markdown_vals:
                    for tv in tv_vals:
                        for win in win_vals:
                            grid.append((acc, dis, mu, md, tv, win))
    print(f"网格: {len(grid)} 组合")
    best_train, best_sh = None, -1e9
    results = []
    for g_ in grid:
        acc, dis, mu, md, tv, win = g_
        coeff = np.array([acc, dis, mu, md, 1.0, 1.0])
        W = build_weights(coeff, tv / 0.15, win, PH, V)
        sh_tr = port_sharpe(W, R, train_idx)
        sh_te = port_sharpe(W, R, test_idx)
        results.append((sh_tr, sh_te, g_))
        if sh_tr > best_sh:
            best_sh, best_train = sh_tr, g_
    results.sort(key=lambda x: -x[0])
    print(f"\n===== 训练期最优 (前5年) =====")
    acc, dis, mu, md, tv, win = best_train
    print(f"吸筹={acc} 派发={dis} markup={mu} markdown={md} 目标波动={tv} 窗口={win}")
    print(f"训练 Sharpe: {best_sh:.3f}")
    shA_tr = port_sharpe(np.ones_like(R), R, train_idx)
    shA_te = port_sharpe(np.ones_like(R), R, test_idx)
    print(f"\n基准: A 恒定 训练 {shA_tr:.3f} / 验证 {shA_te:.3f}")
    for sh_tr, sh_te, g_ in results:
        if g_ == best_train:
            print(
                f"训练最优 → 验证 Sharpe: {sh_te:.3f} (vs A {shA_te:.3f}, Δ{sh_te - shA_te:+.3f})"
            )
    trs = np.array([r[0] for r in results])
    tes = np.array([r[1] for r in results])
    print(
        f"\n参数面: 训练 Sharpe {trs.min():.2f}~{trs.max():.2f} (中位 {np.median(trs):.2f})"
    )
    print(f"验证 Sharpe {tes.min():.2f}~{tes.max():.2f} (中位 {np.median(tes):.2f})")
    corr = np.corrcoef(trs, tes)[0, 1]
    print(f"训练-验证相关: {corr:.3f} (低=过拟合风险)")
    top10_tr = np.argsort(trs)[-10:]
    print(
        f"训练 top10 的验证中位: {np.median(tes[top10_tr]):.3f} vs 验证全网格中位 {np.median(tes):.3f}"
    )
    best_te_idx = np.argmax(tes)
    print(f"全网格验证最优: {tes[best_te_idx]:.3f} (参数 {results[best_te_idx][2]})")


if __name__ == "__main__":
    main()
