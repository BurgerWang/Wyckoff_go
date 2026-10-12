"""Local regressions for historical research isolation and timing contracts."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import wyckoff as wk
from research.common import attach_returns, lag_weights, load_records, metrics
from research.wf_tuning import build_matrices, build_weights


def bars(n=45):
    return pd.DataFrame(
        {"Open": 105.0, "High": 106.0, "Low": 104.0, "Close": 105.0, "Volume": 100.0},
        index=pd.date_range("2025-01-01", periods=n),
    )


def test_no_reversals_have_full_event_schema():
    result = bt.event_study(bars(), "atr", 0.5, 3, horizons=(5, 10), warmup=5)
    assert result["n_events"] == 0
    assert result["base_ok_5d"] == result["base_down_5d"] == 0
    assert np.isnan(result["X_ok_5d"])
    assert np.isnan(result["O_ok_10d"])


def test_validated_numeric_strings_are_used_and_unsorted_data_rejected():
    frame = bars().astype(str)
    assert bt.backtest(frame, "atr", 0.5, 3, warmup=5)["strat"] == 0
    with pytest.raises(ValueError, match="increasing"):
        bt.backtest(frame.iloc[::-1], "atr", 0.5, 3, warmup=5)


def test_delayed_event_is_logged_at_confirmation_and_flat_is_not_bearish(monkeypatch):
    frame = bars()
    pivot, confirmation = frame.index[3], frame.index[6]

    def analyze(columns, scale, prefix, reversal):
        return {
            "events": [
                {
                    "date": pivot,
                    "available_date": confirmation,
                    "event": "AR",
                    "price": 110.0,
                }
            ],
            "phase": {"regime": "unknown"},
        }

    monkeypatch.setattr(wk, "analyze", analyze)
    result = bt.run_wyckoff(frame, "atr", 0.5, 3, warmup=5, horizons=(5,))
    assert result["events"] == [(6, "AR", 110.0)]
    assert result["event_details"][0]["date"] == pivot
    assert result["event_details"][0]["available_date"] == confirmation
    assert result["stats"]["base_up"] == result["stats"]["base_down"] == [0.0]
    assert "AR" not in result["stats"]["n"][5]  # no invented direction


@pytest.mark.parametrize("side,tag,direction", [("accumulation", "AR_UP", 1), ("distribution", "AR_DOWN", -1)])
def test_ar_direction_and_metadata_survive_event_study(monkeypatch, side, tag, direction):
    frame = bars(20)
    frame.iloc[7:] = [105 + direction, 107, 103, 105 + direction, 100]
    observed = {"event": "AR", "date": frame.index[3], "available_date": frame.index[6],
                "price": 110.0, "side": side, "range_id": "r", "status": "confirmed"}
    monkeypatch.setattr(wk, "analyze", lambda *args: {"events": [observed], "phase": {"regime": "unknown"}})
    result = bt.run_wyckoff(frame, "atr", 0.5, 3, warmup=5, horizons=(5,))
    assert result["events"] == [(6, tag, 110.0)]
    assert result["event_details"][0]["side"] == side
    assert result["event_details"][0]["range_id"] == "r"
    assert result["stats"]["ok"][5][tag] == 1.0


@pytest.mark.parametrize("direction", [1, -1])
def test_trade_uses_next_open_risk_and_marks_entry_exit_and_gap(monkeypatch, direction):
    frame = bars()
    if direction == 1:
        frame.iloc[16] = [100, 102, 99, 101, 100]
        frame.iloc[17] = [111, 113, 110, 112, 200]
        frame.iloc[18] = [105, 107, 104, 106, 100]
        frame.iloc[19] = [120, 122, 119, 121, 100]
        frame.iloc[20] = [110, 111, 109, 110, 100]
    else:
        frame.iloc[16] = [110, 111, 108, 109, 100]
        frame.iloc[17] = [99, 100, 97, 98, 200]
        frame.iloc[18] = [105, 106, 103, 104, 100]
        frame.iloc[19] = [120, 121, 118, 119, 100]
        frame.iloc[20] = [130, 131, 129, 130, 100]
    tr = SimpleNamespace(
        top=110,
        bottom=100,
        avg_volume=100,
        bias="accumulation" if direction == 1 else "distribution",
        trend_bars=10,
    )
    monkeypatch.setattr(
        wk,
        "detect_tr",
        lambda columns, box, prefix: (
            tr if prefix.index[-1] == frame.index[15] else None
        ),
    )
    result = bt.run_wyckoff_trade(frame, "atr", 0.5, warmup=14)
    assert result["n_trades"] == 1
    trade = result["trades"][0]
    assert trade[:3] == (frame.index[19], frame.index[20], 120)
    assert trade[3] == (110 if direction == 1 else 130)
    assert trade[4] == "SL"
    assert result["strat"] == pytest.approx(-10 / 120)
    assert result["max_dd"] == pytest.approx(1 - (110 / 120) / (121 / 120))
    assert result["strat"] == pytest.approx(
        np.prod([1 + t[5] for t in result["trades"]]) - 1
    )


def test_ambiguous_bar_and_gap_fills():
    assert bt.barrier_exit(100, 120, 80, 1, 90, 110) == (90, "SL")
    assert bt.barrier_exit(80, 100, 70, 1, 90, 110) == (80, "SL")
    assert bt.barrier_exit(120, 130, 100, -1, 110, 90) == (120, "SL")


EXPERIMENTS = sorted(
    p.stem
    for p in Path("research").glob("*.py")
    if p.stem not in ("__init__", "common")
)


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_research_imports_have_no_io(name, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("research import attempted I/O")

    monkeypatch.setattr(bt, "fetch", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    module = importlib.import_module(f"research.{name}")
    # Reload rebinds the module's classes in place. Other test modules imported
    # them at collection, so put the originals back or `except` stops matching.
    original = dict(vars(module))
    try:
        importlib.reload(module)
        assert capsys.readouterr().out == ""
    finally:
        vars(module).clear()
        vars(module).update(original)


@pytest.mark.parametrize("name", [name for name in EXPERIMENTS
                                  if name not in ("binance_data", "replay_view", "universe",
                                                  "llm_api", "llm_contract")])
def test_experiment_help_parses_before_io(name, monkeypatch, capsys):
    module = importlib.import_module(f"research.{name}")
    monkeypatch.setattr(bt, "fetch", lambda *args: pytest.fail("help downloaded data"))
    if name == "market_study":
        monkeypatch.setattr(module, "load_data", lambda *args, **kwargs: pytest.fail("help downloaded research data"))
    monkeypatch.setattr("sys.argv", [name, "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    option = {"market_study": "--market", "history_data": "--start", "long_study": "--history",
              "phase_validation": "--labels", "llm_pilot": "--packets",
              "llm_prompt": "--pilot"}.get(name, "--symbols")
    assert option in capsys.readouterr().out


def test_backtest_help_and_missing_option_do_not_download(monkeypatch, capsys):
    monkeypatch.setattr(bt, "fetch", lambda *args: pytest.fail("CLI downloaded data"))
    with pytest.raises(SystemExit) as exc:
        bt.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        bt.main(["--years"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "name",
    [
        "entry_study",
        "event_accuracy",
        "tr_stability",
        "tr_meanrev",
        "sl_tp_scan",
        "trade_system",
    ],
)
def test_empty_structure_experiments_complete_locally(name, monkeypatch, capsys):
    module = importlib.import_module(f"research.{name}")
    frame = bars(124)
    calls = []

    def fetch(symbol, years):
        calls.append(symbol)
        return frame

    monkeypatch.setattr(bt, "fetch", fetch)
    monkeypatch.setattr("sys.argv", [name, "--symbols", "A"])
    module.main()
    assert calls == ["A"]  # repeated calls inside the experiment reuse one snapshot.
    assert capsys.readouterr().out


def test_weights_lag_chronologically_without_reordering_output():
    frame = pd.DataFrame(
        {
            "sym": ["A", "B", "A", "B"],
            "date": pd.to_datetime(
                ["2025-01-02", "2025-01-02", "2025-01-01", "2025-01-01"]
            ),
        }
    )
    np.testing.assert_array_equal(lag_weights(frame, [2, 3, 4, 5]), [4, 5, 1, 1])


def test_records_normalize_dates_and_reject_duplicate_keys(tmp_path):
    path = tmp_path / "records.csv"
    frame = pd.DataFrame(
        {
            "sym": ["A", "A"],
            "phase": ["unknown", "unknown"],
            "vol20": [0.1, 0.1],
            "ret": [0.01, -0.01],
            "date": ["2025-03-10 00:00:00-04:00", "2025-03-09 00:00:00-05:00"],
        }
    )
    frame.to_csv(path, index=False)
    records = load_records(path)
    assert records["date"].tolist() == [
        pd.Timestamp("2025-03-09"),
        pd.Timestamp("2025-03-10"),
    ]
    attached = attach_returns(
        records, lambda *args: pytest.fail("stored returns downloaded data"), ["A"]
    )
    assert attached.index.tolist() == [0, 1]
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path)


def test_metrics_include_initial_loss_and_normalize_drawdown():
    assert metrics([-0.2, 0.25, -0.1])[3] == pytest.approx(0.2)
    assert metrics([1.0, -0.25])[3] == pytest.approx(0.25)


def test_turnover_cost_is_single_side_and_includes_initial_exposure():
    from research.vol_robust import portfolio_metrics

    frame = pd.DataFrame(
        {
            "sym": ["A", "A"],
            "ret": [0.0, 0.0],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    result = portfolio_metrics(frame, [1.0, 0.0], fee=0.01)
    assert result[3] == pytest.approx(1 - 0.99**2)
    assert result[4] == pytest.approx(0.01 * 252)


def test_walk_forward_lags_before_calendar_filter_and_clips_actual_target():
    rows = [
        ("A", "2025-01-03", "unknown", 0.1),
        ("A", "2025-01-06", "markup", 0.2),
        ("A", "2025-01-07", "unknown", 0.3),
        ("B", "2025-01-03", "unknown", 0.1),
        ("B", "2025-01-05", "distribution", 0.01),
        ("B", "2025-01-06", "markup", 0.2),
        ("B", "2025-01-07", "unknown", 0.3),
    ]
    frame = pd.DataFrame(rows, columns=["sym", "date", "phase", "vol20"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = 0.01
    frame["vol10"] = frame["vol60"] = frame["vol20"]
    dates, returns, phase, vol = build_matrices(
        frame, lambda *args: pytest.fail("snapshot windows downloaded data"), ["A", "B"]
    )
    assert dates.is_monotonic_increasing
    assert phase[1, 1] == 1  # Sunday's B state applies to Monday.
    assert vol[20][1, 1] == 0.01
    weights = build_weights(np.ones(6), 2 / 3, 20, phase, vol)
    assert weights[1, 1] == 1.5  # clip(0.10 / 0.01), not clip(clip(0.15 / 0.01) * 2/3).


def test_framework_csv_preserves_pivot_and_confirmation_dates(
    tmp_path, monkeypatch, capsys
):
    from research import framework_val

    frame = bars(124)
    pivot, confirmation = frame.index[115], frame.index[121]
    monkeypatch.setattr(bt, "fetch", lambda *args: frame)
    monkeypatch.setattr(
        wk,
        "analyze",
        lambda columns, scale, prefix, reversal: {
            "tr": None,
            "phase": {"regime": "unknown"},
            "events": [
                {
                    "event": "AR",
                    "date": pivot,
                    "available_date": confirmation,
                    "price": 106.0,
                }
            ],
        },
    )
    output = tmp_path / "records.csv"
    monkeypatch.setattr(
        "sys.argv", ["framework_val", "--symbols", "A", "--output", str(output)]
    )
    framework_val.main()
    result = pd.read_csv(output)
    events = json.loads(result.iloc[0]["events_json"])
    assert events[0]["date"] == str(pivot)
    assert events[0]["available_date"] == str(confirmation)
    assert result["ret"].tolist() == [0.0] * 3
    assert {"vol10", "vol20", "vol60"}.issubset(result.columns)
    assert json.loads(result.iloc[1]["events_json"]) == []
