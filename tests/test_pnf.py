import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pnf import (IncrementalPnF, PnF, _extend_scale, build, build_config,
                 consensus, horizontal_count_target, main, make_scale, plot,
                 report, summary, trendline_status, vertical_count_target)


def bars(values):
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({name: values.copy() for name in ("Open", "High", "Low", "Close")},
                         index=pd.date_range("2024-01-01", periods=len(values)))
    frame["Volume"] = 100.0
    frame.attrs["symbol"] = "TEST"
    return frame


def chart(columns, box=1):
    return PnF("TEST", box, scale=make_scale(50, 150, box=box),
               columns=[[d, lo, hi, 1, pd.Timestamp("2024-01-01")]
                        for d, lo, hi in columns])


def test_first_price_and_sub_box_noise_do_not_invent_direction(tmp_path):
    frame = bars([100.8, 100.9, 100.7])
    p = build(frame, 1)
    assert p.direction is None and p.columns == []
    assert "undetermined" in report(p, frame)
    assert consensus([("fixed", p, "", "")], 100.7)["code"] == "FLAT"
    assert "方向未确定" in summary([("fixed", p, "", "")], frame)
    plot(p, frame, tmp_path / "neutral.png")
    assert (tmp_path / "neutral.png").stat().st_size > 1000


def test_close_box_labels_reversals_and_all_bar_dates():
    frame = bars([100, 105, 104.6, 104.7, 102, 103, 105])
    p = build(frame, 1)
    assert [(c[0], c[1], c[2], c[3]) for c in p.columns] == [
        ("X", 101, 105, 4), ("O", 102, 104, 2), ("X", 103, 105, 1)]
    assert [c[4] for c in p.columns] == list(frame.index[[3, 5, 6]])
    assert p.column_start_dates == list(frame.index[[0, 4, 6]])


def test_fractional_o_low_does_not_suppress_valid_high_reversal():
    p = IncrementalPnF(make_scale(95, 110, box=1))
    p.update(105, 105, 105, "2024-01-01")
    p.update(104, 100, 101, "2024-01-02")
    p.update(103, 99.5, 102, "2024-01-03")
    assert [(c[0], c[1], c[2]) for c in p.columns] == [("O", 100, 104), ("X", 101, 103)]


def test_high_low_extension_has_priority_over_same_bar_reversal():
    p = IncrementalPnF(make_scale(90, 120, box=1))
    for day, (h, l, c) in enumerate([(100, 100, 100), (105, 101, 104),
                                    (106, 98, 102), (105, 103, 104)], 1):
        p.update(h, l, c, f"2024-01-{day:02}")
    assert [(c[0], c[1], c[2]) for c in p.columns] == [("X", 101, 106), ("O", 103, 105)]


@pytest.mark.parametrize("kind,kwargs", [("fixed", {"box": 1}), ("pct", {"pct": .01}),
                                         ("traditional", {})])
def test_batch_and_incremental_agree_across_large_gaps(kind, kwargs):
    frame = bars([100, 101, 130, 90, 150, 80, 84])
    scale = make_scale(98, 102, kind, **kwargs)
    incremental = IncrementalPnF(scale, use_close=True)
    for date, row in frame.iterrows():
        incremental.update(row.High, row.Low, row.Close, date)
    batch = build(frame, scale)
    assert batch.columns == incremental.columns
    assert batch.signals == incremental.signals
    assert batch.top <= 150 and batch.bottom >= 80
    assert batch.scale[0] < 80 and batch.scale[-1] > 150
    if kind == "pct":
        np.testing.assert_allclose(batch.scale[1:] / batch.scale[:-1], 1.01)


def test_percentage_grid_does_not_reanchor_when_new_low_is_added():
    a, b = make_scale(90, 120, "pct"), make_scale(60, 120, "pct")
    assert set(a).issubset(set(b))


@pytest.mark.parametrize("price,step", [(.1, .0625), (.5, .125), (2, .25), (10, .5),
                                        (50, 1), (150, 2), (300, 4), (700, 5),
                                        (5000, 50), (30000, 500)])
def test_complete_official_traditional_table(price, step):
    scale = make_scale(price, price, "traditional")
    k = np.searchsorted(scale, price, side="right") - 1
    assert scale[k + 1] - scale[k] == step


def test_traditional_extension_keeps_tier_transitions():
    scale = make_scale(95, 105, "traditional")
    extended = _extend_scale(scale, 1, 1200, 3)
    assert set(scale).issubset(set(extended))
    for price, step in [(2, .25), (50, 1), (150, 2), (300, 4), (700, 5), (1100, 50)]:
        k = np.searchsorted(extended, price, side="right") - 1
        assert extended[k + 1] - extended[k] == step


@pytest.mark.parametrize("kwargs", [{"box": 0}, {"box": -1}, {"box": np.nan},
                                    {"kind": "pct", "pct": 0},
                                    {"kind": "pct", "pct": -.1}])
def test_invalid_scale_parameters_fail_promptly(kwargs):
    with pytest.raises(ValueError):
        make_scale(1, 10, **kwargs)


@pytest.mark.parametrize("scale", [[1], [1, 1], [2, 1], [1, np.inf], [[1, 2], [3, 4]]])
def test_invalid_grids_are_rejected(scale):
    with pytest.raises(ValueError):
        IncrementalPnF(scale)


@pytest.mark.parametrize("reversal", [0, -1, 2.5, True])
def test_invalid_reversal_is_rejected(reversal):
    with pytest.raises(ValueError):
        build(bars([100, 101]), 1, reversal=reversal)


def test_vertical_count_includes_both_end_boxes_and_freezes_after_reversal():
    p = chart([("X", 100, 104), ("O", 101, 103), ("X", 102, 106)])
    result = vertical_count_target(p)
    assert result["box_count"] == 5
    assert result["target"] == 117
    assert result["fixed"] is False
    p.columns.append(["O", 103, 105, 1, pd.Timestamp("2024-01-02")])
    assert vertical_count_target(p)["fixed"] is True


def test_horizontal_count_requires_exit_of_entire_base():
    p = chart([("X", 100, 110), ("O", 104, 109), ("X", 105, 108),
               ("O", 104, 107), ("X", 105, 109)])
    assert horizontal_count_target(p) is None
    p.columns[-1][2] = 111
    result = horizontal_count_target(p)
    assert result["base_cols"] == 5 and result["target"] == 119


def test_horizontal_count_does_not_swallow_chain_overlapping_trend():
    p = chart([("X", 80, 90), ("O", 87, 89), ("X", 88, 100),
               ("O", 97, 99), ("X", 98, 110), ("O", 107, 109), ("X", 108, 120)])
    assert horizontal_count_target(p) is None


def test_subset_consensus_and_beyond_reversal_level_are_handled():
    p = chart([("X", 101, 105)])
    assert consensus([("atr", p, "", "")], 105)["code"] == "WEAK_LONG"
    assert consensus([("atr", p, "", ""), ("pct", p, "", "")], 105)["code"] == "STRONG_LONG"
    assert consensus([("atr", p, "", "")], 98)["code"] == "WEAK_LONG_CRITICAL"


def test_low_price_atr_retains_precision_and_symbol():
    frame = bars([.5, .51, .49, .52])
    p, label = build_config("LOW", frame, "atr", reversal=2)
    assert 0 < p.box < .01
    assert p.symbol == "LOW" and p.reversal == 2 and "历史快照" in label
    with pytest.raises(ValueError, match="ATR box"):
        build_config("FLAT", bars([.5] * 4), "atr")


def test_import_has_no_plotting_side_effects():
    result = subprocess.run([sys.executable, "-c", "import sys; import pnf; "
                             "assert 'matplotlib' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plot_nonuniform_grid_and_one_box_column(tmp_path):
    frame = bars([100, 101, 106, 102, 107])
    p = build(frame, make_scale(98, 108, "pct"))
    plot(p, frame, tmp_path / "percentage.png")
    assert (tmp_path / "percentage.png").stat().st_size > 1000


def test_compatibility_cli_help_does_not_fetch(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--pct" in capsys.readouterr().out


def test_compatibility_cli_error_is_nonzero_and_visible(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("invalid source data")
    monkeypatch.setattr("pnf.run", fail)
    assert main(["TEST", "--atr"]) == 1
    assert "invalid source data" in capsys.readouterr().err
