import numpy as np
import pandas as pd
import pytest

import wyckoff as wk


def bars(n=50, close=105, spread=2):
    return pd.DataFrame({"Open": float(close), "High": close + spread / 2,
                         "Low": close - spread / 2, "Close": float(close), "Volume": 100.0},
                        index=pd.date_range("2025-01-01", periods=n))


def put(df, i, high, low, close, volume):
    df.iloc[i] = [close, high, low, close, volume]


def event(df, i, name, price=105, **extra):
    return {"event": name, "date": df.index[i], "available_date": df.index[i], "price": price, "vol": 1.0, **extra}


def trading_range(df, confirmed=25):
    return wk.TR(0, 4, 110, 100, 5, 105, avg_volume=100,
                 start_date=df.index[20], end_date=df.index[confirmed - 1], confirmed_date=df.index[confirmed])


def test_range_excludes_active_extremes_and_volume_and_counts_completed_columns():
    df = bars()
    columns = [["X", 101, 110, 11, df.index[10]], ["O", 100, 109, 5, df.index[15]],
               ["X", 101, 110, 5, df.index[20]], ["O", 100, 109, 5, df.index[25]],
               ["X", 101, 110, 5, df.index[30]], ["O", 90, 109, 19, df.index[49]]]
    df.loc[df.index[31]:, "Volume"] = 9000
    tr = wk.detect_tr(columns, 1, df)
    assert (tr.bottom, tr.top, tr.width, tr.end_col) == (100, 110, 5, 4)
    assert tr.avg_volume == 100
    assert tr.confirmed_date == df.index[31]
    assert wk.detect_tr(columns[:5], 1, df) is None


def test_range_events_start_after_formation_and_retain_indicator_warmup():
    df = bars()
    put(df, 30, 106, 99, 101, 50)
    put(df, 42, 106, 99, 101, 50)
    tr = trading_range(df, confirmed=35)
    events = wk.detect_events(df, tr, lookback=10)
    springs = [e for e in events if e["event"] == "SPRING"]
    assert [e["date"] for e in springs] == [df.index[42]]


def test_trend_bars_before_entering_congestion_do_not_dilute_its_volume():
    df = bars()
    df.iloc[:6] = [90, 91, 89, 90, 10000]
    columns = [["X", 90, 110, 11, df.index[10]], ["O", 100, 109, 5, df.index[15]],
               ["X", 101, 110, 5, df.index[20]], ["O", 100, 109, 5, df.index[25]],
               ["X", 101, 110, 5, df.index[30]], ["O", 100, 109, 19, df.index[49]]]
    tr = wk.detect_tr(columns, 1, df)
    assert tr.start_date == df.index[6]
    assert tr.avg_volume == 100


def test_sustained_breakout_is_not_repeated_sos():
    df = bars()
    put(df, 30, 115, 102, 112, 300)
    put(df, 31, 116, 103, 113, 300)
    assert [e["date"] for e in wk.detect_events(df, trading_range(df)) if e["event"] == "SOS"] == [df.index[30]]


def test_up_closing_selling_climax_and_ar_confirmation_are_causal():
    df = bars(25)
    for i in range(20):
        c = 120 - i
        put(df, i, c + 1, c - 1, c, 100)
    put(df, 20, 105, 95, 104, 1000)  # closes UP, following a downtrend
    put(df, 21, 110, 102, 109, 100)
    put(df, 22, 108, 100, 101, 100)
    put(df, 23, 120, 101, 119, 100)
    early = wk.detect_events(df.iloc[:23], None)
    assert any(e["event"] == "SC" and e["date"] == df.index[20] for e in early)
    ar = next(e for e in early if e["event"] == "AR")
    assert (ar["date"], ar["available_date"]) == (df.index[21], df.index[22])
    assert ar in wk.detect_events(df, None)


def test_lps_requires_later_confirmation_and_invalidated_pivot_is_removed():
    df = bars(23, close=101)
    tr = wk.TR(0, 4, 102, 98, 5, 100)
    put(df, 20, 104, 101, 103, 300)
    put(df, 21, 102, 100, 101, 50)
    put(df, 22, 103, 101, 102.5, 100)
    sos = event(df, 20, "SOS", 103)
    assert wk._lps_lpsy(df.iloc[:22], tr, [sos]) == []
    lps = wk._lps_lpsy(df, tr, [sos])[0]
    assert (lps["date"], lps["available_date"], lps["price"]) == (df.index[21], df.index[22], 100)
    assert lps["vol"] > 0
    extra = df.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex([df.index[-1] + pd.Timedelta(days=1)])
    df = pd.concat([df, extra])
    put(df, 22, 101.5, 99, 100, 500)  # undercuts old pivot without qualifying
    put(df, 23, 103, 100.5, 102, 50)
    assert wk._lps_lpsy(df, tr, [sos]) == []


def test_optional_spring_and_new_climax_invalidation():
    df = bars()
    events = [event(df, i + 20, name, side="accumulation") for i, name in enumerate(("SC", "AR", "ST", "SOS", "LPS"))]
    assert wk.sequence_state(list(reversed(events)))["acc_complete"]
    events.append(event(df, 27, "BC"))
    result = wk.sequence_state(events)
    assert not result["acc_complete"] and not result["sos_confirmed"]
    assert result["dist_progress"] == 1


def test_new_test_resets_completed_cycle():
    df = bars()
    events = [event(df, 20 + i, name, range_id="observed") for i, name in enumerate(("SPRING", "SOS", "LPS", "SPRING"))]
    state = wk.sequence_state(events)
    assert state["accum_progress"] == 4
    assert state["acc_lps"] is None and not state["sos_confirmed"]


def test_single_breakout_is_phase_d_and_conflicting_test_is_uncertain():
    df = bars()
    tr = trading_range(df)
    events = [event(df, 30, "SOS", range_id="observed", scope="breakout")]
    assert wk.classify_phase(events, tr, [], 112, 1)["phase"] == "D"
    assert wk.classify_phase(events, tr, [], 112, 3)["phase"] == "D"  # closes alone cannot establish E
    context = {"price_structure": {"direction": "up"}}
    events.append(event(df, 33, "LPS", range_id="observed", status="confirmed"))
    assert wk.classify_phase(events, tr, [], 112, 3, context=context)["phase"] == "D"
    context["price_structure"].update(last_high={"date": df.index[35]}, last_low={"date": df.index[33]})
    e_phase = wk.classify_phase(events, tr, [], 112, 3, context=context)
    assert e_phase["phase"] == "E"
    assert e_phase["hypotheses"]["accumulation"]["structure_check"]["status"] == "unavailable"
    ambiguous = wk.classify_phase([event(df, 31, "AMBIGUOUS_TEST")], tr, [], 105)
    assert ambiguous["phase"] == "?" and ambiguous["status"] == "ambiguous"


def test_no_range_never_fabricates_an_a_to_e_stage():
    df = bars()
    result = wk.classify_phase([event(df, 30, "SOS")], None, [], 112)
    assert result["phase"] == "?"


def test_count_is_bounded_and_not_invented_for_unknown_or_dynamic_range():
    df = bars()
    columns = [["X" if i % 2 == 0 else "O", 100, 110, 2, df.index[i + 20]] for i in range(9)]
    tr = wk.TR(2, 6, 110, 100, 5, 105, start_date=df.index[22])
    scale = np.arange(90.0, 121.0)
    e = event(df, 27, "LPS", 105, status="confirmed")
    move = event(df, 26, "SOS", 112, range_confirmed_at=df.index[25], scope="breakout")
    result = wk.wyckoff_count(columns, scale, tr, [move, e])
    assert result["width_bull"] == 5 and result["extension_bull"] == 15
    assert result["bearish"] is None
    assert wk.wyckoff_count(columns, scale, tr, []) is None
    assert wk.wyckoff_count(columns, 90 * 1.01 ** np.arange(40), tr, [move, e]) is None


def test_zero_volume_has_no_volume_events_and_insufficient_data_has_price():
    df = bars()
    df["Volume"] = 0.0
    assert wk.detect_events(df, trading_range(df)) == []
    result = wk.analyze([], np.arange(90.0, 121.0), df.iloc[:30])
    assert result["last_price"] == 105 and result["laws"]["supply_demand"]["status"] == "unavailable"
    warmup = wk.analyze([], np.arange(90.0, 121.0), df.iloc[:20])
    assert warmup["phase"]["status"] == "insufficient_data"


def test_future_or_unsorted_pnf_columns_are_rejected():
    df = bars()
    with pytest.raises(ValueError, match="snapshot"):
        wk.analyze([["X", 100, 110, 2, df.index[-1] + pd.Timedelta(days=1)]], np.arange(90.0, 121.0), df)


def test_ordinary_reversal_does_not_erase_a_known_spring_or_change_baseline():
    import pnf
    closes = list(np.linspace(130, 112, 22)) + [c for c in (100, 110, 100, 110, 100, 110) for _ in range(3)] + [101, 110]
    df = bars(len(closes))
    for i, close in enumerate(closes):
        put(df, i, close + 0.5, close - 0.5, close, 100)
    put(df, len(df) - 2, 102, 99, 101, 60)
    put(df, len(df) - 1, 110, 109.5, 110, 100)
    early_chart = pnf.build(df.iloc[:-1], 1, use_close=True)
    full_chart = pnf.build(df, 1, use_close=True)
    early = wk.analyze(early_chart.columns, early_chart.scale, df.iloc[:-1])
    full = wk.analyze(full_chart.columns, full_chart.scale, df)
    spring = next(e for e in early["events"] if e["event"] == "SPRING")
    assert spring in full["events"]
    assert full["phase"]["phase"] == "B"  # a probe alone is not a confirmed Phase C
    assert early["tr"].confirmed_date == full["tr"].confirmed_date
    assert early["tr"].avg_volume == full["tr"].avg_volume


def test_new_climax_cannot_confirm_opposite_previous_cycle_ar():
    df = bars(27)
    closes = [90] * 10 + list(np.linspace(90, 110, 10)) + [110, 105, 100, 95, 90, 89, 90]
    for i, close in enumerate(closes):
        put(df, i, close + 1, close - 1, close, 100)
    put(df, 20, 117, 109, 110, 300)
    put(df, 25, 91, 80, 89, 350)
    tr = wk.TR(0, 4, 120, 80, 5, 100, start_date=df.index[0], confirmed_date=df.index[20], avg_volume=100)
    events = wk.detect_events(df, tr)
    assert not any(e["event"] == "AR" and e.get("side") == "distribution" and e["available_date"] > df.index[25] for e in events)
    assert any(e["event"] == "SC" and e["date"] == df.index[25] for e in events)


def test_count_uses_the_snapped_row_and_suppresses_nonpositive_targets():
    df = bars()
    columns = [["X" if i % 2 else "O", 8, 10, 2, df.index[i + 20]] for i in range(8)]
    tr = wk.TR(0, 6, 10, 8, 7, 9)
    assert wk.wyckoff_count(columns, np.arange(5.0, 15.0, 0.5), tr, [event(df, 30, "LPSY", 8)]) is None
    columns = [["X", 101, 110, 2, df.index[20 + i]] for i in range(5)]
    tr = wk.TR(0, 4, 110, 100, 5, 105)
    assert wk.wyckoff_count(columns, np.arange(90.0, 121.0), tr, [event(df, 30, "LPS", 100)]) is None
