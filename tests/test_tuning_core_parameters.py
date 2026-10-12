"""Parameter responses preserve confirmation, direction, and frozen geometry."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wyckoff_core.features import compute_features, confirmed_swings
from wyckoff_core.models import CoreConfig, TR
from wyckoff_core.structure import replay_structure
from wyckoff_core.timeline import stage_timeline


NEW_INTEGERS = ('core_atr_period', 'local_trend_bars', 'climax_sequence_max_bars')
NEW_FLOATS = ('range_max_pct', 'range_min_height_atr', 'preliminary_volume_min',
              'preliminary_spread_min', 'climax_recovery_close', 'ar_min_move_atr',
              'st_volume_ratio_max', 'interior_test_spread_max', 'probe_volume_ratio_max',
              'probe_test_spread_max', 'lps_volume_ratio_max', 'lps_spread_ratio_max',
              'lpsy_poor_result_max_atr', 'initiative_spread_min', 'initiative_close_min',
              'initiative_move_min_atr')


def bars(closes=None, n=50):
    closes = np.array([105.] * n if closes is None else closes, dtype=float)
    return pd.DataFrame({'Open': closes, 'High': closes + 1, 'Low': closes - 1,
                         'Close': closes, 'Volume': 100.},
                        index=pd.date_range('2025-01-01', periods=len(closes)))


def replay(df, cfg=None, *, initial=True, features=None, swings=None):
    cfg = cfg or CoreConfig()
    tr = TR(0, 4, 110, 100, 5, 105, start_date=df.index[20],
            confirmed_date=df.index[25], established_date=df.index[25],
            range_id='parameter-test', origin='ohlcv_swings') if initial else None
    return replay_structure(df, compute_features(df, cfg) if features is None else features,
                            confirmed_swings(df, cfg) if swings is None else swings,
                            cfg, initial_range=tr)


def names(result):
    return [e['event'] for e in result['events']]


@pytest.mark.parametrize('field', NEW_INTEGERS + NEW_FLOATS)
@pytest.mark.parametrize('value', [True, 0, -1, float('nan'), float('inf')])
def test_new_fields_reject_nonpositive_nonfinite_or_boolean(field, value):
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: value})


@pytest.mark.parametrize('field', NEW_INTEGERS)
def test_new_integer_fields_reject_fractional_types(field):
    with pytest.raises(ValueError, match='integer'):
        CoreConfig(**{field: 2.0})


@pytest.mark.parametrize('field,value', [('range_max_pct', 1.1), ('climax_recovery_close', .49),
                                        ('climax_recovery_close', 1.01), ('initiative_close_min', .5),
                                        ('initiative_close_min', 1.01)])
def test_fraction_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        CoreConfig(**{field: value})


def test_search_policy_relations_do_not_restrict_legacy_constructor():
    # All these were legal callers before parameterization; search restrictions
    # belong in the versioned tuner rather than the general production API.
    cfg = CoreConfig(trend_window=3, min_range_bars=3, swing_confirmation=4,
                     volume_low=1.1, volume_high=1.15, spread_high=1.1)
    assert cfg.local_trend_bars > cfg.trend_window
    assert cfg.preliminary_volume_min > cfg.volume_high
    assert cfg.preliminary_spread_min > cfg.spread_high


def test_atr_period_changes_ewm_without_changing_preceding_baselines():
    df = bars()
    df.iloc[25] = [110, 111, 109, 110, 1000]
    default = compute_features(df)
    short = compute_features(df, CoreConfig(core_atr_period=2))
    assert default.atr.iloc[25] == pytest.approx(2 + 4 / 14)
    assert short.atr.iloc[25] == 4.0
    pd.testing.assert_frame_equal(default[['volume_baseline', 'spread_baseline']],
                                  short[['volume_baseline', 'spread_baseline']])
    pd.testing.assert_frame_equal(short.iloc[:26], compute_features(df.iloc[:26], CoreConfig(core_atr_period=2)))


@pytest.mark.parametrize('field,value', [('initiative_spread_min', 3), ('initiative_close_min', .9),
                                        ('initiative_move_min_atr', 5)])
def test_initiative_thresholds_filter_both_directions_without_geometry_change(field, value):
    df = bars()
    df.iloc[30] = [113, 114, 109, 113, 300]
    for bull in [True, False]:
        sample = df.copy()
        if not bull:
            sample[['Open', 'Close']] = 210 - df[['Open', 'Close']]
            sample['High'], sample['Low'] = 210 - df.Low, 210 - df.High
        expected = 'SOS' if bull else 'SOW'
        before = replay(sample)
        after = replay(sample, CoreConfig(**{field: value}))
        assert expected in names(before) and expected not in names(after)
        assert before['transitions'] == after['transitions']
        assert before['tr'] == after['tr']


def test_default_sow_fraction_keeps_original_literal_boundary():
    df = bars()
    df.iloc[30] = [102, 105, 100, 102, 300]
    features = compute_features(df)
    features.loc[df.index[30], 'close_position'] = np.nextafter(.3, 1.)
    assert 'SOW' not in names(replay(df, features=features))
    features.loc[df.index[30], 'close_position'] = .3
    assert 'SOW' in names(replay(df, features=features))


def test_probe_volume_filter_preserves_single_bar_recovery_and_frozen_boundaries():
    df = bars()
    df.iloc[30] = [101, 102, 99, 101, 100]
    default = replay(df)
    strict = replay(df, CoreConfig(probe_volume_ratio_max=.8))
    assert 'SPRING' in names(default) and 'SPRING' not in names(strict)
    assert default['tr'] == strict['tr']
    df.iloc[30, df.columns.get_loc('Close')] = 99.5  # No recovery inside.
    df.iloc[30, df.columns.get_loc('Open')] = 99.5
    # v3 distinguishes no recovery yet from a subsequent multi-bar reclaim.
    # A later return may create a candidate, never backfill it onto this bar.
    cfg = CoreConfig(probe_volume_ratio_max=2)
    assert 'SPRING' not in names(replay(df.iloc[:31], cfg))
    recovered = next(e for e in replay(df, cfg)['events'] if e['event'] == 'SPRING')
    assert recovered['subtype'] == 'multi_bar_recovery'
    assert recovered['available_date'] == df.index[31]


@pytest.mark.parametrize('field,value', [('lps_volume_ratio_max', .1), ('lps_spread_ratio_max', .1)])
def test_lps_thresholds_keep_confirmed_reaction_timing(field, value):
    df = bars()
    for i, row in [(30, [113, 114, 109, 113, 300]), (31, [112, 114, 111, 112, 100]),
                   (32, [109, 109.5, 108, 109, 50]), (33, [113, 114, 110, 113, 100]),
                   (34, [114, 115, 111, 114, 100])]:
        df.iloc[i] = row
    default = replay(df.iloc[:35])
    lps = next(e for e in default['events'] if e['event'] == 'LPS')
    assert (lps['date'], lps['available_date']) == (df.index[32], df.index[34])
    assert 'LPS' not in names(replay(df.iloc[:34]))
    strict = replay(df.iloc[:35], CoreConfig(**{field: value}))
    assert 'SOS' in names(strict) and 'LPS' not in names(strict)


@pytest.mark.parametrize('field,value', [('range_max_pct', .05), ('range_min_height_atr', 100)])
def test_range_filters_only_creation_not_existing_geometry(field, value):
    df = bars([c for c in (100, 110) * 6 for _ in range(3)])
    assert replay(df, initial=False)['tr'] is not None
    assert replay(df, CoreConfig(**{field: value}), initial=False)['tr'] is None
    # Supplied ranges are not reconstructed or silently resized by thresholds.
    assert replay(df, CoreConfig(**{field: value}))['tr'].range_id == 'parameter-test'


def test_nondefault_timeline_remains_prefix_stable():
    rng = np.random.default_rng(7)
    df = bars(105 + np.cumsum(rng.normal(0, 1, 120)))
    cfg = replace(CoreConfig(), core_atr_period=10, local_trend_bars=3,
                  range_max_pct=.5, initiative_close_min=.8,
                  probe_volume_ratio_max=.9, climax_sequence_max_bars=40)
    full = stage_timeline(df, cfg)
    for end in [30, 65, 90]:
        prefix = stage_timeline(df.iloc[:end], cfg)
        assert full['rows'][:end] == prefix['rows']
        assert [e for e in full['events'] if e['available_date'] <= df.index[end - 1]] == prefix['events']


def stopping_sequence(*, preliminary=False):
    df = bars()
    df.iloc[15] = [110, 111, 109, 110, 100]
    df.iloc[17] = [99, 100, 99, 99, 100]
    df.iloc[19] = [100, 101, 99, 100, 100]
    df.iloc[20] = [101, 102, 98, 101, 300]
    df.iloc[21] = [107, 108, 106, 107, 100]
    df.iloc[22] = [105, 107, 104, 105, 100]
    df.iloc[30] = [99, 100, 98.5, 99, 50]
    # Known ratios isolate the context/sequence gates from moving baselines.
    features = compute_features(df)
    features[['atr', 'volume_ratio', 'spread_ratio']] = 1.
    features.loc[df.index[20], ['volume_ratio', 'spread_ratio']] = 1.3 if preliminary else 2.
    features.loc[df.index[20], 'close_position'] = .75
    return df, features


@pytest.mark.parametrize('field,value,event', [('local_trend_bars', 1, 'SC'),
                                              ('climax_recovery_close', .8, 'SC'),
                                              ('ar_min_move_atr', 20., 'AR'),
                                              ('climax_sequence_max_bars', 5, 'ST'),
                                              ('st_volume_ratio_max', .8, 'ST')])
def test_stopping_sequence_thresholds_preserve_order_and_raw_contraction(field, value, event):
    df, features = stopping_sequence()
    cfg = CoreConfig(trend_window=3)
    default = replay(df, cfg, initial=False, features=features, swings=[])
    assert [e for e in names(default) if e in ('SC', 'AR', 'ST')] == ['SC', 'AR', 'ST']
    ar = next(e for e in default['events'] if e['event'] == 'AR')
    assert (ar['date'], ar['available_date']) == (df.index[21], df.index[22])
    strict = replay(df, replace(cfg, **{field: value}), initial=False, features=features, swings=[])
    assert event not in names(strict)
    # Permissive ratio thresholds still cannot remove raw contraction checks.
    df.iloc[30, df.columns.get_loc('Volume')] = 400
    assert 'ST' not in names(replay(df, replace(cfg, st_volume_ratio_max=5),
                                   initial=False, features=features, swings=[]))


@pytest.mark.parametrize('field', ['preliminary_volume_min', 'preliminary_spread_min'])
def test_preliminary_thresholds_are_independent_of_climax_thresholds(field):
    df, features = stopping_sequence(preliminary=True)
    cfg = CoreConfig(trend_window=3)
    default = replay(df, cfg, initial=False, features=features, swings=[])
    assert 'PS' in names(default) and 'SC' not in names(default)
    strict = replay(df, replace(cfg, **{field: 1.4}), initial=False, features=features, swings=[])
    assert 'PS' not in names(strict) and 'SC' not in names(strict)


def test_probe_test_spread_cannot_bypass_lower_parent_volume_or_confirmation():
    df = bars()
    df.iloc[30] = [101, 102, 99, 101, 80]
    df.iloc[31] = [103, 104, 101, 103, 100]
    df.iloc[32] = [104, 105, 102, 104, 100]
    df.iloc[36] = [101, 102, 100.5, 101, 30]
    df.iloc[37] = [104, 105, 103, 104, 100]
    df.iloc[38] = [105, 106, 104, 105, 100]
    default = replay(df.iloc[:39])
    test = next(e for e in default['events'] if e['event'] == 'TEST')
    assert test['available_date'] == df.index[38]
    assert 'TEST' not in names(replay(df.iloc[:38]))
    assert 'TEST' not in names(replay(df.iloc[:39], CoreConfig(probe_test_spread_max=.1)))
    df.iloc[36, df.columns.get_loc('Volume')] = 90
    assert 'TEST' not in names(replay(df.iloc[:39], CoreConfig(probe_test_spread_max=5)))


def test_interior_test_spread_filter_preserves_four_pivot_response():
    closes = np.full(52, 105.)
    closes[26:43] = [105, 106, 107, 106, 104.5, 103.5, 103, 104, 106,
                     107, 108.5, 107, 106, 105, 104.5, 106, 107]
    df = bars(closes)
    df.iloc[32] = [103, 104, 101.5, 103, 100]
    df.iloc[40] = [104.5, 105, 103.5, 104.5, 40]
    default = replay(df.iloc[:43])
    test = next(e for e in default['events'] if e.get('parent_type') == 'confirmed_swing')
    assert (test['date'], test['available_date']) == (df.index[40], df.index[42])
    assert not any(e.get('parent_type') == 'confirmed_swing'
                   for e in replay(df.iloc[:43], CoreConfig(interior_test_spread_max=.1))['events'])
    df.iloc[40, df.columns.get_loc('Volume')] = 120
    assert not any(e.get('parent_type') == 'confirmed_swing'
                   for e in replay(df.iloc[:43], CoreConfig(interior_test_spread_max=5))['events'])


def test_high_effort_lpsy_poor_result_gate_uses_prior_bar_atr_feature():
    df = bars()
    for i, row in [(30, [97, 101, 96, 97, 300]), (31, [98, 99, 96, 98, 100]),
                   (32, [101, 102, 100.5, 101, 300]), (33, [97, 100, 96, 97, 100]),
                   (34, [96, 99, 95, 96, 100])]:
        df.iloc[i] = row
    features = compute_features(df)
    assert features.price_change_atr.iloc[32] == pytest.approx(
        (df.Close.iloc[32] - df.Close.iloc[31]) / features.atr.iloc[31])
    permissive = replay(df.iloc[:35], CoreConfig(lpsy_poor_result_max_atr=5))
    lpsy = next(e for e in permissive['events'] if e['event'] == 'LPSY')
    assert lpsy['subtype'] == 'high_effort_poor_result'
    assert (lpsy['date'], lpsy['available_date']) == (df.index[32], df.index[34])
    assert 'LPSY' not in names(replay(df.iloc[:35], CoreConfig(lpsy_poor_result_max_atr=.1)))
