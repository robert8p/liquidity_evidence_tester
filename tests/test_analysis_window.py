from pathlib import Path

import pandas as pd

from app.release_calendar import attach_h41_alignment
from app.research.features import build_net_liquidity, attach_forward_returns_to_features, price_to_weekly
from app.research.net_liquidity import _restrict_aligned_to_analysis_window


def test_release_aligned_analysis_window_prevents_pre_start_target_return_attachment():
    # Macro context starts long before the requested analysis date to allow rolling features.
    idx = pd.date_range('2003-01-01', '2020-12-31', freq='W-WED', tz='UTC')
    fed = pd.DataFrame({'value': range(1_000_000, 1_000_000 + len(idx))}, index=idx)
    tga = pd.DataFrame({'value': range(100_000, 100_000 + len(idx))}, index=idx)
    rrp = pd.DataFrame({'value': [10.0] * len(idx)}, index=idx)
    features = build_net_liquidity(fed, tga, rrp, fed_assets_scale=0.001, tga_scale=0.001, onrrp_scale=1.0)

    aligned_all = attach_h41_alignment(features, instrument_type='crypto')
    aligned_window = _restrict_aligned_to_analysis_window(aligned_all, '2018-01-01', '2020-12-31').set_index('effective_trade_at_utc')

    assert not aligned_window.empty
    assert aligned_window.index.min() >= pd.Timestamp('2018-01-01T00:00:00Z')
    assert aligned_window['observed_at_utc'].min() >= pd.Timestamp('2017-12-27T00:00:00Z')

    # Target data starts at the requested analysis date. The old bug would have attached
    # this first target return to many pre-2018 feature rows. The restricted window prevents that.
    price_idx = pd.date_range('2018-01-05', '2020-12-31', freq='W-FRI', tz='UTC')
    prices = pd.DataFrame({'close': [100 + i for i in range(len(price_idx))]}, index=price_idx)
    analysis = attach_forward_returns_to_features(aligned_window, price_to_weekly(prices), [1, 4])
    assert analysis.index.min() >= pd.Timestamp('2018-01-01T00:00:00Z')
    assert len(analysis) < len(features)


def test_target_history_starting_late_is_not_backfilled_across_years():
    # Release-aligned features start in 2018, but target prices only start in 2021
    # (matching the real Massive Stock Starter response observed in production).
    feature_idx = pd.date_range('2018-01-05T14:30:00Z', '2021-06-30T14:30:00Z', freq='W-FRI')
    features = pd.DataFrame({'liq_impulse_z_52': range(len(feature_idx))}, index=feature_idx)
    price_idx = pd.date_range('2021-05-21T04:00:00Z', '2021-06-30T04:00:00Z', freq='W-FRI')
    prices = pd.Series([100 + i for i in range(len(price_idx))], index=price_idx, name='close')

    analysis = attach_forward_returns_to_features(features, prices, [1, 2])

    pre_coverage = analysis[analysis.index < pd.Timestamp('2021-05-01T00:00:00Z')]
    assert pre_coverage['target_anchor_utc'].isna().all()
    assert pre_coverage['fwd_logret_1w'].isna().all()

    matched = analysis[analysis['target_anchor_utc'].notna()]
    assert not matched.empty
    assert matched.index.min() >= pd.Timestamp('2021-05-14T00:00:00Z')
