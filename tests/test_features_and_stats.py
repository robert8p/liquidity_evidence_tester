import pandas as pd
from app.research.demo import synthetic_macro_and_prices
from app.research.features import build_net_liquidity, attach_forward_returns_to_features, price_to_weekly
from app.research.stats import cross_corr_table, quintile_spread, arx_regression


def test_net_liquidity_features_are_built():
    fed, tga, rrp, btc, _ = synthetic_macro_and_prices(periods_weeks=160)
    feats = build_net_liquidity(fed, tga, rrp, fed_assets_scale=0.001, tga_scale=0.001, onrrp_scale=1.0)
    assert 'net_liquidity_bil' in feats.columns
    assert 'liq_impulse_z_52' in feats.columns
    assert len(feats) > 100


def test_forward_returns_and_stats():
    fed, tga, rrp, btc, _ = synthetic_macro_and_prices(periods_weeks=180)
    feats = build_net_liquidity(fed, tga, rrp, fed_assets_scale=0.001, tga_scale=0.001, onrrp_scale=1.0)
    analysis = attach_forward_returns_to_features(feats, price_to_weekly(btc), [1, 2, 4])
    assert 'fwd_logret_4w' in analysis.columns
    corr = cross_corr_table(analysis, 'liq_impulse_z_52', ['fwd_logret_1w', 'fwd_logret_4w'])
    assert len(corr) == 2
    q = quintile_spread(analysis, 'liq_impulse_z_52', 'fwd_logret_4w')
    assert q['n'] > 30
    r = arx_regression(analysis, 'liq_impulse_z_52', 'fwd_logret_4w')
    assert r['n'] > 30
