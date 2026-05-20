import pandas as pd

from app.research.demo import synthetic_macro_and_prices
from app.research.features import build_net_liquidity, attach_forward_returns_to_features, price_to_weekly
from app.research.stats import screen_feature_grid


def test_signal_variant_screen_runs_for_predeclared_features():
    fed, tga, rrp, btc, _ = synthetic_macro_and_prices(periods_weeks=220)
    feats = build_net_liquidity(fed, tga, rrp, fed_assets_scale=0.001, tga_scale=0.001, onrrp_scale=1.0)
    analysis = attach_forward_returns_to_features(feats, price_to_weekly(btc), [1, 2, 4])
    screen = screen_feature_grid(
        analysis,
        ['liq_impulse_z_52', 'liq_impulse_4w_z_52', 'liquidity_composite_z'],
        ['fwd_logret_1w', 'fwd_logret_4w'],
        min_n=50,
    )
    assert not screen.empty
    assert {'feature', 'target', 'screen_status', 'screen_score'}.issubset(set(screen.columns))
    assert set(screen['feature']).issubset({'liq_impulse_z_52', 'liq_impulse_4w_z_52', 'liquidity_composite_z'})
