from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(5, window // 3)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def weekly_last(df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    return df[[value_col]].resample('W-WED').last().dropna()


def build_net_liquidity(
    fed_assets: pd.DataFrame,
    tga: pd.DataFrame,
    onrrp: pd.DataFrame,
    *,
    fed_assets_scale: float,
    tga_scale: float,
    onrrp_scale: float,
) -> pd.DataFrame:
    fa = weekly_last(fed_assets).rename(columns={'value': 'fed_assets_bil'}) * fed_assets_scale
    tg = weekly_last(tga).rename(columns={'value': 'tga_bil'}) * tga_scale
    rr = weekly_last(onrrp).rename(columns={'value': 'onrrp_bil'}) * onrrp_scale
    df = fa.join(tg, how='inner').join(rr, how='inner')
    df['net_liquidity_bil'] = df['fed_assets_bil'] - df['tga_bil'] - df['onrrp_bil']
    df['liq_impulse_1w_bil'] = df['net_liquidity_bil'].diff()
    df['liq_impulse_4w_bil'] = df['net_liquidity_bil'].diff(4)
    df['liq_impulse_z_52'] = rolling_zscore(df['liq_impulse_1w_bil'], 52)
    df['tga_drain_z_52'] = rolling_zscore(-df['tga_bil'].diff(), 52)
    df['rrp_release_z_52'] = rolling_zscore(-df['onrrp_bil'].diff(), 52)
    return df.dropna(subset=['liq_impulse_1w_bil']).sort_index()


def price_to_weekly(price_df: pd.DataFrame) -> pd.Series:
    if 'close' not in price_df.columns:
        raise ValueError('price_df must contain a close column')
    return price_df['close'].resample('W-FRI').last().dropna()


def forward_log_returns(price: pd.Series, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=price.index)
    for h in horizons:
        out[f'fwd_logret_{h}w'] = np.log(price.shift(-h) / price)
    return out


def attach_forward_returns_to_features(features: pd.DataFrame, weekly_price: pd.Series, horizons: list[int]) -> pd.DataFrame:
    returns = forward_log_returns(weekly_price, horizons)
    # As-of joins: use next available weekly target price after feature effective date.
    left = features.copy().sort_index()
    right = returns.copy().sort_index()
    aligned = pd.merge_asof(left, right, left_index=True, right_index=True, direction='forward')
    return aligned
