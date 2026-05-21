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
    df['liq_impulse_13w_bil'] = df['net_liquidity_bil'].diff(13)

    # Primary and challenger signal definitions. These are deliberately simple,
    # interpretable liquidity constructions; the screen is for falsification, not
    # curve-fitting or model selection for live trading.
    df['liq_impulse_z_52'] = rolling_zscore(df['liq_impulse_1w_bil'], 52)
    df['liq_impulse_4w_z_52'] = rolling_zscore(df['liq_impulse_4w_bil'], 52)
    df['liq_impulse_13w_z_104'] = rolling_zscore(df['liq_impulse_13w_bil'], 104, min_periods=52)
    df['net_liquidity_level_z_156'] = rolling_zscore(df['net_liquidity_bil'], 156, min_periods=52)

    df['tga_drain_z_52'] = rolling_zscore(-df['tga_bil'].diff(), 52)
    df['tga_drain_4w_z_52'] = rolling_zscore(-df['tga_bil'].diff(4), 52)
    df['rrp_release_z_52'] = rolling_zscore(-df['onrrp_bil'].diff(), 52)
    df['rrp_release_4w_z_52'] = rolling_zscore(-df['onrrp_bil'].diff(4), 52)

    df['liquidity_composite_z'] = df[['liq_impulse_z_52', 'tga_drain_z_52', 'rrp_release_z_52']].mean(axis=1)
    df['liquidity_composite_4w_z'] = df[['liq_impulse_4w_z_52', 'tga_drain_4w_z_52', 'rrp_release_4w_z_52']].mean(axis=1)
    return df.dropna(subset=['liq_impulse_1w_bil']).sort_index()


def price_to_weekly(price_df: pd.DataFrame) -> pd.Series:
    if 'close' not in price_df.columns:
        raise ValueError('price_df must contain a close column')
    return price_df['close'].resample('W-FRI').last().dropna()



def price_to_daily(price_df: pd.DataFrame) -> pd.Series:
    """Return one clean daily close series for event-aligned tests.

    Unlike price_to_weekly(), this preserves the first available business-day
    price after a tradable timestamp. This is important for CFTC FX tests: the
    public report is released Friday 15:30 ET, but the conservative first
    evidence anchor should be the next available USD/JPY daily print after the
    Monday FX tradable timestamp, not the following Friday weekly close.
    """
    if 'close' not in price_df.columns:
        raise ValueError('price_df must contain a close column')
    s = price_df['close'].copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError('price_df index must be a DatetimeIndex')
    if s.index.tz is None:
        s.index = s.index.tz_localize('UTC')
    else:
        s.index = s.index.tz_convert('UTC')
    return s.resample('B').last().dropna()


def forward_log_returns_by_business_days(price: pd.Series, horizons_weeks: list[int]) -> pd.DataFrame:
    """Forward log returns using approximately 5 business days per week."""
    out = pd.DataFrame(index=price.index)
    out['target_anchor_utc'] = price.index
    for h in horizons_weeks:
        periods = int(h) * 5
        out[f'fwd_logret_{h}w'] = np.log(price.shift(-periods) / price)
    return out


def attach_daily_forward_returns_to_features(
    features: pd.DataFrame,
    daily_price: pd.Series,
    horizons_weeks: list[int],
    *,
    max_lookahead_days: int = 7,
) -> pd.DataFrame:
    """Attach forward returns from the first available daily target after signal time.

    This is stricter than weekly anchoring for FX/economic-release tests because it
    avoids silently skipping several tradable days. The tolerance prevents early
    feature rows from being matched to a much later first available price history.
    """
    returns = forward_log_returns_by_business_days(daily_price, horizons_weeks)
    left = features.copy().sort_index()
    right = returns.copy().sort_index()
    tolerance = pd.Timedelta(days=max_lookahead_days) if max_lookahead_days is not None else None
    aligned = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction='forward',
        tolerance=tolerance,
    )
    if 'target_anchor_utc' in aligned.columns:
        aligned['target_anchor_utc'] = pd.to_datetime(aligned['target_anchor_utc'], utc=True)
    return aligned

def forward_log_returns(price: pd.Series, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=price.index)
    out['target_anchor_utc'] = price.index
    for h in horizons:
        out[f'fwd_logret_{h}w'] = np.log(price.shift(-h) / price)
    return out


def attach_forward_returns_to_features(
    features: pd.DataFrame,
    weekly_price: pd.Series,
    horizons: list[int],
    *,
    max_lookahead_days: int = 10,
) -> pd.DataFrame:
    """Attach forward returns to release-aligned features without backfilling stale target history.

    The target return row must be the first available weekly target anchor after the
    feature's tradable timestamp, but only if that anchor is close enough to be the
    same/next weekly observation. Without a tolerance, pandas.merge_asof(direction='forward')
    will attach the first target row to every earlier feature row if target price history
    begins later than macro history. That was invalid for a real QQQ run where the equity
    data vendor only returned prices from 2021 while the requested analysis began in 2018.
    """
    returns = forward_log_returns(weekly_price, horizons)
    left = features.copy().sort_index()
    right = returns.copy().sort_index()
    tolerance = pd.Timedelta(days=max_lookahead_days) if max_lookahead_days is not None else None
    aligned = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction='forward',
        tolerance=tolerance,
    )
    if 'target_anchor_utc' in aligned.columns:
        aligned['target_anchor_utc'] = pd.to_datetime(aligned['target_anchor_utc'], utc=True)
    return aligned
