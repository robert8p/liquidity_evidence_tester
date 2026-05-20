from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_macro_and_prices(start: str = '2018-01-01', periods_weeks: int = 360, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods_weeks, freq='W-WED', tz='UTC')
    fed = 4200 + np.cumsum(rng.normal(3, 18, periods_weeks))
    tga = 500 + np.cumsum(rng.normal(0, 25, periods_weeks))
    rrp = 700 + np.cumsum(rng.normal(-1, 30, periods_weeks))
    fed_df = pd.DataFrame({'value': fed * 1000}, index=idx)  # millions
    tga_df = pd.DataFrame({'value': tga * 1000}, index=idx)  # millions
    rrp_df = pd.DataFrame({'value': rrp}, index=idx)  # billions

    net = fed - tga - rrp
    impulse = pd.Series(net, index=idx).diff().fillna(0)
    # Create noisy targets with a weak positive relationship to prior liquidity impulse.
    btc_ret = 0.002 + 0.0009 * impulse.shift(1).fillna(0).to_numpy() + rng.normal(0, 0.07, periods_weeks)
    qqq_ret = 0.001 + 0.00025 * impulse.shift(1).fillna(0).to_numpy() + rng.normal(0, 0.025, periods_weeks)
    btc_price = 12000 * np.exp(np.cumsum(btc_ret))
    qqq_price = 220 * np.exp(np.cumsum(qqq_ret))
    price_idx = pd.date_range(start=start, periods=periods_weeks, freq='W-FRI', tz='UTC')
    btc = pd.DataFrame({'close': btc_price}, index=price_idx)
    qqq = pd.DataFrame({'close': qqq_price}, index=price_idx)
    return fed_df, tga_df, rrp_df, btc, qqq
