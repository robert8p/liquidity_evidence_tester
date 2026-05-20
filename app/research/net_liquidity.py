from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from app.adapters.fred import fetch_fred_csv
from app.adapters.coinapi import fetch_ohlcv
from app.adapters.massive import fetch_daily_bars as fetch_massive_daily
from app.adapters.alpaca import fetch_daily_bars as fetch_alpaca_daily
from app.config import Settings
from app.release_calendar import attach_h41_alignment
from app.research.demo import synthetic_macro_and_prices
from app.research.features import build_net_liquidity, price_to_weekly, attach_forward_returns_to_features
from app.research.stats import cross_corr_table, quintile_spread, arx_regression, expanding_directional_oos, summary_from_oos
from app.research.reporting import write_csv, write_markdown_report
from app.utils import run_id, utc_now_iso, write_json, zip_dir


def _fetch_equity(settings: Settings, symbol: str, start_date: str, end_date: str, raw_dir: Path, warnings: list[str]) -> pd.DataFrame | None:
    if settings.equity_source.lower() == 'alpaca':
        if not (settings.alpaca_key_id and settings.alpaca_secret_key):
            warnings.append('Alpaca credentials missing; equity target skipped.')
            return None
        return fetch_alpaca_daily(symbol, settings.alpaca_key_id, settings.alpaca_secret_key, f'{start_date}T00:00:00Z', f'{end_date}T23:59:59Z', raw_dir=raw_dir)
    if not settings.massive_api_key:
        warnings.append('Massive API key missing; equity target skipped.')
        return None
    return fetch_massive_daily(symbol, settings.massive_api_key, start_date, end_date, raw_dir=raw_dir)


def _fetch_btc(settings: Settings, start_date: str, end_date: str, raw_dir: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not settings.coinapi_key:
        warnings.append('CoinAPI key missing; BTC target skipped.')
        return None
    return fetch_ohlcv(settings.coinapi_btc_symbol_id, settings.coinapi_key, start_iso=f'{start_date}T00:00:00Z', end_iso=f'{end_date}T23:59:59Z', raw_dir=raw_dir)




def _analysis_window_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(f'{start_date}T00:00:00Z')
    end_ts = pd.Timestamp(f'{end_date}T23:59:59Z')
    return start_ts, end_ts


def _restrict_aligned_to_analysis_window(features_aligned: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Keep only rows whose tradable effective timestamp is inside the requested run window.

    v0.1.2 computed features using the full available FRED history, then analysed every
    pre-start macro row. When target prices began at the requested start date, merge_asof
    attached the first available target return to many old macro observations. That created
    repeated artificial forward returns and invalid OOS/quintile metrics. This helper keeps
    the useful pre-start macro history only for rolling feature context, while restricting
    analysed observations to the operator-requested tradable window.
    """
    if 'effective_trade_at_utc' not in features_aligned.columns:
        raise ValueError('features_aligned must include effective_trade_at_utc before window restriction')
    start_ts, end_ts = _analysis_window_bounds(start_date, end_date)
    out = features_aligned.copy()
    out['effective_trade_at_utc'] = pd.to_datetime(out['effective_trade_at_utc'], utc=True)
    return out[(out['effective_trade_at_utc'] >= start_ts) & (out['effective_trade_at_utc'] <= end_ts)].copy()


def _target_analysis(features_aligned: pd.DataFrame, price_df: pd.DataFrame, target_name: str, horizons: list[int], run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    weekly_price = price_to_weekly(price_df)
    analysis = attach_forward_returns_to_features(features_aligned, weekly_price, horizons)
    target_cols = [f'fwd_logret_{h}w' for h in horizons]
    matched_target_rows = int(analysis['target_anchor_utc'].notna().sum()) if 'target_anchor_utc' in analysis.columns else int(analysis[target_cols].notna().any(axis=1).sum())
    first_matched_target_anchor = None
    if 'target_anchor_utc' in analysis.columns and matched_target_rows:
        first_matched_target_anchor = str(analysis['target_anchor_utc'].dropna().min())
    corr = cross_corr_table(analysis, 'liq_impulse_z_52', target_cols)
    qrows = [quintile_spread(analysis, 'liq_impulse_z_52', c) for c in target_cols]
    rrows = [arx_regression(analysis, 'liq_impulse_z_52', c) for c in target_cols]
    oos = expanding_directional_oos(analysis, 'liq_impulse_z_52', f'fwd_logret_{horizons[min(2, len(horizons)-1)]}w')
    write_csv(run_dir / f'{target_name}_analysis_rows.csv', analysis)
    write_csv(run_dir / f'{target_name}_cross_corr.csv', corr)
    pd.DataFrame(qrows).to_csv(run_dir / f'{target_name}_quintile_spreads.csv', index=False)
    pd.DataFrame(rrows).to_csv(run_dir / f'{target_name}_arx_regressions.csv', index=False)
    oos.to_csv(run_dir / f'{target_name}_oos_predictions.csv', index=False)
    best_q = max([x for x in qrows if x.get('spread') is not None], key=lambda x: abs(x['spread']), default={})
    best_r = min([x for x in rrows if x.get('p') is not None], key=lambda x: x['p'], default={})
    metrics = {
        'rows': int(len(analysis.dropna(subset=['liq_impulse_z_52']))),
        'matched_target_rows': matched_target_rows,
        'first_matched_target_anchor_utc': first_matched_target_anchor,
        'best_quintile_target': best_q.get('target'),
        'best_quintile_spread': best_q.get('spread'),
        'best_regression_target': best_r.get('target'),
        'best_regression_coef': best_r.get('coef'),
        'best_regression_t': best_r.get('t'),
        'best_regression_p': best_r.get('p'),
        'oos': summary_from_oos(oos),
    }
    return metrics, analysis, corr


def run_net_liquidity(settings: Settings, *, start_date: str, end_date: str | None, target_symbol: str, include_btc: bool, include_equity: bool, demo_mode: bool, horizons_weeks: list[int]) -> dict:
    rid = run_id('netliq')
    run_dir = settings.data_dir / 'runs' / rid
    raw_dir = settings.data_dir / 'raw' / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    end_date = end_date or date.today().isoformat()

    if demo_mode:
        fed, tga, onrrp, btc, equity = synthetic_macro_and_prices(start=start_date)
        warnings.append('Demo mode used synthetic macro and price data. Do not use this evidence for decisions.')
    else:
        fed = fetch_fred_csv(settings.fred_fed_assets_series, raw_dir=raw_dir)
        tga = fetch_fred_csv(settings.fred_tga_series, raw_dir=raw_dir)
        onrrp = fetch_fred_csv(settings.fred_onrrp_series, raw_dir=raw_dir)
        btc = _fetch_btc(settings, start_date, end_date, raw_dir, warnings) if include_btc else None
        equity = _fetch_equity(settings, target_symbol, start_date, end_date, raw_dir, warnings) if include_equity else None

    feature_base_all = build_net_liquidity(
        fed, tga, onrrp,
        fed_assets_scale=settings.fred_fed_assets_scale_to_billions,
        tga_scale=settings.fred_tga_scale_to_billions,
        onrrp_scale=settings.fred_onrrp_scale_to_billions,
    )
    # Keep the full context file for auditability and rolling z-score reconstruction.
    write_csv(run_dir / 'net_liquidity_features_full_context_unaligned.csv', feature_base_all)

    metrics = {
        'run_id': rid,
        'hypothesis': 'net_liquidity_to_btc_and_nasdaq',
        'created_at_utc': utc_now_iso(),
        'demo_mode': demo_mode,
        'start_date': start_date,
        'end_date': end_date,
        'feature_rows_full_context': int(len(feature_base_all)),
        'targets': {},
        'warnings': warnings,
    }

    if not feature_base_all.empty and feature_base_all.index.min() < pd.Timestamp(f'{start_date}T00:00:00Z'):
        warnings.append('Pre-start macro history was used only for rolling feature context; analysed rows were restricted to the requested release-aligned window.')

    if include_btc and btc is not None and not btc.empty:
        aligned_crypto_all = attach_h41_alignment(feature_base_all, instrument_type='crypto')
        aligned_crypto_window = _restrict_aligned_to_analysis_window(aligned_crypto_all, start_date, end_date).set_index('effective_trade_at_utc')
        write_csv(run_dir / 'BTCUSD_net_liquidity_features_analysis_window.csv', aligned_crypto_window)
        if not aligned_crypto_window.empty:
            m, _, _ = _target_analysis(aligned_crypto_window, btc, 'BTCUSD', horizons_weeks, run_dir)
            m['analysis_window_rows'] = int(len(aligned_crypto_window))
            m['target_price_start_utc'] = str(btc.index.min())
            m['target_price_end_utc'] = str(btc.index.max())
            if m.get('matched_target_rows', 0) < len(aligned_crypto_window):
                warnings.append(f"BTCUSD target coverage warning: only {m.get('matched_target_rows', 0)} of {len(aligned_crypto_window)} release-aligned rows matched target history within the weekly tolerance. Earlier rows were left as NaN instead of being backfilled.")
            metrics['targets']['BTCUSD'] = m
        else:
            warnings.append('BTC target skipped: no release-aligned liquidity rows inside the requested analysis window.')

    if include_equity and equity is not None and not equity.empty:
        aligned_equity_all = attach_h41_alignment(feature_base_all, instrument_type='equity')
        aligned_equity_window = _restrict_aligned_to_analysis_window(aligned_equity_all, start_date, end_date).set_index('effective_trade_at_utc')
        write_csv(run_dir / f'{target_symbol}_net_liquidity_features_analysis_window.csv', aligned_equity_window)
        if not aligned_equity_window.empty:
            m, _, _ = _target_analysis(aligned_equity_window, equity, target_symbol, horizons_weeks, run_dir)
            m['analysis_window_rows'] = int(len(aligned_equity_window))
            m['target_price_start_utc'] = str(equity.index.min())
            m['target_price_end_utc'] = str(equity.index.max())
            if m.get('matched_target_rows', 0) < len(aligned_equity_window):
                warnings.append(f"{target_symbol} target coverage warning: only {m.get('matched_target_rows', 0)} of {len(aligned_equity_window)} release-aligned rows matched target history within the weekly tolerance. Earlier rows were left as NaN instead of being backfilled.")
            metrics['targets'][target_symbol] = m
        else:
            warnings.append(f'{target_symbol} target skipped: no release-aligned liquidity rows inside the requested analysis window.')

    if not metrics['targets']:
        warnings.append('No target analyses were completed. Add CoinAPI/Massive/Alpaca credentials or run demo_mode=true.')

    write_json(run_dir / 'metrics.json', metrics)
    write_markdown_report(run_dir, metrics, warnings)
    pack = zip_dir(run_dir, settings.data_dir / 'packs' / f'{rid}.zip')
    write_json(settings.data_dir / 'runs' / 'latest.json', {'run_id': rid, 'pack': str(pack), 'created_at_utc': metrics['created_at_utc'], 'metrics': metrics})
    return {'run_id': rid, 'run_dir': str(run_dir), 'pack': str(pack), 'metrics': metrics, 'warnings': warnings}
