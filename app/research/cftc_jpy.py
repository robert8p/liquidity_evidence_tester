from __future__ import annotations

from datetime import date
from pathlib import Path
import json
import numpy as np
import pandas as pd

from app.adapters.cftc import fetch_tff_years, extract_jpy_tff_features
from app.adapters.fred import fetch_fred_csv
from app.config import Settings
from app.release_calendar import attach_cftc_alignment
from app.research.features import rolling_zscore, price_to_weekly, attach_forward_returns_to_features
from app.research.stats import cross_corr_table, quintile_spread, arx_regression, expanding_directional_oos, summary_from_oos, validation_label, screen_feature_grid
from app.research.reporting import write_csv
from app.utils import run_id, utc_now_iso, write_json, zip_dir

JPY_SCREEN_FEATURES = [
    'lev_net_oi_z_156',
    'contrarian_usdjpy_z_156',
    'trend_usdjpy_z_156',
    'lev_net_oi_change_1w_z_52',
    'lev_net_oi_change_4w_z_52',
    'short_crowding_z_156',
    'long_crowding_z_156',
    'crowding_abs_z_156',
    'extreme_reversal_signal',
]


def _analysis_window_bounds(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f'{start_date}T00:00:00Z'), pd.Timestamp(f'{end_date}T23:59:59Z')


def _restrict_aligned_to_analysis_window(features_aligned: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if 'effective_trade_at_utc' not in features_aligned.columns:
        raise ValueError('features_aligned must include effective_trade_at_utc before window restriction')
    start_ts, end_ts = _analysis_window_bounds(start_date, end_date)
    out = features_aligned.copy()
    out['effective_trade_at_utc'] = pd.to_datetime(out['effective_trade_at_utc'], utc=True)
    return out[(out['effective_trade_at_utc'] >= start_ts) & (out['effective_trade_at_utc'] <= end_ts)].copy()


def build_jpy_positioning_features(jpy_tff: pd.DataFrame) -> pd.DataFrame:
    """Build simple, pre-declared JPY positioning features.

    The target is USD/JPY. Positive `contrarian_usdjpy_z_156` means leveraged funds are
    unusually long JPY futures; if crowding mean-reverts, USD/JPY would tend to rise.
    Positive `trend_usdjpy_z_156` means leveraged funds are unusually short JPY futures;
    if trend-following pressure persists, USD/JPY would tend to rise.
    """
    df = jpy_tff.copy().sort_index()
    for c in ['lev_net_oi', 'lev_short_oi', 'lev_long_oi']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['lev_net_oi_z_156'] = rolling_zscore(df['lev_net_oi'], 156, min_periods=52)
    df['contrarian_usdjpy_z_156'] = df['lev_net_oi_z_156']
    df['trend_usdjpy_z_156'] = -df['lev_net_oi_z_156']
    df['lev_net_oi_change_1w'] = df['lev_net_oi'].diff()
    df['lev_net_oi_change_4w'] = df['lev_net_oi'].diff(4)
    df['lev_net_oi_change_1w_z_52'] = rolling_zscore(df['lev_net_oi_change_1w'], 52)
    df['lev_net_oi_change_4w_z_52'] = rolling_zscore(df['lev_net_oi_change_4w'], 52)
    df['short_crowding_z_156'] = rolling_zscore(df['lev_short_oi'], 156, min_periods=52)
    df['long_crowding_z_156'] = rolling_zscore(df['lev_long_oi'], 156, min_periods=52)
    df['crowding_abs_z_156'] = rolling_zscore(df['lev_net_oi'].abs(), 156, min_periods=52)
    # Directional, deliberately simple extreme rule: fade the crowded JPY futures side.
    z = df['lev_net_oi_z_156']
    df['extreme_reversal_signal'] = np.select([z >= 1.0, z <= -1.0], [1.0, -1.0], default=0.0)
    return df.dropna(subset=['lev_net_oi']).sort_index()


def _fred_usdjpy_to_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={'value': 'close'}).copy()
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    return out.dropna(subset=['close']).sort_index()[['close']]


def _synthetic_jpy_and_usdjpy(start: str = '2012-01-01', end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = end or date.today().isoformat()
    idx = pd.date_range(start=start, end=end, freq='W-TUE', tz='UTC')
    rng = np.random.default_rng(31415)
    cycle = np.sin(np.arange(len(idx)) / 18.0)
    net = 0.20 * cycle + rng.normal(0, 0.035, len(idx))
    oi = 180_000 + 20_000 * np.cos(np.arange(len(idx)) / 30.0)
    long = (0.28 + np.maximum(net, 0)) * oi
    short = (0.28 + np.maximum(-net, 0)) * oi
    jpy = pd.DataFrame({'market': 'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE', 'open_interest': oi, 'long': long, 'short': short}, index=idx)
    jpy['lev_net_contracts'] = jpy['long'] - jpy['short']
    jpy['lev_net_oi'] = jpy['lev_net_contracts'] / jpy['open_interest']
    jpy['lev_short_oi'] = jpy['short'] / jpy['open_interest']
    jpy['lev_long_oi'] = jpy['long'] / jpy['open_interest']
    days = pd.date_range(start=start, end=end, freq='B', tz='UTC')
    # Build a synthetic USDJPY process with weak dependency on prior positioning crowding.
    weekly_signal = pd.Series(net, index=idx).reindex(days, method='ffill').fillna(0.0)
    daily_ret = 0.0001 + 0.0006 * weekly_signal.values + rng.normal(0, 0.006, len(days))
    close = 90.0 * np.exp(np.cumsum(daily_ret))
    usdjpy = pd.DataFrame({'close': close}, index=days)
    return jpy, usdjpy


def _target_analysis(features_aligned: pd.DataFrame, price_df: pd.DataFrame, horizons: list[int], run_dir: Path, *, screen_features: bool = True) -> dict:
    weekly_price = price_to_weekly(price_df)
    analysis = attach_forward_returns_to_features(features_aligned, weekly_price, horizons, max_lookahead_days=14)
    target_cols = [f'fwd_logret_{h}w' for h in horizons]
    matched_target_rows = int(analysis['target_anchor_utc'].notna().sum()) if 'target_anchor_utc' in analysis.columns else int(analysis[target_cols].notna().any(axis=1).sum())
    analysis_window_rows = int(len(features_aligned))
    target_coverage_ratio = float(matched_target_rows / analysis_window_rows) if analysis_window_rows else None
    first_matched_target_anchor = None
    if 'target_anchor_utc' in analysis.columns and matched_target_rows:
        first_matched_target_anchor = str(analysis['target_anchor_utc'].dropna().min())

    primary_feature = 'contrarian_usdjpy_z_156'
    corr = cross_corr_table(analysis, primary_feature, target_cols)
    qrows = [quintile_spread(analysis, primary_feature, c) for c in target_cols]
    rrows = [arx_regression(analysis, primary_feature, c) for c in target_cols]
    oos_target = f'fwd_logret_{horizons[min(2, len(horizons)-1)]}w'
    oos = expanding_directional_oos(analysis, primary_feature, oos_target)

    write_csv(run_dir / 'USDJPY_analysis_rows.csv', analysis)
    write_csv(run_dir / 'USDJPY_cross_corr.csv', corr)
    pd.DataFrame(qrows).to_csv(run_dir / 'USDJPY_quintile_spreads.csv', index=False)
    pd.DataFrame(rrows).to_csv(run_dir / 'USDJPY_arx_regressions.csv', index=False)
    oos.to_csv(run_dir / 'USDJPY_oos_predictions.csv', index=False)

    screen_summary = {}
    if screen_features:
        screen = screen_feature_grid(analysis, JPY_SCREEN_FEATURES, target_cols)
        screen.to_csv(run_dir / 'USDJPY_signal_variant_screen.csv', index=False)
        usable = screen[screen.get('screen_status', pd.Series(dtype=str)) != 'insufficient_data'] if not screen.empty else screen
        if not usable.empty:
            top = usable.sort_values('screen_score', ascending=False).head(8)
            screen_summary = {
                'candidate_count': int(len(usable)),
                'validated_candidate_count': int((usable['screen_status'] == 'validated_candidate').sum()),
                'top_candidates': top.to_dict(orient='records'),
                'note': 'JPY positioning screen is discovery evidence only; promote nothing without separate walk-forward/live-shadow confirmation.',
            }
        else:
            screen_summary = {'candidate_count': 0, 'validated_candidate_count': 0, 'top_candidates': [], 'note': 'No JPY variants had enough data.'}

    best_q = max([x for x in qrows if x.get('spread') is not None], key=lambda x: abs(x['spread']), default={})
    best_r = min([x for x in rrows if x.get('p') is not None], key=lambda x: x['p'], default={})
    metrics = {
        'rows': int(len(analysis.dropna(subset=[primary_feature]))),
        'analysis_window_rows': analysis_window_rows,
        'matched_target_rows': matched_target_rows,
        'target_coverage_ratio': target_coverage_ratio,
        'first_matched_target_anchor_utc': first_matched_target_anchor,
        'primary_feature': primary_feature,
        'best_quintile_target': best_q.get('target'),
        'best_quintile_spread': best_q.get('spread'),
        'best_regression_target': best_r.get('target'),
        'best_regression_coef': best_r.get('coef'),
        'best_regression_t': best_r.get('t'),
        'best_regression_p': best_r.get('p'),
        'oos': summary_from_oos(oos),
        'signal_variant_screen': screen_summary,
    }
    metrics['validation'] = validation_label(metrics, coverage_ratio=target_coverage_ratio)
    return metrics


def _write_jpy_report(run_dir: Path, metrics: dict, warnings: list[str]) -> Path:
    lines = [
        '# Evidence Tester Report',
        '',
        f"Run ID: `{metrics.get('run_id', 'unknown')}`",
        '',
        '## Purpose',
        '',
        'This pack is evidence for a research hypothesis. It is not a trading signal, order-routing system, or financial advice.',
        '',
        '## Hypothesis',
        '',
        'CFTC Traders in Financial Futures leveraged-fund Japanese-yen positioning is tested as a leading variable for USD/JPY returns. The report date is treated as the Tuesday as-of date, public release as Friday 15:30 ET, and the effective tradable timestamp as the next conservative FX session.',
        '',
        '## Validation summary',
        '',
        '| Target | Status | Main reasons |',
        '|---|---|---|',
    ]
    for target, m in metrics.get('targets', {}).items():
        validation = m.get('validation') or {}
        reasons = validation.get('reasons') or []
        reason_text = '<br>'.join(reasons) if reasons else 'Passed conservative evidence gate.'
        lines.append(f"| {target} | {validation.get('status', 'unknown')} | {reason_text} |")
    lines.extend(['', '## Headline metrics', ''])
    for target, m in metrics.get('targets', {}).items():
        lines.append(f'### {target}')
        lines.append('')
        lines.append('| Metric | Value |')
        lines.append('|---|---:|')
        for k, v in m.items():
            if isinstance(v, float):
                value = f'{v:.6f}'
            elif isinstance(v, dict):
                value = '`' + json.dumps(v, sort_keys=True) + '`'
            else:
                value = str(v)
            lines.append(f'| {k} | {value} |')
        lines.append('')
    lines.extend(['## Signal variant screen', ''])
    lines.append('This section tests pre-declared variants of the JPY positioning signal. It is discovery evidence only; it should not be treated as a trading model-selection result.')
    lines.append('')
    for target, m in metrics.get('targets', {}).items():
        screen = m.get('signal_variant_screen') or {}
        lines.append(f"### {target} signal variants")
        lines.append('')
        lines.append(f"- Candidates screened: {screen.get('candidate_count', 0)}")
        lines.append(f"- Validated candidates under conservative gate: {screen.get('validated_candidate_count', 0)}")
        top = screen.get('top_candidates') or []
        if top:
            lines.append('')
            lines.append('| Rank | Feature | Horizon | Status | Spread | Regression p | OOS excess vs always-long | Long fraction |')
            lines.append('|---:|---|---|---|---:|---:|---:|---:|')
            for i, row in enumerate(top[:8], start=1):
                def fmt(x):
                    return f'{x:.6f}' if isinstance(x, float) else str(x)
                lines.append(
                    f"| {i} | {row.get('feature')} | {row.get('target')} | {row.get('screen_status')} | "
                    f"{fmt(row.get('quintile_spread'))} | {fmt(row.get('regression_p'))} | "
                    f"{fmt(row.get('oos_excess_mean_return_vs_always_long'))} | {fmt(row.get('signal_long_fraction'))} |"
                )
        lines.append('')
    if warnings:
        lines.extend(['## Warnings', ''])
        for w in warnings:
            lines.append(f'- {w}')
        lines.append('')
    lines.extend([
        '## Interpretation discipline',
        '',
        '- Treat positive results as candidates for deeper validation, not permission to trade.',
        '- Reject results that rely on one short sub-period, one horizon, or unaligned release timing.',
        '- Reject OOS results that do not beat simple baselines such as always-long or always-short.',
        '- Promote a hypothesis only after out-of-sample and live-shadow evidence agree with the historical result.',
    ])
    path = run_dir / 'report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def run_cftc_jpy(settings: Settings, *, start_date: str, end_date: str | None, demo_mode: bool, horizons_weeks: list[int], screen_features: bool = True) -> dict:
    rid = run_id('cftcjpy')
    run_dir = settings.data_dir / 'runs' / rid
    raw_dir = settings.data_dir / 'raw' / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    end_date = end_date or date.today().isoformat()

    if demo_mode:
        jpy_tff, usdjpy = _synthetic_jpy_and_usdjpy(start=start_date, end=end_date)
        warnings.append('Demo mode used synthetic CFTC and USDJPY data. Do not use this evidence for decisions.')
    else:
        start_year = max(2010, pd.Timestamp(start_date).year - 1)
        end_year = pd.Timestamp(end_date).year
        raw_cftc = fetch_tff_years(start_year, end_year, raw_dir=raw_dir)
        if raw_cftc.attrs.get('fetch_errors'):
            warnings.append('Some annual CFTC files were not fetched: ' + '; '.join(raw_cftc.attrs.get('fetch_errors')[:5]))
        jpy_tff = extract_jpy_tff_features(raw_cftc)
        usdjpy = _fred_usdjpy_to_price(fetch_fred_csv(settings.fred_usdjpy_series, raw_dir=raw_dir))

    feature_base_all = build_jpy_positioning_features(jpy_tff)
    write_csv(run_dir / 'jpy_positioning_features_full_context_unaligned.csv', feature_base_all)

    aligned_all = attach_cftc_alignment(feature_base_all, instrument_type='fx')
    aligned_window = _restrict_aligned_to_analysis_window(aligned_all, start_date, end_date).set_index('effective_trade_at_utc')
    write_csv(run_dir / 'USDJPY_jpy_positioning_features_analysis_window.csv', aligned_window)

    metrics = {
        'run_id': rid,
        'hypothesis': 'cftc_jpy_positioning_to_usdjpy',
        'created_at_utc': utc_now_iso(),
        'demo_mode': demo_mode,
        'start_date': start_date,
        'end_date': end_date,
        'feature_rows_full_context': int(len(feature_base_all)),
        'analysis_window_rows': int(len(aligned_window)),
        'screen_features': bool(screen_features),
        'screen_feature_names': JPY_SCREEN_FEATURES if screen_features else [],
        'targets': {},
        'warnings': warnings,
    }

    if not feature_base_all.empty and feature_base_all.index.min() < pd.Timestamp(f'{start_date}T00:00:00Z'):
        warnings.append('Pre-start CFTC history was used only for rolling feature context; analysed rows were restricted to the requested release-aligned window.')

    if aligned_window.empty:
        warnings.append('No release-aligned JPY positioning rows inside the requested analysis window.')
    elif usdjpy.empty:
        warnings.append('USDJPY target prices were empty.')
    else:
        m = _target_analysis(aligned_window, usdjpy, horizons_weeks, run_dir, screen_features=screen_features)
        m['analysis_window_rows'] = int(len(aligned_window))
        m['target_price_start_utc'] = str(usdjpy.index.min())
        m['target_price_end_utc'] = str(usdjpy.index.max())
        if m.get('matched_target_rows', 0) < len(aligned_window):
            warnings.append(f"USDJPY target coverage warning: only {m.get('matched_target_rows', 0)} of {len(aligned_window)} release-aligned rows matched target history within the weekly tolerance. Earlier rows were left as NaN instead of being backfilled.")
        metrics['targets']['USDJPY'] = m

    if not metrics['targets']:
        warnings.append('No target analyses were completed. Run demo_mode=true to validate the app or check CFTC/FRED data availability.')

    write_json(run_dir / 'metrics.json', metrics)
    _write_jpy_report(run_dir, metrics, warnings)
    pack = zip_dir(run_dir, settings.data_dir / 'packs' / f'{rid}.zip')
    summary = {
        'run_id': rid,
        'status': 'completed',
        'hypothesis': metrics['hypothesis'],
        'created_at_utc': metrics['created_at_utc'],
        'pack': str(pack),
        'metrics': metrics,
        'warnings': warnings,
    }
    write_json(settings.data_dir / 'runs' / 'latest.json', summary)
    return summary
