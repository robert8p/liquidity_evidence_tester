from __future__ import annotations

import math
import numpy as np
import pandas as pd


def cross_corr_table(df: pd.DataFrame, feature_col: str, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for t in target_cols:
        x = df[feature_col]
        y = df[t]
        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < 20:
            corr = np.nan
        else:
            corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        rows.append({'feature': feature_col, 'target': t, 'n': len(valid), 'corr': corr})
    return pd.DataFrame(rows)


def quintile_spread(df: pd.DataFrame, feature_col: str, target_col: str) -> dict:
    valid = df[[feature_col, target_col]].dropna().copy()
    if len(valid) < 30 or valid[feature_col].nunique() < 5:
        return {'target': target_col, 'n': len(valid), 'top_mean': None, 'bottom_mean': None, 'spread': None, 'hit_rate_top': None}
    valid['q'] = pd.qcut(valid[feature_col], 5, labels=False, duplicates='drop')
    bottom = valid[valid['q'] == valid['q'].min()][target_col]
    top = valid[valid['q'] == valid['q'].max()][target_col]
    return {
        'target': target_col,
        'n': int(len(valid)),
        'top_mean': float(top.mean()),
        'bottom_mean': float(bottom.mean()),
        'spread': float(top.mean() - bottom.mean()),
        'hit_rate_top': float((top > 0).mean()),
        'top_count': int(top.count()),
        'bottom_count': int(bottom.count()),
    }


def _as_2d_array(df: pd.DataFrame | pd.Series) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    # lstsq is more stable than explicitly inverting X'X and avoids scipy/statsmodels.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _normal_two_sided_pvalue(t_stat: float | None) -> float | None:
    if t_stat is None or not np.isfinite(t_stat):
        return None
    # Two-sided normal approximation. Good enough for screening; avoids scipy runtime dependency.
    return float(math.erfc(abs(float(t_stat)) / math.sqrt(2.0)))


def _hac_covariance(X: np.ndarray, residuals: np.ndarray, maxlags: int = 4) -> np.ndarray:
    """Newey-West/HAC covariance estimator implemented with NumPy only.

    This intentionally replaces statsmodels so the app starts reliably on Render even
    when SciPy/statsmodels binary compatibility changes. It is a lightweight research
    screening estimator, not a full econometrics package.
    """
    n, k = X.shape
    if n <= k + 1:
        return np.full((k, k), np.nan)

    xtx_inv = np.linalg.pinv(X.T @ X)
    xu = X * residuals.reshape(-1, 1)
    s = xu.T @ xu
    maxlags = max(0, min(int(maxlags), n - 1))
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        gamma = xu[lag:].T @ xu[:-lag]
        s += weight * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


def arx_regression(df: pd.DataFrame, feature_col: str, target_col: str) -> dict:
    valid = df[[feature_col, target_col]].dropna().copy()
    if len(valid) < 40:
        return {'target': target_col, 'n': len(valid), 'coef': None, 't': None, 'p': None, 'r2': None}
    valid['y_lag1'] = valid[target_col].shift(1)
    valid = valid.dropna()
    if len(valid) < 40:
        return {'target': target_col, 'n': len(valid), 'coef': None, 't': None, 'p': None, 'r2': None}

    y = valid[target_col].to_numpy(dtype=float)
    x_raw = _as_2d_array(valid[[feature_col, 'y_lag1']])
    X = np.column_stack([np.ones(len(valid)), x_raw])
    beta = _ols_beta(X, y)
    fitted = X @ beta
    residuals = y - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    cov = _hac_covariance(X, residuals, maxlags=4)
    se = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else np.full(X.shape[1], np.nan)
    coef = float(beta[1]) if len(beta) > 1 else np.nan
    t_stat = float(beta[1] / se[1]) if len(se) > 1 and se[1] and np.isfinite(se[1]) else None
    return {
        'target': target_col,
        'n': int(len(valid)),
        'coef': coef if np.isfinite(coef) else None,
        't': t_stat,
        'p': _normal_two_sided_pvalue(t_stat),
        'r2': r2,
    }


def expanding_directional_oos(df: pd.DataFrame, feature_col: str, target_col: str, min_train: int = 80) -> pd.DataFrame:
    valid = df[[feature_col, target_col]].dropna().copy()
    preds = []
    if len(valid) <= min_train + 5:
        return pd.DataFrame(columns=['asof', 'pred', 'actual', 'signal', 'correct'])
    for i in range(min_train, len(valid) - 1):
        train = valid.iloc[:i]
        test = valid.iloc[i:i + 1]
        try:
            y_train = train[target_col].to_numpy(dtype=float)
            x_train = train[[feature_col]].to_numpy(dtype=float)
            X_train = np.column_stack([np.ones(len(train)), x_train])
            beta = _ols_beta(X_train, y_train)
            X_test = np.array([[1.0, float(test[feature_col].iloc[0])]])
            pred = float((X_test @ beta)[0])
        except Exception:
            continue
        actual = float(test[target_col].iloc[0])
        signal = 1 if pred > 0 else -1
        preds.append({'asof': str(test.index[0]), 'pred': pred, 'actual': actual, 'signal': signal, 'correct': bool((pred > 0) == (actual > 0))})
    return pd.DataFrame(preds)


def summary_from_oos(oos: pd.DataFrame) -> dict:
    """Summarise OOS predictions against simple baselines.

    Earlier versions reported directional accuracy and signal return alone. That can be
    misleading for upward-drifting assets if the fitted model is mostly or always long.
    This summary explicitly compares the model to always-long/always-short baselines and
    records long-bias so the report can reject confidence-machine results.
    """
    if oos.empty:
        return {
            'n': 0,
            'directional_accuracy': None,
            'mean_signal_return': None,
            'hit_rate_signal_return_positive': None,
            'always_long_mean_return': None,
            'always_long_hit_rate': None,
            'always_short_mean_return': None,
            'always_short_hit_rate': None,
            'excess_mean_return_vs_always_long': None,
            'directional_accuracy_lift_vs_always_long': None,
            'signal_long_fraction': None,
            'signal_short_fraction': None,
            'baseline_note': 'No OOS rows available.',
        }
    actual = oos['actual'].astype(float)
    signal = oos['signal'].astype(float)
    pnl = signal * actual
    always_long_pnl = actual
    always_short_pnl = -actual
    always_long_hit = actual > 0
    always_short_hit = actual < 0
    signal_long_fraction = float((signal > 0).mean())
    signal_short_fraction = float((signal < 0).mean())
    signal_accuracy = float(oos['correct'].mean())
    always_long_accuracy = float(always_long_hit.mean())
    long_bias_warning = signal_long_fraction >= 0.85 or signal_short_fraction >= 0.85
    baseline_note = 'Signal is materially one-sided; compare against baseline before interpreting accuracy.' if long_bias_warning else 'Signal uses both long and short directions.'
    return {
        'n': int(len(oos)),
        'directional_accuracy': signal_accuracy,
        'mean_signal_return': float(pnl.mean()),
        'hit_rate_signal_return_positive': float((pnl > 0).mean()),
        'always_long_mean_return': float(always_long_pnl.mean()),
        'always_long_hit_rate': always_long_accuracy,
        'always_short_mean_return': float(always_short_pnl.mean()),
        'always_short_hit_rate': float(always_short_hit.mean()),
        'excess_mean_return_vs_always_long': float(pnl.mean() - always_long_pnl.mean()),
        'directional_accuracy_lift_vs_always_long': float(signal_accuracy - always_long_accuracy),
        'signal_long_fraction': signal_long_fraction,
        'signal_short_fraction': signal_short_fraction,
        'baseline_note': baseline_note,
    }


def validation_label(metrics: dict, *, coverage_ratio: float | None = None) -> dict:
    """Classify whether a target has passed a blunt evidence gate.

    This is intentionally conservative. It is not a trading recommendation; it stops a
    weak report from sounding better than it is.
    """
    reasons: list[str] = []
    best_p = metrics.get('best_regression_p')
    oos = metrics.get('oos') or {}
    best_spread = metrics.get('best_quintile_spread')
    if coverage_ratio is not None and coverage_ratio < 0.8:
        reasons.append(f'Coverage is limited: {coverage_ratio:.1%} of analysis rows matched target history.')
    if best_p is None or best_p > 0.10:
        reasons.append('Predictive regression is not statistically persuasive at p<=0.10.')
    if oos.get('directional_accuracy_lift_vs_always_long') is None or oos.get('directional_accuracy_lift_vs_always_long') <= 0.02:
        reasons.append('OOS directional accuracy does not beat always-long by at least 2 percentage points.')
    if oos.get('excess_mean_return_vs_always_long') is None or oos.get('excess_mean_return_vs_always_long') <= 0:
        reasons.append('OOS mean signal return does not beat always-long baseline.')
    if oos.get('signal_long_fraction') is not None and (oos.get('signal_long_fraction') >= 0.85 or oos.get('signal_short_fraction') >= 0.85):
        reasons.append('OOS signal is materially one-sided; raw accuracy may just reflect asset drift.')
    if best_spread is None or abs(float(best_spread)) < 0.01:
        reasons.append('Best quintile spread is economically small or unavailable.')
    status = 'validated_candidate' if not reasons else 'not_validated'
    return {'status': status, 'reasons': reasons}


def screen_feature_grid(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    *,
    min_n: int = 80,
) -> pd.DataFrame:
    """Evaluate a small, pre-declared grid of interpretable signal variants.

    This is a discovery screen only. Rows are ranked to decide what deserves deeper
    validation; they are not trading signals and should not be promoted without a
    separate out-of-sample/live-shadow pass.
    """
    rows: list[dict] = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        for target in target_cols:
            if target not in df.columns:
                continue
            valid = df[[feature, target]].dropna()
            if len(valid) < min_n or valid[feature].nunique() < 5:
                rows.append({
                    'feature': feature,
                    'target': target,
                    'n': int(len(valid)),
                    'screen_status': 'insufficient_data',
                    'reason': f'Fewer than {min_n} valid rows or too few feature values.',
                })
                continue
            q = quintile_spread(df, feature, target)
            r = arx_regression(df, feature, target)
            oos = expanding_directional_oos(df, feature, target)
            s = summary_from_oos(oos)
            validation = validation_label({
                'best_regression_p': r.get('p'),
                'best_quintile_spread': q.get('spread'),
                'oos': s,
            }, coverage_ratio=1.0)
            corr = valid[feature].corr(valid[target]) if len(valid) >= min_n else np.nan

            spread = q.get('spread')
            excess = s.get('excess_mean_return_vs_always_long')
            lift = s.get('directional_accuracy_lift_vs_always_long')
            p = r.get('p')
            score = 0.0
            if spread is not None and np.isfinite(spread):
                score += min(abs(float(spread)), 0.25)
            if excess is not None and np.isfinite(excess):
                score += max(float(excess), 0.0) * 2.0
            if lift is not None and np.isfinite(lift):
                score += max(float(lift), 0.0)
            if p is not None and np.isfinite(p):
                score += max(0.0, 0.10 - float(p))

            rows.append({
                'feature': feature,
                'target': target,
                'n': int(len(valid)),
                'corr': None if pd.isna(corr) else float(corr),
                'quintile_spread': q.get('spread'),
                'quintile_top_mean': q.get('top_mean'),
                'quintile_bottom_mean': q.get('bottom_mean'),
                'regression_coef': r.get('coef'),
                'regression_t': r.get('t'),
                'regression_p': r.get('p'),
                'oos_n': s.get('n'),
                'oos_directional_accuracy': s.get('directional_accuracy'),
                'oos_always_long_hit_rate': s.get('always_long_hit_rate'),
                'oos_accuracy_lift_vs_always_long': s.get('directional_accuracy_lift_vs_always_long'),
                'oos_mean_signal_return': s.get('mean_signal_return'),
                'oos_always_long_mean_return': s.get('always_long_mean_return'),
                'oos_excess_mean_return_vs_always_long': s.get('excess_mean_return_vs_always_long'),
                'signal_long_fraction': s.get('signal_long_fraction'),
                'screen_status': validation.get('status'),
                'reason': '; '.join(validation.get('reasons') or []),
                'screen_score': float(score),
            })
    out = pd.DataFrame(rows)
    if not out.empty and 'screen_score' in out.columns:
        out = out.sort_values(['screen_status', 'screen_score'], ascending=[True, False]).reset_index(drop=True)
    return out
