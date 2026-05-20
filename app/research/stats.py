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
    if oos.empty:
        return {'n': 0, 'directional_accuracy': None, 'mean_signal_return': None}
    pnl = oos['signal'] * oos['actual']
    return {
        'n': int(len(oos)),
        'directional_accuracy': float(oos['correct'].mean()),
        'mean_signal_return': float(pnl.mean()),
        'hit_rate_signal_return_positive': float((pnl > 0).mean()),
    }
