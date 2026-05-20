from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def arx_regression(df: pd.DataFrame, feature_col: str, target_col: str) -> dict:
    valid = df[[feature_col, target_col]].dropna().copy()
    if len(valid) < 40:
        return {'target': target_col, 'n': len(valid), 'coef': None, 't': None, 'p': None, 'r2': None}
    valid['y_lag1'] = valid[target_col].shift(1)
    valid = valid.dropna()
    X = sm.add_constant(valid[[feature_col, 'y_lag1']])
    y = valid[target_col]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    return {
        'target': target_col,
        'n': int(model.nobs),
        'coef': float(model.params.get(feature_col, np.nan)),
        't': float(model.tvalues.get(feature_col, np.nan)),
        'p': float(model.pvalues.get(feature_col, np.nan)),
        'r2': float(model.rsquared),
    }


def expanding_directional_oos(df: pd.DataFrame, feature_col: str, target_col: str, min_train: int = 80) -> pd.DataFrame:
    valid = df[[feature_col, target_col]].dropna().copy()
    preds = []
    if len(valid) <= min_train + 5:
        return pd.DataFrame(columns=['asof', 'pred', 'actual', 'signal', 'correct'])
    for i in range(min_train, len(valid) - 1):
        train = valid.iloc[:i]
        test = valid.iloc[i:i + 1]
        X_train = sm.add_constant(train[[feature_col]])
        y_train = train[target_col]
        try:
            model = sm.OLS(y_train, X_train).fit()
            X_test = sm.add_constant(test[[feature_col]], has_constant='add')
            pred = float(model.predict(X_test).iloc[0])
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
