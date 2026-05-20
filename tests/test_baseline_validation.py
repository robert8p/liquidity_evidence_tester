import pandas as pd

from app.research.stats import summary_from_oos, validation_label


def test_oos_summary_compares_signal_to_always_long_baseline():
    oos = pd.DataFrame({
        'pred': [0.1, 0.2, 0.3, 0.4],
        'actual': [0.01, -0.02, 0.03, 0.04],
        'signal': [1, 1, 1, 1],
        'correct': [True, False, True, True],
    })
    summary = summary_from_oos(oos)
    assert summary['signal_long_fraction'] == 1.0
    assert summary['always_long_mean_return'] == summary['mean_signal_return']
    assert summary['excess_mean_return_vs_always_long'] == 0.0
    assert 'one-sided' in summary['baseline_note']


def test_validation_rejects_one_sided_no_baseline_lift():
    metrics = {
        'best_regression_p': 0.5,
        'best_quintile_spread': 0.02,
        'oos': {
            'directional_accuracy_lift_vs_always_long': 0.0,
            'excess_mean_return_vs_always_long': 0.0,
            'signal_long_fraction': 1.0,
            'signal_short_fraction': 0.0,
        },
    }
    label = validation_label(metrics, coverage_ratio=1.0)
    assert label['status'] == 'not_validated'
    assert any('one-sided' in r for r in label['reasons'])
