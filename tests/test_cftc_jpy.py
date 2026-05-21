from pathlib import Path
import pandas as pd

from app.adapters.cftc import extract_jpy_tff_features
from app.release_calendar import cftc_release_time, attach_cftc_alignment
from app.research.cftc_jpy import build_jpy_positioning_features, run_cftc_jpy
from app.config import Settings


def test_extract_jpy_tff_features_flexible_columns():
    df = pd.DataFrame({
        'Market_and_Exchange_Names': ['JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE', 'EURO FX - CHICAGO MERCANTILE EXCHANGE'],
        'Report_Date_as_YYYY-MM-DD': ['2024-01-02', '2024-01-02'],
        'Open_Interest_All': [1000, 2000],
        'Lev_Money_Positions_Long_All': [300, 100],
        'Lev_Money_Positions_Short_All': [500, 200],
    })
    out = extract_jpy_tff_features(df)
    assert len(out) == 1
    assert 'lev_net_oi' in out.columns
    assert abs(float(out['lev_net_oi'].iloc[0]) + 0.2) < 1e-9


def test_cftc_release_alignment_friday_1530_et():
    observed = pd.Timestamp('2024-01-02T00:00:00Z')  # Tuesday
    released = cftc_release_time(observed)
    assert released.day_name() == 'Friday'
    assert released.hour in (19, 20)  # UTC depends on DST; January is 20:30 UTC.


def test_attach_cftc_alignment_fx_not_same_friday():
    idx = pd.DatetimeIndex([pd.Timestamp('2024-01-02T00:00:00Z')])
    df = pd.DataFrame({'lev_net_oi': [0.1]}, index=idx)
    out = attach_cftc_alignment(df, instrument_type='fx')
    assert out['released_at_utc'].iloc[0].day_name() == 'Friday'
    assert out['effective_trade_at_utc'].iloc[0].day_name() == 'Monday'


def test_build_jpy_positioning_features_contains_variants():
    idx = pd.date_range('2020-01-07', periods=180, freq='W-TUE', tz='UTC')
    df = pd.DataFrame({
        'open_interest': 1000,
        'long': [300 + i for i in range(180)],
        'short': [400 - (i % 10) for i in range(180)],
        'lev_net_contracts': [-100 + i for i in range(180)],
        'lev_net_oi': [(-100 + i) / 1000 for i in range(180)],
        'lev_short_oi': [(400 - (i % 10)) / 1000 for i in range(180)],
        'lev_long_oi': [(300 + i) / 1000 for i in range(180)],
    }, index=idx)
    out = build_jpy_positioning_features(df)
    for col in ['contrarian_usdjpy_z_156', 'trend_usdjpy_z_156', 'extreme_reversal_signal']:
        assert col in out.columns
    assert out['contrarian_usdjpy_z_156'].notna().sum() > 10


def test_run_cftc_jpy_demo(tmp_path: Path):
    settings = Settings(DATA_DIR=tmp_path)
    settings.ensure_dirs()
    result = run_cftc_jpy(settings, start_date='2016-01-01', end_date='2024-01-01', demo_mode=True, horizons_weeks=[1, 2, 4], screen_features=True)
    assert result['status'] == 'completed'
    assert Path(result['pack']).exists()
    assert 'USDJPY' in result['metrics']['targets']
    assert (tmp_path / 'runs' / 'latest.json').exists()
