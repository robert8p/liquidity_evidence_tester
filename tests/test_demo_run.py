from pathlib import Path
from app.config import Settings
from app.research.net_liquidity import run_net_liquidity


def test_demo_run_builds_pack(tmp_path: Path):
    settings = Settings(DATA_DIR=tmp_path)
    settings.ensure_dirs()
    result = run_net_liquidity(
        settings,
        start_date='2018-01-01',
        end_date='2021-01-01',
        target_symbol='QQQ',
        include_btc=True,
        include_equity=True,
        demo_mode=True,
        horizons_weeks=[1, 2, 4],
        screen_features=False,
    )
    assert Path(result['pack']).exists()
    assert 'BTCUSD' in result['metrics']['targets']
    assert 'QQQ' in result['metrics']['targets']
