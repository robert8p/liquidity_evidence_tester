from app.config import get_settings
from app.research.net_liquidity import run_net_liquidity

settings = get_settings()
result = run_net_liquidity(
    settings,
    start_date='2018-01-01',
    end_date=None,
    target_symbol='QQQ',
    include_btc=True,
    include_equity=True,
    demo_mode=True,
    horizons_weeks=[1, 2, 4, 8],
)
print(result['pack'])
