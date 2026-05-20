from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_title: str = Field(default='Liquidity Evidence Tester', alias='APP_TITLE')
    app_env: str = Field(default='development', alias='APP_ENV')
    data_dir: Path = Field(default=Path('/tmp/liquidity_evidence_tester'), alias='DATA_DIR')

    coinapi_key: str | None = Field(default=None, alias='COINAPI_KEY')
    alpaca_key_id: str | None = Field(default=None, alias='ALPACA_KEY_ID')
    alpaca_secret_key: str | None = Field(default=None, alias='ALPACA_SECRET_KEY')
    massive_api_key: str | None = Field(default=None, alias='MASSIVE_API_KEY')

    coinapi_btc_symbol_id: str = Field(default='COINBASE_SPOT_BTC_USD', alias='COINAPI_BTC_SYMBOL_ID')
    equity_source: str = Field(default='massive', alias='EQUITY_SOURCE')  # massive | alpaca
    equity_target_symbol: str = Field(default='QQQ', alias='EQUITY_TARGET_SYMBOL')

    fred_fed_assets_series: str = 'WALCL'
    fred_tga_series: str = 'WTREGEN'
    fred_onrrp_series: str = 'RRPONTSYD'

    # FRED WALCL and WTREGEN are commonly millions USD; RRPONTSYD is often billions USD.
    # We normalise all to billions for the net-liquidity feature.
    fred_fed_assets_scale_to_billions: float = 0.001
    fred_tga_scale_to_billions: float = 0.001
    fred_onrrp_scale_to_billions: float = 1.0

    def ensure_dirs(self) -> None:
        for sub in ['raw', 'runs', 'packs', 'tmp']:
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
