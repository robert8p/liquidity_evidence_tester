from __future__ import annotations

from pydantic import BaseModel, Field


class NetLiquidityRunRequest(BaseModel):
    start_date: str = Field(default='2018-01-01', description='YYYY-MM-DD')
    end_date: str | None = Field(default=None, description='YYYY-MM-DD; defaults to today')
    target_symbol: str = Field(default='QQQ', description='Equity ETF target for Nasdaq proxy')
    include_btc: bool = True
    include_equity: bool = True
    demo_mode: bool = False
    horizons_weeks: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    screen_features: bool = Field(default=True, description='Run a pre-declared liquidity signal variant screen.')


class CftcJpyRunRequest(BaseModel):
    start_date: str = Field(default='2012-01-01', description='YYYY-MM-DD')
    end_date: str | None = Field(default=None, description='YYYY-MM-DD; defaults to today')
    demo_mode: bool = False
    horizons_weeks: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    screen_features: bool = Field(default=True, description='Run a pre-declared CFTC JPY signal variant screen.')


class RunSummary(BaseModel):
    run_id: str
    status: str
    hypothesis: str
    created_at_utc: str
    evidence_pack: str | None = None
    metrics: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
