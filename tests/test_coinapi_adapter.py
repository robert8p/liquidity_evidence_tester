from __future__ import annotations

import json
from pathlib import Path

from app.adapters import coinapi


def test_coinapi_fetch_ohlcv_decodes_bytes_payload(monkeypatch, tmp_path: Path):
    payload = [
        {
            "time_period_start": "2024-01-01T00:00:00.0000000Z",
            "price_close": 42000.5,
        },
        {
            "time_period_start": "2024-01-02T00:00:00.0000000Z",
            "price_close": 43000.75,
        },
    ]

    def fake_get_bytes(url, *, headers=None, params=None, timeout=45):
        assert headers and headers.get("X-CoinAPI-Key") == "test-key"
        assert params and params.get("period_id") == "1DAY"
        return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(coinapi, "get_bytes", fake_get_bytes)
    df = coinapi.fetch_ohlcv(
        "COINBASE_SPOT_BTC_USD",
        "test-key",
        start_iso="2024-01-01T00:00:00Z",
        end_iso="2024-01-03T00:00:00Z",
        raw_dir=tmp_path,
    )

    assert len(df) == 2
    assert float(df["close"].iloc[-1]) == 43000.75
    assert any(tmp_path.glob("coinapi_COINBASE_SPOT_BTC_USD_*.json"))
