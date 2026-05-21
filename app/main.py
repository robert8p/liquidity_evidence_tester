from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from app import __version__
from app.config import get_settings
from app.models import NetLiquidityRunRequest, CftcJpyRunRequest
from app.research.net_liquidity import run_net_liquidity
from app.research.cftc_jpy import run_cftc_jpy
from app.utils import read_json

settings = get_settings()
app = FastAPI(title=settings.app_title, version=__version__)


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return f"""
    <!doctype html>
    <html>
    <head>
      <title>{settings.app_title}</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; background: #f7f7f8; color: #111; }}
        .card {{ background: white; padding: 20px; border-radius: 14px; box-shadow: 0 2px 12px rgba(0,0,0,.08); margin-bottom: 18px; }}
        button {{ padding: 10px 14px; border: 0; border-radius: 8px; cursor: pointer; margin-right: 8px; margin-top: 6px; }}
        .primary {{ background: #111827; color: white; }}
        .secondary {{ background: #e5e7eb; color: #111827; }}
        pre {{ background: #111827; color: #f9fafb; padding: 14px; border-radius: 10px; overflow:auto; }}
        input {{ padding: 8px; border: 1px solid #ddd; border-radius: 8px; margin: 4px; }}
        .warn {{ color: #92400e; }}
        .muted {{ color: #4b5563; }}
      </style>
    </head>
    <body>
      <h1>{settings.app_title} <span class="muted">v{__version__}</span></h1>
      <p>Research-only evidence tester. No trading, no order routing, no alerts.</p>

      <div class="card">
        <h2>Run net-liquidity evidence test</h2>
        <p>Tests Fed assets − TGA − ON RRP as a leading signal for BTC and QQQ/Nasdaq proxy. v0.2.0 evidence did not validate this standalone thesis, but the module is retained for audit and reruns.</p>
        <label>Start date <input id="net_start" value="2018-01-01" /></label>
        <label>Equity target <input id="net_target" value="{settings.equity_target_symbol}" /></label>
        <label><input id="net_screen" type="checkbox" checked /> Screen liquidity signal variants</label>
        <br/>
        <button class="primary" onclick="runNet(false)">Run with configured APIs</button>
        <button class="secondary" onclick="runNet(true)">Run demo mode</button>
      </div>

      <div class="card">
        <h2>Run CFTC JPY positioning evidence test</h2>
        <p>Tests CFTC leveraged-fund Japanese-yen positioning as a leading signal for USD/JPY. CFTC release timing is aligned before target returns are calculated.</p>
        <label>Start date <input id="jpy_start" value="2012-01-01" /></label>
        <label><input id="jpy_screen" type="checkbox" checked /> Screen JPY positioning variants</label>
        <br/>
        <button class="primary" onclick="runJpy(false)">Run with CFTC/FRED data</button>
        <button class="secondary" onclick="runJpy(true)">Run demo mode</button>
      </div>

      <div class="card">
        <h2>Evidence packs</h2>
        <button class="secondary" onclick="latest()">Latest run</button>
        <a href="/api/evidence/latest.zip"><button class="secondary">Download latest pack</button></a>
        <p class="warn">Use demo mode to validate the app before trusting any configured API/data run. Demo evidence is synthetic.</p>
      </div>

      <div class="card">
        <h2>Status</h2>
        <pre id="out">Ready.</pre>
      </div>

      <script>
        function setRunning(label) {{
          document.getElementById('out').textContent = label + ' Running... this may take a little while if API calls are used.';
        }}
        async function runNet(demo) {{
          setRunning('Net-liquidity test:');
          const body = {{
            start_date: document.getElementById('net_start').value,
            target_symbol: document.getElementById('net_target').value,
            include_btc: true,
            include_equity: true,
            demo_mode: demo,
            horizons_weeks: [1,2,4,8],
            screen_features: document.getElementById('net_screen').checked
          }};
          const res = await fetch('/api/run/net-liquidity', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        }}
        async function runJpy(demo) {{
          setRunning('CFTC JPY test:');
          const body = {{
            start_date: document.getElementById('jpy_start').value,
            demo_mode: demo,
            horizons_weeks: [1,2,4,8],
            screen_features: document.getElementById('jpy_screen').checked
          }};
          const res = await fetch('/api/run/cftc-jpy', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        }}
        async function latest() {{
          const res = await fetch('/api/runs/latest');
          const data = await res.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        }}
      </script>
    </body>
    </html>
    """


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'version': __version__,
        'data_dir': str(settings.data_dir),
        'has_coinapi_key': bool(settings.coinapi_key),
        'has_massive_api_key': bool(settings.massive_api_key),
        'has_alpaca_keys': bool(settings.alpaca_key_id and settings.alpaca_secret_key),
        'cftc_fred_usdjpy_series': settings.fred_usdjpy_series,
    }


@app.get('/api/config')
def config() -> dict:
    return {
        'version': __version__,
        'data_dir': str(settings.data_dir),
        'equity_source': settings.equity_source,
        'equity_target_symbol': settings.equity_target_symbol,
        'coinapi_btc_symbol_id': settings.coinapi_btc_symbol_id,
        'fred_series': {
            'fed_assets': settings.fred_fed_assets_series,
            'tga': settings.fred_tga_series,
            'onrrp': settings.fred_onrrp_series,
            'usdjpy': settings.fred_usdjpy_series,
        },
        'credentials_present': {
            'coinapi': bool(settings.coinapi_key),
            'massive': bool(settings.massive_api_key),
            'alpaca': bool(settings.alpaca_key_id and settings.alpaca_secret_key),
        },
        'available_tests': ['net_liquidity', 'cftc_jpy'],
    }


@app.post('/api/run/net-liquidity')
def run_net_liquidity_endpoint(req: NetLiquidityRunRequest) -> JSONResponse:
    try:
        result = run_net_liquidity(
            settings,
            start_date=req.start_date,
            end_date=req.end_date,
            target_symbol=req.target_symbol,
            include_btc=req.include_btc,
            include_equity=req.include_equity,
            demo_mode=req.demo_mode,
            horizons_weeks=req.horizons_weeks,
            screen_features=req.screen_features,
        )
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/run/cftc-jpy')
def run_cftc_jpy_endpoint(req: CftcJpyRunRequest) -> JSONResponse:
    try:
        result = run_cftc_jpy(
            settings,
            start_date=req.start_date,
            end_date=req.end_date,
            demo_mode=req.demo_mode,
            horizons_weeks=req.horizons_weeks,
            screen_features=req.screen_features,
        )
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/api/runs/latest')
def latest_run() -> dict:
    latest_path = settings.data_dir / 'runs' / 'latest.json'
    if not latest_path.exists():
        return {'status': 'no_runs_yet', 'message': 'Run demo mode or a configured API test first.'}
    return read_json(latest_path)


@app.get('/api/evidence/latest.zip')
def latest_pack() -> FileResponse:
    latest_path = settings.data_dir / 'runs' / 'latest.json'
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail='No evidence pack exists yet.')
    pack = Path(read_json(latest_path)['pack'])
    if not pack.exists():
        raise HTTPException(status_code=404, detail='Latest evidence pack file is missing.')
    return FileResponse(pack, filename=pack.name, media_type='application/zip')


@app.get('/api/evidence/{run_id}.zip')
def pack_by_id(run_id: str) -> FileResponse:
    pack = settings.data_dir / 'packs' / f'{run_id}.zip'
    if not pack.exists():
        raise HTTPException(status_code=404, detail='Evidence pack not found.')
    return FileResponse(pack, filename=pack.name, media_type='application/zip')
