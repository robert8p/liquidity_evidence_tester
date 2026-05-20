from __future__ import annotations

from pathlib import Path
from typing import Any
import requests
from app.utils import sha256_bytes, utc_now_iso, write_json


class HttpError(RuntimeError):
    pass


def get_bytes(url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: int = 45) -> bytes:
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise HttpError(f"GET {url} failed: {r.status_code} {r.text[:300]}")
    return r.content


def snapshot_payload(raw_dir: Path, source: str, endpoint: str, content: bytes, suffix: str = 'payload') -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    h = sha256_bytes(content)[:16]
    safe_source = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in source)
    path = raw_dir / f"{safe_source}_{h}.{suffix}"
    path.write_bytes(content)
    meta = {
        'source': source,
        'endpoint': endpoint,
        'pulled_at_utc': utc_now_iso(),
        'content_sha256': sha256_bytes(content),
        'payload_path': str(path),
    }
    write_json(path.with_suffix(path.suffix + '.meta.json'), meta)
    return path
