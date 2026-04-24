"""Minimal JSON HTTP helpers (stdlib only)."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Mapping, MutableMapping, Optional


def json_request(
    method: str,
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    json_body: Any = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    """
    Perform an HTTP request and parse a JSON response.

    Returns:
        Tuple of (status_code, parsed_json_or_text).
    """
    h: MutableMapping[str, str] = {}
    if headers:
        h.update(headers)
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        h.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=dict(h), method=method.upper())
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
            code = resp.getcode() or 200
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        code = e.code
    if not raw:
        return code, None
    try:
        return code, json.loads(raw)
    except json.JSONDecodeError:
        return code, raw
