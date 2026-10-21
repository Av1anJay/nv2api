"""NGC backend — the *other* door to preview models like cosmos3-nano-reasoner.

`integrate.api.nvidia.com/v1` does not route cosmos3-nano-reasoner at all (it
answers `404 page not found`, while GA models on the same key answer 403). The
build.nvidia.com playground therefore does not use it either; it posts to:

    POST https://buildapi.ngc.nvidia.com/v3/predict/stateless/<model>
      nv-function-id: <nvcf function uuid>
      nv-captcha-token: <invisible hCaptcha token>      # signed-out path

Verified: no token -> 400 "Captcha required"; bogus token -> 400 "Token is
invalid". There is an authenticated sibling that needs no captcha:

    POST https://buildapi.ngc.nvidia.com/v3/protected/predict/stateless/<model>
      Authorization: Bearer <NGC JWT from authn.nvidia.com>

which without a JWT answers `403 RBAC: access denied`, and with a non-JWT
`401 Jwt is not in the form of Header.Payload.Signature`. So an NGC API key
(ngc.nvidia.com -> Setup -> API Keys) buys a captcha-free route to the same
model, and that is what this module implements.
"""

from __future__ import annotations

import base64
import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

AUTHN_URL = "https://authn.nvidia.com/token"
BUILDAPI = "https://buildapi.ngc.nvidia.com"
CATALOG_PAGE = "https://build.nvidia.com/{publisher}/{model}"


def _authn_url() -> str:
    import os

    return os.environ.get("N2A_AUTHN_URL", AUTHN_URL)


def _buildapi() -> str:
    import os

    return os.environ.get("N2A_BUILDAPI", BUILDAPI)


def _catalog_page() -> str:
    import os

    return os.environ.get("N2A_CATALOG_PAGE", CATALOG_PAGE)

_TOKEN_RE = re.compile(
    r"nvcfFunctionId[^0-9a-f]{0,16}"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)

_ssl = ssl.create_default_context()


def exchange_ngc_key(ngc_key: str, timeout: int = 30) -> tuple[str, int]:
    """NGC API key -> (JWT, seconds_until_expiry).

    NGC's token service takes the key as the *username* with the literal
    password `$oauthtoken`.
    """
    token = base64.b64encode(f"$oauthtoken:{ngc_key}".encode()).decode()
    query = urllib.parse.urlencode({"service": "ngc", "scope": "group/ngc"})
    req = urllib.request.Request(
        f"{_authn_url()}?{query}",
        headers={"Authorization": f"Basic {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl) as resp:
        raw = resp.read().decode(errors="replace")
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - some deployments return the bare JWT
        return raw.strip(), 3000
    jwt = data.get("token") or data.get("access_token") or data.get("id_token")
    if not jwt:
        raise RuntimeError(f"no token in authn response: {raw[:200]}")
    ttl = 3000
    try:
        payload = jwt.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        if "exp" in claims:
            ttl = max(60, int(claims["exp"]) - int(time.time()) - 60)
    except Exception:  # noqa: BLE001
        pass
    return jwt, ttl


class TokenCache:
    """Thread-safe JWT cache keyed by NGC API key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: tuple[str, str, float] | None = None  # key, jwt, expires_at

    def get(self, ngc_key: str) -> str:
        with self._lock:
            if self._entry and self._entry[0] == ngc_key and time.time() < self._entry[2]:
                return self._entry[1]
            jwt, ttl = exchange_ngc_key(ngc_key)
            self._entry = (ngc_key, jwt, time.time() + ttl)
            return jwt


_FID_CACHE: dict[str, str] = {}
_FID_LOCK = threading.Lock()


def discover_function_id(model: str, timeout: int = 20) -> str | None:
    """Scrape `nvcfFunctionId` off the model's build.nvidia.com page.

    Verified against https://build.nvidia.com/nvidia/cosmos3-nano-reasoner ->
    24a385c8-111a-4294-8c73-cf7575cc2595.
    """
    with _FID_LOCK:
        if model in _FID_CACHE:
            return _FID_CACHE[model]
    publisher, _, name = model.partition("/")
    if not name:
        publisher, name = "nvidia", model
    url = _catalog_page().format(publisher=publisher, model=name)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nv2api)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl) as resp:
            html = resp.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return None
    match = _TOKEN_RE.search(html)
    if not match:
        return None
    fid = match.group(1)
    with _FID_LOCK:
        _FID_CACHE[model] = fid
    return fid


def predict_path(model: str, authenticated: bool = True) -> str:
    prefix = "/v3/protected/predict/stateless" if authenticated else "/v3/predict/stateless"
    return f"{prefix}/{model}"


def build_headers(jwt: str, function_id: str | None, stream: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if function_id:
        headers["nv-function-id"] = function_id
    return headers


def probe_entitlement(ngc_key: str, model: str, timeout: int = 60) -> tuple[int, str]:
    """Cheapest possible round trip: does this NGC key reach this model?"""
    try:
        jwt = exchange_ngc_key(ngc_key)[0]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)
    fid = discover_function_id(model)
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
    ).encode()
    req = urllib.request.Request(
        f"{_buildapi()}{predict_path(model)}",
        data=body,
        headers=build_headers(jwt, fid, stream=False),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl) as resp:
            return resp.status, resp.read().decode(errors="replace")[:400]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")[:400]
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)
