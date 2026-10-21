"""NGC backend tests: the captcha-free route to preview models.

Run: python3 tests/test_ngc.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv2api import ngc  # noqa: E402
from nv2api.config import Config  # noqa: E402
from nv2api.server import Handler, Upstream  # noqa: E402
from tests import mock_ngc  # noqa: E402

FAILS: list[str] = []
PASSES = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSES
    if cond:
        PASSES += 1
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name} {detail}")


def post(url: str, payload: dict, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def start_gateway(upstream_base: str, **overrides) -> ThreadingHTTPServer:
    cfg = Config(
        host="127.0.0.1",
        port=0,
        upstream_base=upstream_base,
        upstream_keys=["nvapi-test"],
        upstream_retries=0,
        upstream_timeout=30,
        log_requests=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    Handler.cfg = cfg
    Handler.upstream = Upstream(cfg)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    mock = mock_ngc.start(0)
    base = f"http://127.0.0.1:{mock.server_port}"
    os.environ["N2A_AUTHN_URL"] = f"{base}/token"
    os.environ["N2A_BUILDAPI"] = base
    os.environ["N2A_CATALOG_PAGE"] = base + "/{publisher}/{model}"
    print(f"mock ngc = {base}\n")

    # 1 token exchange
    print("token exchange")
    jwt, ttl = ngc.exchange_ngc_key(mock_ngc.GOOD_NGC_KEY)
    check("good key returns a JWT", jwt.count(".") == 2, jwt[:40])
    check("ttl is sane", 60 <= ttl <= 3600, str(ttl))
    try:
        ngc.exchange_ngc_key("nvapi-wrong")
        check("bad key rejected", False, "no exception")
    except urllib.error.HTTPError as exc:
        check("bad key rejected", exc.code == 401, str(exc.code))

    # 2 function id discovery
    print("function id discovery")
    fid = ngc.discover_function_id(mock_ngc.MODEL)
    check("scraped nvcfFunctionId", fid == mock_ngc.FUNCTION_ID, str(fid))

    # 3 entitlement probe
    print("entitlement probe")
    status, raw = ngc.probe_entitlement(mock_ngc.GOOD_NGC_KEY, mock_ngc.MODEL)
    check("probe reaches model", status == 200, f"{status} {raw[:120]}")

    # 4 gateway: auto backend falls back on 404
    print("gateway auto-fallback")
    # integrate.api mock that 404s for everything -> the real preview behaviour
    dead = ThreadingHTTPServer(("127.0.0.1", 0), _NotFound)
    dead.daemon_threads = True
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    gw = start_gateway(f"http://127.0.0.1:{dead.server_port}/v1", backend="auto", ngc_keys=[mock_ngc.GOOD_NGC_KEY])
    g = f"http://127.0.0.1:{gw.server_port}"
    payload = {
        "model": "cosmos3-nano-reasoner",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
        ],
    }
    status, raw = post(f"{g}/v1/chat/completions", payload)
    body = json.loads(raw) if raw else {}
    check("404 upstream falls through to ngc", status == 200, f"{status} {raw[:160]}")
    check("reports ngc backend", body.get("nv2api_backend") == "ngc", str(body.get("nv2api_backend")))
    check("content from ngc", "NGC backend" in json.dumps(body), str(body)[:160])
    check("think stripped on ngc path", "<think>" not in json.dumps(body.get("choices", [])), str(body)[:160])
    sent = mock_ngc.STATE["requests"][-1]
    check("ngc got no system role", all(m["role"] in ("user", "assistant") for m in sent["body"]["messages"]))
    check("ngc path is the protected predict route", sent["path"].startswith("/v3/protected/predict/stateless/"), sent["path"])

    # 5 streaming through ngc
    print("gateway streaming via ngc")
    status, raw = post(f"{g}/v1/chat/completions", {**payload, "stream": True})
    check("stream 200", status == 200, raw[:160])
    events = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    check("stream terminated", events and events[-1] == "[DONE]", str(events[-1:]))
    text = "".join(
        (json.loads(e)["choices"][0]["delta"].get("content") or "") for e in events if e != "[DONE]"
    )
    check("stream text assembled", "NGC backend speaking." in text, text[:120])

    # 6 forced ngc backend, and auto without ngc keys stays 404
    print("backend modes")
    gw2 = start_gateway(f"http://127.0.0.1:{dead.server_port}/v1", backend="auto", ngc_keys=[])
    g2 = f"http://127.0.0.1:{gw2.server_port}"
    status, raw = post(f"{g2}/v1/chat/completions", payload)
    check("no ngc key -> 404 preserved", status == 404, f"{status} {raw[:100]}")

    gw3 = start_gateway(f"http://127.0.0.1:{dead.server_port}/v1", backend="ngc", ngc_keys=[mock_ngc.GOOD_NGC_KEY])
    g3 = f"http://127.0.0.1:{gw3.server_port}"
    status, raw = post(f"{g3}/v1/chat/completions", payload)
    check("forced ngc works", status == 200, f"{status} {raw[:120]}")

    gw4 = start_gateway(f"http://127.0.0.1:{dead.server_port}/v1", backend="ngc", ngc_keys=["nvapi-wrong"])
    g4 = f"http://127.0.0.1:{gw4.server_port}"
    status, raw = post(f"{g4}/v1/chat/completions", payload)
    check("bad ngc key surfaces 502 with reason", status == 502 and "token exchange" in raw, f"{status} {raw[:140]}")

    print(f"\n{PASSES} passed, {len(FAILS)} failed")
    if FAILS:
        print("failed:", ", ".join(FAILS))
        return 1
    return 0


class _NotFound(BaseHTTPRequestHandler):
    """Mimics integrate.api.nvidia.com answering 404 for an unrouted model."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = b"404 page not found\n"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    sys.exit(main())
