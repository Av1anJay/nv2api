"""End-to-end test: real HTTP calls through nv2api against a mock NIM upstream.

Run: python3 tests/test_e2e.py     (no pytest / no third-party deps)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nv2api.config import Config  # noqa: E402
from nv2api.server import Handler, Upstream  # noqa: E402
from tests import mock_nim  # noqa: E402

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
            return resp.status, resp.read().decode(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(), dict(exc.headers)


def get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def start_gateway(upstream_base: str, **overrides) -> ThreadingHTTPServer:
    cfg = Config(
        host="127.0.0.1",
        port=0,
        upstream_base=upstream_base,
        upstream_keys=["nvapi-test-key"],
        upstream_retries=0,
        upstream_timeout=30,
        split_think=True,
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
    mock = mock_nim.start(0)
    base_up = f"http://127.0.0.1:{mock.server_port}/v1"
    # NVCF asset endpoint is a different host upstream; point it at the mock
    os.environ["N2A_ASSET_API"] = f"http://127.0.0.1:{mock.server_port}/v2/nvcf/assets"
    gw = start_gateway(base_up)
    base = f"http://127.0.0.1:{gw.server_port}"
    print(f"mock={base_up} gateway={base}/v1\n")

    # 1 health
    print("health")
    status, raw = get(f"{base}/healthz")
    check("GET /healthz 200", status == 200, raw[:120])
    check("health reports upstream", json.loads(raw)["upstream"] == base_up)

    # 2 models
    print("models")
    status, raw = get(f"{base}/v1/models")
    ids = [m["id"] for m in json.loads(raw)["data"]] if status == 200 else []
    check("GET /v1/models 200", status == 200, raw[:120])
    check("advertises cosmos3-nano-reasoner", "nvidia/cosmos3-nano-reasoner" in ids, str(ids))

    # 3 non-stream chat, system folding + reasoning split + max_tokens clamp
    print("chat (non-stream)")
    payload = {
        "model": "cosmos3-nano-reasoner",
        "messages": [
            {"role": "system", "content": "You are a physical-world reasoner."},
            {"role": "user", "content": "What happens next in the video?"},
        ],
        "max_tokens": 9000,
        "temperature": 0.3,
        "tools": [{"type": "function", "function": {"name": "nope"}}],
    }
    status, raw, _ = post(f"{base}/v1/chat/completions", payload)
    body = json.loads(raw) if raw else {}
    check("200 from gateway", status == 200, raw[:200])
    msg = (body.get("choices") or [{}])[0].get("message", {})
    check("content has no <think>", "<think>" not in (msg.get("content") or ""), str(msg)[:200])
    check("reasoning_content extracted", bool(msg.get("reasoning_content")), str(msg)[:200])
    check("usage passed through", body.get("usage", {}).get("total_tokens") == 12474, str(body.get("usage")))
    check("warnings reported", "nv2api_warnings" in body, str(body.get("nv2api_warnings")))
    check("model alias resolved", body.get("model") == "nvidia/cosmos3-nano-reasoner", str(body.get("model")))
    sent = mock_nim.STATE["requests"][-1]
    check("upstream got no system role", all(m["role"] in ("user", "assistant") for m in sent["messages"]))
    check("system folded into user text", "physical-world reasoner" in json.dumps(sent["messages"]))
    check("max_tokens clamped to 4096", sent.get("max_tokens") == 4096, str(sent.get("max_tokens")))
    check("tools stripped", "tools" not in sent, str(list(sent)))

    # 4 streaming
    print("chat (stream)")
    status, raw, headers = post(f"{base}/v1/chat/completions", {**payload, "stream": True})
    check("200 stream", status == 200, raw[:200])
    check("sse content-type", "text/event-stream" in (headers.get("Content-Type") or ""), str(headers.get("Content-Type")))
    events = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")]
    check("terminated with [DONE]", events and events[-1] == "[DONE]", str(events[-1:]))
    deltas = [json.loads(e) for e in events if e != "[DONE]"]
    content = "".join((d["choices"][0]["delta"].get("content") or "") for d in deltas)
    reasoning = "".join((d["choices"][0]["delta"].get("reasoning_content") or "") for d in deltas)
    check("stream content clean", "<think>" not in content and "robot arm" not in content, content[:120])
    check("stream reasoning captured", "robot arm" in reasoning, reasoning[:120])
    check("stream content complete", "[517, 742]" in content, content[:200])
    check("finish_reason present", any(d["choices"][0].get("finish_reason") == "stop" for d in deltas))
    check("usage in final chunk", any("usage" in d for d in deltas))

    # 5 legacy completions
    print("legacy /v1/completions")
    status, raw, _ = post(f"{base}/v1/completions", {"model": "cosmos3-nano-reasoner", "prompt": "hi"})
    body = json.loads(raw) if raw else {}
    check("200 legacy", status == 200, raw[:200])
    check("text_completion shape", body.get("object") == "text_completion", str(body)[:150])
    check("has text", bool(body.get("choices", [{}])[0].get("text")))

    # 6 image handling: small inline passes through, big one gets asset-uploaded
    print("image handling")
    small = base64.b64encode(b"\x89PNG" + b"0" * 500).decode()
    post(
        f"{base}/v1/chat/completions",
        {
            "model": "cosmos3-nano-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{small}"}},
                        {"type": "text", "text": "what is this?"},
                    ],
                }
            ],
        },
    )
    sent = mock_nim.STATE["requests"][-1]
    img_part = [p for p in sent["messages"][0]["content"] if p["type"] == "image_url"][0]
    check("small image stays inline", img_part["image_url"]["url"].startswith("data:image/png;base64,"))

    big = base64.b64encode(b"\x89PNG" + b"0" * (300 * 1024)).decode()
    status, raw, _ = post(
        f"{base}/v1/chat/completions",
        {
            "model": "cosmos3-nano-reasoner",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big}"}},
                        {"type": "text", "text": "what is this?"},
                    ],
                }
            ],
        },
    )
    sent = mock_nim.STATE["requests"][-1]
    img_part = [p for p in sent["messages"][0]["content"] if p["type"] == "image_url"][0]
    check("big image became asset ref", ";asset_id," in img_part["image_url"]["url"], img_part["image_url"]["url"][:80])
    check("asset upload actually hit S3 mock", len(mock_nim.STATE["uploads"]) == 1, str(mock_nim.STATE["uploads"]))

    # 7 video passthrough + fps
    print("video")
    post(
        f"{base}/v1/chat/completions",
        {
            "model": "cosmos3-nano-reasoner",
            "fps": 4,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": "https://example.com/a.mp4"}},
                        {"type": "text", "text": "what happens next?"},
                    ],
                }
            ],
        },
    )
    sent = mock_nim.STATE["requests"][-1]
    check("video part forwarded", any(p["type"] == "video_url" for p in sent["messages"][0]["content"]))
    check("fps forwarded", sent.get("frames_per_second") == 4, str(sent.get("frames_per_second")))

    # 8 upstream error passthrough (422 for a bad role injected directly)
    print("error handling")
    gw2 = start_gateway(base_up, upstream_keys=[])
    base2 = f"http://127.0.0.1:{gw2.server_port}"
    status, raw, _ = post(f"{base2}/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    check("missing upstream key surfaces 403", status == 403, f"{status} {raw[:120]}")

    # 9 client auth
    gw3 = start_gateway(base_up, client_keys=["sk-local-1"])
    base3 = f"http://127.0.0.1:{gw3.server_port}"
    status, raw, _ = post(f"{base3}/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    check("auth required -> 401", status == 401, f"{status} {raw[:100]}")
    status, raw, _ = post(
        f"{base3}/v1/chat/completions",
        {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        {"Authorization": "Bearer sk-local-1"},
    )
    check("valid key accepted", status == 200, f"{status} {raw[:120]}")

    print(f"\n{PASSES} passed, {len(FAILS)} failed")
    if FAILS:
        print("failed:", ", ".join(FAILS))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
