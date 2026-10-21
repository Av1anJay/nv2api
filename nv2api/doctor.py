"""`nv2api doctor` — prove the key + model actually work before you trust the proxy.

Checks, in order:
  1. GET  {upstream}/models          -> key valid? model listed?
  2. POST {upstream}/chat/completions -> text-only probe (cheap)
  3. POST {upstream}/chat/completions -> image probe (data URL, ~1 KB)
  4. POST {upstream}/chat/completions -> streaming probe
  5. POST {api.nvcf.nvidia.com/v2/nvcf/assets} -> asset-upload entitlement

Every failure prints NVIDIA's own error text, so a 404 ("page not found" =
model not routed for this key) is distinguishable from a 403 (bad key).
"""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request

from .config import Config
from .translate import build_upstream_request

PNG_1PX = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d4944415478da63fcffff3f0300050001a5f645400000000049454e44ae426082"
    )
).decode()


def _req(url: str, key: str, payload: dict | None = None, timeout: int = 90, method: str | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def _verdict(name: str, status: int, raw: str, *, ok_codes=(200,)) -> bool:
    ok = status in ok_codes
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}  (HTTP {status})")
    if not ok:
        print("       " + raw.strip().replace("\n", " ")[:400])
    return ok


def main(cfg: Config) -> int:
    key = cfg.upstream_keys[0] if cfg.upstream_keys else ""
    model = cfg.default_model
    print(f"upstream : {cfg.upstream_base}")
    print(f"model    : {model}")
    print(f"key      : {'set (' + key[:10] + '…)' if key else 'MISSING (export NVIDIA_API_KEY=...)'}\n")

    results: list[bool] = []

    status, raw = _req(f"{cfg.upstream_base}/models", key, timeout=30)
    ok = _verdict("GET /models", status, raw)
    results.append(ok)
    if ok:
        try:
            ids = [m["id"] for m in json.loads(raw).get("data", [])]
            listed = model in ids
            print(f"       catalog lists {len(ids)} models; '{model}' listed: {listed}")
            if not listed:
                print("       (preview models are callable before they are listed)")
        except Exception:  # noqa: BLE001
            pass

    payload, warnings = build_upstream_request(
        {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Say OK."}], "max_tokens": 32},
        model,
        max_tokens_cap=cfg.max_tokens_cap,
    )
    status, raw = _req(f"{cfg.upstream_base}/chat/completions", key, payload)
    ok = _verdict("POST /chat/completions (text)", status, raw)
    results.append(ok)
    if ok:
        print("       " + raw.strip().replace("\n", " ")[:200])

    img_payload, _ = build_upstream_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"}},
                        {"type": "text", "text": "What colour is this pixel?"},
                    ],
                }
            ],
            "max_tokens": 64,
        },
        model,
        max_tokens_cap=cfg.max_tokens_cap,
    )
    status, raw = _req(f"{cfg.upstream_base}/chat/completions", key, img_payload)
    results.append(_verdict("POST /chat/completions (image)", status, raw))

    stream_payload = dict(payload, stream=True)
    status, raw = _req(f"{cfg.upstream_base}/chat/completions", key, stream_payload)
    ok = _verdict("POST /chat/completions (stream)", status, raw)
    results.append(ok)
    if ok:
        print(f"       {raw.count('data:')} SSE events, ends with [DONE]: {'[DONE]' in raw}")

    asset_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    status, raw = _req(asset_url, key, {"contentType": "image/png", "description": "nv2api doctor"}, timeout=30)
    results.append(_verdict("POST /v2/nvcf/assets (upload entitlement)", status, raw, ok_codes=(200, 201)))

    # --- the other door: buildapi.ngc.nvidia.com ---
    if cfg.ngc_keys:
        from . import ngc as ngc_mod

        print("\n-- NGC backend (buildapi, captcha-free) --")
        try:
            jwt, ttl = ngc_mod.exchange_ngc_key(cfg.ngc_keys[0])
            print(f"[PASS] NGC key -> JWT  (valid ~{ttl // 60} min)")
            ngc_ok = True
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] NGC key -> JWT  ({exc})")
            results.append(False)
            ngc_ok = False
        if ngc_ok:
            fid = ngc_mod.discover_function_id(model)
            print(f"[{'PASS' if fid else 'WARN'}] nvcfFunctionId for {model}: {fid}")
            status, raw = ngc_mod.probe_entitlement(cfg.ngc_keys[0], model)
            ok = _verdict("POST buildapi /v3/protected/predict/stateless (text)", status, raw)
            results.append(ok)
    else:
        print("\n(no NGC_API_KEY set — skipping the buildapi route, which is the only")
        print(" one that carries preview models like cosmos3-nano-reasoner)")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    if not results[1]:
        print(
            "\nIf the text probe 404s while /models works, integrate.api has no route to\n"
            "that model (preview gating). Options:\n"
            "  * set NGC_API_KEY=<ngc.nvidia.com API key> to use the captcha-free\n"
            "    buildapi route (backend=auto falls back automatically)\n"
            "  * or N2A_DEFAULT_MODEL=nvidia/cosmos-reason2-8b for a GA model"
        )
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(Config.load()))
