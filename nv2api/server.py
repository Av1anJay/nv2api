"""HTTP server: OpenAI-compatible surface, stdlib only."""

from __future__ import annotations

import http.client
import json
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import __version__, ngc
from .assets import make_resolver
from .config import Config
from .translate import (
    ThinkStreamSplitter,
    build_upstream_request,
    openai_completion,
    sse,
)

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "authorization, content-type, x-api-key, anthropic-version",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


class KeyRotator:
    def __init__(self, keys: list[str]) -> None:
        self.keys = [k for k in keys if k]
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> str | None:
        if not self.keys:
            return None
        with self._lock:
            key = self.keys[self._i % len(self.keys)]
            self._i += 1
            return key

    def __bool__(self) -> bool:
        return bool(self.keys)


class Upstream:
    """Talks to the NIM API with retries; streams via http.client."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.keys = KeyRotator(cfg.upstream_keys)
        self.ngc_keys = KeyRotator(cfg.ngc_keys)
        self._tokens = ngc.TokenCache()
        parsed = urllib.parse.urlsplit(cfg.upstream_base)
        self.scheme = parsed.scheme or "https"
        self.host = parsed.netloc
        self.prefix = parsed.path.rstrip("/")
        self.ssl_ctx = ssl.create_default_context()

    # ---------- NGC buildapi backend ----------
    def _ngc_jwt(self) -> str:
        key = self.ngc_keys.next()
        if not key:
            raise RuntimeError("no NGC_API_KEY configured")
        return self._tokens.get(key)

    def ngc_post_json(self, model: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST through buildapi.ngc.nvidia.com (captcha-free, needs NGC key)."""
        try:
            jwt = self._ngc_jwt()
        except Exception as exc:  # noqa: BLE001
            return 502, _error_body(f"NGC token exchange failed: {exc}", 502)
        fid = ngc.discover_function_id(model)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{ngc._buildapi()}{ngc.predict_path(model, authenticated=True)}",
            data=body,
            headers=ngc.build_headers(jwt, fid, stream=False),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.upstream_timeout, context=self.ssl_ctx) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, _error_body(exc.read().decode(errors="replace"), exc.code)
        except Exception as exc:  # noqa: BLE001
            return 502, _error_body(str(exc), 502)

    def ngc_open_stream(self, model: str, payload: dict[str, Any]):
        jwt = self._ngc_jwt()
        fid = ngc.discover_function_id(model)
        headers = ngc.build_headers(jwt, fid, stream=True)
        parsed = urllib.parse.urlsplit(ngc._buildapi())
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(
                parsed.netloc, timeout=self.cfg.upstream_timeout, context=self.ssl_ctx
            )
        else:
            conn = http.client.HTTPConnection(parsed.netloc, timeout=self.cfg.upstream_timeout)
        conn.request(
            "POST",
            f"{parsed.path.rstrip('/')}{ngc.predict_path(model, authenticated=True)}",
            body=json.dumps(payload),
            headers=headers,
        )
        return conn, conn.getresponse()

    # ---------- non-streaming ----------
    def post_json(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        last_err: Exception | None = None
        for attempt in range(self.cfg.upstream_retries + 1):
            key = self.keys.next()
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = urllib.request.Request(
                f"{self.cfg.upstream_base}{path}",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    req,
                    timeout=self.cfg.upstream_timeout,
                    context=self.ssl_ctx,
                ) as resp:
                    return resp.status, json.loads(resp.read().decode() or "{}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.cfg.upstream_retries:
                    last_err = exc
                    time.sleep(self.cfg.upstream_retry_backoff * (attempt + 1))
                    continue
                return exc.code, _error_body(body, exc.code)
            except Exception as exc:  # noqa: BLE001 - network errors
                last_err = exc
                if attempt < self.cfg.upstream_retries:
                    time.sleep(self.cfg.upstream_retry_backoff * (attempt + 1))
                    continue
                return 502, _error_body(str(exc), 502)
        return 502, _error_body(str(last_err), 502)

    # ---------- streaming ----------
    def open_stream(self, path: str, payload: dict[str, Any]):
        key = self.keys.next()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        kwargs = {"timeout": self.cfg.upstream_timeout}
        if self.scheme == "https":
            kwargs["context"] = self.ssl_ctx
        conn = cls(self.host, **kwargs)
        conn.request("POST", f"{self.prefix}{path}", body=json.dumps(payload), headers=headers)
        return conn, conn.getresponse()

    def get_models(self) -> tuple[int, dict[str, Any]]:
        key = self.keys.next()
        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.cfg.upstream_base}/models", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30, context=self.ssl_ctx) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, _error_body(exc.read().decode(errors="replace"), exc.code)
        except Exception as exc:  # noqa: BLE001
            return 502, _error_body(str(exc), 502)


def _error_body(raw: str, code: int) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass
    return {
        "error": {
            "message": raw.strip()[:800] or f"upstream returned {code}",
            "type": "upstream_error",
            "code": code,
        }
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"nv2api/{__version__}"
    protocol_version = "HTTP/1.1"

    cfg: Config
    upstream: Upstream
    _models_cache: tuple[float, list[str]] = (0.0, [])

    # ---------------- helpers ----------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.cfg.log_requests:
            sys.stderr.write("[nv2api] %s %s\n" % (self.address_string(), fmt % args))

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, message: str, etype: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": message, "type": etype, "code": code}})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw.decode())
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _authorized(self) -> bool:
        if not self.cfg.client_keys:
            return True
        header = self.headers.get("Authorization") or ""
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        token = token or (self.headers.get("x-api-key") or "").strip()
        return token in self.cfg.client_keys

    # ---------------- verbs ----------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        for k, v in CORS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/healthz", "/health", "/"):
            self._json(
                200,
                {
                    "status": "ok",
                    "version": __version__,
                    "upstream": self.cfg.upstream_base,
                    "backend": self.cfg.backend,
                    "ngc_keys": len(self.cfg.ngc_keys),
                    "default_model": self.cfg.default_model,
                    "upstream_keys": len(self.cfg.upstream_keys),
                    "auth_required": bool(self.cfg.client_keys),
                },
            )
            return
        if path in ("/v1/models", "/models"):
            if not self._authorized():
                self._error(401, "invalid api key", "authentication_error")
                return
            self._json(200, {"object": "list", "data": self._model_list()})
            return
        self._error(404, f"no route for GET {path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._chat()
            return
        if path in ("/v1/completions", "/completions"):
            self._completion()
            return
        self._error(404, f"no route for POST {path}", "not_found")

    # ---------------- routes ----------------
    def _model_list(self) -> list[dict[str, Any]]:
        now = time.time()
        cached_at, cached = Handler._models_cache
        if cached and now - cached_at < self.cfg.models_ttl:
            names = cached
        else:
            names = []
            status, data = self.upstream.get_models()
            if status == 200:
                names = [m.get("id") for m in data.get("data", []) if m.get("id")]
            for extra in self.cfg.public_models():
                if extra not in names:
                    names.append(extra)
            if names:
                Handler._models_cache = (now, names)
        return [
            {"id": n, "object": "model", "created": 0, "owned_by": n.split("/")[0] if "/" in n else "nvidia"}
            for n in names
        ]

    def _chat(self) -> None:
        if not self._authorized():
            self._error(401, "invalid api key", "authentication_error")
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._error(400, str(exc))
            return

        model = self.cfg.resolve_model(body.get("model"))
        api_key = self.upstream.keys.next() if self.upstream.keys else None
        payload, warnings = build_upstream_request(
            body,
            model,
            max_tokens_cap=self.cfg.max_tokens_cap,
            drop_unsupported=self.cfg.drop_unsupported,
            asset_resolver=make_resolver(api_key),
        )
        if warnings and self.cfg.log_requests:
            sys.stderr.write(f"[nv2api] {model}: " + "; ".join(warnings) + "\n")

        if payload.get("stream"):
            self._chat_stream(payload, model, warnings)
            return

        status, data, via = self._post_chat(payload, model)
        if status != 200:
            self._json(status, data if "error" in data else {"error": data})
            return
        result = openai_completion(data, model, split_reasoning=self.cfg.split_think)
        if warnings:
            result["nv2api_warnings"] = warnings
        if via != "integrate":
            result["nv2api_backend"] = via
        self._json(200, result)

    # ---------- backend selection ----------
    def _post_chat(self, payload: dict[str, Any], model: str) -> tuple[int, dict[str, Any], str]:
        """Try the OpenAI surface, then buildapi. Returns (status, body, backend)."""
        mode = self.cfg.backend
        if mode == "ngc":
            status, data = self.upstream.ngc_post_json(model, payload)
            return status, data, "ngc"
        status, data = self.upstream.post_json("/chat/completions", payload)
        # 404 = "page not found": integrate.api has no route for this model at
        # all. Preview endpoints live behind buildapi instead.
        if mode == "auto" and status == 404 and self.cfg.ngc_keys:
            nstatus, ndata = self.upstream.ngc_post_json(model, payload)
            return nstatus, ndata, "ngc"
        return status, data, "integrate"

    def _completion(self) -> None:
        """Legacy /v1/completions -> chat, so old clients keep working."""
        if not self._authorized():
            self._error(401, "invalid api key", "authentication_error")
            return
        try:
            body = self._body()
        except ValueError as exc:
            self._error(400, str(exc))
            return
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        chat_body = {
            "model": body.get("model"),
            "messages": [{"role": "user", "content": prompt or ""}],
            "stream": body.get("stream", False),
        }
        for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
            if key in body:
                chat_body[key] = body[key]
        model = self.cfg.resolve_model(chat_body.get("model"))
        payload, _ = build_upstream_request(
            chat_body,
            model,
            max_tokens_cap=self.cfg.max_tokens_cap,
            asset_resolver=make_resolver(self.upstream.keys.next() if self.upstream.keys else None),
        )
        status, data, _via = self._post_chat(payload, model)
        if status != 200:
            self._json(status, data if "error" in data else {"error": data})
            return
        chat = openai_completion(data, model, split_reasoning=self.cfg.split_think)
        text = chat["choices"][0]["message"].get("content", "")
        self._json(
            200,
            {
                "id": chat["id"],
                "object": "text_completion",
                "created": chat["created"],
                "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": chat["usage"],
            },
        )

    def _chat_stream(self, payload: dict[str, Any], model: str, warnings: list[str]) -> None:
        mode = self.cfg.backend
        try:
            if mode == "ngc":
                conn, resp = self.upstream.ngc_open_stream(model, payload)
            else:
                conn, resp = self.upstream.open_stream("/chat/completions", payload)
                if mode == "auto" and resp.status == 404 and self.cfg.ngc_keys:
                    # integrate.api has no route for this preview model
                    try:
                        resp.read()
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    conn, resp = self.upstream.ngc_open_stream(model, payload)
        except Exception as exc:  # noqa: BLE001
            self._error(502, f"upstream connection failed: {exc}", "upstream_error")
            return

        if resp.status != 200:
            raw = resp.read().decode(errors="replace")
            conn.close()
            self._json(resp.status, _error_body(raw, resp.status))
            return

        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        splitter = ThinkStreamSplitter() if self.cfg.split_think else None

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # HTTP/1.1 + no length: the only way a client sees EOF is a closed socket.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.close_connection = True
        for k, v in CORS.items():
            self.send_header(k, v)
        self.end_headers()

        def chunk(delta: dict[str, Any], finish: str | None = None, usage: dict[str, Any] | None = None):
            event: dict[str, Any] = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                event["usage"] = usage
            self.wfile.write(sse(event))
            self.wfile.flush()

        try:
            chunk({"role": "assistant", "content": ""})
            for raw_line in resp:
                line = raw_line.decode(errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:  # noqa: BLE001
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if isinstance(text, list):
                    text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                finish = choice.get("finish_reason")
                usage = obj.get("usage")

                if text and splitter is not None:
                    r_out, c_out = splitter.feed(text)
                    if r_out:
                        chunk({"reasoning_content": r_out})
                    if c_out:
                        chunk({"content": c_out})
                elif text:
                    chunk({"content": text})
                elif reasoning:
                    chunk({"reasoning_content": reasoning})

                if finish:
                    if splitter is not None:
                        r_tail, c_tail = splitter.flush()
                        if r_tail:
                            chunk({"reasoning_content": r_tail})
                        if c_tail:
                            chunk({"content": c_tail})
                    chunk({}, finish=finish, usage=usage)
            self.wfile.write(sse("[DONE]"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                resp.close()
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def serve(cfg: Config) -> None:
    Handler.cfg = cfg
    Handler.upstream = Upstream(cfg)
    if not cfg.upstream_keys:
        sys.stderr.write("[nv2api] WARNING: no NVIDIA_API_KEY set — upstream calls will 403\n")
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    httpd.daemon_threads = True
    sys.stderr.write(
        f"[nv2api] listening on http://{cfg.host}:{cfg.port}/v1  "
        f"(upstream {cfg.upstream_base}, default model {cfg.default_model})\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[nv2api] shutting down\n")
    finally:
        httpd.server_close()
