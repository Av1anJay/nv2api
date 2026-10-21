"""Mock NGC backend — authn token exchange + buildapi v3 predict.

Lets the NGC path be tested end to end without an NGC API key. Mirrors the real
contract as observed:

  GET/POST /token (Basic auth, user must be "$oauthtoken")
      -> {"token": "<jwt with exp>"}      bad key -> 401
  POST /v3/protected/predict/stateless/<model>
      no/invalid JWT -> 403 "RBAC: access denied"
      JWT that is not JWT-shaped -> 401 "Jwt is not in the form of ..."
      missing nv-function-id -> 400 "Required header 'nv-function-id' is not present."
      wrong function id -> 403 "RBAC: access denied"
  GET  /<publisher>/<model>              -> HTML containing nvcfFunctionId
"""

from __future__ import annotations

import base64
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD_NGC_KEY = "nvapi-good-ngc-key"
FUNCTION_ID = "24a385c8-111a-4294-8c73-cf7575cc2595"
MODEL = "nvidia/cosmos3-nano-reasoner"
STATE = {"requests": [], "tokens": 0}


def _jwt(exp_offset: int = 3600) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'RS256', 'typ': 'JWT'})}.{seg({'sub': 'mock', 'exp': int(time.time()) + exp_offset})}.sig"


class MockNgcHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _token(self):
        """NGC authn: Basic $oauthtoken:<api-key> -> JWT. Works for GET and POST."""
        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Basic "):
            self._json(401, {"detail": "unauthorized"})
            return
        try:
            user, _, password = base64.b64decode(auth[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001
            self._json(401, {"detail": "bad basic auth"})
            return
        if user != "$oauthtoken" or password != GOOD_NGC_KEY:
            self._json(401, {"detail": "bad credentials"})
            return
        STATE["tokens"] += 1
        self._json(200, {"token": _jwt()})

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/token"):
            self._token()
            return
        if self.path.startswith("/nvidia/") or self.path.startswith("/someorg/"):
            html = (
                "<html><script>self.__next_f.push([1,\"{\\\"namespace\\\":\\\"qc69jvmznzxy\\\","
                f"\\\"nvcfFunctionId\\\":\\\"{FUNCTION_ID}\\\"}}\"])</script></html>"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        self._json(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)

        if self.path.startswith("/token"):
            self._token()
            return

        if "/predict/stateless/" in self.path:
            model = self.path.rsplit("/predict/stateless/", 1)[-1]
            auth = self.headers.get("Authorization") or ""
            if not auth.startswith("Bearer "):
                self._json(403, {"detail": "RBAC: access denied"})
                return
            jwt = auth[7:]
            if jwt.count(".") != 2:
                self._json(
                    401,
                    {"detail": "Jwt is not in the form of Header.Payload.Signature with two dots and 3 sections"},
                )
                return
            fid = self.headers.get("nv-function-id") or ""
            if fid != FUNCTION_ID:
                if not fid:
                    self._json(400, {"detail": "Required header 'nv-function-id' is not present."})
                else:
                    self._json(403, {"detail": "RBAC: access denied"})
                return
            body = json.loads(raw or b"{}")
            STATE["requests"].append({"model": model, "body": body, "path": self.path})
            answer = "<think>ok</think>NGC backend speaking."
            if body.get("stream"):
                self._stream(model, answer)
                return
            self._json(
                200,
                {
                    "id": "chatcmpl-ngc",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
            return

        self._json(404, {"detail": "not found"})

    def _stream(self, model, answer):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        for i in range(0, len(answer), 6):
            evt = {
                "id": "chatcmpl-ngc",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": answer[i : i + 6]}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start(port: int = 0) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), MockNgcHandler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
