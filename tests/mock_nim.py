"""Mock NVIDIA NIM upstream — lets us test nv2api without a real nvapi- key.

Emulates the parts of the documented NIM contract we depend on:
  GET  /v1/models
  POST /v1/chat/completions        (JSON + SSE, <think> reasoning)
  POST /v2/nvcf/assets + PUT url   (NVCF asset upload)
It also validates the request the way NIM does: rejects system roles, unknown
fields, max_tokens > 4096 — so a passing test means our translation is right.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "stream",
    "frames_per_second",
    "stop",
}
ALLOWED_ROLES = {"user", "assistant"}
ALLOWED_PART_TYPES = {"text", "image_url", "video_url"}

STATE = {"uploads": {}, "requests": []}


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "nvidia/cosmos3-nano-reasoner", "object": "model"},
                        {"id": "nvidia/cosmos-reason2-8b", "object": "model"},
                    ],
                },
            )
            return
        self._json(404, {"detail": "not found"})

    def do_PUT(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        data = self.rfile.read(length)
        STATE["uploads"][self.path] = (self.headers.get("Content-Type"), len(data))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/v2/nvcf/assets":
            meta = json.loads(raw or b"{}")
            asset_id = str(uuid.uuid4())
            self._json(
                200,
                {
                    "assetId": asset_id,
                    "uploadUrl": f"http://127.0.0.1:{self.server.server_port}/s3/{asset_id}",
                },
            )
            return
        if self.path != "/v1/chat/completions":
            self._json(404, {"detail": "not found"})
            return

        if not (self.headers.get("Authorization") or "").startswith("Bearer "):
            self._json(403, {"status": 403, "title": "Forbidden", "detail": "Authorization failed"})
            return

        body = json.loads(raw)
        STATE["requests"].append(body)
        problem = _validate(body)
        if problem:
            self._json(422, {"status": 422, "title": "Unprocessable Entity", "detail": problem})
            return

        answer = (
            "<think>The clip shows a robot arm above a basket; "
            "the next action is to descend toward the tape.</think>"
            'Move to the tape at [517, 742], then the basket at [477, 412].'
        )
        if body.get("stream"):
            self._stream(body, answer)
            return

        self._json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1765347170,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 11708, "completion_tokens": 766, "total_tokens": 12474},
            },
        )

    def _stream(self, body, answer):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        # deliberately split tags across events to exercise the splitter
        pieces = [answer[i : i + 7] for i in range(0, len(answer), 7)]
        for piece in pieces:
            evt = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": 1765347170,
                "model": body["model"],
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.002)
        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": 1765347170,
            "model": body["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11708, "completion_tokens": 766, "total_tokens": 12474},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def _validate(body: dict) -> str | None:
    extra = set(body) - ALLOWED_FIELDS
    if extra:
        return f"unexpected fields: {sorted(extra)}"
    if not body.get("model"):
        return "model is required"
    if body.get("max_tokens", 0) > 4096:
        return "max_tokens exceeds 4096"
    if body.get("frames_per_second") and not (2 <= body["frames_per_second"] <= 8):
        return "frames_per_second out of range"
    msgs = body.get("messages") or []
    if not msgs:
        return "messages required"
    for m in msgs:
        if m.get("role") not in ALLOWED_ROLES:
            return f"role '{m.get('role')}' not allowed (user/assistant only)"
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") not in ALLOWED_PART_TYPES:
                    return f"content part type '{part.get('type')}' not allowed"
                if part["type"] == "text" and "text" not in part:
                    return "text part missing text"
                if part["type"] in ("image_url", "video_url") and not part[part["type"]].get("url"):
                    return f"{part['type']} missing url"
    roles = [m["role"] for m in msgs]
    for a, b in zip(roles, roles[1:]):
        if a == b:
            return "roles must alternate"
    return None


def start(port: int = 0) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8299
    srv = start(p)
    print(f"mock NIM on http://127.0.0.1:{srv.server_port}/v1", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.shutdown()
