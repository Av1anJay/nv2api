"""Translate OpenAI-shaped requests into what NVIDIA NIM actually accepts,
and translate NIM responses back into the OpenAI shape.

Verified against the OpenAPI spec published on
https://build.nvidia.com/nvidia/cosmos3-nano-reasoner :

  NIMLLMChatCompletionRequest:
    messages  : [{role: "user"|"assistant", content: str | [part, ...]}]
                NOTE: only user/assistant are allowed; system must be folded in.
    model     : string
    temperature, top_p, max_tokens (<=4096), presence_penalty,
    repetition_penalty, seed, stream, frames_per_second (2..8, default 4)
    content parts: text | image_url{url} | video_url{url}
    No tools / response_format / n / stop.

Big images (>180 KB) must be uploaded through the NVCF asset API and referenced
as  <img src="data:image/png;asset_id,<uuid>" />  — handled in assets.py.
"""

from __future__ import annotations

import base64
import json
import re
import time
import uuid
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)

# Fields NIM rejects outright; forwarded only when we know they are safe.
SAFE_PASSTHROUGH = (
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "stop",
)

# Extra fields some clients send that the cosmos spec does not have.
MEDIA_KWARGS = {"frames_per_second": 8, "fps": 8}


def _image_url_from_part(part: dict[str, Any]) -> str | None:
    img = part.get("image_url")
    if isinstance(img, dict):
        return img.get("url")
    if isinstance(img, str):
        return img
    return None


def _video_url_from_part(part: dict[str, Any]) -> str | None:
    vid = part.get("video_url")
    if isinstance(vid, dict):
        return vid.get("url")
    if isinstance(vid, str):
        return vid
    # OpenAI-style passthrough from some clients
    if part.get("type") == "video":
        return part.get("url") or (part.get("video") or {}).get("url")
    return None


def normalize_messages(
    messages: list[dict[str, Any]],
    *,
    asset_resolver=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (nim_messages, warnings).

    - system/developer messages are folded into the first user message
    - OpenAI content-part lists are mapped onto NIM content parts
    - remote/data URLs for images >180KB go through asset_resolver when given
    """
    warnings: list[str] = []
    system_chunks: list[str] = []
    out: list[dict[str, Any]] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")

        if role in ("system", "developer"):
            if isinstance(content, str):
                system_chunks.append(content)
            elif isinstance(content, list):
                system_chunks.extend(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            continue

        if role not in ("user", "assistant"):
            # tool/function results cannot be expressed in the cosmos schema
            warnings.append(f"message role '{role}' dropped (NIM accepts user/assistant only)")
            continue

        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("text", "input_text"):
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif ptype in ("image_url", "input_image", "image"):
                    url = _image_url_from_part(part) or part.get("url")
                    if url and asset_resolver is not None:
                        url = asset_resolver(url, warnings)
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                elif ptype in ("video_url", "input_video", "video"):
                    url = _video_url_from_part(part) or part.get("url")
                    if url:
                        parts.append({"type": "video_url", "video_url": {"url": url}})
        elif content is None:
            parts.append({"type": "text", "text": ""})

        out.append({"role": role, "content": parts})

    if system_chunks and out:
        prefix = "\n\n".join(c.strip() for c in system_chunks if c and c.strip())
        if prefix:
            first = out[0]
            first["content"] = [{"type": "text", "text": prefix}] + first["content"]
    elif system_chunks:
        out.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": "\n\n".join(system_chunks)}],
            }
        )

    # NIM wants alternating roles; merge consecutive same-role turns.
    merged: list[dict[str, Any]] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"].extend(m["content"])
        else:
            merged.append(m)
    return merged, warnings


def build_upstream_request(
    body: dict[str, Any],
    model: str,
    *,
    max_tokens_cap: int = 4096,
    drop_unsupported: bool = True,
    asset_resolver=None,
) -> tuple[dict[str, Any], list[str]]:
    """OpenAI chat.completions body -> NIM body."""
    warnings: list[str] = []
    messages, w = normalize_messages(body.get("messages") or [], asset_resolver=asset_resolver)
    warnings.extend(w)

    payload: dict[str, Any] = {"model": model, "messages": messages}

    for key in SAFE_PASSTHROUGH:
        if key in body and body[key] is not None:
            payload[key] = body[key]

    # OpenAI alias for max_tokens
    if "max_completion_tokens" in body and body["max_completion_tokens"] is not None:
        payload.setdefault("max_tokens", body["max_completion_tokens"])

    if max_tokens_cap:
        current = payload.get("max_tokens")
        if current is None:
            payload["max_tokens"] = max_tokens_cap
        elif isinstance(current, int) and current > max_tokens_cap:
            warnings.append(f"max_tokens {current} clamped to {max_tokens_cap} (NIM limit)")
            payload["max_tokens"] = max_tokens_cap

    # media kwargs: fps for video sampling
    fps = body.get("frames_per_second") or body.get("fps")
    if isinstance(fps, int) and 2 <= fps <= 8:
        payload["frames_per_second"] = fps

    payload["stream"] = bool(body.get("stream"))

    if drop_unsupported:
        for key in ("tools", "tool_choice", "functions", "function_call", "response_format"):
            if body.get(key):
                warnings.append(f"'{key}' dropped: this NIM endpoint does not support function calling")
        if isinstance(body.get("n"), int) and body["n"] > 1:
            warnings.append("'n' dropped: NIM returns a single choice")

    return payload, warnings


def split_think(text: str) -> tuple[str, str | None]:
    """'<think>r</think>answer' -> ('answer', 'r')."""
    if not text:
        return text, None
    found = THINK_RE.findall(text)
    if not found:
        return text, None
    reasoning = "\n\n".join(f.strip() for f in found).strip()
    answer = THINK_RE.sub("", text).strip()
    return answer, reasoning or None


def openai_completion(
    upstream: dict[str, Any],
    model: str,
    *,
    split_reasoning: bool = True,
) -> dict[str, Any]:
    """NIM chat.completion -> OpenAI chat.completion (with reasoning_content)."""
    choice = (upstream.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if split_reasoning and content:
        content, parsed = split_think(content)
        reasoning = reasoning or parsed

    out_message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning:
        out_message["reasoning_content"] = reasoning

    usage = upstream.get("usage") or {}
    return {
        "id": upstream.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": upstream.get("created") or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": out_message,
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0)
            or (usage.get("prompt_tokens", 0) + (usage.get("completion_tokens", 0) or 0)),
        },
    }


class ThinkStreamSplitter:
    """Incrementally splits <think>…</think> out of a streamed answer.

    Holds back a small tail so a tag split across SSE events is not emitted as
    literal text.
    """

    def __init__(self) -> None:
        self.in_think = False
        self.buf = ""
        self.pending = ""  # text we are unsure about (partial tag)

    def feed(self, text: str) -> tuple[str, str]:
        """Returns (reasoning_delta, content_delta)."""
        self.buf += text
        reasoning_out = ""
        content_out = ""
        while self.buf:
            if not self.in_think:
                idx = self.buf.lower().find("<think>")
                if idx == -1:
                    # maybe a partial "<thi" tail: hold it back
                    hold = self._partial_tag(self.buf, "<think>")
                    if hold:
                        content_out += self.buf[: len(self.buf) - hold]
                        self.buf = self.buf[len(self.buf) - hold :]
                    else:
                        content_out += self.buf
                        self.buf = ""
                    break
                content_out += self.buf[:idx]
                self.buf = self.buf[idx + len("<think>") :]
                self.in_think = True
            else:
                idx = self.buf.lower().find("</think>")
                if idx == -1:
                    hold = self._partial_tag(self.buf, "</think>")
                    if hold:
                        reasoning_out += self.buf[: len(self.buf) - hold]
                        self.buf = self.buf[len(self.buf) - hold :]
                    else:
                        reasoning_out += self.buf
                        self.buf = ""
                    break
                reasoning_out += self.buf[:idx]
                self.buf = self.buf[idx + len("</think>") :]
                self.in_think = False
        return reasoning_out, content_out

    def flush(self) -> tuple[str, str]:
        if self.in_think:
            out = (self.buf, "")
        else:
            out = ("", self.buf)
        self.buf = ""
        return out

    @staticmethod
    def _partial_tag(text: str, tag: str) -> int:
        """How many trailing chars could be the start of `tag`."""
        limit = min(len(text), len(tag) - 1)
        lowered = text.lower()
        for size in range(limit, 0, -1):
            if tag.startswith(lowered[-size:]):
                return size
        return 0


def data_url_size(url: str) -> int:
    """Approximate decoded byte size of a data: URL (0 for http(s) URLs)."""
    if not url.startswith("data:"):
        return 0
    try:
        header, _, b64 = url.partition(",")
        if ";base64" not in header:
            return len(b64)
        return len(base64.b64decode(b64, validate=False))
    except Exception:
        return 0


def sse(event: dict[str, Any] | str) -> bytes:
    if isinstance(event, str):
        return f"data: {event}\n\n".encode()
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
