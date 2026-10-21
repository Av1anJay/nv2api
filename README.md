# nv2api

OpenAI-compatible gateway in front of NVIDIA NIM endpoints, built around
`nvidia/cosmos3-nano-reasoner` first — a video/image reasoning model whose NIM
schema is *almost* OpenAI but not quite.

Pure Python stdlib. No fastapi, no uvicorn, no httpx, no venv to maintain.

## Why it exists

The cosmos3 NIM endpoint diverges from `POST /v1/chat/completions` in ways that
break ordinary OpenAI clients:

| OpenAI thing | what NIM does | what nv2api does |
| --- | --- | --- |
| `role: "system"` / `"developer"` | rejected — only `user`/`assistant` allowed | folds system text into the first user turn |
| `content: "string"` | allowed, but parts are richer | normalises to `[{type:text,...}]` |
| `image_url` part | `data:`/URL only up to 180 KB, above that needs an NVCF asset | auto-uploads big images via `/v2/nvcf/assets`, rewrites to `asset_id` ref |
| `video_url` | supported, needs frame rate | forwards `fps` → `frames_per_second` (2–8) |
| `tools` / `response_format` / `n` | unsupported (no function calling) | dropped, reported in `nv2api_warnings` |
| `max_tokens` | hard cap 4096 | clamps and warns |
| reasoning | inlined as `<think>…</think>` in `content` | split into `reasoning_content` (streamed incrementally, tag-safe) |
| `/v1/completions` | absent | emulated on top of chat |
| many `nvapi-` keys | one key per request | round-robin rotation + retry on 429/5xx |

## Quick start

```sh
git clone https://github.com/AvianJay/nv2api && cd nv2api
export NVIDIA_API_KEY=nvapi-xxxxxxxx        # one key, or comma-separated pool
python3 -m nv2api                            # listens on 127.0.0.1:8200
```

```sh
curl http://127.0.0.1:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cosmos3-nano-reasoner",
    "messages": [{"role": "user", "content": [
      {"type": "video_url", "video_url": {"url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos3/agibot.mp4"}},
      {"type": "text", "text": "What can be the next immediate action?"}
    ]}]
  }'
```

Any OpenAI SDK works:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8200/v1", api_key="anything")
r = client.chat.completions.create(model="cosmos3-nano-reasoner", messages=[...])
print(r.choices[0].message.reasoning_content, r.choices[0].message.content)
```

## Endpoints

- `POST /v1/chat/completions` — streaming and non-streaming
- `POST /v1/completions` — legacy text completion, mapped onto chat
- `GET /v1/models` — upstream list plus the configured aliases
- `GET /healthz` — liveness + which upstream/key count is loaded
- `OPTIONS *` — CORS preflight (browser frontends work)

## Configuration

Environment (all optional except the key):

| var | default | meaning |
| --- | --- | --- |
| `NVIDIA_API_KEY` | – | `nvapi-` key(s), comma-separated for a pool |
| `NGC_API_KEY` | – | NGC API key(s) — unlocks the buildapi route for preview models |
| `N2A_BACKEND` | `auto` | `auto` (integrate → buildapi on 404), `integrate`, or `ngc` |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | upstream |
| `N2A_HOST` / `N2A_PORT` | `127.0.0.1` / `8200` | listen address |
| `N2A_API_KEYS` | – | if set, clients must present one of these |
| `N2A_DEFAULT_MODEL` | `nvidia/cosmos3-nano-reasoner` | used when the request omits `model` |
| `N2A_EXTRA_MODELS` | – | extra ids advertised by `/v1/models` |
| `N2A_MAX_TOKENS_CAP` | `4096` | clamp (0 disables) |
| `N2A_SPLIT_THINK` | `true` | `<think>` → `reasoning_content` |
| `N2A_DROP_UNSUPPORTED` | `true` | strip tools/response_format/n |
| `N2A_UPSTREAM_TIMEOUT` | `600` | video reasoning is slow |
| `N2A_UPSTREAM_RETRIES` | `2` | retries on 429/5xx |
| `N2A_ASSET_API` | `https://api.nvcf.nvidia.com/v2/nvcf/assets` | asset upload host |

Or a JSON file: `python3 -m nv2api -c config.json` (see `config.example.json`).

## Running it for real

```sh
# detached, survives ssh logout
screen -dmS nv2api sh -c 'NVIDIA_API_KEY=... python3 -m nv2api >> ~/nv2api.log 2>&1'

# or systemd --user
mkdir -p ~/.config/systemd/user
cp deploy/nv2api.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now nv2api
```

The default bind is `127.0.0.1`. If you expose it, set `N2A_API_KEYS` — an open
gateway burns your NVIDIA quota for anyone who finds it.

## Tests

```sh
python3 tests/test_e2e.py    # 33 assertions: OpenAI surface + translation
python3 tests/test_ngc.py    # 17 assertions: NGC token exchange + buildapi route
```

Both run over real sockets against `tests/mock_nim.py` and `tests/mock_ngc.py`,
which mimic the upstream contracts strictly (reject system roles, unknown
fields, `max_tokens > 4096`, non-alternating roles; require a JWT-shaped bearer
plus `nv-function-id`) — so a green run means the translation layer is right,
not just that the code runs.

## Status / known limits

### Two different NVIDIA surfaces

`integrate.api.nvidia.com/v1` genuinely does **not** carry cosmos3-nano-reasoner
(it answers `404 page not found` while GA models on the same key answer `403`).
The build.nvidia.com playground is not a way around that — it posts to
`buildapi.ngc.nvidia.com/v3/predict/stateless/<model>` with an invisible
hCaptcha token (no token → `400 Captcha required`, bogus token → `400 Token is
invalid`).

That host has an authenticated, **captcha-free** sibling, and nv2api speaks it:

```
POST https://buildapi.ngc.nvidia.com/v3/protected/predict/stateless/<model>
  Authorization: Bearer <JWT from authn.nvidia.com>
  nv-function-id: <uuid, scraped from the model page>
```

`backend: "auto"` (default) calls `integrate.api` first and transparently
retries through buildapi when the answer is `404` — so a preview model just
works, without the client knowing which door it went through (`nv2api_backend`
in the response says which). Set `NGC_API_KEY` (free, from
ngc.nvidia.com → Setup → API Keys) to enable it.

Verified states of the buildapi route, all reproduced in `tests/test_ngc.py`:

| request | answer |
| --- | --- |
| no `Authorization` | `403 RBAC: access denied` |
| non-JWT bearer | `401 Jwt is not in the form of Header.Payload.Signature` |
| JWT, no `nv-function-id` | `400 Required header 'nv-function-id' is not present.` |
| unsigned/captcha path | `400 Captcha required` |

Remaining limits:

- `nvidia/cosmos3-nano-reasoner` is still **preview**: it is absent from the
  public `/v1/models` list, and `integrate.api` 404s it. The NGC route is the
  supported way in; `python3 -m nv2api --doctor` tells you which of the two
  you are hitting and why.
- No function calling: the endpoint genuinely has none, so tools are dropped
  rather than faked.
- NVCF asset uploads are best-effort; on failure the inline data URL is sent
  and a warning is attached.
- Single process, no request queue: throughput is NVIDIA's rate limit, not us.
- The captcha path is deliberately **not** implemented — solving an hCaptcha is
  bot-protection circumvention, and the authenticated route exists.
