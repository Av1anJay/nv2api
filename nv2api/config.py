"""Configuration: env vars first, optional JSON file second."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str) -> list[str]:
    raw = _env(name)
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Config:
    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8200

    # --- upstream ---
    upstream_base: str = "https://integrate.api.nvidia.com/v1"
    # One or more nvapi- keys. Multiple keys are rotated round-robin.
    upstream_keys: list[str] = field(default_factory=list)
    upstream_timeout: int = 600          # seconds; video reasoning is slow
    upstream_connect_timeout: int = 20
    upstream_retries: int = 2            # retries on 429/5xx
    upstream_retry_backoff: float = 1.5  # seconds, multiplied per attempt

    # --- client auth (optional) ---
    # If empty, the gateway accepts any (or no) bearer token. Bind to localhost!
    client_keys: list[str] = field(default_factory=list)

    # --- models ---
    default_model: str = "nvidia/cosmos3-nano-reasoner"
    # Alias -> upstream model id. Aliases are what clients may ask for.
    model_aliases: dict[str, str] = field(default_factory=dict)
    # Models advertised by GET /v1/models even when upstream omits them.
    extra_models: list[str] = field(default_factory=list)
    models_ttl: int = 300
    max_tokens_cap: int = 4096           # 0 disables clamping

    # --- behaviour ---
    split_think: bool = True             # <think>...</think> -> reasoning_content
    drop_unsupported: bool = True        # strip tool/function fields NIM rejects
    log_requests: bool = True

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        # env overrides
        cfg.host = _env("N2A_HOST", cfg.host)
        cfg.port = _env_int("N2A_PORT", cfg.port)
        cfg.upstream_base = _env("NVIDIA_BASE_URL", cfg.upstream_base).rstrip("/")
        if keys := _env_list("NVIDIA_API_KEY"):
            cfg.upstream_keys = keys
        cfg.upstream_timeout = _env_int("N2A_UPSTREAM_TIMEOUT", cfg.upstream_timeout)
        cfg.upstream_retries = _env_int("N2A_UPSTREAM_RETRIES", cfg.upstream_retries)
        if client := _env_list("N2A_API_KEYS"):
            cfg.client_keys = client
        cfg.default_model = _env("N2A_DEFAULT_MODEL", cfg.default_model)
        cfg.max_tokens_cap = _env_int("N2A_MAX_TOKENS_CAP", cfg.max_tokens_cap)
        cfg.split_think = _env_bool("N2A_SPLIT_THINK", cfg.split_think)
        cfg.drop_unsupported = _env_bool("N2A_DROP_UNSUPPORTED", cfg.drop_unsupported)
        cfg.log_requests = _env_bool("N2A_LOG_REQUESTS", cfg.log_requests)
        cfg.models_ttl = _env_int("N2A_MODELS_TTL", cfg.models_ttl)
        if extra := _env_list("N2A_EXTRA_MODELS"):
            cfg.extra_models = extra
        return cfg

    def resolve_model(self, requested: str | None) -> str:
        name = (requested or "").strip() or self.default_model
        if name in self.model_aliases:
            return self.model_aliases[name]
        # "cosmos3-nano-reasoner" -> "nvidia/cosmos3-nano-reasoner"
        if "/" not in name and name:
            return f"nvidia/{name}"
        return name

    def public_models(self) -> list[str]:
        names: list[str] = list(self.model_aliases.keys())
        names.append(self.default_model)
        names.extend(self.extra_models)
        seen: list[str] = []
        for n in names:
            if n and n not in seen:
                seen.append(n)
        return seen
