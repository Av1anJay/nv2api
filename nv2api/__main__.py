"""CLI entry point: `python -m nv2api` or `nv2api`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config
from .server import serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nv2api", description="OpenAI-compatible gateway for NVIDIA NIM")
    ap.add_argument("-c", "--config", default=None, help="path to config.json")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--upstream", default=None, help="upstream base url, e.g. https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default=None, help="default model id")
    ap.add_argument("--doctor", action="store_true", help="probe the upstream with this key and exit")
    ap.add_argument("--version", action="version", version=f"nv2api {__version__}")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.host:
        cfg.host = args.host
    if args.port:
        cfg.port = args.port
    if args.upstream:
        cfg.upstream_base = args.upstream.rstrip("/")
    if args.model:
        cfg.default_model = args.model
    if args.doctor:
        from .doctor import main as doctor_main

        return doctor_main(cfg)
    serve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
