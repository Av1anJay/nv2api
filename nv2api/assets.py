"""NVCF asset upload — needed for images larger than 180 KB.

Flow (docs.api.nvidia.com/cloud-functions/reference/createasset):
  POST https://api.nvcf.nvidia.com/v2/nvcf/assets
       {"contentType": "image/png", "description": "..."}
    -> {"assetId": "<uuid>", "uploadUrl": "<presigned s3 put url>"}
  PUT <uploadUrl>  (raw bytes, Content-Type must match)
Then reference the asset from chat content as
  <img src="data:image/png;asset_id,<uuid>" />
or list it in the NVCF-INPUT-ASSET-REFERENCES header.

Everything here is best-effort: on any failure the caller falls back to the
inline data URL and the warning list records why.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

ASSET_API = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
SIZE_LIMIT = 180 * 1024


def _asset_api() -> str:
    """Overridable so tests (and mirrors) can point at another host."""
    import os

    return os.environ.get("N2A_ASSET_API", ASSET_API)

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "mp4": "video/mp4",
}


def guess_mime(url_or_name: str, default: str = "image/png") -> str:
    name = url_or_name.split("?", 1)[0].rsplit(".", 1)
    if len(name) == 2:
        return _MIME_BY_EXT.get(name[1].lower(), default)
    return default


def create_asset(api_key: str, content_type: str, description: str = "nv2api upload", timeout: int = 60) -> dict[str, Any]:
    body = json.dumps({"contentType": content_type, "description": description}).encode()
    req = urllib.request.Request(
        _asset_api(),
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def upload_asset(api_key: str, data: bytes, content_type: str, timeout: int = 180) -> tuple[str, str]:
    """Upload bytes; returns (asset_id, reference_url). Raises on failure."""
    meta = create_asset(api_key, content_type, timeout=timeout)
    asset_id = meta.get("assetId") or meta.get("asset_id")
    upload_url = meta.get("uploadUrl") or meta.get("upload_url")
    if not asset_id or not upload_url:
        raise RuntimeError(f"unexpected asset response: {list(meta)}")

    put = urllib.request.Request(
        upload_url,
        data=data,
        method="PUT",
        headers={"Content-Type": content_type, "Content-Length": str(len(data))},
    )
    with urllib.request.urlopen(put, timeout=timeout) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"S3 PUT failed: {resp.status}")

    return asset_id, f"data:{content_type};asset_id,{asset_id}"


def make_resolver(api_key: str | None):
    """Returns a callable (url, warnings) -> url for translate.normalize_messages."""

    def resolve(url: str, warnings: list[str]) -> str:
        from .translate import data_url_size

        if not url.startswith("data:"):
            return url  # remote http(s) URLs are fetched by NVIDIA itself
        size = data_url_size(url)
        if size <= SIZE_LIMIT:
            return url
        if not api_key:
            warnings.append(
                f"inline image is {size // 1024} KB (>180 KB) and no upstream key is configured; "
                "NIM may reject it"
            )
            return url
        try:
            header, _, b64 = url.partition(",")
            content_type = header[5:].split(";")[0] or "image/png"
            import base64

            raw = base64.b64decode(b64)
            asset_id, ref = upload_asset(api_key, raw, content_type)
            warnings.append(f"image {size // 1024} KB uploaded as NVCF asset {asset_id[:8]}…")
            return ref
        except Exception as exc:  # noqa: BLE001 - fall back to inline
            warnings.append(f"asset upload failed ({exc}); sending inline data URL")
            return url

    return resolve
