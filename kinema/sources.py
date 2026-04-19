"""a-u.supply API client.

Reads images from the search index, audio from releases or any output index.
Auth via env: AU_BASE_URL (default https://a-u.supply), AU_API_KEY (Bearer).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


def _base_url() -> str:
    return os.environ.get("AU_BASE_URL", "https://a-u.supply").rstrip("/")


def _headers() -> dict[str, str]:
    key = os.environ.get("AU_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


@dataclass
class MediaHit:
    id: str
    media_type: str
    filename: str
    file_path: str
    download_url: str
    duration_seconds: float | None = None


def search_media(
    media_types: list[str],
    filters: dict[str, Any] | None = None,
    query: str | None = None,
    page: int = 1,
    per_page: int = 100,
    sort: str = "created_at:desc",
) -> list[MediaHit]:
    """POST /api/search — returns hits matching filter spec."""
    payload: dict[str, Any] = {
        "media_types": media_types,
        "page": page,
        "per_page": per_page,
        "sort": sort,
    }
    if filters:
        payload["filters"] = filters
    if query:
        payload["query"] = query

    r = requests.post(f"{_base_url()}/api/search", json=payload, headers=_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    return [_hit_from_search(h) for h in data.get("hits", [])]


def _hit_from_search(h: dict[str, Any]) -> MediaHit:
    mid = h["id"]
    return MediaHit(
        id=mid,
        media_type=h["media_type"],
        filename=h.get("filename", mid),
        file_path=h.get("file_path", ""),
        download_url=f"{_base_url()}/api/media/{mid}/file",
        duration_seconds=(h.get("audio_meta") or {}).get("duration_seconds")
        or (h.get("video_meta") or {}).get("duration_seconds"),
    )


def list_releases(status: str = "published", per_page: int = 200) -> list[dict[str, Any]]:
    r = requests.get(
        f"{_base_url()}/api/releases",
        params={"status": status, "per_page": per_page},
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("releases", [])


def get_release(code: str) -> dict[str, Any]:
    r = requests.get(
        f"{_base_url()}/api/releases/{quote(code, safe='')}",
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _absolutize(url: str) -> str:
    """Release/track API responses include relative stream_urls; we need absolute."""
    return url if url.startswith("http") else f"{_base_url()}{url}"


def find_track_stream_url(code: str, track_number: int) -> tuple[str, str]:
    """Return (absolute_stream_url, title) for a 1-based track number in a release."""
    detail = get_release(code)
    for t in detail.get("tracks", []):
        if int(t.get("track_number", -1)) == int(track_number):
            url = t.get("stream_url")
            if not url:
                raise RuntimeError(f"track {track_number} of {code} has no stream_url")
            return _absolutize(url), t.get("title", "")
    raise RuntimeError(f"release {code} has no track number {track_number}")


def random_release_track() -> tuple[str, str]:
    """Pick any track from any published release. Returns (absolute_stream_url, title)."""
    releases = list_releases()
    if not releases:
        raise RuntimeError("no published releases available")
    random.shuffle(releases)
    for rel in releases:
        detail = get_release(rel["product_code"])
        tracks = [t for t in detail.get("tracks", []) if t.get("stream_url")]
        if not tracks:
            continue
        track = random.choice(tracks)
        return _absolutize(track["stream_url"]), track.get("title", "")
    raise RuntimeError("no published releases have streamable tracks")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=_headers(), stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return dest
