#!/usr/bin/env python3
"""Register/update the kinema manifest with a-u.supply.

Fetches the live release catalog and injects a `[params.track]` dropdown
populated with `option_groups` (one group per release, options for each
track). Re-run whenever releases change.

Env: AU_API_KEY (required, admin scope), AU_BASE_URL (default https://a-u.supply).
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "manifest.toml"


def _base_url() -> str:
    return os.environ.get("AU_BASE_URL", "https://a-u.supply").rstrip("/")


def _api(path: str, *, method: str = "GET", body: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    key = os.environ.get("AU_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_base_url() + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _build_track_groups() -> list[dict]:
    """Walk all published releases and produce [params.track].option_groups."""
    rels = _api("/api/releases?status=published&per_page=200").get("releases", [])
    groups: list[dict] = []
    for r in rels:
        code = r["product_code"]
        title = r.get("title", code)
        detail = _api(f"/api/releases/{urllib.parse.quote(code, safe='')}")
        opts = []
        for t in detail.get("tracks", []):
            tid = t.get("id")
            if not tid or not t.get("stream_url"):
                continue
            opts.append({
                "value": f"{code}|{tid}",
                "label": f"{t.get('track_number','?')}. {t.get('title','(untitled)')}",
            })
        if opts:
            groups.append({"label": f"{code} — {title}", "options": opts})
    return groups


def _toml_quote(s: str) -> str:
    """Render a string for TOML — escape backslashes and double quotes."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _render_groups_toml(groups: list[dict]) -> str:
    """Append [[params.track.option_groups]] blocks to the manifest body."""
    out: list[str] = []
    for g in groups:
        out.append("\n[[params.track.option_groups]]")
        out.append(f"label = {_toml_quote(g['label'])}")
        for opt in g["options"]:
            out.append("[[params.track.option_groups.options]]")
            out.append(f"value = {_toml_quote(opt['value'])}")
            out.append(f"label = {_toml_quote(opt['label'])}")
    return "\n".join(out) + "\n"


def main() -> int:
    if not os.environ.get("AU_API_KEY"):
        print("AU_API_KEY env var required (admin scope)", file=sys.stderr)
        return 2

    manifest_text = MANIFEST_PATH.read_text()
    # Sanity-check it parses before we mutate.
    tomllib.loads(manifest_text)

    groups = _build_track_groups()
    print(f"built {len(groups)} release groups, {sum(len(g['options']) for g in groups)} tracks total")

    augmented = manifest_text + _render_groups_toml(groups)
    # Validate the augmented form still parses.
    tomllib.loads(augmented)

    # Try PUT first (update); fall back to POST (create).
    try:
        resp = _api("/api/apps/kinema", method="PUT", body={"manifest_toml": augmented})
        print("updated:", resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            resp = _api("/api/apps", method="POST", body={"manifest_toml": augmented})
            print("created:", {k: resp.get(k) for k in ("name", "enabled", "image")})
        else:
            print("PUT failed:", e.code, e.read().decode()[:500], file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
