#!/usr/bin/env python3
"""Submit one job per recipe, poll all jobs until terminal, print a table.

Designed to run autonomously without per-bash permission prompts. Reads
AU_API_KEY from the a-u.supply .env file. Single Python process; one
sleep between polls; no compound shell pipelines.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ENV_PATH = Path("/home/tube/github/a-u.supply/.env")
RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"
BASE = "https://a-u.supply"
POLL_INTERVAL = 12
DEADLINE_SECONDS = 60 * 30  # 30 min total ceiling


def _load_key() -> str:
    text = ENV_PATH.read_text()
    m = re.search(r"^AU_API_KEY=(.+)$", text, re.M)
    if not m:
        raise SystemExit("AU_API_KEY missing from .env")
    return m.group(1).strip()


def _api(path: str, *, key: str, body: dict | None = None, method: str = "GET") -> dict:
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    key = _load_key()
    recipes = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))
    print(f"verifying {len(recipes)} recipes", flush=True)

    job_for_recipe: dict[str, str] = {}
    for recipe in recipes:
        body = {
            "app_name": "kinema",
            "media_item_ids": [],
            "params": {
                "image_source": "random",
                "image_count": 40,
                "audio_source": "random_release",
                "recipe": recipe,
                "title_card": "none",
                "sec_per_image": 1.5,
            },
        }
        try:
            resp = _api("/api/jobs", key=key, body=body, method="POST")
            jid = resp.get("id") or resp.get("job_id")
            job_for_recipe[recipe] = jid
            print(f"submitted {recipe}: {jid}", flush=True)
        except Exception as e:
            print(f"submit FAIL {recipe}: {e}", flush=True)

    statuses: dict[str, dict] = {}
    start = time.time()
    while True:
        pending = [r for r in job_for_recipe if r not in statuses]
        if not pending:
            break
        if time.time() - start > DEADLINE_SECONDS:
            print(f"DEADLINE — {len(pending)} jobs still in flight", flush=True)
            break
        time.sleep(POLL_INTERVAL)
        for recipe in list(pending):
            jid = job_for_recipe[recipe]
            try:
                resp = _api(f"/api/jobs/{jid}", key=key)
            except Exception as e:
                print(f"poll {recipe} {jid}: {e}", flush=True)
                continue
            job = resp.get("job", resp)
            s = job.get("status")
            if s in ("completed", "failed", "cancelled", "error"):
                statuses[recipe] = {
                    "status": s,
                    "outputs": resp.get("outputs", []),
                    "log_tail": (job.get("log_tail") or "")[-1500:],
                    "id": jid,
                }
                emoji = "OK" if s == "completed" else "FAIL"
                size = ""
                if s == "completed" and resp.get("outputs"):
                    size = f" ({resp['outputs'][0].get('file_size_bytes',0)//1024} KB)"
                print(f"  [{emoji}] {recipe}{size}", flush=True)

    print()
    print("=== SUMMARY ===")
    passed = [r for r, v in statuses.items() if v["status"] == "completed"]
    failed = [r for r, v in statuses.items() if v["status"] != "completed"]
    print(f"passed: {len(passed)}/{len(job_for_recipe)}")
    if failed:
        print(f"failed: {failed}")
        for r in failed:
            v = statuses[r]
            print(f"\n--- {r} ({v['id']}) ---")
            print(v["log_tail"])
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
