#!/usr/bin/env python3
"""Re-verify a specific list of recipes, waiting for build then submitting.

Usage: pass recipe names as $RECIPES env var, comma-separated.
"""

from __future__ import annotations
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

KEY = re.search(r"^AU_API_KEY=(.+)$",
                Path("/home/tube/github/a-u.supply/.env").read_text(), re.M).group(1).strip()
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
BASE = "https://a-u.supply"

RECIPES = os.environ.get("RECIPES", "").split(",")
RECIPES = [r.strip() for r in RECIPES if r.strip()]
if not RECIPES:
    print("set RECIPES=a,b,c", file=sys.stderr); sys.exit(2)


def _wait_for_build():
    print("waiting for kinema image build...", flush=True)
    while True:
        out = subprocess.check_output(
            ["gh", "run", "list", "--repo", "A-U-Supply/kinema",
             "--workflow", "Build and push Docker image", "--limit", "1",
             "--json", "databaseId,status,conclusion"],
            text=True,
        )
        info = json.loads(out)[0]
        status, conclusion = info["status"], info["conclusion"]
        if status == "completed":
            print(f"  build done: {conclusion}", flush=True)
            return conclusion == "success"
        time.sleep(15)


def _api(path, body=None, method="GET"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers=HDR, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _submit(recipe):
    body = {
        "app_name": "kinema",
        "media_item_ids": [],
        "params": {
            "image_source": "random", "image_count": 40,
            "audio_source": "random_release",
            "recipe": recipe, "title_card": "none",
            "sec_per_image": 1.5,
        },
    }
    return _api("/api/jobs", body=body, method="POST")


def main() -> int:
    if not _wait_for_build():
        print("build failed, not resubmitting"); return 1

    job_ids = {}
    for r in RECIPES:
        try:
            resp = _submit(r)
            jid = resp.get("id") or resp.get("job_id")
            job_ids[r] = jid
            print(f"submitted {r}: {jid[:8]}", flush=True)
        except Exception as e:
            print(f"submit FAIL {r}: {e}", flush=True)

    # Poll
    done = {}
    deadline = time.time() + 60 * 40  # 40 min
    while job_ids.keys() - done.keys() and time.time() < deadline:
        time.sleep(15)
        for r, jid in list(job_ids.items()):
            if r in done: continue
            try:
                resp = _api(f"/api/jobs/{jid}")
            except Exception: continue
            job = resp.get("job", resp)
            s = job.get("status")
            if s in ("completed", "failed", "cancelled"):
                done[r] = (s, resp.get("outputs", []), (job.get("log_tail") or "")[-800:])
                size = ""
                if s == "completed" and resp.get("outputs"):
                    size = f" ({resp['outputs'][0].get('file_size_bytes',0)//1024} KB)"
                print(f"  [{'OK' if s=='completed' else 'FAIL'}] {r}{size}", flush=True)

    passed = [r for r, v in done.items() if v[0] == "completed"]
    failed = [r for r, v in done.items() if v[0] != "completed"]
    print(f"\npassed {len(passed)}/{len(job_ids)}: {passed}")
    if failed:
        for r in failed:
            s, outs, log = done[r]
            print(f"\n--- {r} [{s}] ---"); print(log)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
