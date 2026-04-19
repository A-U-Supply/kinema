#!/usr/bin/env python3
"""Cancel all pending/running kinema jobs to free the queue."""

from __future__ import annotations
import json
import re
import urllib.request
from pathlib import Path

ENV = Path("/home/tube/github/a-u.supply/.env")
KEY = re.search(r"^AU_API_KEY=(.+)$", ENV.read_text(), re.M).group(1).strip()
HDR = {"Authorization": f"Bearer {KEY}"}

with urllib.request.urlopen(
    urllib.request.Request("https://a-u.supply/api/jobs?app_name=kinema&per_page=20", headers=HDR),
    timeout=15,
) as r:
    jobs = json.loads(r.read().decode()).get("jobs", [])

for j in jobs:
    if j.get("status") in ("pending", "running"):
        jid = j["id"]
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"https://a-u.supply/api/jobs/{jid}/cancel",
                    headers=HDR, method="POST", data=b"",
                ),
                timeout=15,
            ):
                print(f"cancelled {jid[:8]} ({j.get('params',{}).get('recipe','?')})")
        except Exception as e:
            print(f"fail {jid[:8]}: {e}")
