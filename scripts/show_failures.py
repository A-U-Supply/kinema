#!/usr/bin/env python3
"""Dump log_tails for failed kinema jobs so I can fix each recipe."""

from __future__ import annotations
import json, re, urllib.request
from pathlib import Path

KEY = re.search(r"^AU_API_KEY=(.+)$",
                Path("/home/tube/github/a-u.supply/.env").read_text(), re.M).group(1).strip()
HDR = {"Authorization": f"Bearer {KEY}"}

req = urllib.request.Request(
    "https://a-u.supply/api/jobs?app_name=kinema&status=failed&per_page=40",
    headers=HDR,
)
with urllib.request.urlopen(req, timeout=15) as r:
    jobs = json.loads(r.read().decode()).get("jobs", [])

# Keep only the most recent failure per recipe (jobs come in desc order).
seen: set[str] = set()
for j in jobs:
    recipe = j.get("params", {}).get("recipe", "?")
    if recipe in seen:
        continue
    seen.add(recipe)
    print(f"\n====== {recipe} · {j.get('id','')[:8]} · {j.get('created_at','')} ======")
    log = (j.get("log_tail") or "")[-1500:]
    print(log or "(empty)")
