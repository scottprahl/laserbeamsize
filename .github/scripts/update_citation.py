#!/usr/bin/env python3
"""
Refreshes the version and date-released fields in CITATION.cff
using data from the GitHub release that triggered this workflow.
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
import yaml


REPO = os.getenv("GITHUB_REPOSITORY", "scottprahl/laserbeamsize")
EVENT_FILE = os.getenv("GITHUB_EVENT_PATH")          # set by GitHub Actions


def load_release_payload() -> dict:
    """Return release JSON, using event payload when available."""
    if EVENT_FILE and Path(EVENT_FILE).is_file():
        with open(EVENT_FILE, "r", encoding="utf-8") as fp:
            event = json.load(fp)
        if "release" in event:           # ← graceful guard
            return event["release"]

    # Fallback: hit the API (e.g. for workflow_dispatch runs)
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    headers = {}
    if (token := os.getenv("GITHUB_TOKEN")):
        headers["Authorization"] = f"token {token}"
    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()
    return res.json()


def normalise_version(tag: str) -> str:
    """Strip a leading 'v' or 'V' if present."""
    return tag.lstrip("vV")


def main() -> None:
    release = load_release_payload()
    release_date = release["published_at"][:10]                  # YYYY-MM-DD
    version = normalise_version(release["tag_name"])

    cff_path = Path("CITATION.cff")
    if not cff_path.exists():
        sys.exit("CITATION.cff not found")

    with cff_path.open("r", encoding="utf-8") as fp:
        cff = yaml.safe_load(fp)

    changed = False
    if cff.get("date-released") != release_date:
        cff["date-released"] = release_date
        changed = True
    if cff.get("version") != version:
        cff["version"] = version
        changed = True

    if changed:
        with cff_path.open("w", encoding="utf-8") as fp:
            yaml.safe_dump(cff, fp, sort_keys=False)
        print(
            f"Updated CITATION.cff → version={version}, "
            f"date-released={release_date}"
        )
    else:
        print("CITATION.cff already up-to-date.")


if __name__ == "__main__":
    main()
