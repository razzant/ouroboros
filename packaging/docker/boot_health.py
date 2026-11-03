"""Compose healthcheck: a live container stays healthy unless boot failed.

``GET /api/health`` answers 200 while the process is still booting. ``boot.phase``
``failed`` is the only unhealthy document. A probe that cannot connect, times out,
or sees any other document returns success. There is no start deadline, so a long
boot is not marked failed. The HTTP timeout is only this probe's transport bound.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

_PROBE_TIMEOUT_SEC = 3.0


def phase_is_failed(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    boot = document.get("boot")
    if not isinstance(boot, dict):
        return False
    return boot.get("phase") == "failed"


def probe(url: str, timeout: float = _PROBE_TIMEOUT_SEC) -> int:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            raw = response.read()
    except (OSError, ValueError):
        return 0
    if status != 200:
        return 0
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0
    return 1 if phase_is_failed(document) else 0


def main() -> int:
    raw = os.environ.get("OUROBOROS_SERVER_PORT", "8765").strip() or "8765"
    if not raw.isdigit() or not 1 <= int(raw) <= 65535:
        return 0
    return probe(f"http://127.0.0.1:{raw}/api/health")


if __name__ == "__main__":
    sys.exit(main())
