"""The running VERSION string: the bundled or checkout ``VERSION`` file, else package metadata."""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path


@cache
def get_version() -> str:
    candidates: list[Path] = []

    if getattr(sys, "frozen", False):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            candidates.append(Path(bundle_root) / "VERSION")

    candidates.append(Path(__file__).resolve().parent.parent / "VERSION")

    for candidate in candidates:
        try:
            version_text = candidate.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if version_text:
            return version_text

    try:
        from importlib.metadata import version as package_version

        return package_version("ouroboros")
    except Exception:
        return "0.0.0"
