"""Shared constants and platform primitives for the unix_computer_use skill.

Verbatim extraction from ``plugin.py`` (v7 stream W): the connection-registry
and remote-backend mixins live in sibling leaves and cannot import the plugin
entry module, so the values both they and ``plugin.py`` need are owned here.
``plugin.py`` re-exports every name, keeping its public module surface intact.
"""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
from typing import Any

_TIMEOUT_SEC = 10
# Anthropic computer-use guidance: keep screenshots at/below ~XGA/WXGA so the
# model reasons over a stable, token-cheap coordinate space.
_MAX_IMAGE_W = 1280
_MAX_IMAGE_H = 800
_CONNECTIONS_FILE = "connections.json"
_ACTIVE_CONNECTION_FILE = "active_connection.txt"
_REMOTE_BACKENDS = {"osworld_http", "ssh_macos"}
# Cap a remote /screenshot download (a 1920x1080 PNG is well under 10 MB).
_MAX_REMOTE_SHOT_BYTES = 20 * 1024 * 1024

# Remote backend constants. These are dormant unless a non-local connection is
# explicitly activated in skill state (or by a benchmark runner). The default
# behavior remains local macOS/Linux computer-use.
_OSWORLD_PKGS_PREFIX = (
    "import pyautogui; import time; import platform; "
    "pyautogui.FAILSAFE = False; "
    "{command}"
)


def _osworld_result_ok(out: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed verdict for an OSWorld /execute round-trip: the in-VM server
    returns HTTP 200 even on nonzero exit, so require 200 AND (dict body)
    status=="success" AND returncode==0 when present."""
    if int(out.get("status") or 0) != 200:
        return False, f"HTTP {out.get('status')}"
    result = out.get("result")
    if not isinstance(result, dict):
        return False, "unexpected non-JSON /execute response"
    status = str(result.get("status") or "").strip().lower()
    if status and status != "success":
        return False, str(result.get("message") or result.get("error") or f"status={status}")[:1000]
    returncode = result.get("returncode")
    if returncode is not None:
        try:
            rc = int(returncode)
        except Exception:
            return False, f"non-integer returncode {returncode!r}"
        if rc != 0:
            err = str(result.get("error") or result.get("output") or "").strip()
            return False, (err or f"guest command exited {rc}")[:1000]
    return True, ""


def _png_dimensions(path: pathlib.Path) -> tuple[int, int]:
    """Physical (pixel) width/height from a PNG IHDR; (0, 0) on failure."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
    except Exception:
        pass
    return 0, 0


def _png_intact(path: pathlib.Path) -> bool:
    """Full-decode integrity check, not just the IHDR.

    A truncated/zero-padded PNG keeps a valid 24-byte header, so
    ``_png_dimensions`` alone cannot see the damage; in the v6.81.1 OSWorld run
    such a file passed header checks, survived ``_downscale`` (which swallows
    the PIL error and returns the corrupt source) and then killed the whole
    task with a non-retryable provider 400 ("Could not process image"). Decode
    the WHOLE image before ever reporting ok:true. Without PIL, fall back to
    requiring the IEND trailer — weaker, but it still catches truncation.
    """
    try:
        from PIL import Image
    except Exception:
        try:
            with open(path, "rb") as fh:
                fh.seek(max(0, path.stat().st_size - 16))
                return b"IEND" in fh.read()
        except Exception:
            return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _run(cmd: list[str], *, timeout: int = _TIMEOUT_SEC) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""
