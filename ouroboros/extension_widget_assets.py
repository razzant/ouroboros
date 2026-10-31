"""Finite browser-file declarations and capture for opt-in ESM widgets.

The extension publication owns the bytes. A declaration names files, never a
directory or a dependency graph; browser URL resolution supplies the graph.
"""
from __future__ import annotations

import pathlib
import re

from ouroboros.contracts.plugin_api import ExtensionRegistrationError

MAX_WIDGET_FILES = 256
MAX_WIDGET_BYTES = 32 * 1024 * 1024
WIDGET_ASSET_TYPES = {
    ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
    ".json": "application/json", ".txt": "text/plain", ".wasm": "application/wasm",
    ".bin": "application/octet-stream", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
    ".avif": "image/avif", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".woff": "font/woff", ".woff2": "font/woff2", ".ttf": "font/ttf", ".otf": "font/otf",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".mp4": "video/mp4", ".webm": "video/webm",
}


def widget_asset_path(value: object) -> bool:
    """Canonical, case-sensitive POSIX browser path; no escaping or URL syntax."""
    return (isinstance(value, str) and len(value) <= 240
            and all(re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", part)
                    for part in value.split("/"))
            and pathlib.PurePosixPath(value).suffix in WIDGET_ASSET_TYPES)


def validate_widget_files(render: dict) -> None:
    """Called by the common manifest/PluginAPI/child-catalog render validator."""
    mode = render.get("entry_type", "classic")
    if mode not in ("classic", "module"):
        raise ExtensionRegistrationError("module widget entry_type must be classic or module")
    if mode == "classic":
        if "assets" in render:
            raise ExtensionRegistrationError("widget assets require entry_type: module")
        return
    entry = render.get("entry")
    if not widget_asset_path(entry) or not entry.endswith((".js", ".mjs")):
        raise ExtensionRegistrationError("ESM widget entry must be a relative .js / .mjs browser path")
    assets = render.setdefault("assets", [])
    if (not isinstance(assets, list) or len(assets) >= MAX_WIDGET_FILES
            or not all(widget_asset_path(path) for path in assets)):
        raise ExtensionRegistrationError(f"widget assets must list fewer than {MAX_WIDGET_FILES} browser file paths")
    if len(set([entry, *assets])) != len(assets) + 1:
        raise ExtensionRegistrationError("widget assets must be unique and exclude entry (included automatically)")


def capture_widget_sources(skill, render: dict) -> dict[str, bytes]:
    """Capture only this tab's declared files, from the exact reviewed release."""
    from ouroboros.extension_contributions import capture_browser_sources

    if skill is None:
        raise ExtensionRegistrationError("ESM widget capture requires a loaded reviewed skill")
    paths = {render["entry"], *render["assets"]}
    root = skill.skill_dir.resolve()
    for rel in paths:
        target = root / rel
        if any(part.is_symlink() for part in (target, *target.parents) if part != root and root in part.parents):
            raise ExtensionRegistrationError(f"widget asset {rel!r} must not be a symlink")
    try:
        sources = capture_browser_sources(skill, skill.content_hash, paths=paths, max_bytes=MAX_WIDGET_BYTES)
        if sources.keys() != paths:
            raise ValueError("declared widget files are missing or excluded from the reviewed payload")
        for rel, content in sources.items():
            if rel.endswith((".js", ".mjs", ".css", ".json", ".txt", ".svg")):
                content.decode("utf-8")
        return sources
    except (OSError, ValueError, RuntimeError) as exc:
        raise ExtensionRegistrationError(f"widget assets: {exc}") from exc
