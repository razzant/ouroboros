"""Desktop MainApi JS bridge for the pywebview shell.

Loopback-guarded file downloads/saves/open-in-default-app and native
confirmation wrappers, extracted from the root ``launcher.py`` so the
launcher stays inside the size-ratchet module budget. All file URLs are
validated against the local Ouroboros server origin before any fetch.
"""

import base64
import logging
import pathlib
import shutil
import tempfile
import urllib.parse
import urllib.request

log = logging.getLogger("launcher.bridge")

# Wired by the launcher before the window is created.
_request_runtime_mode_change = None
_request_auto_grant_reviewed_skills_change = None
_request_skill_key_grant = None
_load_settings = None
_open_external_url = None
_request_native_attention = None
get_window = None  # callable() -> pywebview window | None
__actual_port = 0


def _native_confirm(title, message) -> bool:
    window = get_window() if get_window is not None else None
    return bool(window and window.create_confirmation_dialog(title, message))


def _resolve_bridge_file_url(raw_url: str) -> str:
    """Validate a loopback file-bridge URL, returning the resolved full URL.

    Shared SSOT for both the download-to-Downloads and open-in-default-app
    bridge methods so the loopback guard cannot drift between them.
    """
    full_url = urllib.parse.urljoin(f"http://127.0.0.1:{_actual_port}", str(raw_url or ""))
    parsed = urllib.parse.urlparse(full_url)
    if parsed.scheme != "http":
        raise ValueError("file URL must be http://")
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("desktop file access is limited to the local Ouroboros server")
    if parsed.port != _actual_port:
        raise ValueError("file URL port must match the local Ouroboros server")
    if parsed.path != "/api/files/download" and not parsed.path.startswith(("/api/extensions/", "/api/tasks/")):
        raise ValueError("file URL path must be /api/files/download, /api/extensions/<skill>/... or /api/tasks/...")
    return full_url

def _unique_bridge_target(directory: pathlib.Path, filename: str) -> pathlib.Path:
    safe_name = pathlib.Path(str(filename or "download")).name or "download"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / safe_name
    stem, suffix = target.stem, target.suffix
    counter = 1
    while target.exists():
        target = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return target

def _fetch_bridge_url_to(full_url: str, target: pathlib.Path) -> None:
    with urllib.request.urlopen(full_url, timeout=60) as resp, target.open("wb") as fh:  # noqa: S310 - localhost validated above
        shutil.copyfileobj(resp, fh)

class MainApi:
    @staticmethod
    def _native_confirm(title: str, message: str) -> bool:
        return _native_confirm(title, message)

    def request_runtime_mode_change(self, mode: str) -> dict:
        try:
            return _request_runtime_mode_change(mode, self._native_confirm)
        except Exception as exc:
            log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Native confirmation failed: {exc}"}

    def confirm_runtime_mode_change(self, mode: str) -> dict:
        """Confirm a mode change without writing it.

        The SPA persists the selected mode through the owner HTTP endpoint.
        Keeping this bridge side-effect free lets older shells fall back to
        the same in-app confirmation instead of normalizing newer modes
        such as Cyber Pro through their stale local enum.
        """
        try:
            mode_text = str(mode or "").strip().lower()
            if mode_text not in {"light", "advanced", "pro", "cyber_pro"}:
                return {"confirmed": False, "error": "Unknown runtime mode."}
            settings = _load_settings()
            current = normalize_runtime_mode(settings.get("OUROBOROS_RUNTIME_MODE"))
            message = (
                f"Change Ouroboros runtime mode from {current} to {mode_text}?\n\n"
                "The new mode is saved through the owner endpoint and takes effect after restart."
            )
            return {"confirmed": bool(_native_confirm("Confirm Runtime Mode Change", message))}
        except Exception as exc:
            log.warning("Runtime mode native confirmation failed: %s", exc, exc_info=True)
            return {"confirmed": False, "error": f"Native confirmation failed: {exc}"}

    def request_auto_grant_reviewed_skills_change(self, enabled: bool) -> dict:
        try:
            return _request_auto_grant_reviewed_skills_change(bool(enabled), self._native_confirm)
        except Exception as exc:
            log.warning("Reviewed-skill auto-grant confirmation failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Native confirmation failed: {exc}"}

    def request_skill_key_grant(self, skill: str, keys: list) -> dict:
        try:
            return _request_skill_key_grant(skill, keys, self._native_confirm)
        except Exception as exc:
            log.warning("Skill grant native confirmation failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Native confirmation failed: {exc}"}

    def download_file_to_downloads(self, url: str, filename: str, open_external: bool = False) -> dict:
        try:
            full_url = _resolve_bridge_file_url(url)
            target = _unique_bridge_target(pathlib.Path.home() / "Downloads", filename)
            _fetch_bridge_url_to(full_url, target)
            if open_external:
                open_path_external(target)
            return {"ok": True, "path": str(target)}
        except Exception as exc:
            log.warning("Desktop file download failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    def open_external_url(self, url: str) -> dict:
        return _open_external_url(url)
    def request_attention(self, sound: bool = True) -> dict:
        return _request_native_attention(get_window().show if get_window() else None, sound=bool(sound))

    def save_bytes_to_downloads(self, filename: str, b64: str) -> dict:
        try:
            target = _unique_bridge_target(pathlib.Path.home() / "Downloads", filename)
            target.write_bytes(base64.b64decode(str(b64 or ""), validate=True))
            return {"ok": True, "path": str(target)}
        except Exception as exc:
            log.warning("Desktop save-to-Downloads failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

    def open_file_with_default_app(self, url: str, filename: str) -> dict:
        """Open a delivered file in the OS default app (external window).

        Fetches the loopback file into a private temp dir (NOT ~/Downloads)
        and hands it to the platform default handler. This never navigates
        the in-app WKWebView, which was the original fullscreen-lockup bug.
        """
        try:
            full_url = _resolve_bridge_file_url(url)
            # Per-open private dir: mkdtemp atomically creates a fresh 0700
            # directory, so a pre-placed symlink/dir at a shared temp path
            # cannot redirect the write (hardens over a fixed shared root).
            open_root = pathlib.Path(tempfile.mkdtemp(prefix="ouroboros-open-"))
            target = _unique_bridge_target(open_root, filename)
            _fetch_bridge_url_to(full_url, target)
            open_path_external(target)
            return {"ok": True, "path": str(target)}
        except Exception as exc:
            log.warning("Desktop open-in-default-app failed: %s", exc, exc_info=True)
            return {"ok": False, "error": str(exc)}

# Prune stale externally-opened temp copies from previous sessions (privacy + disk).
