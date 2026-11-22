"""Capture targets: the display holding an application's window, or one exact window of it (macOS).

Mixed into the plugin's ``_ComputerUse`` beside ``PlaceMixin``. It uses the seams the plugin and the place mixin supply
(``_native``, ``_target_error``, ``_plat``, ``_helper_python``, ``_run_cmd``, ``_json_text``, ``_png_ok``, ``_png_dims``,
``_downscale``, ``api``) and imports neither module, so the plugin can import it.

A display capture is the screenshot's coordinate observation (SKILL.md, ``screenshot``). A window capture is
capture-only: one ScreenCaptureKit image of the named window, taken by the helper process ``macos_capture.py``. It is
accepted only when that process ended normally (exit 0, one answer line, ``ok``) and its PNG decodes whole at the size
it reported; anything else is a failure and the file is removed. It carries no ``observation_id``, transform or
display, and never falls back to a display capture.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import time
import uuid
from typing import Any

from .mac_input import permission_refusal, proved, with_refusal
from .macos_capture import window_selector

_WINDOW_TIMEOUT_MS = 5000  # both ScreenCaptureKit completion handlers together (macos_capture.SPEC)
# The helper's hard stop: interpreter start and PyObjC imports, the launch-free resolution (an lsappinfo read up to
# 8 s), then the capture's own budget.
_WINDOW_PROCESS_SEC = 30
_WINDOW_NOTE = ("Capture-only: one window's pixels, not display coordinates, so no observation_id is supplied. "
                "Act through element_act/ax_tree, or take a screenshot without window= for coordinate input.")


def window_source(window: dict[str, Any]) -> str:
    """The image's capture source: the same for every capture of one window of one process."""
    return f"window:{window['pid']}:{window['window_id']}"


def helper_answer(stdout: str) -> dict[str, Any] | None:
    """The helper's one answer: exactly one non-empty stdout line holding a JSON object, else None."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    try:
        parsed = json.loads(lines[0])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class CaptureMixin:
    """Which pixels a screenshot takes: a display (``_capture_display``) or one window (``_window_screenshot``)."""

    def _capture_display(self, app: str = "") -> dict[str, Any]:
        """The display to capture: the main one, or the one holding ``app``'s front window (bounds from CoreGraphics).

        The answer says how the display was chosen (``selected_by``: default, app, or fallback when the RESOLVED app's
        window could not be located) so a main-display capture is never presented as the named app's display. With a
        display list, an app not resolved (refused, or no answer naming its ``target``) is an ``error``, uncaptured;
        without one the app is not looked up (``app_display_unread``): the default capture stays capture-only."""
        res = self._native("displays", timeout=10)
        if res.get("screen_recording") == "denied":  # screencapture would exit 0 with the wallpaper alone
            return {"reason": "permission_denied", **permission_refusal("screen_recording"), "note": "nothing was captured",
                    "error": "this process may not record the screen (the Screen Recording grant); only the owner can grant it"}
        displays = res.get("displays") if res.get("ok") else None
        query = str(app or "").strip()
        if not displays:
            reason = res.get("error") or "no display list"
            return {"display_unread": reason, **({"app_display_unread": reason} if query else {})}
        # The grant as the display list read it travels with the display: only "granted" proves it.
        chosen = {**next((d for d in displays if d.get("main")), displays[0]), "selected_by": "default",
                  "screen_recording": res.get("screen_recording")}
        if query:
            tree = self._native("observe", json.dumps({"app": query, "max_nodes": 1, "timeout_ms": 1500}), timeout=15)
            target = tree.get("target")
            if tree.get("stage") == "resolve" or not isinstance(target, dict):  # unresolved or unknown: never a fallback
                return {"error": self._target_error(query, tree)}
            node = (tree.get("nodes") or [{}])[0] if tree.get("ok") else {}
            if "x" in node and "w" in node:
                cx, cy = node["x"] + node["w"] // 2, node["y"] + node["h"] // 2
                for d in displays:
                    if d["x"] <= cx < d["x"] + d["width"] and d["y"] <= cy < d["y"] + d["height"]:
                        return {**d, "selected_by": "app", "app": target, "screen_recording": res.get("screen_recording")}
                chosen["app_window_unread"] = f"the window's centre ({cx},{cy}) lies on no active display"
            else:
                chosen["app_window_unread"] = str(tree.get("error") or tree.get("reason") or "the window's bounds could not be read")
            chosen["selected_by"] = "fallback"
        return chosen

    def _window_screenshot(self, *, connection: tuple[str, dict[str, Any]], job_id: str, app: Any, window: Any,
                           max_width: int, max_height: int) -> str:
        """One exact window of ``app`` on this Mac as a capture-only image (no remote, no display fallback)."""
        name, conn = connection
        backend = str(conn.get("backend") or "local").lower()

        def refuse(reason: str, error: str, **extra: Any) -> str:
            return self._json_text({"ok": False, "reason": reason, "error": error, **extra})

        if backend != "local":
            return refuse("unsupported_backend", f"window= captures one window of this Mac; the active connection {name!r} "
                          f"is {backend}. Use screenshot without window= there.")
        if self._plat() != "macos":
            return refuse("unsupported_platform", f"window= capture is local macOS 14+ only (this is {self._plat()})")
        _, why = window_selector(window)
        if why:
            return refuse("bad_option", why)
        if not isinstance(app, str) or not app.strip():
            return refuse("app_required", "window= needs app=: the application that owns the window")
        out_dir = pathlib.Path(self.api.skill_job_dir(job_id or uuid.uuid4().hex[:8])) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"window-{int(time.time())}-{uuid.uuid4().hex[:8]}.png"  # a new name for every capture
        spec = {"app": app, "window": window, "output": str(out_path), "timeout_ms": _WINDOW_TIMEOUT_MS}
        cmd = [self._helper_python(), "-I", "-B", str(pathlib.Path(__file__).with_name("macos_capture.py")),
               json.dumps(spec, ensure_ascii=True)]
        try:
            rc, out, err = self._run_cmd(cmd, timeout=_WINDOW_PROCESS_SEC)
        except subprocess.TimeoutExpired:
            self._discard_window_output(out_path)
            return refuse("helper_stopped", f"the capture helper was stopped after {_WINDOW_PROCESS_SEC} s; nothing is kept")
        except OSError as exc:
            return refuse("helper_not_started", f"the capture helper could not start ({type(exc).__name__}: {exc})")
        answer = helper_answer(out)
        if rc != 0 or not answer or answer.get("ok") is not True:
            self._discard_window_output(out_path)
            if rc == 1 and answer and answer.get("ok") is False:
                return self._json_text(with_refusal(answer))  # the helper's own refusal: it ended normally and says why
            return refuse("helper_failed", f"the capture helper ended with exit {rc} without a complete answer "
                          f"({(err or out).strip()[-300:] or 'no output'})")
        return self._accept_window_capture(answer, out_path, max_width=max_width, max_height=max_height)

    def _accept_window_capture(self, answer: dict[str, Any], out_path: pathlib.Path, *, max_width: int,
                               max_height: int) -> str:
        """The helper's successful answer as the screenshot result, once its PNG is the whole image it reported."""
        facts = answer.get("window") if isinstance(answer.get("window"), dict) else {}
        target = answer.get("app") if isinstance(answer.get("app"), dict) else {}
        reported = (answer.get("image_width"), answer.get("image_height"))
        problem = ""
        if answer.get("output") != str(out_path) or not out_path.is_file():
            problem = "the helper reported no image at the requested path"
        elif not self._png_ok(out_path):
            problem = "the helper's PNG does not decode whole"
        elif self._png_dims(out_path) != reported:
            problem = f"the PNG is {self._png_dims(out_path)}, the helper reported {reported}"
        elif not isinstance(facts.get("window_id"), int) or facts.get("pid") != target.get("pid") or facts.get("pid") is None:
            problem = "the helper's window does not belong to the resolved application"
        if problem:
            self._discard_window_output(out_path)
            return self._json_text({"ok": False, "reason": "capture_failed", "error": problem, "window": facts or None})
        px_w, px_h = reported
        max_w = max(320, min(int(max_width or self._MAX_IMAGE_W), 4096))
        max_h = max(240, min(int(max_height or self._MAX_IMAGE_H), 4096))
        img_path, img_w, img_h = self._downscale(out_path, max_w, max_h)
        result: dict[str, Any] = {
            "ok": True, "path": str(img_path), "backend": "ScreenCaptureKit", "capture_kind": "window",
            "image_width": img_w, "image_height": img_h, "capture_width_px": px_w, "capture_height_px": px_h,
            "downscaled": img_path != out_path, "auto_attach_image": str(img_path),
            "auto_attach_source": window_source(facts), "app": target, "window": facts,
            "mapped_input_available": False, "coordinate_note": _WINDOW_NOTE,
            "cursor_included": answer.get("cursor_included"), "shadow_included": answer.get("shadow_included"),
            "point_pixel_scale": answer.get("point_pixel_scale"), "capture_ms": answer.get("capture_ms"),
        }
        if img_path != out_path:
            result["full_resolution_path"] = str(out_path)
        return self._json_text(proved(result, "screen_recording"))

    @staticmethod
    def _discard_window_output(out_path: pathlib.Path) -> None:
        """Remove a capture that was not accepted, with any partial file a stopped helper left beside it."""
        for path in (out_path, *out_path.parent.glob(out_path.name + ".*.part")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
