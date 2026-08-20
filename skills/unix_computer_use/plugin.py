"""Unix computer-use extension: screenshot + desktop input for macOS/Linux.

The implementation is intentionally substrate-thin: it detects already-installed
platform tools and returns clear unsupported/capability errors instead of
downloading binaries. All GUI actions are meant to run UNDER HUMAN OBSERVATION;
the agent should prefer semantic application APIs when available and ask before
destructive or sensitive UI actions.

Coordinate contract (normalization): ``screenshot`` may downscale the captured
image to fit WXGA (1280x800, Anthropic computer-use guidance) and persists the
exact image->input transform in skill state. Input tools (click/move/drag/...)
accept coordinates in the LAST SCREENSHOT's image space by default and remap
them through that stored transform; pass ``raw=true`` to bypass remapping and
address native input coordinates directly.

Permission caveat (honest): macOS TCC state (Screen Recording / Accessibility)
is NOT probed — ``screencapture``/``cliclick`` exit 0 even when permission is
denied (capturing only the wallpaper, or dropping synthetic input), so callers
must ensure the host has granted these. Wayland support depends on ydotool
(input) and grim/gnome-screenshot (capture); X11 uses xdotool/scrot. Windows
support is deferred to a separate skill.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import platform
import re
import shutil
import time
import urllib.request
import uuid
from typing import Any

from .lib.cu_connections import _ConnectionRegistryMixin
from .lib.cu_remote_backends import _RemoteBackendMixin
from .lib.cu_runtime import (  # noqa: F401 - re-exported module surface
    _ACTIVE_CONNECTION_FILE,
    _CONNECTIONS_FILE,
    _MAX_IMAGE_H,
    _MAX_IMAGE_W,
    _MAX_REMOTE_SHOT_BYTES,
    _OSWORLD_PKGS_PREFIX,
    _REMOTE_BACKENDS,
    _TIMEOUT_SEC,
    _json,
    _osworld_result_ok,
    _png_dimensions,
    _png_intact,
    _run,
)

_TRANSFORM_FILE = "coord_transform.json"
_AX_MAX_ELEMENTS = 120
_PYAUTOGUI_MODS = {
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt",
    "shift": "shift",
    "cmd": "winleft", "command": "winleft", "super": "winleft", "meta": "winleft", "win": "winleft",
    "fn": "fn",
}


_PYAUTOGUI_BASE_ALIASES = {
    "return": "enter", "enter": "enter", "esc": "esc", "escape": "esc",
    # Canonical "delete"=BACKWARD (matches _X11_KEY_ALIASES); pyautogui "delete"=FORWARD, so swap.
    "del": "backspace", "delete": "backspace", "backspace": "backspace", "fwd-delete": "delete",
    "space": "space", "tab": "tab", "home": "home", "end": "end",
    "page-down": "pagedown", "pagedown": "pagedown", "page_down": "pagedown", "pgdn": "pagedown",
    "page-up": "pageup", "pageup": "pageup", "page_up": "pageup", "pgup": "pageup",
    "arrow-down": "down", "arrow-up": "up", "arrow-left": "left", "arrow-right": "right",
    "down": "down", "up": "up", "left": "left", "right": "right",
}


def _macos_logical_size() -> tuple[int, int]:
    """Logical (point) DESKTOP size via AppleScript; (0, 0) on failure.

    cliclick consumes logical points while ``screencapture`` writes physical
    pixels; exposing both recovers the Retina scale. On a MULTI-DISPLAY Mac
    this is the union of all displays while a no-argument ``screencapture``
    captures only the main display — the derived scale is then approximate.
    """
    try:
        rc, out, _ = _run([
            "osascript", "-e",
            'tell application "Finder" to get bounds of window of desktop',
        ])
        if rc == 0:
            parts = [p.strip() for p in out.replace(",", " ").split()]
            nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
            if len(nums) >= 4:
                return nums[2] - nums[0], nums[3] - nums[1]
    except Exception:
        pass
    return 0, 0


def _which(name: str) -> str:
    return shutil.which(name) or ""


def _coerce_point(x: Any, y: Any) -> tuple[int, int]:
    """Normalize pointer coordinates from the shapes models actually emit.

    Accepted: ints/floats; numeric strings ("663"); and the recurring
    malformation where BOTH coordinates arrive in one string — ``x="663, 500"``
    with ``y`` absent, or with ``y`` a single number duplicating one of the
    pair's. The v6.81.0 OSWorld run hit exactly this 109 times across 100
    tasks, each costing a full round on a ValueError.

    ABSENT (None / empty string) and UNPARSEABLE are different facts: an absent
    y may be recovered from a pair in x; a non-empty y that parses to nothing
    ("invalid") or to several numbers ("500, 42") is contradictory input and
    raises ValueError naming both values — a genuinely garbled call must fail
    loudly, not be guessed into a real pointer action.
    """
    def parse(value: Any, *, name: str) -> list[int] | None:
        """None == genuinely absent; a list is always non-empty; garbage raises."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"boolean is not a coordinate: {name}={value!r}")
        if isinstance(value, (int, float)):
            return [int(round(value))]
        text = str(value)
        if not text.strip():
            return None
        found = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not found:
            raise ValueError(f"cannot parse {name} coordinate from {value!r}")
        return [int(round(float(n))) for n in found]

    xs = parse(x, name="x")
    ys = parse(y, name="y")
    if xs is None:
        raise ValueError(f"missing x coordinate (y={y!r})")
    if len(xs) == 1:
        if ys is None:
            raise ValueError(f"missing y coordinate (x={x!r})")
        if len(ys) == 1:
            return xs[0], ys[0]
        raise ValueError(f"ambiguous y coordinate: x={x!r}, y={y!r}")
    if len(xs) == 2:
        # The whole point arrived in x. Accept a truly absent y, or a SINGLE
        # y that merely repeats one of the pair's numbers (the observed
        # duplication shape). A multi-number or disagreeing y contradicts the
        # pair and must not be silently resolved either way.
        if ys is None or (len(ys) == 1 and ys[0] in xs):
            return xs[0], xs[1]
        raise ValueError(f"ambiguous coordinates: x={x!r} carries a pair but y={y!r} disagrees")
    raise ValueError(f"cannot parse coordinates from x={x!r}, y={y!r}")


def _platform() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    if system == "windows":
        return "windows"
    return system or "unknown"


def _session_type() -> str:
    """linux display-session type: wayland | x11 | unknown (macos: native)."""
    if _platform() == "macos":
        return "native"
    raw = str(os.environ.get("XDG_SESSION_TYPE") or "").strip().lower()
    if raw in ("wayland", "x11"):
        return raw
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "unknown"


# Aliases accepted by the `key`/`hold_key` tools, mapped per backend.
_KEY_ALIASES = {
    "pagedown": "page-down", "page_down": "page-down", "pgdn": "page-down",
    "pageup": "page-up", "page_up": "page-up", "pgup": "page-up",
    "down": "arrow-down", "up": "arrow-up", "left": "arrow-left", "right": "arrow-right",
    "arrowdown": "arrow-down", "arrowup": "arrow-up", "arrowleft": "arrow-left", "arrowright": "arrow-right",
    "enter": "return", "escape": "esc", "del": "delete", "backspace": "delete",
    "space": "space", "tab": "tab", "home": "home", "end": "end",
}
_X11_KEY_ALIASES = {
    "page-down": "Page_Down", "page-up": "Page_Up",
    "arrow-down": "Down", "arrow-up": "Up", "arrow-left": "Left", "arrow-right": "Right",
    "return": "Return", "esc": "Escape", "delete": "BackSpace", "fwd-delete": "Delete",
    "space": "space", "tab": "Tab", "home": "Home", "end": "End",
    "cmd": "super", "command": "super", "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt", "shift": "shift",
    "super": "super", "meta": "Meta_L",
    # X11 keysyms are case-sensitive; lowercase fN must map back to FN.
    **{f"f{i}": f"F{i}" for i in range(1, 17)},
}
_MAC_MODS = {
    "cmd": "cmd", "command": "cmd", "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt", "shift": "shift", "fn": "fn",
}
# Linux additionally accepts the super/meta modifier names directly.
_ALL_MODS = {**_MAC_MODS, "super": "super", "meta": "meta"}


class _ComputerUse(_ConnectionRegistryMixin, _RemoteBackendMixin):
    def __init__(self, api: Any) -> None:
        self.api = api
        self.state_dir = pathlib.Path(api.get_state_dir())
        self.state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # capabilities / coordinate transform
    # ------------------------------------------------------------------

    def _capabilities(self) -> dict[str, Any]:
        plat = _platform()
        session = _session_type()
        return {
            "platform": plat,
            "session_type": session,
            "screenshot": {
                "screencapture": bool(_which("screencapture")),
                "gnome_screenshot": bool(_which("gnome-screenshot")),
                "scrot": bool(_which("scrot")),
                "grim": bool(_which("grim")),
            },
            "input": {
                "cliclick": bool(_which("cliclick")),
                "xdotool": bool(_which("xdotool")),
                "ydotool": bool(_which("ydotool")),
                "wtype": bool(_which("wtype")),
                "osascript": bool(_which("osascript")),
            },
            "downscale": {
                "sips": bool(_which("sips")),
                "magick": bool(_which("magick") or _which("convert")),
            },
            "notes": [
                "ALL actions are for supervised, low-risk local workflows under human observation.",
                "macOS requires Screen Recording for screenshot and Accessibility for input.",
                "macOS TCC permission state is NOT verified here: screencapture/cliclick exit 0 "
                "even when denied (wallpaper-only capture / dropped input). Ensure grants are in place.",
                "macOS scroll is unsupported (cliclick has no scroll-wheel command); use key paging instead.",
                "macOS middle-click and mouse_down/up for non-left buttons are unsupported via cliclick.",
                "Linux X11 needs xdotool (input) + gnome-screenshot/scrot (capture).",
                "Linux Wayland needs ydotool (input; ydotoold running) or wtype (typing only) "
                "+ grim or gnome-screenshot (capture). xdotool does NOT work on Wayland.",
                "Input coordinates default to the LAST screenshot's image space (auto-remapped); "
                "pass raw=true for native input coordinates.",
                "Windows support is deferred to a separate skill.",
            ],
            "permission_state_verified": False,
        }

    def capabilities(self) -> str:
        name, conn = self._active_connection()
        payload = {"ok": True, **self._capabilities()}
        payload["active_connection"] = name
        payload["active_backend"] = str(conn.get("backend") or "local")
        if self._is_remote():
            payload["remote"] = {k: v for k, v in conn.items() if "key" not in k.lower() and "secret" not in k.lower()}
        return _json(payload)

    def _save_transform(self, data: dict[str, Any]) -> None:
        try:
            (self.state_dir / _TRANSFORM_FILE).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_transform(self) -> dict[str, Any]:
        try:
            return json.loads((self.state_dir / _TRANSFORM_FILE).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _map_xy(self, x: int, y: int, *, raw: bool) -> tuple[int, int, dict[str, Any]]:
        """Map screenshot-image coordinates to native input coordinates.

        Returns (input_x, input_y, note). Without a stored transform (no
        screenshot yet, or raw=true) coordinates pass through unchanged.
        """
        if raw:
            return int(x), int(y), {"coordinate_space": "raw"}
        tf = self._load_transform()
        sx = float(tf.get("sx") or 0.0)
        sy = float(tf.get("sy") or 0.0)
        if sx <= 0.0 or sy <= 0.0:
            return int(x), int(y), {"coordinate_space": "raw (no stored screenshot transform)"}
        return (
            int(round(int(x) * sx)),
            int(round(int(y) * sy)),
            {
                "coordinate_space": "screenshot",
                "mapped_from": [int(x), int(y)],
                "transform_ts": tf.get("ts"),
            },
        )

    # ------------------------------------------------------------------
    # screenshot (with WXGA normalization)
    # ------------------------------------------------------------------

    def _downscale(self, src: pathlib.Path, max_w: int, max_h: int) -> tuple[pathlib.Path, int, int]:
        """Downscale PNG to fit max_w x max_h; returns (path, w, h) (src on no-op/failure)."""
        px_w, px_h = _png_dimensions(src)
        if px_w <= 0 or px_h <= 0 or (px_w <= max_w and px_h <= max_h):
            return src, px_w, px_h
        ratio = min(max_w / px_w, max_h / px_h)
        new_w = max(1, int(px_w * ratio))
        new_h = max(1, int(px_h * ratio))
        dest = src.with_name(src.stem + f"-{new_w}x{new_h}.png")
        # PRIMARY: in-process PIL — deterministic and dependency-light so the
        # downscale (and therefore the stored image->input coord_transform) ALWAYS
        # happens. Without this the skill silently no-oped on hosts lacking
        # sips/ImageMagick, returning a full-resolution image with an identity
        # transform while view_image re-downscaled independently — a coordinate-
        # space mismatch that made every click land off-target.
        try:
            from PIL import Image
            resample = getattr(Image, "Resampling", Image).LANCZOS
            with Image.open(src) as im:
                im.convert("RGB").resize((new_w, new_h), resample).save(dest, format="PNG")
            gw, gh = _png_dimensions(dest)
            if dest.exists() and gw > 0 and gh > 0:
                return dest, gw, gh
        except Exception:
            pass
        # FALLBACK: external resizers (macOS sips / ImageMagick) when PIL is absent.
        if _which("sips"):  # macOS built-in
            cmd = ["sips", "-z", str(new_h), str(new_w), str(src), "--out", str(dest)]
        elif _which("magick"):
            cmd = ["magick", str(src), "-resize", f"{new_w}x{new_h}!", str(dest)]
        elif _which("convert"):
            cmd = ["convert", str(src), "-resize", f"{new_w}x{new_h}!", str(dest)]
        else:
            return src, px_w, px_h
        try:
            rc, _, _ = _run(cmd, timeout=15)
        except Exception:
            return src, px_w, px_h
        if rc != 0 or not dest.exists():
            return src, px_w, px_h
        got_w, got_h = _png_dimensions(dest)
        return dest, (got_w or new_w), (got_h or new_h)

    def screenshot(self, *, job_id: str = "manual", max_width: int = _MAX_IMAGE_W,
                   max_height: int = _MAX_IMAGE_H) -> str:
        _conn_name, conn = self._active_connection()
        if conn.get("disabled"):
            return self._disabled_connection_error(_conn_name, conn)
        backend = str(conn.get("backend") or "local").lower()
        if backend == "osworld_http":
            return self._osworld_screenshot(conn, max_width=max_width, max_height=max_height)
        if backend == "ssh_macos":
            return self._ssh_macos_screenshot(conn, max_width=max_width, max_height=max_height)
        job_dir = self.api.skill_job_dir(job_id or uuid.uuid4().hex[:8])
        out_dir = pathlib.Path(job_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"screenshot-{int(time.time())}.png"
        plat = _platform()
        session = _session_type()
        if plat == "macos" and _which("screencapture"):
            cmd = ["screencapture", "-x", str(out_path)]
        elif plat == "linux" and session == "wayland" and _which("grim"):
            cmd = ["grim", str(out_path)]
        elif plat == "linux" and _which("gnome-screenshot"):
            cmd = ["gnome-screenshot", "-f", str(out_path)]
        elif plat == "linux" and session != "wayland" and _which("scrot"):
            cmd = ["scrot", str(out_path)]
        else:
            return _json({
                "ok": False,
                "error": "no supported screenshot backend found for this session "
                         f"(platform={plat}, session={session})",
                "capabilities": self._capabilities(),
            })
        try:
            rc, stdout, stderr = _run(cmd, timeout=15)
        except Exception as exc:  # noqa: BLE001 - surface host permission errors plainly
            return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}", "cmd": cmd})
        if rc != 0 or not out_path.exists():
            return _json({"ok": False, "error": stderr.strip() or stdout.strip() or f"exit {rc}", "cmd": cmd})

        px_w, px_h = _png_dimensions(out_path)
        max_w = max(320, min(int(max_width or _MAX_IMAGE_W), 4096))
        max_h = max(240, min(int(max_height or _MAX_IMAGE_H), 4096))
        img_path, img_w, img_h = self._downscale(out_path, max_w, max_h)

        # Input-space size: macOS input consumes LOGICAL points; X11/Wayland
        # input consumes the same pixel space the capture was taken in.
        input_w, input_h = px_w, px_h
        scale_note = ""
        approx = False
        if plat == "macos":
            log_w, log_h = _macos_logical_size()
            if log_w > 0 and log_h > 0:
                input_w, input_h = log_w, log_h
                scale = round(px_w / log_w, 4) if log_w else 0.0
                if scale and abs(scale - round(scale)) > 0.02 and abs(scale * 2 - round(scale * 2)) > 0.02:
                    approx = True
                # A real device scale is >=1.0; a sub-1 ratio means the logical
                # bounds span MULTIPLE displays while screencapture grabbed only
                # the main one — the transform is wrong even if the ratio looks
                # "clean" (e.g. two identical Retinas give exactly 0.5).
                if scale and scale < 0.95:
                    approx = True
                scale_note = (
                    "macOS input uses LOGICAL points; this transform already maps "
                    "image pixels to logical points."
                )
            else:
                # Logical size unavailable (e.g. Automation TCC denied): the
                # pixel-based transform would be 2x off for cliclick's logical
                # points on Retina. Flag it honestly instead of silently lying.
                approx = True
                scale_note = (
                    "WARNING: macOS logical desktop size unavailable — the "
                    "transform maps to PIXELS, but cliclick consumes LOGICAL "
                    "points (2x off on Retina). Grant Automation access or "
                    "prefer ax_tree marks with raw=true."
                )
        result: dict[str, Any] = {
            "ok": True,
            "path": str(img_path),
            "backend": cmd[0],
            "image_width": img_w, "image_height": img_h,
            "capture_width_px": px_w, "capture_height_px": px_h,
            "input_width": input_w, "input_height": input_h,
            "downscaled": img_path != out_path,
            "auto_attach_image": str(img_path),
        }
        if img_path != out_path:
            result["full_resolution_path"] = str(out_path)
        if img_w > 0 and img_h > 0 and input_w > 0 and input_h > 0:
            sx = round(input_w / img_w, 6)
            sy = round(input_h / img_h, 6)
            transform = {
                "sx": sx, "sy": sy,
                "image_w": img_w, "image_h": img_h,
                "input_w": input_w, "input_h": input_h,
                "platform": plat, "session": session,
                "approx": approx, "ts": time.time(),
            }
            self._save_transform(transform)
            result["coord_transform"] = transform
            result["coordinate_note"] = (
                "Pass coordinates read off THIS image directly to click/move/drag — "
                "they are auto-remapped through coord_transform (image -> input space). "
                + scale_note
                + (" Multi-display scale is approximate; prefer ax_tree marks." if approx else "")
            )
        return _json(result)

    # ------------------------------------------------------------------
    # pointer actions
    # ------------------------------------------------------------------

    def _backend_unavailable(self, action: str) -> str:
        return _json({
            "ok": False,
            "error": f"no supported {action} backend found "
                     f"(platform={_platform()}, session={_session_type()})",
            "capabilities": self._capabilities(),
        })

    def click(self, *, x: int, y: int | None = None, button: str = "left", double: bool = False,
              triple: bool = False, raw: bool = False) -> str:
        plat = _platform()
        button = str(button or "left").strip().lower()
        if button not in ("left", "right", "middle"):
            return _json({"ok": False, "error": f"unknown button {button!r} (use left/right/middle)"})
        try:
            x, y = _coerce_point(x, y)
        except ValueError as exc:
            return _json({"ok": False, "error": str(exc)})
        if int(x) < 0 or int(y) < 0:
            return _json({"ok": False, "error": f"negative coordinates not allowed ({x},{y})"})
        ix, iy, note = self._map_xy(x, y, raw=raw)
        repeat = 3 if triple else (2 if double else 1)
        if self._is_remote():
            _name, conn = self._active_connection()
            return self._remote_pyautogui(
                conn,
                f"pyautogui.click({ix}, {iy}, clicks={repeat}, interval=0.1, button={button!r})",
                note=note,
            )
        if plat == "macos" and _which("cliclick"):
            if button == "middle":
                return _json({"ok": False, "error": "middle-click unsupported on macOS (cliclick has no middle button)"})
            if button == "right":
                op = "rc"
            elif triple:
                op = "tc"
            elif double:
                op = "dc"
            else:
                op = "c"
            cmd = ["cliclick", f"{op}:{ix},{iy}"]
        elif plat == "linux" and _session_type() == "wayland" and _which("ydotool"):
            button_code = {"left": "0xC0", "middle": "0xC2", "right": "0xC1"}[button]
            move = ["ydotool", "mousemove", "--absolute", "-x", str(ix), "-y", str(iy)]
            rc, out, err = self._try(move)
            if rc != 0:
                return _json({"ok": False, "error": err or out or f"exit {rc}", "cmd": move})
            cmd = ["ydotool", "click", "--repeat", str(repeat), button_code]
        elif plat == "linux" and _which("xdotool"):
            button_id = {"left": "1", "middle": "2", "right": "3"}[button]
            cmd = ["xdotool", "mousemove", str(ix), str(iy), "click", "--repeat", str(repeat), button_id]
        else:
            return self._backend_unavailable("click")
        return self._exec_input(cmd, extra=note)

    def move(self, *, x: int, y: int | None = None, raw: bool = False) -> str:
        plat = _platform()
        try:
            x, y = _coerce_point(x, y)
        except ValueError as exc:
            return _json({"ok": False, "error": str(exc)})
        if int(x) < 0 or int(y) < 0:
            return _json({"ok": False, "error": f"negative coordinates not allowed ({x},{y})"})
        ix, iy, note = self._map_xy(x, y, raw=raw)
        if self._is_remote():
            _name, conn = self._active_connection()
            return self._remote_pyautogui(conn, f"pyautogui.moveTo({ix}, {iy})", note=note)
        if plat == "macos" and _which("cliclick"):
            cmd = ["cliclick", f"m:{ix},{iy}"]
        elif plat == "linux" and _session_type() == "wayland" and _which("ydotool"):
            cmd = ["ydotool", "mousemove", "--absolute", "-x", str(ix), "-y", str(iy)]
        elif plat == "linux" and _which("xdotool"):
            cmd = ["xdotool", "mousemove", str(ix), str(iy)]
        else:
            return self._backend_unavailable("mouse-move")
        return self._exec_input(cmd, extra=note)

    def left_click_drag(self, *, start_x: Any, start_y: Any = None, end_x: Any = None,
                        end_y: Any = None, raw: bool = False) -> str:
        plat = _platform()
        try:
            start_x, start_y = _coerce_point(start_x, start_y)
            end_x, end_y = _coerce_point(end_x, end_y)
        except ValueError as exc:
            return _json({"ok": False, "error": str(exc)})
        for v in (start_x, start_y, end_x, end_y):
            if int(v) < 0:
                return _json({"ok": False, "error": "negative coordinates not allowed"})
        sx_, sy_, note = self._map_xy(start_x, start_y, raw=raw)
        ex_, ey_, _ = self._map_xy(end_x, end_y, raw=raw)
        if self._is_remote():
            _name, conn = self._active_connection()
            return self._remote_pyautogui(
                conn,
                f"pyautogui.moveTo({sx_}, {sy_}); pyautogui.dragTo({ex_}, {ey_}, duration=0.5, button='left')",
                note=note,
            )
        if plat == "macos" and _which("cliclick"):
            cmd = ["cliclick", f"dd:{sx_},{sy_}", f"dm:{ex_},{ey_}", f"du:{ex_},{ey_}"]
        elif plat == "linux" and _session_type() == "wayland" and _which("ydotool"):
            # ydotool encodes press/release in the button byte: 0x40|btn=down,
            # 0x80|btn=up (0xC0|btn = full click). There are no --down/--up flags.
            steps = [
                ["ydotool", "mousemove", "--absolute", "-x", str(sx_), "-y", str(sy_)],
                ["ydotool", "click", "0x40"],
                ["ydotool", "mousemove", "--absolute", "-x", str(ex_), "-y", str(ey_)],
                ["ydotool", "click", "0x80"],
            ]
            return self._exec_sequence(steps, extra=note)
        elif plat == "linux" and _which("xdotool"):
            cmd = ["xdotool", "mousemove", str(sx_), str(sy_), "mousedown", "1",
                   "mousemove", str(ex_), str(ey_), "mouseup", "1"]
        else:
            return self._backend_unavailable("drag")
        return self._exec_input(cmd, extra=note)

    def mouse_down(self, *, x: Any = None, y: Any = None, button: str = "left", raw: bool = False) -> str:
        return self._mouse_press(x=x, y=y, button=button, raw=raw, press=True)

    def mouse_up(self, *, x: Any = None, y: Any = None, button: str = "left", raw: bool = False) -> str:
        return self._mouse_press(x=x, y=y, button=button, raw=raw, press=False)

    def _mouse_press(self, *, x: int, y: int, button: str, raw: bool, press: bool) -> str:
        plat = _platform()
        button = str(button or "left").strip().lower()
        if button not in ("left", "right", "middle"):
            return _json({"ok": False, "error": f"unknown button {button!r} (use left/right/middle)"})
        # Coordinates are OPTIONAL here ("press where the pointer is"). ABSENT is
        # None or the legacy -1 sentinel — both mean the same and are mapped to
        # None BEFORE coercion, so mouse_down(x="663, 500") recovers the pair
        # exactly like click does instead of treating the sentinel as a
        # conflicting y (v6.81.1 review round 5). A provided coordinate goes
        # through the same normalizer as click/move.
        x = None if (isinstance(x, int) and x == -1) else x
        y = None if (isinstance(y, int) and y == -1) else y
        if x is None and y is None:
            has_xy = False
            x = y = -1
        else:
            try:
                x, y = _coerce_point(x, y)
            except ValueError as exc:
                return _json({"ok": False, "error": str(exc)})
            has_xy = int(x) >= 0 and int(y) >= 0
        ix = iy = 0
        note: dict[str, Any] = {}
        if has_xy:
            ix, iy, note = self._map_xy(x, y, raw=raw)
        if self._is_remote():
            _name, conn = self._active_connection()
            fn = "mouseDown" if press else "mouseUp"
            code = (f"pyautogui.{fn}(x={ix}, y={iy}, button={button!r})" if has_xy
                    else f"pyautogui.{fn}(button={button!r})")
            return self._remote_pyautogui(conn, code, note=note)
        if plat == "macos" and _which("cliclick"):
            if button != "left":
                return _json({"ok": False, "error": "mouse_down/up supports only the left button on macOS (cliclick dd/du)"})
            op = "dd" if press else "du"
            target = f"{op}:{ix},{iy}" if has_xy else f"{op}:."
            cmd = ["cliclick", target]
        elif plat == "linux" and _session_type() == "wayland" and _which("ydotool"):
            # press = 0x40|btn, release = 0x80|btn (left +0, right +1, middle +2).
            offset = {"left": 0, "right": 1, "middle": 2}[button]
            code = (0x40 if press else 0x80) | offset
            steps = []
            if has_xy:
                steps.append(["ydotool", "mousemove", "--absolute", "-x", str(ix), "-y", str(iy)])
            steps.append(["ydotool", "click", f"0x{code:02X}"])
            return self._exec_sequence(steps, extra=note)
        elif plat == "linux" and _which("xdotool"):
            button_id = {"left": "1", "middle": "2", "right": "3"}[button]
            cmd = ["xdotool"]
            if has_xy:
                cmd += ["mousemove", str(ix), str(iy)]
            cmd += ["mousedown" if press else "mouseup", button_id]
        else:
            return self._backend_unavailable("mouse press")
        return self._exec_input(cmd, extra=note)

    def cursor_position(self) -> str:
        plat = _platform()
        if self._is_remote():
            _name, conn = self._active_connection()
            return self._remote_pyautogui(conn, "print('pos', pyautogui.position())")
        if plat == "macos" and _which("cliclick"):
            try:
                rc, out, err = _run(["cliclick", "p"])
            except Exception as exc:  # noqa: BLE001
                return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if rc == 0:
                nums = [p for p in out.replace(",", " ").split() if p.lstrip("-").isdigit()]
                if len(nums) >= 2:
                    return _json({"ok": True, "x": int(nums[0]), "y": int(nums[1]),
                                  "coordinate_space": "input (macOS logical points)"})
            return _json({"ok": False, "error": err.strip() or out.strip() or f"exit {rc}"})
        if plat == "linux" and _which("xdotool"):
            try:
                rc, out, err = _run(["xdotool", "getmouselocation"])
            except Exception as exc:  # noqa: BLE001
                return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if rc == 0:
                fields = dict(
                    part.split(":", 1) for part in out.split() if ":" in part
                )
                try:
                    return _json({"ok": True, "x": int(fields.get("x", "0")), "y": int(fields.get("y", "0")),
                                  "coordinate_space": "input (X11 pixels)"})
                except ValueError:
                    pass
            return _json({"ok": False, "error": err.strip() or out.strip() or f"exit {rc}"})
        return self._backend_unavailable("cursor-position")

    # ------------------------------------------------------------------
    # keyboard actions
    # ------------------------------------------------------------------

    def type_text(self, *, text: str, interval_ms: int = 0) -> str:
        plat = _platform()
        text = str(text or "")
        if self._is_remote():
            _name, conn = self._active_connection()
            interval = max(0, int(interval_ms or 0)) / 1000.0
            typewrite_code = f"pyautogui.typewrite({text!r}, interval={interval!r})"
            backend = str(conn.get("backend") or "").lower()
            # typewrite silently drops non-ASCII, so paste it via the in-VM clipboard
            # (base64-safe), like official OSWorld agents; ssh_macos `t:` handles unicode.
            #
            # ASCII is NOT automatically safe either: pyautogui types shift-symbols by
            # pressing the unshifted key with SHIFT held, and on the guest's keymap that
            # mis-resolves for the angle brackets. Measured byte-for-byte in the
            # v6.81.1 OSWorld run (task gimp/62f7fd55): the agent typed
            # `<?xml version="1.0"...` and the file on disk read `>?xml version="1.0"...`
            # — every `<` arrived as `>` (hexdump `3e` where `3c` was sent), silently
            # corrupting any XML/HTML/shell-redirect the agent types. Route those through
            # the same clipboard path instead of trusting the keystroke translation.
            # Multi-line and long payloads also go via clipboard: typewrite presses
            # Enter for each "\n" (submitting forms / running partial shell lines)
            # and long keystroke streams drop characters under load. Forensics on
            # the v6.81.1 run counted lost paragraph breaks and leading spaces in
            # retyped documents as concrete task failures.
            if backend == "osworld_http" and (any(ord(ch) > 127 for ch in text)
                                              or "<" in text or ">" in text
                                              or "\n" in text or len(text) > 200):
                b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                # Chord selection happens INSIDE this one guest call: a terminal
                # ignores Ctrl+V (it wants Ctrl+Shift+V) yet the hotkey itself
                # succeeds, so a terminal paste would report ok:true having typed
                # nothing — and sending both chords everywhere would paste twice
                # in an ordinary field. One round trip, no guessing on the host.
                clip_code = (
                    "import base64, subprocess, pyperclip; "
                    f"pyperclip.copy(base64.b64decode('{b64}').decode('utf-8')); "
                    "_cls = ''\n"
                    "try:\n"
                    "    _cls = subprocess.run(['xdotool','getactivewindow','getwindowclassname'],"
                    "capture_output=True,text=True,timeout=5).stdout.strip().lower()\n"
                    "except Exception:\n"
                    "    _cls = ''\n"
                    "_term = any(t in _cls for t in "
                    "('terminal','xterm','konsole','alacritty','kitty'))\n"
                    "pyautogui.hotkey('ctrl','shift','v') if _term else pyautogui.hotkey('ctrl','v')\n"
                    "print('paste_target=' + ('terminal' if _term else 'field'))"
                )
                out = self._remote_pyautogui(
                    conn, clip_code, note={"method": "clipboard"}, timeout=60,
                )
                try:
                    if json.loads(out).get("ok"):
                        return out
                except Exception:
                    pass
                # Clipboard path failed (pyperclip/xclip absent) — typewrite fallback.
                return self._remote_pyautogui(
                    conn, typewrite_code, note={"method": "typewrite", "clipboard_fallback": True}, timeout=60,
                )
            return self._remote_pyautogui(conn, typewrite_code, note={"method": "typewrite"}, timeout=60)
        if plat == "macos" and _which("cliclick"):
            cmd = ["cliclick", f"t:{text}"]
        elif plat == "linux" and _session_type() == "wayland" and (_which("ydotool") or _which("wtype")):
            # `--` terminates option parsing so text starting with '-' is typed,
            # not parsed as flags.
            if _which("ydotool"):
                cmd = ["ydotool", "type", "--key-delay", str(max(0, int(interval_ms or 0))), "--", text]
            else:
                cmd = ["wtype", "--", text]
        elif plat == "linux" and _which("xdotool"):
            cmd = ["xdotool", "type", "--delay", str(max(0, int(interval_ms or 0))), "--", text]
        else:
            return self._backend_unavailable("typing")
        # Long text / per-char delay can exceed the 10s input cap; allow up to
        # the registered 20s tool budget.
        return self._exec_input(cmd, timeout=18)

    def _parse_combo(self, combo: str) -> tuple[list[str], str, str]:
        """(modifiers, base_key, error). Split ONLY on '+' (key names contain '-').

        The base key is alias-normalized when a lowercase alias matches; the
        ORIGINAL token is preserved otherwise (X11 keysyms like ``F5`` or
        ``XF86AudioPlay`` are case-sensitive). The original base is stashed on
        ``self._last_base_raw`` for backend-specific case handling.
        """
        raw_parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not raw_parts:
            return [], "", "keys is required"
        parts = [p.lower() for p in raw_parts]
        mod_tokens = parts[:-1]
        unknown = [t for t in mod_tokens if t not in _ALL_MODS]
        if unknown:
            return [], "", f"unknown key modifier(s): {unknown}"
        self._last_base_raw = raw_parts[-1]
        base = _KEY_ALIASES.get(parts[-1], parts[-1])
        return mod_tokens, base, ""

    def key(self, *, keys: str) -> str:
        combo = str(keys or "").strip()
        if not combo:
            return _json({"ok": False, "error": "keys is required"})
        # WHITESPACE = a SEQUENCE of combos, executed in order ("Left Left",
        # "shift+Right shift+Right"). Splitting only on '+' made both shapes fail,
        # and fail differently: the modified form errored with a nonsense modifier
        # ("unknown key modifier(s): ['right shift']"), while the bare form
        # SILENTLY no-opped — `pyautogui.press('left left')` is simply ignored, so
        # the agent believed a keypress happened that never did. Both were measured
        # in the v6.81.1 OSWorld run; the silent one is the more dangerous.
        steps = combo.split()
        if len(steps) > 1:
            results = []
            for step in steps:
                out = self.key(keys=step)
                results.append(json.loads(out))
                if not results[-1].get("ok"):
                    return _json({"ok": False, "error": f"step {step!r} failed: "
                                                        f"{results[-1].get('error')}",
                                  "steps_done": len(results) - 1, "steps_total": len(steps)})
            return _json({"ok": True, "sequence": steps, "steps": len(steps)})
        plat = _platform()
        mods, base, err = self._parse_combo(combo)
        if err:
            return _json({"ok": False, "error": err})
        if self._is_remote():
            _name, conn = self._active_connection()
            tokens = [_PYAUTOGUI_MODS.get(m, m) for m in mods]
            base_raw = getattr(self, "_last_base_raw", base)
            b = _PYAUTOGUI_BASE_ALIASES.get(base, base_raw if len(base_raw) == 1 else base)
            tokens.append(b)
            code = f"pyautogui.press({tokens[0]!r})" if len(tokens) == 1 else "pyautogui.hotkey(" + ", ".join(repr(t) for t in tokens) + ")"
            return self._remote_pyautogui(conn, code)
        if plat == "macos" and _which("cliclick"):
            non_mac = [t for t in mods if t not in _MAC_MODS]
            if non_mac:
                return _json({"ok": False, "error": f"modifier(s) {non_mac} are not available on macOS"})
            inner = f"t:{base}" if len(base) == 1 else f"kp:{base}"
            if mods:
                mac_mods = [_MAC_MODS[t] for t in mods]
                cmd = ["cliclick", f"kd:{','.join(mac_mods)}", inner, f"ku:{','.join(mac_mods)}"]
            else:
                cmd = ["cliclick", inner]
        elif plat == "linux" and _session_type() == "wayland":
            # ydotool `key` accepts ONLY raw KEYCODE:STATE pairs (e.g. 28:1
            # 28:0) and treats anything else as a no-op delay with exit 0 —
            # a silent fake success. Be honest instead: key combos are
            # unsupported on Wayland; use type_text (ydotool/wtype) for text.
            return _json({
                "ok": False,
                "error": (
                    "key combos are unsupported on Wayland (ydotool key takes raw "
                    "keycodes only); use type_text for text input or an X11 session"
                ),
                "capabilities": self._capabilities(),
            })
        elif plat == "linux" and _which("xdotool"):
            # Aliases map to canonical keysyms; unknown multi-char tokens keep
            # their ORIGINAL case (X11 keysyms like XF86AudioPlay, KP_Enter).
            base_raw = getattr(self, "_last_base_raw", base)
            x_base = _X11_KEY_ALIASES.get(base, base if len(base) == 1 else base_raw)
            x_mods = [_X11_KEY_ALIASES.get(m, m) for m in mods]
            x_combo = "+".join(x_mods + [x_base])
            cmd = ["xdotool", "key", x_combo]
        else:
            return self._backend_unavailable("key")
        return self._exec_input(cmd)

    def hold_key(self, *, keys: str, duration_ms: int = 500) -> str:
        combo = str(keys or "").strip()
        if not combo:
            return _json({"ok": False, "error": "keys is required"})
        duration = max(50, min(10_000, int(duration_ms or 500)))
        plat = _platform()
        tokens = [p.strip() for p in combo.split("+") if p.strip()]
        if not tokens:
            return _json({"ok": False, "error": "keys is required"})
        if self._is_remote():
            _name, conn = self._active_connection()
            pya = [_PYAUTOGUI_MODS.get(t.lower(), (t if len(t) == 1 else t.lower())) for t in tokens]
            downs = "; ".join(f"pyautogui.keyDown({t!r})" for t in pya)
            ups = "; ".join(f"pyautogui.keyUp({t!r})" for t in reversed(pya))
            return self._remote_pyautogui(conn, f"{downs}; time.sleep({duration/1000.0!r}); {ups}", timeout=max(12, duration // 1000 + 8))
        if plat == "macos" and _which("cliclick"):
            # cliclick `kd:`/`ku:` accept ONLY modifiers (cmd/ctrl/alt/shift/fn);
            # `kp:` is press-AND-RELEASE — it cannot hold. Be honest: only
            # pure-modifier combos are holdable on macOS. `w:` is WAIT in ms.
            lowered = [t.lower() for t in tokens]
            non_mods = [t for t, low in zip(tokens, lowered) if low not in _MAC_MODS]
            if non_mods:
                return _json({
                    "ok": False,
                    "error": (
                        f"cliclick cannot hold non-modifier key(s) {non_mods}; "
                        "only modifier combos (cmd/ctrl/alt/shift/fn) are holdable on macOS"
                    ),
                })
            held = ",".join(_MAC_MODS[low] for low in lowered)
            cmd = ["cliclick", f"kd:{held}", f"w:{duration}", f"ku:{held}"]
            return self._exec_input(cmd, timeout=max(12, duration // 1000 + 4))
        if plat == "linux" and _session_type() == "wayland":
            # ydotool's key syntax is raw KEYCODE:STATE pairs and silently
            # ignores anything else; no honest hold contract — unsupported.
            return _json({"ok": False, "error": "hold_key is unsupported on Wayland (ydotool has no reliable keydown/keyup names)"})
        if plat == "linux" and _which("xdotool"):
            mods, base, err = self._parse_combo(combo)
            if err:
                return _json({"ok": False, "error": err})
            # Same case-preservation as key(): unknown multi-char tokens keep
            # their ORIGINAL case (X11 keysyms are case-sensitive).
            base_raw = getattr(self, "_last_base_raw", base)
            x_base = _X11_KEY_ALIASES.get(base, base if len(base) == 1 else base_raw)
            x_mods = [_X11_KEY_ALIASES.get(m, m) for m in mods]
            x_combo = "+".join(x_mods + [x_base]) if (x_mods or x_base) else x_base
            steps = [["xdotool", "keydown", x_combo]]
            return self._hold_then_release(steps, [["xdotool", "keyup", x_combo]], duration)
        return self._backend_unavailable("hold-key")

    def _hold_then_release(self, down_steps: list[list[str]], up_steps: list[list[str]],
                           duration_ms: int) -> str:
        for cmd in down_steps:
            rc, out, err = self._try(cmd)
            if rc != 0:
                return _json({"ok": False, "error": err or out or f"exit {rc}", "cmd": cmd})
        time.sleep(min(10.0, duration_ms / 1000.0))
        for cmd in up_steps:
            rc, out, err = self._try(cmd)
            if rc != 0:
                return _json({"ok": False, "error": err or out or f"exit {rc}", "cmd": cmd})
        return _json({"ok": True, "held_ms": duration_ms})

    def wait(self, *, ms: int = 500) -> str:
        delay = max(0, min(10_000, int(ms or 0)))
        time.sleep(delay / 1000.0)
        return _json({"ok": True, "waited_ms": delay})

    # ------------------------------------------------------------------
    # scroll / windows / accessibility
    # ------------------------------------------------------------------

    def scroll(self, *, clicks: int = 3, direction: str = "down") -> str:
        direction = str(direction or "down").lower()
        amount = max(1, min(20, abs(int(clicks or 1))))
        plat = _platform()
        if self._is_remote():
            _name, conn = self._active_connection()
            # X11 pyautogui = one wheel detent per unit (like the local path): 1:1, no multiplier.
            if direction in ("up", "down"):
                delta = amount if direction == "up" else -amount
                return self._remote_pyautogui(conn, f"pyautogui.scroll({delta})")
            if direction in ("left", "right"):
                delta = amount if direction == "right" else -amount
                return self._remote_pyautogui(conn, f"pyautogui.hscroll({delta})")
            return _json({"ok": False, "error": f"unknown scroll direction {direction!r} (use up/down/left/right)"})
        if plat == "macos":
            # cliclick has NO scroll-wheel command — its `w:` is WAIT. Faking it
            # returned ok:true while nothing scrolled. Be honest: page via the
            # `key` tool (page-down / arrow keys) on macOS instead.
            return _json({
                "ok": False,
                "error": "scroll is unsupported on macOS (no cliclick scroll-wheel command); "
                         "use the key tool with page-down/page-up or arrow keys instead",
                "capabilities": self._capabilities(),
            })
        if plat == "linux" and _session_type() == "wayland" and _which("ydotool"):
            # ydotool wheel: positive y = up, negative = down (REL_WHEEL units).
            delta = amount if direction == "up" else -amount
            if direction not in ("up", "down"):
                return _json({"ok": False, "error": f"unsupported scroll direction {direction!r} on Wayland (use up/down)"})
            cmd = ["ydotool", "mousemove", "--wheel", "-x", "0", "-y", str(delta)]
            return self._exec_input(cmd)
        if plat == "linux" and _which("xdotool"):
            button = {"down": "5", "up": "4", "left": "6", "right": "7"}.get(direction)
            if button is None:
                return _json({"ok": False, "error": f"unknown scroll direction {direction!r} (use up/down/left/right)"})
            cmd = ["xdotool", "click", "--repeat", str(amount), button]
            return self._exec_input(cmd)
        return self._backend_unavailable("scroll")

    def window_list(self) -> str:
        plat = _platform()
        if self._is_remote():
            _name, conn = self._active_connection()
            if conn.get("disabled"):
                return self._disabled_connection_error(_name, conn)
            backend = str(conn.get("backend") or "").lower()
            if backend == "osworld_http":
                return self.remote_exec(command='DISPLAY=:0 wmctrl -lG 2>/dev/null || wmctrl -l 2>/dev/null || true', timeout=15)
            if backend == "ssh_macos":
                rc, out, err = self._ssh_run(conn, 'osascript -e \'tell application "System Events" to get the name of every process whose background only is false\'', timeout=15)
                if rc == 0:
                    return _json({"ok": True, "backend": backend, "windows": [p.strip() for p in out.split(",") if p.strip()]})
                return _json({"ok": False, "backend": backend, "error": err.strip() or out.strip() or f"exit {rc}"})
        if plat == "macos" and _which("osascript"):
            script = (
                'tell application "System Events" to get the name of every process '
                'whose background only is false'
            )
            try:
                rc, stdout, stderr = _run(["osascript", "-e", script])
            except Exception as exc:  # noqa: BLE001
                return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if rc == 0:
                return _json({"ok": True, "platform": plat, "windows": [p.strip() for p in stdout.split(",") if p.strip()]})
            return _json({"ok": False, "error": stderr.strip() or stdout.strip() or f"exit {rc}"})
        if plat == "linux" and _which("wmctrl"):
            try:
                rc, stdout, stderr = _run(["wmctrl", "-l"])
            except Exception as exc:  # noqa: BLE001
                return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if rc == 0:
                windows = [line.strip() for line in stdout.splitlines() if line.strip()]
                return _json({"ok": True, "platform": plat, "windows": windows})
            return _json({"ok": False, "error": stderr.strip() or stdout.strip() or f"exit {rc}"})
        return self._backend_unavailable("window listing")

    _AX_SCRIPT = """
on run
  set out to ""
  tell application "System Events"
    set frontProc to first process whose frontmost is true
    set procName to name of frontProc
    tell frontProc
      if (count of windows) is 0 then return "PROC\\t" & procName
      set theWindow to window 1
      set out to "PROC\\t" & procName & "\\nWIN\\t" & (name of theWindow)
      set elems to entire contents of theWindow
      set maxN to %d
      set n to 0
      repeat with el in elems
        if n is greater than or equal to maxN then exit repeat
        try
          set elRole to role of el
          if elRole is in {"AXButton", "AXTextField", "AXTextArea", "AXLink", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXComboBox", "AXMenuButton", "AXTab", "AXMenuItem"} then
            set elPos to position of el
            set elSize to size of el
            set elTitle to ""
            try
              set elTitle to title of el
            end try
            if elTitle is "" then
              try
                set elTitle to name of el
              end try
            end if
            set out to out & "\\nEL\\t" & elRole & "\\t" & elTitle & "\\t" & (item 1 of elPos) & "\\t" & (item 2 of elPos) & "\\t" & (item 1 of elSize) & "\\t" & (item 2 of elSize)
            set n to n + 1
          end if
        end try
      end repeat
    end tell
  end tell
  return out
end run
"""

    def ax_tree(self) -> str:
        """Set-of-marks accessibility snapshot of the frontmost window.

        macOS: numbered interactive elements (role/title/center in INPUT
        coordinates — pass them with raw=true, or remap to screenshot space
        yourself). Falls back to the visible-process list when the AX walk
        fails (TCC denied, AppleScript timeout, no window). Linux: process
        list only (a full AT-SPI walk is out of substrate-thin scope).
        """
        plat = _platform()
        if self._is_remote():
            _name, conn = self._active_connection()
            if conn.get("disabled"):
                return self._disabled_connection_error(_name, conn)
            backend = str(conn.get("backend") or "").lower()
            if backend == "osworld_http":
                target = self._connection_target(conn)
                if not target:
                    return _json({"ok": False, "error": "osworld_http connection has no target/target_file"})
                try:
                    with urllib.request.urlopen(target + "/accessibility", timeout=20) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                    return _json({"ok": True, "backend": backend, "accessibility_tree": raw[:80_000], "truncated": len(raw) > 80_000})
                except Exception as exc:  # noqa: BLE001
                    return _json({"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}"})
            # Remote macOS AX over SSH would need a more careful per-target TCC
            # setup; degrade honestly instead of faking a set-of-marks tree.
            return self.window_list()
        if plat == "macos" and _which("osascript"):
            script = self._AX_SCRIPT % _AX_MAX_ELEMENTS
            try:
                rc, stdout, stderr = _run(["osascript", "-e", script], timeout=20)
            except Exception as exc:  # noqa: BLE001
                rc, stdout, stderr = 1, "", f"{type(exc).__name__}: {exc}"
            if rc == 0 and stdout.strip():
                marks: list[dict[str, Any]] = []
                proc_name = ""
                window_name = ""
                for line in stdout.splitlines():
                    parts = line.rstrip("\n").split("\t")
                    if parts[0] == "PROC" and len(parts) > 1:
                        proc_name = parts[1]
                    elif parts[0] == "WIN" and len(parts) > 1:
                        window_name = parts[1]
                    elif parts[0] == "EL" and len(parts) >= 7:
                        try:
                            x, y, w, h = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
                        except ValueError:
                            continue
                        marks.append({
                            "id": len(marks) + 1,
                            "role": parts[1].replace("AX", "", 1),
                            "title": parts[2],
                            "center_x": x + w // 2,
                            "center_y": y + h // 2,
                            "x": x, "y": y, "w": w, "h": h,
                        })
                return _json({
                    "ok": True, "platform": plat, "frontmost": proc_name,
                    "window": window_name, "marks": marks,
                    "truncated": len(marks) >= _AX_MAX_ELEMENTS,
                    "coordinate_note": (
                        "mark coordinates are INPUT-space (macOS logical points): "
                        "click them with raw=true."
                    ),
                })
            # Honest fallback: process list (the old best-effort surface).
            fb_rc, fb_out, fb_err = self._try([
                "osascript", "-e",
                'tell application "System Events" to get the name of every process whose background only is false',
            ])
            if fb_rc == 0:
                return _json({
                    "ok": True, "platform": plat, "marks": [],
                    "visible_processes": [p.strip() for p in fb_out.split(",") if p.strip()],
                    "degraded": "AX element walk failed; returned process list only "
                                f"({(stderr or '').strip() or 'no window / AX denied'})",
                })
            return _json({"ok": False, "error": (fb_err or stderr or "AX walk failed").strip()})
        if plat == "linux":
            if _which("wmctrl"):
                rc, out, err = self._try(["wmctrl", "-l"])
                if rc == 0:
                    return _json({
                        "ok": True, "platform": plat, "marks": [],
                        "windows": [line.strip() for line in out.splitlines() if line.strip()],
                        "degraded": "set-of-marks unsupported on Linux (no AT-SPI walk); window list only",
                    })
            return self._backend_unavailable("accessibility tree")
        return self._backend_unavailable("accessibility tree")

    # ------------------------------------------------------------------
    # exec helpers
    # ------------------------------------------------------------------

    def _try(self, cmd: list[str], *, timeout: int = _TIMEOUT_SEC) -> tuple[int, str, str]:
        try:
            return _run(cmd, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return 1, "", f"{type(exc).__name__}: {exc}"

    def _exec_input(self, cmd: list[str], *, timeout: int = _TIMEOUT_SEC,
                    extra: dict[str, Any] | None = None) -> str:
        rc, stdout, stderr = self._try(cmd, timeout=timeout)
        if rc != 0:
            return _json({"ok": False, "error": stderr.strip() or stdout.strip() or f"exit {rc}", "cmd": cmd})
        payload: dict[str, Any] = {"ok": True, "cmd": cmd}
        if extra:
            payload.update(extra)
        return _json(payload)

    def _exec_sequence(self, steps: list[list[str]], *, extra: dict[str, Any] | None = None) -> str:
        for cmd in steps:
            rc, stdout, stderr = self._try(cmd)
            if rc != 0:
                return _json({"ok": False, "error": stderr.strip() or stdout.strip() or f"exit {rc}", "cmd": cmd})
        payload: dict[str, Any] = {"ok": True, "steps": len(steps)}
        if extra:
            payload.update(extra)
        return _json(payload)

    def remote_exec(self, *, command: str, timeout: int = 60) -> str:
        """Run a shell command on the active remote connection.

        For local backend, this refuses to run so existing local computer-use does
        not become a general shell tool. Use ordinary Ouroboros shell/file tools
        for local work; remote_exec is for inspecting a selected remote machine.
        """
        _name, conn = self._active_connection()
        if conn.get("disabled"):
            return self._disabled_connection_error(_name, conn)
        backend = str(conn.get("backend") or "local").lower()
        cmd = str(command or "").strip()
        if not cmd:
            return _json({"ok": False, "error": "command is required"})
        timeout = max(5, min(300, int(timeout or 60)))
        try:
            if backend == "osworld_http":
                out = self._osworld_execute(conn, ["bash", "-lc", cmd], timeout=timeout)
                ok, err = _osworld_result_ok(out)
                payload: dict[str, Any] = {"ok": ok, "backend": backend, "status": out["status"], "result": out["result"]}
                if not ok:
                    payload["error"] = err
                return _json(payload)
            if backend == "ssh_macos":
                rc, stdout, stderr = self._ssh_run(conn, cmd, timeout=timeout)
                return _json({"ok": rc == 0, "backend": backend, "returncode": rc, "output": stdout, "error": stderr})
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}"})
        return _json({"ok": False, "backend": backend, "error": "remote_exec requires an active remote backend"})


def register(api: Any) -> None:
    impl = _ComputerUse(api)
    _xy = {"x": {"type": "integer"}, "y": {"type": "integer"}, "raw": {"type": "boolean", "default": False}}
    api.register_tool("list_connections", lambda: impl.list_connections(), description="List configured local/remote computer-use connections and the active backend.", schema={"type": "object", "properties": {}}, timeout_sec=5)
    api.register_tool("add_connection", impl.add_connection, description="Add or update a remote computer-use connection (no secrets stored; SSH uses existing ssh config/agent).", schema={"type": "object", "properties": {"name": {"type": "string"}, "backend": {"type": "string"}, "target": {"type": "string", "default": ""}, "target_file": {"type": "string", "default": ""}, "host": {"type": "string", "default": ""}, "user": {"type": "string", "default": ""}, "port": {"type": "integer", "default": 22}, "ssh_alias": {"type": "string", "default": ""}, "enabled": {"type": "boolean", "default": True}, "activate": {"type": "boolean", "default": False}}, "required": ["name", "backend"]}, timeout_sec=10)
    api.register_tool("test_connection", impl.test_connection, description="Run a safe non-mutating health check for a configured connection.", schema={"type": "object", "properties": {"name": {"type": "string", "default": ""}}}, timeout_sec=30)
    api.register_tool("activate_connection", impl.activate_connection, description="Select a configured connection as the active target for screenshot/input tools.", schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, timeout_sec=5)
    api.register_tool("use_local", lambda: impl.use_local(), description="Return computer-use tools to the local desktop backend.", schema={"type": "object", "properties": {}}, timeout_sec=5)
    api.register_tool("clear_active_connection", lambda: impl.clear_active_connection(), description="Alias for use_local: clear any active remote connection.", schema={"type": "object", "properties": {}}, timeout_sec=5)
    api.register_tool("capabilities", lambda: impl.capabilities(), description="Report available computer-use backends, session type, and coordinate contract.", schema={"type": "object", "properties": {}}, timeout_sec=5)
    api.register_tool("screenshot", impl.screenshot, description="Capture the desktop, downscale to <=WXGA, persist the image->input coordinate transform, and return the PNG path. The image is attached to the conversation automatically — no view_image call is needed to see it.", schema={"type": "object", "properties": {"job_id": {"type": "string", "default": "manual"}, "max_width": {"type": "integer", "default": _MAX_IMAGE_W}, "max_height": {"type": "integer", "default": _MAX_IMAGE_H}}}, timeout_sec=25)
    # y is deliberately NOT in `required` for click/move: models recurringly emit the
    # whole point inside x ("663, 500") with y omitted, and a required-field rejection
    # then fires BEFORE the handler's _coerce_point can recover the pair.
    api.register_tool("click", impl.click, description="Click at screenshot-space coordinates (auto-remapped; raw=true for native input space).", schema={"type": "object", "properties": {**_xy, "button": {"type": "string", "default": "left"}, "double": {"type": "boolean", "default": False}, "triple": {"type": "boolean", "default": False}}, "required": ["x"]}, timeout_sec=10)
    # Models call these names unprompted (111 such calls in one OSWorld run, all
    # previously "Unknown tool"): register them as thin aliases of click.
    api.register_tool("double_click", lambda **kw: impl.click(**{**kw, "double": True}), description="Alias of click with double=true.", schema={"type": "object", "properties": {**_xy, "button": {"type": "string", "default": "left"}}, "required": ["x"]}, timeout_sec=10)
    api.register_tool("triple_click", lambda **kw: impl.click(**{**kw, "triple": True}), description="Alias of click with triple=true.", schema={"type": "object", "properties": {**_xy, "button": {"type": "string", "default": "left"}}, "required": ["x"]}, timeout_sec=10)
    api.register_tool("move", impl.move, description="Move the mouse pointer to screenshot-space coordinates.", schema={"type": "object", "properties": _xy, "required": ["x"]}, timeout_sec=10)
    # start_y/end_y are recoverable from a pair packed into start_x/end_x, so only
    # the x-side fields stay required (same rationale as click/move above).
    api.register_tool("left_click_drag", impl.left_click_drag, description="Press the left button at start coordinates, drag to end coordinates, release.", schema={"type": "object", "properties": {"start_x": {"type": "integer"}, "start_y": {"type": "integer"}, "end_x": {"type": "integer"}, "end_y": {"type": "integer"}, "raw": {"type": "boolean", "default": False}}, "required": ["start_x", "end_x"]}, timeout_sec=15)
    api.register_tool("mouse_down", impl.mouse_down, description="Press and hold a mouse button (left only on macOS), optionally at coordinates.", schema={"type": "object", "properties": {**_xy, "button": {"type": "string", "default": "left"}}}, timeout_sec=10)
    api.register_tool("mouse_up", impl.mouse_up, description="Release a held mouse button, optionally at coordinates.", schema={"type": "object", "properties": {**_xy, "button": {"type": "string", "default": "left"}}}, timeout_sec=10)
    api.register_tool("cursor_position", lambda: impl.cursor_position(), description="Report the current pointer position in native input coordinates.", schema={"type": "object", "properties": {}}, timeout_sec=10)
    api.register_tool("type_text", impl.type_text, description="Type text into the focused application.", schema={"type": "object", "properties": {"text": {"type": "string"}, "interval_ms": {"type": "integer", "default": 0}}, "required": ["text"]}, timeout_sec=20)
    api.register_tool("key", impl.key, description="Press a key or modifier combo (e.g. return, cmd+s, ctrl+l, page-down). Split combos with '+'.", schema={"type": "object", "properties": {"keys": {"type": "string"}}, "required": ["keys"]}, timeout_sec=10)
    api.register_tool("hold_key", impl.hold_key, description="Hold a key or modifier combo for a duration (50-10000 ms).", schema={"type": "object", "properties": {"keys": {"type": "string"}, "duration_ms": {"type": "integer", "default": 500}}, "required": ["keys"]}, timeout_sec=15)
    api.register_tool("scroll", impl.scroll, description="Scroll the active view (X11/Wayland; unsupported on macOS — use key paging).", schema={"type": "object", "properties": {"clicks": {"type": "integer", "default": 3}, "direction": {"type": "string", "default": "down"}}}, timeout_sec=10)
    api.register_tool("wait", impl.wait, description="Sleep up to 10s so the UI can settle before the next observation.", schema={"type": "object", "properties": {"ms": {"type": "integer", "default": 500}}}, timeout_sec=12)
    api.register_tool("window_list", lambda: impl.window_list(), description="List visible desktop windows/processes when a backend is available.", schema={"type": "object", "properties": {}}, timeout_sec=10)
    api.register_tool("ax_tree", lambda: impl.ax_tree(), description="Set-of-marks accessibility snapshot of the frontmost window (macOS; numbered clickable elements) with honest degradation.", schema={"type": "object", "properties": {}}, timeout_sec=25)
    api.register_tool("remote_exec", impl.remote_exec, description="Run a shell command on the active remote connection only; refuses on local backend. OSWorld/Linux guest: each call is a FRESH `bash -lc` starting in $HOME. SSH Mac: each call runs through the remote account's login shell via ssh. Either way this is NOT the visible desktop terminal: it does not inherit that terminal's working directory and leaves nothing in its shell history.", schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 60}}, "required": ["command"]}, timeout_sec=120)
