"""Remote computer-use backends (OSWorld HTTP, SSH macOS) for unix_computer_use.

Verbatim extraction from ``plugin.py`` (v7 stream W). These helpers are dormant
unless a non-local connection is explicitly activated in skill state (or by a
benchmark runner); the default behaviour of the skill remains local
macOS/Linux computer-use. ``_ComputerUse`` mixes this class in, so every method
keeps its exact name, signature and body.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import subprocess
import time
import urllib.request
import uuid
from typing import Any

from .cu_runtime import (
    _MAX_IMAGE_H,
    _MAX_IMAGE_W,
    _MAX_REMOTE_SHOT_BYTES,
    _OSWORLD_PKGS_PREFIX,
    _json,
    _osworld_result_ok,
    _png_dimensions,
    _png_intact,
    _run,
)


class _RemoteBackendMixin:
    """OSWorld HTTP and SSH-macOS execution, screenshot and health-check helpers."""

    def _connection_target(self, conn: dict[str, Any]) -> str:
        target = str(conn.get("target") or "").strip()
        if not target and conn.get("target_file"):
            # Path confinement: only read a target_file that lives inside this
            # skill's OWN state dir (where add_connection / a benchmark runner
            # publishes it). Refuse any path outside it so the tool cannot be
            # used to read arbitrary files elsewhere on disk.
            try:
                candidate = pathlib.Path(str(conn["target_file"])).expanduser().resolve()
                base = self.state_dir.resolve()
                if candidate == base or base in candidate.parents:
                    target = candidate.read_text(encoding="utf-8").strip()
                else:
                    target = ""
            except Exception:
                target = ""
        return target.rstrip("/")

    def _osworld_execute(self, conn: dict[str, Any], command: list[str], *, timeout: int = 60) -> dict[str, Any]:
        target = self._connection_target(conn)
        payload = json.dumps({"command": command, "shell": False}).encode("utf-8")
        req = urllib.request.Request(target + "/execute", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 0) or resp.getcode())
            body = resp.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(body)
        except Exception:
            parsed = body[:1000]
        return {"status": status, "result": parsed}

    @staticmethod
    def _ssh_macos_key_name(key: str) -> str:
        low = str(key or "").strip().lower()
        return {
            "enter": "return", "return": "return", "esc": "esc", "escape": "esc",
            # Input is a PYAUTOGUI key name: its "delete" is forward delete (cliclick fwd-delete).
            "delete": "fwd-delete", "backspace": "delete", "pagedown": "page-down",
            "pageup": "page-up", "down": "arrow-down", "up": "arrow-up",
            "left": "arrow-left", "right": "arrow-right", "winleft": "cmd",
            "super": "cmd", "meta": "cmd",
        }.get(low, key)

    def _ssh_macos_cliclick_for_pyautogui(self, code: str) -> tuple[list[str], str]:
        """Translate the pyautogui snippets this skill emits into cliclick args."""
        text = str(code or "").strip()
        m = re.search(r"pyautogui\.click\((\d+),\s*(\d+),\s*clicks=(\d+).*button=([\"'])([^\"']+)\4", text)
        if m:
            x, y, clicks, button = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(5)
            if button == "right":
                return [f"rc:{x},{y}"], ""
            if button == "middle":
                return [], "middle-click unsupported by cliclick"
            op = "tc" if clicks >= 3 else ("dc" if clicks == 2 else "c")
            return [f"{op}:{x},{y}"], ""
        m = re.search(r"pyautogui\.moveTo\((\d+),\s*(\d+)\).*pyautogui\.dragTo\((\d+),\s*(\d+)", text)
        if m:
            sx, sy, ex, ey = map(int, m.groups())
            return [f"dd:{sx},{sy}", f"dm:{ex},{ey}", f"du:{ex},{ey}"], ""
        m = re.search(r"pyautogui\.moveTo\((\d+),\s*(\d+)\)", text)
        if m:
            return [f"m:{int(m.group(1))},{int(m.group(2))}"], ""
        m = re.search(r"pyautogui\.(mouseDown|mouseUp)\(x=(\d+),\s*y=(\d+),\s*button=([\"'])([^\"']+)\4", text)
        if m:
            fn, x, y, button = m.group(1), int(m.group(2)), int(m.group(3)), m.group(5)
            if button != "left":
                return [], "mouseDown/mouseUp supports only left button via cliclick"
            return [f"{'dd' if fn == 'mouseDown' else 'du'}:{x},{y}"], ""
        m = re.search(r"pyautogui\.(mouseDown|mouseUp)\(button=([\"'])([^\"']+)\2", text)
        if m:
            fn, button = m.group(1), m.group(3)
            if button != "left":
                return [], "mouseDown/mouseUp supports only left button via cliclick"
            return [f"{'dd' if fn == 'mouseDown' else 'du'}:."], ""
        m = re.search(r"pyautogui\.typewrite\((?P<q>[\"'])(?P<txt>.*?)(?P=q),\s*interval=", text)
        if m:
            return [f"t:{m.group('txt')}"], ""
        m = re.search(r"pyautogui\.press\(([\"'])([^\"']+)\1\)", text)
        if m:
            return [f"kp:{self._ssh_macos_key_name(m.group(2))}"], ""
        m = re.search(r"pyautogui\.hotkey\((.*)\)", text)
        if m:
            toks = [t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()]
            if not toks:
                return [], "empty hotkey"
            mods = [self._ssh_macos_key_name(t) for t in toks[:-1]]
            base = self._ssh_macos_key_name(toks[-1])
            if mods:
                held = ",".join(mods)
                return [f"kd:{held}", f"kp:{base}", f"ku:{held}"], ""
            return [f"kp:{base}"], ""
        if "pyautogui.scroll" in text or "pyautogui.hscroll" in text:
            return [], "scroll unsupported via cliclick; use key page-down/page-up"
        return [], f"unsupported pyautogui snippet for ssh_macos/cliclick: {text[:120]}"

    def _remote_pyautogui(self, conn: dict[str, Any], code: str, *, note: dict[str, Any] | None = None, timeout: int = 30) -> str:
        if conn.get("disabled"):
            return self._disabled_connection_error(str(self._read_connections().get("active") or "?"), conn)
        backend = str(conn.get("backend") or "").lower()
        try:
            if backend == "osworld_http":
                wrapped = _OSWORLD_PKGS_PREFIX.format(command=code)
                out = self._osworld_execute(conn, ["python", "-c", wrapped], timeout=timeout)
                ok, err = _osworld_result_ok(out)
                payload: dict[str, Any] = {"ok": ok, "backend": backend, "status": out["status"], "execute_result": out["result"]}
                if not ok:
                    payload["error"] = err
            elif backend == "ssh_macos":
                cliclick_args, err = self._ssh_macos_cliclick_for_pyautogui(code)
                if err:
                    return _json({"ok": False, "backend": backend, "error": err, "code": code})
                remote = "cliclick " + " ".join(shlex.quote(arg) for arg in cliclick_args)
                rc, stdout, stderr = self._ssh_run(conn, remote, timeout=timeout)
                payload = {"ok": rc == 0, "backend": backend, "returncode": rc, "output": stdout, "error": stderr}
            else:
                return _json({"ok": False, "error": f"unsupported remote backend {backend!r}"})
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}", "code": code})
        if note:
            payload.update(note)
        return _json(payload)

    def _remote_screenshot_result(
        self,
        *,
        backend: str,
        raw_path: pathlib.Path,
        max_width: int,
        max_height: int,
        input_w: int,
        input_h: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        px_w, px_h = _png_dimensions(raw_path)
        if px_w <= 0 or px_h <= 0 or not _png_intact(raw_path):
            # Not a fully decodable PNG — don't claim success on garbage; a valid
            # 24-byte header over zero-padded data must fail here, not rounds
            # later as a provider 400. Drop the file.
            try:
                raw_path.unlink()
            except OSError:
                pass
            return _json({"ok": False, "backend": backend, "error": "remote screenshot is not a fully decodable PNG"})
        if input_w <= 0 or input_h <= 0:
            input_w, input_h = px_w, px_h
        max_w = max(320, min(int(max_width or _MAX_IMAGE_W), 4096))
        max_h = max(240, min(int(max_height or _MAX_IMAGE_H), 4096))
        img_path, img_w, img_h = self._downscale(raw_path, max_w, max_h)
        # Path confinement: the downscaled image already lives under the skill's
        # own job dir; return it directly for view_image, never copied elsewhere.
        view_path = img_path
        result: dict[str, Any] = {
            "ok": True,
            "path": str(view_path),
            "backend": backend,
            "image_width": img_w,
            "image_height": img_h,
            "capture_width_px": px_w,
            "capture_height_px": px_h,
            "input_width": input_w,
            "input_height": input_h,
            "downscaled": img_path != raw_path,
            "view_image_ready": True,
            # Typed opt-in for the host's same-round image attachment (v6.81.1).
            # DISTINCT from view_image_ready, which only ever meant "a path you
            # may view manually" — reusing it would retroactively change the
            # contract of every result that already carries it.
            "auto_attach_image": str(view_path),
        }
        if img_path != raw_path:
            result["full_resolution_path"] = str(raw_path)
        if extra:
            result.update(extra)
        if img_w > 0 and img_h > 0 and input_w > 0 and input_h > 0:
            sx = round(input_w / img_w, 6)
            sy = round(input_h / img_h, 6)
            transform = {
                "sx": sx, "sy": sy,
                "image_w": img_w, "image_h": img_h,
                "input_w": input_w, "input_h": input_h,
                "platform": backend, "session": "remote",
                "approx": False, "ts": time.time(),
            }
            self._save_transform(transform)
            result["coord_transform"] = transform
            result["coordinate_note"] = (
                "Pass coordinates read off THIS image directly to click/move/drag — "
                "they are auto-remapped through coord_transform (image -> remote input space)."
            )
        return _json(result)

    def _osworld_screenshot(self, conn: dict[str, Any], *, max_width: int, max_height: int) -> str:
        target = self._connection_target(conn)
        if not target:
            return _json({"ok": False, "error": "osworld_http connection has no target/target_file"})
        out_dir = pathlib.Path(self.api.skill_job_dir("osworld_http")) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"screenshot-{int(time.time())}-{uuid.uuid4().hex[:6]}.png"
        # Bounded re-fetch on a corrupt payload: a truncated body from the guest
        # is transient (mid-write read), but once persisted it used to survive
        # header-only checks and detonate rounds later as a provider 400.
        last_err = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(target + "/screenshot", timeout=20) as resp:
                    data = resp.read(_MAX_REMOTE_SHOT_BYTES + 1)
            except Exception as exc:  # noqa: BLE001
                return _json({"ok": False, "error": f"/screenshot failed: {type(exc).__name__}: {exc}", "backend": "osworld_http"})
            if not data:
                return _json({"ok": False, "error": "/screenshot returned empty body", "backend": "osworld_http"})
            if len(data) > _MAX_REMOTE_SHOT_BYTES:
                return _json({"ok": False, "error": f"/screenshot exceeded {_MAX_REMOTE_SHOT_BYTES} byte cap", "backend": "osworld_http"})
            # Write-then-validate-then-rename: the published path never holds
            # a partially written or undecodable image.
            tmp_path = out_path.with_suffix(".part")
            tmp_path.write_bytes(data)
            if _png_intact(tmp_path):
                tmp_path.rename(out_path)
                break
            last_err = f"undecodable PNG ({len(data)} bytes) on attempt {attempt + 1}/3"
            try:
                tmp_path.unlink()
            except OSError:
                pass
            time.sleep(0.5)
        else:
            return _json({"ok": False, "error": f"screenshot_corrupt: {last_err}", "backend": "osworld_http"})
        px_w, px_h = _png_dimensions(out_path)
        return self._remote_screenshot_result(
            backend="osworld_http",
            raw_path=out_path,
            max_width=max_width,
            max_height=max_height,
            input_w=px_w,
            input_h=px_h,
            # NOT `target`: the bridge URL is control-plane, and putting it in an
            # agent-visible result is how an agent learns where the harness lives.
            # Measured in the v6.81.1 OSWorld run: one agent read the port out of a
            # screenshot result and curled `<bridge>/evaluate` looking for the grader
            # (it failed only because remote_exec runs inside the guest, where that
            # port is not the host's — containment by luck of topology, not design).
            # The host keeps the target in bridge.json for observability.
            extra={"backend_endpoint": "osworld_http"},
        )

    def _test_osworld(self, conn: dict[str, Any], name: str) -> str:
        target = self._connection_target(conn)
        if not target:
            return _json({"ok": False, "connection": name, "backend": "osworld_http", "error": "missing target/target_file"})
        try:
            with urllib.request.urlopen(target + "/screenshot", timeout=10) as resp:
                raw = resp.read(32)
            out = self._osworld_execute(conn, ["python", "-c", "import pyautogui; print(pyautogui.size())"], timeout=20)
            return _json({
                "ok": bool(raw) and _osworld_result_ok(out)[0],
                "connection": name,
                "backend": "osworld_http",
                "target": target,
                "screenshot_bytes_probe": len(raw),
                "execute_probe": out,
            })
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "connection": name, "backend": "osworld_http", "target": target, "error": f"{type(exc).__name__}: {exc}"})

    def _ssh_destination(self, conn: dict[str, Any]) -> list[str]:
        alias = str(conn.get("ssh_alias") or "").strip()
        if alias:
            return [alias]
        host = str(conn.get("host") or "").strip()
        user = str(conn.get("user") or "").strip()
        port = int(conn.get("port") or 22)
        dest = f"{user}@{host}" if user else host
        return ["-p", str(port), dest] if port != 22 else [dest]

    def _ssh_scp_source(self, conn: dict[str, Any], remote_path: str) -> list[str]:
        """scp source args: '-P <port>' (scp uses capital P) plus a SINGLE
        '<destination>:<remote_path>' token (scp needs the source as one arg)."""
        alias = str(conn.get("ssh_alias") or "").strip()
        if alias:
            return [f"{alias}:{remote_path}"]
        host = str(conn.get("host") or "").strip()
        user = str(conn.get("user") or "").strip()
        port = int(conn.get("port") or 22)
        dest = f"{user}@{host}" if user else host
        src = f"{dest}:{remote_path}"
        return ["-P", str(port), src] if port != 22 else [src]

    def _ssh_run(self, conn: dict[str, Any], command: str, *, timeout: int = 30) -> tuple[int, str, str]:
        ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", *self._ssh_destination(conn), command]
        return _run(ssh_cmd, timeout=timeout)

    def _ssh_macos_screenshot(self, conn: dict[str, Any], *, max_width: int, max_height: int) -> str:
        out_dir = pathlib.Path(self.api.skill_job_dir("ssh_macos")) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        remote_path = f"/tmp/ouroboros-shot-{int(time.time())}-{uuid.uuid4().hex[:6]}.png"
        rc, stdout, stderr = self._ssh_run(conn, f"screencapture -x {remote_path!r}", timeout=20)
        if rc != 0:
            return _json({"ok": False, "backend": "ssh_macos", "error": stderr.strip() or stdout.strip() or f"exit {rc}"})
        dest = out_dir / pathlib.Path(remote_path).name
        scp_cmd = ["scp", "-q", *self._ssh_scp_source(conn, remote_path), str(dest)]
        try:
            proc = subprocess.run(scp_cmd, text=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            return _json({"ok": False, "backend": "ssh_macos", "error": f"scp failed: {type(exc).__name__}: {exc}"})
        if proc.returncode != 0 or not dest.exists():
            return _json({"ok": False, "backend": "ssh_macos", "error": proc.stderr.strip() or proc.stdout.strip() or f"scp exit {proc.returncode}"})
        rc, out, _err = self._ssh_run(conn, "osascript -e 'tell application \"Finder\" to get bounds of window of desktop'", timeout=10)
        input_w = input_h = 0
        if rc == 0:
            parts = [p.strip() for p in out.replace(",", " ").split()]
            nums = [int(p) for p in parts if p.lstrip("-").isdigit()]
            if len(nums) >= 4:
                input_w, input_h = nums[2] - nums[0], nums[3] - nums[1]
        return self._remote_screenshot_result(
            backend="ssh_macos",
            raw_path=dest,
            max_width=max_width,
            max_height=max_height,
            input_w=input_w,
            input_h=input_h,
            extra={"host": str(conn.get("ssh_alias") or conn.get("host") or "")},
        )

    def _test_ssh_macos(self, conn: dict[str, Any], name: str) -> str:
        rc, stdout, stderr = self._ssh_run(
            conn,
            "printf 'host='; hostname; printf '\\nuser='; whoami; printf '\\nos='; sw_vers -productVersion 2>/dev/null; printf '\\n'; command -v screencapture; command -v cliclick || true",
            timeout=15,
        )
        ok = rc == 0 and "screencapture" in stdout
        hint = ""
        if rc != 0:
            hint = (
                "SSH auth failed. Put the private key in ~/.ssh/<name>, chmod 600 it, "
                "and add Host/User/IdentityFile to ~/.ssh/config; then retry test_connection."
            )
        elif "cliclick" not in stdout:
            hint = "Install cliclick on the Mac (e.g. brew install cliclick) and grant Accessibility permission."
        return _json({"ok": ok, "connection": name, "backend": "ssh_macos", "output": stdout, "error": stderr, "hint": hint})
