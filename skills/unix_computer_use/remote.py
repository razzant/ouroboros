"""Remote desktop backends of unix_computer_use: OSWorld HTTP guests and Macs over SSH.

Mixed into the plugin's ``_ComputerUse``; the plugin supplies ``_run_cmd``, ``_json_text``,
``_png_dims``, ``_png_ok``, ``_downscale``, ``_save_transform``, ``_display_geometry``, ``_active_connection`` and the
connection registry; the place mixin supplies ``_begin``/``_finish`` (one receipt contract for
every transport). The OSWorld clipboard routing and the fail-closed result reading were measured
on the benchmark (v6.81.x forensics) and must not drift from it. Dormant unless a non-local
connection is active; the local desktop is never touched from here.

Targets on a remote are resolved from the remote's own window inventory (``wmctrl -lpxG`` rows in
an OSWorld guest, ``lsappinfo list`` on a Mac) and activated by window id or pid: a name is never
interpolated into a remote command, and nothing is launched. Every input is wrapped so the same
remote call reads who is in front before and after it and withholds the input when a targeted
application is no longer in front (``__TARGET_LOST__``).
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import shlex
import subprocess
import time
import urllib.request
import uuid
from typing import Any

from .macos_native import parse_info, parse_list, resolve
from .place import parse_wmctrl_rows, parse_x11_front, window_id_int, x11_resolve, x11_same

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
_PYAUTOGUI_MODS = {
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt",
    "shift": "shift",
    "cmd": "winleft", "command": "winleft", "super": "winleft", "meta": "winleft", "win": "winleft",
    "fn": "fn",
}


def _osworld_completion(out):
    """The pinned producer always returns a command exit code, including on failure.

    OSWorld-V2 c261cb57 pins osworld-server a3cc3f0c64e463f020d1a44780307e9b46cbcab1:
    https://github.com/xlang-ai/osworld-server/blob/a3cc3f0c64e463f020d1a44780307e9b46cbcab1/src/http_routes.py#L113-L148
    Its status='success' means subprocess.run returned, NOT that the command succeeded.
    Missing/malformed completion after send is unknown, never a retry signal.
    """
    result = out.get("result")
    if type(out.get("status")) is not int or out["status"] != 200 or not isinstance(result, dict):
        return False, False, "unexpected HTTP status or non-JSON /execute response"
    rc = result.get("returncode")
    if type(rc) is not int:
        return False, False, "missing or non-integer command returncode"
    status = result.get("status", "success")
    if status != "success" or rc != 0:
        return False, status == "success", str(result.get("message") or result.get("error") or
                                               f"guest command status={status!r}, exit {rc}")[:1000]
    return True, True, ""


def _osworld_result_ok(out: dict[str, Any]) -> tuple[bool, str]:
    ok, _completed, error = _osworld_completion(out)
    return ok, error


# Complete kp vocabulary, from cliclick 5.1 KeyPressAction.getSupportedKeycodes:
# https://github.com/BlueM/cliclick/blob/d786bc5faf8830dd30584e731b166e504927dbad/Actions/KeyPressAction.m
_CLICLICK_KEYS = set("return enter esc tab space delete fwd-delete arrow-up arrow-down arrow-left arrow-right "
                     "home end page-up page-down num-clear num-equals num-divide num-multiply num-minus num-plus "
                     "num-enter mute volume-up volume-down brightness-up brightness-down play-pause play-previous "
                     "play-next keys-light-toggle keys-light-up keys-light-down".split()) | {
                         f"f{i}" for i in range(1, 17)} | {f"num-{i}" for i in range(10)}
_CLICLICK_MODS = {"cmd", "ctrl", "alt", "shift", "fn"}


def cliclick_keys(mods, base, *, hold_ms=None):
    if any(m not in _CLICLICK_MODS for m in mods):
        raise ValueError(f"unsupported cliclick modifiers {mods!r}")
    if hold_ms is not None:
        if base not in _CLICLICK_MODS:
            raise ValueError(f"cliclick cannot hold non-modifier key {base!r}; only modifier combos are holdable")
        held = ",".join([*mods, base])
        return [f"kd:{held}", f"w:{hold_ms}", f"ku:{held}"]
    if len(base) == 1 and base.isprintable():
        inner = f"t:{base}"
    elif base.lower() in _CLICLICK_KEYS:
        inner = f"kp:{base.lower()}"
    else:
        raise ValueError(f"unknown cliclick key {base!r}")
    held = ",".join(mods)
    return [f"kd:{held}", inner, f"ku:{held}"] if mods else [inner]


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

# The one X11 front reader (guest shell): ``<window id>|<pid>|<base64 title>`` on one line, nothing when unread.
# The title travels base64-encoded: whatever a window is called, it can never look like a marker or add a line.
_X11_FRONT_SH = ("W=$(xdotool getactivewindow 2>/dev/null); if [ -n \"$W\" ]; then "
                 "printf '%s|%s|b64:%s' \"$W\" \"$(xdotool getwindowpid \"$W\" 2>/dev/null)\" "
                 "\"$(printf '%s' \"$(xdotool getwindowname \"$W\" 2>/dev/null)\" | base64 | tr -d '\\n')\"; fi")
# The same reader inside the guest python that runs an input: it cannot raise (a probe failure is an unknown
# place, never a failed input), prints the markers the receipt is read from (each on its own line, the title
# base64-encoded), and withholds a targeted input when the expected window is not in front (``__TARGET_LOST__``)
# or could not be read at all (``__FRONT_UNREAD__``).
_OSWORLD_PROBE = (
    "import subprocess as _sp, base64 as _b64\n"
    "def _front():\n"
    "    try:\n"
    "        w = _sp.run(['xdotool', 'getactivewindow'], capture_output=True, text=True, timeout=3).stdout.strip()\n"
    "        if not w:\n"
    "            return ''\n"
    "        pn = _sp.run(['xdotool', 'getwindowpid', w, 'getwindowname', w], capture_output=True, text=True, errors='replace', timeout=3).stdout\n"
    "        p, sep, n = pn.partition(chr(10))\n"
    "        if not sep:\n"  # xdotool ends a chain at its first failing command: a window without a pid keeps its name
    "            p, n = '', _sp.run(['xdotool', 'getwindowname', w], capture_output=True, text=True, errors='replace', timeout=3).stdout\n"
    "        return w + '|' + p.strip() + '|b64:' + _b64.b64encode(n.rstrip(chr(10)).encode('utf-8', 'replace')).decode('ascii')\n"
    "    except Exception:\n"
    "        return ''\n"
    "_b = _front()\n"
    "print()\n"
    "print('__FRONT_BEFORE__' + _b)\n"
    "if {expect!r} and not _b:\n"
    "    print('__FRONT_UNREAD__')\n"
    "    raise SystemExit(3)\n"
    "if {expect!r} and _b.split('|')[0] not in {expect!r}:\n"
    "    print('__TARGET_LOST__')\n"
    "    raise SystemExit(3)\n"
    "{command}\n"
    "print()\n"
    "print('__FRONT_AFTER__' + _front())\n"
)
# Mac over SSH: who is in front (one line), the input wrapper and the pre-check of a targeted input.
_SSH_FRONT_READ = ("lsappinfo info -only pid -only bundleid -only name \"$(lsappinfo front 2>/dev/null)\" 2>/dev/null"
                   " | tr '\\n' ' '")
_SSH_INPUT_WRAPPER = ("B=$({read}); echo; echo \"__FRONT_BEFORE__$B\"; {precheck}"
                      "{command}; RC=$?; A=$({read}); echo; echo \"__FRONT_AFTER__$A\"; exit $RC")
_SSH_PRECHECK = ("P=$(printf '%s' \"$B\" | grep -o '\"pid\"=[0-9]*' | tr -dc 0-9); "
                 "if [ -z \"$P\" ]; then echo __FRONT_UNREAD__; exit 3; fi; "
                 "if [ \"$P\" != \"{pid}\" ]; then echo __TARGET_LOST__; exit 3; fi; ")
# Activation by pid (only an integer is interpolated): System Events brings the process forward; the same call
# polls who is in front up to the wait and reads it before and after.
_SSH_ACTIVATE = ("B=$({read}); osascript -e 'tell application \"System Events\" to set frontmost of "
                 "(first process whose unix id is {pid}) to true' 2>&1; RC=$?; i=0; while [ $i -lt {polls} ]; do "
                 "F=$(lsappinfo info -only pid \"$(lsappinfo front 2>/dev/null)\" 2>/dev/null | tr -dc 0-9); "
                 "[ \"$F\" = \"{pid}\" ] && break; sleep 0.1; i=$((i+1)); done; "
                 "A=$({read}); echo; echo \"__FRONT_BEFORE__$B\"; echo \"__FRONT_AFTER__$A\"; exit $RC")
# The Mac's inventory: the listing's own exit code is kept (a pipeline's last command would mask it).
_SSH_INVENTORY = "lsappinfo list; RC=$?; echo; echo __FRONT__; {read}; exit $RC"
# SSH paste: the clipboard write and the paste chord are two stages, and the receipt says which ones ran.
_SSH_PASTE = "printf %s {text} | pbcopy && echo __CLIPBOARD_WRITTEN__ && cliclick kd:cmd t:v ku:cmd"
_MARKER_LINE = re.compile(r"^__(FRONT_BEFORE|FRONT_AFTER|TARGET_LOST|FRONT_UNREAD|CLIPBOARD_WRITTEN)__.*$\n?", re.M)


def cliclick_point(x: int, y: int) -> str:
    """A cliclick point: an absolute negative coordinate (a display left of or above the main one) takes the ``=``
    prefix, because a bare ``-`` means a move relative to the pointer."""
    return f"{'=' if int(x) < 0 else ''}{int(x)},{'=' if int(y) < 0 else ''}{int(y)}"


def parse_markers(text: str, parse: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Marker LINES (``__FRONT_BEFORE__…``, ``__FRONT_AFTER__…``, ``__TARGET_LOST__``, ``__FRONT_UNREAD__``) ->
    (before, after, stop) where ``stop`` is ``"target_lost"``, ``"front_unread"`` or ``""``. Only a whole line is a
    marker, and the readers encode titles, so text an application shows can never be read as one."""
    found: dict[str, dict[str, Any] | None] = {}
    for label in ("BEFORE", "AFTER"):
        m = re.search(rf"^__FRONT_{label}__(.*)$", text or "", re.M)
        found[label] = parse(m.group(1).strip()) if m else None
    stop = ""
    if re.search(r"^__TARGET_LOST__$", text or "", re.M):
        stop = "target_lost"
    elif re.search(r"^__FRONT_UNREAD__$", text or "", re.M):
        stop = "front_unread"
    return found["BEFORE"], found["AFTER"], stop


def split_ssh_inventory(out: str) -> tuple[str, str]:
    """(listing, front text) of an SSH inventory read: the marker is matched at the start of a line (the last one), so
    an application name that contains it is data."""
    marks = list(re.finditer(r"^__FRONT__", out, re.M))
    if not marks:
        return out, ""
    return out[:marks[-1].start()], out[marks[-1].end():]


def _info_or_none(raw: str) -> dict[str, Any] | None:
    info = parse_info(raw)
    return info if info.get("pid") else None


def strip_markers(text: str) -> str:
    return _MARKER_LINE.sub("", text or "").strip()


def _clean_listing_text(text: str) -> str:
    """Error text of the guest listing without the script's own separator and front line."""
    keep = [ln for ln in str(text or "").splitlines()
            if ln.strip() != "__ACTIVE__" and not re.match(r"^\d+\|\d*\|b64:", ln.strip())]
    return strip_markers("\n".join(keep))


class RemoteBackends:
    """OSWorld HTTP + SSH Mac transports (see module docstring)."""

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

    def _validate_remote_keys(self, conn, tokens):
        # PyAutoGUI may silently ignore unknown names or entries with no backend mapping.
        code = ("import json, pyautogui; "
                f"print(json.dumps({{'unsupported_keys': [k for k in {tokens!r} "
                "if not pyautogui.platformModule.keyboardMapping.get(k)]}))")
        try:
            out = self._osworld_execute(conn, ["python", "-c", code], timeout=10)
            ok, error = _osworld_result_ok(out)
            if not ok:
                raise ValueError(error)
            unsupported = json.loads(out["result"]["output"])["unsupported_keys"]
            if not isinstance(unsupported, list):
                raise ValueError("invalid key validation response")
        except Exception as exc:
            raise ValueError(f"guest key validation unavailable: {exc}") from exc
        if unsupported:
            raise ValueError(f"unsupported guest key(s): {unsupported!r}")

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
        m = re.search(r"pyautogui\.click\((-?\d+),\s*(-?\d+),\s*clicks=(\d+).*button=([\"'])([^\"']+)\4", text)
        if m:
            x, y, clicks, button = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(5)
            if button == "right":
                return [f"rc:{cliclick_point(x, y)}"], ""
            if button == "middle":
                return [], "middle-click unsupported by cliclick"
            op = "tc" if clicks >= 3 else ("dc" if clicks == 2 else "c")
            return [f"{op}:{cliclick_point(x, y)}"], ""
        m = re.search(r"pyautogui\.moveTo\((-?\d+),\s*(-?\d+)\).*pyautogui\.dragTo\((-?\d+),\s*(-?\d+)", text)
        if m:
            sx, sy, ex, ey = map(int, m.groups())
            return [f"dd:{cliclick_point(sx, sy)}", f"dm:{cliclick_point(ex, ey)}", f"du:{cliclick_point(ex, ey)}"], ""
        m = re.search(r"pyautogui\.moveTo\((-?\d+),\s*(-?\d+)\)", text)
        if m:
            return [f"m:{cliclick_point(int(m.group(1)), int(m.group(2)))}"], ""
        m = re.search(r"pyautogui\.(mouseDown|mouseUp)\(x=(-?\d+),\s*y=(-?\d+),\s*button=([\"'])([^\"']+)\4", text)
        if m:
            fn, x, y, button = m.group(1), int(m.group(2)), int(m.group(3)), m.group(5)
            if button != "left":
                return [], "mouseDown/mouseUp supports only left button via cliclick"
            return [f"{'dd' if fn == 'mouseDown' else 'du'}:{cliclick_point(x, y)}"], ""
        m = re.search(r"pyautogui\.(mouseDown|mouseUp)\(button=([\"'])([^\"']+)\2", text)
        if m:
            fn, button = m.group(1), m.group(3)
            if button != "left":
                return [], "mouseDown/mouseUp supports only left button via cliclick"
            return [f"{'dd' if fn == 'mouseDown' else 'du'}:."], ""
        m = re.search(r"pyautogui\.typewrite\((?P<q>[\"'])(?P<txt>.*?)(?P=q),\s*interval=", text)
        if m:
            return [f"t:{m.group('txt')}"], ""
        if text.startswith(("pyautogui.press(", "pyautogui.hotkey(")):
            import ast
            try:
                call = ast.parse(text, mode="eval").body
                tokens = [ast.literal_eval(arg) for arg in call.args]
                keys = [self._ssh_macos_key_name(t) for t in tokens]
                return cliclick_keys(keys[:-1], keys[-1]), ""
            except (ValueError, SyntaxError, AttributeError, IndexError) as exc:
                return [], str(exc)
        if "pyautogui.scroll" in text or "pyautogui.hscroll" in text:
            return [], "scroll unsupported via cliclick; use key page-down/page-up"
        return [], f"unsupported pyautogui snippet for ssh_macos/cliclick: {text[:120]}"

    def _remote_pyautogui(self, conn: dict[str, Any], code: str, *, note: dict[str, Any] | None = None, timeout: int = 30,
                          input_effect: bool = True, target: str = "", ctx: dict[str, Any] | None = None, later: bool = False,
                          raise_policy: str = "allow") -> str:
        return self._json_text(self._remote_input(conn, code, later=later, note=note, timeout=timeout, input_effect=input_effect,
                                                  target=target, ctx=ctx, raise_policy=raise_policy))

    def _remote_input(self, conn: dict[str, Any], code: str, *, note: dict[str, Any] | None = None, timeout: int = 30,
                      input_effect: bool = True, target: str = "", ctx: dict[str, Any] | None = None,
                      ssh_command: str = "", later: bool = False, raise_policy: str = "allow") -> dict[str, Any]:
        """One remote input (or read, ``input_effect=False``). With ``target`` the application is resolved and brought
        forward first (``_begin``); with ``ctx`` (a later step of one call) the earlier switch is reused and the guest
        pre-check withholds the input when that application is no longer in front."""
        if conn.get("disabled"):
            return json.loads(self._disabled_connection_error("remote", conn))
        backend = str(conn.get("backend") or "").lower()
        if backend == "ssh_macos" and not ssh_command:
            # Translate before any switch: a snippet cliclick cannot express is refused without moving the front.
            cliclick_args, err = self._ssh_macos_cliclick_for_pyautogui(code)
            if err:
                return {"ok": False, "backend": backend, "error": err, "code": code}
            ssh_command = "cliclick " + " ".join(shlex.quote(arg) for arg in cliclick_args)
        if input_effect and ctx is None:
            refusal, ctx = self._begin(target, raise_policy=raise_policy, connection=("remote", conn))
            if refusal:
                return refusal
        shared = ctx if ctx is not None else {}
        ctx = dict(shared)
        expect = ((ctx.get("target") or {}).get("resolved") or {}) if input_effect else {}
        activated = bool((ctx.get("target") or {}).get("activated"))
        try:
            if backend == "osworld_http":
                if input_effect:
                    # The guest checks the active window id against the target's windows before it acts.
                    ids = [str(window_id_int(w)) for w in (expect.get("window_ids") or []) if window_id_int(w) is not None]
                    probe = _OSWORLD_PROBE.format(expect=ids, command=code)
                    wrapped = _OSWORLD_PKGS_PREFIX.format(command=probe)
                else:
                    wrapped = _OSWORLD_PKGS_PREFIX.format(command=code)
                out = self._osworld_execute(conn, ["python", "-c", wrapped], timeout=timeout)
                result = out.get("result")
                text = str(result.get("output") or "") if isinstance(result, dict) else ""
                before, after, stop = parse_markers(text, parse_x11_front)
                if stop:
                    return self._stopped(ctx, shared, before, backend, stop, later=later)
                ok, completion_known, err = _osworld_completion(out)
                if isinstance(result, dict) and input_effect:
                    result = {**result, "output": strip_markers(text)}
                payload: dict[str, Any] = {"ok": ok, "backend": backend, "status": out["status"], "execute_result": result,
                                           "dispatched": True if completion_known else None}
                if not ok:
                    payload["error"] = err
                    if not completion_known:
                        payload.update({"reason": "completion_unknown", "transport_failed": True})
                stderr = ""
            elif backend == "ssh_macos":
                remote = ssh_command
                if input_effect:
                    precheck = _SSH_PRECHECK.format(pid=int(expect["pid"])) if expect.get("pid") else ""
                    remote = _SSH_INPUT_WRAPPER.format(read=_SSH_FRONT_READ, precheck=precheck, command=remote)
                rc, stdout, stderr = self._ssh_run(conn, remote, timeout=timeout)
                before, after, stop = parse_markers(stdout, _info_or_none)
                if stop:
                    return self._stopped(ctx, shared, before, backend, stop, later=later)
                # The wrapper prints `__FRONT_BEFORE__` before the input and `__FRONT_AFTER__` after it. Both received:
                # the wrapper completed and `rc` is the input's own exit. A reply cut short (a refused connection, a
                # channel broken after the input may have run) leaves the outcome unknown: never "not sent".
                # ssh returns the remote exit status only after the session ended normally: exit 0 is completion too.
                completed = (rc == 0 or bool(re.search(r"^__FRONT_AFTER__", stdout, re.M))) if input_effect else True
                started = completed or bool(re.search(r"^__FRONT_BEFORE__", stdout, re.M))
                payload = {"ok": rc == 0 and completed, "backend": backend, "returncode": rc, "output": strip_markers(stdout),
                           "dispatched": True if completed else None}
                if not completed:
                    payload.update({"transport_failed": True, "reason": "ssh_failed",
                                    "error": f"ssh ended without the wrapper's completion marker ({stderr.strip() or f'exit {rc}'}); "
                                             + ("the input stage had started when the reply was cut short" if started
                                                else "nothing of the wrapper's output came back")})
                elif rc != 0:
                    payload["error"] = stderr.strip() or strip_markers(stdout) or f"exit {rc}"
                if note and "clipboard_replaced" in note:
                    # The clipboard stage reports itself. A received marker is a fact whatever came after; no marker
                    # from a completed wrapper means the `&&` chain stopped at the write (the chord never ran); no
                    # marker from a cut-short reply is an unknown stage, not a failed one.
                    if re.search(r"^__CLIPBOARD_WRITTEN__$", stdout, re.M):
                        payload["clipboard_replaced"] = True
                    elif completed:
                        payload["clipboard_replaced"] = False
                        if rc != 0:
                            payload.update({"dispatched": False, "reason": "clipboard_failed",
                                            "error": f"the clipboard write failed on the remote Mac ({payload.get('error')}); nothing was pasted"})
                    else:
                        note = {k: v for k, v in note.items() if k != "clipboard_replaced"}  # an unknown stage makes no claim
            else:
                return {"ok": False, "error": f"unsupported remote backend {backend!r}"}
        except Exception as exc:  # noqa: BLE001 - the transport failed; an earlier observed switch is still a fact
            failed: dict[str, Any] = {"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}", "code": code,
                                      "transport_failed": True}
            if input_effect:
                failed.update({"dispatched": None,
                               "note": "the remote call did not answer: whether the input ran is unknown (do not assume it did not)"})
                if note:  # what was attempted (the method); a clipboard stage with no answer makes no claim
                    failed.update({k: v for k, v in note.items() if k != "clipboard_replaced" and k not in failed})
            if ctx.get("target"):
                failed["target"] = ctx["target"]
            if ctx.get("front_before") is not None:
                failed["front_before"] = ctx["front_before"]
            if activated:
                failed["observed_effect"] = True
                failed["activation_done"] = True
            return failed
        if note:
            payload.update({k: v for k, v in note.items() if k not in payload})
        if not input_effect:
            return payload
        if payload["ok"]:
            payload["observed_effect"] = True
        else:
            if activated:
                payload["observed_effect"] = True  # the switch ran before the input failed
                payload["activation_done"] = True
            if payload.get("clipboard_replaced"):
                payload["observed_effect"] = True  # the clipboard write completed before the paste chord failed
                payload["clipboard_done"] = True
        ctx["delivered_to"] = before
        ctx.setdefault("front_before", before)
        payload = self._finish(payload, ctx, stderr, after=after)
        shared["front_last"] = ctx.get("front_last")  # the call's last front read, for a sequence's aggregate
        return payload

    def _stopped(self, ctx: dict[str, Any], shared: dict[str, Any], front: dict[str, Any] | None, backend: str,
                 reason: str, *, later: bool = False) -> dict[str, Any]:
        """The remote pre-check withheld the input (another window in front, or the front unread): nothing was sent;
        a switch observed earlier in the same call keeps its own mark. ``later``: a later step of one call."""
        target = ctx.get("target") or {}
        res = {"reason": reason, "front": front, "error": "the remote front reader printed nothing", "before_step": later,
               "activated": target.get("activated")}
        out: dict[str, Any] = {"ok": False, "backend": backend, "reason": reason, "target": target,
                               "error": self._target_error(str(target.get("requested") or ""), res)}
        if front:
            out["front"] = front
            shared["front_last"] = front
        if ctx.get("front_before") is not None:
            out["front_before"] = ctx["front_before"]
        if target.get("activated"):
            out["observed_effect"] = True
            out["activation_done"] = True
        return out

    # --- remote place readers ------------------------------------------------------------------

    def _guest_windows(self, conn: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
        """OSWorld guest: (wmctrl rows, front window, error). One /execute call."""
        # The listing travels base64-encoded as one line: a window title can never look like the separator.
        script = (f"export DISPLAY=:0; L=$(wmctrl -lpxG 2>/dev/null); RC=$?; printf '%s' \"$L\" | base64 | tr -d '\\n'; echo; "
                  f"echo __ACTIVE__; {_X11_FRONT_SH}; echo; exit $RC")
        try:
            out = self._osworld_execute(conn, ["bash", "-lc", script], timeout=20)
        except Exception as exc:  # noqa: BLE001
            return [], None, f"{type(exc).__name__}: {exc}"
        ok, err = _osworld_result_ok(out)
        text = str((out.get("result") or {}).get("output") if isinstance(out.get("result"), dict) else "")
        m = re.search(r"^__ACTIVE__$", text, re.M)
        if not m:
            return [], None, "the guest printed no window listing (the script did not run as written)"
        # The listing is the LAST line before the separator: a login shell (`bash -lc`) may greet before the script runs.
        lines = [ln.strip() for ln in text[:m.start()].splitlines() if ln.strip()]
        encoded, active = (lines[-1] if lines else ""), text[m.end():].strip()
        try:
            listing = base64.b64decode(encoded + "=" * (-len(encoded) % 4), validate=True).decode("utf-8", "replace") if encoded else ""
        except (ValueError, UnicodeError):
            return [], None, "the guest listing could not be decoded (the script did not run as written)"
        rows = parse_wmctrl_rows(listing)
        front = parse_x11_front(active.splitlines()[0] if active else "", rows)
        return rows, front, ("" if ok else (_clean_listing_text(err) or "wmctrl failed in the guest"))

    def _remote_front(self, conn: dict[str, Any]) -> dict[str, Any] | None:
        """Who is in front on the remote now (OSWorld: the X11 window; SSH Mac: the application); None = unread."""
        backend = str(conn.get("backend") or "").lower()
        try:
            if backend == "osworld_http":
                out = self._osworld_execute(conn, ["bash", "-lc", f"export DISPLAY=:0; {_X11_FRONT_SH}; echo"], timeout=15)
                result = out.get("result")
                return parse_x11_front(str(result.get("output") or "").strip()) if isinstance(result, dict) else None
            if backend == "ssh_macos":
                rc, out, _err = self._ssh_run(conn, _SSH_FRONT_READ, timeout=15)
                return _info_or_none(out) if rc == 0 else None
        except Exception:  # noqa: BLE001 - an unread front is an unknown, not an error of the caller
            return None
        return None

    def _remote_activate_dict(self, conn: dict[str, Any], app: str, wait_ms: int, *, raise_policy: str = "allow") -> dict[str, Any]:
        """Resolve ``app`` from the remote's own inventory, bring it forward by id/pid (never by name, never a
        launch) and read who is in front afterwards. Same result shape as the local helper's ``activate``."""
        backend = str(conn.get("backend") or "").lower()
        app = str(app or "").strip()
        wait_ms = max(200, int(wait_ms))
        polls = max(2, wait_ms // 100)
        if backend == "osworld_http":
            rows, before, err = self._guest_windows(conn)
            if err and not rows:
                return {"ok": False, "backend": backend, "reason": "unsupported", "error": f"no window list in the guest ({err})"}
            res = x11_resolve(app, rows)
            if not res.get("ok"):
                return {**res, "backend": backend}
            target = res["target"]
            if x11_same(before, target):
                return {"ok": True, "backend": backend, "target": target, "front_before": before, "front": before,
                        "activated": False, "already_front": True, "verified": True, "waited_ms": 0}
            if raise_policy == "never":
                return {"ok": False, "backend": backend, "target": target, "front_before": before, "front": before,
                        "activated": False, "verified": False, "reason": "target_lost" if before else "front_unread",
                        "resolved": target}
            wid = window_id_int(target["window_id"])
            script = (f"export DISPLAY=:0; wmctrl -i -a {shlex.quote(str(target['window_id']))} 2>/dev/null; RC=$?; i=0; "
                      f"while [ $i -lt {polls} ]; do W=$(xdotool getactivewindow 2>/dev/null); [ \"$W\" = \"{wid}\" ] && break; "
                      f"sleep 0.1; i=$((i+1)); done; echo; printf '__FRONT_AFTER__'; {_X11_FRONT_SH}; echo; exit $RC")
            t0 = time.monotonic()
            try:
                out = self._osworld_execute(conn, ["bash", "-lc", script], timeout=max(20, wait_ms // 1000 + 15))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "backend": backend, "reason": "activate_failed", "error": f"{type(exc).__name__}: {exc}", "target": target}
            waited = int((time.monotonic() - t0) * 1000)
            ok, err = _osworld_result_ok(out)
            result = out.get("result")
            text = str(result.get("output") or "") if isinstance(result, dict) else ""
            _b, after, _stop = parse_markers(text, lambda raw: parse_x11_front(raw, rows))
            return self._activation_result(backend, target, before, after, "wmctrl -i -a", ok, err, waited, same=x11_same)
        if backend == "ssh_macos":
            rc, out, err = self._ssh_run(conn, _SSH_INVENTORY.format(read=_SSH_FRONT_READ), timeout=20)
            if rc != 0:
                return {"ok": False, "backend": backend, "reason": "apps_unread",
                        "error": f"the remote Mac's applications could not be listed ({err.strip() or strip_markers(out) or f'exit {rc}'})"}
            listing, front_text = split_ssh_inventory(out)
            apps = parse_list(listing)
            if not apps:
                return {"ok": False, "backend": backend, "reason": "apps_unread",
                        "error": "lsappinfo list named no application on the remote Mac"}
            before = _info_or_none(front_text)
            matches = resolve(app, apps)
            if not matches:
                return {"ok": False, "backend": backend, "reason": "not_running", "running": [a["app"] for a in apps][:40]}
            if len(matches) > 1:
                return {"ok": False, "backend": backend, "reason": "ambiguous",
                        "candidates": [{"app": a["app"], "pid": a["pid"], "app_id": a["app_id"]} for a in matches[:10]]}
            target = {"app": matches[0]["app"], "pid": matches[0]["pid"], "app_id": matches[0]["app_id"]}
            if before and before.get("pid") == target["pid"]:
                return {"ok": True, "backend": backend, "target": target, "front_before": before, "front": before,
                        "activated": False, "already_front": True, "verified": True, "waited_ms": 0}
            if raise_policy == "never":
                return {"ok": False, "backend": backend, "target": target, "front_before": before, "front": before,
                        "activated": False, "verified": False, "reason": "target_lost" if before else "front_unread",
                        "resolved": target}
            script = _SSH_ACTIVATE.format(read=_SSH_FRONT_READ, pid=int(target["pid"]), polls=polls)
            t0 = time.monotonic()
            rc, out, err = self._ssh_run(conn, script, timeout=max(20, wait_ms // 1000 + 15))
            waited = int((time.monotonic() - t0) * 1000)
            _b, after, _stop = parse_markers(out, _info_or_none)
            return self._activation_result(backend, target, before, after, "system_events", rc == 0,
                                           (err.strip() or strip_markers(out) or f"exit {rc}")[:300], waited)
        return {"ok": False, "reason": "unsupported", "error": f"unsupported remote backend {backend!r}"}

    @staticmethod
    def _activation_result(backend: str, target: dict[str, Any], before: dict[str, Any] | None, after: dict[str, Any] | None,
                           method: str, ok: bool, err: str, waited_ms: int, same: Any = None) -> dict[str, Any]:
        same = same or (lambda front, tgt: bool(front) and front.get("pid") == tgt.get("pid"))
        verified = bool(same(after, target))
        activated: bool | None = (verified and not same(before, target)) if before else (None if verified else False)
        out: dict[str, Any] = {"ok": verified, "backend": backend, "target": target, "front_before": before, "front": after,
                               "activated": activated, "verified": verified, "method": method, "waited_ms": waited_ms}
        if not verified:
            if not ok:
                out.update({"reason": "activate_failed", "error": strip_markers(err)})
            elif after is None:
                out.update({"reason": "front_unread", "activated": None,
                            "error": "the remote front reader printed nothing after the switch"})
            else:
                out["reason"] = "not_frontmost_after_activate"
        return out

    def _remote_window_list(self, conn: dict[str, Any], query: str, limit: int) -> str:
        backend = str(conn.get("backend") or "").lower()
        if backend == "osworld_http":
            rows, front, err = self._guest_windows(conn)
            if err and not rows:
                return self._json_text({"ok": False, "backend": backend, "reason": "unsupported",
                                        "error": f"the guest's window list could not be read (wmctrl/xdotool: {err})"})
            if front:
                for row in rows:
                    row["frontmost"] = window_id_int(row["window_id"]) == window_id_int(front.get("window_id"))
            if query:
                rows = [r for r in rows if query in r["app"].lower() or query in r["title"].lower() or query == f"pid:{r['pid']}"]
            payload: dict[str, Any] = {"ok": True, "backend": backend, "frontmost": front, "windows": rows[:limit],
                                       "truncated": len(rows) > limit, "limit": limit}
            if front is None:
                payload["front_unread"] = "xdotool getactivewindow named no window in the guest"
            if err:
                payload["warning"] = err
            return self._json_text(payload)
        if backend == "ssh_macos":
            rc, out, err = self._ssh_run(conn, _SSH_INVENTORY.format(read=_SSH_FRONT_READ), timeout=20)
            if rc != 0:
                return self._json_text({"ok": False, "backend": backend, "reason": "apps_unread",
                                        "error": f"lsappinfo list failed on the remote Mac ({err.strip() or strip_markers(out) or f'exit {rc}'})"})
            listing, front_text = split_ssh_inventory(out)
            front = _info_or_none(front_text)
            apps = parse_list(listing)
            if not apps:
                return self._json_text({"ok": False, "backend": backend, "reason": "apps_unread",
                                        "error": "lsappinfo list named no application on the remote Mac"})
            rows = [{"app": a["app"], "pid": a["pid"], "app_id": a["app_id"], **({"frontmost": a["pid"] == front["pid"]} if front else {})}
                    for a in apps]
            if query:
                rows = [r for r in rows if query in r["app"].lower() or query == r["app_id"].lower() or query == f"pid:{r['pid']}"]
            payload = {"ok": True, "backend": backend, "frontmost": front, "windows": rows[:limit], "titles_available": False,
                       "truncated": len(rows) > limit, "limit": limit,
                       "note": "window titles are not read over SSH; app names and the frontmost app are"}
            if front is None:
                payload["front_unread"] = "lsappinfo named no frontmost application on the remote Mac"
            return self._json_text(payload)
        return self._json_text({"ok": False, "backend": backend, "error": f"unsupported remote backend {backend!r}"})

    def _remote_ax_tree(self, conn: dict[str, Any]) -> str:
        backend = str(conn.get("backend") or "").lower()
        if backend == "osworld_http":
            target = self._connection_target(conn)
            if not target:
                return self._json_text({"ok": False, "error": "osworld_http connection has no target/target_file"})
            try:
                with urllib.request.urlopen(target + "/accessibility", timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return self._json_text({"ok": True, "backend": backend, "accessibility_tree": raw[:80_000], "truncated": len(raw) > 80_000,
                                        "note": "the guest's whole accessibility tree; app, root, max_nodes and timeout_ms do not apply here"})
            except Exception as exc:  # noqa: BLE001
                return self._json_text({"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}"})
        return self._json_text({"ok": False, "backend": backend, "reason": "unsupported",
                                "error": "no accessibility reader over SSH; use window_list and the app's own commands through remote_exec"})

    def _remote_screenshot_result(
        self,
        *,
        backend: str,
        raw_path: pathlib.Path,
        max_width: int,
        max_height: int,
        input_w: int,
        input_h: int,
        input_x: int = 0,
        input_y: int = 0,
        extra: dict[str, Any] | None = None,
        approx: bool = False,
        mapping_available: bool = True,
        connection_identity: str = "",
        resolved_target: str = "",
    ) -> str:
        px_w, px_h = self._png_dims(raw_path)
        if px_w <= 0 or px_h <= 0 or not self._png_ok(raw_path):
            # Not a fully decodable PNG — don't claim success on garbage; a valid
            # 24-byte header over zero-padded data must fail here, not rounds
            # later as a provider 400. Drop the file.
            try:
                raw_path.unlink()
            except OSError:
                pass
            return self._json_text({"ok": False, "backend": backend, "error": "remote screenshot is not a fully decodable PNG"})
        if mapping_available and (input_w <= 0 or input_h <= 0):
            input_w, input_h = px_w, px_h
        max_w = max(320, min(int(max_width or self._MAX_IMAGE_W), 4096))
        max_h = max(240, min(int(max_height or self._MAX_IMAGE_H), 4096))
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
            "input_width": input_w if mapping_available else None,
            "input_height": input_h if mapping_available else None,
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
        if not mapping_available:
            result["mapped_input_available"] = False
            result["coordinate_note"] = "Capture only: exact primary display geometry is unavailable; use raw=true for native coordinates."
        if mapping_available and img_w > 0 and img_h > 0 and input_w > 0 and input_h > 0:
            sx = round(input_w / img_w, 6)
            sy = round(input_h / img_h, 6)
            transform = {
                "sx": sx, "sy": sy,
                "image_w": img_w, "image_h": img_h,
                "input_w": input_w, "input_h": input_h,
                "platform": backend, "session": "remote",
                "approx": bool(approx), "ts": time.time(),
                "ox": input_x, "oy": input_y,
                "display_geometry": {"x": input_x, "y": input_y, "w": input_w, "h": input_h},
                "observation_id": uuid.uuid4().hex,
                "connection_identity": connection_identity or self._connection_identity(), "backend": backend,
                "resolved_target": resolved_target,
            }
            if self._persist_transform(result, transform):
                result["coord_transform"] = {k: v for k, v in transform.items() if k != "resolved_target"}
                result["observation_id"] = transform["observation_id"]
                result["coordinate_note"] = (
                    "Pass this observation_id with coordinates read off THIS image to click/move/drag — "
                    "they are remapped through coord_transform (image -> remote input space)."
                    + (" The input scale is approximate." if approx else "")
                )
        return self._json_text(result)

    def _osworld_screenshot(self, conn: dict[str, Any], *, max_width: int, max_height: int,
                            connection_name: str = "") -> str:
        target = self._connection_target(conn)
        if not target:
            return self._json_text({"ok": False, "error": "osworld_http connection has no target/target_file"})
        identity = self._connection_identity(connection=(connection_name, conn), resolved_target=target)
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
                return self._json_text({"ok": False, "error": f"/screenshot failed: {type(exc).__name__}: {exc}", "backend": "osworld_http"})
            if not data:
                return self._json_text({"ok": False, "error": "/screenshot returned empty body", "backend": "osworld_http"})
            if len(data) > _MAX_REMOTE_SHOT_BYTES:
                return self._json_text({"ok": False, "error": f"/screenshot exceeded {_MAX_REMOTE_SHOT_BYTES} byte cap", "backend": "osworld_http"})
            # Write-then-validate-then-rename: the published path never holds
            # a partially written or undecodable image.
            tmp_path = out_path.with_suffix(".part")
            tmp_path.write_bytes(data)
            if self._png_ok(tmp_path):
                tmp_path.rename(out_path)
                break
            last_err = f"undecodable PNG ({len(data)} bytes) on attempt {attempt + 1}/3"
            try:
                tmp_path.unlink()
            except OSError:
                pass
            time.sleep(0.5)
        else:
            return self._json_text({"ok": False, "error": f"screenshot_corrupt: {last_err}", "backend": "osworld_http"})
        px_w, px_h = self._png_dims(out_path)
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
            connection_identity=identity, resolved_target=target,
        )

    def _test_osworld(self, conn: dict[str, Any], name: str) -> str:
        target = self._connection_target(conn)
        if not target:
            return self._json_text({"ok": False, "connection": name, "backend": "osworld_http", "error": "missing target/target_file"})
        try:
            with urllib.request.urlopen(target + "/screenshot", timeout=10) as resp:
                raw = resp.read(32)
            out = self._osworld_execute(conn, ["python", "-c", "import pyautogui; print(pyautogui.size())"], timeout=20)
            return self._json_text({
                "ok": bool(raw) and _osworld_result_ok(out)[0],
                "connection": name,
                "backend": "osworld_http",
                "target": target,
                "screenshot_bytes_probe": len(raw),
                "execute_probe": out,
            })
        except Exception as exc:  # noqa: BLE001
            return self._json_text({"ok": False, "connection": name, "backend": "osworld_http", "target": target, "error": f"{type(exc).__name__}: {exc}"})

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
        return self._run_cmd(ssh_cmd, timeout=timeout)

    def _ssh_macos_screenshot(self, conn: dict[str, Any], *, max_width: int, max_height: int,
                              connection_name: str = "") -> str:
        identity = self._connection_identity(connection=(connection_name, conn))
        out_dir = pathlib.Path(self.api.skill_job_dir("ssh_macos")) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        remote_path = f"/tmp/ouroboros-shot-{int(time.time())}-{uuid.uuid4().hex[:6]}.png"
        rc, stdout, stderr = self._ssh_run(conn, f"screencapture -x -m {remote_path!r}", timeout=20)
        if rc != 0:
            return self._json_text({"ok": False, "backend": "ssh_macos", "error": stderr.strip() or stdout.strip() or f"exit {rc}"})
        dest = out_dir / pathlib.Path(remote_path).name
        scp_cmd = ["scp", "-q", *self._ssh_scp_source(conn, remote_path), str(dest)]
        try:
            proc = subprocess.run(scp_cmd, text=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
        except Exception as exc:  # noqa: BLE001
            return self._json_text({"ok": False, "backend": "ssh_macos", "error": f"scp failed: {type(exc).__name__}: {exc}"})
        if proc.returncode != 0 or not dest.exists():
            return self._json_text({"ok": False, "backend": "ssh_macos", "error": proc.stderr.strip() or proc.stdout.strip() or f"scp exit {proc.returncode}"})
        # Capture and live validation use the same exact primary-display geometry reader.
        geometry = self._display_geometry({}, connection=(connection_name, conn)) or {}
        return self._remote_screenshot_result(
            backend="ssh_macos",
            raw_path=dest,
            max_width=max_width,
            max_height=max_height,
            input_w=geometry.get("w", 0),
            input_h=geometry.get("h", 0),
            input_x=geometry.get("x", 0),
            input_y=geometry.get("y", 0),
            extra={"host": str(conn.get("ssh_alias") or conn.get("host") or "")},
            mapping_available=bool(geometry),
            connection_identity=identity,
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
        return self._json_text({"ok": ok, "connection": name, "backend": "ssh_macos", "output": stdout, "error": stderr, "hint": hint})


    def _remote_type_text(self, conn: dict[str, Any], text: str, interval_ms: int = 0, target: str = "", via: str = "",
                          raise_policy: str = "allow") -> str:
        """type_text on a remote backend (OSWorld clipboard routing measured on the bench; SSH Mac types or pastes)."""
        text = str(text or "")
        interval = max(0, int(interval_ms or 0)) / 1000.0
        typewrite_code = f"pyautogui.typewrite({text!r}, interval={interval!r})"
        backend = str(conn.get("backend") or "").lower()
        if backend == "ssh_macos" and via == "paste":
            # The clipboard of the remote Mac is replaced (said in the receipt); the text travels as one quoted argument.
            command = _SSH_PASTE.format(text=shlex.quote(text))
            return self._json_text(self._remote_input(conn, "paste", note={"method": "paste", "clipboard_replaced": True, "chars": len(text)},
                                                      timeout=60, target=target, ssh_command=command, raise_policy=raise_policy))
        ctx: dict[str, Any] | None = None
        if target:
            # One switch for the whole call: a typewrite fallback reuses it instead of bringing the app forward again.
            refusal, ctx = self._begin(target, raise_policy=raise_policy, connection=("remote", conn))
            if refusal:
                return self._json_text(refusal)
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
        if backend == "osworld_http" and (via == "paste" or any(ord(ch) > 127 for ch in text)
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
                conn, clip_code, note={"method": "clipboard"}, timeout=60, target=target, ctx=ctx,
                raise_policy=raise_policy,
            )
            try:
                parsed = json.loads(out)
                if parsed.get("ok") or parsed.get("reason") or parsed.get("transport_failed"):
                    # Done, withheld/refused (target lost, front unread), or the answer was lost after the program may
                    # have pasted: none of these is retried by typing.
                    return out
            except Exception:
                pass
            # Clipboard path failed (pyperclip/xclip absent) — typewrite fallback, same switch.
            return self._remote_pyautogui(
                conn, typewrite_code, note={"method": "typewrite", "clipboard_fallback": True}, timeout=60, target=target, ctx=ctx,
                raise_policy=raise_policy,
            )
        return self._remote_pyautogui(conn, typewrite_code, note={"method": "typewrite"}, timeout=60, target=target,
                                      ctx=ctx, raise_policy=raise_policy)

    def remote_exec(self, *, command: str, timeout: int = 60) -> str:
        """Run a shell command on the active remote connection.

        For local backend, this refuses to run so existing local computer-use does
        not become a general shell tool. Use ordinary Ouroboros shell/file tools
        for local work; remote_exec is for inspecting a selected remote machine.
        """
        _name, conn = self._connection_snapshot()
        if conn.get("disabled"):
            return self._disabled_connection_error(_name, conn)
        backend = str(conn.get("backend") or "local").lower()
        cmd = str(command or "").strip()
        if not cmd:
            return self._json_text({"ok": False, "error": "command is required"})
        timeout = max(5, min(300, int(timeout or 60)))
        try:
            if backend == "osworld_http":
                out = self._osworld_execute(conn, ["bash", "-lc", cmd], timeout=timeout)
                ok, err = _osworld_result_ok(out)
                payload: dict[str, Any] = {"ok": ok, "backend": backend, "status": out["status"], "result": out["result"]}
                if not ok:
                    payload["error"] = err
                return self._json_text(payload)
            if backend == "ssh_macos":
                rc, stdout, stderr = self._ssh_run(conn, cmd, timeout=timeout)
                return self._json_text({"ok": rc == 0, "backend": backend, "returncode": rc, "output": stdout, "error": stderr})
        except Exception as exc:  # noqa: BLE001
            return self._json_text({"ok": False, "backend": backend, "error": f"{type(exc).__name__}: {exc}"})
        return self._json_text({"ok": False, "backend": backend, "error": "remote_exec requires an active remote backend"})
