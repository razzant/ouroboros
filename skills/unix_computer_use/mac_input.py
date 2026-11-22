"""macOS input of unix_computer_use: the operations ``input_events.js`` posts, and how a refusal for a missing grant reads.

One program serves the local Mac (``osascript -l JavaScript input_events.js '<spec>'``) and a Mac over SSH (its source
through that Mac's own ``osascript -e``), so nothing is installed on either. This module builds the program's argv
from operations, names the keys (``kVK_*`` virtual key codes, which do not depend on the keyboard layout), translates
the pyautogui snippets the remote paths emit into the same operations, reads the program's answer, and composes the
one structured field a refusal for a missing macOS grant carries (``permission_refusal``).
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import shlex
import time
from typing import Any

INPUT_EVENTS_JS = pathlib.Path(__file__).with_name("input_events.js")
EVENT_TAG = 0x4F55524F  # "OURO" in kCGEventSourceUserData of every posted event (the same constant as the program's)
# Receipt methods of programs that refuse BEFORE posting anything (a missing grant, a key no layout gives): such a
# refusal is a known "not sent". The paste chord is not one of them: its clipboard write ran first.
REFUSES_BEFORE_POSTING = frozenset({"command_chord", "cg_events", "unicode_events"})
_POSTING_METHODS = REFUSES_BEFORE_POSTING | {"paste"}
MODIFIERS = frozenset({"cmd", "ctrl", "alt", "shift", "fn"})
BUTTONS = ("left", "right", "middle")
_FKEYS = (122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106)
KEY_CODES = {
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "esc": 53, "fwd-delete": 117,
    "home": 115, "end": 119, "page-up": 116, "page-down": 121,
    "arrow-left": 123, "arrow-right": 124, "arrow-down": 125, "arrow-up": 126,
    "num-clear": 71, "num-equals": 81, "num-divide": 75, "num-multiply": 67, "num-minus": 78, "num-plus": 69,
    "num-enter": 76, "num-decimal": 65,
    **{f"num-{n}": code for n, code in enumerate((82, 83, 84, 85, 86, 87, 88, 89, 91, 92))},
    **{f"f{n}": code for n, code in enumerate(_FKEYS, 1)},
}
# System keys: NX key types of the NSSystemDefined events (subtype 8) a keyboard's media keys send, the names the
# former helper (cliclick) took for them.
SYSTEM_KEYS = {"volume-up": 0, "volume-down": 1, "brightness-up": 2, "brightness-down": 3, "mute": 7, "play-pause": 16,
               "play-next": 17, "play-previous": 18, "keys-light-up": 21, "keys-light-down": 22, "keys-light-toggle": 23}

# The one place the refusal's field names live (a later contract may rename them).
_REFUSAL_FIELD = "refusal"
_GRANT_PANES = {"accessibility": "Accessibility", "post_events": "Accessibility",
                "screen_recording": "Screen & System Audio Recording"}
# The reasons the native helpers already give a missing grant (macos_native, macos_capture, macos_stream).
_GRANT_REASONS = {"accessibility_untrusted": "accessibility", "screen_recording_denied": "screen_recording"}


# Typing budget per grapheme, a delivery limit, not a judgement. Measured 21.11.2026 on hosted Macs typing 600
# characters at interval_ms=0 with a pause after each key down and up: 15.44 ms (macos-15-intel), 19.56 and 21.86 ms
# (arm64 macos-15 / macos-26), where 10 ms ran into the deadline at 485 and 514 characters; the pause after the down
# is gone since, and 40 ms keeps about twice the slowest measured rate.
TYPE_MS_PER_CHAR = 40
# type_text's registered timeout_sec: the host kills the child then, counting from its start; the last TYPE_RECEIPT_SEC
# of it are kept for the child's start before the handler and for the receipt after the program.
TYPE_HOST_SEC = 180
TYPE_RECEIPT_SEC = 10
_clock = time.monotonic  # the host's deadline counts on this clock too (a test double replaces this name only)


def type_seconds(text: str, interval_ms: int = 0) -> int:
    """The helper timeout for typing ``text``: a fixed part for starting the program and reading the layout, plus the
    per-character budget and the asked interval; the program's own deadline is 2 s below it."""
    need = 10 + len(text) * (max(0, int(interval_ms or 0)) + TYPE_MS_PER_CHAR) // 1000
    return min(TYPE_HOST_SEC - TYPE_RECEIPT_SEC, need)


# An SSH input call whose ssh exited 0 while the remote sh printed none of the wrapper's lines: it ran no script.
SCRIPT_NOT_RUN = {"dispatched": False, "reason": "remote_script_not_run",
                  "error": "ssh exited 0 but the remote shell printed none of the input wrapper's lines, so the command never "
                           "reached it and nothing was sent: check that ssh passes stdin to the remote command for this "
                           "host (a StdinNull yes in the ssh configuration stops it)"}


def typing_until(ops: list[dict[str, Any]], seconds: float) -> float | None:
    """The deadline of an SSH Mac's program: typed text alone counts from the call's start (its counts must come
    back before the host's deadline); every other operation keeps its own fixed timeout."""
    return deadline_in(seconds) if any(op.get("op") == "type" for op in ops) else None


def deadline_in(seconds: float) -> float:
    """The monotonic moment ``seconds`` from now, read when the call begins: the helper must be done by then, so the
    time its target switch takes comes out of the program's budget (``fit``), never out of the host's deadline."""
    return round(_clock() + float(seconds), 3)


def left_of(began: float, seconds: int) -> int:
    """What is left of ``seconds`` that began at ``began`` (a ``deadline_in(0)`` stamp), in whole seconds, at least 1:
    the time a call spent before building its program (an SSH Mac's inventory and switch) is not given again."""
    return max(1, int(began + seconds - _clock()))


def fit(cmd: list[str], timeout: int, *, margin_ms: int = 2000) -> tuple[list[str], int]:
    """``cmd`` and its helper timeout cut to what is left before the spec's ``until`` (``deadline_in``), just before it
    runs: the program's own deadline falls ``margin_ms`` before the helper's, which falls at ``until``, so a text the
    switch delayed stops with its counts (chars_sent) instead of being killed by the host without them. Any other
    argv, or one without ``until``, is returned as it is."""
    if ops_of(cmd) is None:
        return cmd, timeout
    spec = json.loads(cmd[-1])
    end = spec.pop("until", None)
    if not isinstance(end, (int, float)):
        return cmd, timeout
    left_ms = int((end - _clock()) * 1000)
    spec["budget_ms"] = max(500, min(int(spec.get("budget_ms") or 8000), left_ms - margin_ms))
    wait = max(min(int(timeout), -(-left_ms // 1000)), -(-(spec["budget_ms"] + margin_ms) // 1000))
    return [*cmd[:-1], json.dumps(spec, ensure_ascii=True, separators=(",", ":"))], wait


def fit_shell(command: str, timeout: int) -> tuple[str, int]:
    """``fit`` for the shell-quoted program an SSH Mac runs (3 s of margin for the round trip)."""
    try:
        words = shlex.split(command)
    except ValueError:
        return command, timeout
    fitted, wait = fit(words, timeout, margin_ms=3000)
    return (shlex.join(fitted), wait) if fitted is not words else (command, timeout)


def permission_refusal(grant: str) -> dict[str, Any]:
    """The one structured field of a refusal for a missing macOS grant, in the shape the host's other capabilities
    use: the capability is unavailable, which permission it lacks, that only the owner can repair it and where
    (System Settings; nothing here asks the system to prompt). ``reason`` beside it stays the string code."""
    return {_REFUSAL_FIELD: {"capability_unavailable": True, "permission": grant, "repair": "owner",
                             "where": f"System Settings > Privacy & Security > {_GRANT_PANES[grant]}"}}


def proved(payload: dict[str, Any], permission: str) -> dict[str, Any]:
    """A successful answer names the permission the call just proved it holds (``permissions_available``)."""
    if payload.get("ok") is True:
        payload["permissions_available"] = sorted({*payload.get("permissions_available", []), permission})
    return payload


def with_refusal(payload: dict[str, Any]) -> dict[str, Any]:
    """``payload`` with the structured refusal field when its reason is a missing grant (the same field everywhere)."""
    grant = _GRANT_REASONS.get(str(payload.get("reason") or ""))
    return {**payload, **permission_refusal(grant)} if grant else payload


def argv(ops: list[dict[str, Any]], *, remote: bool = False, budget_ms: int = 8000, until: float | None = None) -> list[str]:
    """The program's argv: the file locally, its source over SSH (shell-quoted word by word by the caller), and the
    spec as the one word after it (ASCII JSON, so no locale on either end can change it). ``until`` (``deadline_in``)
    is for ``fit`` alone, which takes it out before the program runs."""
    body: dict[str, Any] = {"ops": ops, "budget_ms": int(budget_ms), **({"until": until} if until is not None else {})}
    spec = json.dumps(body, ensure_ascii=True, separators=(",", ":"))
    program = ["-e", INPUT_EVENTS_JS.read_text(encoding="utf-8")] if remote else [str(INPUT_EVENTS_JS)]
    return ["osascript", "-l", "JavaScript", *program, spec]


def ops_of(cmd: Any) -> list[dict[str, Any]] | None:
    """The operations of an argv this module built, else None."""
    if not isinstance(cmd, list) or cmd[:3] != ["osascript", "-l", "JavaScript"] or len(cmd) < 5:
        return None
    if not (cmd[3] == str(INPUT_EVENTS_JS) or (cmd[3] == "-e" and len(cmd) == 6
                                                and cmd[4] == INPUT_EVENTS_JS.read_text(encoding="utf-8"))):
        return None
    try:
        return json.loads(cmd[-1])["ops"]
    except (ValueError, KeyError, TypeError):
        return None


def method_of(ops: list[dict[str, Any]]) -> str:
    return "unicode_events" if any(op.get("op") == "type" for op in ops) else "cg_events"


def shown(cmd: list[str]) -> list[str]:
    """An input program argv as a receipt shows it: the spec readable (the text as typed, not escaped six times
    longer), the program's source over SSH named rather than repeated."""
    ops = ops_of(cmd)
    if ops is None:
        return cmd
    spec = json.loads(cmd[-1])
    spec.pop("until", None)  # a moment on this host's clock, meaningless in a receipt
    return [*cmd[:3], cmd[3] if cmd[3] != "-e" else "-e input_events.js", json.dumps(spec, ensure_ascii=False)]


def answer_fields(stdout: str, stderr: str = "") -> dict[str, Any]:
    """What the program reports it posted: ``posted`` events and, for text, ``chars_sent``, ``chars_total`` and
    ``override_only``; its answer on success (stdout), its ``__POSTED__`` line when it stopped on an error (stderr)."""
    lines = [line.partition("__POSTED__ ")[2] for line in str(stderr or "").splitlines() if line.startswith("__POSTED__ ")]
    for line in [*lines, *reversed(str(stdout or "").strip().splitlines())]:
        try:
            answer = json.loads(line)
        except ValueError:
            continue
        if isinstance(answer, dict):
            return {k: answer[k] for k in ("posted", "chars_sent", "chars_total", "override_only") if isinstance(answer.get(k), int)}
    return {}


def resolve_argv(prepared: list[Any], *, remote: bool = False) -> tuple[list[str] | None, list[str], list[str]]:
    """For a key sequence: one ``resolve`` call (posts nothing) that finds every character its steps name before the
    first step is sent, and which steps those are; (argv or None when no step names a character, chars, command chars).
    A Command chord step (the chord program's argv) names its character as its last word."""
    chars, command = [], []
    for cmd in prepared:
        for op in ops_of(cmd) or []:
            if isinstance(op.get("char"), str):
                chars.append(op["char"])
        if isinstance(cmd, list) and ops_of(cmd) is None and cmd[:3] == ["osascript", "-l", "JavaScript"] and len(cmd[-1]) == 1:
            command.append(cmd[-1])
    if not chars and not command:
        return None, [], []
    return argv([{"op": "resolve", "chars": chars, "command_chars": command}], remote=remote), chars, command


def position_of(output: Any) -> dict[str, Any]:
    """The pointer the program read (``{"x":..,"y":..}`` on its last line) as receipt fields, else nothing."""
    try:
        at = json.loads(str(output or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
    if not (isinstance(at, dict) and all(type(at.get(k)) is int for k in ("x", "y"))):
        return {}
    return {"x": at["x"], "y": at["y"], "coordinate_space": "input (macOS logical points)"}


def resolve_refusal(rc: int, stdout: str, stderr: str) -> dict[str, Any] | None:
    """The ``resolve`` answer (read on the target after any switch) as a refusal of the whole key sequence: no right
    to post first, then a character no key gives; an answer that cannot be read refuses too, since then nothing is
    known about the layout the steps would land in. None only when every character has its key."""
    try:
        answer = json.loads(stdout.strip().splitlines()[-1]) if rc == 0 else None
    except (ValueError, IndexError):
        answer = None
    if isinstance(answer, dict) and answer.get("post_events") is False:  # the grant answers before any layout fact
        return {"ok": False, "reason": "permission_denied", **permission_refusal("post_events"), "dispatched": False,
                "note": "the input was not sent",
                "error": "this process may not post input events (the Accessibility grant), so no step was sent"}
    if not isinstance(answer, dict) or answer.get("layout_unread") or not isinstance(answer.get("missing"), list):
        why = "the layout has no key data" if isinstance(answer, dict) and answer.get("layout_unread") else (
            (stderr or stdout or f"exit {rc}").strip()[:300])
        return {"ok": False, "reason": "layout_unread", "dispatched": False, "note": "the input was not sent",
                "error": f"the characters of the key sequence could not be looked up in the active keyboard layout ({why}), "
                         "so no step was sent"}
    if not (answer.get("missing") or answer.get("missing_command")):
        return None
    missing, command = list(answer.get("missing") or []), list(answer.get("missing_command") or [])
    layout = answer.get("layout") or "the active keyboard layout"
    return {"ok": False, "reason": "untypable_characters", "untyped": (missing + command)[:20], "dispatched": False,
            "note": "the input was not sent",
            "error": (f"no key of {layout} gives {', '.join(missing + command)}"
                      + (" (alone or with Shift or Option)" if missing else "") + (" under Command" if command else "")
                      + ", so no step of the sequence was sent: switch the layout, name the shortcut by its character "
                        "under Command, type the text with type_text, or use the app's own commands")}


def posting_seen(payload: dict[str, Any], posted: Any) -> bool:
    """Evidence that events went out: the input program's own count above zero, or the paste chord's line naming the
    key it pressed (a paste or a Command shortcut). An exit 0 with no answer (a cut reply, a program that printed
    nothing) proves no grant."""
    if isinstance(posted, int):
        return posted > 0
    return payload.get("method") in ("paste", "command_chord") and bool(re.search(r"\+ key code \d+ \(", str(payload.get("helper_stderr") or "")))


def with_grants(payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """A successful multi-step answer names the grants its steps proved (each step's own ``permissions_available``)."""
    for result in results:
        for grant in result.get("permissions_available") or []:
            proved(payload, grant)
    return payload


def posted_effect(payload: dict[str, Any]) -> dict[str, Any]:
    """What the input program said it posted decides the input's mark: an answer that posted nothing (an empty text)
    carries no input mark, and one that stopped after posting some keeps the mark and says how much went out. An
    answer that never came back (a lost SSH reply) has no count and is left as it is: unknown, never "not sent"."""
    posted = payload.get("posted")
    if payload.get("ok") is not True:
        payload.pop("permissions_available", None)  # a receipt demoted after the first look proves nothing
    elif payload.get("method") in _POSTING_METHODS and posting_seen(payload, posted):
        proved(payload, "post_events")  # events went out: the right to post them is a fact, not a preflight
    if not isinstance(posted, int):
        return payload
    if payload.get("ok") and posted == 0:
        payload.pop("observed_effect", None)
        payload["note"] = "the input program posted nothing (there was nothing to type): no input reached the screen"
    elif not payload.get("ok") and posted > 0:
        payload["observed_effect"] = True
        typed = f" ({payload['chars_sent']} of {payload.get('chars_total')} characters)" if "chars_sent" in payload else ""
        payload["note"] = f"the input program posted {posted} event(s){typed} before it stopped (see error); the rest was not sent"
    else:
        return payload
    if (payload.get("target") or {}).get("activated"):
        payload.update(observed_effect=True, activation_done=True)  # the switch before it is an effect of its own
    return payload


def key_op(mods: list[str], base: str, *, hold_ms: int | None = None) -> dict[str, Any]:
    """One key chord (or a hold of it) as an operation; raises ValueError for a modifier or key macOS has no code for.
    A single printable character is found in the active layout by the program; a named key is its virtual key code."""
    if any(m not in MODIFIERS for m in mods):
        raise ValueError(f"unsupported macOS modifiers {mods!r}")
    op: dict[str, Any] = {"op": "hold" if hold_ms is not None else "keys", "mods": list(mods)}
    if hold_ms is not None:
        op["ms"] = int(hold_ms)
    if base in MODIFIERS:
        op["mods"].append(base)  # a modifier alone: pressed as a modifier, its flag on the way down
        return op
    if len(base) == 1 and base.isprintable() and not base.isspace():
        op["char"] = base
    elif base.lower() in KEY_CODES:
        op["code"] = KEY_CODES[base.lower()]
    elif base.lower() in SYSTEM_KEYS:
        op["system"] = SYSTEM_KEYS[base.lower()]
    else:
        raise ValueError(f"unknown macOS key {base!r}")
    return op


# pyautogui key names (the remote snippets) -> this module's names; pyautogui's "delete" is forward delete.
_PYAUTOGUI_KEYS = {"enter": "return", "escape": "esc", "delete": "fwd-delete", "backspace": "delete",
                   "pagedown": "page-down", "pageup": "page-up", "down": "arrow-down", "up": "arrow-up",
                   "left": "arrow-left", "right": "arrow-right", "winleft": "cmd", "super": "cmd", "meta": "cmd",
                   "command": "cmd", "option": "alt", "control": "ctrl"}


def _call(text: str) -> tuple[str, list[Any], dict[str, Any]]:
    """(name, literal args, literal keywords) of ONE ``pyautogui.<name>(...)`` call; raises on anything else."""
    call = ast.parse(text, mode="eval").body
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)):
        raise ValueError("not a call")
    return call.func.attr, [ast.literal_eval(a) for a in call.args], {k.arg: ast.literal_eval(k.value) for k in call.keywords}


def ops_for_pyautogui(code: str) -> tuple[list[dict[str, Any]], str]:
    """The pyautogui snippets this skill's remote paths emit, as operations for a Mac over SSH; (ops, error)."""
    text = str(code or "").strip()
    if re.fullmatch(r"print\(.*pyautogui\.position\(\)\)", text):
        return [{"op": "position"}], ""
    try:
        m = re.fullmatch(r"(pyautogui\.moveTo\(.*?\)); (pyautogui\.dragTo\(.*\))", text)
        if m:
            (_n, (sx, sy), _k), (_m, (ex, ey), kw) = _call(m.group(1)), _call(m.group(2))
            return [{"op": "drag", "from": [sx, sy], "to": [ex, ey], "button": kw.get("button", "left")}], ""
        name, args, kw = _call(text)
        if name == "click":
            return [{"op": "click", "x": args[0], "y": args[1], "button": kw.get("button", "left"),
                     "count": max(1, min(3, int(kw.get("clicks", 1))))}], ""
        if name == "moveTo":
            return [{"op": "move", "x": args[0], "y": args[1]}], ""
        if name in ("mouseDown", "mouseUp"):
            op = {"op": "button", "down": name == "mouseDown", "button": kw.get("button", "left")}
            if "x" in kw:
                op.update(x=kw["x"], y=kw["y"])
            return [op], ""
        if name == "typewrite":
            return [{"op": "type", "text": str(args[0]), "interval_ms": int(float(kw.get("interval", 0)) * 1000)}], ""
        if name in ("press", "hotkey"):
            keys = [_PYAUTOGUI_KEYS.get(str(t).lower(), str(t).lower() if len(str(t)) > 1 else str(t)) for t in args]
            return [key_op(keys[:-1], keys[-1])], ""
        if name in ("scroll", "hscroll"):
            return [{"op": "scroll", "dy": int(args[0])} if name == "scroll" else {"op": "scroll", "dx": -int(args[0])}], ""
    except (ValueError, SyntaxError, IndexError, TypeError) as exc:
        return [], f"unsupported pyautogui snippet for ssh_macos ({exc}): {text[:120]}"
    return [], f"unsupported pyautogui snippet for ssh_macos: {text[:120]}"
