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
from typing import Any

INPUT_EVENTS_JS = pathlib.Path(__file__).with_name("input_events.js")
EVENT_TAG = 0x4F55524F  # "OURO" in kCGEventSourceUserData of every posted event (the same constant as the program's)
# Receipt methods of programs that refuse BEFORE posting anything (a missing grant, a key no layout gives): such a
# refusal is a known "not sent". The paste chord is not one of them: its clipboard write ran first.
REFUSES_BEFORE_POSTING = frozenset({"command_chord", "cg_events", "unicode_events"})
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
    **{name: code for name, code in (("cmd", 55), ("shift", 56), ("alt", 58), ("ctrl", 59), ("fn", 63))},
}

# The one place the refusal's field names live (a later contract may rename them).
_REFUSAL_FIELD = "refusal"
_GRANT_PANES = {"accessibility": "Accessibility", "post_events": "Accessibility",
                "screen_recording": "Screen & System Audio Recording"}


def permission_refusal(grant: str) -> dict[str, Any]:
    """The one structured field of a refusal for a missing macOS grant: its kind, which grant, and that only the owner
    can give it (System Settings; nothing here asks the system to prompt)."""
    return {_REFUSAL_FIELD: {"kind": "permission", "grant": grant, "fixable_by": "owner",
                             "where": f"System Settings > Privacy & Security > {_GRANT_PANES[grant]}"}}


def argv(ops: list[dict[str, Any]], *, remote: bool = False, budget_ms: int = 8000) -> list[str]:
    """The program's argv: the file locally, its source over SSH (shell-quoted word by word by the caller), and the
    spec as the one word after it (ASCII JSON, so no locale on either end can change it)."""
    spec = json.dumps({"ops": ops, "budget_ms": int(budget_ms)}, ensure_ascii=True, separators=(",", ":"))
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


def answer_fields(stdout: str) -> dict[str, Any]:
    """What the program reports about typed text (``chars_sent``, ``chars_total``, ``override_only``), else nothing."""
    for line in reversed(str(stdout or "").strip().splitlines()):
        try:
            answer = json.loads(line)
        except ValueError:
            continue
        if isinstance(answer, dict):
            return {k: answer[k] for k in ("chars_sent", "chars_total", "override_only") if isinstance(answer.get(k), int)}
    return {}


def key_op(mods: list[str], base: str, *, hold_ms: int | None = None) -> dict[str, Any]:
    """One key chord (or a hold of it) as an operation; raises ValueError for a modifier or key macOS has no code for.
    A single printable character is found in the active layout by the program; a named key is its virtual key code."""
    if any(m not in MODIFIERS for m in mods):
        raise ValueError(f"unsupported macOS modifiers {mods!r}")
    op: dict[str, Any] = {"op": "hold" if hold_ms is not None else "keys", "mods": list(mods)}
    if hold_ms is not None:
        op["ms"] = int(hold_ms)
        if base in MODIFIERS:
            op["mods"].append(base)  # a pure-modifier hold
            return op
    if len(base) == 1 and base.isprintable() and not base.isspace():
        op["char"] = base
    elif base.lower() in KEY_CODES:
        op["code"] = KEY_CODES[base.lower()]
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
