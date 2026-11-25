"""Shared fixtures of the unix_computer_use tests: the plugin loaded from its payload, a fake host ``API`` (with
``StopAPI``, a PluginAPI 2.6 host answering each Stop reading) and a faked local Mac (``mac``), whose OS boundary is
``subprocess.run``; ``inputs`` and ``receipt`` read what a call recorded and returned."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use"


def load_plugin():
    spec = importlib.util.spec_from_file_location("unix_computer_use_targeting_plugin", SKILL / "plugin.py",
                                                  submodule_search_locations=[str(SKILL)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class API:
    def __init__(self, state: Path) -> None:
        self.state, self.tools, self.meta = state, {}, {}

    def get_state_dir(self) -> str:
        return str(self.state)

    def skill_job_dir(self, job_id: str) -> Path:
        path = self.state / "jobs" / job_id
        (path / "output").mkdir(parents=True, exist_ok=True)
        return path

    def register_tool(self, name, handler, **metadata):
        self.tools[name] = handler
        self.meta[name] = metadata


SAFARI = ("Safari", 100, "com.apple.Safari")
KEYNOTE = ("Keynote", 200, "com.apple.iWork.Keynote")


def app_dict(triple) -> dict:
    return dict(zip(("app", "pid", "app_id"), triple))


def helper_op(argv: list[str]) -> list[str]:
    """The native helper's op and args from its argv (``python -I -B macos_native.py <op> ...``)."""
    idx = next(i for i, part in enumerate(argv) if str(part).endswith("macos_native.py"))
    return [str(a) for a in argv[idx + 1:]]


CHORD_POSTED = "paste chord: Command + key code 9 (v under Command in com.apple.keylayout.ABC)\n"
def ran(command: str, stdout: str = "") -> str:
    """What an SSH Mac answers when the skill's input wrapper ran: its two place lines around the program's own output
    (unread fronts here). A reply without them is a script that never reached the remote shell."""
    if "__FRONT_AFTER__" in command and "__FRONT_" not in stdout:
        return f"__FRONT_BEFORE__\n{stdout}\n__FRONT_AFTER__\n"
    return stdout


EVENTS_POSTED = '{"posted":2}\n'  # the input program's answer on success: what it posted


def observing(helper):
    """``observe`` (resolve and tree in ONE helper process), for a fake without its own answer to it ("no fake"),
    answered from that fake's ``resolve`` and ``tree`` answers."""
    def answer(args, state):
        out = helper(args, state)
        if args[0] != "observe" or not str(out.get("error") or "").startswith("no fake"):
            return out
        spec = json.loads(args[1])
        found = helper(["resolve", spec["app"]], state) if spec["app"] else {"ok": True, "candidates": [app_dict(state["front"])]}
        if not found.get("ok"):
            return {**found, "stage": "resolve"}
        target = found["candidates"][0]
        return {**helper(["tree", str(target["pid"]), str(spec["max_nodes"]), str(spec["timeout_ms"])], state), "target": target}
    return answer


def is_input(argv: list[str]) -> bool:
    """A call of the macOS input program (``osascript -l JavaScript input_events.js '<spec>'``)."""
    return (argv[:3] == ["osascript", "-l", "JavaScript"] and len(argv) == 5 and argv[3].endswith("input_events.js")
            and json.loads(argv[-1])["ops"][0]["op"] != "resolve")  # a lookup that posts nothing is not an input


def ops(argv: list[str]) -> list[dict]:
    """The operations an input program call was given."""
    return json.loads(argv[-1])["ops"]


def key(code: int | None = None, *mods: str, char: str | None = None) -> dict:
    """One ``keys`` operation: a virtual key code or a character, under ``mods``."""
    return {"op": "keys", "mods": list(mods), **({"code": code} if code is not None else {"char": char})}


DOWN, RIGHT, RETURN, TAB = key(125), key(124), key(36), key(48)
# The resolve answer of a layout that gives every character a key sequence names (the program's own words).
RESOLVED_ALL = '{"layout":"com.apple.keylayout.ABC","post_events":true,"missing":[],"missing_command":[]}\n'


def mac(tmp_path, monkeypatch, *, front=SAFARI, helper=None, events=(0, EVENTS_POSTED, ""), lsappinfo=True, on_input=None,
        chord=(0, "", CHORD_POSTED), resolved=None):
    """A faked local Mac: lsappinfo reports ``state['front']``; the native helper answers through ``helper(args, state)``;
    ``on_input(argv, state)`` runs when the input program is called (a key that moves the front, for instance) and it
    answers ``events`` (exit code, stdout, stderr); the paste chord program answers ``chord``."""
    module = load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(module, "_session_type", lambda: "native")
    tools = {"screencapture", "osascript", "pbcopy"} | ({"lsappinfo"} if lsappinfo else set())
    monkeypatch.setattr(module, "_which", lambda name: f"/usr/bin/{name}" if name in tools else "")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    state = {"front": front, "calls": [], "helper": [], "clipboard": None, "chords": [], "resolved": []}

    def default_helper(args, state):
        op = args[0]
        if op == "activate":
            query = args[1]
            if query in ("Keynote", "pid:200", "com.apple.iWork.Keynote"):
                before = app_dict(state["front"])
                state["front"] = KEYNOTE
                target = app_dict(KEYNOTE)
                if before["pid"] == 200:
                    return {"ok": True, "target": target, "front_before": before, "front": target, "activated": False,
                            "already_front": True, "verified": True, "waited_ms": 0}
                return {"ok": True, "target": target, "front_before": before, "front": target, "activated": True,
                        "verified": True, "method": "ax_frontmost", "waited_ms": 120}
            if query == "Stuck":
                before = app_dict(state["front"])
                return {"ok": False, "reason": "not_frontmost_after_activate", "target": {"app": "Stuck", "pid": 300, "app_id": "x.stuck"},
                        "front_before": before, "front": before, "activated": False, "verified": False, "method": "system_events", "waited_ms": 2000}
            if query == "Term":
                return {"ok": False, "reason": "ambiguous", "candidates": [{"app": "Terminal", "pid": 1}, {"app": "Termius", "pid": 2}]}
            return {"ok": False, "reason": "not_running", "running": ["Safari", "Keynote"], "candidates": []}
        if op == "resolve":
            if args[1] in ("Keynote", "pid:200"):
                return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
            return {"ok": False, "reason": "not_running", "candidates": [], "running": ["Safari"]}
        if op == "perms":
            return {"ok": True, "accessibility": "trusted", "screen_recording": "granted", "post_events": "granted", "listen_events": "denied"}
        if op == "displays":
            return {"ok": True, "displays": [{"id": 1, "main": True, "x": 0, "y": 0, "width": 1728, "height": 1117, "index": 1},
                                             {"id": 2, "main": False, "x": 1728, "y": 0, "width": 1920, "height": 1080, "index": 2}]}
        if op == "tree":
            return {"ok": True, "pid": int(args[1]), "root": "AXFocusedWindow", "window": "Deck.key", "visited": 1, "truncated": False,
                    "truncated_by": [], "errors": {}, "elapsed_ms": 5, "marks": [],
                    "nodes": [{"id": 1, "depth": 0, "role": "Window", "label": "Deck.key", "x": 1800, "y": 100, "w": 800, "h": 600}],
                    "limits": {"max_nodes": int(args[2]), "timeout_ms": int(args[3])}}
        if op == "apps":
            return {"ok": True, "frontmost": app_dict(state["front"]), "titles_available": True,
                    "apps": [{"app": "Safari", "pid": 100, "app_id": "com.apple.Safari", "frontmost": state["front"][1] == 100, "scriptable": True,
                              "windows": [{"title": "Ouroboros", "minimized": False, "main": True, "bounds": [0, 0, 800, 600], "window_id": 39}]},
                             {"app": "Keynote", "pid": 200, "app_id": "com.apple.iWork.Keynote", "frontmost": state["front"][1] == 200,
                              "scriptable": True, "windows": [{"title": "Deck.key", "minimized": False, "main": True}]}],
                    "truncated": False, "limit": int(args[1]), "total": 2}
        return {"ok": False, "error": f"no fake for {op}"}

    answer = observing(helper or default_helper)

    def run(cmd, *a, **k):
        argv = [str(part) for part in cmd]
        state["calls"].append(argv)
        if argv[:2] == ["lsappinfo", "front"]:
            return subprocess.CompletedProcess(argv, 0, "ASN:0x0-0x1:\n", "")
        if argv[:2] == ["lsappinfo", "info"]:
            name, pid, bid = state["front"]
            return subprocess.CompletedProcess(argv, 0, f'"pid"={pid}\n"CFBundleIdentifier"="{bid}"\n"LSDisplayName"="{name}"\n', "")
        if any(part.endswith("macos_native.py") for part in argv):
            args = helper_op(argv)
            state["helper"].append(args)
            out = answer(args, state)
            return subprocess.CompletedProcess(argv, 0 if out.get("ok") else 1, json.dumps(out) + "\n", "")
        if argv[:3] == ["osascript", "-l", "JavaScript"] and argv[3].endswith("input_events.js") and not is_input(argv):
            state["resolved"].append(ops(argv)[0])  # a key sequence's lookup of its characters: posts nothing
            return subprocess.CompletedProcess(argv, *(resolved or (0, RESOLVED_ALL, "")))
        if is_input(argv):
            if on_input:
                on_input(argv, state)
            rc, out, err = events
            return subprocess.CompletedProcess(argv, rc, out, err)
        if argv[0] == "pbcopy":
            state["clipboard"] = k.get("input")
            state["clipboard_contract"] = (k.get("encoding"), (k.get("env") or {}).get("LC_ALL"))
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:3] == ["osascript", "-l", "JavaScript"]:
            state["chords"].append(argv)
            rc, out, err = chord
            return subprocess.CompletedProcess(argv, rc, out, err)
        if argv[0] == "screencapture":
            from PIL import Image
            Image.new("RGB", (64, 40), (1, 2, 3)).save(Path(argv[-1]), format="PNG")
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    api = API(tmp_path / "state")
    module.register(api)
    return module, api, state


def inputs(state) -> list[list[dict]]:
    """The operations of every input program call, in order."""
    return [ops(argv) for argv in state["calls"] if is_input(argv)]


def receipt(text: str) -> dict:
    out = json.loads(text)
    assert isinstance(out, dict)
    return out


CONTINUE = {"status": "continue", "reason": "", "note": ""}
STOPPED = {"status": "stop", "reason": "cancelled", "note": "Stop is in force for this task"}


class StopAPI(API):
    """A host serving PluginAPI 2.6: ``answers`` are its successive readings of the call's Stop (then ``continue``);
    each reading is recorded in ``calls`` beside the OS calls, so its place among them is visible."""

    def __init__(self, state_dir: Path, calls: list, answers) -> None:
        super().__init__(state_dir)
        self.calls, self.answers = calls, list(answers)

    def get_stop_state(self) -> dict:
        self.calls.append(["get_stop_state"])
        return dict(self.answers.pop(0)) if self.answers else dict(CONTINUE)
