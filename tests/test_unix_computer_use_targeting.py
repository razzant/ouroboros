"""unix_computer_use: inputs addressed to an application, place receipts, bounded element reads.

The 27.10 incident class: the model pressed cmd+tab and an arrow key into whatever was in front and read
``observed_effect: true`` (the helper exited 0) as "the slide changed". Every test here fakes only the OS
boundary (``subprocess.run``: cliclick, lsappinfo, wmctrl/xdotool, the native helper's JSON; ``urlopen`` toward
an OSWorld guest; ``ssh``) and asserts both directions: the guard fires (nothing is sent, no mark) and stays
quiet (one input, the mark, the place). The same receipt contract is pinned on every transport.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

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
    """The native helper's op and args from its argv (``python -I macos_native.py <op> ...``)."""
    idx = next(i for i, part in enumerate(argv) if str(part).endswith("macos_native.py"))
    return [str(a) for a in argv[idx + 1:]]


def mac(tmp_path, monkeypatch, *, front=SAFARI, helper=None, cliclick=(0, "", ""), lsappinfo=True, on_input=None):
    """A faked local Mac: lsappinfo reports ``state['front']``; the native helper answers through ``helper(args, state)``;
    ``on_input(argv, state)`` runs when cliclick is called (a key that moves the front, for instance)."""
    module = load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "macos")
    monkeypatch.setattr(module, "_session_type", lambda: "native")
    tools = {"cliclick", "screencapture", "osascript", "pbcopy"} | ({"lsappinfo"} if lsappinfo else set())
    monkeypatch.setattr(module, "_which", lambda name: f"/usr/bin/{name}" if name in tools else "")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    state = {"front": front, "calls": [], "helper": [], "clipboard": None}

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

    answer = helper or default_helper

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
        if argv[0] == "cliclick":
            if on_input:
                on_input(argv, state)
            rc, out, err = cliclick
            return subprocess.CompletedProcess(argv, rc, out, err)
        if argv[0] == "pbcopy":
            state["clipboard"] = k.get("input")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "screencapture":
            from PIL import Image
            Image.new("RGB", (64, 40), (1, 2, 3)).save(Path(argv[-1]), format="PNG")
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    api = API(tmp_path / "state")
    module.register(api)
    return module, api, state


def inputs(state) -> list[list[str]]:
    return [argv for argv in state["calls"] if argv[0] == "cliclick"]


def receipt(text: str) -> dict:
    out = json.loads(text)
    assert isinstance(out, dict)
    return out


# --- an input aimed at an application (macOS) -----------------------------------------------------

def test_a_targeted_key_is_sent_only_after_the_target_is_verified_in_front(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["observed_effect"] is True, out
    assert inputs(state) == [["cliclick", "kp:arrow-right"]], "exactly one input, after the switch"
    assert state["helper"][0][:2] == ["activate", "Keynote"], "one helper call resolves, activates and verifies"
    assert out["target"]["requested"] == "Keynote" and out["target"]["activated"] is True and out["target"]["verified"] is True
    assert out["delivered_to"] == app_dict(KEYNOTE)
    assert out["front_before"] == app_dict(SAFARI)
    assert "front_after" not in out, "the front did not change again after the input"
    assert "no UI change" in out["note"] and "Keynote" in out["note"]


def test_a_target_that_did_not_come_forward_sends_nothing(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="right", app="Stuck"))
    assert out["ok"] is False and "observed_effect" not in out, out
    assert out["reason"] == "not_frontmost_after_activate" and "Safari" in out["error"] and "nothing was sent" in out["error"]
    assert out["target"]["verified"] is False and inputs(state) == [], "the guard fires: no cliclick at all"


@pytest.mark.parametrize("query, reason", [("NoSuchApp", "not_running"), ("Term", "ambiguous")])
def test_an_unresolved_target_refuses_before_any_input(tmp_path, monkeypatch, query, reason):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["click"](x=10, y=20, raw=True, app=query))
    assert out["ok"] is False and out["reason"] == reason and "observed_effect" not in out, out
    assert inputs(state) == []
    if reason == "ambiguous":
        assert [c["app"] for c in out["candidates"]] == ["Terminal", "Termius"] and "pid:" in out["error"]


def test_an_untargeted_input_keeps_working_and_names_who_received_it(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="right"))
    assert out["ok"] is True and out["observed_effect"] is True
    assert inputs(state) == [["cliclick", "kp:arrow-right"]] and state["helper"] == [], "no activation without app="
    assert out["delivered_to"]["app"] == "Safari" and "target" not in out and "front_before" not in out
    # app="" is what a voice model sends for an unused optional field: identical to omitting it.
    state["calls"].clear()
    out2 = receipt(api.tools["key"](keys="right", app=""))
    assert out2["delivered_to"]["app"] == "Safari" and state["helper"] == []


def test_without_lsappinfo_the_place_is_unknown_and_the_input_still_goes(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, lsappinfo=False)
    out = receipt(api.tools["key"](keys="return"))
    assert out["ok"] is True and out["observed_effect"] is True
    assert out["delivered_to"] == "unknown" and "unknown" in out["note"] and "front_after" not in out
    assert inputs(state) == [["cliclick", "kp:return"]]


def test_a_front_change_after_the_input_is_reported(tmp_path, monkeypatch):
    def switch(argv, state):
        state["front"] = KEYNOTE  # cmd+tab moved the front
    module, api, state = mac(tmp_path, monkeypatch, on_input=switch)
    out = receipt(api.tools["key"](keys="cmd+tab"))
    assert out["delivered_to"]["app"] == "Safari" and out["front_after"]["app"] == "Keynote", out


def test_a_dropped_input_is_an_honest_failure(tmp_path, monkeypatch):
    """cliclick exits 0 and only warns on stderr when macOS has no Accessibility grant for the posting process."""
    warning = "WARNING: Accessibility privileges not enabled. Many actions may fail.\n"
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(0, "", warning))
    out = receipt(api.tools["key"](keys="right"))
    assert out["ok"] is False and "observed_effect" not in out, out
    assert out["reason"] == "permission_denied" and out["permission"] == {"accessibility": "untrusted", "source": "helper stderr"}
    assert "Accessibility" in out["helper_stderr"]
    # A switch observed before the dropped input keeps its own mark.
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "permission_denied" and out["observed_effect"] is True and out["activation_done"] is True
    # Other stderr noise is kept beside a successful dispatch.
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(0, "", "some note\n"))
    out = receipt(api.tools["key"](keys="right"))
    assert out["ok"] is True and out["observed_effect"] is True and out["helper_stderr"] == "some note"


def test_characters_the_layout_cannot_type_are_an_honest_failure(tmp_path, monkeypatch):
    """Measured on the live probe (27.10.2026): under a Cyrillic layout cliclick typed none of 'ouroboros probe',
    printed one 'Unable to get key code for <c>' line per letter and exited 0. The hint names the method used."""
    stderr = "".join(f"Unable to get key code for {c} (see README-Characters.md)\n" for c in "ourbs")
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(0, "", stderr))
    out = receipt(api.tools["type_text"](text="ouroboros probe"))
    assert out["ok"] is False and "observed_effect" not in out, out
    assert out["reason"] == "untypable_characters" and out["untyped"] == list("ourbs") and "via=paste" in out["error"]
    shortcut = receipt(api.tools["key"](keys="cmd+s"))
    assert shortcut["reason"] == "untypable_characters" and "layout" in shortcut["error"] and "via=paste" not in shortcut["error"]
    pasted = receipt(api.tools["type_text"](text="x", via="paste"))
    assert pasted["reason"] == "untypable_characters" and "paste chord" in pasted["error"] and "via=paste" not in pasted["error"]
    module, api, state = mac(tmp_path, monkeypatch)
    assert receipt(api.tools["type_text"](text="ouroboros probe"))["observed_effect"] is True, "a clean dispatch keeps the mark"


def test_an_activation_that_ran_before_a_failed_input_keeps_its_mark_and_says_the_input_is_unknown(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(1, "", "boom"))
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["observed_effect"] is True and out["activation_done"] is True, out
    assert out["target"]["activated"] is True and out["error"] == "boom"
    assert "delivered_to" not in out and out["front_at_dispatch"]["app"] == "Keynote"
    assert "unknown" in out["note"] and "dispatched while" not in out["note"] and "switch to Keynote" in out["note"]


# --- key sequences: one switch, every later step re-checked, every recipient named ---------------

def test_a_key_sequence_brings_the_target_forward_once_and_rechecks_before_each_step(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="down down return", app="Keynote"))
    assert out["ok"] is True and out["steps"] == 3 and out["observed_effect"] is True
    assert [args[0] for args in state["helper"]] == ["activate"], "one activation for the whole sequence"
    assert len(inputs(state)) == 3 and out["target"]["activated"] is True and out["delivered_to"]["app"] == "Keynote"
    assert "steps_delivered_to" not in out and "no UI change" in out["note"]
    reads = [argv for argv in state["calls"] if argv[:2] == ["lsappinfo", "front"]]
    assert len(reads) >= 2, "the front is read again before the later steps"


def test_a_targeted_sequence_stops_when_its_app_leaves_the_front(tmp_path, monkeypatch):
    """The 27.10 shape inside one call: cmd+tab (sent to Keynote) moves the front to Safari; the arrow is NOT sent."""
    def switch(argv, state):
        if "kp:tab" in argv:
            state["front"] = SAFARI
    module, api, state = mac(tmp_path, monkeypatch, on_input=switch)
    out = receipt(api.tools["key"](keys="cmd+tab right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "target_lost" and out["steps_done"] == 1 and out["steps_total"] == 2, out
    assert inputs(state) == [["cliclick", "kd:cmd", "kp:tab", "ku:cmd"]], "the arrow was never sent"
    assert out["observed_effect"] is True, "the first step ran"
    assert out["delivered_to"]["app"] == "Keynote" and out["front"]["app"] == "Safari" and out["front_after"]["app"] == "Safari"
    assert "no longer in front" in out["error"] and "nothing more was sent" in out["error"]


def test_an_untargeted_sequence_names_every_recipient_when_they_differ(tmp_path, monkeypatch):
    def switch(argv, state):
        if "kp:tab" in argv:
            state["front"] = KEYNOTE
    module, api, state = mac(tmp_path, monkeypatch, on_input=switch)
    out = receipt(api.tools["key"](keys="cmd+tab right"))
    assert out["ok"] is True and out["steps"] == 2 and len(inputs(state)) == 2
    assert out["delivered_to"]["app"] == "Safari" and out["front_after"]["app"] == "Keynote"
    assert [r["app"] for r in out["steps_delivered_to"]] == ["Safari", "Keynote"], out


def test_a_sequence_refused_at_its_first_step_carries_the_refusal(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="down return", app="Term"))
    assert out["ok"] is False and out["reason"] == "ambiguous" and [c["app"] for c in out["candidates"]] == ["Terminal", "Termius"]
    assert inputs(state) == [] and "observed_effect" not in out


def test_mouse_up_and_move_never_change_the_front(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    for tool in ("mouse_up", "move"):
        assert "app" not in api.meta[tool]["schema"]["properties"], tool
    out = receipt(api.tools["mouse_down"](x=5, y=6, raw=True, app="Keynote"))
    assert out["ok"] is True and out["target"]["activated"] is True and inputs(state)[-1] == ["cliclick", "dd:5,6"]


# --- activate_app ---------------------------------------------------------------------------------

def test_activate_app_marks_only_a_switch_it_observed(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["activate_app"](app="Keynote"))
    assert out["ok"] is True and out["activated"] is True and out["observed_effect"] is True
    assert out["front"]["app"] == "Keynote" and out["front_before"]["app"] == "Safari" and inputs(state) == []
    again = receipt(api.tools["activate_app"](app="Keynote"))
    assert again["ok"] is True and again["activated"] is False and "observed_effect" not in again, "already in front: no effect"
    assert again["target"]["already_front"] is True and again["target"]["verified"] is True
    refused = receipt(api.tools["activate_app"](app="Stuck"))
    assert refused["ok"] is False and "observed_effect" not in refused and refused["reason"] == "not_frontmost_after_activate"
    assert receipt(api.tools["activate_app"](app=""))["ok"] is False


def test_activate_app_does_not_claim_a_switch_when_the_previous_front_was_unknown(tmp_path, monkeypatch):
    def helper(args, state):
        return {"ok": True, "target": app_dict(KEYNOTE), "front_before": None, "front": app_dict(KEYNOTE),
                "activated": None, "verified": True, "method": "ax_frontmost", "waited_ms": 90}
    module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    out = receipt(api.tools["activate_app"](app="Keynote"))
    assert out["ok"] is True and out["activated"] is None and "observed_effect" not in out and "unknown" in out["note"]
    sent = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert sent["ok"] is True and sent["observed_effect"] is True and sent["target"]["verified"] is True, "verified in front: the input goes"


# --- window_list, capabilities, ax_tree ------------------------------------------------------------

def test_window_list_names_the_front_and_the_scriptable_apps(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["window_list"]())
    assert out["ok"] is True and out["frontmost"]["app"] == "Safari" and out["titles_available"] is True
    rows = {row["app"]: row for row in out["windows"]}
    assert rows["Keynote"]["frontmost"] is False and rows["Keynote"]["scriptable"] is True
    assert rows["Safari"]["windows"][0]["title"] == "Ouroboros"
    only = receipt(api.tools["window_list"](app="keynote"))
    assert [row["app"] for row in only["windows"]] == ["Keynote"]
    # The helper not answering is a failure, not an empty list.
    module, api, state = mac(tmp_path, monkeypatch, helper=lambda args, st: {"ok": False, "error": "no answer"})
    out = receipt(api.tools["window_list"]())
    assert out["ok"] is False and "windows" not in out and out["frontmost"]["app"] == "Safari"


def test_capabilities_report_real_grants_and_targeting(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    caps = receipt(api.tools["capabilities"]())
    assert caps["permission_state_verified"] is True
    assert caps["permissions"] == {"accessibility": "trusted", "screen_recording": "granted", "post_events": "granted", "listen_events": "denied"}
    assert caps["frontmost"]["app"] == "Safari" and caps["targeting"]["app"] == "supported"
    module, api, state = mac(tmp_path, monkeypatch, helper=lambda args, st: {"ok": False, "error": "no answer"})
    caps = receipt(api.tools["capabilities"]())
    assert caps["permission_state_verified"] is False and "error" in caps["permissions"] and caps["targeting"]["app"] == "unavailable"


def test_ax_tree_reads_a_named_app_without_bringing_it_forward(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["ax_tree"](app="Keynote", max_nodes=40, timeout_ms=500))
    assert out["ok"] is True and out["app"]["pid"] == 200 and out["window"] == "Deck.key"
    assert [args[0] for args in state["helper"]] == ["resolve", "tree"], "resolve, then read; never activate"
    assert state["helper"][1][1:] == ["200", "40", "500", "window", "false"]
    assert state["front"] == SAFARI, "reading does not move the front"
    missing = receipt(api.tools["ax_tree"](app="Nope"))
    assert missing["ok"] is False and missing["reason"] == "not_running"


# --- type_text via=paste and the display transform -------------------------------------------------

def test_type_text_via_paste_writes_the_clipboard_after_the_target_is_verified(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["type_text"](text="Привет, мир " * 40, via="paste", app="Keynote"))
    assert out["ok"] is True and out["method"] == "paste" and out["clipboard_replaced"] is True
    assert state["clipboard"].startswith("Привет") and inputs(state) == [["cliclick", "kd:cmd", "t:v", "ku:cmd"]]
    assert out["delivered_to"]["app"] == "Keynote"
    activate_at = next(i for i, argv in enumerate(state["calls"]) if any(p.endswith("macos_native.py") for p in argv))
    copy_at = next(i for i, argv in enumerate(state["calls"]) if argv[0] == "pbcopy")
    assert activate_at < copy_at, "the clipboard is written only after the target is verified in front"
    plain = receipt(api.tools["type_text"](text="hi"))
    assert plain["ok"] is True and inputs(state)[-1] == ["cliclick", "t:hi"] and "method" not in plain
    # A refusal leaves the owner's clipboard alone and says nothing was sent.
    module, api, state = mac(tmp_path, monkeypatch)
    refused = receipt(api.tools["type_text"](text="secret", via="paste", app="NoSuchApp"))
    assert refused["ok"] is False and state["clipboard"] is None and inputs(state) == []
    assert receipt(api.tools["type_text"](text="x", via="clipboard"))["ok"] is False, "an unknown via is refused, not typed"


def test_screenshot_transform_uses_display_bounds_and_the_target_display(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (3456, 2234))
    out = receipt(api.tools["screenshot"](job_id="main"))
    assert out["ok"] is True and out["display"]["index"] == 1 and out["display"]["selected_by"] == "default"
    tf = out["coord_transform"]
    assert (tf["input_w"], tf["input_h"], tf["ox"], tf["oy"]) == (1728, 1117, 0, 0) and tf["approx"] is False
    assert inputs(state) == [] and ["screencapture", "-x", out["full_resolution_path"]] in state["calls"]
    # app= : the display holding that app's window is captured and the transform carries its origin.
    out = receipt(api.tools["screenshot"](job_id="deck", app="Keynote"))
    assert out["display"]["index"] == 2 and out["display"]["selected_by"] == "app" and out["app"]["pid"] == 200
    assert any(argv[:4] == ["screencapture", "-x", "-D", "2"] for argv in state["calls"])
    assert (out["coord_transform"]["ox"], out["coord_transform"]["oy"]) == (1728, 0)
    click = receipt(api.tools["click"](x=100, y=50, observation_id=out["observation_id"]))
    assert click["ok"] is True
    sx = out["coord_transform"]["sx"]
    assert inputs(state)[-1] == ["cliclick", f"c:{round(100 * sx) + 1728},{round(50 * out['coord_transform']['sy'])}"]


def test_screenshot_with_an_unlocatable_window_says_it_captured_the_main_display(tmp_path, monkeypatch):
    def helper(args, state):
        if args[0] == "tree":
            return {"ok": False, "reason": "accessibility_untrusted", "pid": int(args[1])}
        if args[0] == "resolve":
            return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
        if args[0] == "displays":
            return {"ok": True, "displays": [{"id": 1, "main": True, "x": 0, "y": 0, "width": 1728, "height": 1117, "index": 1}]}
        return {"ok": False, "error": "no fake"}
    module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (3456, 2234))
    out = receipt(api.tools["screenshot"](job_id="fallback", app="Keynote"))
    assert out["ok"] is True and "app" not in out, "the main display is not presented as the named app's"
    assert out["display"]["selected_by"] == "fallback" and "accessibility_untrusted" in out["display"]["app_window_unread"]


def test_screenshot_with_unreadable_displays_has_no_mapped_input(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, helper=lambda args, st: {"ok": False, "error": "locked"})
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (800, 600))
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (0, 0))
    out = receipt(api.tools["screenshot"](job_id="fallback"))
    assert out["ok"] is True and "display" not in out and out["mapped_input_available"] is False
    assert "observation_id" not in out and "coord_transform" not in out
    assert out["input_geometry_evidence"] == "unavailable" and out["input_width"] is None
    assert "locked" in out["coordinate_note"] and "Capture only" in out["coordinate_note"]
    assert receipt(api.tools["click"](x=10, y=10, observation_id=""))["dispatched"] is False
    assert inputs(state) == []
    # A clean-looking Finder ratio still cannot prove which display was captured.
    monkeypatch.setattr(module, "_macos_logical_size", lambda: (400, 300))
    out = receipt(api.tools["screenshot"](job_id="finder"))
    assert out["input_width"] == 400 and out["mapped_input_available"] is False
    assert out["input_geometry_evidence"] == "finder_desktop_union"
    assert "observation_id" not in out and "coord_transform" not in out


# --- registration -------------------------------------------------------------------------------

def test_surface_exposes_targeting_on_input_tools(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    assert "activate_app" in api.tools and "element_act" in api.tools and "clear_active_connection" not in api.tools
    assert len(api.tools) == 25
    for tool in ("click", "double_click", "triple_click", "left_click_drag", "mouse_down", "type_text", "key", "hold_key", "scroll"):
        assert api.meta[tool]["schema"]["properties"]["app"] == {"type": "string", "default": ""}, tool
    lead = api.meta["key"]["description"]
    assert lead.startswith("Press keys in the front app or app=") and "not that the app reacted" in lead[:120]
    assert "run_command" in lead
    for tool in ("click", "double_click", "triple_click", "left_click_drag", "mouse_down", "hold_key", "scroll"):
        assert "app=" in api.meta[tool]["description"][:120], tool
    assert all(api.meta[tool]["timeout_sec"] >= 45 for tool in ("click", "key", "type_text", "activate_app")), "a switch plus an input fits the budget"


# --- Linux X11 ----------------------------------------------------------------------------------

WMCTRL = ("0x00000001  0 4242 10 20 800 600 navigator.Firefox host Browser - Docs\n"
          "0x00000002  0 5151 0 0 1024 768 keynote.Keynote host Deck\n"
          "0x00000003  0 5151 0 0 300 200 keynote.Keynote host Inspector\n")
TITLES = {"1": "Browser - Docs", "2": "Deck", "3": "Inspector"}
PIDS = {"1": "4242", "2": "5151", "3": "5151"}


def x11(tmp_path, monkeypatch, *, active="1", xdotool=True, switch_to=None, wmctrl=True):
    """A faked X11 desktop: ``xdotool getactivewindow`` prints the active window id (decimal), wmctrl the rows."""
    module = load_plugin()
    # This fixture models X11 on hosts that need not have libX11 installed.
    monkeypatch.setattr(module._ComputerUse, "_x11_keysym", staticmethod(lambda _name: None))
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    tools = ({"wmctrl"} if wmctrl else set()) | ({"xdotool"} if xdotool else set())
    monkeypatch.setattr(module, "_which", lambda name: f"/usr/bin/{name}" if name in tools else "")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    state = {"active": active, "calls": []}

    def run(cmd, *a, **k):
        argv = [str(part) for part in cmd]
        state["calls"].append(argv)
        if argv == ["wmctrl", "-lpxG"]:
            return subprocess.CompletedProcess(argv, 0, WMCTRL, "")
        if argv[:3] == ["wmctrl", "-i", "-a"]:
            if switch_to is not None:
                state["active"] = switch_to
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv == ["xdotool", "getactivewindow"]:
            return subprocess.CompletedProcess(argv, 0 if state["active"] else 1, (state["active"] + "\n") if state["active"] else "", "")
        if argv[:2] == ["xdotool", "getwindowpid"]:
            return subprocess.CompletedProcess(argv, 0, PIDS.get(argv[2], "") + "\n", "")
        if argv[:2] == ["xdotool", "getwindowname"]:
            return subprocess.CompletedProcess(argv, 0, TITLES.get(argv[2], "") + "\n", "")
        assert argv[:2] != ["xdotool", "getwindowclassname"], "the reader must not depend on a command old xdotool builds lack"
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    api = API(tmp_path / "state")
    module.register(api)
    return module, api, state


def test_x11_targeted_key_activates_through_wmctrl_and_verifies_with_xdotool(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch, switch_to="2")
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["observed_effect"] is True, out
    assert ["wmctrl", "-i", "-a", "0x00000002"] in state["calls"]
    assert [argv for argv in state["calls"] if argv[:2] == ["xdotool", "key"]] == [["xdotool", "key", "Right"]]
    assert out["target"]["activated"] is True and out["target"]["verified"] is True
    assert out["delivered_to"] == {"window_id": "0x00000002", "pid": 5151, "app": "keynote.Keynote", "title": "Deck"}
    assert out["front_before"]["pid"] == 4242
    module, api, state = x11(tmp_path, monkeypatch)  # Firefox is already in front: no switch, one key
    already = receipt(api.tools["key"](keys="right", app="Firefox"))
    assert already["ok"] is True and already["target"]["activated"] is False and already["target"]["already_front"] is True
    assert not any(argv[:3] == ["wmctrl", "-i", "-a"] for argv in state["calls"])


def test_x11_refuses_when_the_switch_cannot_be_verified_or_matched(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch)  # wmctrl runs, xdotool keeps answering Firefox
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "not_frontmost_after_activate" and "observed_effect" not in out
    assert not any(argv[:2] == ["xdotool", "key"] for argv in state["calls"])
    missing = receipt(api.tools["key"](keys="right", app="Emacs"))
    assert missing["ok"] is False and missing["reason"] == "not_running" and "keynote.Keynote" in missing["running"]


def test_x11_sends_nothing_when_the_front_cannot_be_read_after_the_switch(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch, switch_to="")  # the switch ran; xdotool then names no window
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "front_unread" and "observed_effect" not in out, out
    assert not any(argv[:2] == ["xdotool", "key"] for argv in state["calls"]) and "delivered_to" not in out
    assert out["target"]["activated"] is None and out["target"]["verified"] is False
    unread = receipt(api.tools["activate_app"](app="Keynote"))
    assert unread["ok"] is False and unread["reason"] == "front_unread" and "front" not in unread and "observed_effect" not in unread


def test_x11_matches_a_class_or_title_exactly_and_never_a_substring(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="right", app="Docs"))  # only Firefox's TITLE contains it
    assert out["ok"] is False and out["reason"] == "not_running" and "observed_effect" not in out
    assert out["candidates"][0]["app"] == "navigator.Firefox" and "exact" in out["error"]
    assert not any(argv[:2] == ["xdotool", "key"] for argv in state["calls"])
    by_title = receipt(api.tools["key"](keys="right", app="Browser - Docs"))
    assert by_title["ok"] is True and by_title["target"]["resolved"]["matched_by"] == "title"
    # Two windows of one application are one target by class; two windows named by title are distinct.
    module, api, state = x11(tmp_path, monkeypatch, switch_to="3")
    inspector = receipt(api.tools["key"](keys="right", app="Inspector"))
    assert inspector["ok"] is True and inspector["delivered_to"]["window_id"] == "0x00000003"


def test_x11_without_xdotool_refuses_a_target_and_lists_windows_without_a_front(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch, xdotool=False)
    out = receipt(api.tools["type_text"](text="hi", app="Keynote"))
    assert out["ok"] is False and "backend" in out["error"], "typing on X11 needs xdotool: an honest backend error"
    switch = receipt(api.tools["activate_app"](app="Keynote"))
    assert switch["ok"] is False and switch["reason"] == "unsupported" and "xdotool" in switch["error"]
    rows = receipt(api.tools["window_list"]())
    assert rows["ok"] is True and rows["frontmost"] is None and "xdotool" in rows["front_unread"]
    assert all("frontmost" not in r for r in rows["windows"]), "an unread front is not rendered as false"
    caps = receipt(api.tools["capabilities"]())
    assert caps["targeting"]["app"] == "unsupported" and "xdotool" in caps["targeting"]["note"]


def test_x11_window_list_rows_and_wayland_refusal(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch)
    out = receipt(api.tools["window_list"]())
    assert out["ok"] is True and out["frontmost"]["pid"] == 4242 and out["frontmost"]["app"] == "navigator.Firefox"
    assert [(r["app"], r["frontmost"]) for r in out["windows"]] == [("navigator.Firefox", True), ("keynote.Keynote", False), ("keynote.Keynote", False)]
    assert receipt(api.tools["capabilities"]())["targeting"]["app"] == "supported"
    monkeypatch.setattr(module, "_session_type", lambda: "wayland")
    monkeypatch.setattr(module, "_which", lambda name: "")
    refusal = receipt(api.tools["window_list"]())
    assert refusal["ok"] is False and refusal["reason"] == "unsupported" and "Wayland" in refusal["error"]
    assert receipt(api.tools["key"](keys="right", app="Keynote"))["ok"] is False


# --- the native helper's pure parts (no desktop needed) ----------------------------------------

def test_native_parsers_and_resolution():
    from skills.unix_computer_use import macos_native as native

    info = native.parse_info('"pid"=406\n"CFBundleIdentifier"="com.apple.loginwindow"\n"LSDisplayName"="loginwindow"\n')
    assert info == {"pid": 406, "app_id": "com.apple.loginwindow", "app": "loginwindow"}
    one_line = native.parse_info('"pid"=100 "CFBundleIdentifier"="com.google.Chrome" "LSDisplayName"="Google Chrome" ')
    assert one_line == {"pid": 100, "app_id": "com.google.Chrome", "app": "Google Chrome"}, "spaces between fields, names with spaces"
    listing = (' 1) "loginwindow" ASN:0x0-0x3003: \n    bundleID="com.apple.loginwindow"\n    bundle path="/System/Library/CoreServices/loginwindow.app"\n'
               '    pid = 406 type="UIElement" flavor=3\n'
               ' 2) "Терминал" ASN:0x0-0x5005: \n    bundleID="com.apple.Terminal"\n    bundle path="/System/Applications/Utilities/Terminal.app"\n'
               '    pid = 878 type="Foreground" flavor=3\n'
               ' 3) "Keynote" ASN:0x0-0x6006: \n    bundleID="com.apple.iWork.Keynote"\n    bundle path="/Applications/Keynote.app"\n'
               '    pid = 900 type="Foreground" flavor=3\n')
    apps = native.parse_list(listing)
    assert [a["app"] for a in apps] == ["Терминал", "Keynote"], "background-only processes are not applications"
    assert native.resolve("keynote", apps)[0]["pid"] == 900
    assert native.resolve("COM.APPLE.TERMINAL", apps)[0]["pid"] == 878
    assert native.resolve("Terminal", apps)[0]["pid"] == 878, "the executable's bundle name matches a localized display name"
    assert native.resolve("pid:900", apps)[0]["app"] == "Keynote"
    assert native.resolve("pid:1", apps) == [] and native.resolve("", apps) == []
    assert native.resolve("Key", apps) == [], "no prefix guessing: an unknown name resolves to nothing"
    assert native.resolve("900", apps) == [], "bare digits are a name, not a pid (pid:<n> is the form)"


def test_native_walk_is_bounded_and_says_which_bound_cut_it():
    from skills.unix_computer_use import macos_native as native

    graph = {"root": ["a", "b", "web"], "a": [f"a{i}" for i in range(50)], "b": [], "web": ["w1", "w2"]}

    def read(el):
        role = "AXWebArea" if el == "web" else ("AXWindow" if el == "root" else "AXButton")
        return 0, [role, None, el, None, None, (10, 20), (30, 40)]

    def children(el, n):
        kids = list(graph.get(el, []))
        return 0, kids[:n], len(kids)

    clock = iter([0.0] * 1000)
    small = native.walk("root", read, children, max_nodes=5, deadline_s=10, now=lambda: next(clock))
    assert small["visited"] == 5 and small["truncated"] is True and small["truncated_by"] == ["max_nodes", "web_content_skipped"]
    whole = native.walk("root", read, children, max_nodes=100, deadline_s=10, now=lambda: next(clock))
    assert whole["visited"] == 54 and whole["truncated_by"] == ["web_content_skipped"], "web content is skipped and named"
    with_web = native.walk("root", read, children, max_nodes=100, deadline_s=10, skip_children=(), now=lambda: next(clock))
    assert with_web["visited"] == 56 and with_web["truncated"] is False
    ticks = iter([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    late = native.walk("root", read, children, max_nodes=100, deadline_s=1, now=lambda: next(ticks, 5.0))
    assert late["truncated_by"][0] == "timeout" and late["visited"] >= 1
    marks = native.marks_from(whole["nodes"])
    assert marks and marks[0] == {"id": 1, "role": "Button", "title": "a", "center_x": 25, "center_y": 40, "x": 10, "y": 20, "w": 30, "h": 40}
    assert all(m["role"] != "Window" for m in marks) and len(marks) == 52

    def busy(el):
        return (-25204, []) if el == "a" else read(el)
    partial = native.walk("root", busy, children, max_nodes=100, deadline_s=10, now=lambda: next(clock))
    assert partial["errors"] == {"CannotComplete": 1} and partial["visited"] == 3, "a busy node is counted, not walked"
    assert partial["truncated_by"] == ["read_errors", "web_content_skipped"], "its unrequested subtree is a named cut"


def test_native_walk_names_a_wide_node_it_could_not_finish_and_releases_every_element_once():
    """A node wider than the node budget is cut by max_nodes and says how many children were not read (no
    hidden per-node cap); every fetched element is released exactly once, even when a read raises."""
    from skills.unix_computer_use import macos_native as native

    wide = {"root": ["list"], "list": [f"row{i}" for i in range(200)]}

    def read(el):
        return 0, ["AXRow" if el.startswith("row") else "AXGroup", None, el, None, None, (0, 0), (10, 10)]

    requested = []

    def children(el, n):
        requested.append((el, n))
        kids = list(wide.get(el, []))
        return 0, kids[:n], len(kids)

    released = []
    out = native.walk("root", read, children, max_nodes=50, deadline_s=10, release=released.append)
    assert out["visited"] == 50 and out["truncated_by"] == ["max_nodes"]
    assert requested[:2] == [("root", 50), ("list", 49)], "children are requested up to the remaining budget plus one"
    assert out["nodes"][1]["children_unread"] == 151
    assert sorted(released) == sorted(["root", "list"] + [f"row{i}" for i in range(49)]) and len(released) == len(set(released))

    def raising(el):
        if el == "list":
            raise RuntimeError("provider went away")
        return read(el)
    released.clear()
    with pytest.raises(RuntimeError):
        native.walk("root", raising, children, max_nodes=50, deadline_s=10, release=released.append)
    assert released == ["root", "list"], "the root and the element being read are released once each"
    assert native._bounds((float("nan"), 1.0), (2.0, 3.0)) is None and native._bounds((1.0, 2.0), (3.5, 4.5)) == [1, 2, 3, 4]


# --- cycle 2: window identity, stage facts, generated programs ---------------------------------

def test_a_sequence_whose_first_combo_is_invalid_keeps_the_switch_mark(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="bogus+right down", app="Keynote"))
    assert out["ok"] is False and "bogus" in out["error"] and out["steps_done"] == 0 and inputs(state) == []
    assert "observed_effect" not in out and not any("activate" in str(c) for c in state["calls"]), "validation precedes activation"
    assert out["dispatched"] is False


def test_a_failed_paste_helper_still_reports_the_clipboard_write(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(1, "", "boom"))
    out = receipt(api.tools["type_text"](text="secret", via="paste", app="Keynote"))
    assert out["ok"] is False and out["clipboard_replaced"] is True and out["method"] == "paste" and state["clipboard"] == "secret"
    assert out["observed_effect"] is True and out["activation_done"] is True and out["clipboard_done"] is True and "unknown" in out["note"]
    # Without a switch the completed clipboard write is still an observed effect of its own.
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(1, "", "boom"))
    out = receipt(api.tools["type_text"](text="secret", via="paste"))
    assert out["ok"] is False and out["observed_effect"] is True and out["clipboard_done"] is True and "activation_done" not in out
    warning = "WARNING: Accessibility privileges not enabled. Many actions may fail.\n"
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(0, "", warning))
    dropped = receipt(api.tools["type_text"](text="secret", via="paste"))
    assert dropped["reason"] == "permission_denied" and dropped["clipboard_done"] is True and dropped["observed_effect"] is True
    assert "clipboard write" in dropped["note"]


def test_a_failed_clipboard_write_sends_nothing_and_says_so(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    original = module.subprocess.run

    def broken_pbcopy(cmd, *a, **k):
        if [str(p) for p in cmd][0] == "pbcopy":
            raise module.subprocess.TimeoutExpired(cmd, 10)
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", broken_pbcopy)
    out = receipt(api.tools["type_text"](text="x", via="paste", app="Keynote"))
    assert out["ok"] is False and "TimeoutExpired" in out["error"] and inputs(state) == [], out
    assert out["note"].startswith("the input was not sent") and out["activation_done"] is True and out["reason"] == "clipboard_failed"
    assert out["dispatched"] is False and "front_at_dispatch" not in out and out["front"]["app"] == "Keynote"


def test_a_demoted_step_inside_a_sequence_is_reported_as_dispatched_with_its_facts(tmp_path, monkeypatch):
    warning = "WARNING: Accessibility privileges not enabled. Many actions may fail.\n"
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(0, "", warning))
    out = receipt(api.tools["key"](keys="down return"))
    assert out["ok"] is False and out["reason"] == "permission_denied" and out["steps_done"] == 0
    assert "failed" in out["error"] and "not sent" not in out["error"] and out["permission"]["accessibility"] == "untrusted"
    assert out["dispatched"] is True and "dropped" in out["note"], "the demoted step keeps its own explanation"
    # After completed steps the explanation is said about the next step, beside what completed.
    module, api, state = mac(tmp_path, monkeypatch)
    calls = {"n": 0}
    original = module.subprocess.run

    def second_dropped(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if argv[0] == "cliclick":
            calls["n"] += 1
            if calls["n"] == 2:
                return subprocess.CompletedProcess(argv, 0, "", warning)
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", second_dropped)
    out = receipt(api.tools["key"](keys="down return up"))
    assert out["steps_done"] == 1 and out["reason"] == "permission_denied" and out["observed_effect"] is True and calls["n"] == 2, out
    assert out["note"].startswith("1 of 3 step(s) completed") and "dropped" in out["note"] and "the rest were not sent" in out["note"], out["note"]


def test_a_failed_helper_with_an_unread_front_is_still_a_dispatched_input(tmp_path, monkeypatch):
    """The reader going silent is not proof that nothing was sent: the helper ran, its outcome is unknown."""
    module, api, state = mac(tmp_path, monkeypatch, cliclick=(1, "", "boom"), lsappinfo=False)
    out = receipt(api.tools["key"](keys="right down"))
    assert out["ok"] is False and out["steps_done"] == 0 and out["dispatched"] is True
    assert "failed" in out["error"] and "unknown" in out["note"] and "not sent" not in out["note"]
    module, api, state = mac(tmp_path, monkeypatch, lsappinfo=False)
    calls = {"n": 0}
    original = module.subprocess.run

    def second_fails(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if argv[0] == "cliclick":
            calls["n"] += 1
            if calls["n"] == 2:
                return subprocess.CompletedProcess(argv, 1, "", "boom")
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", second_fails)
    out = receipt(api.tools["key"](keys="right down return"))
    assert out["steps_done"] == 1 and out["dispatched"] is True and calls["n"] == 2, "the second helper ran and failed; the third was not sent"
    assert "dispatched and did not report success" in out["note"] and "the rest were not sent" in out["note"]
    # A dispatched step whose front could not be read is listed as an unknown recipient, not left out.
    module, api, state = mac(tmp_path, monkeypatch)
    seen = {"inputs": 0}
    original = module.subprocess.run

    def front_unread_after_first(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if argv[0] == "cliclick":
            seen["inputs"] += 1
            if seen["inputs"] == 2:
                return subprocess.CompletedProcess(argv, 1, "", "boom")
        if argv[0] == "lsappinfo" and seen["inputs"] >= 1:
            return subprocess.CompletedProcess(argv, 1, "", "")
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", front_unread_after_first)
    out = receipt(api.tools["key"](keys="right down return"))
    assert out["steps_done"] == 1 and out["dispatched"] is True and seen["inputs"] == 2, out
    assert out["steps_delivered_to"][0]["app"] == "Safari" and out["steps_delivered_to"][1] == "unknown", out


def test_cliclick_gets_absolute_negative_coordinates_for_a_display_left_of_the_main_one(tmp_path, monkeypatch):
    def helper(args, state):
        if args[0] == "displays":
            return {"ok": True, "displays": [{"id": 1, "main": True, "x": 0, "y": 0, "width": 1728, "height": 1117, "index": 1},
                                             {"id": 2, "main": False, "x": -1920, "y": -200, "width": 1920, "height": 1080, "index": 2}]}
        if args[0] == "resolve":
            return {"ok": True, "candidates": [app_dict(KEYNOTE)]}
        if args[0] == "tree":
            return {"ok": True, "nodes": [{"id": 1, "depth": 0, "role": "Window", "x": -1800, "y": -100, "w": 800, "h": 600}]}
        return {"ok": False, "error": "no fake"}
    module, api, state = mac(tmp_path, monkeypatch, helper=helper)
    monkeypatch.setattr(module, "_png_dimensions", lambda _p: (3840, 2160))
    shot = receipt(api.tools["screenshot"](job_id="left", app="Keynote"))
    assert shot["display"]["index"] == 2 and (shot["coord_transform"]["ox"], shot["coord_transform"]["oy"]) == (-1920, -200)
    click = receipt(api.tools["click"](x=100, y=50, observation_id=shot["observation_id"]))
    assert click["ok"] is True and inputs(state)[-1] == ["cliclick", "c:=-1870,=-175"], "a bare '-' would be a relative move"
    assert receipt(api.tools["click"](x=-10, y=20, raw=True))["ok"] is True and inputs(state)[-1] == ["cliclick", "c:=-10,20"]
    assert receipt(api.tools["move"](x=5, y=5, raw=True))["ok"] is True and inputs(state)[-1] == ["cliclick", "m:5,5"]


LIBRE = ("0x00000001  0 4242 10 20 800 600 navigator.Firefox host Browser - Docs\n"
         "0x00000004  0 7000 0 0 1024 768 libreoffice.libreoffice-writer host Letter - LibreOffice Writer\n"
         "0x00000005  0 7000 0 0 1024 768 libreoffice.libreoffice-calc host Sheet - LibreOffice Calc\n"
         "0x00000006  0 0 0 0 640 480 xterm.XTerm host xterm\n"
         "0x00000007  0 0 0 0 640 480 OldJava.OldJava host Old Java App\n"
         "0x00000008  0 8000 0 0 640 480 gimp-2.10.Gimp-2.10 host GNU Image Manipulation Program\n"
         "0x00000009  0 9000 0 0 640 480 org.gnome.Nautilus.Org.gnome.Nautilus host Home\n")


def test_x11_a_class_is_its_own_windows_not_every_window_of_its_process(tmp_path, monkeypatch):
    """LibreOffice Writer and Calc share one process: naming Writer with Calc in front is a switch, never 'already front'."""
    monkeypatch.setattr(sys.modules[__name__], "WMCTRL", LIBRE)
    monkeypatch.setattr(sys.modules[__name__], "PIDS", {**PIDS, "4": "7000", "5": "7000", "6": "0", "7": "0"})
    monkeypatch.setattr(sys.modules[__name__], "TITLES", {**TITLES, "4": "Letter - LibreOffice Writer", "5": "Sheet - LibreOffice Calc", "6": "xterm", "7": "Old Java App"})
    module, api, state = x11(tmp_path, monkeypatch, active="5", switch_to="4")
    out = receipt(api.tools["key"](keys="ctrl+v", app="libreoffice-writer"))
    assert out["ok"] is True and ["wmctrl", "-i", "-a", "0x00000004"] in state["calls"], out
    assert out["target"]["activated"] is True and out["delivered_to"]["window_id"] == "0x00000004"
    rows = receipt(api.tools["window_list"]())
    assert [(r["app"], r["frontmost"]) for r in rows["windows"] if r["pid"] == 7000] == [("libreoffice.libreoffice-writer", True), ("libreoffice.libreoffice-calc", False)]
    module, api, state = x11(tmp_path, monkeypatch, active="5")  # the switch does not happen
    stuck = receipt(api.tools["key"](keys="ctrl+v", app="libreoffice-writer"))
    assert stuck["ok"] is False and stuck["reason"] == "not_frontmost_after_activate" and not any(argv[:2] == ["xdotool", "key"] for argv in state["calls"])
    assert receipt(api.tools["activate_app"](app="libreoffice-writer"))["ok"] is False, "Calc in front is not 'Writer already in front'"
    # pid: names any window of the process; a window the X server has no pid for is itself.
    module, api, state = x11(tmp_path, monkeypatch, active="5")
    assert receipt(api.tools["key"](keys="ctrl+v", app="pid:7000"))["target"]["already_front"] is True
    module, api, state = x11(tmp_path, monkeypatch, active="6", switch_to="7")
    old = receipt(api.tools["key"](keys="return", app="OldJava"))
    assert old["ok"] is True and old["target"]["activated"] is True and ["wmctrl", "-i", "-a", "0x00000007"] in state["calls"]
    module, api, state = x11(tmp_path, monkeypatch, active="6")
    assert receipt(api.tools["key"](keys="return", app="OldJava"))["reason"] == "not_frontmost_after_activate", "pid 0 == pid 0 is not 'the same window'"


def test_x11_dotted_classes_are_named_whole_or_by_either_part(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "WMCTRL", LIBRE)
    module, api, state = x11(tmp_path, monkeypatch, active="8")
    assert receipt(api.tools["key"](keys="right", app="Gimp-2.10"))["ok"] is True
    assert receipt(api.tools["key"](keys="right", app="gimp-2.10.Gimp-2.10"))["ok"] is True
    module, api, state = x11(tmp_path, monkeypatch, active="9")
    assert receipt(api.tools["key"](keys="right", app="org.gnome.Nautilus"))["ok"] is True
    assert receipt(api.tools["key"](keys="right", app="gnome"))["reason"] == "not_running", "a fragment between dots is not a name"


def focus_inspector_after_a_key(module, state, monkeypatch):
    """The first key opens the Inspector window of the same process (a sibling window comes forward)."""
    original = module.subprocess.run

    def run(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if argv[:2] == ["xdotool", "key"]:
            state["active"] = "3"
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", run)


def test_x11_a_window_named_by_title_stops_a_sequence_when_its_sibling_comes_forward(tmp_path, monkeypatch):
    module, api, state = x11(tmp_path, monkeypatch, active="2")
    focus_inspector_after_a_key(module, state, monkeypatch)
    out = receipt(api.tools["key"](keys="right down", app="Deck"))
    assert out["ok"] is False and out["reason"] == "target_lost" and out["steps_done"] == 1, out
    assert "is not one of the target's windows" in out["error"] and "0x00000003" in out["error"] and "0x00000002" in out["error"], out["error"]
    assert [argv for argv in state["calls"] if argv[:2] == ["xdotool", "key"]] == [["xdotool", "key", "Right"]]
    assert out["front_after"]["window_id"] == "0x00000003" and out["delivered_to"]["window_id"] == "0x00000002"
    # The same two keys named by class: any window of Keynote is Keynote, both are sent, both recipients named.
    module, api, state = x11(tmp_path, monkeypatch, active="2")
    focus_inspector_after_a_key(module, state, monkeypatch)
    both = receipt(api.tools["key"](keys="right down", app="Keynote"))
    assert both["ok"] is True and [r["window_id"] for r in both["steps_delivered_to"]] == ["0x00000002", "0x00000003"]
    assert both["front_after"]["window_id"] == "0x00000003", "a sibling window in front is a different place"
    module, api, state = x11(tmp_path, monkeypatch, active="2")
    focus_inspector_after_a_key(module, state, monkeypatch)
    one = receipt(api.tools["key"](keys="right"))
    assert one["delivered_to"]["window_id"] == "0x00000002" and one["front_after"]["window_id"] == "0x00000003"


def test_an_untargeted_sequence_counts_an_unknown_recipient(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    original = module.subprocess.run
    calls = {"front": 0}

    def flaky_lsappinfo(cmd, *a, **k):
        argv = [str(p) for p in cmd]
        if argv[:2] == ["lsappinfo", "front"]:
            calls["front"] += 1
            if calls["front"] >= 3:  # the reader goes silent before the second step
                return subprocess.CompletedProcess(argv, 1, "", "")
        return original(cmd, *a, **k)
    monkeypatch.setattr(module.subprocess, "run", flaky_lsappinfo)
    out = receipt(api.tools["key"](keys="down return"))
    assert out["ok"] is True and out["steps_delivered_to"][0]["app"] == "Safari" and out["steps_delivered_to"][1] == "unknown", out


def test_native_reads_say_which_attributes_failed_and_never_render_a_failed_read_as_false():
    from skills.unix_computer_use import macos_native as native

    vals, failed = native.split_attr_errors(["AXTitle", "AXMinimized", "AXMain"], ["Deck", ("axerror", -25204), ("axerror", -25212)])
    assert vals == ["Deck", None, None] and failed == {"AXMinimized": "CannotComplete"}, "ordinary absence is not a failure"
    busy = [("axerror", -25204)] * len(native._WALK_ATTRS)

    def read(el):
        return (0, busy) if el == "b" else (0, ["AXButton", None, el, None, None, (1, 1), (2, 2)])

    def children(el, n):
        kids = {"root": ["a", "b"]}.get(el, [])
        return 0, kids[:n], len(kids)
    out = native.walk("root", read, children, max_nodes=10, deadline_s=5)
    assert out["visited"] == 2 and out["errors"] == {name + ":CannotComplete": 1 for name in native._WALK_ATTRS}
    assert out["truncated"] is True and out["truncated_by"] == ["read_errors"], "a subtree never requested is a named cut"
    # A node wider than the budget whose fetched children all fail is still a cut, named.
    wide = {"root": [f"row{i}" for i in range(100)]}

    def read_failing_rows(el):
        return (0, busy) if el.startswith("row") else (0, ["AXGroup", None, el, None, None, (0, 0), (9, 9)])

    def children_wide(el, n):
        kids = wide.get(el, [])
        return 0, kids[:n], len(kids)
    cut = native.walk("root", read_failing_rows, children_wide, max_nodes=5, deadline_s=5)
    assert cut["visited"] == 1 and cut["nodes"][0]["children_unread"] == 95 and cut["truncated_by"] == ["max_nodes", "read_errors"]
    # A children read the app refused is a read error, named as such, never a budget cut.

    def children_refused(el, n):
        return (-25204, [], 10) if el == "root" else (0, [], 0)
    refused = native.walk("root", read_failing_rows, children_refused, max_nodes=150, deadline_s=5)
    assert refused["truncated_by"] == ["read_errors"] and refused["nodes"][0]["children_unread"] == 10 and refused["errors"] == {"children:CannotComplete": 1}
    # Fewer children than the count within the budget (the read disagreed with the count): a read error, not a budget cut.

    def children_race(el, n):
        return (0, ["c1", "c2", "c3"], 10) if el == "root" else (0, [], 0)
    race = native.walk("root", read_failing_rows, children_race, max_nodes=150, deadline_s=5)
    assert race["visited"] == 4 and race["nodes"][0]["children_unread"] == 7 and race["truncated_by"] == ["read_errors"] and race["errors"] == {}
    # A count with no value answered for the children: named too.

    def children_no_value(el, n):
        return (-25212, [], 4) if el == "root" else (0, [], 0)
    no_value = native.walk("root", read_failing_rows, children_no_value, max_nodes=150, deadline_s=5)
    assert no_value["nodes"][0]["children_unread"] == 4 and no_value["truncated_by"] == ["read_errors"] and no_value["errors"] == {}


def test_native_activation_rules_without_a_desktop(monkeypatch):
    from skills.unix_computer_use import macos_native as native

    monkeypatch.setattr(native, "op_resolve", lambda q: {"ok": True, "candidates": [app_dict(KEYNOTE)]})
    monkeypatch.setattr(native, "_AX", lambda: None)
    reads = iter([None, app_dict(KEYNOTE)])
    monkeypatch.setattr(native, "front_app", lambda: next(reads, None))
    monkeypatch.setattr(native, "_activate_pid", lambda ax, pid, wait_ms: ("ax_frontmost", app_dict(KEYNOTE), 40))
    out = native.op_activate("Keynote", 500)
    assert out["ok"] is True and out["activated"] is None and out["verified"] is True, "unknown before: a verified front, not a claimed switch"
    monkeypatch.setattr(native, "front_app", lambda: app_dict(SAFARI))
    monkeypatch.setattr(native, "_activate_pid", lambda ax, pid, wait_ms: ("system_events", None, 2000))
    unread = native.op_activate("Keynote", 500)
    assert unread["ok"] is False and unread["reason"] == "front_unread" and unread["activated"] is None
