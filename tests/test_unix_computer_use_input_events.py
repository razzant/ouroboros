"""unix_computer_use: pointer, wheel, keys and text on a Mac as CoreGraphics events (#63), locally and over SSH.

``skills/unix_computer_use/input_events.js`` posts input, so here it runs byte for byte under Node against inert
stand-ins of the bridge it uses (``tests/fixtures/input_events_inert.cjs``: no ObjC, no CoreGraphics, nothing posted,
the pointer never moves) and is compiled without running. ``mac_input.py`` builds its argv, names the keys and
translates the remote pyautogui snippets; those are checked as plain functions. No real event is sent anywhere.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skills.unix_computer_use import mac_input
from tests.group_run import run_in_group

PROGRAM = Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use" / "input_events.js"
INERT = Path(__file__).resolve().parent / "fixtures" / "input_events_inert.cjs"
TAG = 0x4F55524F
SHIFT, COMMAND = 0x20000, 0x100000
ABC = {"id": "com.apple.keylayout.ABC", "planes": {0: {0: "a", 4: "h", 34: "i", 49: " ", 1: "s"}, 2: {0: "A", 4: "H"}}}
RUSSIAN = {"id": "com.apple.keylayout.Russian", "planes": {0: {4: "р", 34: "ш"}, 2: {4: "Р"}}}


def runs(*scenarios: dict) -> list[dict]:
    node = shutil.which("node")
    assert node, "Node runs the input program against inert stand-ins"
    done = run_in_group([node, str(INERT), str(PROGRAM), json.dumps(scenarios)], timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


def spec(*ops: dict, budget_ms: int = 8000) -> dict:
    return {"ops": list(ops), "budget_ms": budget_ms}


def clean(run: dict) -> None:
    """A finished call holds nothing, released every object it made and failed with nothing."""
    assert run["error"] is None and run["held"] == [] and run["leaked"] == [], run


def test_clicks_carry_their_button_count_mark_and_point_as_two_doubles():
    middle, right = runs({"spec": spec({"op": "click", "x": 10, "y": -20, "button": "middle", "count": 2})},
                         {"spec": spec({"op": "click", "x": 3, "y": 4, "button": "right", "count": 1})})
    for run in (middle, right):
        clean(run)
        assert run["mouse_signature"] == ["void*", ["void*", "unsigned int", "double", "double", "unsigned int"]]
        assert all(row["tap"] == 0 and row["tag"] == TAG for row in run["posted"]), "HID tap, the skill's mark on each"
        assert run["source_state"] == 1 and json.loads(run["returned"]) == {"posted": len(run["posted"])}
    # A move to the point first, then down/up pairs whose click state counts 1, 2 (AppKit's double click).
    assert [(r["mouse"], r["at"], r["button"], r.get("clicks")) for r in middle["posted"]] == [
        (5, [10, -20], 2, None), (25, [10, -20], 2, 1), (26, [10, -20], 2, 1), (25, [10, -20], 2, 2), (26, [10, -20], 2, 2)]
    assert [r["mouse"] for r in right["posted"]] == [5, 3, 4] and right["posted"][1]["button"] == 1


def test_a_drag_travels_and_a_move_under_a_held_button_is_a_drag():
    drag, held_move, plain_move = runs({"spec": spec({"op": "drag", "from": [0, 0], "to": [120, 60], "button": "left"})},
                                       {"spec": spec({"op": "move", "x": 5, "y": 6}), "held_buttons": [0]},
                                       {"spec": spec({"op": "move", "x": 5, "y": 6})})
    for run in (drag, held_move, plain_move):
        clean(run)
    kinds = [r["mouse"] for r in drag["posted"]]
    assert kinds == [5, 1] + [6] * 12 + [2] and drag["posted"][-1]["at"] == [120, 60] and drag["posted"][2]["at"] == [10, 5]
    assert [r["mouse"] for r in held_move["posted"]] == [6] and [r["mouse"] for r in plain_move["posted"]] == [5]


def test_a_button_down_stays_down_and_a_release_without_a_point_is_at_the_pointer():
    down, up = runs({"spec": spec({"op": "button", "down": True, "button": "right", "x": 1, "y": 2})},
                    {"spec": spec({"op": "button", "down": False, "button": "left"}), "pointer": [100, 1000], "screen_height": 1117})
    assert down["error"] is None and down["held"] == ["b1"] and [r["mouse"] for r in down["posted"]] == [3], "asked to stay down"
    clean(up)
    assert up["posted"] == [{"tap": 0, "tag": TAG, "mouse": 2, "at": [100, 117], "button": 0, "clicks": 1}], "AppKit's y flipped"


def test_text_goes_one_grapheme_per_event_pair_in_any_layout():
    text = "Hi 👋🏽 Привет\né" + "\u0301" * 25
    abc, russian = runs({"spec": spec({"op": "type", "text": text, "interval_ms": 7}), "layout": ABC},
                        {"spec": spec({"op": "type", "text": "Hi"}), "layout": RUSSIAN})
    clean(abc)
    clean(russian)
    downs = [r for r in abc["posted"] if r["down"]]
    assert [r["unicode"] for r in downs][:4] == ["H", "i", " ", "👋🏽"], "a surrogate pair and its modifier in one event"
    assert (downs[0]["key"], downs[0]["flags"]) == (4, SHIFT) and (downs[1]["key"], downs[1]["flags"]) == (34, 0)
    assert downs[3]["key"] == 0x5A and "\r" in [r["unicode"] for r in downs] and any(r["key"] == 36 for r in downs)
    # A grapheme over 20 UTF-16 units (é with 25 combining accents) goes as its code points.
    assert [r["unicode"] for r in downs[-26:]] == ["é"] + ["\u0301"] * 25
    answer = json.loads(abc["returned"])
    assert answer["method"] == "unicode_events" and answer["chars_sent"] == answer["chars_total"] == len(downs)
    assert answer["override_only"] == sum(1 for r in downs if r["key"] == 0x5A), "characters without a key of their own"
    assert all(r["flags"] in (0, SHIFT) for r in abc["posted"]), "the Option the keyboard holds never rides along"
    assert 0.007 in abc["delays"] and all(r["tap"] == 1 and r["tag"] == TAG for r in abc["posted"])
    # Latin under a Russian layout: no key gives it, the string still goes (the former helper typed nothing here).
    assert [r["unicode"] for r in russian["posted"] if r["down"]] == ["H", "i"]
    assert json.loads(russian["returned"])["override_only"] == 2


def test_a_chord_finds_its_key_in_the_layout_or_refuses_before_posting():
    shifted, missing, named = runs({"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "A"}), "layout": ABC},
                                   {"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "c"}), "layout": RUSSIAN},
                                   {"spec": spec({"op": "keys", "mods": ["cmd", "alt"], "code": 36})})
    clean(shifted)
    assert [(r["key"], r["down"], r["flags"]) for r in shifted["posted"]] == [
        (59, True, 0x40000), (56, True, 0x60000), (0, True, 0x60000), (0, False, 0x60000), (56, False, 0x40000), (59, False, 0)]
    assert missing["posted"] == [] and "CGEventPost" not in missing["calls"] and missing["returned"] == ""
    assert missing["log"] == ["Unable to get key code for c (no key of com.apple.keylayout.Russian gives c alone or with "
                              "Shift); nothing was sent"], "the words the receipt composer reads"
    assert "CGPreflightPostEventAccess" not in missing["calls"], "found before the right to post is even read"
    clean(named)
    assert [r["key"] for r in named["posted"]] == [55, 58, 36, 36, 58, 55] and named["posted"][2]["flags"] == COMMAND | 0x80000


def test_without_the_right_to_post_nothing_is_posted_and_reading_the_pointer_needs_no_right():
    refused, position = runs({"spec": spec({"op": "scroll", "dy": -3}), "post": False},
                             {"spec": spec({"op": "position"}), "post": False, "pointer": [40, 1000], "screen_height": 1117})
    assert refused["posted"] == [] and refused["returned"] == "" and refused["leaked"] == []
    assert refused["log"] == ["WARNING: Accessibility privileges not enabled (no right to post input events); nothing was sent"]
    assert "CGEventSourceCreate" not in refused["calls"], "no event exists"
    clean(position)
    assert json.loads(position["returned"]) == {"x": 40, "y": 117} and position["posted"] == []
    assert "CGPreflightPostEventAccess" not in position["calls"]


def test_the_wheel_posts_one_line_per_step():
    down, sideways = runs({"spec": spec({"op": "scroll", "dy": -3})}, {"spec": spec({"op": "scroll", "dx": 2})})
    clean(down)
    clean(sideways)
    assert [(r["scroll"], r["units"], r["wheels"], r["tap"]) for r in down["posted"]] == [([-1, 0], 1, 2, 0)] * 3
    assert [r["scroll"] for r in sideways["posted"]] == [[0, 1], [0, 1]]


def test_a_hold_lets_up_what_it_pressed_on_error_and_at_its_own_deadline():
    ok, broken, late = runs({"spec": spec({"op": "hold", "mods": ["cmd"], "char": "a", "ms": 400}), "layout": ABC},
                            {"spec": spec({"op": "hold", "mods": ["cmd", "shift"], "code": 49, "ms": 400}), "fail": "post:3"},
                            {"spec": spec({"op": "hold", "mods": [], "code": 49, "ms": 400},
                                          {"op": "keys", "mods": [], "code": 36}, budget_ms=500), "tick_ms": 30})
    clean(ok)
    assert [(r["key"], r["down"]) for r in ok["posted"]] == [(55, True), (0, True), (0, False), (55, False)] and 0.4 in ok["delays"]
    # The key up failed: Shift, then Command, then the key are let up again (last pressed first) and the error stands.
    assert broken["error"] == "injected post failure" and broken["held"] == [] and broken["leaked"] == []
    assert [(r["key"], r["down"]) for r in broken["posted"]] == [(55, True), (56, True), (49, True), (49, False), (56, False), (55, False)]
    assert "own deadline" in late["error"] and late["held"] == [] and late["leaked"] == []
    assert not any(r["key"] == 36 for r in late["posted"]), "nothing after the deadline but the release"


def test_events_the_system_cannot_make_fail_the_call_with_nothing_sent():
    unmade, bad = runs({"spec": spec({"op": "click", "x": 1, "y": 1}, {"op": "keys", "mods": [], "code": 36}), "fail": "event:3"},
                       {"argv": []})
    assert unmade["posted"] == [] and "no key event" in unmade["error"] and unmade["leaked"] == []
    assert bad["posted"] == [] and "nothing was sent" in bad["error"]


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("osacompile"), reason="JavaScript for Automation is macOS")
def test_the_input_program_compiles_without_running(tmp_path):
    """osacompile parses and compiles the program into a file and runs nothing (running it would post input)."""
    done = subprocess.run(["osacompile", "-l", "JavaScript", "-o", str(tmp_path / "input.scpt"), str(PROGRAM)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and (tmp_path / "input.scpt").exists(), done.stderr


def test_the_argv_hands_one_ascii_spec_locally_and_the_source_over_ssh():
    local = mac_input.argv([{"op": "type", "text": "Привет"}], budget_ms=5000)
    assert local[:4] == ["osascript", "-l", "JavaScript", str(PROGRAM)] and local[-1].isascii()
    assert json.loads(local[-1]) == {"ops": [{"op": "type", "text": "Привет"}], "budget_ms": 5000}
    remote = mac_input.argv([{"op": "move", "x": 1, "y": 2}], remote=True)
    assert remote[3:5] == ["-e", PROGRAM.read_text(encoding="utf-8")] and len(remote) == 6
    assert mac_input.ops_of(local) == [{"op": "type", "text": "Привет"}] and mac_input.ops_of(remote) == [{"op": "move", "x": 1, "y": 2}]
    assert mac_input.ops_of(["osascript", "-l", "JavaScript", "paste_chord.js"]) is None and mac_input.ops_of(["xdotool"]) is None
    assert mac_input.method_of(mac_input.ops_of(local)) == "unicode_events" and mac_input.method_of([{"op": "move"}]) == "cg_events"
    assert mac_input.answer_fields('{"posted":4,"chars_sent":2,"chars_total":2,"override_only":1}\n') == {
        "chars_sent": 2, "chars_total": 2, "override_only": 1}
    assert mac_input.answer_fields("") == {} and mac_input.answer_fields('{"posted":3}') == {}


def test_the_remote_snippets_translate_to_the_same_operations():
    cases = {
        "pyautogui.click(10, -20, clicks=3, interval=0.1, button='middle')": [{"op": "click", "x": 10, "y": -20, "button": "middle", "count": 3}],
        "pyautogui.moveTo(1, 2); pyautogui.dragTo(3, 4, duration=0.5, button='left')": [{"op": "drag", "from": [1, 2], "to": [3, 4], "button": "left"}],
        "pyautogui.moveTo(5, 6)": [{"op": "move", "x": 5, "y": 6}],
        "pyautogui.mouseDown(x=1, y=2, button='right')": [{"op": "button", "down": True, "button": "right", "x": 1, "y": 2}],
        "pyautogui.mouseUp(button='middle')": [{"op": "button", "down": False, "button": "middle"}],
        "pyautogui.typewrite('it\\'s\\nok', interval=0.02)": [{"op": "type", "text": "it's\nok", "interval_ms": 20}],
        "pyautogui.hotkey('winleft', 'shift', 's')": [{"op": "keys", "mods": ["cmd", "shift"], "char": "s"}],
        "pyautogui.press('enter')": [{"op": "keys", "mods": [], "code": 36}],
        "pyautogui.scroll(-4)": [{"op": "scroll", "dy": -4}],
        "pyautogui.hscroll(2)": [{"op": "scroll", "dx": -2}],
        "print('pos', pyautogui.position())": [{"op": "position"}],
    }
    for code, expected in cases.items():
        assert mac_input.ops_for_pyautogui(code) == (expected, ""), code
    for code in ("pyautogui.write('x')", "import os", "pyautogui.press('no-such-key')"):
        ops, error = mac_input.ops_for_pyautogui(code)
        assert ops == [] and "unsupported pyautogui snippet" in error, code


def test_a_grant_refusal_is_one_structured_field():
    assert mac_input.permission_refusal("screen_recording") == {"refusal": {
        "kind": "permission", "grant": "screen_recording", "fixable_by": "owner",
        "where": "System Settings > Privacy & Security > Screen & System Audio Recording"}}
    assert set(mac_input.permission_refusal("accessibility")) == {"refusal"}
