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
ABC = {"id": "com.apple.keylayout.ABC", "planes": {0: {0: "a", 4: "h", 34: "i", 49: " ", 1: "s"}, 2: {0: "A", 4: "H"},
                                                  8: {0: "å", 1: "ß"}, 10: {0: "Å"}, 1: {5: "g", 9: "v"}}}
# US International: the apostrophe key is dead (it starts a composition), so it never stands for "'".
US_INTL = {"id": "com.apple.keylayout.USInternational-PC", "planes": {0: {39: "'", 0: "a"}, 2: {39: '"'}}, "dead": {0: [39], 2: [39]}}
NUMERIC_PAD, SECONDARY_FN, CTRL, OPTION = 0x200000, 0x800000, 0x40000, 0x80000
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
    # One pause per grapheme, after its key up: a real delay() costs several ms (the hosted runs of 21.11.2026).
    assert len(abc["delays"]) == answer["chars_total"], abc["delays"]
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
                              "Shift or Option); nothing was sent"], "the words the receipt composer reads"
    # The right to post is read first: without it the answer is the grant, not a layout hint (#REVP1 P8).
    denied, = runs({"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "c"}), "layout": RUSSIAN, "post": False})
    assert denied["posted"] == [] and denied["log"] == [
        "WARNING: Accessibility privileges not enabled (no right to post input events); nothing was sent"], denied
    clean(named)
    assert [r["key"] for r in named["posted"]] == [55, 58, 36, 36, 58, 55] and named["posted"][2]["flags"] == COMMAND | 0x80000


def test_a_key_keeps_its_class_flags_and_never_a_held_modifier():
    """A keyboard puts Fn and NumericPad on an arrow, Fn on the F-keys and the navigation block, NumericPad on the
    keypad; macOS ignores Ctrl+arrow without them. The stand-in's new key events carry those bits and the Option the
    keyboard holds: the program keeps the first and drops the second."""
    arrow, function, keypad, clear, letter = runs(*({"spec": spec({"op": "keys", "mods": ["ctrl"], "code": code})} for code in (123, 96, 83, 71)),
                                                  {"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "a"}), "layout": ABC})
    # Keypad clear (71) is the keypad key a keyboard gives Fn, not NumericPad (measured on CoreGraphics' own flags).
    for run, bits in ((arrow, NUMERIC_PAD | SECONDARY_FN), (function, SECONDARY_FN), (keypad, NUMERIC_PAD), (clear, SECONDARY_FN),
                      (letter, 0)):
        clean(run)
        key = [r for r in run["posted"] if r["key"] not in (59,)]
        assert [r["flags"] for r in key] == [CTRL | bits] * 2, run["posted"]
        assert all(not r["flags"] & OPTION for r in run["posted"]), "the held Option never rides along"
    assert [r["flags"] for r in arrow["posted"] if r["key"] == 59] == [CTRL, 0], "the modifier itself carries no class bit"


def test_a_modifier_alone_is_pressed_as_a_modifier():
    alone, = runs({"spec": spec({"op": "keys", "mods": ["shift"]})})
    clean(alone)
    assert [(r["key"], r["down"], r["flags"]) for r in alone["posted"]] == [(56, True, SHIFT), (56, False, 0)]


def test_system_keys_and_option_characters_are_keys_again():
    """The former helper took media names (NSSystemDefined, subtype 8, data1 = key type << 16 | state << 8) and found
    characters under Option; both are back. The system event belongs to its NSEvent: the program never releases it."""
    play, held, option = runs({"spec": spec({"op": "keys", "mods": [], "system": 16})},
                              {"spec": spec({"op": "hold", "mods": [], "system": 0, "ms": 150})},
                              {"spec": spec({"op": "keys", "mods": ["cmd"], "char": "Å"}), "layout": ABC})
    for run in (play, held, option):
        clean(run)  # a system event retained and not released, or released once too often, fails here
    # The bridge hands each -CGEvent out autoreleased: the program retains its own reference and releases it once.
    assert play["calls"].count("CFRetain") == 2 and play["retained"] == [0, 0], play
    assert [(r["system"], r["down"], r["flags"], r["subtype"], r["tap"], r["tag"]) for r in play["posted"]] == [
        (16, True, 0xa00, 8, 0, TAG), (16, False, 0xb00, 8, 0, TAG)]
    assert [r["system"] for r in held["posted"]] == [0, 0] and 0.15 in held["delays"]
    assert [(r["key"], r["down"]) for r in option["posted"]] == [(55, True), (56, True), (58, True), (0, True), (0, False),
                                                                 (58, False), (56, False), (55, False)]
    assert option["posted"][3]["flags"] == COMMAND | SHIFT | OPTION


def test_a_key_that_arms_a_composition_never_stands_for_its_character():
    """A custom layout can type a character and leave a composition armed (a nonzero dead-key state): that key is not
    the character's key. Typing it goes on the neutral key with its string; a chord naming it is refused."""
    arming = {"id": "custom", "planes": {0: {39: "'", 0: "a"}}, "arming": {0: [39]}}
    typed, chord = runs({"spec": spec({"op": "type", "text": "'"}), "layout": arming},
                        {"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "'"}), "layout": arming})
    clean(typed)
    assert [(r["key"], r["unicode"]) for r in typed["posted"] if r["down"]] == [(0x5A, "'")]
    assert chord["posted"] == [] and chord["log"][0].startswith("Unable to get key code for '"), chord
    plain, = runs({"spec": spec({"op": "type", "text": "a"}), "layout": arming})
    assert [r["key"] for r in plain["posted"]] == [0, 0], "a key that arms nothing keeps its character"


def test_a_dead_key_never_stands_for_its_character_and_a_crlf_is_one_return():
    typed, chord, crlf = runs({"spec": spec({"op": "type", "text": "a'"}), "layout": US_INTL},
                              {"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "'"}), "layout": US_INTL},
                              {"spec": spec({"op": "type", "text": "a\r\nb\r\n"}), "layout": ABC})
    clean(typed)
    clean(crlf)
    downs = [r for r in typed["posted"] if r["down"]]
    assert [(r["key"], r["unicode"]) for r in downs] == [(0, "a"), (0x5A, "'")] and json.loads(typed["returned"])["override_only"] == 1
    assert chord["posted"] == [] and chord["log"][0].startswith("Unable to get key code for '"), chord
    assert [(r["key"], r["unicode"]) for r in crlf["posted"] if r["down"]] == [(0, "a"), (36, "\r"), (0x5A, "b"), (36, "\r")]
    assert json.loads(crlf["returned"])["chars_total"] == 4


def test_the_counts_of_what_went_out_survive_an_empty_text_and_a_stop_mid_text():
    empty, stopped = runs({"spec": spec({"op": "type", "text": ""})},
                          {"spec": spec({"op": "type", "text": "hih", "interval_ms": 300}, budget_ms=500), "layout": ABC})
    clean(empty)
    assert json.loads(empty["returned"]) == {"posted": 0, "method": "unicode_events", "chars_total": 0, "chars_sent": 0,
                                             "override_only": 0}
    assert "own deadline" in stopped["error"] and stopped["held"] == [] and stopped["leaked"] == []
    line = next(entry for entry in stopped["log"] if entry.startswith("__POSTED__ "))
    report = json.loads(line.split(" ", 1)[1])
    assert report["chars_sent"] == len([r for r in stopped["posted"] if not r["down"]]) and 0 < report["chars_sent"] < 3, report
    assert report["chars_total"] == 3 and report["posted"] == len(stopped["posted"]), report


def test_a_character_counts_when_its_key_goes_down_and_each_press_has_its_own_release():
    """A stop between a key's down and its up: the character went down, so it counts, and the release posted in the
    catch counts as posted; overrides count when they go out, not when planned. In "aA" a stop after the first press
    lets that press up (no Shift), not the shifted release of the last pair."""
    stopped, two, cut = runs({"spec": spec({"op": "type", "text": "ab", "interval_ms": 0}), "layout": ABC, "fail": "post:1"},
                             {"spec": spec({"op": "type", "text": "aA"}), "layout": ABC, "fail": "post:1"},
                             {"spec": spec({"op": "type", "text": "a🙂", "interval_ms": 600}, budget_ms=500), "layout": ABC})
    for run in (stopped, two, cut):
        assert run["held"] == [] and run["leaked"] == [], run
    report = json.loads(next(e for e in stopped["log"] if e.startswith("__POSTED__ ")).split(" ", 1)[1])
    assert report["chars_sent"] == 1 and report["posted"] == len(stopped["posted"]) == 2, (report, stopped["posted"])
    assert [(r["key"], r["down"], r["flags"]) for r in two["posted"]] == [(0, True, 0), (0, False, 0)], "its own release"
    late = json.loads(next(e for e in cut["log"] if e.startswith("__POSTED__ ")).split(" ", 1)[1])
    assert late["override_only"] == 0 and late["chars_sent"] == 1, "the emoji never went out, so it is not counted"


def test_a_sequence_lookup_answers_the_grant_first_then_the_keys_and_posts_nothing():
    """Without the right to post the lookup answers that alone and never reads the layout; with it, the keys no layout
    key gives; a layout without key data is said as unread, never as "no key gives it"."""
    lookup = spec({"op": "resolve", "chars": ["å", "Ж", "a"], "command_chars": ["g", "п"]})
    denied, found, unread = runs({"spec": lookup, "layout": ABC, "post": False}, {"spec": lookup, "layout": ABC},
                                 {"spec": lookup, "layout": {"id": "com.apple.inputmethod.Kotoeri", "planes": None}})
    for run in (denied, found, unread):
        assert run["posted"] == [] and run["error"] is None and run["leaked"] == [] and "CGEventSourceCreate" not in run["calls"], run
    assert json.loads(denied["returned"]) == {"post_events": False}
    assert not any(call.startswith("TIS") for call in denied["calls"]), "the grant answers before any layout read"
    assert json.loads(found["returned"]) == {"layout": "com.apple.keylayout.ABC", "post_events": True, "missing": ["Ж"],
                                             "missing_command": ["п"]}
    assert json.loads(unread["returned"]) == {"post_events": True, "layout_unread": True}
    chord, = runs({"spec": spec({"op": "keys", "mods": ["ctrl"], "char": "a"}), "layout": {"id": "x", "planes": None}})
    assert chord["posted"] == [] and chord["log"] == [
        "WARNING: the active keyboard layout could not be read (no layout data); nothing was sent"], chord


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


def test_a_release_after_an_error_is_the_prepared_event_even_when_the_system_makes_no_more():
    """The key up fails to post and no further event can be made: the release already built goes out, so nothing
    stays down (a second fault no longer leaves the key pressed)."""
    broken, = runs({"spec": spec({"op": "hold", "mods": ["cmd"], "code": 36, "ms": 50}), "fail": ["post:2", "event:4", "event:5"]})
    assert broken["error"] == "injected post failure" and broken["held"] == [] and broken["leaked"] == [], broken
    assert [(r["key"], r["down"]) for r in broken["posted"]] == [(55, True), (36, True), (36, False), (55, False)]


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
        "posted": 4, "chars_sent": 2, "chars_total": 2, "override_only": 1}
    assert mac_input.answer_fields("") == {} and mac_input.answer_fields('{"posted":3}') == {"posted": 3}
    stopped = 'typed\n__POSTED__ {"posted":3,"method":"unicode_events","chars_total":9,"chars_sent":1,"override_only":0}\n'
    assert mac_input.answer_fields("", stopped) == {"posted": 3, "chars_sent": 1, "chars_total": 9, "override_only": 0}
    # The receipt shows the operations readable: the text as typed, the SSH source named, not repeated.
    shown = mac_input.shown(local)
    assert shown[:4] == local[:4] and "Привет" in shown[-1] and "\\u" not in shown[-1] and len(shown[-1]) < len(local[-1])
    assert mac_input.shown(remote)[3] == "-e input_events.js" and mac_input.shown(["xdotool", "key"]) == ["xdotool", "key"]
    assert mac_input.position_of('{"x":3,"y":-4}\n') == {"x": 3, "y": -4, "coordinate_space": "input (macOS logical points)"}
    assert mac_input.position_of("pos 3 4") == {} and mac_input.position_of(None) == {}
    # The typing budget is the measured per-character rate with room (40 ms), plus the fixed start.
    assert mac_input.type_seconds("x" * 600) == 10 + 600 * 40 // 1000 and mac_input.type_seconds("x" * 9000) == 170
    assert mac_input.type_seconds("x" * 100, interval_ms=60) == 10 + 100 * 100 // 1000


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
        "capability_unavailable": True, "permission": "screen_recording", "repair": "owner",
        "where": "System Settings > Privacy & Security > Screen & System Audio Recording"}}
    # A success names what it proved, and only a success does.
    assert mac_input.proved({"ok": True}, "accessibility") == {"ok": True, "permissions_available": ["accessibility"]}
    assert mac_input.proved({"ok": False}, "accessibility") == {"ok": False}
    assert set(mac_input.permission_refusal("accessibility")) == {"refusal"}
    # The native helpers' own reasons for a missing grant gain the same field; any other reason stays as it was.
    for reason, grant in (("accessibility_untrusted", "accessibility"), ("screen_recording_denied", "screen_recording")):
        assert mac_input.with_refusal({"ok": False, "reason": reason})["refusal"]["permission"] == grant
    assert mac_input.with_refusal({"ok": False, "reason": "no_window"}) == {"ok": False, "reason": "no_window"}


def test_a_sequence_with_a_character_no_key_gives_is_refused_whole_and_an_unread_lookup_sends_nothing():
    steps = [mac_input.argv([{"op": "keys", "mods": [], "code": 36}]), mac_input.argv([{"op": "keys", "mods": [], "char": "Ж"}]),
             ["osascript", "-l", "JavaScript", "paste_chord.js", "g"]]
    cmd, chars, command = mac_input.resolve_argv(steps)
    assert mac_input.ops_of(cmd) == [{"op": "resolve", "chars": ["Ж"], "command_chars": ["g"]}] and (chars, command) == (["Ж"], ["g"])
    assert mac_input.resolve_argv(steps[:1]) == (None, [], [])
    refusal = mac_input.resolve_refusal(0, '{"layout":"com.apple.keylayout.ABC","missing":["Ж"],"missing_command":[]}', "")
    assert refusal["reason"] == "untypable_characters" and refusal["untyped"] == ["Ж"] and refusal["dispatched"] is False
    assert "ABC" in refusal["error"] and "no step" in refusal["error"]
    assert mac_input.resolve_refusal(0, '{"missing":[],"missing_command":[]}', "") is None
    # Unread is a refusal too (nothing is known about the layout the steps would land in), and the grant comes first.
    for unread in ((1, "", "boom"), (0, "", ""), (0, "not json", "")):
        out = mac_input.resolve_refusal(*unread)
        assert out["reason"] == "layout_unread" and out["dispatched"] is False, unread
    denied = mac_input.resolve_refusal(0, '{"post_events":false,"missing":["Ж"],"missing_command":[]}', "")
    assert denied["reason"] == "permission_denied" and denied["refusal"]["permission"] == "post_events" and denied["dispatched"] is False


def test_what_was_posted_decides_the_mark():
    empty = mac_input.posted_effect({"ok": True, "observed_effect": True, "posted": 0})
    assert "observed_effect" not in empty and "nothing" in empty["note"]
    switched = mac_input.posted_effect({"ok": True, "observed_effect": True, "posted": 0, "target": {"activated": True}})
    assert switched["observed_effect"] is True and switched["activation_done"] is True
    partial = mac_input.posted_effect({"ok": False, "posted": 4, "chars_sent": 2, "chars_total": 9})
    assert partial["observed_effect"] is True and "4 event(s) (2 of 9 characters)" in partial["note"]
    assert mac_input.posted_effect({"ok": True, "observed_effect": True, "posted": 3}) == {"ok": True, "observed_effect": True, "posted": 3}
    assert mac_input.posted_effect({"ok": False, "dispatched": None}) == {"ok": False, "dispatched": None}, "no count: unknown"
