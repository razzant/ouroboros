"""unix_computer_use: the paste chord of type_text(via="paste") on a Mac, locally and over SSH (#2589).

Measured in a macOS 26.4 guest on an AppKit Edit menu (the change's .review-evidence): the menu matched the
Command-plane characters of the key event, so typing the layout's "v" (cliclick t:v) missed Paste under an ok receipt
on Russian - PC (Option added) and Dvorak - QWERTY Command (its typing-plane "v" is "." under Command), and a fixed
US-V key missed it on Dvorak ("k"), while the key whose Command plane is "v", posted with exactly the Command flag,
pasted on all six layouts. ``skills/unix_computer_use/paste_chord.js`` posts input, so here it runs byte for byte under
Node against inert stand-ins of the bridge it uses (``tests/fixtures/paste_chord_inert.cjs``: no ObjC, no CoreGraphics,
nothing posted), is compiled without running, and goes through a real shell to stand-ins of the Mac's own tools.
key() sends Command and one character through the same program, given that character as its argument (8.11.2026: under
a Cyrillic layout cliclick refused cmd+g in desktop Slack, "Unable to get key code for g"); every other combo keeps
cliclick.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.group_run import run_in_group
from tests.test_unix_computer_use_targeting import CHORD_POSTED, SAFARI, inputs, load_plugin, mac, receipt

CHORD = Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use" / "paste_chord.js"
INERT = Path(__file__).resolve().parent / "fixtures" / "paste_chord_inert.cjs"
COMMAND = 0x100000  # kCGEventFlagMaskCommand
SHIFT = 0x20000  # kCGEventFlagMaskShift
# From the guest's measured layout tables (#2589), by UCKeyTranslate modifier state: 0 none, 1 Command, 8 Option,
# 9 Command+Option.
ABC = {"id": "com.apple.keylayout.ABC", "planes": {0: {9: "v", 47: "."}, 1: {9: "v", 47: "."}, 8: {9: "√"}}}
RUSSIAN_PC = {"id": "com.apple.keylayout.RussianWin", "planes": {0: {9: "м", 47: "ю"}, 1: {9: "v", 47: "."}, 8: {9: "v"}}}
DVORAK = {"id": "com.apple.keylayout.Dvorak", "planes": {0: {9: "k", 47: "v"}, 1: {9: "k", 47: "v"}}}
DVORAK_QWERTY_CMD = {"id": "com.apple.keylayout.DVORAK-QWERTYCMD", "planes": {0: {9: "k", 47: "v"}, 1: {9: "v", 47: "."}}}
TURKMEN = {"id": "com.apple.keylayout.Turkmen", "planes": {0: {9: "ý"}, 1: {9: "ý"}, 9: {9: "√"}}}


def chord_runs(*scenarios: dict) -> list[dict]:
    """The chord program run under Node against the inert stand-ins: one trace per scenario."""
    node = shutil.which("node")
    assert node, "Node runs the chord program against inert stand-ins"
    done = run_in_group([node, str(INERT), str(CHORD), json.dumps(scenarios)], timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


def chord(key: int, *, shift: bool = False) -> list[list]:
    """[tap, key code, down, flags]: Command down, the key down and up with exactly Command, Command up; session tap.
    With shift, Shift goes down after Command and up before it, and the key carries both flags."""
    held = COMMAND | (SHIFT if shift else 0)
    return [[1, 55, True, COMMAND], *([[1, 56, True, held]] if shift else []), [1, key, True, held], [1, key, False, held],
            *([[1, 56, False, COMMAND]] if shift else []), [1, 55, False, 0]]


def test_the_paste_chord_is_command_and_the_key_that_gives_v_under_command(tmp_path, monkeypatch):
    module = load_plugin()
    assert list(module.PASTE_CHORD_ARGV) == ["osascript", "-l", "JavaScript", str(CHORD)]
    layouts = ((ABC, 9), (RUSSIAN_PC, 9), (DVORAK, 47), (DVORAK_QWERTY_CMD, 9))
    runs = chord_runs(*({"layout": layout} for layout, _key in layouts))
    for (layout, key), run in zip(layouts, runs):
        assert run["error"] is None and run["returned"] == "", run
        assert run["log"] == [f"paste chord: Command + key code {key} (v under Command in {layout['id']})"]
        # Exactly the Command flag: the Option the keyboard holds (each new event's own flags here) never rides along.
        assert run["posted"] == chord(key) and run["held"] == [] and run["source_state"] == 1, run  # HID system state
        assert [call for call in run["calls"] if call in ("CGEventPost", "delay")] == ["CGEventPost", "delay"] * 4
        # Key-down action, Command alone (cmdKey >> 8), the keyboard type in use, no dead keys; the lowest code wins.
        assert run["translated"] == [[code, 0, 1, 198, 1, 4] for code in range(key + 1)], run["translated"]
        # The layout data is read only while its input source lives (the stand-ins refuse anything else), and each
        # object the program made is released once: the input source, the four events and their event source.
        assert run["leaked"] == [] and sorted(run["released"]) == ["CGEvent"] * 4 + ["CGEventSource", "TISInputSource"]
    assert runs[0]["log"][0] + "\n" == CHORD_POSTED, "the receipt tests' chord line is the one the program prints"
    # key("cmd+v") is the same chord given "v" and never touches the clipboard (no pbcopy): only type_text writes it.
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="cmd+v"))
    assert out["ok"] is True and out["method"] == "command_chord" and state["clipboard"] is None, out
    assert inputs(state) == [] and state["chords"] == [[*module.PASTE_CHORD_ARGV, "v"]]
    assert not any(argv[0] == "pbcopy" for argv in state["calls"])


def test_a_refused_paste_chord_posts_nothing_in_the_words_the_receipt_reads(tmp_path, monkeypatch):
    turkmen, unread, no_data, denied = chord_runs(
        {"layout": TURKMEN}, {"layout": None}, {"layout": {"id": "com.apple.inputmethod.Kotoeri", "planes": None}},
        {"layout": ABC, "post": False})
    for run in (turkmen, unread, no_data, denied):
        assert run["error"] is None and run["posted"] == [] and "CGEventSourceCreate" not in run["calls"], run
        assert run["leaked"] == [] and run["released"] == ([] if run is unread else ["TISInputSource"]), run
    assert turkmen["translated"] == [[code, 0, 1, 198, 1, 4] for code in range(128)], "every key code, then the refusal"
    assert turkmen["log"] == ["Unable to get key code for v (no key of com.apple.keylayout.Turkmen gives v under Command); "
                              "the paste chord was not sent"]
    assert unread["log"] == no_data["log"] == ["Unable to get key code for v (the active keyboard layout could not be read); "
                                               "the paste chord was not sent"]
    assert denied["log"] == ["WARNING: Accessibility privileges not enabled (no right to post input events); "
                             "the paste chord was not sent"]
    for run, reason in ((turkmen, "untypable_characters"), (unread, "untypable_characters"), (denied, "permission_denied")):
        module, api, state = mac(tmp_path, monkeypatch, chord=(0, "", run["log"][0] + "\n"))
        out = receipt(api.tools["type_text"](text="x", via="paste"))
        assert out["ok"] is False and out["reason"] == reason and out["dispatched"] is True and out["clipboard_done"] is True, out
        assert "paste chord was not sent" in out["error"] and "paste chord was not sent" in out["note"], "refused, not dropped"


def test_a_paste_chord_the_system_cannot_finish_is_never_ok_and_leaves_no_key_down():
    """An event the system cannot make (any of the four) fails the chord before anything is posted. A call that throws
    after Command (or the key) went down still lets up what went down, the key before Command and each at most once,
    and nothing goes down again. Every object is released and the original error ends the program: osascript exits 1,
    so the receipt is never ok (test_a_failed_paste_helper_still_reports_the_clipboard_write)."""
    full = chord(9)
    faults = [{"fail": f"{call}:{n}"} for call in ("event", "flags", "post", "delay") for n in range(4)]
    faults += [{"fail": f"post:{n}", "landed": True} for n in range(4)]  # it threw after the event reached the system
    for fault, run in zip(faults, chord_runs(*({"layout": ABC, **fault} for fault in faults))):
        call, n = fault["fail"].split(":")
        assert run["log"] == [] and run["held"] == [] and run["leaked"] == [], (fault, run)
        if call in ("event", "flags"):
            assert run["posted"] == [] and run["error"] == (
                "injected flags failure" if call == "flags" else
                f"CGEventCreateKeyboardEvent made no event for key code {full[int(n)][1]}; the paste chord was not sent"), run
            continue
        assert run["error"] == f"injected {call} failure", ("the original error", fault, run)
        took = int(n) + (call == "delay" or bool(fault.get("landed")))  # the chord's own events the system took
        assert run["posted"][:took] == full[:took], (fault, run)
        assert run["posted"][took:] in ([], full[2:3], full[3:], full[2:]), ("only key up, then Command up", fault, run)
    # No event source: the events are made without one, as CGEventCreateKeyboardEvent allows, and NULL is never released.
    (run,) = chord_runs({"layout": ABC, "no_event_source": True})
    assert run["error"] is None and run["posted"] == full and run["leaked"] == [] and "CGEventSource" not in run["released"]


# Key codes by Apple's kVK_ANSI_* constants (G 5, ] 30, comma 43, slash 44; Dvorak's / and ] sit on [ 33 and = 24); the
# Command plane of a Russian layout is Latin, as the guest's measured tables show for v (RUSSIAN_PC above). States 2 and 3
# are Shift and Command+Shift, from what the guest's fixture logged for Command+Shift events on 12.11.2026 (characters
# and charactersIgnoringModifiers): ABC and Dvorak ignore Shift under Command (state 3 is state 1), RussianWin and
# Dvorak - QWERTY Command have Command+Shift characters of their own.
ABC_KEYS = {"id": "com.apple.keylayout.ABC", "planes": {0: {5: "g", 30: "]", 43: ",", 44: "/"}, 1: {5: "g", 30: "]", 43: ",", 44: "/"},
                                                        2: {5: "G", 30: "}", 43: "<", 44: "?"}, 3: {5: "g", 30: "]", 43: ",", 44: "/"}}}
RUSSIAN_KEYS = {"id": "com.apple.keylayout.RussianWin", "planes": {0: {5: "п", 26: "7", 43: "б", 44: "."},
                                                                   1: {5: "g", 26: "7", 30: "]", 43: ",", 44: "/"},
                                                                   2: {5: "П", 26: "?", 30: "Ъ", 43: "Б", 44: ","},
                                                                   3: {5: "G", 26: "&", 30: "}", 43: "<", 44: "?"}}}
DVORAK_KEYS = {"id": "com.apple.keylayout.Dvorak", "planes": {1: {24: "]", 33: "/"}, 2: {24: "}", 33: "?"}, 3: {24: "]", 33: "/"}}}
QWERTY_CMD_KEYS = {"id": "com.apple.keylayout.DVORAK-QWERTYCMD", "planes": {1: {24: "=", 30: "]", 33: "[", 44: "/"},
                                                                          2: {24: "}", 30: "+", 33: "?", 44: "Z"},
                                                                          3: {24: "+", 30: "}", 33: "{", 44: "?"}}}


def shift_pass(layout: dict, key: int) -> list[int]:
    """The modifier states the second pass reads, key by key up to ``key``: Command+Shift, Command, and the Shift plane
    where the two are the same (the layout ignores Shift under Command there)."""
    planes = layout["planes"]
    return [state for code in range(key + 1) for state in
            ([3, 1, 2] if planes[3].get(code, "") == planes[1].get(code, "") else [3, 1])]


def command_line(char: str, key: int, layout: dict, held: str = "Command") -> str:
    return f"Command chord: {held} + key code {key} ({char} under {held} in {layout['id']})"


def test_a_command_chord_posts_the_key_that_gives_its_character_under_command():
    """The character arrives as the program's argument: Command and the key whose Command plane gives it, with exactly
    the Command flag, under a layout whose typing plane has no such character (the Cyrillic g) and under ABC."""
    cases = ((RUSSIAN_KEYS, "g", 5), (RUSSIAN_KEYS, ",", 43), (ABC_KEYS, "g", 5), (DVORAK_QWERTY_CMD, "v", 9))
    for (layout, char, key), run in zip(cases, chord_runs(*({"layout": layout, "argv": [char]} for layout, char, _k in cases))):
        assert run["error"] is None and run["log"] == [command_line(char, key, layout)], run
        assert run["posted"] == chord(key) and run["held"] == [] and run["leaked"] == [], run
        assert run["translated"] == [[code, 0, 1, 198, 1, 4] for code in range(key + 1)], "the Command plane only"
    # Without a word after the program (osascript hands run an empty list) it is the paste chord, word for word.
    no_word, bare = chord_runs({"layout": ABC, "argv": []}, {"layout": ABC})
    assert no_word["log"] == bare["log"] == [CHORD_POSTED.strip()] and no_word["posted"] == bare["posted"] == chord(9)
    # A character only Command and Shift give is the same resolver's second pass, read as an app reads that chord: the
    # Command+Shift characters, or where the layout ignores Shift under Command the Shift plane's (cmd+? and cmd+} on ABC
    # and Dvorak, the keys cliclick's typed Shift sent before; not Dvorak - QWERTY Command's =, whose Command+Shift is +).
    # The whole chord with Shift is made before anything is posted, never resent.
    shifted = ((ABC_KEYS, "?", 44), (ABC_KEYS, "}", 30), (RUSSIAN_KEYS, "?", 44), (RUSSIAN_KEYS, "}", 30),
               (DVORAK_KEYS, "?", 33), (DVORAK_KEYS, "}", 24), (QWERTY_CMD_KEYS, "?", 44), (QWERTY_CMD_KEYS, "}", 30))
    for (layout, char, key), run in zip(shifted, chord_runs(*({"layout": layout, "argv": [char]} for layout, char, _k in shifted))):
        assert run["error"] is None and run["log"] == [command_line(char, key, layout, "Command+Shift")], run
        assert run["posted"] == chord(key, shift=True) and run["held"] == [] and run["leaked"] == [], run
        assert [state for _code, _action, state, *_rest in run["translated"]] == [1] * 128 + shift_pass(layout, key), run
    # A failure once Shift is down still lets the key (if down), Shift and Command up, each once.
    failed, = chord_runs({"layout": ABC_KEYS, "argv": ["?"], "fail": "post:2", "landed": True})
    assert failed["error"] == "injected post failure" and failed["posted"] == chord(44, shift=True) and failed["held"] == []


def test_a_refused_command_chord_posts_nothing_and_a_malformed_argument_reaches_no_layout():
    """A character no key gives under Command or Command+Shift (a Cyrillic layout's own letter: its Command planes are
    Latin, and the typing plane is never tried), and no right to post, refuse in the receipt's words with nothing posted.
    An argument that is not one printable character ends the program before any binding: osascript exits 1."""
    shifted, denied, *malformed = chord_runs(
        {"layout": RUSSIAN_KEYS, "argv": ["п"]}, {"layout": RUSSIAN_KEYS, "argv": ["g"], "post": False},
        *({"layout": ABC_KEYS, "argv": argv} for argv in (["gg"], ["g", "h"], [" "], ["\u0000"], [""])))
    assert shifted["log"] == ["Unable to get key code for п (no key of com.apple.keylayout.RussianWin gives п under Command "
                              "or Command+Shift); the Command chord was not sent"], shifted
    assert [state for _code, _action, state, *_rest in shifted["translated"]] == [1] * 128 + shift_pass(RUSSIAN_KEYS, 127)
    assert denied["log"] == ["WARNING: Accessibility privileges not enabled (no right to post input events); "
                             "the Command chord was not sent"], denied
    for run in (shifted, denied):
        assert run["error"] is None and run["posted"] == [] and "CGEventSourceCreate" not in run["calls"] and run["leaked"] == []
    for run in malformed:
        assert run["error"].startswith("the Command chord takes one printable character") and run["calls"] == [], run


def test_only_command_and_one_character_takes_the_chord_and_every_other_combo_keeps_cliclick(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch)
    out = receipt(api.tools["key"](keys="cmd+G"))  # a letter's case under a modifier is not a Shift
    assert out["ok"] is True and out["method"] == "command_chord" and out["keys"] == "cmd+G", out
    assert state["chords"] == [[*module.PASTE_CHORD_ARGV, "g"]] and inputs(state) == [] and state["clipboard"] is None
    assert out["delivered_to"]["app"] == "Safari" and out["observed_effect"] is True
    state["chords"].clear()
    receipt(api.tools["key"](keys="cmd+shift+g cmd+alt+g ctrl+g cmd+return cmd+tab g"))
    receipt(api.tools["hold_key"](keys="cmd", duration_ms=100))
    assert state["chords"] == [] and inputs(state) == [
        ["cliclick", "kd:cmd,shift", "t:g", "ku:cmd,shift"], ["cliclick", "kd:cmd,alt", "t:g", "ku:cmd,alt"],
        ["cliclick", "kd:ctrl", "t:g", "ku:ctrl"], ["cliclick", "kd:cmd", "kp:return", "ku:cmd"],
        ["cliclick", "kd:cmd", "kp:tab", "ku:cmd"], ["cliclick", "t:g"], ["cliclick", "kd:cmd", "w:100", "ku:cmd"]]
    # Each step's tool is checked before anything is switched or sent: no osascript, no chord, nothing at all.
    module, api, state = mac(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_which", lambda name: "" if name == "osascript" else f"/usr/bin/{name}")
    refused = receipt(api.tools["key"](keys="down cmd+g", app="Keynote"))
    assert refused["dispatched"] is False and "osascript is not installed" in refused["error"], refused
    assert state["helper"] == [] and inputs(state) == [] and state["chords"] == []


REFUSED_G = ("Unable to get key code for g (no key of com.apple.keylayout.Turkmen gives g under Command or Command+Shift); "
             "the Command chord was not sent\n")
DENIED = "WARNING: Accessibility privileges not enabled (no right to post input events); the Command chord was not sent\n"


def test_a_refused_shortcut_says_it_was_not_sent_and_a_failed_one_that_its_outcome_is_unknown(tmp_path, monkeypatch):
    """The program's own refusal happens before any event exists: nothing was posted, a known fact (dispatched false,
    who was in front, never delivered_to), unlike cliclick's "may have been typed". A switch before it keeps its mark.
    A program that ran and did not report success may have posted part of the chord: unknown, never "not sent"."""
    module, api, state = mac(tmp_path, monkeypatch, chord=(0, "", REFUSED_G))
    out = receipt(api.tools["key"](keys="cmd+g"))
    assert (out["ok"], out["reason"], out["dispatched"], out["untyped"]) == (False, "untypable_characters", False, ["g"]), out
    assert "with or without Shift" in out["error"] and "(cmd+g, not the layout's own letter)" in out["error"], out
    assert "shortcut was not sent" in out["error"] and "may have" not in out["error"], out
    assert out["note"] == "the shortcut was not sent (see error); this tool observed no UI change", out
    assert "observed_effect" not in out and "delivered_to" not in out and out["front"]["app"] == "Safari", out
    switched = receipt(api.tools["key"](keys="cmd+g", app="Keynote"))
    assert switched["observed_effect"] is True and switched["activation_done"] is True and switched["dispatched"] is False
    assert switched["note"] == "the switch to Keynote ran; the input after it: the shortcut was not sent (see error)", switched
    module, api, state = mac(tmp_path, monkeypatch, chord=(0, "", DENIED))
    out = receipt(api.tools["key"](keys="cmd+g"))
    assert (out["ok"], out["reason"], out["dispatched"]) == (False, "permission_denied", False), out
    assert out["error"] == "this process may not post input events (the Accessibility grant), so the shortcut was not sent"
    module, api, state = mac(tmp_path, monkeypatch, chord=(1, "", "execution error: Error: injected post failure (-2700)"))
    out = receipt(api.tools["key"](keys="cmd+g"))
    assert out["ok"] is False and out["dispatched"] is True and "observed_effect" not in out and "reason" not in out, out
    assert "unknown" in out["note"] and "not sent" not in out["note"] and out["front_at_dispatch"]["app"] == "Safari", out
    # Over SSH the same program answers in the same words, and the receipt reads them the same way.
    from tests.test_unix_computer_use_remote_targeting import mac_over_ssh, ssh
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, input_stderr=REFUSED_G))
    out = receipt(api.tools["key"](keys="cmd+g"))
    assert (out["ok"], out["reason"], out["dispatched"], out["method"]) == (False, "untypable_characters", False, "command_chord")
    assert "observed_effect" not in out and "delivered_to" not in out and out["front"]["app"] == "Safari", out


def test_a_refused_shortcut_stops_its_sequence_with_the_earlier_steps_kept(tmp_path, monkeypatch):
    module, api, state = mac(tmp_path, monkeypatch, chord=(0, "", REFUSED_G))
    out = receipt(api.tools["key"](keys="down cmd+g return"))
    assert out["ok"] is False and (out["steps_done"], out["steps_total"], out["dispatched"]) == (1, 3, False), out
    assert out["error"].startswith("step 'cmd+g' not sent: no key of the active keyboard layout gives g under Command")
    assert out["observed_effect"] is True and out["reason"] == "untypable_characters", "the arrow ran before it"
    assert out["note"] == ("1 of 3 step(s) completed while Safari was in front; the next step: the shortcut was not sent "
                           "(see error); this tool observed no UI change; the rest were not sent"), out
    assert inputs(state) == [["cliclick", "kp:arrow-down"]] and len(state["chords"]) == 1, "return was never sent"


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="the remote command is a POSIX shell command")
def test_an_ssh_shortcut_hands_its_character_to_the_same_program_as_a_separate_word(tmp_path, monkeypatch):
    """key() on a Mac over SSH runs the chord program's source through the Mac's own osascript with the character as
    the word after it. The whole remote command (front reads, input) runs here in a real POSIX shell against
    stand-ins that record what they received: each character arrives byte for byte, whatever the shell makes of it,
    and the program is unchanged. Nothing is posted (osascript here is a recorder)."""
    from tests.test_unix_computer_use_remote_targeting import ssh
    real_run, sh = subprocess.run, shutil.which("sh")
    bin_dir, out_dir = tmp_path / "bin", tmp_path / "out"
    bin_dir.mkdir()
    out_dir.mkdir()
    (bin_dir / "osascript").write_text('#!/bin/sh\nn=$(ls "$OUT" | wc -l | tr -d " ")\n'
                                       'for a in "$@"; do printf "%s\\000" "$a"; done > "$OUT/$n"\n'
                                       'echo "Command chord: Command + key code 5 (recorded)" >&2\n', encoding="utf-8", newline="\n")
    (bin_dir / "lsappinfo").write_text('#!/bin/sh\n[ "$1" = front ] && echo "ASN:0x0-0x1:" || '
                                       'printf \'"pid"=100\\n"CFBundleIdentifier"="com.apple.Safari"\\n"LSDisplayName"="Safari"\\n\'\n',
                                       encoding="utf-8", newline="\n")
    for tool in ("osascript", "lsappinfo"):
        (bin_dir / tool).chmod(0o755)

    def shell(command):
        done = real_run([sh, "-c", command], capture_output=True, text=True, timeout=20,
                        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "OUT": str(out_dir)})
        return done.returncode, done.stdout, done.stderr
    module, api, calls = ssh(tmp_path, monkeypatch, shell)
    chars = ["g", "'", '"', "$", "`", "\\", ";", "ё"]
    out = receipt(api.tools["key"](keys=" ".join(f"cmd+{c}" for c in chars)))
    assert out["ok"] is True and out["steps"] == len(chars), out
    source = CHORD.read_text(encoding="utf-8")
    for n, char in enumerate(chars):
        argv = [a.decode() for a in (out_dir / str(n)).read_bytes().split(b"\0")[:-1]]
        assert argv == ["-l", "JavaScript", "-e", source, char], (char, argv[-1:])
    assert all("cliclick kd:" not in argv[-1] for argv in calls)


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("osacompile"), reason="JavaScript for Automation is macOS")
def test_the_paste_chord_program_compiles_without_running(tmp_path):
    """osacompile parses and compiles the program into a file and runs nothing (running it would post input)."""
    done = subprocess.run(["osacompile", "-l", "JavaScript", "-o", str(tmp_path / "chord.scpt"), str(CHORD)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and (tmp_path / "chord.scpt").exists(), done.stderr


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("sh"), reason="the remote command is a POSIX shell command")
def test_ssh_paste_hands_the_exact_text_and_chord_program_through_a_real_shell(tmp_path):
    """The remote command as a POSIX shell parses it, with stand-ins for pbcopy and osascript that record what they
    received: the text byte for byte (quotes, $, backticks, Cyrillic), pbcopy told it is UTF-8, and osascript given
    the chord program unchanged. Nothing is posted (osascript here is a recorder)."""
    module = load_plugin()
    remote = sys.modules[module.__name__ + ".remote"]
    bin_dir, out_dir = tmp_path / "bin", tmp_path / "out"
    bin_dir.mkdir()
    out_dir.mkdir()
    (bin_dir / "pbcopy").write_text('#!/bin/sh\nprintf %s "$LC_ALL" > "$OUT/locale"\ncat > "$OUT/clipboard"\n',
                                    encoding="utf-8", newline="\n")
    (bin_dir / "osascript").write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\000" "$a"; done > "$OUT/argv"\n',
                                       encoding="utf-8", newline="\n")
    for tool in ("pbcopy", "osascript"):
        (bin_dir / tool).chmod(0o755)
    text = "Привет 'мир' $HOME `id` \\ \"q\" %s"
    done = subprocess.run([shutil.which("sh"), "-c", remote.ssh_paste_command(text)], capture_output=True, text=True, timeout=20,
                          env={"PATH": f"{bin_dir}:/usr/bin:/bin", "OUT": str(out_dir)})
    assert done.returncode == 0 and done.stdout == "__CLIPBOARD_WRITTEN__\n", done
    assert (out_dir / "clipboard").read_bytes() == text.encode("utf-8") and (out_dir / "locale").read_text() == "en_US.UTF-8"
    argv = (out_dir / "argv").read_bytes().split(b"\0")[:-1]
    assert [a.decode() for a in argv] == ["-l", "JavaScript", "-e", CHORD.read_text(encoding="utf-8")], "the program, unchanged"


LOCALE = Path("/usr/bin/locale")
PASTED = "Привет, мир — “q” ✓ 🐍 'x' $HOME"
# A fresh interpreter runs the plugin's clipboard writer (plugin._pipe, what type_text via=paste calls) and prints its
# own text encoding, the charmap a pbcopy it started would have inherited untold (`locale charmap` makes pbcopy's own
# setlocale(LC_ALL, "") + nl_langinfo(CODESET)) and the writer's exit code. The text is an ASCII literal in the program.
WRITER = ("import importlib.util, locale, subprocess, sys\n"
          "spec = importlib.util.spec_from_file_location('ucu_writer', sys.argv[1] + '/plugin.py',"
          " submodule_search_locations=[sys.argv[1]])\n"
          "module = importlib.util.module_from_spec(spec)\n"
          "sys.modules[spec.name] = module\n"
          "spec.loader.exec_module(module)\n"
          "untold = subprocess.run(['/usr/bin/locale', 'charmap'], capture_output=True, text=True).stdout.strip()\n"
          f"print(locale.getpreferredencoding(False), untold, module._pipe(['pbcopy'], {ascii(PASTED)})[0])\n")


@pytest.mark.skipif(sys.platform != "darwin" or not LOCALE.exists(),
                    reason="pbcopy decodes with the macOS C library's locale")
def test_the_local_clipboard_write_is_utf8_and_tells_pbcopy_so_whatever_locale_the_process_inherited(tmp_path):
    """pbcopy decodes stdin with the locale it inherits, the C encoding without one. A Finder or launchd start has none
    and was already safe (PEP 538 coerces LC_CTYPE=C.UTF-8 for Python and its children), but a shell or headless start
    can hand down LC_ALL=C, no locale with coercion off (both read UTF-8 as US-ASCII) or a legacy locale (Python then
    writes KOI8-R, which has no ✓). The writer runs in a fresh interpreter per start against a recorder pbcopy: what
    it read and the charmap it would decode with. No pasteboard is touched."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pbcopy").write_text('#!/bin/sh\n/usr/bin/locale charmap > "$OUT/charmap"\n'
                                    'exec cat > "$OUT/clipboard"\n', encoding="utf-8", newline="\n")
    (bin_dir / "pbcopy").chmod(0o755)
    starts = {"finder": [], "lc_all_c": ["LC_ALL=C", "LANG=C"], "coercion_off": ["PYTHONCOERCECLOCALE=0"],
              "koi8_r": ["LANG=ru_RU.KOI8-R"]}
    untold = {}
    for name, start in starts.items():
        out = tmp_path / name
        out.mkdir()
        # `env -u` makes each start's locale exactly the named one: tests/conftest.py fills a child's missing LC_ALL
        # and LANG with C (the hermetic test environment), which would turn every start into lc_all_c.
        done = run_in_group(["/usr/bin/env", "-u", "LC_ALL", "-u", "LANG", *start, sys.executable, "-S", "-B", "-c",
                             WRITER, str(CHORD.parent)], timeout=60,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "OUT": str(out)})
        assert done.returncode == 0, (name, done.stdout, done.stderr)
        encoding, untold[name], rc = done.stdout.split()
        assert rc == "0" and (out / "charmap").read_text().strip() == "UTF-8", (name, done.stdout)
        assert (out / "clipboard").read_bytes() == PASTED.encode("utf-8"), (name, encoding)
    # The controls: what a pbcopy that was not told would have decoded with. Only the Finder/launchd start was safe.
    assert untold == {"finder": "UTF-8", "lc_all_c": "US-ASCII", "coercion_off": "US-ASCII", "koi8_r": "KOI8-R"}, untold
