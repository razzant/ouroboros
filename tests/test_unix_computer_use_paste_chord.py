"""unix_computer_use: the paste chord of type_text(via="paste") on a Mac, locally and over SSH (#2589).

Measured in a macOS 26.4 guest on an AppKit Edit menu (the change's .review-evidence): the menu matched the
Command-plane characters of the key event, so typing the layout's "v" (cliclick t:v) missed Paste under an ok receipt
on Russian - PC (Option added) and Dvorak - QWERTY Command (its typing-plane "v" is "." under Command), and a fixed
US-V key missed it on Dvorak ("k"), while the key whose Command plane is "v", posted with exactly the Command flag,
pasted on all six layouts. ``skills/unix_computer_use/paste_chord.js`` posts input, so here it runs byte for byte under
Node against inert stand-ins of the bridge it uses (``tests/fixtures/paste_chord_inert.cjs``: no ObjC, no CoreGraphics,
nothing posted), is compiled without running, and goes through a real shell to stand-ins of the Mac's own tools.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.group_run import run_in_group
from tests.test_unix_computer_use_targeting import CHORD_POSTED, inputs, load_plugin, mac, receipt

CHORD = Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use" / "paste_chord.js"
INERT = Path(__file__).resolve().parent / "fixtures" / "paste_chord_inert.cjs"
COMMAND = 0x100000  # kCGEventFlagMaskCommand
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


def chord(key: int) -> list[list]:
    """[tap, key code, down, flags]: Command down, the key down and up with exactly Command, Command up; session tap."""
    return [[1, 55, True, COMMAND], [1, key, True, COMMAND], [1, key, False, COMMAND], [1, 55, False, 0]]


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
    # key() keeps its meaning: a shortcut still types its letter in the active layout (cliclick), untouched by the chord.
    module, api, state = mac(tmp_path, monkeypatch)
    receipt(api.tools["key"](keys="cmd+v"))
    assert inputs(state) == [["cliclick", "kd:cmd", "t:v", "ku:cmd"]] and state["chords"] == []


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
