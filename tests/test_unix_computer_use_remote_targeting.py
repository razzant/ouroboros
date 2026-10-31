"""unix_computer_use on the remote backends: OSWorld guests and Macs over SSH (see test_unix_computer_use_targeting.py).

The same receipt contract as the local path, pinned at the transport boundary (``urlopen`` toward a guest, ``ssh``): a target
is resolved from the remote's own inventory and activated by window id or pid, the guest program is executed here against a
fake pyautogui/xdotool, every exit keeps the facts it observed, and nothing a window is called can forge a marker.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.request

import pytest

from tests.test_unix_computer_use_targeting import API, KEYNOTE, SAFARI, load_plugin, receipt

# --- remote targets: OSWorld guest and a Mac over SSH ------------------------------------------

import base64


def listing(front: str, rows: str | None = None) -> str:
    """What the guest's listing script prints: the base64 rows on one line, `__ACTIVE__` on its own, then the front."""
    rows = GUEST_ROWS if rows is None else rows
    return base64.b64encode(rows.encode()).decode() + "\n__ACTIVE__\n" + front + "\n"


GUEST_ROWS = ("0x00000001  0 4242 10 20 800 600 navigator.Firefox vm Browser\n"
              "0x00000002  0 5151 0 0 1024 768 keynote.Keynote vm Deck - Keynote\n"
              "0x00000003  0 5151 0 0 300 200 keynote.Keynote vm Inspector - Keynote\n")
FIREFOX_FRONT = "1|4242|Browser"
KEYNOTE_FRONT = "2|5151|Deck - Keynote"


class Guest:
    """``urlopen`` toward an OSWorld guest; ``answer(command)`` -> (returncode, output) per /execute call."""

    def __init__(self, answer):
        self.answer, self.commands, self.validations = answer, [], []

    def __call__(self, request, timeout=None):
        command = json.loads(request.data.decode("utf-8"))["command"]
        from tests.test_unix_computer_use_safety_contract import guest_key_validation
        validation = guest_key_validation(command[-1])
        if validation:
            self.validations.append(command)
            body = validation["result"]
        else:
            self.commands.append(command)
            returncode, output = self.answer(command[-1], len(self.commands))
            body = {"status": "success", "output": output, "error": "", "returncode": returncode}

        class Response:
            status = 200

            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return Response(json.dumps(body).encode("utf-8"))


def osworld(tmp_path, monkeypatch, answer):
    module = load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    monkeypatch.setattr(module, "_which", lambda name: "")
    api = API(tmp_path / "state")
    module.register(api)
    api.tools["add_connection"](name="vm", backend="osworld_http", target="http://guest.invalid:5000", activate=True)
    guest = Guest(answer)
    monkeypatch.setattr(urllib.request, "urlopen", guest)
    return module, api, guest


def test_osworld_targeted_input_resolves_from_the_guest_rows_activates_by_window_id_and_reads_the_front(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        return 0, f"__FRONT_BEFORE__{KEYNOTE_FRONT}\n__FRONT_AFTER__{KEYNOTE_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["observed_effect"] is True, out
    inventory, activate, action = guest.commands
    assert inventory[:2] == ["bash", "-lc"] and "wmctrl -lpxG" in inventory[2]
    assert "wmctrl -i -a 0x00000002" in activate[2] and "Keynote" not in activate[2], "activation by window id, never by name"
    assert "pyautogui.press('right')" in action[2] and "__TARGET_LOST__" in action[2] and "['2', '3']" in action[2], "the guest checks the active window against every Keynote window"
    assert out["target"]["activated"] is True and out["target"]["resolved"]["matched_by"] == "class"
    assert out["delivered_to"] == {"window_id": "0x00000002", "pid": 5151, "title": "Deck - Keynote"}
    assert out["front_before"] == {"window_id": "0x00000001", "pid": 4242, "title": "Browser", "app": "navigator.Firefox"}
    assert "__FRONT" not in json.dumps(out["execute_result"]), "the markers never reach the model as command output"


def test_osworld_refuses_when_the_guest_window_did_not_come_forward_or_does_not_exist(tmp_path, monkeypatch):
    def stays(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        return 0, "__FRONT_AFTER__" + FIREFOX_FRONT + "\n"
    module, api, guest = osworld(tmp_path, monkeypatch, stays)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "not_frontmost_after_activate" and "observed_effect" not in out
    assert len(guest.commands) == 2, "the action was never sent"
    out = receipt(api.tools["click"](x=1, y=2, raw=True, app="Emacs"))
    assert out["ok"] is False and out["reason"] == "not_running" and len(guest.commands) == 3, "no activation for an absent window"
    hinted = receipt(api.tools["click"](x=1, y=2, raw=True, app="Deck"))
    assert hinted["ok"] is False and hinted["reason"] == "not_running" and hinted["candidates"][0]["title"] == "Deck - Keynote"


def test_osworld_already_front_and_same_process_windows(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(KEYNOTE_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__3|5151|Inspector - Keynote\n"
        return 0, f"__FRONT_BEFORE__{KEYNOTE_FRONT}\n__FRONT_AFTER__{KEYNOTE_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["target"]["already_front"] is True and len(guest.commands) == 2, "no activation when already in front"
    switch = receipt(api.tools["activate_app"](app="Inspector - Keynote"))
    assert switch["ok"] is True and switch["activated"] is True and switch["observed_effect"] is True, switch
    assert switch["front"]["window_id"] == "0x00000003", "a window named by title is that window, even in the same process"
    again = receipt(api.tools["activate_app"](app="Keynote"))
    assert again["ok"] is True and again["activated"] is False and "observed_effect" not in again, "already in front: no mark"


def test_osworld_untargeted_input_names_the_front_and_tolerates_a_silent_probe(tmp_path, monkeypatch):
    def answer(code, index):
        if "pyautogui" in code:
            return 0, f"__FRONT_BEFORE__{FIREFOX_FRONT}\n__FRONT_AFTER__{FIREFOX_FRONT}\n"
        return 0, listing(FIREFOX_FRONT)
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="return"))
    assert out["ok"] is True and out["delivered_to"] == {"window_id": "0x00000001", "pid": 4242, "title": "Browser"} and "front_after" not in out
    assert len(guest.commands) == 1 and "pyautogui.press('enter')" in guest.commands[0][2], "no activation without app="
    rows = receipt(api.tools["window_list"]())
    assert rows["ok"] is True and rows["frontmost"]["pid"] == 4242 and [r["frontmost"] for r in rows["windows"]] == [True, False, False]
    # No markers at all (an older guest image): the action's exit code still governs, the place is unknown.
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, ""))
    out = receipt(api.tools["key"](keys="return"))
    assert out["ok"] is True and out["observed_effect"] is True and out["delivered_to"] == "unknown"
    empty = receipt(api.tools["window_list"]())
    assert empty["ok"] is False and empty["reason"] == "unsupported" and "no window listing" in empty["error"], "no listing marker = the script did not run"


def test_osworld_guest_precheck_withholds_an_input_when_the_target_left_the_front(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        return 3, f"__FRONT_BEFORE__{FIREFOX_FRONT}\n__TARGET_LOST__\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "target_lost", out
    assert out["observed_effect"] is True and out["activation_done"] is True, "the switch that was observed before keeps its mark"
    assert out["front"]["pid"] == 4242 and "nothing more was sent" in out["error"]


def test_osworld_window_list_failure_is_not_an_empty_desktop(tmp_path, monkeypatch):
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (127, "\n__ACTIVE__\n"))
    rows = receipt(api.tools["window_list"]())
    assert rows["ok"] is False and rows["reason"] == "unsupported" and "wmctrl" in rows["error"]
    assert receipt(api.tools["key"](keys="right", app="Keynote"))["reason"] == "unsupported"


LISTING = (' 1) "Safari" ASN:0x0-0x1: \n    bundleID="com.apple.Safari"\n    bundle path="/Applications/Safari.app"\n    pid = 100 type="Foreground" flavor=3\n'
           ' 2) "Keynote" ASN:0x0-0x2: \n    bundleID="com.apple.iWork.Keynote"\n    bundle path="/Applications/Keynote.app"\n    pid = 200 type="Foreground" flavor=3\n'
           ' 3) "Google Chrome" ASN:0x0-0x3: \n    bundleID="com.google.Chrome"\n    bundle path="/Applications/Google Chrome.app"\n    pid = 300 type="Foreground" flavor=3\n'
           ' 4) "Don\'t Sleep" ASN:0x0-0x4: \n    bundleID="com.example.dontsleep"\n    bundle path="/Applications/Don\'t Sleep.app"\n    pid = 400 type="Foreground" flavor=3\n')


def info_line(triple) -> str:
    name, pid, bid = triple
    return f'"pid"={pid} "CFBundleIdentifier"="{bid}" "LSDisplayName"="{name}" '


def ssh(tmp_path, monkeypatch, script):
    """A Mac over SSH: ``script(command)`` -> (rc, stdout[, stderr]) for every ssh call."""
    module = load_plugin()
    monkeypatch.setattr(module, "_platform", lambda: "linux")
    monkeypatch.setattr(module, "_session_type", lambda: "x11")
    monkeypatch.setattr(module, "_which", lambda name: "")
    calls = []

    def run(cmd, *a, **k):
        argv = [str(part) for part in cmd]
        calls.append(argv)
        answer = script(argv[-1])
        rc, out, err = (answer + ("",))[:3]
        return subprocess.CompletedProcess(argv, rc, out, err)
    monkeypatch.setattr(module.subprocess, "run", run)
    api = API(tmp_path / "state")
    module.register(api)
    api.tools["add_connection"](name="mac", backend="ssh_macos", host="mac.invalid", user="owner", activate=True)
    return module, api, calls


def mac_over_ssh(front, after=None, input_stderr=""):
    """Scripts: the listing reports ``front``; an activation ends with ``after`` (or ``front``) in front."""
    def script(command):
        if "lsappinfo list" in command:
            return 0, LISTING + "__FRONT__" + info_line(front)
        if "System Events" in command:
            return 0, f"__FRONT_BEFORE__{info_line(front)}\n__FRONT_AFTER__{info_line(after or front)}\n"
        if "cliclick" in command:
            now = after or front
            stage = "__CLIPBOARD_WRITTEN__\n" if "pbcopy" in command else ""
            return 0, f"__FRONT_BEFORE__{info_line(now)}\n{stage}__FRONT_AFTER__{info_line(now)}\n", input_stderr
        return 0, ""
    return script


def test_ssh_targeted_input_resolves_from_lsappinfo_and_activates_by_pid(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, KEYNOTE))
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["observed_effect"] is True, out
    assert "lsappinfo list" in calls[0][-1] and "unix id is 200" in calls[1][-1] and "to activate" not in calls[1][-1]
    assert "cliclick kp:arrow-right" in calls[2][-1] and '"$P" != "200"' in calls[2][-1], "the same ssh call checks the pid before the input"
    assert out["delivered_to"]["app"] == "Keynote" and out["target"]["activated"] is True and out["front_before"]["app"] == "Safari"
    assert out["output"] == "", "the probe markers never reach the model as command output"
    assert "error" not in out


def test_ssh_names_with_spaces_or_apostrophes_are_never_interpolated(tmp_path, monkeypatch):
    chrome = ("Google Chrome", 300, "com.google.Chrome")
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, chrome))
    out = receipt(api.tools["key"](keys="right", app="Google Chrome"))
    assert out["ok"] is True and out["delivered_to"]["app"] == "Google Chrome" and out["target"]["verified"] is True, out
    sleepy = ("Don't Sleep", 400, "com.example.dontsleep")
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, sleepy))
    out = receipt(api.tools["key"](keys="right", app="Don't Sleep"))
    assert out["ok"] is True and "unix id is 400" in calls[1][-1]
    assert all("Don't" not in argv[-1] for argv in calls), "the name itself never enters a remote command"


def test_ssh_refuses_a_not_running_app_before_any_activation(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI))
    out = receipt(api.tools["key"](keys="right", app="Xcode"))
    assert out["ok"] is False and out["reason"] == "not_running" and "Keynote" in out["running"]
    assert len(calls) == 1 and "System Events" not in calls[0][-1] and "to activate" not in calls[0][-1], "nothing is launched"
    stuck = receipt(api.tools["type_text"](text="x", app="Keynote"))
    assert stuck["ok"] is False and stuck["reason"] == "not_frontmost_after_activate" and "observed_effect" not in stuck
    assert not any("cliclick" in argv[-1] for argv in calls)


def test_ssh_already_front_sends_without_a_switch_and_marks_no_activation(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(KEYNOTE))
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is True and out["target"]["already_front"] is True and "front_before" not in out
    assert len(calls) == 2 and "cliclick" in calls[1][-1]
    again = receipt(api.tools["activate_app"](app="Keynote"))
    assert again["ok"] is True and again["activated"] is False and "observed_effect" not in again


def test_ssh_input_dropped_by_the_remote_mac_is_an_honest_failure(tmp_path, monkeypatch):
    warning = "WARNING: Accessibility privileges not enabled. Many actions may fail.\n"
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, input_stderr=warning))
    out = receipt(api.tools["key"](keys="right"))
    assert out["ok"] is False and out["reason"] == "permission_denied" and "observed_effect" not in out, out


def test_ssh_paste_writes_the_remote_clipboard(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI))
    out = receipt(api.tools["type_text"](text="Привет 'мир'", via="paste"))
    assert out["ok"] is True and out["method"] == "paste" and out["clipboard_replaced"] is True
    assert "pbcopy" in calls[-1][-1] and "cliclick kd:cmd t:v ku:cmd" in calls[-1][-1]


def run_guest_program(command: list[str], *, fronts: list[str], raise_probe: bool = False) -> tuple[list[str], str, int]:
    """Execute the python program the plugin sent to an OSWorld guest against a fake pyautogui and a fake xdotool
    (``fronts``: what each getactivewindow/pid/name triple answers, in order). Returns (actions, stdout, exit code)."""
    import io
    import types
    assert command[:2] == ["python", "-c"]
    actions: list[str] = []
    pya = types.SimpleNamespace(FAILSAFE=True, press=lambda k: actions.append(f"press {k}"),
                                hotkey=lambda *k: actions.append("hotkey " + "+".join(k)),
                                typewrite=lambda t, interval=0: actions.append(f"typewrite {t}"), click=lambda *a, **k: actions.append("click"))
    reads = list(fronts)

    class FakeRun:
        def __init__(self, out: str) -> None:
            self.stdout = out

    def fake_run(argv, **_k):
        if raise_probe:
            raise TimeoutError("xdotool hung")
        if argv[1] == "getactivewindow":
            return FakeRun(reads[0].split("|")[0] if reads else "")
        if argv[1] == "getwindowpid" and "getwindowname" in argv:
            wid, pid, title = reads[0].split("|", 2)
            assert argv[2] == wid and argv[4] == wid, argv
            if not pid:
                return FakeRun("")  # xdotool: getwindowpid fails for a window without _NET_WM_PID and ends the chain
            reads.pop(0) if len(reads) > 1 else None
            return FakeRun(pid + "\n" + title + "\n")
        if argv[1] == "getwindowname":
            wid, pid, title = reads[0].split("|", 2)
            assert argv[2] == wid and not pid, argv
            reads.pop(0) if len(reads) > 1 else None
            return FakeRun(title + "\n")
        raise AssertionError(argv)
    fake_sp = types.SimpleNamespace(run=fake_run)
    buf = io.StringIO()
    namespace = {"__name__": "__guest__"}
    real_modules = {k: sys.modules.get(k) for k in ("pyautogui", "subprocess", "platform")}
    sys.modules["pyautogui"] = pya
    sys.modules["subprocess"] = fake_sp
    import contextlib
    code = 0
    try:
        with contextlib.redirect_stdout(buf):
            try:
                exec(command[2], namespace)
            except SystemExit as exc:
                code = int(exc.code or 0)
    finally:
        for k, v in real_modules.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)
    return actions, buf.getvalue(), code


def test_the_generated_guest_program_reads_titles_safely_and_withholds_a_targeted_input(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        return 0, f"__FRONT_BEFORE__{KEYNOTE_FRONT}\n__FRONT_AFTER__{KEYNOTE_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    receipt(api.tools["key"](keys="return"))
    untargeted = guest.commands[-1]
    receipt(api.tools["key"](keys="return", app="Keynote"))
    targeted = guest.commands[-1]
    from skills.unix_computer_use import place, remote as remote_mod
    markers = lambda text: remote_mod.parse_markers(text, place.parse_x11_front)  # noqa: E731
    # A window title that looks like a marker travels base64-encoded: the markers stay whole lines and parse back exactly.
    spoof = "Doc __FRONT_AFTER__9|999|x\n__TARGET_LOST__"
    actions, out, code = run_guest_program(untargeted, fronts=[f"1|4242|{spoof}", f"1|4242|{spoof}"])
    assert actions == ["press enter"] and code == 0
    before, after, stop = markers(out)
    assert stop == "" and before["pid"] == 4242 and before["title"] == spoof and after["title"] == spoof
    # The probe cannot raise: an untargeted action still runs and the place is unknown.
    actions, out, code = run_guest_program(untargeted, fronts=[], raise_probe=True)
    assert actions == ["press enter"] and code == 0 and markers(out)[0] is None
    # Targeted: the active window is checked against the target's windows before the action; another window
    # withholds it, an unread front too; a sibling Keynote window is still Keynote.
    actions, out, code = run_guest_program(targeted, fronts=["2|5151|Deck - Keynote", "2|5151|Deck - Keynote"])
    assert actions == ["press enter"] and code == 0
    actions, out, code = run_guest_program(targeted, fronts=["3|5151|Inspector", "3|5151|Inspector"])
    assert actions == ["press enter"] and code == 0
    actions, out, code = run_guest_program(targeted, fronts=["1|4242|Browser"])
    assert actions == [] and code == 3 and markers(out)[2] == "target_lost"
    actions, out, code = run_guest_program(targeted, fronts=[], raise_probe=True)
    assert actions == [] and code == 3 and markers(out)[2] == "front_unread"


def test_osworld_an_unread_front_before_a_targeted_input_is_front_unread_not_lost(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        return 3, "\n__FRONT_BEFORE__\n__FRONT_UNREAD__\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "front_unread" and "no longer" not in out["error"], out
    assert out["activation_done"] is True, "the observed switch keeps its mark"


def test_osworld_a_withheld_paste_is_never_retyped_and_a_real_clipboard_failure_reuses_the_switch(tmp_path, monkeypatch):
    def withheld(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        return 3, f"\n__FRONT_BEFORE__{FIREFOX_FRONT}\n__TARGET_LOST__\n"
    module, api, guest = osworld(tmp_path, monkeypatch, withheld)
    out = receipt(api.tools["type_text"](text="<b>Привет</b>\nrm -rf build", app="Keynote"))
    assert out["ok"] is False and out["reason"] == "target_lost" and len(guest.commands) == 3, "no typewrite fallback after a refusal"
    assert not any("typewrite" in c[-1] for c in guest.commands)

    def clipboard_broken(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        if "pyperclip" in code:
            return 1, f"\n__FRONT_BEFORE__{KEYNOTE_FRONT}\nno xclip\n"
        return 0, f"\n__FRONT_BEFORE__{KEYNOTE_FRONT}\n__FRONT_AFTER__{KEYNOTE_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, clipboard_broken)
    out = receipt(api.tools["type_text"](text="<b>x</b>", app="Keynote"))
    assert out["ok"] is True and out["method"] == "typewrite" and out["clipboard_fallback"] is True
    assert sum("wmctrl -i -a" in c[-1] for c in guest.commands) == 1, "one switch for the whole call"


def test_osworld_activation_with_an_unread_after_read_is_front_unread_and_reports_its_wait(tmp_path, monkeypatch):
    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "\n__FRONT_AFTER__\n"
        return 0, ""
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["activate_app"](app="Keynote"))
    assert out["ok"] is False and out["reason"] == "front_unread" and "observed_effect" not in out, out
    good = osworld(tmp_path, monkeypatch, lambda code, index: (0, listing(FIREFOX_FRONT)) if "wmctrl -lpxG" in code
                   else (0, "\n__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"))
    switched = receipt(good[1].tools["activate_app"](app="Keynote"))
    assert switched["ok"] is True and isinstance(switched["target"]["waited_ms"], int)


def test_ssh_stage_facts_survive_a_transport_failure_and_an_empty_inventory_is_unread(tmp_path, monkeypatch):
    def script(command):
        if "lsappinfo list" in command:
            return 0, LISTING + "__FRONT__" + info_line(SAFARI)
        if "System Events" in command:
            return 0, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__FRONT_AFTER__{info_line(KEYNOTE)}\n"
        raise subprocess.TimeoutExpired(command, 30)
    module, api, calls = ssh(tmp_path, monkeypatch, script)
    out = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert out["ok"] is False and "TimeoutExpired" in out["error"], out
    assert out["observed_effect"] is True and out["activation_done"] is True and out["target"]["activated"] is True
    module, api, calls = ssh(tmp_path, monkeypatch, lambda command: (0, "__FRONT__" + info_line(SAFARI)))
    empty = receipt(api.tools["key"](keys="right", app="Keynote"))
    assert empty["ok"] is False and empty["reason"] == "apps_unread" and len(calls) == 1
    module, api, calls = ssh(tmp_path, monkeypatch, lambda command: (127, "", "lsappinfo: command not found"))
    assert receipt(api.tools["window_list"]())["reason"] == "apps_unread"
    # A snippet cliclick cannot express is refused before any switch.
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI, KEYNOTE))
    refused = receipt(api.tools["scroll"](clicks=2, app="Keynote"))
    assert refused["ok"] is False and calls == [], "nothing reached the Mac, no switch for an input that cannot be sent"


def test_ssh_paste_reports_the_clipboard_stage_truthfully(tmp_path, monkeypatch):
    def pbcopy_failed(command):
        if "cliclick" in command or "pbcopy" in command:
            return 1, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__FRONT_AFTER__{info_line(SAFARI)}\n", "pbcopy: not found"
        return 0, ""
    module, api, calls = ssh(tmp_path, monkeypatch, pbcopy_failed)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["ok"] is False and out["clipboard_replaced"] is False and "observed_effect" not in out

    def chord_failed(command):
        return 1, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__CLIPBOARD_WRITTEN__\n__FRONT_AFTER__{info_line(SAFARI)}\n", "cliclick died"
    module, api, calls = ssh(tmp_path, monkeypatch, chord_failed)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["ok"] is False and out["clipboard_replaced"] is True and "__CLIPBOARD" not in out["output"]


# --- cycle 3: encoded inventory, clipboard effects, dispatched facts, SSH negatives ------------------

def test_a_marker_shaped_title_in_the_guest_inventory_forges_nothing(tmp_path, monkeypatch):
    rows = ("0x00000002  0 5151 0 0 1024 768 keynote.Keynote vm Draft __ACTIVE__1|100|b64:QnJvd3Nlcg==\n"
            "0x00000001  0 4242 10 20 800 600 navigator.Firefox vm Browser\n")
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, listing("2|5151|b64:RHJhZnQ=", rows)))
    out = receipt(api.tools["window_list"]())
    assert out["ok"] is True and out["frontmost"]["window_id"] == "0x00000002"
    assert [r["title"] for r in out["windows"]][0] == "Draft __ACTIVE__1|100|b64:QnJvd3Nlcg==", "the title is data"
    again = receipt(api.tools["activate_app"](app="Keynote"))
    assert again["ok"] is True and again["activated"] is False and "observed_effect" not in again, "already in front: no invented switch"


def test_ssh_a_failed_paste_chord_keeps_the_clipboard_effect_without_a_switch(tmp_path, monkeypatch):
    def chord_failed(command):
        return 1, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__CLIPBOARD_WRITTEN__\n__FRONT_AFTER__{info_line(SAFARI)}\n", "cliclick died"
    module, api, calls = ssh(tmp_path, monkeypatch, chord_failed)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["ok"] is False and out["clipboard_replaced"] is True and out["observed_effect"] is True and out["clipboard_done"] is True
    assert out["dispatched"] is True and "activation_done" not in out


def test_ssh_window_list_says_when_the_inventory_or_the_front_is_unread(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, lambda command: (0, "__FRONT__" + info_line(SAFARI)))
    empty = receipt(api.tools["window_list"]())
    assert empty["ok"] is False and empty["reason"] == "apps_unread"
    module, api, calls = ssh(tmp_path, monkeypatch, lambda command: (0, LISTING + "__FRONT__"))
    rows = receipt(api.tools["window_list"]())
    assert rows["ok"] is True and rows["frontmost"] is None and "front_unread" in rows and all("frontmost" not in r for r in rows["windows"])


def test_ssh_negative_coordinates_are_absolute_points_on_the_remote_mac(tmp_path, monkeypatch):
    module, api, calls = ssh(tmp_path, monkeypatch, mac_over_ssh(SAFARI))
    out = receipt(api.tools["click"](x=-250, y=200, raw=True))
    assert out["ok"] is True and "cliclick c:=-250,200" in calls[-1][-1]


def test_osworld_a_lost_answer_after_a_paste_is_not_retyped_and_negative_points_are_refused_for_a_guest(tmp_path, monkeypatch):
    def lost(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        raise TimeoutError("the guest did not answer")
    module, api, guest = osworld(tmp_path, monkeypatch, lost)
    out = receipt(api.tools["type_text"](text="<b>x</b>", app="Keynote"))
    assert out["ok"] is False and out["transport_failed"] is True and "unknown" in out["note"], out
    assert not any("typewrite" in c[-1] for c in guest.commands), "a paste whose answer was lost may have run: never retyped"
    assert out["activation_done"] is True
    # A guest desktop has no screen space left of or above its origin: a negative point is refused before any call.
    guest.commands.clear()
    refused = receipt(api.tools["click"](x=-5, y=20, raw=True))
    assert refused["ok"] is False and "negative" in refused["error"] and guest.commands == []
    for tool in ("mouse_down", "mouse_up"):
        pressed = receipt(api.tools[tool](x=-5, y=20, raw=True))
        assert pressed["ok"] is False and "negative" in pressed["error"] and guest.commands == [], (tool, pressed)
    monkeypatch.setattr(module, "_platform", lambda: "macos")  # the host's own space never decides for a guest
    assert receipt(api.tools["click"](x=-5, y=20, raw=True))["ok"] is False and guest.commands == []
    # A stored transform from another display or backend that maps a point outside the guest is refused the same way.
    impl = api.tools["click"].__self__
    geometry = {"x": 0, "y": 0, "w": 1920, "h": 1080}
    monkeypatch.setattr(impl, "_display_geometry", lambda _tf, **_kw: geometry)
    impl._save_transform({"sx": 1.0, "sy": 1.0, "ox": -1920, "oy": 0, "ts": 1, "platform": "macos",
                          "observation_id": "other-display", "display_geometry": geometry, "image_w": 1920, "image_h": 1080,
                          "connection_identity": impl._connection_identity(), "backend": impl._active_backend_name()})
    mapped = receipt(api.tools["click"](x=100, y=100, observation_id="other-display"))
    assert mapped["ok"] is False and "stored screenshot transform" in mapped["error"] and guest.commands == [], mapped
    for tool, args in (("move", {"x": 100, "y": 100}), ("left_click_drag", {"start_x": 100, "start_y": 100, "end_x": 200, "end_y": 200}),
                       ("mouse_down", {"x": 100, "y": 100}), ("mouse_up", {"x": 100, "y": 100})):
        out = receipt(api.tools[tool](**args, observation_id="other-display"))
        assert out["ok"] is False and "stored screenshot transform" in out["error"] and guest.commands == [], (tool, out)
    # Untargeted: a lost answer after the clipboard program is final here too (the base retyped); a lost read carries no dispatch fact.
    plain = receipt(api.tools["type_text"](text="<b>x</b>"))
    assert plain["ok"] is False and plain["transport_failed"] is True and plain["dispatched"] is None and plain["method"] == "clipboard", plain
    assert "clipboard_replaced" not in plain and not any("typewrite" in c[-1] for c in guest.commands)
    pos = receipt(api.tools["cursor_position"]())
    assert pos["ok"] is False and pos["transport_failed"] is True and "dispatched" not in pos and "note" not in pos, pos


def test_osworld_a_pid_less_window_keeps_its_title_in_the_probe(tmp_path, monkeypatch):
    """xdotool ends a command chain at its first failure, so a window without _NET_WM_PID gets its name read alone."""
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, f"__FRONT_BEFORE__{FIREFOX_FRONT}\n__FRONT_AFTER__{FIREFOX_FRONT}\n"))
    receipt(api.tools["key"](keys="return"))
    from skills.unix_computer_use import place, remote as remote_mod
    actions, out, code = run_guest_program(guest.commands[-1], fronts=["7||Old Java App", "7||Old Java App"])
    before, after, stop = remote_mod.parse_markers(out, place.parse_x11_front)
    assert actions == ["press enter"] and code == 0 and stop == "", out
    assert before == {"window_id": "0x00000007", "title": "Old Java App"} and after == before, out


@pytest.mark.skipif(not (shutil.which("bash") and shutil.which("base64")), reason="the guest shell reader runs under bash")
def test_the_guest_shell_reader_encodes_the_exact_title_and_survives_a_pid_less_window(tmp_path):
    """The shell reader (activation, listing, capabilities) and the python probe agree on a title: no trailing newline."""
    from skills.unix_computer_use import place, remote as remote_mod
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "xdotool"
    fake.write_text("\n".join(["#!/bin/sh", 'case "$1" in', "  getactivewindow) echo 7 ;;",
                                '  getwindowpid) [ -n "$XPID" ] && echo "$XPID" || exit 1 ;;',
                                "  getwindowname) printf '%s\\n' \"$XTITLE\" ;;", "esac", ""]),
                    encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    # The bash the PATH names: a bare "bash" makes Windows look in System32 first, which starts WSL.
    bash = shutil.which("bash")
    env = {**os.environ, "PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]),
           "XTITLE": "Draft __ACTIVE__ - Keynote", "XPID": "5151"}
    out = subprocess.run([bash, "-c", remote_mod._X11_FRONT_SH], capture_output=True, text=True, env=env, timeout=30).stdout
    assert place.parse_x11_front(out) == {"window_id": "0x00000007", "pid": 5151, "title": "Draft __ACTIVE__ - Keynote"}, out
    env["XPID"] = ""
    out = subprocess.run([bash, "-c", remote_mod._X11_FRONT_SH], capture_output=True, text=True, env=env, timeout=30).stdout
    assert place.parse_x11_front(out) == {"window_id": "0x00000007", "title": "Draft __ACTIVE__ - Keynote"}, out


def test_ssh_a_failed_clipboard_write_sends_nothing_and_says_so(tmp_path, monkeypatch):
    def write_failed(command):
        if "pbcopy" in command:  # the `&&` chain stops at pbcopy: no marker, no chord
            return 1, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__FRONT_AFTER__{info_line(SAFARI)}\n", "pbcopy: command not found"
        return mac_over_ssh(SAFARI)(command)
    module, api, calls = ssh(tmp_path, monkeypatch, write_failed)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["ok"] is False and out["reason"] == "clipboard_failed" and out["dispatched"] is False, out
    assert out["clipboard_replaced"] is False and "observed_effect" not in out and "not sent" in out["note"]


def test_osworld_a_lost_reply_after_a_key_is_an_unknown_outcome_never_a_withheld_input(tmp_path, monkeypatch):
    """The guest program may have pressed the key before the host's read died: the aggregate says so and withholds the rest."""
    def lost(code, index):
        if "pyautogui.press" in code:
            raise TimeoutError("reply lost after guest input")
        return 0, f"__FRONT_BEFORE__{FIREFOX_FRONT}\n__FRONT_AFTER__{FIREFOX_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, lost)
    out = receipt(api.tools["key"](keys="right down"))
    assert out["ok"] is False and out["dispatched"] is None and out["transport_failed"] is True and out["steps_done"] == 0, out
    assert "outcome unknown" in out["error"] and "not sent" not in out["error"]
    assert "did not answer" in out["note"] and "not sent" not in out["note"], out["note"]
    assert sum("pyautogui.press" in c[-1] for c in guest.commands) == 1, "the second step is withheld after an unknown outcome"
    # The second step's answer lost after the first ran: the aggregate says which step is unknown, never "not sent".
    presses = {"n": 0}

    def lost_second(code, index):
        if "pyautogui.press" in code:
            presses["n"] += 1
            if presses["n"] == 2:
                raise TimeoutError("reply lost after guest input")
        return 0, f"__FRONT_BEFORE__{FIREFOX_FRONT}\n__FRONT_AFTER__{FIREFOX_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, lost_second)
    out = receipt(api.tools["key"](keys="right down return"))
    assert out["steps_done"] == 1 and out["dispatched"] is None and presses["n"] == 2, out
    head, _, _rest = out["note"].partition("the rest")
    assert head.startswith("1 of 3 step(s) completed") and "did not answer" in head and "not sent" not in head, out["note"]


def test_osworld_a_login_shell_greeting_before_the_listing_hides_nothing(tmp_path, monkeypatch):
    """`bash -lc` may print a greeting before the script runs: the listing is the last line before the separator."""
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, "Welcome to the guest\n" + listing(FIREFOX_FRONT)))
    rows = receipt(api.tools["window_list"]())
    assert rows["ok"] is True and len(rows["windows"]) == 3 and rows["frontmost"]["window_id"] == "0x00000001", rows
    module, api, guest = osworld(tmp_path, monkeypatch, lambda code, index: (0, "not base64!\n__ACTIVE__\n" + FIREFOX_FRONT + "\n"))
    bad = receipt(api.tools["window_list"]())
    assert bad["ok"] is False and "decode" in bad["error"], bad


def test_ssh_an_application_name_containing_the_front_marker_is_data(tmp_path, monkeypatch):
    rows = LISTING + ' 5) "My __FRONT__ Tool" ASN:0x0-0x5: \n    bundleID="com.example.front"\n    pid = 500 type="Foreground" flavor=3\n'
    module, api, calls = ssh(tmp_path, monkeypatch, lambda command: (0, rows + "\n__FRONT__\n" + info_line(SAFARI)))
    out = receipt(api.tools["window_list"]())
    assert out["ok"] is True and [r["app"] for r in out["windows"]][-1] == "My __FRONT__ Tool" and out["frontmost"]["pid"] == 100, out


def test_ssh_a_reply_cut_short_is_an_unknown_outcome_never_a_withheld_input(tmp_path, monkeypatch):
    """ssh exit 255 without the wrapper's completion marker: a refused connection or a channel broken after the input
    may have run; the receipt says unknown, keeps a received clipboard fact, and never says "not sent"."""
    def refused(command):
        if "cliclick" in command:  # nothing of the wrapper's output came back
            return 255, "", "ssh: connect to host mac.invalid port 22: Connection refused"
        return mac_over_ssh(SAFARI, KEYNOTE)(command)  # the switch itself succeeds
    module, api, calls = ssh(tmp_path, monkeypatch, refused)
    out = receipt(api.tools["key"](keys="right"))
    assert out["ok"] is False and out["dispatched"] is None and out["transport_failed"] is True and out["reason"] == "ssh_failed", out
    assert "unknown" in out["note"] and "not sent" not in out["note"] and "Connection refused" in out["error"] and "observed_effect" not in out
    ran = receipt(api.tools["key"](keys="right", app="Keynote"))  # a switch that ran before it keeps its mark
    assert ran["dispatched"] is None and ran["activation_done"] is True and ran["observed_effect"] is True, ran

    paste = receipt(api.tools["type_text"](text="x", via="paste"))  # the same refusal on a paste: never a clipboard failure
    assert paste["dispatched"] is None and paste["reason"] == "ssh_failed" and "clipboard_replaced" not in paste and paste["method"] == "paste", paste

    def timed_out(command):  # the ssh call itself raised: the paste's method is kept, its clipboard stage makes no claim
        if "cliclick" in command:
            raise subprocess.TimeoutExpired(command, 60)
        return mac_over_ssh(SAFARI)(command)
    module, api, calls = ssh(tmp_path, monkeypatch, timed_out)
    lost = receipt(api.tools["type_text"](text="x", via="paste"))
    assert lost["transport_failed"] is True and lost["dispatched"] is None and lost["method"] == "paste" and "clipboard_replaced" not in lost, lost

    def cut_short(command):  # the input stage had started, then the channel broke: the clipboard stage is unknown
        if "cliclick" in command:
            return 255, f"__FRONT_BEFORE__{info_line(SAFARI)}\n", "client_loop: send disconnect: Broken pipe"
        return mac_over_ssh(SAFARI)(command)
    module, api, calls = ssh(tmp_path, monkeypatch, cut_short)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["dispatched"] is None and out["reason"] == "ssh_failed" and "clipboard_replaced" not in out and "observed_effect" not in out, out
    assert "started" in out["error"] and "unknown" in out["note"] and "not sent" not in out["note"]

    def cut_after_write(command):  # the clipboard marker came back before the break: a fact that is kept
        if "cliclick" in command:
            return 255, f"__FRONT_BEFORE__{info_line(SAFARI)}\n__CLIPBOARD_WRITTEN__\n", "client_loop: send disconnect: Broken pipe"
        return mac_over_ssh(SAFARI)(command)
    module, api, calls = ssh(tmp_path, monkeypatch, cut_after_write)
    out = receipt(api.tools["type_text"](text="x", via="paste"))
    assert out["dispatched"] is None and out["clipboard_replaced"] is True and out["clipboard_done"] is True and out["observed_effect"] is True, out


def test_osworld_an_unread_front_before_a_later_step_says_so(tmp_path, monkeypatch):
    """The remote pre-check of a later step that cannot read the front withholds it and says which step, never 'the switch was requested'."""
    presses = {"n": 0}

    def answer(code, index):
        if "wmctrl -lpxG" in code:
            return 0, listing(FIREFOX_FRONT)
        if "wmctrl -i -a" in code:
            return 0, "__FRONT_AFTER__" + KEYNOTE_FRONT + "\n"
        presses["n"] += 1
        if presses["n"] == 2:
            return 3, "\n__FRONT_BEFORE__\n__FRONT_UNREAD__\n"
        return 0, f"__FRONT_BEFORE__{KEYNOTE_FRONT}\n__FRONT_AFTER__{KEYNOTE_FRONT}\n"
    module, api, guest = osworld(tmp_path, monkeypatch, answer)
    out = receipt(api.tools["key"](keys="right down", app="Keynote"))
    assert out["ok"] is False and out["steps_done"] == 1 and out["reason"] == "front_unread" and out["dispatched"] is False, out
    assert "before the next step" in out["error"] and "nothing more was sent" in out["error"] and "switch was requested" not in out["error"], out["error"]
