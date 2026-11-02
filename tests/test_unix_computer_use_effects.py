"""unix_computer_use input receipts: ``observed_effect`` is the producer's own fact (#1919).

A successful input call carries ``observed_effect: true``: the input command ran and reported
success. That is dispatch strength, not proof that the screen changed or that the task's goal was
met (macOS cliclick exits 0 even under a denied Accessibility grant). A call that fails after
earlier steps ran keeps ``ok: false`` with its diagnostics and still carries the mark for what ran.
Reads, waits, refusals, a first failed process and an unknown timeout never carry it, nor does
``remote_exec``. Every call goes through the handler the skill registers; only the OS and network
boundary (``subprocess.run``, ``urlopen``) is faked, so no desktop, SSH or guest is touched.
"""
from __future__ import annotations

import importlib.util
import sys
import io
import json
import subprocess
import urllib.request
from pathlib import Path

import pytest

PLUGIN = Path(__file__).resolve().parents[1] / "skills" / "unix_computer_use" / "plugin.py"
TIMEOUT = subprocess.TimeoutExpired(["fixture"], 10)


def load_plugin():
    spec = importlib.util.spec_from_file_location("unix_computer_use_effects_plugin", PLUGIN, submodule_search_locations=[str(PLUGIN.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class API:
    def __init__(self, state: Path) -> None:
        self.state, self.tools = state, {}

    def get_state_dir(self) -> str:
        return str(self.state)

    def skill_job_dir(self, job_id: str) -> Path:
        path = self.state / "jobs" / job_id
        (path / "output").mkdir(parents=True, exist_ok=True)
        return path

    def register_tool(self, name, handler, **_metadata):
        self.tools[name] = handler


def png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (64, 40), (7, 11, 13)).save(path, format="PNG")


def is_probe(argv: list[str]) -> bool:
    """A place fact the skill reads around an input (who is in front, the native helper, a window listing)."""
    return (argv[:1] == ["lsappinfo"] or argv[:2] == ["xdotool", "getactivewindow"] or argv[:2] == ["wmctrl", "-lpxG"]
            or any(str(part).endswith("macos_native.py") for part in argv))


class Transport:
    """``subprocess.run`` as the skill sees it. Every argv is recorded; ``answer(argv, index)`` returns
    a return code, ``(code, stdout)``, or an exception to raise. Screen captures get a real PNG."""

    def __init__(self, answer=None):
        self.calls: list[list[str]] = []   # the inputs and reads a test pins
        self.probes: list[list[str]] = []  # front reads, the native helper, window listings: the receipt's place facts
        self.answer = answer or (lambda argv, index: 0)

    def __call__(self, cmd, *args, **kwargs):
        argv = [str(part) for part in cmd]
        if is_probe(argv):
            self.probes.append(argv)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        self.calls.append(argv)
        outcome = self.answer(argv, len(self.calls) - 1)
        if isinstance(outcome, BaseException):
            raise outcome
        code, stdout = outcome if isinstance(outcome, tuple) else (outcome, "")
        if code == 0 and argv[0] in {"screencapture", "scp"}:
            png(Path(argv[-1]))
        return subprocess.CompletedProcess(cmd, code, stdout, "" if code == 0 else f"exit status {code} from {argv[0]}")


def fail_at(*indexes, outcome=1):
    """Fail the calls with these indexes (``outcome``: a code or an exception); everything else succeeds."""
    return lambda argv, index: outcome if index in indexes else 0


def desktop(tmp_path, monkeypatch, platform, session="native", tools=(), answer=None):
    """The registered handlers of a local desktop: the platform, its installed tools and its process boundary."""
    module = load_plugin()
    # Name lookup is an OS boundary, like this fixture's simulated xdotool process.
    monkeypatch.setattr(module._ComputerUse, "_x11_keysym", staticmethod(lambda _name: None))
    transport = Transport(answer)
    monkeypatch.setattr(module, "_platform", lambda: platform)
    monkeypatch.setattr(module, "_session_type", lambda: session)
    monkeypatch.setattr(module, "_which", lambda name: f"/usr/bin/{name}" if name in tools else "")
    monkeypatch.setattr(module.subprocess, "run", transport)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    api = API(tmp_path / "state")
    module.register(api)
    return api.tools, transport


def ssh_mac(tmp_path, monkeypatch, answer=None):
    """The registered handlers driving a Mac over SSH; every process (ssh, scp) is the fake boundary."""
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=(), answer=answer)
    assert json.loads(tools["add_connection"](name="mac", backend="ssh_macos", host="mac.invalid",
                                              user="owner", activate=True))["ok"] is True
    return tools, transport


def receipt(text: str) -> dict:
    payload = json.loads(text)
    assert isinstance(payload, dict)
    return payload


def marked(payload: dict) -> bool:
    return payload.get("observed_effect") is True


# --- local macOS (cliclick) -----------------------------------------------------------------------

MAC = ("cliclick", "osascript", "screencapture")


@pytest.mark.parametrize("tool, args, argv", [
    ("click", {"x": 10, "y": 20, "raw": True}, ["cliclick", "c:10,20"]),
    ("double_click", {"x": 10, "y": 20, "raw": True}, ["cliclick", "dc:10,20"]),
    ("move", {"x": 5, "y": 6, "raw": True}, ["cliclick", "m:5,6"]),
    ("left_click_drag", {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "raw": True},
     ["cliclick", "dd:1,2", "dm:3,4", "du:3,4"]),
    ("mouse_down", {"x": 5, "y": 6, "raw": True}, ["cliclick", "dd:5,6"]),
    ("mouse_up", {}, ["cliclick", "du:."]),
    ("type_text", {"text": "hello"}, ["cliclick", "t:hello"]),
    ("key", {"keys": "cmd+s"}, ["cliclick", "kd:cmd", "t:s", "ku:cmd"]),
    ("hold_key", {"keys": "shift", "duration_ms": 200}, ["cliclick", "kd:shift", "w:200", "ku:shift"]),
])
def test_a_successful_mac_input_carries_the_mark(tmp_path, monkeypatch, tool, args, argv):
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC)
    result = receipt(tools[tool](**args))
    assert result["ok"] is True and result["observed_effect"] is True, result
    assert transport.calls == [argv] and result["cmd"] == argv, "the receipt still names what ran"


@pytest.mark.parametrize("outcome", [1, TIMEOUT])
def test_a_failed_or_timed_out_single_process_carries_no_mark(tmp_path, monkeypatch, outcome):
    """One process whose outcome is unknown (a nonzero exit of a multi-argument cliclick, or a timeout)."""
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC, answer=fail_at(0, outcome=outcome))
    for tool, args in (("click", {"x": 1, "y": 2, "raw": True}), ("key", {"keys": "cmd+s"}),
                       ("type_text", {"text": "x"})):
        result = receipt(tools[tool](**args))
        assert result["ok"] is False and not marked(result), result
        assert result["error"] and result["cmd"][0] == "cliclick", "the diagnostics stay"
        transport.calls.clear()
        transport.answer = fail_at(0, outcome=outcome)


def test_mac_refusals_reads_and_waits_carry_no_mark(tmp_path, monkeypatch):
    def answer(argv, _index):
        if argv[:2] == ["cliclick", "p"]:
            return 0, "12,34\n"
        if argv[0] == "osascript":
            return 0, ("0, 0, 1440, 900" if "bounds" in argv[-1] else "PROC\tFinder\nWIN\tDesktop")
        return 0
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC, answer=answer)
    refusals = [tools["scroll"](clicks=3), tools["click"](x=1, y=1, button="middle"),
                tools["hold_key"](keys="a"), tools["mouse_down"](x=1, y=1, button="right")]
    assert transport.calls == [], "a refusal runs nothing"
    reads = [tools["cursor_position"](), tools["window_list"](), tools["ax_tree"](), tools["capabilities"](),
             tools["screenshot"](), tools["wait"](ms=1), tools["list_connections"](), tools["test_connection"](),
             tools["remote_exec"](command="touch /tmp/x")]
    for text in refusals:
        assert receipt(text)["ok"] is False and not marked(receipt(text)), text
    for text in reads:
        assert not marked(receipt(text)), text
    assert receipt(reads[0])["ok"] is True and receipt(reads[4])["ok"] is True, "the reads really answered"
    assert {argv[0] for argv in transport.calls} == {"cliclick", "osascript", "screencapture"}
    assert any(str(argv[-2]).endswith("macos_native.py") for argv in transport.probes), "window_list/ax_tree ask the native helper"


def test_a_mac_key_sequence_keeps_the_mark_of_the_steps_that_ran(tmp_path, monkeypatch):
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC)
    result = receipt(tools["key"](keys="cmd+a cmd+c"))
    assert result["ok"] is True and result["steps"] == 2 and marked(result), result

    transport.answer = fail_at(1)
    transport.calls.clear()
    result = receipt(tools["key"](keys="cmd+a cmd+c"))
    assert result["ok"] is False and marked(result), "cmd+a ran before cmd+c failed"
    assert (result["steps_done"], result["steps_total"]) == (1, 2) and "cmd+c" in result["error"], result

    transport.answer = fail_at(0)
    transport.calls.clear()
    result = receipt(tools["key"](keys="cmd+a cmd+c"))
    assert result["ok"] is False and not marked(result) and result["steps_done"] == 0, result
    assert len(transport.calls) == 1, "nothing ran after the first failure"

    transport.answer = fail_at()
    transport.calls.clear()
    result = receipt(tools["key"](keys="cmd+a bogusmod+x"))
    assert result["ok"] is False and not marked(result) and "bogusmod" in result["error"], result
    assert transport.calls == [], "prevalidation refuses the whole sequence before its first input"


# --- local Linux: X11 (xdotool) and Wayland (ydotool) ----------------------------------------------

@pytest.mark.parametrize("tool, args", [
    ("click", {"x": 3, "y": 4, "raw": True}),
    ("left_click_drag", {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "raw": True}),
    ("mouse_up", {"button": "right"}),
    ("type_text", {"text": "-n hello"}),
    ("key", {"keys": "ctrl+l"}),
    ("scroll", {"clicks": 2, "direction": "down"}),
    ("hold_key", {"keys": "ctrl", "duration_ms": 100}),
])
def test_a_successful_x11_input_carries_the_mark(tmp_path, monkeypatch, tool, args):
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=("xdotool", "wmctrl"))
    result = receipt(tools[tool](**args))
    assert result["ok"] is True and marked(result), result
    assert transport.calls and all(argv[0] == "xdotool" for argv in transport.calls)


def test_an_x11_hold_keeps_the_mark_when_only_the_release_fails(tmp_path, monkeypatch):
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=("xdotool",), answer=fail_at(1))
    result = receipt(tools["hold_key"](keys="ctrl", duration_ms=100))
    assert result["ok"] is False and marked(result), "ctrl went down; its release failed"
    assert result["cmd"] == ["xdotool", "keyup", "ctrl"] and result["error"], result
    assert (result["steps_done"], result["steps_total"]) == (1, 2)

    for outcome in (1, TIMEOUT):
        transport.calls.clear()
        transport.answer = fail_at(0, outcome=outcome)
        result = receipt(tools["hold_key"](keys="ctrl", duration_ms=100))
        assert result["ok"] is False and not marked(result) and result["steps_done"] == 0, result
        assert transport.calls == [["xdotool", "keydown", "ctrl"]], "no release is sent for a press that failed"


def test_x11_reads_and_refusals_carry_no_mark(tmp_path, monkeypatch):
    def answer(argv, _index):
        return (0, "x:5 y:6 screen:0 window:1") if argv[1:2] == ["getmouselocation"] else (0, "0x01 host Editor\n")
    tools, _transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=("xdotool", "wmctrl"), answer=answer)
    for text in (tools["cursor_position"](), tools["window_list"](), tools["ax_tree"](),
                 tools["scroll"](direction="sideways"), tools["key"](keys="hyper+x")):
        assert not marked(receipt(text)), text


def test_a_multi_process_input_reports_which_step_ran_and_a_helper_that_never_started(tmp_path, monkeypatch):
    """`dispatched` is the fact that a helper ran: a failing later step is dispatched (its outcome unknown), a
    failing first step too, and a helper the OS could not start is not."""
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",), answer=fail_at(1))
    out = receipt(tools["click"](x=10, y=20, raw=True))
    assert out["ok"] is False and out["steps_done"] == 1 and out["dispatched"] is True and out["observed_effect"] is True, out
    assert "dispatched and did not report success" in out["note"] and "not sent" not in out["error"]
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",), answer=fail_at(0))
    first = receipt(tools["click"](x=10, y=20, raw=True))
    assert first["ok"] is False and first["steps_done"] == 0 and first["dispatched"] is True and "unknown" in first["note"], first
    assert "not sent" not in first["note"] and "observed_effect" not in first
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=("xdotool",), answer=fail_at(1))
    held = receipt(tools["hold_key"](keys="a", duration_ms=50))
    assert held["ok"] is False and held["steps_done"] == 1 and held["dispatched"] is True and "dispatched and did not report success" in held["note"], held
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",))
    assert receipt(tools["click"](x=10, y=20, raw=True))["dispatched"] is True, "a successful multi-process input says its helpers ran"
    missing = FileNotFoundError(2, "No such file or directory", "cliclick")
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC, answer=fail_at(0, outcome=missing))
    never = receipt(tools["click"](x=10, y=20, raw=True))
    assert never["ok"] is False and never["dispatched"] is False and never["reason"] == "helper_not_launched", never
    assert "not sent" in never["note"] and "returncode" not in never and "observed_effect" not in never
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",), answer=fail_at(1, outcome=missing))
    step = receipt(tools["click"](x=10, y=20, raw=True))
    assert step["dispatched"] is False and step["steps_done"] == 1 and "the next step was not sent" in step["note"], step


def test_a_press_at_a_negative_point_is_refused_where_the_desktop_has_no_such_space(tmp_path, monkeypatch):
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "x11", tools=("xdotool",))
    for tool in ("mouse_down", "mouse_up"):
        out = receipt(tools[tool](x=-5, y=20, raw=True))
        assert out["ok"] is False and "negative" in out["error"] and transport.calls == [], (tool, out)
    assert receipt(tools["mouse_down"](x=5, y=6, raw=True))["ok"] is True and transport.calls[-1] == ["xdotool", "mousemove", "5", "6", "mousedown", "1"]
    assert receipt(tools["mouse_up"]())["ok"] is True, "omitted coordinates still release at the pointer"


@pytest.mark.parametrize("tool, args, steps", [
    ("click", {"x": 3, "y": 4, "raw": True}, 2),
    ("left_click_drag", {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "raw": True}, 4),
    ("mouse_down", {"x": 5, "y": 6, "raw": True}, 2),
    ("mouse_up", {}, 1),
])
def test_a_wayland_sequence_marks_what_ran(tmp_path, monkeypatch, tool, args, steps):
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",))
    result = receipt(tools[tool](**args))
    assert result["ok"] is True and marked(result) and len(transport.calls) == steps, result

    for failing in range(steps):
        for outcome in (1, TIMEOUT):
            transport.calls.clear()
            transport.answer = fail_at(failing, outcome=outcome)
            result = receipt(tools[tool](**args))
            assert result["ok"] is False and result["error"], result
            assert result["steps_done"] == failing and result["steps_total"] == steps, result
            assert result["cmd"] == transport.calls[-1] and len(transport.calls) == failing + 1
            assert marked(result) is (failing > 0), (tool, failing, outcome, result)


def test_wayland_text_scroll_and_refusals(tmp_path, monkeypatch):
    tools, transport = desktop(tmp_path, monkeypatch, "linux", "wayland", tools=("ydotool",))
    for tool, args in (("type_text", {"text": "hi"}), ("scroll", {"clicks": 1, "direction": "up"})):
        assert marked(receipt(tools[tool](**args))), tool
    transport.calls.clear()
    for text in (tools["key"](keys="ctrl+l"), tools["hold_key"](keys="ctrl"),
                 tools["scroll"](direction="left"), tools["cursor_position"]()):
        assert receipt(text)["ok"] is False and not marked(receipt(text)), text
    assert transport.calls == []


def test_an_empty_sequence_is_no_effect(tmp_path, monkeypatch):
    module = load_plugin()
    monkeypatch.setattr(module.subprocess, "run", Transport())
    impl = module._ComputerUse(API(tmp_path / "state"))
    for text in (impl._exec_sequence([]), impl._hold_then_release([], [], 50, ctx={})):
        assert receipt(text)["ok"] is True and not marked(receipt(text)), text


# --- SSH Mac --------------------------------------------------------------------------------------

def remote(argv: list[str]) -> str:
    assert argv[0] == "ssh" and "BatchMode=yes" in argv, argv
    return argv[-1]


@pytest.mark.parametrize("tool, args, command", [
    ("click", {"x": 10, "y": 20, "raw": True}, "cliclick c:10,20"),
    ("triple_click", {"x": 10, "y": 20, "raw": True}, "cliclick tc:10,20"),
    ("move", {"x": 5, "y": 6, "raw": True}, "cliclick m:5,6"),
    ("left_click_drag", {"start_x": 1, "start_y": 2, "end_x": 3, "end_y": 4, "raw": True},
     "cliclick dd:1,2 dm:3,4 du:3,4"),
    ("mouse_down", {"x": 5, "y": 6, "raw": True}, "cliclick dd:5,6"),
    ("type_text", {"text": "hello"}, "cliclick t:hello"),
    ("key", {"keys": "return"}, "cliclick kp:return"),
    ("key", {"keys": "cmd+tab"}, "cliclick kd:cmd kp:tab ku:cmd"),
])
def test_a_successful_ssh_input_carries_the_mark(tmp_path, monkeypatch, tool, args, command):
    tools, transport = ssh_mac(tmp_path, monkeypatch)
    result = receipt(tools[tool](**args))
    assert result["ok"] is True and result["returncode"] == 0 and marked(result), result
    assert len(transport.calls) == 1 and command in remote(transport.calls[0]), transport.calls
    assert "__FRONT_BEFORE__" in remote(transport.calls[0]), "the same ssh call reads who is in front"


@pytest.mark.parametrize("outcome", [1, 255, TIMEOUT])
def test_a_failed_ssh_input_carries_no_mark(tmp_path, monkeypatch, outcome):
    """A cliclick failure, an SSH failure (255) or a timeout: the one remote process's outcome is unknown."""
    tools, transport = ssh_mac(tmp_path, monkeypatch, answer=fail_at(0, outcome=outcome))
    result = receipt(tools["click"](x=1, y=2, raw=True))
    assert result["ok"] is False and not marked(result) and result["error"], result
    assert len(transport.calls) == 1


def test_ssh_reads_refusals_and_remote_exec_carry_no_mark(tmp_path, monkeypatch):
    def answer(argv, _index):
        command = argv[-1] if argv[0] == "ssh" else ""
        if "bounds of window" in command:
            return 0, "0, 0, 1440, 900"
        if "background only" in command:
            return 0, "Finder, Safari"
        if "lsappinfo list" in command:  # an inventory that names nothing is apps_unread; a read names one application
            return 0, (' 1) "Finder" ASN:0x0-0x1: \n    bundleID="com.apple.finder"\n    pid = 100 type="Foreground" flavor=3\n'
                       '__FRONT__"pid"=100 "CFBundleIdentifier"="com.apple.finder" "LSDisplayName"="Finder" ')
        return (0, "host=mac\nuser=owner\n/usr/sbin/screencapture\n/opt/homebrew/bin/cliclick") if "hostname" in command else 0
    tools, transport = ssh_mac(tmp_path, monkeypatch, answer=answer)
    refused = [tools["scroll"](clicks=2), tools["click"](x=1, y=1, button="middle", raw=True),
               tools["hold_key"](keys="cmd"), tools["cursor_position"]()]
    assert transport.calls == [], "nothing unsupported reaches the Mac"
    reads = [tools["remote_exec"](command="true"), tools["window_list"](), tools["test_connection"](),
             tools["screenshot"](), tools["capabilities"](), tools["wait"](ms=1)]
    refusal = tools["ax_tree"]()  # no accessibility reader over SSH: an honest refusal, never a window list as success
    for text in refused + reads + [refusal]:
        assert not marked(receipt(text)), text
    assert all(receipt(text)["ok"] is True for text in reads), reads
    assert receipt(refusal)["ok"] is False and receipt(refusal)["reason"] == "unsupported"
    assert {argv[0] for argv in transport.calls} == {"ssh", "scp"}


def test_an_ssh_key_sequence_keeps_the_mark_of_the_steps_that_ran(tmp_path, monkeypatch):
    tools, transport = ssh_mac(tmp_path, monkeypatch)
    result = receipt(tools["key"](keys="down down return"))
    assert result["ok"] is True and result["steps"] == 3 and marked(result), result

    transport.calls.clear()
    transport.answer = fail_at(2)
    result = receipt(tools["key"](keys="down down return"))
    assert result["ok"] is False and marked(result) and result["steps_done"] == 2, result
    assert [c for c in ("cliclick kp:arrow-down", "cliclick kp:arrow-down", "cliclick kp:return")] == \
        [next(c for c in ("cliclick kp:arrow-down", "cliclick kp:return") if c in remote(argv)) for argv in transport.calls]

    transport.calls.clear()
    transport.answer = fail_at(0, outcome=255)
    result = receipt(tools["key"](keys="down return"))
    assert result["ok"] is False and not marked(result) and result["steps_done"] == 0, result


def test_a_disabled_or_missing_connection_never_marks(tmp_path, monkeypatch):
    tools, transport = ssh_mac(tmp_path, monkeypatch)
    tools["add_connection"](name="mac", backend="ssh_macos", host="mac.invalid", enabled=False, activate=True)
    for text in (tools["click"](x=1, y=1), tools["type_text"](text="x"), tools["key"](keys="return return")):
        assert receipt(text)["ok"] is False and not marked(receipt(text)), text
    assert transport.calls == []


# --- OSWorld (the marks it already set) -----------------------------------------------------------

class Response(io.BytesIO):
    status = 200


class Guest:
    """``urlopen`` toward an OSWorld guest: each /execute answers with the next returncode (the last repeats)."""

    def __init__(self, codes):
        self.codes, self.commands, self.validations = list(codes), [], []

    def __call__(self, request, timeout=None):
        from tests.test_unix_computer_use_safety_contract import guest_key_validation
        command = json.loads(request.data.decode("utf-8"))["command"][-1]
        validation = guest_key_validation(command)
        if validation:
            self.validations.append(command)
            return Response(json.dumps(validation["result"]).encode("utf-8"))
        self.commands.append(command)
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        answer = {"status": "success", "output": "", "error": "" if code == 0 else "guest failed", "returncode": code}
        return Response(json.dumps(answer).encode("utf-8"))


def osworld(tmp_path, monkeypatch, codes):
    tools, _transport = desktop(tmp_path, monkeypatch, "linux", "x11")
    tools["add_connection"](name="vm", backend="osworld_http", target="http://guest.invalid:5000", activate=True)
    guest = Guest(codes)
    monkeypatch.setattr(urllib.request, "urlopen", guest)
    return tools, guest


def test_an_osworld_key_sequence_keeps_its_steps_marks(tmp_path, monkeypatch):
    tools, guest = osworld(tmp_path, monkeypatch, [0])
    result = receipt(tools["key"](keys="Left Left"))
    assert result["ok"] is True and result["steps"] == 2 and marked(result), result
    assert all("pyautogui.press('left')" in command for command in guest.commands)

    tools, guest = osworld(tmp_path, monkeypatch, [0, 1])
    result = receipt(tools["key"](keys="ctrl+a ctrl+c"))
    assert result["ok"] is False and marked(result) and result["steps_done"] == 1, result

    tools, guest = osworld(tmp_path, monkeypatch, [1])
    result = receipt(tools["key"](keys="ctrl+a ctrl+c"))
    assert result["ok"] is False and not marked(result) and len(guest.commands) == 1, result


def test_osworld_reads_and_remote_exec_stay_unmarked(tmp_path, monkeypatch):
    tools, _guest = osworld(tmp_path, monkeypatch, [0])
    assert marked(receipt(tools["click"](x=1, y=2, raw=True)))
    for text in (tools["cursor_position"](), tools["remote_exec"](command="touch /tmp/x")):
        assert receipt(text)["ok"] is True and not marked(receipt(text)), text


def test_a_refused_combo_is_a_not_sent_input_on_every_refusal_path(tmp_path, monkeypatch):
    tools, transport = desktop(tmp_path, monkeypatch, "macos", tools=MAC)
    for keys in ("bogus+right", "super+x"):
        out = receipt(tools["key"](keys=keys))
        assert out["ok"] is False and out["dispatched"] is False and "not sent" in out["note"] and transport.calls == [], (keys, out)
    held = receipt(tools["hold_key"](keys="a"))
    assert held["ok"] is False and held["dispatched"] is False and "not sent" in held["note"] and transport.calls == [], held
