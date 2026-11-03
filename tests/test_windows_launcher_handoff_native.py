"""Native Windows proof of the launcher handoff and of its one notification-area icon.

Windows only; nothing here is proven by the stand-ins in ``test_launcher_source_handoff.py`` or
``test_desktop_notifications.py``. Every case requires ``OUROBOROS_WINDOWS_DESKTOP_QA=1``:
the terminal and legacy controls deliberately open visible consoles. ``pythonw.exe``
(a GUI-subsystem interpreter without a console, like the packaged app) hands over through the real
``launcher_bootstrap.exec_source_launcher``; a launcher started in a console of its own (a terminal run) hands
over through it too; and ``pythonw.exe`` starts an older app's successor the way its C runtime's exec did (a
plain start from a console-less parent, which Windows gives a visible console) running the real prologue taken
from launcher.py with an explicit PyInstaller archive descriptor pointing at pythonw's real GUI image.
A manual source run with seed argv is the negative control. Their oracle is the requirement, no console window anyone can see, plus the terminal run's
console reaching its successor; whether a windowless successor reports console clients is recorded in the JUnit
properties as evidence, never assumed. Use the base interpreter (and its adjacent pythonw), since a venv
redirector adds a second console client and changes PID identity. The icon case also needs pythonnet;
it hands one silent balloon to the icon it creates and removes it. No case
clicks a balloon, counts the shell's icons or runs Panic: those stay manual Windows qualification.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.serial, pytest.mark.skipif(sys.platform != "win32", reason="native Windows consoles"),
              pytest.mark.skipif(os.environ.get("OUROBOROS_WINDOWS_DESKTOP_QA") != "1",
                                 reason="interactive desktop probes require OUROBOROS_WINDOWS_DESKTOP_QA=1")]

REPO = pathlib.Path(__file__).resolve().parents[1]
PYTHON = getattr(sys, "_base_executable", sys.executable)
PYTHONW = pathlib.Path(PYTHON).with_name("pythonw.exe")

PROBE = r'''
import ctypes, json, os, pathlib, sys
kernel, user = ctypes.WinDLL("kernel32"), ctypes.WinDLL("user32")
kernel.GetConsoleWindow.restype = ctypes.c_void_p
user.IsWindowVisible.argtypes = (ctypes.c_void_p,)
window, clients = kernel.GetConsoleWindow(), (ctypes.c_uint32 * 8)()
facts = {"argv": sys.argv, "pid": os.getpid(), "window_visible": bool(window and user.IsWindowVisible(window)),
         "console_window": window or 0, "console_clients": kernel.GetConsoleProcessList(clients, 8),
         "mark": os.environ.get("OUROBOROS_LAUNCHER_HANDOFF")}
receipt = pathlib.Path(os.environ["NATIVE_RECEIPT"])
receipt.with_suffix(".writing").write_text(json.dumps(facts), encoding="utf-8")
receipt.with_suffix(".writing").replace(receipt)
'''

HANDING_LAUNCHER = r'''
import ctypes, json, pathlib, subprocess, sys
from ouroboros import launcher_bootstrap as lb
out = pathlib.Path(sys.argv[1])
kernel = ctypes.WinDLL("kernel32")
kernel.GetConsoleWindow.restype = ctypes.c_void_p
(out / "handing.json").write_text(json.dumps({"console_window": kernel.GetConsoleWindow() or 0,
    "console_clients": kernel.GetConsoleProcessList((ctypes.c_uint32 * 8)(), 8)}), encoding="utf-8")
class Log:
    def info(self, *args, **kwargs):
        pass
    warning = error = info
context = lb.BootstrapContext(bundle_dir=out, repo_dir=pathlib.Path(sys.argv[2]), data_dir=out,
                              settings_path=out / "settings.json", embedded_python=sys.argv[3], app_version="0",
                              hidden_run=lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
                              save_settings=None, log=Log())  # the import probe passes; the handoff is the subject
note = lambda name: (out / name).write_text("1", encoding="utf-8")
lb.exec_source_launcher(context, ["--launch-intent", "owner"], release_lock=lambda: note("released"),
                        acquire_lock=lambda: note("acquired") or True, before_release=lambda: note("before_release"),
                        on_failure=lambda: note("restored"), desktop=True, macos=False)
note("carried_on")
'''

OLD_APP = r'''
import json, pathlib, subprocess, sys
child = subprocess.Popen([sys.argv[1], sys.argv[2], "--seed-bundle", sys.argv[3]])  # no flags, like its CRT exec
pathlib.Path(sys.argv[4]).write_text(json.dumps({"pid": child.pid, "code": child.wait()}), encoding="utf-8")
'''


def await_receipt(path: pathlib.Path, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"no receipt at {path}"
        time.sleep(0.1)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def base_interpreter_identity():
    """A redirector cannot witness sole-client or unchanged-PID assertions."""
    child = subprocess.Popen([PYTHON, "-c", "import os; print(os.getpid())"], stdout=subprocess.PIPE,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    stdout, _ = child.communicate(timeout=20)
    assert child.returncode == 0
    if int(stdout.strip()) != child.pid:
        pytest.skip("the base interpreter is also a redirector; direct native PID identity is unavailable")


@pytest.fixture
def console_less_parent():
    if not PYTHONW.exists():
        pytest.skip("no pythonw.exe beside this interpreter")
    return str(PYTHONW)


def hand_over(tmp_path, command, **popen):
    """Run HANDING_LAUNCHER as ``command``: the handing process's console facts and its successor's receipt."""
    out, repo = tmp_path / "out", tmp_path / "Ouroboros data" / "repo dir"
    out.mkdir()
    repo.mkdir(parents=True)
    (repo / "launcher.py").write_text(PROBE, encoding="utf-8")
    (tmp_path / "handing.py").write_text(HANDING_LAUNCHER, encoding="utf-8")
    receipt = out / "successor.json"
    env = {**os.environ, "PYTHONPATH": str(REPO), "NATIVE_RECEIPT": str(receipt)}
    env.pop("OUROBOROS_LAUNCHER_HANDOFF", None)
    parent = subprocess.Popen([*command, str(tmp_path / "handing.py"), str(out), str(repo), PYTHON], env=env,
                              **popen)
    assert parent.wait(timeout=60) == 0, "the launcher leaves through os._exit(0) once its successor exists"
    facts = await_receipt(receipt)
    assert facts["argv"] == [str(repo / "launcher.py"), "--launch-intent", "owner"], "spaced paths arrive intact"
    assert facts["mark"] == "1"
    assert {path.name for path in out.iterdir()} >= {"released", "before_release"}
    assert not (out / "acquired").exists() and not (out / "carried_on").exists()
    return json.loads((out / "handing.json").read_text(encoding="utf-8")), facts


def test_native_handoff_from_a_console_less_launcher_opens_no_console_window(tmp_path, console_less_parent,
                                                                            record_property):
    """Decision 1: the packaged app's case. Its successor shows no console window; spaced paths stay intact."""
    handing, facts = hand_over(tmp_path, [console_less_parent])
    record_property("handing_console", json.dumps(handing))
    record_property("successor_console", json.dumps(facts))  # evidence of what CREATE_NO_WINDOW gave it
    assert handing["console_clients"] == 0, "precondition: the handing process has no console"
    assert not facts["window_visible"], "no console window anyone can see"


def test_native_terminal_run_keeps_its_console_for_the_successor(tmp_path, record_property):
    """The intentional CLI contract: a launcher attached to a console (here one of its own, as a terminal gives)
    hands over into that same console instead of a hidden one."""
    handing, facts = hand_over(tmp_path, [PYTHON], creationflags=subprocess.CREATE_NEW_CONSOLE)
    record_property("handing_console", json.dumps(handing))
    record_property("successor_console", json.dumps(facts))
    assert handing["console_clients"] >= 1 and handing["console_window"], "precondition: an attached console"
    assert facts["console_window"] == handing["console_window"], "the successor runs in the launcher's own console"


def test_native_older_apps_successor_leaves_the_console_window_its_handoff_opened(tmp_path, console_less_parent,
                                                                                    record_property):
    """Decision 2: the real prologue, from launcher.py, in a process Windows gave a visible console."""
    source = (REPO / "launcher.py").read_text(encoding="utf-8")
    prologue = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_rehome_legacy_handoff")
    successor = tmp_path / "old app successor.py"
    initial_probe = "if not os.environ.get('OUROBOROS_LAUNCHER_HANDOFF'):\n" + "\n".join(
        "    " + line for line in PROBE.replace("NATIVE_RECEIPT", "NATIVE_INITIAL").splitlines()) + "\n"
    successor.write_text("import os, subprocess, sys, pathlib\n\n__file__ = " + repr(str(REPO / "launcher.py")) + "\n"
                         + initial_probe + ast.get_source_segment(source, prologue)
                         + "\n\n_rehome_legacy_handoff()\n" + PROBE, encoding="utf-8")
    (tmp_path / "old_app.py").write_text(OLD_APP, encoding="utf-8")
    seed, first, receipt = str(PYTHONW.parent / "_internal"), tmp_path / "first.json", tmp_path / "receipt.json"
    # A genuine GUI PE image and native console, with the old bootloader descriptor supplied explicitly.
    # This is a pythonw surrogate, not an executed historical packaged archive.
    env = {**os.environ, "NATIVE_RECEIPT": str(receipt), "_PYI_ARCHIVE_FILE": str(PYTHONW),
           "NATIVE_INITIAL": str(tmp_path / "initial.json")}
    env.pop("OUROBOROS_LAUNCHER_HANDOFF", None)
    subprocess.run([console_less_parent, str(tmp_path / "old_app.py"), PYTHON, str(successor), seed,
                    str(first)], env=env, timeout=60, check=True)
    started = await_receipt(first)
    initial = await_receipt(tmp_path / "initial.json")
    record_property("legacy_initial_console", json.dumps(initial))
    assert initial["window_visible"] and initial["console_clients"] == 1, "precondition: a sole visible legacy console"
    facts = await_receipt(receipt)
    record_property("restarted_console", json.dumps(facts))  # evidence of what CREATE_NO_WINDOW gave it
    assert started["code"] == 0, "the first successor leaves at once, so Windows closes the console it opened"
    assert facts["pid"] != started["pid"] and facts["argv"][1:] == ["--seed-bundle", seed]
    assert not facts["window_visible"] and facts["mark"] is None, "no visible console window; the mark consumed"


def test_native_manual_source_cli_with_seed_argument_keeps_its_console(tmp_path, record_property):
    """Negative control for legacy repair: seed provenance alone never detaches intentional source CLI."""
    source = (REPO / "launcher.py").read_text(encoding="utf-8")
    prologue = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "_rehome_legacy_handoff")
    entry, receipt = tmp_path / "manual source.py", tmp_path / "manual.json"
    entry.write_text("import os, subprocess, sys, pathlib\n__file__ = " + repr(str(REPO / "launcher.py")) + "\n"
                     + ast.get_source_segment(source, prologue) + "\n_rehome_legacy_handoff()\n" + PROBE,
                     encoding="utf-8")
    env = {**os.environ, "NATIVE_RECEIPT": str(receipt)}
    for key in ("_PYI_ARCHIVE_FILE", "OUROBOROS_LAUNCHER_HANDOFF"):
        env.pop(key, None)
    child = subprocess.Popen([PYTHON, str(entry), "--seed-bundle", str(PYTHONW.parent / "_internal")],
                             env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    assert child.wait(timeout=60) == 0
    facts = await_receipt(receipt)
    record_property("manual_source_console", json.dumps(facts))
    assert facts["pid"] == child.pid, "the intentional source process was never replaced"
    assert facts["console_clients"] >= 1 and facts["console_window"], "the original console remains attached"


@pytest.mark.skipif(os.environ.get("OUROBOROS_WINDOWS_DESKTOP_QA") != "1",
                    reason="needs an interactive Windows desktop and OUROBOROS_WINDOWS_DESKTOP_QA=1")
def test_native_icon_takes_a_silent_balloon_and_is_disposed_at_stop():
    """Part of decision 1A on the real shell: the icon comes up, a Sound-off balloon is handed to it (the shell's
    answer, or WinForms' unanswered call), an accepted one becomes the click target, and ``stop`` disposes the
    icon. Not observed here: a click, the shell's icon count, Panic."""
    pytest.importorskip("clr")
    from ouroboros import desktop_notifications as dn
    from ouroboros import launcher_tray

    opened: list = []
    background = SimpleNamespace(shutdown=threading.Event(), window=None, exit_launcher=lambda: None,
                                 show_window=lambda: opened.append("window"), open_notification=opened.append)
    tray = launcher_tray.WindowsTray(background)
    assert tray.start() and tray.ready.wait(10), "the notification-area icon came up"
    try:
        answer = dn.WindowsNotifier(lambda token: None, tray).notify(
            "Ouroboros native QA", "A silent balloon on the one icon", False, "n1-native")
        assert answer["status"] in ("submitted", "unknown"), answer
        if answer["status"] == "submitted":
            assert answer["sound"] == "off" and tray._latest == "n1-native"
    finally:
        tray.stop(wait=3.0)
    assert tray._disposed.is_set() and not tray.ready.is_set(), "removed before an exit would leave it"
    assert dn.WindowsNotifier(lambda token: None, tray).notify("t", "", True, "n2")["status"] == "unavailable"
