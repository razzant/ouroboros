"""Sandboxed Chromium started inside a tracked command (run_command's producer): what its containment holds.

The command's child launches Playwright Chromium WITH its sandbox (``chromium_sandbox=True``, never
``--no-sandbox``, no fallback) and opens a local ``data:`` page. Windows bears every tracked child into its own
Job (``shell_process.contained_popen``): while the page is open each process of this run (the driver, the
browser, its renderer, GPU and utility processes) must be a member of the command's Job, and after a normal
exit, Stop now or Panic no member is left. POSIX keeps the command's process group; Playwright starts Chromium
in its own group there, outside it, so POSIX shows only that Chromium ends once its driver does, by measured
identity (PID and birth). Every case requires the page to work and nothing of this run to outlive the bound.

Opt-in, never in a default run: ``-m browser``. ``OUROBOROS_EXPECT_BROWSER_ENGINES`` naming ``chromium`` turns a
sandboxed Chromium that cannot launch here into a failure instead of a skip. The existing Chromium pin under
``PLAYWRIGHT_BROWSERS_PATH`` selects the executable, with no older-browser fallback. Kept apart from
``test_browser_tools_smoke.py``: its lanes include a Linux container run as root, where Chromium's sandbox cannot
start, so this case would fail there under ``OUROBOROS_EXPECT_BROWSER_ENGINES``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import warnings

import pytest

from ouroboros import chromium_runtime, config, platform_layer
from ouroboros.process_containment import pid_is_zombie
from ouroboros.tools import shell_process

pytestmark = [pytest.mark.browser]
_EXPECTED_BROWSER_ENGINES = {item.strip().lower()
                             for item in os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES", "").split(",")
                             if item.strip()}
_STOP_BOUND_SEC = 5.0  # the bound under test: Stop now's own output read waits 5 s for released pipes
_START_BOUND_SEC = 120.0  # Chromium's start on a loaded host, not the bound under test

_CHILD = r'''
import json, os, pathlib, sys, time
from playwright.sync_api import sync_playwright

ready, go, profile, executable = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
with sync_playwright() as playwright:
    context = playwright.chromium.launch_persistent_context(
        profile, headless=True, chromium_sandbox=True,
        executable_path=executable)
    page = context.pages[0] if context.pages else context.new_page()
    page.goto("data:text/html,<title>contained smoke</title><p id=v>tracked</p>")
    facts = {"title": page.title(), "text": page.inner_text("#v")}
    print(json.dumps(facts), flush=True)
    partial = ready.with_name(ready.name + ".partial")  # renamed whole: never read half-written
    partial.write_text(json.dumps(facts), encoding="utf-8")
    os.replace(partial, ready)
    while not go.exists():  # held open until the case lets it close normally, or stops it
        time.sleep(0.05)
    context.close()
'''


@pytest.fixture(scope="module")
def sandboxed_chromium():
    """Skip (fail when the lane expects Chromium) unless Playwright launches a sandboxed Chromium here at all."""
    refuse = pytest.fail if "chromium" in _EXPECTED_BROWSER_ENGINES else pytest.skip
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        refuse(f"Playwright is not importable: {exc}")
    root = chromium_runtime.browsers_root(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""))
    pin = chromium_runtime.host_pin()
    executable = chromium_runtime.installed_executable(root, pin) if root is not None else None
    if executable is None:
        refuse("the current Chromium pin is not installed under PLAYWRIGHT_BROWSERS_PATH")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True, chromium_sandbox=True, timeout=60_000,
                executable_path=str(executable))
            version = browser.version
            browser.close()
    except Exception as exc:  # not installed, or its sandbox cannot start on this host at all
        refuse(f"a sandboxed Chromium cannot launch here: {type(exc).__name__}: {exc}")
    return version, str(executable)


def _process_rows() -> dict:
    """``{pid: (ppid, command line)}`` for every process this user can see."""
    if os.name == "nt":
        import psutil

        rows = {}
        for process in psutil.process_iter(["pid", "ppid", "cmdline"]):
            rows[process.info["pid"]] = (process.info["ppid"], " ".join(process.info["cmdline"] or ()))
        return rows
    out = subprocess.run(["ps", "-axww", "-o", "pid=,ppid=,command="], capture_output=True, text=True,
                         timeout=10).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            rows[int(parts[0])] = (int(parts[1]), parts[2])
    return rows


def _descendants(rows: dict, root: int) -> set:
    found, stack = set(), [root]
    while stack:
        parent = stack.pop()
        for pid, (ppid, _cmd) in rows.items():
            if ppid == parent and pid not in found:
                found.add(pid)
                stack.append(pid)
    return found


def _tree(root: int, profile: str) -> dict:
    """This run's processes while the page is open, each pinned by birth: the command's descendants, and the
    browser its unique profile names (outside the command's group on POSIX) with the browser's children."""
    rows = _process_rows()
    pids = _descendants(rows, root)
    for browser in [pid for pid, (_ppid, cmd) in rows.items()
                    if profile in cmd and "--type=" not in cmd and pid != root]:
        pids |= {browser, *_descendants(rows, browser)}
    return {pid: (rows[pid][1], platform_layer.process_start_time(pid)) for pid in sorted(pids) if pid in rows}


def _gone(pid: int, birth: str) -> bool:
    if platform_layer.pid_provably_gone(pid) or pid_is_zombie(pid):
        return True
    now = platform_layer.process_start_time(pid)
    return bool(now and birth) and now != birth  # that PID names another process now


def _outliving(tree: dict, bound: float) -> dict:
    """What of ``tree`` still runs after ``bound`` seconds (read at least once)."""
    deadline = time.monotonic() + bound
    while True:
        tree = {pid: facts for pid, facts in tree.items() if not _gone(pid, facts[1])}
        if not tree or time.monotonic() >= deadline:
            return tree
        time.sleep(0.05)


def _end_leftovers(tree: dict) -> None:
    """End only positively identified leftovers; an unreadable birth is no permission to signal a PID."""
    for pid, (_cmd, birth) in tree.items():
        if _gone(pid, birth):
            continue
        current = platform_layer.process_start_time(pid)
        if not birth or not current:
            warnings.warn(f"Smoke cleanup unconfirmed for PID {pid}: birth identity unavailable", RuntimeWarning)
        elif current == birth:
            platform_layer.force_kill_pid(pid)


def _kinds(tree: dict) -> list:
    return sorted(next((part.split("=", 1)[1] for part in cmd.split() if part.startswith("--type=")), "browser/driver")
                  for cmd, _birth in tree.values())


@pytest.mark.parametrize("ending", ["normal exit", "Stop now", "Panic"])
def test_sandboxed_chromium_inside_a_tracked_command_ends_with_its_command(sandboxed_chromium, tmp_path,
                                                                         monkeypatch, ending):
    registry: set = set()
    monkeypatch.setattr(shell_process, "_active_subprocesses", registry)
    monkeypatch.setattr(shell_process, "_panic_requested", False)
    monkeypatch.setattr(shell_process, "_spawning_subprocesses", 0)
    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.2)
    script, ready, go, profile = (tmp_path / "child.py", tmp_path / "page.ready", tmp_path / "page.go",
                                  tmp_path / "profile")
    script.write_text(_CHILD, encoding="utf-8")
    version, executable = sandboxed_chromium
    print(json.dumps({"chromium_version": version, "executable": executable, "ending": ending}))
    pressed, outcome = threading.Event(), {}

    def command():
        try:
            outcome["result"] = shell_process._tracked_subprocess_run(
                [sys.executable, "-I", str(script), str(ready), str(go), str(profile), executable], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=600,
                stop=(lambda: True if pressed.is_set() else None) if ending == "Stop now" else None)
        except BaseException as exc:  # judged below
            outcome["error"] = exc
        outcome["ended"] = time.monotonic()

    runner = threading.Thread(target=command, name="tracked-chromium-command", daemon=True)
    runner.start()
    tree: dict = {}
    try:
        started = time.monotonic()
        while not ready.exists():
            assert runner.is_alive(), ("the same Chromium launched outside the command but never opened the page "
                                       f"inside it: {outcome}")
            assert time.monotonic() - started < _START_BOUND_SEC, "Chromium never opened the page"
            time.sleep(0.05)
        assert json.loads(ready.read_text(encoding="utf-8")) == {"title": "contained smoke", "text": "tracked"}
        [proc] = list(registry)
        tree = _tree(proc.pid, str(profile))
        kinds = _kinds(tree)
        assert "renderer" in kinds, f"no renderer of this run while its page is open: {kinds}"
        assert not [pid for pid, (cmd, _birth) in tree.items() if "--no-sandbox" in cmd], "its sandbox is off"
        job = proc.__dict__.get("_ouroboros_job")
        if os.name == "nt":
            from ouroboros.process_overload import _in_job

            assert job is not None, "the command was not born into its own Job (log: 'runs without its own Job')"
            # Chromium's processes and its driver; a console host the system attaches is not a member by design
            ours = {pid: cmd for pid, (cmd, _birth) in tree.items()
                    if str(profile) in cmd or "--type=" in cmd or "run-driver" in cmd}
            outside = {pid: cmd[:160] for pid, cmd in ours.items() if not _in_job(pid, job)}
            assert not outside, f"processes of this run outside the command's Job while the page is open: {outside}"

        if ending == "normal exit":
            go.write_text("close", encoding="utf-8")
            runner.join(_START_BOUND_SEC)
            assert not runner.is_alive() and "error" not in outcome, outcome
            result = outcome["result"]
            assert result.returncode == 0 and '"title": "contained smoke"' in result.stdout, result
        elif ending == "Stop now":
            pressed.set()
            stopped_at = time.monotonic()
            runner.join(_STOP_BOUND_SEC + 10)
            assert not runner.is_alive(), "Stop now never ended the command's wait"
            assert isinstance(outcome.get("error"), shell_process.CommandStopped), outcome
            assert outcome["ended"] - stopped_at < _STOP_BOUND_SEC, (
                f"the command ended {outcome['ended'] - stopped_at:.1f}s after Stop now: its readers were held")
            assert '"title": "contained smoke"' in outcome["error"].output, "the output so far is kept"
        else:
            receipts = shell_process.kill_all_tracked_subprocesses(request_only=True)
            requested_at = time.monotonic()
            assert [row["pid"] for row in receipts if row.get("requested")] == [proc.pid], receipts
            report = shell_process.kill_all_tracked_subprocesses()  # Panic's settle pass: no wait
            assert report is None or report["state"] == "unconfirmed", report
            if report is None and os.name == "nt":  # the Job is what Windows settlement measures
                assert not _outliving(tree, 0), "a clean settlement while this run's processes still ran"
            runner.join(_STOP_BOUND_SEC + 10)
            assert not runner.is_alive(), "the Panic request never released the command's readers"
            assert outcome["ended"] - requested_at < _STOP_BOUND_SEC, (
                f"the command ended {outcome['ended'] - requested_at:.1f}s after Panic: its readers were held")

        left = _outliving(tree, _STOP_BOUND_SEC)
        assert not left, f"{ending}: processes of this run outlived the bound: {_kinds(left)} {sorted(left)}"
        if os.name == "nt":
            assert platform_layer.job_has_running_member(job) is False, "a member of the command's Job still runs"
        assert shell_process.unsettled_tracked_children() is None
        assert shell_process.kill_all_tracked_subprocesses(deadline=time.monotonic() + 5) is None
        assert proc not in registry, "only an observed exit leaves the registry"
    finally:
        go.write_text("close", encoding="utf-8")
        for proc in list(registry):
            if proc.poll() is None:
                shell_process._kill_process_group(proc)
        _end_leftovers(tree)
        runner.join(30)
