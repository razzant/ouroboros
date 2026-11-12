#!/usr/bin/env python3
"""Install the UI lane's browsers and their Ubuntu packages with every download bounded (#2895).

    python3 -I -S scripts/ci_browser_install.py [--pinned-chromium] [--attempt-seconds N]
        [--python PATH] <browser>...

`playwright install --with-deps <browser>...` is one command with no deadline of its own: when an
Ubuntu mirror served its packages at about 30 KB/s, it held the install step until the step's
45-minute ceiling (#2895). This helper does the same work in three stages and bounds the two that
wait on a server:

1. packages: `apt-get update`, the packages `playwright install-deps --dry-run <browser>...`
   reports missing (Playwright 1.62 simulates its own `apt-get install`; Xvfb and the fonts come
   with them), then `apt-get install --download-only` of those. Nothing is installed yet. The
   update exits 0 even when an index could not be fetched and the old one stays, so only the report
   tells whether the indices are usable: update and report are one step, whose retry refreshes
   the indices again. A report that reads is used as it is, whichever indices it came from.
2. dpkg: `apt-get install --no-download` of exactly those packages, from the archives stage 1
   fetched. It never waits on a server, so it is neither stopped nor repeated here, not even when
   this helper is interrupted: its failure ends the install, and only the step's ceiling bounds it.
3. browsers: `playwright install <browser>...`, then with --pinned-chromium the browser tools'
   Chrome for Testing (`python -m ouroboros.chromium_runtime install`).

An attempt of a bounded stage has --attempt-seconds (600, the owner's ~10 minutes) for all of its
commands, and the stage gets ONE more attempt, starting again at the step that failed or was still
running at the deadline: apt keeps the archives it completed and resumes a partial one; Playwright and
chromium_runtime skip a browser already installed. A second failure ends the install.

A command still running at the deadline is stopped with its process group: SIGTERM, SIGKILL after
STOP_GRACE_SEC, then the group must be empty within another grace. Without root, apt-get runs under
`sudo -n`, in sudo's process group (no terminal, so no pty): this user's SIGTERM reaches sudo, which
relays it to apt-get, while SIGKILL stops only sudo, never root's apt-get or its download methods.
A member left after both signals is therefore reported and the install refused: no retry starts
beside it. Root's processes are beyond this helper, so that member can keep running until the runner
goes away, and only the group is checked: nothing here proves every descendant gone.

The exit status is 0 once everything is installed, else 1 (128 + the signal when interrupted).
--python names the interpreter that runs Playwright and chromium_runtime (default: this one).
Stdlib only, Python 3.10.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ATTEMPT_SEC = 600.0  # one attempt of a download stage: the owner's ~10 minutes (#2895)
ATTEMPTS = 2  # the first attempt and ONE retry
STOP_GRACE_SEC = 5.0  # after SIGTERM, after SIGKILL, and for the stopped group to empty
APT_INSTALL = ("install", "-y", "--no-install-recommends")  # the options of Playwright's own install-deps
# `playwright install-deps --dry-run` (Playwright 1.62) simulates its `apt-get install` and prints
# either ALL_PRESENT (exit 0) or the MISSING header and one indented package per line (exit 1).
ALL_PRESENT = "All system dependencies are installed."
MISSING = re.compile(r"Missing system dependencies \((\d+)\):")
PACKAGE = re.compile(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?")
BROWSER = re.compile(r"[a-z][a-z-]*")


class Refused(Exception):
    """The install cannot complete; the message names the stage, the command and why."""


class Retryable(Refused):
    """A refusal the stage's one retry may still repair; raised again if that retry meets it too."""


def _say(message: str) -> None:
    print(f"ci_browser_install: {message}", flush=True)


def _annotate(kind: str, title: str, message: str) -> None:
    """A GitHub workflow command; its data escapes as the runner reads it."""
    data = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{kind} title={title}::{data}", flush=True)


def _stop(process: subprocess.Popen) -> bool:
    """Stop the command's process group (module docstring); True once the group has no member left."""
    for stop in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, stop)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass  # Every member left belongs to root: nothing this user sends can reach them.
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=STOP_GRACE_SEC)
    deadline = time.monotonic() + STOP_GRACE_SEC
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)


def run_bounded(command: list[str], deadline: float, *, capture: bool = False) -> tuple[int | None, str]:
    """Run ``command`` in its own process group until it exits or the monotonic ``deadline`` passes.

    Returns its exit status, None when the deadline stopped it, and its standard output when
    captured (otherwise it shares this process's streams). Refused when a stopped command's
    group still has members.
    """
    left = deadline - time.monotonic()
    shown = " ".join(command)
    _say("$ " + shown)
    if left <= 0:
        _say("not started: this attempt has no time left")
        return None, ""
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=REPO, start_new_session=True,
                                   stdout=subprocess.PIPE if capture else None,
                                   text=True if capture else None)
    except OSError as exc:
        raise Refused(f"cannot start {command[0]}: {exc}") from exc
    try:
        output, _ = process.communicate(timeout=left)
    except subprocess.TimeoutExpired:
        gone = _stop(process)
        if process.stdout is not None:
            process.stdout.close()
        _say(f"stopped after {time.monotonic() - started:.0f} s: the attempt's deadline passed")
        if not gone:
            raise Refused(f"`{shown}` outlived SIGTERM and SIGKILL: its process group {process.pid} still has "
                          "members this user cannot signal, so no retry starts beside them") from None
        return None, ""
    except BaseException:  # An interrupt of this helper stops the running download first.
        if not _stop(process):
            _say(f"interrupted: process group {process.pid} of `{shown}` still has members this user cannot signal")
        raise
    _say(f"exit {process.returncode} after {time.monotonic() - started:.0f} s")
    return process.returncode, output or ""


def run_stage(name: str, steps: list, attempt_sec: float) -> int:
    """Run a bounded stage's ``(label, step)`` pairs in order and return the attempts it took.

    ``step(deadline)`` is True once its work is done. ONE retry, with a fresh deadline, starts
    again at the first step that is not; a step that cannot succeed raises Refused itself,
    and one that failed for a stated reason the retry may repair raises Retryable.
    """
    pending = list(steps)
    for attempt in range(1, ATTEMPTS + 1):
        deadline = time.monotonic() + attempt_sec
        try:
            while pending and pending[0][1](deadline):
                pending.pop(0)
        except Retryable:
            if attempt == ATTEMPTS:
                raise
        if not pending:
            _say(f"{name}: done in attempt {attempt}")
            return attempt
        label = pending[0][0]
        if attempt < ATTEMPTS:
            _annotate("warning", "Browser install retried",
                      f"{name}: {label} failed or outlived the attempt's {attempt_sec:.0f} s; "
                      "its one retry starts again there")
    raise Refused(f"{name}: {label} failed in both attempts of {attempt_sec:.0f} s")


def missing_packages(report: str, status: int) -> list[str]:
    """The packages a Playwright 1.62 `install-deps --dry-run` report names; any other report is Retryable."""
    lines = report.splitlines()
    if status == 0 and ALL_PRESENT in (line.strip() for line in lines):
        return []
    for index, line in enumerate(lines):
        header = MISSING.fullmatch(line.strip())
        if header is None:
            continue
        count, rows = int(header[1]), lines[index + 1:index + 1 + int(header[1])]
        names = [row.strip() for row in rows]
        if (status != 0 and count > 0 and len(rows) == count and all(row.startswith("  ") for row in rows)
                and all(PACKAGE.fullmatch(name) for name in names)):
            return names
        break
    raise Retryable(f"packages: `playwright install-deps --dry-run` exited {status} without the report "
                    f"this helper reads (Playwright 1.62); its output: {report.strip()[-2000:] or 'none'}")


def _as_root() -> list[str]:
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo", "-n", "--"]
    raise Refused("packages: apt-get needs root, and this user has no sudo")


def _install_packages(command: list[str]) -> None:
    """The dpkg stage: one run, never stopped or repeated here (module docstring)."""
    _say("$ " + " ".join(command))
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=REPO)
    except OSError as exc:
        raise Refused(f"dpkg: cannot start {command[0]}: {exc}") from exc
    try:
        status = process.wait()
    except BaseException:
        _say(f"interrupted during dpkg: process {process.pid} is left to finish; an installation is never stopped here")
        raise
    _say(f"exit {status} after {time.monotonic() - started:.0f} s")
    if status != 0:
        raise Refused(f"dpkg: apt-get install --no-download exited {status}; an installation failure is "
                      "not a download, so it is not retried")


def install(browsers: list[str], *, pinned_chromium: bool = False, attempt_sec: float = ATTEMPT_SEC,
            python: str = sys.executable) -> None:
    """Install ``browsers`` and their packages in the three stages of the module docstring."""
    root, missing = _as_root(), []

    def update_and_resolve(deadline: float) -> bool:
        if run_bounded([*root, "apt-get", "update"], deadline)[0] != 0:
            return False
        status, report = run_bounded([python, "-m", "playwright", "install-deps", "--dry-run", *browsers],
                                     deadline, capture=True)
        if status is None:
            return False
        print(report, end="", flush=True)
        missing[:] = missing_packages(report, status)
        return True

    run_stage("packages", [
        ("apt-get update and its dependency report (playwright install-deps --dry-run)", update_and_resolve),
        ("apt-get install --download-only", lambda deadline: not missing or run_bounded(
            [*root, "apt-get", *APT_INSTALL, "--download-only", *missing], deadline)[0] == 0),
    ], attempt_sec)
    if missing:
        _install_packages([*root, "apt-get", *APT_INSTALL, "--no-download", *missing])
    steps = [("playwright install", lambda deadline: run_bounded(
        [python, "-m", "playwright", "install", *browsers], deadline)[0] == 0)]
    if pinned_chromium:
        steps.append(("chromium_runtime install", lambda deadline: run_bounded(
            [python, "-m", "ouroboros.chromium_runtime", "install"], deadline)[0] == 0))
    run_stage("browsers", steps, attempt_sec)


def _browser(value: str) -> str:
    if not BROWSER.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a Playwright browser name: {value!r}")
    return value


def _seconds(value: str) -> float:
    seconds = float(value)
    if not (math.isfinite(seconds) and seconds > 0):
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return seconds


def _interrupted(signum: int, _frame) -> None:
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts/ci_browser_install.py",
                                     description="Install Playwright browsers and their Ubuntu packages, "
                                                 "every download bounded and retried once (#2895).")
    parser.add_argument("browsers", nargs="+", type=_browser, help="Playwright browser names, e.g. chromium webkit")
    parser.add_argument("--pinned-chromium", action="store_true",
                        help="also install the browser tools' pinned Chrome for Testing")
    parser.add_argument("--attempt-seconds", type=_seconds, default=ATTEMPT_SEC,
                        help=f"one attempt of a download stage (default {ATTEMPT_SEC:.0f})")
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter that runs Playwright and chromium_runtime (default: this one)")
    args = parser.parse_args(argv)
    # Like SIGINT: stop the running download (its own session hears no signal sent to this one), then exit.
    previous = {stop: signal.signal(stop, _interrupted) for stop in (signal.SIGTERM, signal.SIGHUP)}
    started = time.monotonic()
    try:
        install(args.browsers, pinned_chromium=args.pinned_chromium, attempt_sec=args.attempt_seconds,
                python=args.python)
    except Refused as refusal:
        _say(f"refused: {refusal}")
        _annotate("error", "Browser install refused", str(refusal))
        return 1
    finally:
        for stop, handler in previous.items():
            signal.signal(stop, handler)
    _say(f"installed in {time.monotonic() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
