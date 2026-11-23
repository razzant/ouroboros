#!/usr/bin/env python3
"""One candidate launch on a Mac: its first opening and, only when that window is usable, one ordinary Restart.

The narrow manual path of the desktop workflow for #2791 (window_renewal.md "First opening only"). It reuses the
renewal arm, observer, opening record and window instruments for a single launch of the dispatched commit: no
baseline, Update, Panic, cold relaunch, task or Send, and nothing is ever tried twice. The first chat document is only
read, never navigated to, and the Restart waits for positive native evidence from that window. ``--budget`` bounds
the scenario's waits; the ordinary Quit and cleanup follow it.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # run as a script under -I: this instrument's own checkout
    sys.path.insert(0, str(REPO))

from devtools.macos_wkwebview.arm_source import sha256  # noqa: E402
from devtools.macos_wkwebview.launcher_evidence import launcher_diagnostics  # noqa: E402
from devtools.macos_wkwebview.native_window import public_reason, same_window, visible_key_view  # noqa: E402
from devtools.macos_wkwebview.window_renewal import (  # noqa: E402
    EXIT_CODES,
    Arm,
    host_facts,
    instrument_identity,
    now,
    wait_for,
    write_json,
)

SCHEMA = "ouroboros.macos-window-first-open/1"
SCOPE = ("one launch of the dispatched commit: its first opening, read without navigating, and only when that window was "
         "usable on its chat document with positive native evidence, one ordinary Restart in the same session; no "
         "Update, Panic, relaunch, task, Send or retry")
TYPING = ("typed", "typed_pixels", "window_typed_pixels", "window_pixels")  # the first window's own probe calibrates them
CHECKS = {"first_open": ("window_usable", *TYPING),
          "restart": ("generation_new", "view_new", "installed", "document_ready", "main_active_after", *TYPING)}
RESTART_RESERVE = 240  # seconds of budget one Restart needs to settle and be read; with less left, none is sent


def judge(report: dict) -> dict:
    """PASS: a usable first window, then one Restart that renewed its view in the same window with visible input.
    That is one observed Restart, not a finished repair. FAIL needs a demonstrated failure; anything missing is a gap."""
    if report.get("unsupported"):
        return {"verdict": "UNSUPPORTED", "failures": [], "gaps": [report["unsupported"]]}
    arm = report.get("candidate") or {}
    checks = arm.get("checks") or {}
    first = checks.get("first_open") or {}
    failures, gaps = [], [*report.get("gaps", ()), *(f"candidate: {gap}" for gap in arm.get("gaps", ()))]
    for phase, names in CHECKS.items():
        for name in names:
            value, key = (checks.get(phase) or {}).get(name), f"{phase}.{name}"
            if value is None:
                gaps.append(f"{key} was not established")
            elif value is False and (name not in TYPING or (phase == "restart" and first.get(name) is True)):
                failures.append(f"{key} is false")
            elif value is False:
                gaps.append(f"{key} is false, and nothing earlier in this session showed that probe holding")
    return {"verdict": "FAIL" if failures else "INCONCLUSIVE" if gaps else "PASS", "failures": failures, "gaps": gaps}


class FirstOpen(Arm):
    """The renewal arm reduced to one launch: the same preparation, observer, opening record and teardown."""

    def __init__(self, sha, parent, out, checkpoint, deadline):
        super().__init__("candidate", sha, parent, out, checkpoint)
        self.checks = {phase: dict.fromkeys(names) for phase, names in CHECKS.items()}
        self.deadline = deadline

    def run(self):
        first = self.checks["first_open"]
        try:  # the harness import inside prepare names checkout modules: its failure lands in the report
            self.prepare()
            self.phase = "first launch"
            if not self.start_session(self.tree["src"] / "launcher.py"):  # nothing goes to a window never usable
                raise RuntimeError("the first window was not usable; no Restart was sent")
            first["window_usable"] = True
            self.phase = "first window"
            self.evidence("first", self.win.state(), first)
            if not all(first[name] is True for name in TYPING):  # an absent or contradictory observation is no basis
                raise RuntimeError("the first window gave no positive native readability and input evidence; "
                                   "no Restart was sent")
            if self.remaining(RESTART_RESERVE) < RESTART_RESERVE:
                raise RuntimeError("the budget left too little time to settle a Restart; none was sent")
            self.phase = "restart"
            self.restart()
        except Exception as exc:
            self.gaps.append(f"{self.phase} stopped ({public_reason(exc)})")
            if self.launcher is not None:  # a launch that is ending may still leave its own exit and fatal class
                wait_for(lambda: self.launcher.poll() is not None, 10)
        finally:
            if first["window_usable"] is None and self.launcher is not None and self.launcher.poll() is not None:
                first["window_usable"] = False  # it ended before its window was ever usable; still running says nothing
            if self.openings:  # the same launch read again where the scenario ended, before the fixture's own Quit
                self.facts["launch_at_end"] = self.openings[-1].project(len(self.servers))
            self.teardown()

    def open_main(self):
        """In place of the journey's own: read whether this launch's window already shows its first chat document.
        Nothing is navigated, closed, reopened or clicked, so a window that does not show it stays unusable."""
        w, port = self.win, self.generation["port"]
        current = self.settle(None)  # reads only: an installed loopback view on this generation's port
        if not (current and current["pid"] == self.launcher.pid and current["home"] and visible_key_view(current, current)):
            raise RuntimeError("this launch's view was not visible, key and on screen at its chat document")
        document = w.document(port)  # its own server's root, complete, with the product's socket client
        if not document:
            raise RuntimeError("the chat document was not ready")
        if w.shows_room(None, current, origin=document["origin"]) is not True or w.until(
                lambda: w.visible("#chat-input", editable=True)) is not True:
            raise RuntimeError("Main was not the mounted, editable room")
        if not w.until(lambda: w.page("return window.__ouroWs?.isConnected() === true"), 30):
            raise RuntimeError("the chat document's WebSocket was not connected")
        return True

    def evidence(self, label, target, checks):
        """One usable window: its native identity and document, one WindowServer image of that window alone, and
        native typing in Main measured in that image and in WebKit's own snapshot. Nothing is sent."""
        if not target:
            raise RuntimeError(f"the {label} window's identity was not observed")
        w = self.win
        self.facts[label + "_window"] = target
        self.facts[label + "_document"] = self.observe(label + " document", lambda: w.document(self.generation["port"]))
        self.facts[label + "_image"] = self.observe(label + " window image", lambda: w.call(
            "capture", kind="window", name=label + "-window", target=target))
        self.observe(label + " native typing", lambda: w.typing(
            target, "#chat-input", label, checks, self.facts.setdefault(label + "_pixels", {})))

    def restart(self):
        """One ordinary Settings Restart of this session, requested once, then its successor read and nothing more."""
        checks, w = self.checks["restart"], self.win
        old = w.state()
        if not (old and visible_key_view(old, old)):
            raise RuntimeError("the window was not the visible key view; no Restart was sent")
        self.checkpoint()
        started = time.time()
        self.generation_step("restart")  # a lost reply is observed, never sent again
        self.checkpoint()
        current = self.observe("restart observe view", lambda: self.settle(old))
        settled = self.observe("restart document", lambda: w.document(self.generation["port"]))
        self.facts["restart_diagnostics"] = launcher_diagnostics(self.root, started, time.time())
        if current:
            checks["view_new"] = True if current["view"] != old["view"] else None  # an old view at the timer: below
            checks["installed"] = bool(current["installed"] and old["installed"] and same_window(old, current))
        checks["document_ready"] = True if settled else None
        if current and settled:
            checks["main_active_after"] = self.observe("restart landed room", lambda: w.shows_room(
                None, current, origin=settled["origin"]))
            if checks["main_active_after"]:
                self.evidence("restart", current, checks)
        last = self.observe("restart final view", w.state)
        self.facts["restart_window"] = {"before": old, "after": current, "final": last}
        self.reconcile_view("restart", old, current, last, checks, CHECKS["restart"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=600.0, help="seconds for the scenario's waits; teardown follows")
    args = parser.parse_args(argv)
    if not re.fullmatch("[a-f0-9]{40}", args.candidate):
        parser.error("the full 40-character candidate commit ID is required")
    if not 60 <= args.budget <= 1800:
        parser.error("--budget must be between 60 and 1800 seconds")
    out = args.out.resolve()
    if out == REPO or REPO in out.parents or (out.exists() and any(out.iterdir())):
        parser.error("output must be new or empty and outside the checkout")
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.budget
    report = {"schema": SCHEMA, "scope": SCOPE, "started_at": now(), "host": host_facts(),
              "instrument": {**instrument_identity(), "first_open.py": sha256(Path(__file__))}, "candidate": None,
              "judgement": {"verdict": "INCONCLUSIVE", "gaps": ["the native run has not reached its final observation"]}}
    parent = os.environ.get("OUROBOROS_TEST_TEMP_ROOT")
    if sys.platform != "darwin" or not parent:
        report["unsupported"] = "requires macOS and a safe_test disposable root"
    else:
        def checkpoint():
            report["candidate"] = case.result()
            write_json(out / "report.json", report)
        try:
            case = FirstOpen(args.candidate, Path(parent), out, checkpoint, deadline)
            checkpoint()
            case.run()
        except Exception as exc:
            report.setdefault("gaps", []).append(f"candidate stopped before its final observation ({public_reason(exc)})")
    report.update(judgement=judge(report), finished_at=now())
    write_json(out / "report.json", report)
    verdict = report["judgement"]["verdict"]
    print(f"macOS first opening {verdict}")
    return EXIT_CODES[verdict]


if __name__ == "__main__":
    sys.exit(main())
