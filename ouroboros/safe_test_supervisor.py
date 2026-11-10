"""Stage two of scripts/safe_test.py, entered only after environment replacement.

This is a command lifetime owner, not a recovery daemon. Its guardian ends the
command's tree if this owner is SIGKILLed, but then nothing records a result
and scratch stays; an exit or a matching filename never proves deletion safe.
A launch that failed stays red, yet with its result recorded and its first reap
clean it holds no scratch. Stage one prints the one final SAFE_TEST_RETAINED/RELEASED line.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from ouroboros.process_containment import ProcessContainer
from ouroboros.test_environment import (
    isolated_environment, kept_facts, release_except_retained, retain_tree, retained_folders,
)
from ouroboros.test_evidence import (
    EVIDENCE_ENV, EVIDENCE_POINTER, FileOutput, describe_gaps, evidence_directory, retain_uncaptured_source,
    source_evidence, write_record, remove_scratch_tree, report_diagnostic,
)

_GRACE_SEC = 10  # an INT root's export window; SIGTERM to forced reap (GNU timeout --kill-after)
_TIMEOUT_STATUS = 124  # GNU timeout's status for an expired deadline
_LISTED_MARKERS = 5  # release diagnostics name a few markers, never grant deletion


def release_scratch(root: Path) -> dict:
    """Release declared *direct* children, never unknown command data or symlinks. A retention marker
    keeps its own folder, and at the top of ``root`` all of it (``retained_folders``)."""
    kept = retained_folders(root)
    if root in kept:
        markers = [found for held in kept.values() for found in held]
        listed = ", ".join(map(str, markers[:_LISTED_MARKERS]))
        more = f" (+{len(markers) - _LISTED_MARKERS} more)" if len(markers) > _LISTED_MARKERS else ""
        return {"released": [], "retained": [str(root)], "kept": kept_facts(kept),
                "reason": f"retention marker or unreadable subtree: {listed}{more}"}
    # Declarations say reproducible, not idle. The caller already reaped its
    # container; no independent consumer may call this as a reaper.
    released = release_except_retained(root, kept, remove_scratch_tree, declared_only=True)
    # The launcher's empty defaults carry no unique content. Remove empty
    # directories bottom-up only; an arbitrary HOME/file is not test scratch.
    defaults = isolated_environment(root, Path(__file__).resolve().parents[1], source={}, create=False)
    skeleton = {Path(value) for value in defaults.values()
                if os.path.isabs(value) and Path(value).is_relative_to(root)}
    for path in sorted(skeleton | {root}, key=lambda path: len(path.parts), reverse=True):
        if path.is_symlink():
            continue
        try:
            path.rmdir()
        except OSError:
            pass
    return {"released": released, "retained": [str(root)] if root.exists() else [], "kept": kept_facts(kept)}


class _LiveDisplay(threading.Thread):
    """Copy the command's regular-file logs to this owner's stdio while it runs.

    One chunk in memory at a time, and no pipe EOF to wait for. The files stay
    the canonical result; a failed display write only stops displaying."""

    def __init__(self, root: Path):
        super().__init__(daemon=True)
        self.sources = [(root / "stdout.bin", sys.stdout), (root / "stderr.bin", sys.stderr)]
        self.offsets = [0, 0]
        self.error = None
        self.done = threading.Event()

    def run(self):
        while not self.done.wait(0.1):
            self._copy()
        self._copy()

    def _copy(self):
        for index, (path, target) in enumerate(self.sources):
            target = getattr(target, "buffer", None)
            if self.error is not None or target is None:
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(self.offsets[index])
                    while chunk := stream.read(64 * 1024):
                        target.write(chunk)
                        target.flush()
                        self.offsets[index] += len(chunk)
            except (OSError, ValueError) as exc:  # ValueError: stdio closed under us
                self.error = exc

    def finish(self):
        self.done.set()
        self.join(timeout=60)
        return self.error or (RuntimeError("display did not drain") if self.is_alive() else None)


def _evidence_directory(repo: Path, root: Path, explicit, owner_data_root=None) -> tuple[Path, str]:
    """The run's evidence; only an unwritable DEFAULT parent falls back into retained scratch."""
    try:
        return evidence_directory(repo, parent=explicit, owner_data_root=owner_data_root), ""
    except OSError as exc:
        if explicit or os.environ.get(EVIDENCE_ENV):
            raise  # A selected destination is never silently redirected.
        fallback = f"default evidence parent unwritable ({exc}); evidence kept inside retained scratch"
        if not retain_tree(root, fallback):
            raise
        return evidence_directory(repo, parent=root / "evidence"), fallback


def _record_source(repo: Path, destination: Path) -> str:
    """The tested recipe. A gap is named, never a refusal to run or a complete source."""
    try:
        record = source_evidence(repo, destination)
    except Exception as exc:  # describing the source failed, not the command
        gap = f"recipe not recorded ({type(exc).__name__}: {exc})"
        try:
            write_record(destination / "source.json", {"repository": str(repo), "source_snapshot": "incomplete",
                                                       "error": gap})
        except OSError:
            pass  # result.json and stderr still name the gap
    else:
        if not record.get("gaps"):
            return ""
        gap = "untracked inputs not recorded: " + describe_gaps(record["gaps"])
    retain_uncaptured_source(repo, gap)
    report_diagnostic(f"TEST_SOURCE_INCOMPLETE {destination}: {gap} (the command still runs)")
    return gap


def main(root, repo, command, evidence_parent=None, keep=False, timeout=None, owner_data_root=None) -> int:
    root, repo = Path(root), Path(repo)
    evidence = None
    proc = None
    container = ProcessContainer(lifeline=True)
    code, reap_error, evidence_error, fallback, source_gap = 2, "command not started", "", "", ""
    stopped = launch_error = escalation = ""
    reaped = False
    execution = "not_run"
    requested: list[int] = []
    previous = {}

    def request_stop(signum, _frame):
        requested.append(signum)  # The first decides; later ones never interrupt teardown.

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.signal(signum, request_stop)
    try:
        if keep:  # before the command: a test's folder goes at its end unless its root is held
            retain_tree(root, "debug retention")
        evidence, fallback = _evidence_directory(repo, root, evidence_parent, owner_data_root)
        if fallback:
            report_diagnostic(f"SAFE_TEST_EVIDENCE_FALLBACK {fallback}: {evidence}")
        (root / EVIDENCE_POINTER).write_text(str(evidence), encoding="utf-8")
        env = {**os.environ, EVIDENCE_ENV: str(evidence)}
        from ouroboros.test_environment import retain_diagnostic_outputs
        retain_diagnostic_outputs(root, env, cwd=repo)
        # All ordinary UI producers already honour this explicit destination.
        env.setdefault("OUROBOROS_UI_EVIDENCE_DIR", str(evidence / "ui"))
        source_gap = _record_source(repo, evidence / "source")
        write_record(evidence / "owner.json", {"supervisor_pid": os.getpid(), "scratch": str(root),
                     "command": command, "state": "starting (not a terminal outcome)"})
        print(f"SAFE_TEST_EVIDENCE {evidence}", file=sys.stderr, flush=True)
        output = FileOutput(root, "command")
        display = _LiveDisplay(output.root)
        try:
            if requested:
                stopped = f"signal {requested[0]}"
            else:
                # No pipe EOF wait: first wait for the direct child, then reap ALL
                # attributed descendants, then close/read the regular log files.
                execution = "attempted"
                try:
                    proc = container.spawn(command, cwd=repo, env=env,
                                           stdout=output.stdout, stderr=output.stderr)
                    execution = "executed"
                except Exception as exc:
                    # The command's own red result, not lost evidence: the reap and the
                    # records below still decide the scratch.
                    launch_error = f"{type(exc).__name__}: {exc}"
                    execution = "attempted" if container.root_created else "not_run"
                    report_diagnostic(f"SAFE_TEST_ERROR {launch_error}")
                if proc is not None:
                    display.start()
                    deadline = None if timeout is None else time.monotonic() + timeout
                    while proc.poll() is None and not requested and (deadline is None or time.monotonic() < deadline):
                        time.sleep(0.05)
                    if proc.poll() is None:
                        stopped = f"signal {requested[0]}" if requested else "timeout"
                        report_diagnostic(f"SAFE_TEST_INTERRUPTED by {stopped}: stopping the command and every process it started"
                                          if requested else f"SAFE_TEST_TIMEOUT after {timeout:g}s: stopping the command "
                                          "and every process it started")
                        if requested and requested[0] == signal.SIGINT and container.interrupt_root():
                            # spawn's new session cannot receive the terminal's INT.
                            # Give pytest time to export sessionfinish/JUnit first.
                            try:
                                proc.wait(timeout=_GRACE_SEC)
                            except subprocess.TimeoutExpired:
                                # The wait, not the output, observed it: the TERM below may cut whatever
                                # the root still had to do, such as a nested runner's final report.
                                escalation = (f"root still running after its {_GRACE_SEC} s interrupt window: escalated "
                                              "to TERM and the forced reap; its own completion is unconfirmed")
                                report_diagnostic(f"SAFE_TEST_INTERRUPT_ESCALATED {escalation}")
                        # Also when the root already exited: what it left gets TERM grace.
                        container.terminate_tree(_GRACE_SEC)
            reap_error = container.reap()
            reaped = True
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    reap_error = reap_error or "direct child did not exit after reap"
            if stopped:
                code = _TIMEOUT_STATUS if stopped == "timeout" else 128 + requested[0]
            elif proc is not None:
                code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
            if display.is_alive() and (failure := display.finish()) is not None:
                output.failed("display", failure)
            output.finish(command=command, returncode=None if proc is None else proc.returncode,
                          stopped=stopped, interrupt_escalation=escalation, containment_error=reap_error,
                          launch_error=launch_error)
            evidence_error = "; ".join(output.errors)
        finally:
            output.close()
    except BaseException as exc:
        evidence_error = f"{type(exc).__name__}: {exc}"
        report_diagnostic(f"SAFE_TEST_ERROR {evidence_error}")
    finally:
        if not reaped:
            # spawn may have created a process before raising. Even without a
            # returned handle, the container still holds its first reap's answer.
            reap_error = container.reap() or reap_error
        container.close()
        cleanup = {"released": [], "retained": [str(root)]}
        why = ("debug retention" if keep else reap_error or evidence_error)
        # Both writes carry every observed fact; only the evidence error, cleanup and retention move on.
        facts = {"command": command, "returncode": code, "containment_error": reap_error, "launch_error": launch_error,
                 "source_gap": source_gap, "scratch": str(root), "execution": execution, "stopped": stopped,
                 "interrupt_escalation": escalation, "evidence_fallback": fallback}
        if evidence is not None and not why:
            # Commit the result before deleting anything it describes. A failed
            # evidence write is itself a reason to keep the source tree.
            try:
                write_record(evidence / "result.json", {**facts, "evidence_error": evidence_error, "cleanup": "pending"})
                (root / EVIDENCE_POINTER).unlink()
                cleanup = release_scratch(root)
                why = cleanup.get("reason", "")
            except OSError as exc:
                why = evidence_error = f"evidence/cleanup incomplete: {exc}"
        if why:
            retain_tree(root, why)  # a marker beneath the root already holds its own folder, and only that
        kept = cleanup.pop("kept", None)
        if kept is None:  # the root's own hold: nothing was walked, and only its reason matters
            kept = [{"path": str(root), "reasons": [why.split("\n")[0]]}]
        reason = why or (f"{kept[0]['path']}: {kept[0]['reasons'][0]}" if kept else "")
        if evidence is not None:
            try:
                write_record(evidence / "result.json", {**facts, "evidence_error": evidence_error, "cleanup": cleanup,
                                                        "retention_reason": reason, "kept": kept,
                                                        "containment_tokens": [container.token] if kept else []})
            except OSError as exc:
                report_diagnostic(f"SAFE_TEST_EVIDENCE_INCOMPLETE {exc}")
        if reason:
            report_diagnostic(f"SAFE_TEST_RETENTION {reason}")
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    # Command status stays the status. Containment/evidence failures have their
    # own result fields and are not manufactured test failures.
    return code
