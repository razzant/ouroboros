"""Explicit historical seal audit (owner CLI or rebuild job), never automatic at boot.

The audit is diagnostic, O(history), and uses the unchanged monetary/archive
owners in a separate interpreter. It cannot hold server readiness or the GIL.
Stop never waits on the launch lock, spawn, output or exit confirmation: the
spawner publishes its handle then rechecks Stop, closing both race orderings.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

_REPORT_BYTES = 4096


def spawn_maintenance_job(data_root, repo_dir, purpose, *, args=(), **popen):
    """The audit, the maintenance child and the storage count share session custody, a dedicated
    group and containment env. ARGS follow the child's standard arguments; POPEN overrides the
    stream defaults (a stdin pipe for a child that reads its input)."""
    from ouroboros.config import (SETTINGS_PATH, get_deliverables_root, get_subagent_projects_root,
                                  get_subagent_worktree_root, resolve_app_root)
    from ouroboros.launch_roots import BASIS_ENV, WorkRootUnavailable
    from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX
    from ouroboros.process_custody import current_custody_session_id, lifeline_parent_env, spawn_supervised

    env = {key: os.environ[key] for key in (
        "PATH", "HOME", "USERPROFILE", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
    ) if key in os.environ}
    # Dropping this token hides the descendant from test/Panic containment reaping.
    env.update({key: value for key, value in os.environ.items() if key.startswith(CONTAINMENT_ENV_PREFIX)})
    # The three work roots as THIS process resolved them (setting, environment or default): a child
    # that counts or retires them must see the configured ones (Docker keeps them outside HOME).
    # An undecided one stays unset, so the child also gets this host's APP_ROOT and basis: left to
    # its own default APP_ROOT it would pick the historical folder the host refused (#1239).
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo_dir), OUROBOROS_DATA_DIR=str(data_root),
               OUROBOROS_REPO_DIR=str(repo_dir), OUROBOROS_SETTINGS_PATH=str(SETTINGS_PATH),
               OUROBOROS_APP_ROOT=str(resolve_app_root()), **lifeline_parent_env())
    for key, root in (("OUROBOROS_SUBAGENT_WORKTREE_ROOT", get_subagent_worktree_root),
                      ("OUROBOROS_SUBAGENT_PROJECTS_ROOT", get_subagent_projects_root),
                      ("OUROBOROS_DELIVERABLES_ROOT", get_deliverables_root)):
        try:
            env[key] = root()
        except WorkRootUnavailable:  # undecided: the child meets the same refusal, never a guess
            pass
    if os.environ.get(BASIS_ENV):
        env[BASIS_ENV] = os.environ[BASIS_ENV]
    # Never share the server group: custody-publication failure kills this group.
    return spawn_supervised(
        [sys.executable, "-m", "ouroboros." + purpose, "--data-root", str(data_root),
         "--custody-session", current_custody_session_id(), *args],
        drive_root=data_root, purpose=purpose, scope="session", cwd=str(repo_dir), env=env,
        **{"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, **popen},
    )


def read_job_lines(process, on_line, signal, *, line_bytes=16 * _REPORT_BYTES):
    """Streaming IPC, the report reader's sibling for a child that talks while it works (the
    storage count): ON_LINE gets each complete line; a line past the bound signals the captured
    child and ends the read; the pipe always closes."""
    try:
        while True:
            raw = process.stdout.readline(line_bytes + 1)
            if not raw:
                break
            if len(raw) > line_bytes:
                signal(process)
                break
            on_line(raw)
        return process.wait()
    finally:
        process.stdout.close()


def read_job_report(process, signal):
    """Bounded IPC; overflow signals the captured child and the pipe always closes."""
    try:
        raw = process.stdout.read(_REPORT_BYTES + 1)
        if len(raw) > _REPORT_BYTES:
            signal(process)
        return raw, process.wait()
    finally:
        process.stdout.close()


def finish_maintenance_job(job, process, data_root, started, parse, record):
    """Publish custody before rechecking Stop; owners retain report schemas and error policy."""
    with job._lock:
        job._process = process
    if job._stopped.is_set():
        job._signal(process)
    record(data_root, phase="started", pid=process.pid)
    raw, exit_code = read_job_report(process, job._signal)
    fields = parse(raw)
    phase = ("stopped" if job._stopped.is_set() else "unknown" if not fields
             else "failed" if exit_code else fields.pop("status", "unknown"))
    fields.pop("status", None)
    record(data_root, phase=phase, exit_code=exit_code,
           duration_seconds=time.monotonic() - started, **fields)
    return fields


class HistoricalAudit:
    def __init__(self):
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._launched = False
        self._process = None

    def start(self, data_root: Path, repo_dir: Path) -> None:
        with self._lock:
            if self._launched or self._stopped.is_set():
                return
            self._launched = True
        # No filesystem, spawning or waiting under the state lock. Failure is
        # terminal for this invocation; a later explicit invocation may retry.
        try:
            threading.Thread(target=self._run, args=(Path(data_root), Path(repo_dir)),
                             name="startup-historical-audit", daemon=True).start()
        except Exception as exc:
            self._record(data_root, "failed", exception_class=type(exc).__name__)

    def stop(self) -> None:
        self._stopped.set()
        process = self._process
        if process is not None:
            self._signal(process)

    @staticmethod
    def _signal(process) -> None:
        # The child spawns no descendants and runs in its OWN process group
        # (no Job breakaway), so no signal aimed at it can reach the server or
        # its siblings. Popen.kill signals only our captured child, portably,
        # with no exit wait or audit lock.
        try:
            process.kill()
        except OSError:
            pass

    @staticmethod
    def _record(data_root, phase, **facts):
        from ouroboros.utils import append_jsonl, utc_now_iso
        append_jsonl(Path(data_root) / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "startup_historical_audit", "phase": phase, **facts,
        })

    def _run(self, data_root, repo_dir):
        started = time.monotonic()
        try:
            if self._stopped.is_set():
                self._record(data_root, "stopped")
                return
            process = spawn_maintenance_job(data_root, repo_dir, "startup_historical_audit")
            finish_maintenance_job(self, process, data_root, started, _report_fields, self._record)
        except Exception as exc:
            # Output has a closed, bounded schema. No raw exception, path or
            # archived payload can reach the parent diagnostic record.
            self.stop()
            self._record(data_root, "failed", exception_class=type(exc).__name__,
                         duration_seconds=time.monotonic() - started)


def _report_fields(raw):
    try:
        value = json.loads(raw) if len(raw) <= _REPORT_BYTES else None
        if not isinstance(value, dict) or value.get("status") not in {"completed", "unknown", "failed"}:
            return {}
        fields = {"status": value["status"]}
        for key in ("facts_written", "manifests_checked", "wall_seconds", "cpu_seconds"):
            number = value.get(key)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                return {}
            fields[key] = number
        if value.get("exception_class"):
            name = value["exception_class"]
            if isinstance(name, str) and name.isidentifier() and len(name) <= 100:
                fields["exception_class"] = name
        return fields
    except (ValueError, UnicodeError, TypeError, OverflowError):
        return {}


audit = HistoricalAudit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--custody-session", default="")
    args = parser.parse_args()
    os.environ["OUROBOROS_DATA_DIR"] = str(Path(args.data_root).resolve())
    # Roots are bound before importing readers that capture config globals.
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline
    adopt_session_id(args.custody_session)
    start_parent_lifeline(label="startup_historical_audit")
    started, cpu = time.monotonic(), time.process_time()
    record = {"status": "failed", "facts_written": 0, "manifests_checked": 0}
    try:
        from ouroboros.model_send_seal import reconcile_model_send_seals
        result = reconcile_model_send_seals(Path(args.data_root))
        record.update(status=result["status"], facts_written=result["facts_written"],
                      manifests_checked=result["manifests_checked"])
    except Exception as exc:
        record["exception_class"] = type(exc).__name__
    record.update(wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu)
    print(json.dumps(record, separators=(",", ":")), flush=True)
    return 1 if record["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
