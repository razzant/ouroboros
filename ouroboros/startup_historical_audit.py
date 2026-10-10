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
        from ouroboros.config import SETTINGS_PATH
        from ouroboros.process_custody import current_custody_session_id, spawn_supervised

        started = time.monotonic()
        try:
            if self._stopped.is_set():
                self._record(data_root, "stopped")
                return
            from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX
            env = {key: os.environ[key] for key in (
                "PATH", "HOME", "USERPROFILE", "SystemRoot", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
            ) if key in os.environ}
            # Containment membership is an env token every descendant inherits:
            # a child that drops it is invisible to the container reap that a
            # test fixture (or Panic) relies on to prove this data root quiet.
            env.update({key: value for key, value in os.environ.items()
                        if key.startswith(CONTAINMENT_ENV_PREFIX)})
            env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo_dir),
                       OUROBOROS_DATA_DIR=str(data_root), OUROBOROS_REPO_DIR=str(repo_dir),
                       OUROBOROS_SETTINGS_PATH=str(SETTINGS_PATH))
            # Dedicated group by default: spawn_supervised kills the child's
            # whole POSIX group when the custody record cannot be written, so
            # sharing the server's group would let a custody-write failure
            # SIGKILL the server and every sibling under it.
            process = spawn_supervised(
                [sys.executable, "-m", "ouroboros.startup_historical_audit", "--data-root", str(data_root),
                 "--custody-session", current_custody_session_id()],
                drive_root=data_root, purpose="startup_historical_audit", scope="session",
                cwd=str(repo_dir), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._process = process
            if self._stopped.is_set():
                self._signal(process)
            self._record(data_root, "started", pid=process.pid)
            try:
                raw = process.stdout.read(_REPORT_BYTES + 1)
                if len(raw) > _REPORT_BYTES:
                    self._signal(process)
                exit_code = process.wait()
            finally:
                process.stdout.close()
            fields = _report_fields(raw)
            phase = ("stopped" if self._stopped.is_set() else
                     "unknown" if not fields else
                     "failed" if exit_code else fields.pop("status", "unknown"))
            fields.pop("status", None)
            self._record(data_root, phase, exit_code=exit_code,
                         duration_seconds=time.monotonic() - started, **fields)
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
