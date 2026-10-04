"""Harmless real processes used only by the isolated lifecycle qualification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def process_body():
    marker = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    children = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
                for _ in range(count)]
    child_pids = {("child_pid" if count == 1 else f"child_pid_{index + 1}"): child.pid
                  for index, child in enumerate(children)}
    marker.write_text(json.dumps({"pid": os.getpid(), **child_pids}), encoding="utf-8")
    time.sleep(3600)


class ProcessCohort:
    """Spawn through production executor/custody APIs, preserving native identities.

    The operator drives these APIs in a separate process, so this proves durable
    cross-process cleanup. It does not claim model-selected tool dispatch or a live
    Claudexor engine. The retained daemon-shaped process is outside the server Job.
    """
    def __init__(self, root, source, task_id, wait_until, daemon_descendants=1):
        from ouroboros import process_custody, workspace_executor
        from ouroboros.claudexor_daemon import CUSTODY_PURPOSE
        self.root = root
        self.data = root / "data"
        self.workspace = root / "process-workspace"
        self.workspace.mkdir()
        self.executor = workspace_executor
        self.daemon = None
        self.foreground = None
        self.foreground_outcome = {}
        self.markers = {}
        entries = [json.loads(line) for line in (self.data / "state/process_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
        sessions = {entry["session_id"] for entry in entries if str(entry.get("purpose", "")).startswith("worker:")}
        if len(sessions) != 1:
            raise RuntimeError(f"expected exactly one worker custody generation, got {sessions}")
        process_custody.adopt_session_id(sessions.pop())
        self.ctx = SimpleNamespace(drive_root=self.data, task_id=task_id, executor_ref={
            "type": "local", "id": "shutdown-qualification", "network": "host",
            "workspace_host_path": str(self.workspace), "workspace_backend_path": "/qualification",
        })
        self.interpreter = getattr(sys, "_base_executable", sys.executable)
        self.script = Path(__file__).resolve()
        self.wait_until = wait_until
        self.daemon_purpose = CUSTODY_PURPOSE
        self.process_custody = process_custody
        self.daemon_descendants = daemon_descendants

    def _command(self, label):
        count = self.daemon_descendants if label == "retained-daemon" else 1
        return [self.interpreter, str(self.script), str(self.workspace / (label + ".json")), str(count)]

    def _ready(self, label):
        marker = self.workspace / (label + ".json")
        def load():
            try:
                return json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
        self.markers[label] = self.wait_until(load, 15, label + " native PID marker")

    def start(self):
        def execute():
            try:
                result = self.executor.execute(self.ctx, self._command("foreground"), self.workspace, 600)
                self.foreground_outcome.update(returncode=result.returncode)
            except Exception as exc:
                self.foreground_outcome.update(error=repr(exc))
        self.foreground = threading.Thread(target=execute, daemon=True)
        self.foreground.start()
        self._ready("foreground")
        self.service = self.executor.start_service(self.ctx, name="qualification-service",
            cmd=self._command("service"), host_cwd=self.workspace, cwd_root="active_workspace",
            readiness={}, outputs=[], before_outputs={}, keep_alive=True)
        self._ready("service")
        self.daemon = self.process_custody.spawn_supervised(self._command("retained-daemon"),
            drive_root=self.data, purpose=self.daemon_purpose, scope="daemon", cwd=self.workspace,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._ready("retained-daemon")
        if self.daemon.pid != self.markers["retained-daemon"]["pid"]:
            raise RuntimeError("daemon Popen identity differs from native payload PID")
        expected = self.daemon.pid
        observed = self.process_custody.live_daemon_root_pids(self.data,
            purposes={self.daemon_purpose}, retained_purposes={self.daemon_purpose}, strict=True)
        if expected not in observed:
            raise RuntimeError("native daemon fingerprint was not admitted by production custody reader")
        child_pids = set(self.markers["retained-daemon"].values()) - {expected}
        if len(child_pids) != self.daemon_descendants:
            raise RuntimeError("retained daemon marker count does not match the requested geometry")
        if os.name != "nt":
            census = [tuple(map(int, row.split())) for row in subprocess.check_output(
                ["ps", "-axo", "pid=,ppid="], text=True).splitlines() if row.strip()]
            actual_children = {pid for pid, ppid in census if ppid == expected}
            if actual_children != child_pids:
                raise RuntimeError("native PID/PPID census differs from retained daemon fixture markers")
        else:
            import psutil
            actual_children = {child.pid for child in psutil.Process(expected).children()}
            if actual_children != child_pids:
                raise RuntimeError("native Windows PID/PPID census differs from retained daemon markers")
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in (self.data / "state/workspace_executor_processes").glob("*.json")]
        expected_pids = {self.markers[name]["pid"] for name in ("foreground", "service")}
        if not expected_pids <= {record.get("host_pid") for record in records}:
            raise RuntimeError("foreground/service durable records do not match payload PIDs")
        return {"native_pids": self.markers, "service": self.service,
                "executor_records": records,
                "daemon_retained_by_custody": sorted(observed),
                "daemon_descendants": self.daemon_descendants,
                "driver_scope": "production APIs from fixture process; source cleanup uses durable records",
                "real_claudexor_engine": "NOT_RUN"}

    def after_quit(self):
        from ouroboros.platform_layer import pid_is_alive
        from ouroboros.process_containment import pid_is_zombie
        def live(pid):
            return pid_is_alive(pid) and not pid_is_zombie(pid)
        state = {label: {role: live(pid) for role, pid in ids.items()}
                 for label, ids in self.markers.items()}
        expected = all(not alive for label in ("foreground", "service") for alive in state[label].values())
        expected = expected and all(state["retained-daemon"].values())
        return {"alive_after_quit_before_fixture_cleanup": state, "passed": expected}

    def cleanup(self):
        from ouroboros.platform_layer import kill_process_tree
        self.executor.kill_all_foreground(self.data)
        self.executor.kill_all_services(self.data)
        if self.daemon is not None:
            kill_process_tree(self.daemon)
            self.daemon.wait(timeout=10)
        if self.foreground is not None:
            self.foreground.join(10)


if __name__ == "__main__":
    process_body()
