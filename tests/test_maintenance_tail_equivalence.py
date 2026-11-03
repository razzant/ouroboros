"""Completion contract checked before and after sharing the two maintenance tails."""
from __future__ import annotations

import io
import json

import pytest

from ouroboros import maintenance
from ouroboros import startup_historical_audit as audit


@pytest.mark.parametrize("kind", ["maintenance", "audit"])
@pytest.mark.parametrize("case", [
    "completed", "unknown", "failed", "nonzero", "malformed", "bool_count", "negative",
    "overflow", "stop_during_spawn", "stop_after_publication", "read_error", "record_error", "already_stopped",
])
def test_maintenance_completion_contract(tmp_path, monkeypatch, kind, case):
    module = maintenance if kind == "maintenance" else audit
    job = maintenance.MaintenanceChild() if kind == "maintenance" else audit.HistoricalAudit()
    counters = ({"jobs": {"evidence_packing": {"state": "done", "items": 1}}}
                if kind == "maintenance" else
                {"facts_written": 1, "manifests_checked": 2, "wall_seconds": 0, "cpu_seconds": 0})
    status = case if case in {"unknown", "failed"} else "completed"
    payload = {"status": status, **counters, "extra": "discarded"}
    first = next(iter(counters))
    if case in {"bool_count", "negative"}:
        payload[first] = True if case == "bool_count" else -1
    raw = json.dumps(payload).encode("utf-8")
    if case == "malformed":
        raw = b"{"
    if case == "overflow":
        raw = b" " * (audit._REPORT_BYTES + 1)
    signals, records, spawned = [], [], []

    class Pipe(io.BytesIO):
        def read(self, size=-1):
            assert job._process is child, "publish custody before reading the child"
            assert size == audit._REPORT_BYTES + 1
            if case == "stop_after_publication":
                job.stop()
            if case == "read_error":
                raise OSError("synthetic pipe failure")
            return super().read(size)

    class Child:
        pid = 4242
        stdout = Pipe(raw)

        def kill(self):
            assert job._process is self
            signals.append(self.pid)

        def wait(self):
            return -9 if signals else 1 if case == "nonzero" else 0

    child = Child()

    def spawn(root, repo, purpose):
        assert root == repo == tmp_path and job._process is None
        spawned.append(purpose)
        if case == "stop_during_spawn":
            job.stop()
            assert signals == [], "Stop cannot signal an unpublished handle"
        return child

    def record(root, phase, **facts):
        assert root == tmp_path
        if "duration_seconds" in facts:
            assert facts.pop("duration_seconds") >= 0
        records.append({"phase": phase, **facts})
        if phase == "started":
            assert job._process is child
            if case == "record_error":
                raise OSError("synthetic record failure")

    monkeypatch.setattr(module, "spawn_maintenance_job", spawn)
    if kind == "maintenance":
        monkeypatch.setattr(maintenance, "due", lambda _root: ["evidence_packing"])
        monkeypatch.setattr(maintenance, "_record", record)
    else:
        monkeypatch.setattr(job, "_record", record)
    if case == "already_stopped":
        job.stop()
    job._run(tmp_path, tmp_path)

    if case == "already_stopped":
        assert spawned == signals == []
        assert records == ([] if kind == "maintenance" else [{"phase": "stopped"}])
        assert job._process is None
        return
    assert spawned == ["maintenance" if kind == "maintenance" else "startup_historical_audit"]
    assert records[0] == {"phase": "started", "pid": 4242}
    errors = case in {"read_error", "record_error"}
    stopped = case in {"stop_during_spawn", "stop_after_publication"}
    invalid = case in {"malformed", "bool_count", "negative", "overflow"}
    assert signals == ([4242] if stopped or case == "overflow" or (errors and kind == "audit") else [])
    if errors:
        assert records[1:] == [{"phase": "failed", "exception_class": "OSError"}]
    else:
        phase = "stopped" if stopped else "unknown" if invalid else "failed" if case == "nonzero" else status
        assert records[1:] == [{"phase": phase, "exit_code": child.wait(), **({} if invalid else counters)}]
    assert child.stdout.closed is (case != "record_error")
    assert job._process is (None if kind == "maintenance" else child)
    assert job._stopped.is_set() is (stopped or (errors and kind == "audit"))
