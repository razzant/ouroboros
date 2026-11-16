"""A real server and project worker; only the memory reading and model are synthetic."""
from __future__ import annotations

import json
import os

import pytest

from tests.system_e2e.harness import (
    ArtifactOracle, KeylessIsolatedServer, LANE_MOCK, ModelGate, ScriptedStubModel,
    keyless_settings, require_lane, wait_durable_result, wait_until, write_settings_file,
)
from tests.system_e2e.test_presence_resilience import s33_candidate  # noqa: F401
from tests._stability_evidence import retain_server_evidence


class _MemoryServer(KeylessIsolatedServer):
    def _env(self):
        env = super()._env()
        env["PYTHONPATH"] = os.pathsep.join((str(self.data_root / "hook"), str(self.clone)))
        env["STABILITY_MEMORY_FILE"] = str(self.data_root / "memory-verdict")
        return env


_HOOK = '''import sys
if sys.argv[0].endswith("server.py"):
    import os, pathlib
    from ouroboros.host_memory import MemorySnapshot
    from supervisor import pool_admission, queue, worker_pool_lifecycle
    def measured():
        assert not queue._queue_lock._is_owned(), "memory probe under Q"
        assert not worker_pool_lifecycle._WORKER_LIFECYCLE_LOCK._is_owned(), "memory probe under lifecycle"
        value = pathlib.Path(os.environ["STABILITY_MEMORY_FILE"]).read_text()
        return None if value == "unknown" else MemorySnapshot(16 * 2**30, 12 * 2**30, value == "critical")
    pool_admission.physical_memory = measured
'''


@pytest.mark.integration
@pytest.mark.parametrize("recovered", ["healthy", "unknown"])
def test_real_project_wait_cancel_readiness_and_completion(s33_candidate, tmp_path, recovered):  # noqa: F811
    require_lane(LANE_MOCK)
    data = tmp_path / "data"
    (data / "hook").mkdir(parents=True)
    (data / "hook" / "sitecustomize.py").write_text(_HOOK)
    verdict = data / "memory-verdict"
    verdict.write_text("critical")
    gate = ModelGate(lambda body: "STABILITY-RUN" in json.dumps(body), timeout=180)
    facts = {"recovered": recovered, "tree": s33_candidate.identity}
    with ScriptedStubModel(final_answer="Final answer: stability complete.", gate=gate) as stub:
        write_settings_file(data / "settings.json", keyless_settings(
            stub, OUROBOROS_MAX_WORKERS=1,
            OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(tmp_path / "worktrees")))
        server = _MemoryServer(s33_candidate, data, data / "settings.json")
        oracle = ArtifactOracle(data)
        try:
            server.start(ready_timeout=300)
            cancelled = server.submit("Answer briefly. STABILITY-CANCEL")

            def held(tid):
                snap = oracle.queue_snapshot()
                return (any(row.get("id") == tid for row in snap.get("pending", []))
                        and snap.get("memory_admission", {}).get("reason") == "memory_pressure_critical")

            assert wait_until(lambda: held(cancelled), 60), oracle.queue_snapshot()
            assert not gate.arrived.is_set()
            assert server.cancel_task(cancelled)["status"] in (200, 202)
            facts["cancelled"] = wait_durable_result(oracle, cancelled, timeout=120)["status"]
            assert facts["cancelled"] == "cancelled"
            running = server.submit("Answer briefly. STABILITY-RUN")
            assert wait_until(lambda: held(running), 60), oracle.queue_snapshot()
            assert not gate.arrived.is_set(), "critical queued task reached the model"
            verdict.write_text(recovered)
            assert gate.arrived.wait(120), oracle.queue_snapshot()
            assert running in oracle.running_ids()
            # Admission is not a running-work killer. The existing worker finishes
            # the already-dispatched round even after a new critical reading.
            verdict.write_text("critical")
            gate.release.set()
            facts["completed"] = wait_durable_result(oracle, running, timeout=180)["status"]
            assert facts["completed"] == "completed"
            events = oracle.events()
            facts["ready_pids"] = sorted({row["pid"] for row in events
                                           if row.get("type") == "worker_ready"})
            assert len(facts["ready_pids"]) >= 2, "conversation and project children must both report readiness"
        finally:
            gate.release.set()
            server.stop()  # real contained children, positive physical teardown
            facts["teardown"] = "confirmed"
            retain_server_evidence(data, facts)
            print("STABILITY_POOL_FACTS", json.dumps(facts, default=str))
