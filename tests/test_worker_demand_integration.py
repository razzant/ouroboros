"""Keyless native small-install and burst measurements on the serving candidate.

Run under safe_test with `-m integration -o addopts=''`. JUnit properties retain
the raw PID/birth/RSS/latency samples, cap and candidate identity. Parent qualification
can run the same census against its pristine base for A/B; these assertions pin
the candidate, not a fabricated baseline. Linux/Windows/packaged need native runs.
"""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from ouroboros.install_identity import acting_as
from ouroboros.platform_layer import pid_provably_gone, process_start_time
from ouroboros.process_custody import _read_ledger_strict
from ouroboros.queue_observation import load_queue_document
from tests.candidate_checkout import candidate_checkout
from tests.system_e2e.harness import (
    ArtifactOracle, ScriptedStubModel, classify_call, keyless_settings, start_server,
    wait_durable_result, wait_until,
)

pytestmark = [pytest.mark.integration]


def native_worker_census(root):
    """Ledger ownership plus positive native liveness and measured current RSS."""
    readable, ledger = _read_ledger_strict(root)
    if not readable:
        return None
    rows = []
    for row in ledger:
        if not str(row.get("purpose") or "").startswith("worker:"):
            continue
        pid = row["pid"]
        if pid_provably_gone(pid):
            continue
        birth = process_start_time(pid)
        if not birth and pid_provably_gone(pid):
            continue
        fingerprint = row["fingerprint"]
        assert birth and birth == (fingerprint.get("start_time_boot") or fingerprint.get("start_time"))
        rss = None
        if sys.platform in {"darwin", "linux"}:
            measured = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                      capture_output=True, text=True, timeout=5)
            if measured.returncode == 0 and measured.stdout.strip():
                rss = int(measured.stdout.strip()) * 1024
        rows.append({"pid": pid, "birth": birth, "rss_bytes": rss, "purpose": row["purpose"]})
    return rows


def worker_ended(pid, birth):
    if pid_provably_gone(pid):
        return True
    current = process_start_time(pid)
    return bool(current and current != birth)


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_project_workers_follow_demand_and_retire_between_bursts(tmp_path, record_property, cap):
    release = threading.Event()
    calls, lock = [], threading.Lock()

    def hold(body):
        if body.get("tools") and classify_call(body) == "agent":
            with lock:
                calls.append(time.monotonic())
            assert release.wait(120), "test did not release its keyless model call"

    source = Path(__file__).resolve().parents[1]
    measurements = {"cap": cap, "platform": sys.platform, "samples": []}
    with candidate_checkout(source, tmp_path / "candidate", origin_proof=True) as candidate:
        measurements["candidate"] = candidate.identity
        with ScriptedStubModel(gate=hold) as stub:
            started = time.monotonic()
            # Both are explicit choices: an unmarked 10/3 is a legacy default.
            # The loopback model must admit the whole burst. Physical memory
            # admission stays live, so the largest case needs room for ten workers.
            settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=cap, _owner_cap_format=1,
                                        OUROBOROS_MODEL_MAX_CONCURRENCY=cap, _model_cap_format=1)
            server = start_server(candidate, tmp_path / "instance", settings)
            oracle = ArtifactOracle(server.data_root)

            def census(phase, since, task_ids):
                count = len(task_ids)

                def observed():
                    snapshot = load_queue_document(server.data_root)
                    if ({row["id"] for row in snapshot.get("running", [])} != task_ids
                            or snapshot.get("worker_total") != count
                            or snapshot.get("acting_worker_count") != count
                            or snapshot.get("assignable_idle_workers") != 0
                            or snapshot.get("lane_carrier_total") != 1):
                        return None
                    rows = native_worker_census(server.data_root)
                    if rows is None or len(rows) != count + 1:
                        return None
                    ready = {(row["pid"], row["process_birth"], f"worker:{row['worker_id']}")
                             for row in oracle.events("worker_ready")}
                    if all((row["pid"], row["birth"], row["purpose"]) in ready for row in rows):
                        return snapshot, rows

                observation = wait_until(observed, 120)
                assert observation, (phase, task_ids, load_queue_document(server.data_root),
                                     native_worker_census(server.data_root))
                snapshot, rows = observation
                assert snapshot["max_workers"] == cap
                assert not snapshot.get("worker_pool_disabled_reason")
                project_slots = {f"worker:{row['worker_id']}" for row in snapshot["running"]}
                projects = [row for row in rows if row["purpose"] in project_slots]
                assert len(project_slots) == len(projects) == count
                assert {row["purpose"] for row in projects} == project_slots
                carrier, = [row for row in rows if row["purpose"] not in project_slots]
                sample = {"phase": phase, "elapsed_sec": time.monotonic() - since, "snapshot": snapshot,
                          "project_workers": projects, "conversation_carrier": carrier}
                measurements["samples"].append(sample)
                return (carrier["pid"], carrier["birth"]), {(row["pid"], row["birth"]) for row in projects}

            try:
                carrier, _ = census("ready_idle", started, set())
                retired = set()
                for phase, count in (("first_task", 1), ("burst_0", cap), ("burst_1", cap)):
                    release.clear()
                    with lock:
                        calls.clear()
                    before = time.monotonic()
                    ids = {server.submit(f"Independent item {phase}/{i}. Answer briefly.") for i in range(count)}
                    assert len(ids) == count and all(ids)
                    assert wait_until(lambda: len(calls) == count, 120), (phase, load_queue_document(server.data_root))
                    spare, projects = census(phase, before, ids)
                    assert spare == carrier and carrier not in projects
                    assert len(projects) == count and projects.isdisjoint(retired)
                    release.set()
                    # Read the launched test install's own durable results with its identity.
                    with acting_as(server.data_root):
                        for tid in ids:
                            assert wait_durable_result(oracle, tid)["status"] == "completed"
                    assert census(f"idle_after_{phase}", before, set()) == (carrier, set())
                    assert wait_until(lambda: all(worker_ended(pid, birth) for pid, birth in projects), 90), projects
                    retired.update(projects)
            finally:
                release.set()
                record_property("native_pool_measurements", json.dumps(measurements))
                record_property("retained_data_root", str(server.data_root))
                server.stop()
