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

from ouroboros.platform_layer import pid_is_alive, process_start_time
from ouroboros.process_custody import _read_ledger
from tests.candidate_checkout import candidate_checkout
from tests.system_e2e.harness import (
    ArtifactOracle, ScriptedStubModel, keyless_settings, start_server,
    wait_durable_result, wait_until,
)

pytestmark = [pytest.mark.integration]


def native_worker_census(root):
    """Ledger ownership plus positive native liveness and measured current RSS."""
    rows = []
    for row in _read_ledger(root):
        if not str(row.get("purpose") or "").startswith("worker:"):
            continue
        pid = row["pid"]
        if not pid_is_alive(pid):
            continue
        birth = process_start_time(pid)
        fingerprint = row["fingerprint"]
        assert birth == (fingerprint.get("start_time_boot") or fingerprint.get("start_time"))
        rss = None
        if sys.platform in {"darwin", "linux"}:
            measured = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                      capture_output=True, text=True, timeout=5)
            if measured.returncode == 0 and measured.stdout.strip():
                rss = int(measured.stdout.strip()) * 1024
        rows.append({"pid": pid, "birth": birth, "rss_bytes": rss, "purpose": row["purpose"]})
    return rows


@pytest.mark.parametrize("cap", [1, 3, 10])
def test_small_install_then_repeated_burst_keeps_one_spare_without_lowering_cap(tmp_path, record_property, cap):
    release = threading.Event()
    calls, lock = [], threading.Lock()

    def hold(body):
        if body.get("tools"):
            with lock:
                calls.append(time.monotonic())
            assert release.wait(120), "test did not release its keyless model call"

    source = Path(__file__).resolve().parents[1]
    measurements = {"cap": cap, "platform": sys.platform, "samples": []}
    with candidate_checkout(source, tmp_path / "candidate", origin_proof=True) as candidate:
        with ScriptedStubModel(gate=hold) as stub:
            started = time.monotonic()
            server = start_server(candidate, tmp_path / "instance",
                                  keyless_settings(stub, OUROBOROS_MAX_WORKERS=cap))
            oracle = ArtifactOracle(server.data_root)

            def census(phase, since):
                rows = native_worker_census(server.data_root)
                sample = {"phase": phase, "elapsed_sec": time.monotonic() - since,
                          "workers": rows, "snapshot": oracle.queue_snapshot()}
                measurements["samples"].append(sample)
                return rows

            try:
                assert wait_until(lambda: oracle.queue_snapshot().get("assignable_idle_workers") == 1, 90)
                initial = census("ready_idle", started)
                assert len(initial) == 1
                # The first small task uses the already-ready process, with no rotation.
                before = time.monotonic()
                tid = server.submit("Complete this small task with a short answer.")
                assert wait_until(lambda: len(calls) == 1, 90)
                assert census("first_task", before)[0]["pid"] == initial[0]["pid"]
                release.set()
                assert wait_durable_result(oracle, tid)["status"] == "completed"
                assert wait_until(lambda: not oracle.running_ids(), 60)
                for wave in range(2):
                    release.clear()
                    with lock:
                        calls.clear()
                    before = time.monotonic()
                    ids = [server.submit(f"Independent burst item {wave}/{i}; answer briefly.")
                           for i in range(cap)]
                    assert wait_until(lambda: len(oracle.running_ids()) == cap and len(calls) == cap, 120)
                    current = census(f"burst_{wave}", before)
                    assert len(current) == cap
                    assert oracle.queue_snapshot()["acting_worker_count"] == cap
                    release.set()
                    for tid in ids:
                        assert wait_durable_result(oracle, tid)["status"] == "completed"
                    assert wait_until(lambda: not oracle.running_ids()
                                      and oracle.queue_snapshot().get("worker_total") == 1
                                      and len(native_worker_census(server.data_root)) == 1, 90)
                    idle = census(f"idle_after_{wave}", before)
                    assert idle[0]["pid"] == initial[0]["pid"]
                assert not oracle.queue_snapshot().get("worker_pool_disabled_reason")
            finally:
                release.set()
                record_property("native_pool_measurements", json.dumps(measurements))
                record_property("retained_data_root", str(server.data_root))
                server.stop()
