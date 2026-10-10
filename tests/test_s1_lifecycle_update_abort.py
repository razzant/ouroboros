"""Opt-in REAL same-server managed-update ABORT consumer; host execution is owed.

Run only via unchanged scripts/safe_test.py with OUROBOROS_E2E_DEEP=mock.
tests/test_update_abort_retention.py owns the abort matrix with simulated OS
processes; this one case keeps every process real. A fixture-only newer release
fails production's pre-restart smoke, so /api/update/apply really quiesces,
checks out, smokes, rolls back and recreates the pool INSIDE the same server:
no restart, exec, boot or ACK follows. Nothing in that path is mocked; only the
model is an HTTP loopback peer. Not a version-migration or merge claim.
"""
from __future__ import annotations

import os
import subprocess
import time

import pytest

from devtools.benchmarks.common.server_runner import _api, _api_status
from ouroboros.server_process import read_service_bindings
from tests.s1_lifecycle_process_support import (
    SOURCE, jsonl, product_hashes, read_json, scenario, submit, wait_for, write_json,
)
from tests.system_e2e.harness import ArtifactOracle, LANE_MOCK, require_lane
from tests.test_s1_lifecycle_process import (
    _initial_read, _money_preserved, _restored_read, _rows, _usage, _worker_owner,
)

pytestmark = [pytest.mark.serial, pytest.mark.integration, pytest.mark.skipif(
    os.name != "posix", reason="This harness qualifies local POSIX process facts; Windows remains separate")]


def _git(cwd, *args):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_real_update_abort_returns_saved_work_through_a_recreated_pool_in_the_same_server(tmp_path_factory):
    require_lane(LANE_MOCK)
    # Short root: macOS AF_UNIX limit, as in the companion lifecycle file.
    root = tmp_path_factory.mktemp("s1")
    with scenario(root, managed=True) as (server, model, workspace, evidence):
        task_id = submit(server, workspace, "root")
        saved = _initial_read(server, model, "root", task_id, evidence)
        old_pid = int(_worker_owner(server, task_id)["local_answer_owner_pid"])
        before = _usage(server, [task_id], evidence, "before")
        # A fixture-only newer release that production's pre-restart smoke
        # (py_compile server.py) must refuse. It never reaches server memory.
        upstream = root / "managed-source"
        with (upstream / "server.py").open("a", encoding="utf-8") as stream:
            stream.write("\ndef s1_update_abort_fixture(:\n")
        subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-q", "-am", "Test-only release failing the pre-restart smoke"],
                       cwd=upstream, check=True)
        plan = _api(server.base_url, "POST", "/api/update/preflight", {}, timeout=180).get("merge_plan", {})
        write_json(evidence / "update-plan.json", plan)
        assert plan.get("base_sha") == _git(server.clone, "rev-parse", "HEAD"), plan
        assert plan.get("target_sha") == _git(upstream, "rev-parse", "HEAD"), plan
        start = time.monotonic()
        response = _api_status(server.base_url, "POST", "/api/update/apply", {
            "strategy": "replace", "expected_base_sha": plan["base_sha"],
            "expected_target_sha": plan["target_sha"], "confirm_recovery": True,
        }, timeout=180)
        write_json(evidence / "update-response.json", {**response, "elapsed_sec": time.monotonic() - start})
        body = response["body"]
        assert response["status"] == 409 and body.get("rolled_back") is True, response
        assert body.get("error") == "pre-restart smoke failed" and body["smoke"]["returncode"] != 0, body
        oracle = ArtifactOracle(server.data_root)
        rolled = oracle.supervisor_rows("managed_update_rolled_back")
        assert [row.get("reason") for row in rolled] == ["replace_pre_restart_smoke_failed"], rolled
        # Same process and boot, original product bytes, nothing owed to a boot.
        assert server.proc.poll() is None, "The abort became an application exit"
        assert len(jsonl(evidence / "boots.jsonl")) == 1, "An application restart happened"
        assert read_service_bindings(server.data_root)["main"]["pid"] == server.proc.pid
        assert _git(server.clone, "rev-parse", "HEAD") == plan["base_sha"]
        assert product_hashes(server.clone) == product_hashes(SOURCE), "Rollback did not restore product bytes"
        assert not (server.clone / ".git/ouroboros-update-tx.json").exists()
        _restored_read(server, model, "root", task_id, saved, evidence)
        running = wait_for(lambda: _rows(server, "running").get(task_id), "same-ID successor in recreated pool")
        assert running["_attempt"] == 2 and running["_working_recovery"]["cause"] == "update_aborted", running
        new_pid = int(_worker_owner(server, task_id)["local_answer_owner_pid"])
        assert new_pid != old_pid and wait_for(lambda: _gone(old_pid), "stopped worker process gone", 30)
        ready = [row for row in oracle.events("worker_ready") if int(row.get("pid") or 0) == new_pid]
        assert ready and ready[-1].get("git_sha") == plan["base_sha"], ("Recreated pool runs other code", ready)
        txdir = server.data_root / "state/delegate_recovery_transactions"
        transactions = [read_json(path) for path in sorted(txdir.glob("*.json")) if path.name != "active.json"]
        assert not any(tx.get("status") == "normal_exit_acknowledged" or tx.get("returns_restored_at")
                       for tx in transactions), ("An abort fabricated a restart acknowledgement", transactions)
        _money_preserved(before, _usage(server, [task_id], evidence, "after"))
        write_json(evidence / "abort-observation.json", {
            "server_pid": server.proc.pid, "old_worker_pid": old_pid, "new_worker_pid": new_pid,
            "worker_ready": ready, "rolled_back": rolled, "running": running, "transactions": transactions,
            "model_requests": model.count("root"),
        })
