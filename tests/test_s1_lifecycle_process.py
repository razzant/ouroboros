"""Opt-in REAL server/worker consumers; prepared here, host execution is owed.

Run only via unchanged scripts/safe_test.py with OUROBOROS_E2E_DEEP=mock.
No lifecycle function, ACK, queue, admission, clock or checkpoint is mocked.
The model is an HTTP loopback peer. The non-signal worker fault exits a real
worker(23); a separate SIGKILL case MUST stay terminal under existing policy.

Managed update means current-candidate replace recovery through /api/update/apply:
real quiesce, transaction, stop, smoke, direct exec, ACK and boot-finalize. It
does NOT claim qualification of a version migration or a three-way merge.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest

from devtools.benchmarks.common.server_runner import _api, _api_status
from ouroboros import usage_accounting as ua
from ouroboros.budget_pause import budget_pause_row
from ouroboros.process_containment import pid_environment_assignment_state
from ouroboros.working_checkpoint import checkpoint_path
from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
from tests.s1_lifecycle_process_support import (
    checkpoint, jsonl, read_calls, read_json, request, scenario, submit,
    wait_for, watch_no_change, write_json,
)
from tests.system_e2e.harness import ArtifactOracle, LANE_MOCK, message_text, require_lane

pytestmark = [pytest.mark.serial, pytest.mark.integration, pytest.mark.skipif(
    os.name != "posix", reason="This harness qualifies local POSIX process facts; Windows remains separate")]


@pytest.fixture
def s1_root(tmp_path_factory):
    require_lane(LANE_MOCK)
    # macOS AF_UNIX limit: pytest's long test-function directory names leave
    # insufficient room for production worker/control socket suffixes.
    return tmp_path_factory.mktemp("s1")


def _rows(server, kind):
    return {r["id"]: r["task"] for r in ArtifactOracle(server.data_root).queue_snapshot().get(kind, [])}


def _usage(server, ids, evidence, label):
    rows = [row for row in ua.read_usage_records(server.data_root, final_only=True)
            if row.get("task_id") in ids]
    write_json(evidence / f"usage-{label}.json", rows)
    return rows


def _money_preserved(before, after):
    by_id = {row["attempt_id"]: row for row in after}
    assert before, "The real worker never reached physical usage accounting"
    for row in before:
        assert row["attempt_id"] in by_id, ("money identity lost", row)
        current = by_id[row["attempt_id"]]
        if row.get("cost_final"):
            assert current.get("cost_final") and current.get("cost_usd") == row.get("cost_usd"), current
        if row.get("state") == "dispatched":
            assert not current.get("cost_final"), "An unanswered old request must not become final/free"


def _initial_read(server, model, key, task_id, evidence):
    wait_for(lambda: model.count(key) == 1, f"{key} real tool result followed by held model request")
    saved = checkpoint(server, task_id)
    assert saved["task_id"] == task_id and saved["task_attempt"] == 1
    assert saved["working"]["boundary"] == "ready"
    assert saved["working"]["seq"] >= 3, saved["working"]  # ready/pre-effect/post-batch/next ready
    tools = [message_text(m) for m in saved["messages"] if m.get("role") == "tool"]
    assert sum(text.count(model.tokens[key]) for text in tools) == 1
    calls = wait_for(lambda: read_calls(server, task_id), f"{key} durable read invocation")
    assert len(calls) == 1 and not calls[0].get("is_error"), calls
    write_json(evidence / f"checkpoint-before-{key}.json", saved)
    write_json(evidence / f"read-before-{key}.json", calls)
    return saved


def _restored_read(server, model, key, task_id, before, evidence):
    wait_for(lambda: model.count(key) >= 2, f"{key} actual restored worker model request")
    assert model.count(key) == 2, (key, model.count(key))
    resumed = model.ready[key][1]
    old_results = [m for m in before["messages"] if m.get("role") == "tool"
                   and model.tokens[key] in message_text(m)]
    new_results = [m for m in resumed["messages"] if m.get("role") == "tool"
                   and model.tokens[key] in message_text(m)]
    assert len(old_results) == len(new_results) == 1
    assert message_text(new_results[0]) == message_text(old_results[0]), "Saved cognition changed"
    after = checkpoint(server, task_id, 2)
    assert after["round_idx"] >= before["round_idx"], "Round count reset"
    assert after["cost_ceiling"] == before["cost_ceiling"], "Original task ceiling changed"
    assert len(read_calls(server, task_id)) == 1, "A completed tool was invoked again"
    assert not checkpoint_path(server.data_root, task_id, 1).exists(), "Saved source was not consumed"
    write_json(evidence / f"checkpoint-after-{key}.json", after)


def _prior_pause(server, model, workspace, evidence):
    task_id = submit(server, workspace, "prior")
    _initial_read(server, model, "prior", task_id, evidence)
    request(server, f"/api/tasks/{task_id}/pause", {"request_id": "s1-prior-pause"})
    pause = wait_for(lambda: (row if (row := budget_pause_row(server.data_root, task_id))
                             and row.get("state") == "paused" else None), "prior owner Pause settled")
    write_json(evidence / "prior-pause.json", pause)
    wait_for(lambda: task_id in _rows(server, "pending"), "prior pause queue carrier")
    return task_id, pause


def _assert_prior_pause(server, model, task_id, before):
    after = budget_pause_row(server.data_root, task_id)
    assert after.get("state") == "paused" and after.get("pause_id") == before["pause_id"], after
    assert model.count("prior") == 1, "The lifecycle released an earlier owner Pause"
    assert not (after.get("grant") or {}).get("consumed_at"), after


def _second_boot(server, evidence, *, direct_exec):
    if direct_exec:
        wait_for(lambda: len(jsonl(evidence / "boots.jsonl")) == 2, "production direct-exec second boot", 180)
        server._wait_ready(150)
        server.prove_served()
    else:
        server.stop()  # custody sweep recorded separately from product exit, before next boot
        server.start()
    boots = jsonl(evidence / "boots.jsonl")
    assert len(boots) == 2 and boots[0]["boot_id"] != boots[1]["boot_id"], boots
    assert (boots[0]["pid"] == boots[1]["pid"]) == direct_exec, boots


def _ack(server, root_id, child_id, queue_id, prior_id, evidence):
    txdir = server.data_root / "state/delegate_recovery_transactions"
    rows = [read_json(path) for path in txdir.glob("*.json") if path.name != "active.json"]
    assert len(rows) == 1, rows
    [tx] = rows
    assert tx["status"] == "normal_exit_acknowledged" and tx["ack_source"] == "direct_exec_successor", tx
    assert tx["exit_code"] == 42 and tx["supervisor_pid"] == server.proc.pid, tx
    assert {root_id, child_id} <= set(tx["return_ids"]), tx
    assert queue_id in tx["queue_ids"] and prior_id not in tx["return_ids"] + tx["queue_ids"], tx
    assert tx.get("returns_restored_at"), "Fresh return token was not consumed by boot"
    write_json(evidence / "ack-readback.json", tx)


@pytest.mark.parametrize("door", ["quit", "crash", "restart", "managed_update", "panic"])
def test_real_server_door_to_second_boot_keeps_cognition_queue_and_prior_pause(s1_root, door):
    require_lane(LANE_MOCK)
    with scenario(s1_root, tree=True, managed=door == "managed_update") as (server, model, workspace, evidence):
        prior_id, prior_pause = _prior_pause(server, model, workspace, evidence)
        root_id = submit(server, workspace, "root")
        root_saved = _initial_read(server, model, "root", root_id, evidence)
        oracle = ArtifactOracle(server.data_root)
        children = wait_for(lambda: oracle.child_task_ids(root_id), "real schedule_subagent child")
        assert len(children) == 1
        child_id = children[0]
        child_saved = _initial_read(server, model, "child", child_id, evidence)
        assert oracle.task_result(child_id)["root_task_id"] == root_id
        queued_id = submit(server, workspace, "queued")
        wait_for(lambda: queued_id in _rows(server, "pending"), "accepted unstarted queue carrier")
        assert model.count("queued") == 0, "Fixture did not reach a runnable queue waiting for capacity"
        ids = [root_id, child_id, queued_id, prior_id]
        write_json(evidence / "task-ids.json", dict(zip(("root", "child", "queued", "prior"), ids)))
        write_json(evidence / "snapshot-before.json", oracle.queue_snapshot())
        before = _usage(server, ids, evidence, "before")
        start = time.monotonic()
        if door == "quit":
            # Genuine private launcher Quit pipe -> request_shutdown -> lifespan.
            rc = server.quit()
            assert rc == 0, rc
        elif door == "crash":
            os.kill(server.proc.pid, signal.SIGKILL)
            rc = server.proc.wait(30)
            assert rc == -signal.SIGKILL
        elif door == "panic":
            # The endpoint may lose the HTTP reply to hard exit; record it exactly.
            response = _api_status(server.base_url, "POST", "/api/command", {"cmd": "/panic"}, timeout=30)
            write_json(evidence / "panic-response.json", response)
            rc = server.proc.wait(30)
        elif door == "restart":
            request(server, "/api/command", {"cmd": "/restart"})
            rc = None  # successful exec keeps this PID, not a fabricated waitpid(42)
        else:
            plan = _api(server.base_url, "POST", "/api/update/preflight", {}, timeout=180).get("merge_plan", {})
            write_json(evidence / "update-plan.json", plan)
            assert plan.get("base_sha") and plan.get("target_sha") != plan["base_sha"], plan
            # The newer fixture commit changes no product bytes.
            base_tree = subprocess.check_output(["git", "rev-parse", plan["base_sha"] + "^{tree}"], cwd=server.clone)
            target_tree = subprocess.check_output(["git", "rev-parse", plan["target_sha"] + "^{tree}"], cwd=server.clone)
            assert base_tree == target_tree
            result = request(server, "/api/update/apply", {
                "strategy": "replace", "expected_base_sha": plan["base_sha"],
                "expected_target_sha": plan["target_sha"], "confirm_recovery": True,
            })
            write_json(evidence / "update-response.json", result)
            assert result.get("restarting") is True, result
            rc = None
        write_json(evidence / "door.json", {"door": door, "observed_exit": rc,
                                            "elapsed_sec": time.monotonic() - start})
        returning = door in {"restart", "managed_update"}
        _second_boot(server, evidence, direct_exec=returning)
        if returning:
            _ack(server, root_id, child_id, queued_id, prior_id, evidence)
        else:
            pending = _rows(server, "pending")
            assert set(ids) <= set(pending), pending
            for task_id in (root_id, child_id, queued_id):
                hold = budget_hold_fact(pending[task_id])
                assert hold and hold["reason"] == HOLD_SAVED_WORK, (task_id, hold)
            counts = {key: model.count(key) for key in model.ready}
            watch_no_change(lambda: {key: model.count(key) for key in model.ready}, counts,
                            f"{door} cannot auto-return work")
            if door == "panic":
                _assert_prior_pause(server, model, prior_id, prior_pause)
                _money_preserved(before, _usage(server, ids, evidence, "after"))
                return  # Panic never acquires an automatic Resume promise.
            for task_id in (root_id, child_id):
                request(server, f"/api/tasks/{task_id}/resume")
        _restored_read(server, model, "root", root_id, root_saved, evidence)
        _restored_read(server, model, "child", child_id, child_saved, evidence)
        # Both restored workers hold model requests. Stop those two explicitly
        # to make real capacity; the queued task must preserve its door's grant.
        for task_id in (child_id, root_id):
            request(server, f"/api/tasks/{task_id}/cancel")
        if not returning:
            watch_no_change(lambda: model.count("queued"), 0, "Quit/crash queue still needs Resume")
            request(server, f"/api/tasks/{queued_id}/resume")
        wait_for(lambda: model.count("queued") == 1, "formerly runnable queue dispatched once")
        assert oracle.task_result(queued_id).get("admitted_dispatch_attempt") == 1
        _assert_prior_pause(server, model, prior_id, prior_pause)
        _money_preserved(before, _usage(server, ids, evidence, "after"))
        if door == "managed_update":
            finalized = wait_for(lambda: oracle.supervisor_rows("managed_update_finalized"), "real boot update-finalize")
            assert finalized[-1]["head"] == plan["target_sha"], finalized
        write_json(evidence / "snapshot-after.json", oracle.queue_snapshot())


def _worker_owner(server, task_id):
    def sent():
        return [r for r in ua.read_usage_records(server.data_root, final_only=True)
                if r.get("task_id") == task_id and r.get("state") == "dispatched"
                and r.get("local_answer_owner_pid") and r.get("local_answer_owner_birth")]
    rows = wait_for(sent, "physical worker receiver identity")
    pids = {r["pid"] for r in read_json(server.data_root / "state/worker_pids.json").get("workers", [])}
    live = [r for r in rows if int(r["local_answer_owner_pid"]) in pids]
    assert len(live) == 1, live
    row = live[0]
    pid = int(row["local_answer_owner_pid"])
    assert pid != server.proc.pid and os.getpgid(pid) == pid
    assert pid_environment_assignment_state(pid, "OUROBOROS_DATA_DIR", str(server.data_root)) == "present"
    return row


def _fault_worker(server, task_id, evidence, *, signal_kill=False, label="first"):
    owner = _worker_owner(server, task_id)
    pid = int(owner["local_answer_owner_pid"])
    if not signal_kill:
        assert (server.injection / f"installed-{pid}.json").is_file(), "Worker exit injector was not loaded"
        write_json(server.injection / f"permit-{pid}.json", {"pid": pid, "exit_code": 23})
    write_json(evidence / f"worker-owner-{label}.json", owner)
    os.kill(pid, signal.SIGKILL if signal_kill else signal.SIGUSR1)
    expected = -int(signal.SIGKILL) if signal_kill else 23
    dead = wait_for(lambda: [r for r in ArtifactOracle(server.data_root).supervisor_rows("worker_dead_detected")
                            if r.get("busy_task_id") == task_id and r.get("exitcode") == expected
                            and r.get("attempt") == (1 if label == "first" else 2)],
                    "supervisor's native Process exit observation", 120)
    write_json(evidence / f"worker-dead-{label}.json", dead)
    assert server.proc.poll() is None, "Worker death became whole-app death"


@pytest.mark.parametrize("signal_kill", [False, True], ids=["exit23-existing-retry", "sigkill-no-retry"])
def test_real_pool_worker_death_preserves_existing_retry_policy_and_saved_cognition(s1_root, signal_kill):
    require_lane(LANE_MOCK)
    with scenario(s1_root) as (server, model, workspace, evidence):
        task_id = submit(server, workspace, "root")
        saved = _initial_read(server, model, "root", task_id, evidence)
        before = _usage(server, [task_id], evidence, "before")
        _fault_worker(server, task_id, evidence, signal_kill=signal_kill)
        oracle = ArtifactOracle(server.data_root)
        if signal_kill:
            result = wait_for(lambda: (r if (r := oracle.task_result(task_id)).get("status") == "failed" else None),
                              "signal crash remains terminal")
            assert result["reason_code"] == "worker_crash_signal", result
            watch_no_change(lambda: model.count("root"), 1, "SIGKILL must not gain retry eligibility")
        else:
            _restored_read(server, model, "root", task_id, saved, evidence)
            running = wait_for(lambda: _rows(server, "running").get(task_id), "same-ID retry in real pool")
            assert running["_attempt"] == 2 and running["_working_recovery"]["cause"] == "worker_crash", running
            # The candidate's existing QUEUE_MAX_RETRIES is 1. A second real
            # non-signal loss must not buy a third try. No setting is changed.
            _fault_worker(server, task_id, evidence, label="second")
            result = wait_for(lambda: (r if (r := oracle.task_result(task_id)).get("status") == "failed" else None),
                              "existing retry exhaustion")
            assert result["reason_code"] == "worker_crash_retry_exhausted", result
            watch_no_change(lambda: model.count("root"), 2, "retry count must not expand")
        assert len(read_calls(server, task_id)) == 1
        _money_preserved(before, _usage(server, [task_id], evidence, "after"))
        write_json(evidence / "terminal-result.json", result)


def test_real_panic_has_no_working_save_prerequisite(s1_root):
    require_lane(LANE_MOCK)
    with scenario(s1_root) as (server, model, workspace, evidence):
        task_id = submit(server, workspace, "root")
        _initial_read(server, model, "root", task_id, evidence)
        path = checkpoint_path(server.data_root, task_id, 1)
        # Test-only filesystem fault: replace the working DIRECTORY with a file.
        # Both loading and saving at the standard path now fail, independent of
        # uid/ACL privileges. Keep the original checkpoint as evidence first.
        saved_dir = path.parent.with_name("s1-original-working")
        path.parent.rename(saved_dir)
        path.parent.write_text("S1 deliberate non-directory storage fault\n")
        try:
            start = time.monotonic()
            response = _api_status(server.base_url, "POST", "/api/command", {"cmd": "/panic"}, timeout=30)
            rc = server.proc.wait(30)
            write_json(evidence / "panic-unwritable.json", {
                "response": response, "observed_exit": rc, "elapsed_sec": time.monotonic() - start,
                "working_location_is_file": path.parent.is_file(),
            })
            assert path.parent.is_file(), "Fixture fault was not present through Panic"
            _second_boot(server, evidence, direct_exec=False)
            watch_no_change(lambda: model.count("root"), 1, "Panic with no loadable save must never auto-return")
        finally:
            # Restore only our disposable filesystem fault, after custody ends.
            server.stop()
            path.parent.unlink()
            saved_dir.rename(path.parent)
