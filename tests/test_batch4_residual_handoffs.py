"""Residual Pause replay, pending recovery, and atomic validation consumers."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def test_pause_replay_after_resume_and_later_generation_has_no_queue_effect(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    workers.PENDING.append({"id": "root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "root", "admitted_dispatch": "none"})
    first = request_owner_pause("root", request_id="P")
    assert first["ok"]
    assert q.resume_budget_paused_task("root")["ok"]
    before = q.QUEUE_SNAPSHOT_PATH.read_bytes()
    monkeypatch.setattr(owner_pause, "_CACHE", {})  # fresh process observes durable action history
    replay = request_owner_pause("root", request_id="P")
    assert replay["ok"] and replay["duplicate"] and replay["state"] == "released"
    assert replay["fence_id"] == first["fence_id"]
    assert not q.BUDGET_ROOT_FENCES
    assert before == q.QUEUE_SNAPSHOT_PATH.read_bytes()
    second = request_owner_pause("root", request_id="Q")
    assert second["ok"] and second["fence_id"] != first["fence_id"]
    before = q.QUEUE_SNAPSHOT_PATH.read_bytes()
    replay = request_owner_pause("root", request_id="P")
    assert replay["state"] == "released" and replay["fence_id"] == first["fence_id"]
    assert q.BUDGET_ROOT_FENCES["root"]["fence_id"] == second["fence_id"]
    assert before == q.QUEUE_SNAPSHOT_PATH.read_bytes()
    assert q.resume_budget_paused_task("root")["ok"]
    assert request_owner_pause("root", request_id="P")["state"] == "released"
    assert not q.BUDGET_ROOT_FENCES


def test_pause_accepted_while_closed_remembers_every_acknowledged_request(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    workers.PENDING.append({"id": "root", "type": "task", "_attempt": 1, "chat_id": 0,
                            "root_task_id": "root", "admitted_dispatch": "none"})
    first = request_owner_pause("root", request_id="P")
    alias = request_owner_pause("root", request_id="other-tab")
    assert alias["fence_id"] == first["fence_id"] and alias["duplicate"]
    assert q.resume_budget_paused_task("root")["ok"]
    monkeypatch.setattr(owner_pause, "_CACHE", {})
    assert request_owner_pause("root", request_id="other-tab")["state"] == "released"
    assert not q.BUDGET_ROOT_FENCES


def _pending(tmp_path):
    from ouroboros import delegate_custody as dc
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    assert dc.record_start_requested(tmp_path, run_id="", task_id="child", root_task_id="root",
        invocation_id="inv-exact", idempotency_key="inv-exact", max_seconds=60,
        request={"prompt": "original full request"}, project_id="", project_owned=False, route="test")
    return dc.pending_invocations(tmp_path)[0]


def test_pending_recovery_pause_before_concrete_handoff_retains_same_invocation(tmp_path):
    from ouroboros import delegate_custody as dc, owner_pause
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    record = _pending(tmp_path)
    owner_pause.install_fence(tmp_path, "root", request_id="P")
    gateway = SimpleNamespace(start_run=lambda *_a, **_kw: pytest.fail("new handoff after Pause"))
    result = _recover_pending_invocation(tmp_path, gateway, record)
    assert result["action"] == "invocation_retained" and result["reason"] == "owner_pause"
    assert dc.pending_invocations(tmp_path) == [record]


def test_pending_recovery_handoff_is_excluded_with_pause_and_wait_is_outside_lock(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as dc, owner_pause
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    record = _pending(tmp_path)
    accepted, release = threading.Event(), threading.Event()
    real_submit = ThreadPoolExecutor.submit
    submissions, observed, results, errors = [], [], [], []
    def submit(executor, function, *args, **kwargs):
        def delayed():
            assert release.wait(5)
            return function(*args, **kwargs)
        future = real_submit(executor, delayed)
        submissions.append(True)
        accepted.set()
        return future
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    def start(body, *, idempotency_key):
        # The handed request may enter its SDK after Pause has been accepted.
        observed.append((body, idempotency_key, owner_pause.read_fence(tmp_path, "root")["state"]))
        return {}  # Unknown accepted handle stays pending, not proven no-effect.
    def recover():
        try:
            results.append(_recover_pending_invocation(tmp_path, SimpleNamespace(start_run=start), record))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=recover)
    thread.start()
    try:
        assert accepted.wait(5), "recovery must use the concrete executor handoff"
        owner_pause.install_fence(tmp_path, "root", request_id="P")
        assert thread.is_alive() and not observed
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert submissions == [True]
    assert record["request"] is None and record["request_ref"]
    assert observed == [({"prompt": "original full request"}, record["invocation_id"], "requested")]
    assert results[0]["action"] == "recovery_pending"
    assert dc.pending_invocations(tmp_path) == [record]


@pytest.mark.parametrize("failure", ["parse", "late_hunk", "missing_delete", "existing_add"])
def test_atomic_patch_validation_refusal_retires_only_its_exact_handoff(tmp_path, monkeypatch, failure):
    from tests.test_batch4_producer_custody import _registry, _consumers
    from ouroboros.task_results import load_task_result

    registry, q, workers = _registry(tmp_path, monkeypatch)
    workspace = registry._ctx.workspace_root
    (workspace / "a.txt").write_text("old\n")
    good = "*** Update File: a.txt\n@@\n-old\n+new\n"
    patches = {
        "parse": "not a patch",
        "late_hunk": good + "*** Update File: a.txt\n@@\n-absent\n+unused\n",
        "missing_delete": good + "*** Delete File: missing.txt\n",
        "existing_add": good + "*** Add File: a.txt\n+duplicate\n",
    }
    result = registry.execute_result("apply_patch", {"patch": patches[failure]})
    assert result.status != "ok", result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert (workspace / "a.txt").read_text() == "old\n"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _consumers(tmp_path, registry, q, workers, held=False)


def test_patch_phase2_failure_is_not_validation_no_effect(tmp_path, monkeypatch):
    from tests.test_batch4_producer_custody import _registry
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import edit_ops

    registry, _, _ = _registry(tmp_path, monkeypatch)
    workspace = registry._ctx.workspace_root
    real_write = edit_ops.write_text
    def fail_second(path, text):
        if path.name == "b.txt":
            raise OSError("disk error after first write")
        return real_write(path, text)
    monkeypatch.setattr(edit_ops, "write_text", fail_second)
    result = registry.execute_result("apply_patch", {"patch":
        "*** Add File: a.txt\n+first\n*** Add File: b.txt\n+second\n"})
    assert (workspace / "a.txt").read_text() == "first\n"
    assert result.status != "ok"
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not (workspace / 'b.txt').exists()
    assert not load_task_result(tmp_path, "root").get("launch_handoffs"), "the failed write body returned"


def test_patch_validation_cannot_retire_an_independent_unknown_handoff(tmp_path, monkeypatch):
    from tests.test_batch4_producer_custody import _registry
    from ouroboros import owner_pause
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import conflicting_writers

    registry, q, _ = _registry(tmp_path, monkeypatch)
    with owner_pause.tool_handoff(registry._ctx, "unsettled"):
        owner_pause.run_operation(registry._ctx, lambda: "opaque acknowledgement")
        before = load_task_result(tmp_path, "root").get("launch_handoffs", {})
        result = registry.execute_result("apply_patch", {"patch": "invalid patch"})
        assert result.meta["operation_outcome"] == "completed_no_effect"
        assert load_task_result(tmp_path, "root").get("launch_handoffs", {}) == before
        assert any(row["kind"] == "tool_handoff" for row in conflicting_writers(q, "root"))


@pytest.mark.parametrize("surface", ["accept_step", "http"])
def test_pause_latch_contention_acknowledges_the_durable_fence_and_the_same_id_completes_it(
        tmp_path, monkeypatch, surface):
    import asyncio

    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_owner_controls_http_custody import _Holder, _request

    q, _, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    workers.PENDING.append({"id": "root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "root", "admitted_dispatch": "none"})
    real_install = owner_pause.install_fence
    holders = []
    def install_then_contend(*args, **kwargs):
        result = real_install(*args, **kwargs)
        holders.append(_Holder(lambda: owner_pause.launch_lock(tmp_path, "root"), 3))
        return result
    monkeypatch.setattr(owner_pause, "install_fence", install_then_contend)

    def press():
        if surface == "accept_step":
            return 200, request_owner_pause("root", request_id="P")
        from ouroboros.gateway.task_pause import api_task_pause

        response = asyncio.run(api_task_pause(_request("/api/tasks/root/pause", "root", {"request_id": "P"})))
        return response.status_code, json.loads(response.body)
    try:
        status, pending = press()
        assert not holders[0].timed_out.is_set(), "queue must not wait for this root's launch lock"
        fence = owner_pause.read_fence(tmp_path, "root")
        assert fence["request_id"] == "P" and owner_pause.fence_closed(fence)
        # Truthful: the Pause is durable and every launch gate refuses; only its
        # queue latch waits. Never "refused" (it was once a 503 over a closed fence).
        assert status == (202 if surface == "http" else 200)
        assert pending["ok"] is True and pending["latch_pending"] is True, pending
        assert (pending["fence_id"], pending["state"], pending["duplicate"]) == (fence["fence_id"], "requested", False)
        assert "root" not in q.BUDGET_ROOT_FENCES
    finally:
        for holder in holders:
            holder.finish()
    monkeypatch.setattr(owner_pause, "install_fence", real_install)
    status, replay = press()
    assert status == 200 and replay["ok"] and replay["duplicate"] and not replay.get("latch_pending")
    assert replay["fence_id"] == fence["fence_id"]
    assert owner_pause.read_fence(tmp_path, "root")["generation"] == fence["generation"]
    assert q.BUDGET_ROOT_FENCES["root"]["fence_id"] == fence["fence_id"]


def _queued_retry_root(tmp_path, monkeypatch):
    """A crash-requeued root: attempt 2 of the same id, its launch claim carried."""
    from ouroboros.task_results import write_task_result

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    write_task_result(tmp_path, "root", "interrupted", root_task_id="root")
    row = {"id": "root", "type": "task", "chat_id": 0, "_attempt": 2, "root_task_id": "root",
           "admitted_dispatch": "possible"}
    workers.PENDING.append(row)
    return q, workers, row


def test_queued_retry_root_settles_paused_and_resume_releases_its_one_admitted_dispatch(tmp_path, monkeypatch):
    from ouroboros.owner_pause import read_fence
    from supervisor.continuation_admission import conflicting_writers
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_budget_pause_holds import _idle_worker

    q, workers, row = _queued_retry_root(tmp_path, monkeypatch)
    sent = []
    _idle_worker(workers, sent)
    ack = request_owner_pause("root", request_id="P")
    assert ack["ok"] and ack["state"] == "paused", ack
    assert read_fence(tmp_path, "root")["state"] == "paused"
    # Continue's census is not the Pause's: a latched possible row still counts there.
    assert [b["kind"] for b in conflicting_writers(q, "root")] == ["dispatchable_member"]
    workers.assign_tasks()
    assert sent == [] and workers.PENDING == [row]
    resumed = q.resume_budget_paused_task("root")
    assert resumed["ok"] and resumed["owner_pause_released"] and not resumed["never_started"], resumed
    assert read_fence(tmp_path, "root")["state"] == "released"
    assert "root" not in q.BUDGET_ROOT_FENCES
    # Same row, same attempt, same dispatch evidence: never re-classified as unrun.
    assert workers.PENDING == [row] and "_budget_pause_hold" not in row
    assert (row["_attempt"], row["admitted_dispatch"]) == (2, "possible")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [(task["id"], task["_attempt"]) for task in sent] == [("root", 2)]


def test_queued_retry_resume_waits_for_its_earlier_attempts_handed_work(tmp_path, monkeypatch):
    from ouroboros.owner_pause import operation_start, read_fence, tool_handoff
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor import owner_pause_control
    from supervisor.owner_pause_control import request_owner_pause, settle_requested_owner_pauses

    q, workers, row = _queued_retry_root(tmp_path, monkeypatch)
    monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
    source = SimpleNamespace(task_id="root", root_task_id="root", drive_root=tmp_path)
    with pytest.raises(TimeoutError):  # attempt 1 handed an operation whose outcome is unknown
        with tool_handoff(source, "external_write"):
            with operation_start(source):
                raise TimeoutError("request sent, response unknown")
    assert load_task_result(tmp_path, "root").get("launch_handoffs", {})
    ack = request_owner_pause("root", request_id="P")
    assert ack["ok"] and ack["state"] == "requested"
    refused = q.resume_budget_paused_task("root")
    assert refused["error"] == "owner_pause_effects_unsettled", refused
    assert [b["kind"] for b in refused["blockers"]] == ["tool_handoff"]
    assert read_fence(tmp_path, "root")["state"] == "requested"
    assert q.BUDGET_ROOT_FENCES["root"]["cause"] == "owner_pause"
    assert workers.PENDING == [row] and "_budget_pause_hold" not in row
    assert settle_requested_owner_pauses(q, now=100.0) == []
    write_task_result(tmp_path, "root", "interrupted", launch_handoffs={})  # positively settled
    assert settle_requested_owner_pauses(q, now=110.0) == ["root"]
    assert read_fence(tmp_path, "root")["state"] == "paused"
    assert q.resume_budget_paused_task("root")["ok"]
    assert (row["_attempt"], row["admitted_dispatch"]) == (2, "possible")


def test_a_latched_member_with_unknown_dispatch_still_holds_the_pause(tmp_path, monkeypatch):
    from supervisor.events_budget import hold_budget_row
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause

    q, workers, row = _queued_retry_root(tmp_path, monkeypatch)
    child = {"id": "child", "type": "task", "chat_id": 0, "_attempt": 1, "root_task_id": "root",
             "parent_task_id": "root", "admitted_dispatch": "possible"}
    hold_budget_row(child, reason="dispatch_outcome_unknown", result_root=tmp_path)
    workers.PENDING.append(child)
    assert request_owner_pause("root", request_id="P")["state"] == "requested"
    workers.PENDING.remove(child)
    assert refresh_owner_pause_tree("root") == "paused"


def test_exact_root_resume_counts_a_latched_retry_child_like_the_settlement(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, budget_hold_fact, install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_owner_pause import _owner_park

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    write_task_result(tmp_path, "solo", "running", chat_id=0, root_task_id="solo")
    write_task_result(tmp_path, "retry-child", "interrupted", chat_id=0, root_task_id="solo")
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    child = {"id": "retry-child", "type": "task", "chat_id": 0, "_attempt": 2, "root_task_id": "solo",
             "parent_task_id": "solo", "delegation_role": "subagent", "admitted_dispatch": "possible"}
    workers.PENDING.append(child)
    assert request_owner_pause("solo", request_id="P")["state"] == "requested"
    row = _owner_park(tmp_path, monkeypatch, "solo", [], root="solo")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=q.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "solo", budget_pause.exact_pause_marker(row)["checkpoint"])
    assert read_fence(tmp_path, "solo")["state"] == "paused"
    granted = q.resume_budget_paused_task("solo")
    assert granted["ok"] and granted["exact_continuation"], granted
    # Root Resume selects the root alone: the child waits for its own selection (Q9).
    assert budget_hold_fact(child)["reason"] == HOLD_ROOT_FENCE_LIFTED
    assert (child["_attempt"], child["admitted_dispatch"]) == (2, "possible")


def test_queued_retry_root_resume_leaves_its_queued_members_to_explicit_selection(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, budget_hold_fact
    from supervisor.owner_pause_control import request_owner_pause

    q, workers, row = _queued_retry_root(tmp_path, monkeypatch)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    child = {"id": "child", "type": "task", "chat_id": 0, "_attempt": 1, "root_task_id": "root",
             "parent_task_id": "root", "delegation_role": "subagent", "admitted_dispatch": "none"}
    workers.PENDING.append(child)
    sent = []
    for wid in (0, 1):
        workers.WORKERS[wid] = affirm_ready(SimpleNamespace(wid=wid, busy_task_id=None, reaping=False,
                                               in_q=SimpleNamespace(put=lambda t: sent.append(dict(t)) if t.get("type") != "worker_quiescence" else None)))
    assert request_owner_pause("root", request_id="P")["state"] == "paused"
    assert q.resume_budget_paused_task("root")["ok"]
    assert budget_hold_fact(child)["reason"] == HOLD_ROOT_FENCE_LIFTED
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]
    assert q.resume_budget_paused_task("child")["ok"]  # the owner's own selection
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root", "child"]
