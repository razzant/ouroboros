"""Preparation refusal and final child admission keep their original authority."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
import threading

import pytest

from ouroboros import headless, task_tree_ledger
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.task_results import write_task_result
from supervisor import events_schedule_task as schedule, task_admission, task_lifecycle, workspace_preparation as prep
from supervisor.events import dispatch_event
from tests.test_workspace_preparation import env as env, _WAIT_SEC

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("cause", ["closed", "thread_start_failed"])
def test_entry_refusal_without_executor_retains_drive_and_does_not_block_dispatch(env, monkeypatch, cause):
    event = env.request("unstarted")
    drive = Path(event["child_drive_root"])
    drive.mkdir(parents=True)
    evidence = drive / "producer-evidence.txt"
    evidence.write_text("retained producer state", encoding="utf-8")
    reached, release = Queue(), threading.Event()

    def held_cleanup(*_args, **_kwargs):
        reached.put("cleanup")
        assert release.wait(_WAIT_SEC)

    monkeypatch.setattr(headless, "remove_subagent_task_drive", held_cleanup)
    if cause == "closed":
        prep.close_generation(env.gen)
    else:
        real_start = threading.Thread.start

        def cannot_start(thread):
            if thread.name == "workspace-preparation":
                raise OSError("thread start failed")
            return real_start(thread)

        monkeypatch.setattr(threading.Thread, "start", cannot_start)

    def dispatch():
        dispatch_event(event, env.ctx)
        dispatch_event({"type": "budget_root_fence", "task_id": "independent",
                        "resource_limit": {"scope": "root", "root_task_id": "other-root"}}, env.ctx)
        reached.put("control-drained")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(dispatch)
        try:
            assert reached.get(timeout=_WAIT_SEC) == "control-drained"
        finally:
            release.set()
        future.result(timeout=_WAIT_SEC)
    row = env.row("unstarted")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["workspace_preparation"]["phase"] == "refused"
    expected = prep.INTERRUPTED_BY_GENERATION_EXIT if cause == "closed" else prep.EXECUTOR_UNAVAILABLE
    assert row["reason_code"] == expected
    assert task_lifecycle.BUDGET_ROOT_FENCES["other-root"]["status"] == "paused"
    assert evidence.read_text(encoding="utf-8") == "retained producer state"
    assert env.pending_ids() == [] and env.gen.wait_idle(_WAIT_SEC)


@pytest.mark.parametrize("limit, reason", [
    ("max_children", "delegation_rights_max_children"),
    ("may_fan_out", "delegation_rights_may_fan_out"),
    ("hard", "active_subagent_hard_cap"),
    ("constraint", "delegation_constraint_child_cap"),
    ("soft", ""),
    ("advisory", ""),
])
def test_readonly_admission_between_preparation_checks_and_enqueue_keeps_caps(env, monkeypatch, limit, reason):
    from supervisor import queue

    parent_budget = {"max_children": 1} if limit == "max_children" else {"may_fan_out": False} if limit == "may_fan_out" else {}
    write_task_result(env.data, "parent", "running", delegation_role="root", root_task_id="root",
                      task_contract=build_task_contract({"delegation_budget": parent_budget}))
    monkeypatch.setattr(task_tree_ledger, "DATA_DIR", env.data)
    if limit in {"constraint", "advisory"}:
        recorded = task_tree_ledger.tree_ledger_append(
            "root", "delegation_constraint", "one child at this boundary", task_id="parent",
            payload={"directive": "cap_children", "scope": {"max_children": 1},
                     "rationale": "one child at this boundary", "advisory": limit == "advisory"},
        )
        assert recorded.startswith("OK"), recorded
    if limit == "soft":
        monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "1")
    if limit == "hard":
        # Existing physical ceiling, not a smaller replacement policy: one
        # readonly admission consumes the last slot while the copy is ready.
        with queue._queue_lock:
            queue.PENDING.extend({"id": f"seed-{i}", "root_task_id": "root",
                                  "parent_task_id": "other-parent", "delegation_role": "subagent"}
                                 for i in range(schedule.MAX_ACTIVE_SUBAGENTS_HARD_CAP - 1))

    checked, release = threading.Event(), threading.Event()
    original = prep.PreparationClaim.admit

    def before_handover(claim, step):
        # The copy lane's checks are done; the final admission step is about to be handed to the
        # admission lane, where every admission step is FIFO: a sibling admitted meanwhile lands first.
        if claim.task_id == "acting":
            checked.set()
            assert release.wait(_WAIT_SEC), "test did not release final admission"
        return original(claim, step)

    monkeypatch.setattr(prep.PreparationClaim, "admit", before_handover)
    event = env.request("acting")
    assert env.dispatch_returns(event)
    try:
        assert checked.wait(_WAIT_SEC), "acting child never reached final admission"
        sibling = env.readonly("readonly", root="root")
        sibling["parent_task_id"] = "parent"
        write_task_result(env.data, "readonly", "requested", parent_task_id="parent")
        assert env.dispatch_returns(sibling)
        deadline = threading.Event()
        for _ in range(int(_WAIT_SEC * 100)):
            if env.row("readonly")["status"] == "scheduled":
                break
            deadline.wait(0.01)
        assert env.row("readonly")["status"] == "scheduled"
    finally:
        release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("acting")
    if reason:
        assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted", row
        assert row["reason_code"] == reason
        assert "acting" not in env.pending_ids()
        assert not Path(row["workspace_preparation"]["resource"]["path"]).exists()
    else:
        assert row["status"] == "scheduled" and env.pending_ids().count("acting") == 1
        assert Path(row["workspace_root"]).is_dir()
        if limit == "soft":
            assert any(kwargs.get("progress_meta", {}).get("queued_behind_active_cap")
                       for _chat, _text, kwargs in env.ctx.sent if kwargs.get("task_id") == "acting")


@pytest.mark.parametrize("seam", ["owned", "depth", "final", "refusal", "recovery_refusal"])
def test_prepared_child_sql_contention_releases_q_and_keeps_exact_marker(env, monkeypatch, seam):
    from ouroboros import store
    from supervisor import queue
    from tests.test_admission_contention import database_reader

    event = env.request("busy")
    claim = prep.PreparationClaim.live(env.gen, event, env.ctx, {
        "parent_id": "parent", "root_task_id": "root", "chat_id": 7, "delegation_role": "subagent"})
    assert task_admission.claim_requested_child(env.ctx, "busy", claim.token)
    assert prep._transition(env.data, "busy", claim.token, claim.marker(prep.QUEUED), expected=None)
    if seam == "final":
        assert prep._transition(env.data, "busy", claim.token, claim.marker(prep.COPYING), expected=(prep.QUEUED,))
        assert prep._transition(env.data, "busy", claim.token, claim.marker(prep.READY), expected=(prep.COPYING,))
        assert claim.enter_admission()
        monkeypatch.setattr(claim, "enter_admission", lambda: True)  # boundary already durable before the held DB
    before = env.row("busy")
    if seam == "recovery_refusal":
        claim = prep.PreparationClaim.recovered(env.data, before, before[prep.MARKER])
        claim.ctx.send_with_budget = env.ctx.send_with_budget

    def invoke():
        if seam == "owned":
            with pytest.raises(store.StoreBusy):
                task_admission.subagent_schedule_owned(env.ctx, "busy", admission_token=claim.token)
        elif seam == "depth":
            with pytest.raises(store.StoreBusy):
                task_admission.parse_schedule_task_depth(
                    env.ctx, event, tid="busy", chat_id=7, delegation_role="subagent", parent_id="parent",
                    root_task_id="root", role="builder", desc="change busy", expected_output="patch",
                    constraints="", task_context="", preparation=claim,
                )
        elif seam == "final":
            # Off a lane the gate waits only the caller's budget: zero here, so the held store refuses at once.
            with pytest.raises(store.StoreBusy), store.acquisition_budget(0):
                task_admission.enqueue_subagent_with_scheduled_result(
                    env.ctx, {"id": "busy"}, result_fields={}, admitted_task_contract={},
                    admitted_depth_provenance={}, direct_child_count=0, pending_ref=queue.PENDING, preparation=claim,
                )
        else:
            with pytest.raises(store.StoreBusy):
                claim.refuse("preparation ended", prep.INTERRUPTED_BY_GENERATION_EXIT)

    probed, controlled = threading.Event(), threading.Event()
    under_q = []
    real_connect, real_snapshot = store.connect, env.ctx.persist_queue_snapshot

    def connecting(path, tier, **kwargs):
        # The moved paths open storage only OFF Q; a connection while Q is owned is the old defect.
        if queue._queue_lock._is_owned():
            under_q.append(kwargs.get("wait_sec", 0))
        probed.set()
        return real_connect(path, tier, **kwargs)

    def snapshot(reason=""):
        if reason == "budget_root_fenced":
            controlled.set()  # inside the real control handler's Q section
        return real_snapshot(reason)

    monkeypatch.setattr(env.ctx, "persist_queue_snapshot", snapshot)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with database_reader(env.data, exclusive=True) as release_database:
            monkeypatch.setattr(store, "connect", connecting)
            future = pool.submit(invoke)
            try:
                assert probed.wait(3), "prepared-child read never reached the held SQLite store"
                assert under_q == [], "a prepared-child owner opened SQLite while holding Q"
                control = pool.submit(dispatch_event, {
                    "type": "budget_root_fence", "task_id": "independent",
                    "resource_limit": {"scope": "root", "root_task_id": "other-root"},
                }, env.ctx)
                assert controlled.wait(3), "the contended preparation held Q against a real control"
                future.result(timeout=_WAIT_SEC)  # refused at once off Q, while the store is still held
            finally:
                release_database()
            control.result(timeout=_WAIT_SEC)
    assert env.row("busy") == before  # no invented never-admitted receipt or newer-owner overwrite
    assert env.pending_ids() == []
    assert task_lifecycle.BUDGET_ROOT_FENCES["other-root"]["status"] == "paused"
    claim.release()
