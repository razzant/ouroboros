from __future__ import annotations

import json
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import model_sleep, owner_pause, owner_wait, server_restart
from ouroboros.gateway import control
from ouroboros.model_wait import TaskModelWait
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.review_pause import live_detached_operations
from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
from supervisor import git_ops, workers
from supervisor.active_activity import (
    DirectActivityRegistry,
    get_direct_activity_registry,
)
from supervisor.budget_resume import resume_warm_owner_pause_root
from supervisor.owner_pause_control import request_owner_pause
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_direct_chat_turn_owner_control import _live_chat_agent


@pytest.fixture(autouse=True)
def _isolated_direct_registry_and_short_drain(monkeypatch: pytest.MonkeyPatch):
    get_direct_activity_registry().clear()
    workers.open_repo_writer_admission()
    orig_wait = DirectActivityRegistry.wait_until_empty
    monkeypatch.setattr(
        DirectActivityRegistry,
        "wait_until_empty",
        lambda self, timeout=0.05: orig_wait(self, min(float(timeout), 0.05)),
    )
    yield
    workers.open_repo_writer_admission()
    get_direct_activity_registry().clear()


def _direct_turn_setup(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
):
    queue_mod, state_mod, workers_mod = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state_mod, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers_mod, "kill_workers_for_update", lambda **kwargs: [])
    write_task_result(
        tmp_path,
        task_id,
        STATUS_RUNNING,
        chat_id=0,
        _is_direct_chat=True,
        admitted_dispatch_attempt=1,
    )
    _live_chat_agent(monkeypatch, task_id=task_id)
    ctx, limit = _loop_ctx(tmp_path, task_id, direct=True)
    ctx.pending_events, ctx.current_chat_id = [], 0
    waiter = TaskModelWait(
        task={"id": task_id, "budget_drive_root": str(tmp_path)},
        drive_root=tmp_path,
        event_queue=None,
        worker_slot_held=False,
    )
    waiter.tool_context, ctx.model_wait_context = ctx, waiter
    ctx.owner_wait_callback = owner_wait.direct_owner_wait
    return queue_mod, ctx, limit


def _wait_for_owner_wait_state(
    drive_root: pathlib.Path,
    task_id: str,
    *,
    expected_reason: str,
    timeout: float = 5.0,
) -> dict:
    # The durable row is written before the registry mark, so wait for both:
    # returning on the row alone races the mark (review rl-20261022T144146).
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wait = (load_task_result(drive_root, task_id) or {}).get("owner_wait") or {}
        entry = get_direct_activity_registry().get(task_id)
        if (
            wait.get("state") == "waiting"
            and wait.get("reason") == expected_reason
            and isinstance(wait.get("source_ref"), dict)
            and entry is not None
            and entry.parked_wait is not None
        ):
            return wait
        time.sleep(0.02)
    raise RuntimeError(
        f"direct turn {task_id} did not enter owner_wait(reason={expected_reason!r}) within {timeout}s"
    )


def test_update_quiesce_proceeds_when_warm_owner_paused_direct_turn_reviews_settled(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2191 negative reproduction: a direct turn parked warm in direct_owner_wait
    under owner_pause whose detached reviews have already settled and with no tool
    or model call in flight must not block _quiesce_repo_writers / _stop_repo_writers."""
    task_id = "390ae3e9"
    _queue, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    if request_owner_pause(task_id, request_id="press").get("state") != "requested":
        raise RuntimeError("owner pause fence was not requested")
    fence = owner_pause.read_fence(tmp_path, task_id)
    registry = get_direct_activity_registry()

    def _run_direct_turn() -> None:
        try:
            owner_wait.park_owner_pause_warm(
                limit,
                ctx,
                fence=fence,
                detached=[{"owner_id": "review-settled-1", "surface": "multi_model_review"}],
            )
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_direct_turn, daemon=True)
    worker.start()
    try:
        wait = _wait_for_owner_wait_state(tmp_path, task_id, expected_reason="owner_pause")
        if (
            wait.get("state") != "waiting"
            or wait.get("reason") != "owner_pause"
            or not isinstance(wait.get("source_ref"), dict)
            or live_detached_operations(task_id) != []
            or not worker.is_alive()
        ):
            raise RuntimeError("preconditions for settled warm owner-pause direct turn were not met")

        assert server_restart._live_running_task_ids(SimpleNamespace(RUNNING={})) == []
        blockers = control._quiesce_repo_writers("smart")
        if blockers:
            if blockers != [f"active:{task_id}"]:
                raise RuntimeError(f"unexpected blocker payload: {blockers!r}")
            failure = control._fence_failure(blockers)
            payload = json.loads(failure.body.decode("utf-8"))
            if (
                failure.status_code != 409
                or payload.get("reason") != "update_writer_fence_blocked"
                or payload.get("blockers") != [f"active:{task_id}"]
                or payload.get("restart_required") is not False
            ):
                raise RuntimeError(f"unexpected fence failure response: {payload!r}")
        assert blockers == []
    finally:
        workers.open_repo_writer_admission()
        resume_warm_owner_pause_root(task_id)
        worker.join(timeout=5.0)
        if worker.is_alive():
            raise RuntimeError("warm-parked direct turn thread did not exit after resume_warm_owner_pause_root")


def test_update_quiesce_blocks_live_unparked_busy_direct_turn(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2191 control case: a live, unparked busy direct turn (no owner_wait) must
    continue to block _quiesce_repo_writers / _stop_repo_writers with 'active:<id>'."""
    task_id = "direct-busy-turn"
    _direct_turn_setup(tmp_path, monkeypatch, task_id)
    registry = get_direct_activity_registry()
    entered = threading.Event()
    release = threading.Event()

    def _run_busy_direct_turn() -> None:
        try:
            entered.set()
            release.wait(timeout=5.0)
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_busy_direct_turn, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=2.0)
        assert (load_task_result(tmp_path, task_id) or {}).get("owner_wait") is None
        assert server_restart._live_running_task_ids(SimpleNamespace(RUNNING={})) == [task_id]

        blockers = control._quiesce_repo_writers("smart")
        assert blockers == [f"active:{task_id}"]

        failure = control._fence_failure(blockers)
        assert failure.status_code == 409
        payload = json.loads(failure.body.decode("utf-8"))
        assert payload["reason"] == "update_writer_fence_blocked"
        assert payload["blockers"] == [f"active:{task_id}"]
        assert payload["restart_required"] is False
        assert task_id in payload["error"]
        assert "No work was stopped" in payload["error"]
        assert "restart Ouroboros" in payload["error"]
    finally:
        release.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()


@pytest.mark.parametrize(
    ("task_id", "wait_reason"),
    [
        ("direct-quiz-wait", "owner"),
        ("direct-warm-sleep", "sleep"),
    ],
)
def test_update_quiesce_proceeds_for_direct_quiz_wait_and_warm_sleep(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    wait_reason: str,
) -> None:
    """#2191 item 2: a direct turn parked in direct_owner_wait via wait_after_tools
    for either escalate(wait_for_answer=True) (reason='owner') or warm model sleep
    await_messages(mode='warm') (reason='sleep') stays registered in
    DirectActivityRegistry and hits the same drain_repo_writers refusal."""
    _queue, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    if wait_reason == "owner":
        ctx._owner_wait_requested = "quiz-direct-1"
    else:
        armed = model_sleep.request_sleep(
            ctx,
            model_sleep.selectors(ctx, wake_after_sec=60),
            model_sleep.MODE_WARM,
        )
        if armed.get("reason") != "sleep_armed":
            raise RuntimeError(f"warm model sleep was not armed: {armed!r}")
    registry = get_direct_activity_registry()

    def _run_direct_wait() -> None:
        try:
            owner_wait.wait_after_tools(
                ctx,
                limit.messages,
                limit.llm_trace,
                limit.accumulated_usage,
                limit.round_idx,
                limit.tool_schemas,
                limit.owner_msg_seen,
            )
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_direct_wait, daemon=True)
    worker.start()
    try:
        wait = _wait_for_owner_wait_state(tmp_path, task_id, expected_reason=wait_reason)
        if (
            wait.get("state") != "waiting"
            or wait.get("reason") != wait_reason
            or not isinstance(wait.get("source_ref"), dict)
            or not worker.is_alive()
        ):
            raise RuntimeError(f"preconditions for direct {wait_reason} wait were not met")

        assert server_restart._live_running_task_ids(SimpleNamespace(RUNNING={})) == []
        blockers = control._quiesce_repo_writers("smart")
        if blockers:
            if blockers != [f"active:{task_id}"]:
                raise RuntimeError(f"unexpected blocker payload: {blockers!r}")
        assert blockers == []
    finally:
        workers.open_repo_writer_admission()
        write_owner_message(
            tmp_path,
            "wake from test cleanup",
            task_id=task_id,
            msg_id=f"wake-{task_id}",
        )
        worker.join(timeout=5.0)
        if worker.is_alive():
            raise RuntimeError(f"direct {wait_reason} wait thread did not exit after owner message")


def test_update_drain_waits_for_inflight_detached_review_and_proceeds_when_settled(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: a warm owner-paused direct turn with an in-flight detached review blocks
    drain while the review is live, and unblocks as soon as the review finishes."""
    from types import SimpleNamespace
    from ouroboros import review_pause

    task_id = "direct-review-inflight"
    _queue, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    assert request_owner_pause(task_id, request_id="press-inflight").get("state") == "requested"
    fence = owner_pause.read_fence(tmp_path, task_id)
    registry = get_direct_activity_registry()
    op = SimpleNamespace(
        owner_id="op-inflight-1",
        task_id=task_id,
        surface="multi_model_review",
        retry_key="rk-1",
        identity={},
        closed=False,
        author_detached=True,
    )
    with review_pause._LOCK:
        review_pause._LIVE[op.owner_id] = op

    def _finish_op() -> None:
        with review_pause._LOCK:
            op.closed = True
            review_pause._LIVE.pop(op.owner_id, None)

    def _run_direct_turn() -> None:
        try:
            owner_wait.park_owner_pause_warm(
                limit,
                ctx,
                fence=fence,
                detached=[{"owner_id": op.owner_id, "surface": "multi_model_review"}],
            )
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_direct_turn, daemon=True)
    worker.start()
    try:
        _wait_for_owner_wait_state(tmp_path, task_id, expected_reason="owner_pause")
        assert live_detached_operations(task_id) != []

        # While the detached review is still live, drain blocks with active:<task_id>.
        assert control._quiesce_repo_writers("smart") == [f"active:{task_id}"]

        # When the detached review finishes during the drain window, drain succeeds.
        def _finish_soon() -> None:
            time.sleep(0.015)
            _finish_op()

        finisher = threading.Thread(target=_finish_soon, daemon=True)
        finisher.start()
        try:
            orig_wait = DirectActivityRegistry.wait_until_empty
            monkeypatch.setattr(
                DirectActivityRegistry,
                "wait_until_empty",
                lambda self, timeout=0.5: orig_wait(self, min(float(timeout), 0.5)),
            )
            assert control._quiesce_repo_writers("smart") == []
        finally:
            finisher.join(timeout=2.0)
    finally:
        _finish_op()
        workers.open_repo_writer_admission()
        resume_warm_owner_pause_root(task_id)
        worker.join(timeout=5.0)
        assert not worker.is_alive()


@pytest.mark.parametrize("wait_kind", ["owner_pause", "owner", "sleep"])
def test_update_handoff_and_boot_retention_for_all_three_parked_direct_wait_kinds(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    wait_kind: str,
) -> None:
    """R3/R5: after _quiesce_repo_writers and an acknowledged update restart:
    - owner_pause becomes a held cold exact pause in PENDING continued by Resume;
    - owner (quiz wait) becomes a cold _owner_wait_resume row in PENDING (not held,
      not converted to a fresh model round, preserving a mailbox answer that arrived
      during the closed update window);
    - sleep (warm model sleep) becomes a cold _owner_wait_resume row in PENDING
      without saved_sleep_recovery hold so it wakes on its own schedule."""
    from ouroboros import delegate_recovery
    from supervisor.events_budget import budget_hold_fact
    from supervisor.queue_snapshot import restore_pending_from_snapshot
    from supervisor.queue_transitions import resume_budget_paused_task

    task_id = f"direct-carry-{wait_kind}"
    queue_mod, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    registry = get_direct_activity_registry()

    if wait_kind == "owner_pause":
        assert request_owner_pause(task_id, request_id=f"pause-{task_id}").get("state") == "requested"
        fence = owner_pause.read_fence(tmp_path, task_id)

        def _target() -> None:
            try:
                owner_wait.park_owner_pause_warm(
                    limit,
                    ctx,
                    fence=fence,
                    detached=[{"owner_id": "settled-1", "surface": "multi_model_review"}],
                )
            finally:
                registry.unregister(task_id)

        expected_reason = "owner_pause"
    else:
        if wait_kind == "owner":
            ctx._owner_wait_requested = "quiz-carry-1"
            expected_reason = "owner"
        else:
            armed = model_sleep.request_sleep(
                ctx,
                model_sleep.selectors(ctx, wake_after_sec=60),
                model_sleep.MODE_WARM,
            )
            assert armed.get("reason") == "sleep_armed"
            expected_reason = "sleep"

        def _target() -> None:
            try:
                owner_wait.wait_after_tools(
                    ctx,
                    limit.messages,
                    limit.llm_trace,
                    limit.accumulated_usage,
                    limit.round_idx,
                    limit.tool_schemas,
                    limit.owner_msg_seen,
                )
            finally:
                registry.unregister(task_id)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    saved_wait = None
    try:
        _wait_for_owner_wait_state(tmp_path, task_id, expected_reason=expected_reason)
        assert control._quiesce_repo_writers("smart") == []
        assert queue_mod.persist_queue_snapshot(reason="managed_update") is True

        if wait_kind == "owner":
            # Answer arrives while admission is closed: stays in mailbox, turn stays waiting.
            write_owner_message(
                tmp_path,
                "1. A — carry all parked turns",
                task_id=task_id,
                msg_id="quiz_answer:carry-1",
            )
            time.sleep(0.1)
            still_waiting = (load_task_result(tmp_path, task_id) or {}).get("owner_wait") or {}
            assert still_waiting.get("state") == "waiting"

        saved_wait = dict((load_task_result(tmp_path, task_id, strict=True) or {})["owner_wait"])
        tx = delegate_recovery.active_restart_transaction(tmp_path)
        assert tx.get("transaction_id")
        delegate_recovery._write_restart_transaction(
            tmp_path,
            {**tx, "status": "normal_exit_acknowledged", "exit_code": 42},
        )
    finally:
        # Release the in-memory thread before boot restore mutates the same task_id.
        ctx.model_wait_context.control_reason = lambda: "cancelled"
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        if saved_wait is not None:
            owner_wait.set_owner_wait(tmp_path, task_id, saved_wait)
        workers.open_repo_writer_admission()

    queue_mod.PENDING.clear()
    queue_mod.RUNNING.clear()
    queue_mod.init(tmp_path)
    queue_mod.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    restored = restore_pending_from_snapshot()
    assert restored == 1
    assert len(queue_mod.PENDING) == 1
    item = queue_mod.PENDING[0]
    assert item["id"] == task_id
    assert item.get("_is_direct_chat") is True

    if wait_kind == "owner_pause":
        assert isinstance(item.get("_budget_pause"), dict)
        assert item["_budget_pause"].get("reason") == "owner"
        assert item["_budget_pause"].get("exact_continuation") is True
        assert not item.get("_owner_wait_resume")
        resumed = resume_budget_paused_task(task_id)
        assert resumed.get("ok") is True
    elif wait_kind == "owner":
        from types import SimpleNamespace

        assert isinstance(item.get("_owner_wait_resume"), dict)
        assert item["_owner_wait_resume"].get("reason") == "owner"
        assert not item.get("_budget_pause")
        assert budget_hold_fact(item) is None
        ctx.model_wait_context.control_reason = lambda: ""
        ctx.owner_wait_resume = item["_owner_wait_resume"]
        saved = owner_wait.load_owner_wait(ctx)
        assert saved is not None
        monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
        messages, trace, usage, seen = [], {}, {}, set()
        resumed_tuple = owner_wait.resume_native_loop(
            SimpleNamespace(_ctx=ctx), saved, messages, trace, usage, seen,
        )
        assert resumed_tuple[4] == limit.round_idx
        assert any(
            "continued from its saved owner wait after a planned restart" in str(m.get("content") or "")
            for m in messages
        )
        assert owner_wait.OwnerMailboxPeek().pending(tmp_path, task_id, seen, 1) is True
    else:
        assert isinstance(item.get("_owner_wait_resume"), dict)
        assert item["_owner_wait_resume"].get("reason") == "sleep"
        assert not item.get("_budget_pause")
        assert budget_hold_fact(item) is None
        ctx.model_wait_context.control_reason = lambda: ""
        ctx.owner_wait_resume = item["_owner_wait_resume"]
        saved = owner_wait.load_owner_wait(ctx)
        assert saved is not None


def test_direct_owner_wait_wake_gate_blocks_continuation_and_allows_terminal_stop(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: while repo_writer_admission is closed, a mailbox wake is held in the park
    loop (park mark preserved, owner_wait stays 'waiting'), whereas a terminal Stop
    exits direct_owner_wait immediately without waiting for admission to open."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor.active_activity import is_parked_direct_entry

    task_id = "direct-wake-gate"
    _queue, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    ctx._owner_wait_requested = "quiz-gate-1"
    registry = get_direct_activity_registry()
    outcomes: list[str] = []

    def _run_wait() -> None:
        try:
            checkpoint = owner_wait.checkpoint_owner_wait(
                ctx,
                limit.messages,
                limit.llm_trace,
                limit.accumulated_usage,
                limit.round_idx,
                limit.tool_schemas,
                limit.owner_msg_seen,
            )
            outcomes.append(owner_wait.direct_owner_wait(ctx, checkpoint))
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_wait, daemon=True)
    worker.start()
    try:
        _wait_for_owner_wait_state(tmp_path, task_id, expected_reason="owner")
        workers.close_repo_writer_admission("managed update (smart)")

        write_owner_message(
            tmp_path,
            "answer while gate closed",
            task_id=task_id,
            msg_id="msg-while-closed",
        )
        time.sleep(0.15)
        assert worker.is_alive()
        entry = registry.get(task_id)
        assert entry is not None and is_parked_direct_entry(entry)
        assert (load_task_result(tmp_path, task_id) or {}).get("owner_wait", {}).get("state") == "waiting"

        # Terminal Stop exits immediately even while admission remains closed.
        request_cancel(tmp_path, task_id, requested_by="owner")
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert outcomes == ["control:cancelled"]
    finally:
        workers.open_repo_writer_admission()
        if worker.is_alive():
            request_cancel(tmp_path, task_id, requested_by="cleanup")
            worker.join(timeout=5.0)


def test_aborted_update_with_live_parked_direct_turn_keeps_single_live_owner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: if an update aborts after _quiesce_repo_writers while the parked direct turn
    is still alive in DirectActivityRegistry, open_repo_writer_admission_after_update_abort
    must not create a duplicate PENDING row or corrupt owner_wait, and the live turn
    wakes normally once admission reopens."""
    task_id = "direct-abort-live"
    queue_mod, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    ctx._owner_wait_requested = "quiz-abort-1"
    registry = get_direct_activity_registry()

    def _run_wait() -> None:
        try:
            owner_wait.wait_after_tools(
                ctx,
                limit.messages,
                limit.llm_trace,
                limit.accumulated_usage,
                limit.round_idx,
                limit.tool_schemas,
                limit.owner_msg_seen,
            )
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_wait, daemon=True)
    worker.start()
    try:
        _wait_for_owner_wait_state(tmp_path, task_id, expected_reason="owner")
        assert control._quiesce_repo_writers("smart") == []
        assert workers.repo_writer_admission_closed() != ""

        # Simulate update abort before restart: reopen via abort recovery.
        assert workers.open_repo_writer_admission_after_update_abort() is True
        assert workers.repo_writer_admission_closed() == ""
        assert queue_mod.PENDING == []
        assert (load_task_result(tmp_path, task_id) or {}).get("owner_wait", {}).get("state") == "waiting"

        # Now an owner answer wakes the live thread cleanly.
        write_owner_message(
            tmp_path,
            "answer after abort",
            task_id=task_id,
            msg_id="msg-after-abort",
        )
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert (load_task_result(tmp_path, task_id) or {}).get("owner_wait", {}).get("state") == "resumed"
    finally:
        workers.open_repo_writer_admission()
        if worker.is_alive():
            write_owner_message(tmp_path, "cleanup", task_id=task_id, msg_id="msg-cleanup")
            worker.join(timeout=5.0)


@pytest.mark.parametrize("gate_reason", ["managed update (smart)", "managed_update_abort_unconfirmed"])
def test_resume_during_closed_update_admission_refuses_with_notice_and_preserves_pause(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_reason: str,
) -> None:
    """R5: pressing Resume while repo_writer_admission is closed (including on the
    managed_update_abort_unconfirmed path) is refused with a visible owner notice and
    typed update_in_progress outcome, leaving the pause fence intact."""
    from supervisor.queue_transitions import resume_budget_paused_task

    task_id = f"direct-resume-gate-{gate_reason.split()[0]}"
    queue_mod, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    notices: list[tuple] = []
    monkeypatch.setattr(
        queue_mod,
        "send_with_budget",
        lambda chat_id, text, **kw: notices.append((chat_id, text, kw)),
    )
    assert request_owner_pause(task_id, request_id=f"pause-{task_id}").get("state") == "requested"
    fence = owner_pause.read_fence(tmp_path, task_id)
    registry = get_direct_activity_registry()

    def _run_direct_turn() -> None:
        try:
            owner_wait.park_owner_pause_warm(
                limit,
                ctx,
                fence=fence,
                detached=[{"owner_id": "settled-r5", "surface": "multi_model_review"}],
            )
        finally:
            registry.unregister(task_id)

    worker = threading.Thread(target=_run_direct_turn, daemon=True)
    worker.start()
    try:
        _wait_for_owner_wait_state(tmp_path, task_id, expected_reason="owner_pause")
        workers.close_repo_writer_admission(gate_reason)

        refused = resume_budget_paused_task(task_id)
        assert refused["ok"] is False
        assert refused["error"] == "update_in_progress"
        assert refused["reason"] == gate_reason
        assert refused["action"] == "resume_after_restart"
        assert "An update is in progress; press Resume after the restart." in refused["message"]
        assert len(notices) == 1
        assert "press Resume after the restart" in notices[0][1]
        assert notices[0][2].get("system_type") == "managed_update_notice"
        assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, task_id))
        assert worker.is_alive()

        # Once admission reopens, Resume succeeds and wakes the thread.
        workers.open_repo_writer_admission()
        resumed = resume_budget_paused_task(task_id)
        assert resumed["ok"] is True
        worker.join(timeout=5.0)
        assert not worker.is_alive()
    finally:
        workers.open_repo_writer_admission()
        resume_warm_owner_pause_root(task_id)
        worker.join(timeout=5.0)


def test_late_finalize_after_admitted_wake_keeps_the_resuming_turn_a_drain_blocker(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding (Astra, rl-20261022T140822): an admitted wake followed by the
    update closing admission and a late Wrap up must not be re-parked by a second
    admission check; the resuming stack stays visible to the update drain."""
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, request_cancel
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id

    task_id = "late-wrap-up"
    _, ctx, limit = _direct_turn_setup(tmp_path, monkeypatch, task_id)
    ctx._owner_wait_requested = "quiz-race"
    checkpoint = owner_wait.checkpoint_owner_wait(
        ctx, limit.messages, limit.llm_trace, limit.accumulated_usage,
        limit.round_idx, limit.tool_schemas, limit.owner_msg_seen,
    )
    write_owner_message(tmp_path, "answer", task_id=task_id, msg_id="answer")
    registry = get_direct_activity_registry()
    real_admit = owner_wait._admit_direct_continuation_wake
    admissions: list[bool] = []
    drains: list[list[str]] = []

    def ordered_admit(*args):
        admitted = real_admit(*args)
        admissions.append(admitted)
        if len(admissions) == 1 and admitted:
            workers.close_repo_writer_admission("managed_update:smart")
            stop = request_cancel(tmp_path, task_id, requested_by="owner",
                                  requested_stop_policy=STOP_POLICY_FINALIZE)
            write_owner_message(tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, task_id=task_id,
                                msg_id=owner_stop_control_id(stop), kind=KIND_FINALIZE_NOW)
            drains.append(list(workers.drain_repo_writers(timeout=0)))
        return admitted

    monkeypatch.setattr(owner_wait, "_admit_direct_continuation_wake", ordered_admit)
    try:
        outcome = owner_wait.direct_owner_wait(ctx, checkpoint)
        assert admissions == [True], "an admitted continuation is never re-checked"
        assert drains == [[task_id]], "the resuming stack must remain a drain blocker"
        assert registry.get(task_id).parked_wait is None
        assert workers.drain_repo_writers(timeout=0) == [task_id]
        assert outcome.startswith("control:finalize")
        assert load_task_result(tmp_path, task_id)["owner_wait"]["state"] == "resumed"
    finally:
        workers.open_repo_writer_admission()


def test_update_admission_notice_falls_back_to_the_saved_chat_address(
    tmp_path: pathlib.Path,
) -> None:
    """Review finding (Astra, advisory): a queue row without chat_id still gets the
    Resume notice through the task's saved chat address."""
    from supervisor.budget_resume import _restart_or_update_refusal

    task_id = "resume-notice-fallback"
    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=7, _is_direct_chat=True)
    sent: list[tuple] = []
    q = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        repo_writer_admission_closed=lambda: "managed update (smart)",
        repo_writer_task_allowed=lambda _task: False,
        send_with_budget=lambda chat_id, text, **kw: sent.append((chat_id, text, kw)),
    )
    refused = _restart_or_update_refusal(q, {"id": task_id}, tmp_path, task_id)
    assert refused and refused["error"] == "update_in_progress"
    assert [(chat_id, kw.get("system_type")) for chat_id, _text, kw in sent] == [
        (7, "managed_update_notice")]
