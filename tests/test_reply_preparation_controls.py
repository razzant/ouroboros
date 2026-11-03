"""A retained working run never defeats a later control on its sleeping author."""
import pytest

from tests.test_reply_preparation import world, body, _Supervisor, _observed, _chats, _intent  # noqa: F401

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("control", ["owner_pause", "stop", "restart"])
def test_cold_retained_run_obeys_later_control(world, body, monkeypatch, control):  # noqa: F811
    from ouroboros import budget_pause as bp, model_sleep, sleep_custody as sc
    from ouroboros.cancel_intents import STOP_POLICY_IMMEDIATE
    from ouroboros.owner_pause import install_fence
    from supervisor.events_budget import install_exact_budget_pause
    from tests import test_cold_external_custody as cold
    from tests._budget_pause_exact_helpers import _supervisor_ctx

    env = cold._setup(world.root, monkeypatch)
    monkeypatch.setattr(sc, "_runtime", lambda _root: body.runtime)
    monkeypatch.setattr(world.workers, "get_event_q", lambda **_kw: _Supervisor(world))
    env.task.update(chat_id=6, metadata={"origin_message_ref": {
        "chat_id": 6, "client_message_id": "m-control", "ts": "2026-10-17T00:00:00+00:00"}})
    assert _observed(world, "run-one", run_state="running")
    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")

    def host_tick(_ctx):
        sc.reconcile(env.q, runtime=body.runtime)
        return ""

    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    with pytest.raises(bp.BudgetPauseRequested) as raised:
        bp.enter_cold_sleep(env.limit)
    pause = raised.value.pause
    install_exact_budget_pause(_supervisor_ctx(world.root, env.workers, env.q, [], []), "sleeper",
                               bp.exact_pause_marker(pause)["checkpoint"], evt=bp.pause_event(env.task, pause))
    bp.end_dispatch_fence("sleeper")
    [parked] = env.workers.PENDING
    assert sc.carrier_receipt(parked)["runs"] == [["run-one", "invoke-one"]]
    assert _chats(world) == [6]
    if control == "owner_pause":
        install_fence(world.root, "sleeper", request_id="pause-retained-run")
    elif control == "stop":
        _intent(world, "sleeper", policy=STOP_POLICY_IMMEDIATE)
    else:
        # The existing restart carrier; no process restart is simulated here.
        parked["_budget_pause_hold"] = {"reason": "owner_restart_hold"}
    assert _chats(world) == [], "a retained sleep receipt cannot outrank the later control"


def test_retry_uses_logical_root_stop_without_hiding_graceful_summary(world):  # noqa: F811
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE
    from ouroboros.gateway.tasks import _record_stop
    from ouroboros.task_results import write_task_result
    from tests.test_reply_preparation import _pooled

    old = "logical-root"
    write_task_result(world.root, old, "failed", chat_id=6, root_task_id=old,
                      superseded_by="retry-root", retry_task_id="retry-root", delegation_role="root")
    meta = _pooled(world, "retry-root", chat_id=6)
    lineage = {"root_task_id": old, "original_task_id": old, "timeout_retry_from": old,
               "supersedes_task_id": old, "delegation_role": "root"}
    meta["task"].update(lineage)
    write_task_result(world.root, "retry-root", "running", chat_id=6, **lineage)
    assert _chats(world) == [6]
    _intent(world, old, policy=STOP_POLICY_FINALIZE, scope="cascade")
    assert _chats(world) == [6], "the summary is still being prepared"
    intent = _record_stop("retry-root", cascade=True, stop_action_id="stop-retry")
    assert intent and intent["task_id"] == old, "Stop is retained against the logical root"
    assert _chats(world) == [], "the retry must see its logical root's immediate Stop"


def test_reply_refresh_does_not_load_the_optional_project_question_registry(world, monkeypatch):  # noqa: F811
    from tests.test_reply_preparation import _pooled

    def unwanted_read(*_args, **_kwargs):
        reads.append(True)
        return []

    reads = []
    monkeypatch.setattr("ouroboros.projects_registry.list_reserved_projects", unwanted_read)
    _pooled(world, "one-main-reply", chat_id=1)
    assert _chats(world) == [1]  # first routing read populates the existing project-thread cache
    reads.clear()
    for _ in range(3):
        assert _chats(world) == [1]
    assert reads == [], "typing needs live waits, not the full project/question directory"


def test_display_wait_marks_do_not_spare_idle_without_an_actual_work_lease():
    from supervisor.cognitive_operations import _active_operation_progressing

    leases = {"wait": {"kind": "in_slot_wait", "until_ts": 20},
              "expired": {"kind": "review", "until_ts": 5}}
    meta = {"active_operation_leases": leases}
    assert _active_operation_progressing(meta, 10) is False
    assert "wait" in leases and "expired" not in leases
    leases["tool"] = {"kind": "tool", "until_ts": 20}
    assert _active_operation_progressing(meta, 10) is True
    assert _active_operation_progressing(meta, 21) is False
    assert "active_operation_leases" not in meta


def test_cascade_stop_of_a_waited_helper_fences_its_whole_subtree(world):  # noqa: F811
    import time
    from types import SimpleNamespace
    from ouroboros.gateway.tasks import _record_stop
    from ouroboros.model_sleep import declare_in_slot_wait
    from tests.test_reply_preparation import _pooled

    _pooled(world, "waiting-root", chat_id=3)
    _pooled(world, "working-child", chat_id=3, parent="waiting-root")
    grandchild = _pooled(world, "working-grandchild", chat_id=3, parent="working-child")
    grandchild["task"]["root_task_id"] = "waiting-root"
    ctx = SimpleNamespace(task_id="waiting-root", task_attempt=1, event_queue=_Supervisor(world),
                          current_chat_id=3)
    declare_in_slot_wait(ctx, "tasks:children", time.time() + 60)
    assert _chats(world) == [3]
    assert _record_stop("working-child", cascade=True, stop_action_id="stop-child")
    assert _chats(world) == [], "a subtree Stop fences the helper before its RUNNING row disappears"
