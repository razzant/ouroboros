"""Install/update reviews keep caller authority after the author leaves the tool."""
import contextlib
import queue

import pytest

from ouroboros import model_wait, review_ledger
import ouroboros.review_substrate as substrate
from tests.test_review_skill import harness  # noqa: F401 -- shared offline review fixture


@pytest.mark.parametrize("operation", ["install", "update"])
@pytest.mark.parametrize("author_ended", [False, True])
def test_source_tool_review_keeps_the_real_caller_after_the_tool_returns(harness, operation, author_ended):  # noqa: F811 -- imported pytest fixture
    """The install/update starts one ordinary wave; an HTTP Skills review remains manual."""
    import contextvars
    import threading
    from ouroboros import skill_review_runner as runner, skill_source_install as install
    from tests.test_skill_source_tools import _typed_call
    from tests.test_skill_review_runner import _reset_queue
    from ouroboros.skill_review_runner import skill_review_ui_projection
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope
    from ouroboros.task_results import write_task_result, STATUS_RUNNING

    h = harness
    h.ctx.current_chat_id = 218403133
    h.ctx.project_id = "skills-project"
    h.ctx.budget_drive_root = str(h.drive)
    h.ctx.task_lifecycle_bound = True
    h.ctx.event_queue = queue.Queue()
    write_task_result(h.drive, h.ctx.task_id, STATUS_RUNNING)
    h.ctx.task_metadata = {"root_task_id": h.ctx.task_id, "budget_drive_root": str(h.drive),
                           "owner_source": {"kind": "quiz", "quiz_id": "selected-install"}}
    scope = contextvars.ContextVar("test_install_review_scope", default="lost")
    token = scope.set("original-host-scope")
    entered, release = threading.Event(), threading.Event()
    observed, payloads = [], []
    physical_scopes, physical_waits = [], []
    physical = substrate.ReviewCoordinator._run_slot

    def observe_physical(self, *args, **kwargs):
        physical_scopes.append(current_usage_scope())
        physical_waits.append(model_wait.current_model_wait())
        return physical(self, *args, **kwargs)

    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", observe_physical)
    original = runner.run_skill_review_lifecycle_blocking

    def deferred(ctx, name, **kwargs):
        observed.append((ctx, scope.get(), dict(getattr(ctx, "task_metadata", None) or {}), kwargs["source"]))
        entered.set()
        assert release.wait(15), "test did not release the review"
        payload = original(ctx, name, **kwargs)
        payloads.append(payload)
        return payload

    h.monkeypatch.setattr(runner, "run_skill_review_lifecycle_blocking", deferred)
    installed = {"ok": True, "name": "weather", "needs_review": True, "source": {"name": "review source"}}
    h.monkeypatch.setattr(install, "install_item", lambda *a, **kw: dict(installed))
    h.monkeypatch.setattr(install, "update_installed", lambda *a, **kw: dict(installed))
    _reset_queue()
    try:
        args = {"id": "hub/weather", "reason": "weather task"} if operation == "install" else {"name": "weather", "mode": "replace"}
        with contextlib.ExitStack() as author:
            author_wait = author.enter_context(model_wait.task_model_wait_scope(
                task={"id": h.ctx.task_id, "chat_id": h.ctx.current_chat_id, "metadata": h.ctx.task_metadata},
                drive_root=h.drive, event_queue=h.ctx.event_queue, worker_slot_held=True))
            author_wait.tool_context = h.ctx
            h.ctx.model_wait_context = author_wait
            author.enter_context(usage_scope(UsageScope(
                drive_root=h.drive, task_id=h.ctx.task_id, root_task_id=h.ctx.task_id,
                billing_group_id="original-work", billing_group_limit_usd=17.5,
                global_limit_usd=100, root_limit_usd=20, root_limit_source="owner")))
            _typed_call("skill_" + operation, h.ctx, **args, reviewers=["t1"],
                        **({"review_reason": "small payload"} if operation == "install" else {"reason": "small payload"}))
            assert entered.wait(15)
            [thread] = [t for t in threading.enumerate() if t.name == "skill-install-review-weather"]
            assert thread.is_alive() and not payloads, "the tool returned while its review still runs"
            while not h.ctx.event_queue.empty():
                h.ctx.event_queue.get_nowait()  # The source tool's own synchronous wait already ended.
            if author_ended:
                author.close()
                write_task_result(h.drive, h.ctx.task_id, "completed")
                assert author_wait.closed
            if not author_ended:
                h.ctx.task_metadata["root_task_id"] = "next-tool-must-not-relocate-the-wave"
            h.ctx.current_chat_id = 1
            release.set()
            thread.join(120)
            assert not thread.is_alive()
    finally:
        release.set()
        scope.reset(token)
        _reset_queue()
    [payload] = payloads
    captured, host_scope, metadata, source = observed[0]
    assert captured is not h.ctx and captured.event_queue is h.ctx.event_queue
    assert captured.task_id == h.ctx.task_id and captured.current_chat_id == 218403133
    assert captured.budget_drive_root == str(h.drive) and captured.project_id == "skills-project"
    assert captured.task_lifecycle_bound == h.ctx.task_lifecycle_bound
    [billing] = physical_scopes
    assert billing.task_id == h.ctx.task_id and billing.root_task_id == h.ctx.task_id
    assert billing.billing_group_id == "original-work" and billing.billing_group_limit_usd == 17.5
    assert billing.global_limit_usd == 100 and billing.root_limit_usd == 20
    assert billing.review_slot_id == "t1" and not billing.non_task_operation
    assert metadata["owner_source"] == {"kind": "quiz", "quiz_id": "selected-install"}
    assert host_scope == "original-host-scope" and source == "tool"
    record = review_ledger.load_record(h.drive, payload["record_id"])
    assert record["task_id"] == h.ctx.task_id
    assert record["root_task_id"] == h.ctx.task_id
    facts = record["brief"]["skill"]
    assert facts["source"] == "tool" and facts["group_id"] == f"task:{h.ctx.task_id}:weather"
    assert [send["slot_id"] for send in h.sends] == ["t1"], "no second review for task presentation"
    projection = skill_review_ui_projection(h.drive, "weather")
    assert projection["current"]["presentation_owner_task_id"] == h.ctx.task_id
    assert projection["current"]["chat_id"] == 218403133
    assert not hasattr(h.ctx, "_skill_review_source"), "the async carrier must not mutate the next tool call"
    # The captured caller still reads its original task's Pause authority,
    # not a detached manual identity, after the source tool returned.
    from ouroboros.owner_pause import install_fence, member_fence, fence_closed
    if not author_ended:
        install_fence(h.drive, captured.task_id, request_id="after-install")
        assert fence_closed(member_fence(captured))
    [review_wait] = physical_waits
    assert review_wait is not author_wait and review_wait.owner_id.startswith("review-operation-")
    assert review_wait.task["metadata"]["root_task_id"] == h.ctx.task_id
    assert review_wait.task["chat_id"] == 218403133
    assert review_wait.worker_slot_held is False and review_wait.tool_context is None
    events = list(h.ctx.event_queue.queue)
    assert not [event for event in events if event.get("type") == "typing_start"], events
    assert not [event for event in events if event.get("kind") == "in_slot_wait"], events


@pytest.mark.parametrize("control", ["none", "pause", "stop"])
def test_background_source_review_retains_real_pre_dispatch_controls(harness, control):  # noqa: F811 -- imported pytest fixture
    import threading
    from ouroboros import skill_review_runner as runner, skill_source_install as install
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result
    from tests.test_skill_review_runner import _reset_queue

    h = harness
    h.ctx.current_chat_id = 7
    h.ctx.event_queue = queue.Queue()
    h.ctx.task_metadata = {"root_task_id": h.ctx.task_id, "budget_drive_root": str(h.drive)}
    write_task_result(h.drive, h.ctx.task_id, "running")
    entered, release = threading.Event(), threading.Event()
    original = runner.run_skill_review_lifecycle_blocking
    observed_controls, payloads = [], []
    physical, refusals, transport_calls = substrate.ReviewCoordinator._run_slot, [], []

    class Transport:
        @model_wait.model_waitable
        def chat(self, *, model, model_role="reviewer", **kwargs):
            transport_calls.append(model)
            return {}, {}

    def controlled_send(self, request, slot, **kwargs):
        try:
            Transport().chat(model=slot.model)
        except model_wait.ModelWaitInterrupted as exc:
            refusals.append(exc.control_reason)
            raise
        return physical(self, request, slot, **kwargs)

    def deferred(*args, **kwargs):
        entered.set()
        assert release.wait(15)
        observed_controls.append(model_wait.current_model_wait().control_reason(strict_current=True))
        result = original(*args, **kwargs)
        payloads.append(result)
        return result

    h.monkeypatch.setattr(runner, "run_skill_review_lifecycle_blocking", deferred)
    h.monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", controlled_send)
    _reset_queue()
    try:
        with model_wait.task_model_wait_scope(task={"id": h.ctx.task_id, "chat_id": 7,
                "metadata": h.ctx.task_metadata}, drive_root=h.drive, event_queue=h.ctx.event_queue,
                worker_slot_held=True) as author:
            author.tool_context = h.ctx
            install.start_review(h.drive, "weather", ctx=h.ctx, reviewers=["t1"], reason="small payload")
            assert entered.wait(15)
        assert author.closed
        if control == "pause":
            install_fence(h.drive, h.ctx.task_id, request_id="after-source-tool")
        else:
            write_task_result(h.drive, h.ctx.task_id, "completed")
            if control == "stop":
                request_cancel(h.drive, h.ctx.task_id, reason="owner stop", source="test", allow_settled_target=True)
        [thread] = [t for t in threading.enumerate() if t.name == "skill-install-review-weather"]
        release.set()
        thread.join(120)
        assert not thread.is_alive()
    finally:
        release.set()
        _reset_queue()
    assert len(h.sends) == len(transport_calls) == (1 if control == "none" else 0)
    assert observed_controls == (["cancelled"] if control == "stop" else [None])
    assert refusals == (["cancelled"] if control == "stop" else [])
    assert len(payloads) == 1 and (payloads[0]["status"] == "clean") == (control == "none"), payloads
    assert not [e for e in h.ctx.event_queue.queue if e.get("type") == "typing_start"]


def test_synchronous_skill_review_still_announces_its_author_wait(harness):  # noqa: F811 -- imported pytest fixture
    from ouroboros.task_results import write_task_result
    from tests.test_review_skill import _review

    h = harness
    h.ctx.event_queue = queue.Queue()
    write_task_result(h.drive, h.ctx.task_id, "running")
    with model_wait.task_model_wait_scope(task={"id": h.ctx.task_id}, drive_root=h.drive,
            event_queue=h.ctx.event_queue, worker_slot_held=True) as author:
        author.tool_context = h.ctx
        assert _review(h, reviewers=["t1"], reason="small payload").status == "clean"
    events = list(h.ctx.event_queue.queue)
    assert [e["phase"] for e in events if e.get("kind") == "in_slot_wait"] == ["started", "finished"]
    assert any(e.get("type") == "typing_start" for e in events)
