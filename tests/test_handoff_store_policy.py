"""Local admission must not lend its no-wait policy to accepted physical work."""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from ouroboros import _usage_wait, owner_pause, store, usage_accounting as usage
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_admission_contention import database_reader, publish
from tests._cases import chosen_rows

SURFACES = ("ordinary", "run_operation", "submit_tool", "handed_inline_tool", "model_send",
            "model_send_async", "async_operation", "async_preparation")


@pytest.fixture
def source(tmp_path):
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    return SimpleNamespace(task_id="root", root_task_id="root", task_metadata={}, drive_root=tmp_path,
                           budget_drive_root=str(tmp_path))


def _invoke(surface, source, body):
    if surface == "ordinary":
        return body()
    if surface == "run_operation":
        return owner_pause.run_operation(source, body)
    if surface == "submit_tool":
        with ThreadPoolExecutor(max_workers=1) as executor:
            return owner_pause.submit_tool(source, "probe", executor.submit, body).result()
    if surface == "handed_inline_tool":
        from ouroboros.loop_tool_execution import StatefulToolExecutor
        executor = StatefulToolExecutor()

        def handed():
            with owner_pause.tool_handoff(source, "probe") as claim:
                assert claim["handed"]
                thread = threading.get_ident()
                def inline():
                    assert threading.get_ident() == thread
                    return body()
                return owner_pause.run_tool_handler(source, inline)

        try:
            return owner_pause.submit_tool(source, "probe", executor.submit, handed).result()
        finally:
            executor.shutdown()
    reservation = SimpleNamespace(scope=source, attempt_id="physical-probe")
    if surface == "model_send":
        return _usage_wait.model_send(reservation, body)

    async def complete():
        assert not store._ADMISSION_NO_WAIT.get(), "Task creation copied the admission guard"
        return await asyncio.to_thread(body)

    async def start():
        if surface == "model_send_async":
            return await _usage_wait.model_send_async(reservation, complete, retain_on_cancel=lambda *_: None)
        if surface == "async_operation":
            return await owner_pause.submit_async_operation(source, complete)
        return await owner_pause.submit_async_preparation(complete, source=source)

    return asyncio.run(start())


@pytest.mark.parametrize("surface", chosen_rows(SURFACES, 0, 7))
@pytest.mark.parametrize("budget", [None, 0.8])
def test_handed_body_waits_out_real_sqlite_contention_with_its_original_policy(source, surface, budget):
    effects = []
    def body():
        assert store._WAIT_BUDGET.get() == budget
        assert not store._ADMISSION_NO_WAIT.get()
        effects.append("body")
        with database_reader(source.drive_root, exclusive=True) as release:
            timer = threading.Timer(0.25, release)
            timer.start()
            try:
                write_task_result(source.drive_root, "root", "running", note="physical-result")
            finally:
                timer.join(3)
        return "done"

    with store.acquisition_budget(budget) if budget is not None else contextlib.nullcontext():
        assert _invoke(surface, source, body) == "done"
        assert store._WAIT_BUDGET.get() == budget and not store._ADMISSION_NO_WAIT.get()
    assert effects == ["body"]
    assert load_task_result(source.drive_root, "root", strict=True)["note"] == "physical-result"


@pytest.mark.parametrize("surface", SURFACES[1:])
@pytest.mark.parametrize("budget", [0, 0.03])
def test_post_handoff_busy_preserves_deliberate_budget_and_never_repeats_effect(source, surface, budget):
    effect = source.drive_root / "physical-effect.txt"
    def body():
        assert store._WAIT_BUDGET.get() == budget and not store._ADMISSION_NO_WAIT.get()
        with effect.open("a", encoding="utf-8") as target:
            target.write("effect\n")
        with database_reader(source.drive_root, exclusive=True):
            write_task_result(source.drive_root, "root", "running", note="after-effect")
        pytest.fail("the explicit physical budget must expire before this held writer is released")

    with store.acquisition_budget(budget):
        with pytest.raises(store.StoreBusy):
            _invoke(surface, source, body)
        assert store._WAIT_BUDGET.get() == budget and not store._ADMISSION_NO_WAIT.get()
    assert effect.read_text(encoding="utf-8").splitlines() == ["effect"]
    assert "note" not in load_task_result(source.drive_root, "root", strict=True)


@pytest.mark.parametrize("surface", SURFACES[1:])
def test_physical_context_keeps_owner_usage_review_and_arbitrary_bindings(source, surface):
    sentinel = contextvars.ContextVar("handoff-test-sentinel")
    marker = object()
    token = sentinel.set(marker)
    scope = usage.UsageScope(drive_root=source.drive_root, task_id="root", root_task_id="root")
    try:
        with task_model_wait_scope(task={"id": "root", "_attempt": 5}, drive_root=source.drive_root,
                                   event_queue=None, worker_slot_held=False) as waiter, \
                usage.usage_scope(scope), usage.physical_attempt_limit(2), \
                owner_pause.review_episode("root") as review, store.acquisition_budget(0.8):
            expected = dict(contextvars.copy_context().items())
            def body():
                for variable, value in expected.items():
                    assert variable.get() is value, variable.name
                assert sentinel.get() is marker and usage.current_usage_scope() is scope
                from ouroboros.model_wait import current_model_wait
                assert current_model_wait() is waiter and owner_pause._REVIEW_EPISODE.get() is review
                assert not store._ADMISSION_NO_WAIT.get()
                if surface.startswith("model_send"):
                    assert owner_pause.model_handed_off("physical-probe")
                return "context retained"
            assert _invoke(surface, source, body) == "context retained"
            assert not store._ADMISSION_NO_WAIT.get() and store._WAIT_BUDGET.get() == 0.8
    finally:
        sentinel.reset(token)


@pytest.mark.parametrize("pause", [False, True])
def test_nested_start_has_fresh_admission_and_outer_body_holds_no_launch_lock(source, monkeypatch, pause):
    entered, resume = threading.Event(), threading.Event()
    seen = []
    validate = owner_pause._validate_launch
    def checked(*args, **kwargs):
        seen.append((store._ADMISSION_NO_WAIT.get(), store._WAIT_BUDGET.get()))
        return validate(*args, **kwargs)
    monkeypatch.setattr(owner_pause, "_validate_launch", checked)

    def outer():
        assert not store._ADMISSION_NO_WAIT.get()
        entered.set()
        assert resume.wait(5)
        return owner_pause.run_operation(source, lambda: write_task_result(
            source.drive_root, "root", "running", note="nested physical result"))

    with store.acquisition_budget(0.8), ThreadPoolExecutor(max_workers=1) as executor:
        future = owner_pause.submit_tool(source, "outer", executor.submit, outer)
        try:
            assert entered.wait(3)
            with owner_pause.launch_lock(source.drive_root, "root", wait=False):
                pass
            if pause:
                owner_pause.install_fence(source.drive_root, "root", request_id="between-handoffs")
        finally:
            resume.set()
        if pause:
            with pytest.raises(owner_pause.OwnerPauseRefused):
                future.result()
        else:
            assert future.result()["note"] == "nested physical result"
    assert seen == [(True, 0.8), (True, 0.8)]


def test_failed_submission_restores_transient_parent_scope_without_losing_other_context(source):
    def rejected(*_args):
        assert not store._ADMISSION_NO_WAIT.get()
        assert store._WAIT_BUDGET.get() == 0.6
        raise RuntimeError("executor refused the local handoff")
    with store.acquisition_budget(0.6), store.admission_acquisition():
        with pytest.raises(RuntimeError, match="executor refused"):
            owner_pause.submit_tool(source, "probe", rejected, lambda: pytest.fail("body started"))
        assert store._ADMISSION_NO_WAIT.get() and store._WAIT_BUDGET.get() == 0.6
    assert not store._ADMISSION_NO_WAIT.get() and store._WAIT_BUDGET.get() is None


@pytest.mark.parametrize(("asynchronous", "reason", "stage"), [
    (False, "contention", "submit"),
    (False, "identity_unreadable", "claim"),
    (True, "permission", "claim"),
    (True, "kernel_refused", "submit"),
])
def test_launch_lock_outcome_keeps_proven_unsent_distinct_from_contention(
    source, monkeypatch, stage, reason, asynchronous,
):
    from ouroboros.llm_attempt import _PhysicalSendNotStarted
    from ouroboros import queue_observation
    publish(source.drive_root)
    lock = owner_pause.launch_lock
    locks, sends, waits = [], [], []
    @contextlib.contextmanager
    def acquire(*args, **kwargs):
        locks.append(True)
        if len(locks) == (1 if stage == "claim" else 2):
            raise owner_pause.LaunchBusy(reason)
        with lock(*args, **kwargs):
            yield
    monkeypatch.setattr(owner_pause, "launch_lock", acquire)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _seconds: waits.append(True))
    request = usage.AttemptRequest(model="test", provider="test", task_id="root", root_task_id="root",
                                   drive_root=source.drive_root, reservation_usd=0.01)
    def send():
        sends.append(True)
        return "answer"
    async def send_async():
        return send()
    def run():
        if asynchronous:
            return asyncio.run(usage.execute_physical_attempt_async(
                request, send_async, extractor=lambda _r: ({}, 0.1, True)))
        return usage.execute_physical_attempt(request, send, extractor=lambda _r: ({}, 0.1, True))
    if reason == "contention":
        assert run() == "answer" and sends == [True]
    else:
        with pytest.raises(_PhysicalSendNotStarted) as failure:
            run()
        assert isinstance(failure.value.__cause__, owner_pause.LaunchBusy)
        assert failure.value.__cause__.reason == reason and not sends and not waits
    (row,) = usage.read_usage_records(source.drive_root, final_only=True)
    assert row["state"] == ("settled" if reason == "contention" else "released")
