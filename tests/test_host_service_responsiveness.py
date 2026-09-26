"""The Host Service keeps the owner's surfaces responsive while Presence turns wait (TZ3).

Before: token/grant/admission reads ran inline on the ASGI loop the owner's API shares, and
each admitted turn — including its wait on the presence gate and its model/quota waits — ran
on the shared default executor. These tests pin the replacement: blocking checks leave the
loop, a queued turn holds no thread, an executing turn owns its thread, a cancelled HTTP wait
neither stops the work nor frees its capacity early, and a retry of the same event joins it.
"""

import asyncio
import contextvars
import json
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros import presence_runner
from ouroboros.gateway import host_service
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import PresenceTurnError, PresenceTurnGate
from tests.test_host_service_api import _seed_presence_behavior, _seed_token

TOKEN = "presence-token"
BUDGET = "telegram-bot:presence"


@pytest.fixture(autouse=True)
def _isolated_gates(monkeypatch):
    """The configured-gate cache is process-global; each test gets its own, at two active turns."""
    monkeypatch.setattr(presence_runner, "_GATES", {})
    monkeypatch.setenv("OUROBOROS_PRESENCE_MAX_ACTIVE", "2")


def _event(source_event_id: str, conversation_id: str = "room-1") -> dict:
    return {
        "source_event_id": source_event_id, "provider": "telegram", "account_id": "bot-1",
        "conversation_id": conversation_id, "thread_id": "topic-1", "conversation_key": "ignored",
        "actor": {"platform_actor_id": "user-7"}, "conversation": {}, "message": {}, "text": "Hello",
    }


def _request(app, body=None):
    async def payload():
        return body

    return SimpleNamespace(app=app, headers={"x-skill-token": TOKEN}, json=payload)


def _turn(app, binding_id: str, source_event_id: str, conversation_id: str = "room-1"):
    return host_service._api_presence_turn(
        _request(app, {"binding_id": binding_id, "event": _event(source_event_id, conversation_id)}))


def _presence_app(tmp_path: pathlib.Path, runner, *, account_wide: bool = False):
    _seed_token(tmp_path, skill="telegram-bot", token=TOKEN, permissions=["presence"],
                manifest_permissions=["presence"])
    binding_id = _seed_presence_behavior(tmp_path, account_wide=account_wide)
    app = create_host_service_app(tmp_path, presence_runner=runner)
    return app, binding_id, app.state.host_service_context


def _answer(**kwargs):
    event_id = kwargs["event"].source_event_id
    return SimpleNamespace(outcome="message", text=f"answer {event_id}", task_id=event_id, work_ref="")


async def _until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


def _turn_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("presence-turn-")]


def test_queued_and_executing_turns_leave_the_default_executor_free(tmp_path):
    """Four live turns, two gate slots, ONE default-executor worker: the worker stays free.

    Before, each turn waited inside ``asyncio.to_thread`` — the first parked turn was that
    worker, and the owner's reads (and every Host authentication) queued behind a model.
    """
    entered, release = [], threading.Event()

    def runner(**kwargs):
        entered.append(kwargs["event"].source_event_id)
        assert release.wait(10)
        return _answer(**kwargs)

    app, binding_id, ctx = _presence_app(tmp_path, runner, account_wide=True)

    async def scenario():
        shared = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shared-default")
        asyncio.get_running_loop().set_default_executor(shared)
        turns = [asyncio.create_task(_turn(app, binding_id, f"e-{index}", f"room-{index}")) for index in range(4)]
        try:
            await _until(lambda: len(entered) == 2 and ctx._inflight[BUDGET] == 4)
            await asyncio.sleep(0.2)
            assert len(entered) == 2, "the gate admits two turns; two stay queued"
            assert len(ctx.presence_turns.live()) == 4
            assert len(_turn_threads()) == 2, "a queued turn holds no thread"
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 2) == "free"
            identity = await asyncio.wait_for(host_service._api_identity(_request(app)), 5)
            assert identity.status_code == 200
        finally:
            release.set()
        responses = await asyncio.wait_for(asyncio.gather(*turns), 10)
        assert sorted(json.loads(response.body)["text"] for response in responses) == [
            f"answer e-{index}" for index in range(4)]
        shared.shutdown(wait=True)

    asyncio.run(scenario())
    assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())


def test_retries_join_turns_whose_waiters_were_cancelled_even_at_full_budget(tmp_path):
    """Cancelled waiters (one executing turn, one queued) leave both turns running with their
    capacity; retries join them although the in-flight budget is full, and each event runs once."""
    calls, entered, release = [], threading.Event(), threading.Event()

    def runner(**kwargs):
        calls.append(kwargs["event"].source_event_id)
        entered.set()
        assert release.wait(10)
        return _answer(**kwargs)

    app, binding_id, ctx = _presence_app(tmp_path, runner)

    async def scenario():
        # One conversation: e-0 executes, e-1..e-4 queue behind it; five turns fill the budget.
        waiters = {}
        for index in range(5):
            waiters[f"e-{index}"] = asyncio.create_task(_turn(app, binding_id, f"e-{index}"))
            await _until(lambda: ctx._inflight[BUDGET] == index + 1)
            if index == 0:
                await _until(entered.is_set)
        for event_id in ("e-0", "e-2"):
            waiters[event_id].cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiters.pop(event_id)
        await asyncio.sleep(0.1)
        assert calls == ["e-0"]
        assert ctx._inflight[BUDGET] == 5, "a cancelled wait does not return its turn's capacity"
        assert len(ctx.presence_turns.live()) == 5

        refused = await _turn(app, binding_id, "e-5")
        assert refused.status_code == 429, "a NEW event still meets the budget"
        retries = {event_id: asyncio.create_task(_turn(app, binding_id, event_id)) for event_id in ("e-0", "e-2")}
        await asyncio.sleep(0.2)
        assert not any(retry.done() for retry in retries.values()), "a retry joins; it is never refused"
        assert calls == ["e-0"] and ctx._inflight[BUDGET] == 5

        release.set()
        for event_id, retry in retries.items():
            response = await asyncio.wait_for(retry, 10)
            assert response.status_code == 200
            assert json.loads(response.body)["text"] == f"answer {event_id}"
        for response in await asyncio.wait_for(asyncio.gather(*waiters.values()), 10):
            assert response.status_code == 200

    asyncio.run(scenario())
    assert sorted(calls) == [f"e-{index}" for index in range(5)], "every event ran exactly once"
    assert ctx.presence_turns.live() == [] and not any(ctx._inflight.values())


def test_turn_thread_carries_the_admitting_request_context(tmp_path):
    """A raw thread starts with an empty context; the turn must see what to_thread carried."""
    from ouroboros.config import runtime_setting, task_settings_scope
    from ouroboros.settings_integrity import TaskSettingsSnapshot

    probe = contextvars.ContextVar("tz3_probe", default="lost")
    seen = {}

    def runner(**kwargs):
        seen.update(probe=probe.get(), setting=runtime_setting("OUROBOROS_TZ3_CONTEXT_PROBE"),
                    thread=threading.current_thread().name)
        return _answer(**kwargs)

    app, binding_id, _ctx = _presence_app(tmp_path, runner)

    async def scenario():
        probe.set("carried")
        with task_settings_scope(TaskSettingsSnapshot(settings={}, environ={"OUROBOROS_TZ3_CONTEXT_PROBE": "task"})):
            return await _turn(app, binding_id, "e-ctx")

    assert asyncio.run(scenario()).status_code == 200
    assert seen["thread"].startswith("presence-turn-")
    assert seen["probe"] == "carried" and seen["setting"] == "task"


def test_a_thread_that_cannot_start_returns_capacity_gate_and_identity(tmp_path, monkeypatch):
    calls = []
    app, binding_id, ctx = _presence_app(tmp_path, lambda **kwargs: calls.append(1) or _answer(**kwargs))
    start = threading.Thread.start

    def refuse_turn_threads(thread):
        if thread.name.startswith("presence-turn-"):
            raise RuntimeError("can't start new thread")
        return start(thread)

    monkeypatch.setattr(threading.Thread, "start", refuse_turn_threads)
    refused = asyncio.run(_turn(app, binding_id, "e-1"))
    assert refused.status_code == 503
    assert json.loads(refused.body)["error"] == "presence turn thread could not start"
    assert calls == [] and ctx.presence_turns.live() == [] and not any(ctx._inflight.values())

    monkeypatch.setattr(threading.Thread, "start", start)
    # The gate lease went back too: the same conversation admits at once.
    retried = asyncio.run(asyncio.wait_for(_turn(app, binding_id, "e-1"), 5))
    assert retried.status_code == 200 and calls == [1]


@pytest.mark.parametrize("outcome", ["answered", "refused"])
def test_a_turn_retires_from_the_live_set_before_its_outcome_is_published(outcome):
    """The live set is custody the Host reads the moment a waiter sees the terminal.

    Settlement runs on the turn's own thread while the waiter wakes on the loop, so an id
    retired only after the outcome is set can outlive a response already on the wire (CI saw
    a settled turn still live right after its 409 and after twelve 200s). Capacity returns
    first, the id retires, and only then does the shared future publish — for a result and
    for a refusal alike; the probe reads the live set at the exact publication call.
    """
    from contextlib import ExitStack
    from ouroboros.presence_runner import PresenceTurnExecutions, PresenceTurnLease

    executions, capacity, published = PresenceTurnExecutions(), [], []

    async def admit():
        return PresenceTurnLease("room", ExitStack())

    def run(_lease):
        if outcome == "refused":
            raise PresenceTurnError("presence_start_unwritable", "source_event_id", turn_ref="turn-1")
        return outcome

    async def scenario():
        execution, started = executions.start_or_join(
            "turn-1", reserve=lambda: capacity.append("held") or True, release=lambda: capacity.remove("held"),
            admit=admit, run=run)
        assert started
        for name in ("set_result", "set_exception"):
            original = getattr(execution.result, name)

            def publish(*args, _original=original):
                published.append((executions.live(), list(capacity)))
                return _original(*args)

            setattr(execution.result, name, publish)
        waiter = asyncio.wait_for(asyncio.wrap_future(execution.result), 5)
        if outcome == "refused":
            with pytest.raises(PresenceTurnError):
                await waiter
        else:
            assert await waiter == "answered"
        # No quiescence wait: the waiter observes the outcome, so the id is already gone.
        assert executions.live() == [] and capacity == []

    asyncio.run(scenario())
    assert published == [([], [])]


@pytest.mark.parametrize("failure_at", ["context", "constructor"])
def test_thread_preparation_failure_releases_admitted_work(failure_at, monkeypatch):
    from contextlib import ExitStack
    from ouroboros.presence_runner import (
        PresenceTurnExecutions, PresenceTurnLease, PresenceTurnNotStarted,
    )

    executions, capacity, gate = PresenceTurnExecutions(), [], []

    async def admit():
        resources = ExitStack()
        gate.append("held")
        resources.callback(gate.remove, "held")
        return PresenceTurnLease("room", resources)

    def fail(*args, **kwargs):
        raise RuntimeError("thread preparation refused")

    async def scenario():
        # Install faults after asyncio's own Context preparation. Only the
        # admitted execution's thread preparation is under test.
        execution, _ = executions.start_or_join(
            "turn-preparation", reserve=lambda: capacity.append("held") or True,
            release=lambda: capacity.remove("held"), admit=admit,
            run=lambda lease: pytest.fail("no work may start"),
        )
        with monkeypatch.context() as patch:
            patch.setattr(contextvars if failure_at == "context" else threading,
                          "copy_context" if failure_at == "context" else "Thread", fail)
            await execution.admission
        with pytest.raises(PresenceTurnNotStarted):
            execution.result.result()

    asyncio.run(scenario())
    assert capacity == [] and gate == [] and executions.live() == []


@pytest.mark.parametrize("admission_started", [False, True])
def test_an_interrupted_admission_settles_waiters_and_capacity(admission_started):
    """A loop that cancels a queued turn (shutdown) must not strand its waiters or its slot,
    including a cancel that lands before the admission coroutine ever ran."""
    from ouroboros.presence_runner import PresenceTurnExecutions, PresenceTurnNotStarted

    executions, capacity = PresenceTurnExecutions(), []

    async def never_admitted():
        await asyncio.Event().wait()

    async def scenario():
        execution, started = executions.start_or_join(
            "turn-1", reserve=lambda: capacity.append("held") or True, release=lambda: capacity.remove("held"),
            admit=never_admitted, run=lambda _lease: pytest.fail("must not run"))
        assert started and capacity == ["held"]
        if admission_started:
            await asyncio.sleep(0.05)
        execution.admission.cancel()
        with pytest.raises(PresenceTurnNotStarted):
            await asyncio.wait_for(asyncio.wrap_future(execution.result), 2)

    asyncio.run(scenario())
    assert capacity == [] and executions.live() == []


def test_blocking_checks_run_off_the_event_loop(tmp_path, monkeypatch):
    """Token discovery and presence admission read disk; the loop keeps ticking through both."""
    import ouroboros.presence_admission as presence_admission

    app, binding_id, ctx = _presence_app(tmp_path, _answer)
    authenticate, admit = ctx.authenticate_token_payload, presence_admission.admit_presence_turn

    def slow_authenticate(raw_token):
        time.sleep(0.3)
        return authenticate(raw_token)

    def slow_admit(**kwargs):
        time.sleep(0.3)
        return admit(**kwargs)

    monkeypatch.setattr(ctx, "authenticate_token_payload", slow_authenticate)
    monkeypatch.setattr(presence_admission, "admit_presence_turn", slow_admit)

    async def ticks_during(awaitable):
        ticks, done = 0, asyncio.Event()

        async def tick():
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        ticker = asyncio.create_task(tick())
        try:
            response = await awaitable
        finally:
            done.set()
            await ticker
        return response.status_code, ticks

    async def scenario():
        return (await ticks_during(host_service._api_identity(_request(app))),
                await ticks_during(_turn(app, binding_id, "e-1")))

    (identity_status, identity_ticks), (turn_status, turn_ticks) = asyncio.run(scenario())
    assert identity_status == 200 and turn_status == 200
    # Inline on the loop, each 0.3 s check would be one tick.
    assert identity_ticks > 10 and turn_ticks > 20


def test_coroutine_admission_keeps_the_cross_process_gate_contract(tmp_path):
    """``admit`` takes the same file locks ``run`` takes: another gate instance (another process
    in production) waits for the conversation and the slot cap, and cancellation leaks nothing."""
    host = PresenceTurnGate(1, state_root=tmp_path)
    other = PresenceTurnGate(1, state_root=tmp_path)
    entered = threading.Event()

    async def scenario():
        lease = await host.admit("conversation-a")
        waiting = asyncio.create_task(host.admit("conversation-a"))
        await asyncio.sleep(0.1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        blocked = threading.Thread(target=other.run, args=("conversation-b", entered.set))
        blocked.start()
        try:
            await asyncio.sleep(0.3)
            assert not entered.is_set(), "the one cross-process slot is held by the lease"
        finally:
            lease.release()
        await asyncio.to_thread(blocked.join, 5)
        assert entered.is_set()
        # Nothing the cancelled admission took stayed held: the conversation admits at once.
        again = await asyncio.wait_for(host.admit("conversation-a"), 2)
        again.release()
        again.release()  # idempotent

    asyncio.run(scenario())


def test_an_admitted_turn_runs_only_under_its_own_conversation(tmp_path):
    from ouroboros.presence_runner import PresenceTurnEvent, run_presence_turn

    lease = asyncio.run(PresenceTurnGate(1).admit("telegram:bot-1:room-2:topic-1"))
    event = PresenceTurnEvent(
        source_event_id="e-1", provider="telegram", account_id="bot-1", conversation_id="room-1",
        thread_id="topic-1", conversation_key="telegram:bot-1:room-1:topic-1", actor={"id": "u"},
        conversation={}, message={}, text="Hello",
    )
    try:
        with pytest.raises(PresenceTurnError) as refused:
            run_presence_turn(admission=SimpleNamespace(binding_id="b"), event=event, repo_dir=tmp_path,
                              drive_root=tmp_path, admitted=lease,
                              agent_factory=lambda **_kwargs: pytest.fail("must not run"))
        assert refused.value.code == "presence_admission_conversation_mismatch"
    finally:
        lease.release()


def test_same_source_id_with_different_room_or_text_never_joins_or_replays(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def runner(**kwargs):
        from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id
        from ouroboros.task_results import write_task_result

        calls.append(kwargs["event"].text)
        entered.set()
        assert release.wait(10)
        event = kwargs["event"]
        answer = _answer(**kwargs)
        write_task_result(tmp_path, presence_turn_task_id(binding, event.source_event_id), "completed",
                          metadata={"source": "presence", "presence": {"binding_id": binding},
                                    "presence_event_identity": presence_event_identity(binding, event),
                                    "presence_result_text": answer.text},
                          terminal_origin="model_final", result=answer.text)
        return answer

    app, binding, ctx = _presence_app(tmp_path, runner, account_wide=True)

    async def scenario():
        first = asyncio.create_task(_turn(app, binding, "same-id"))
        await _until(entered.is_set)
        other_room = await _turn(app, binding, "same-id", "room-2")
        different = _event("same-id")
        different["text"] = "changed message"
        other_text = await host_service._api_presence_turn(_request(app, {"binding_id": binding, "event": different}))
        for response in (other_room, other_text):
            body = json.loads(response.body)
            assert response.status_code == 409 and body["code"] == "presence_event_identity_conflict"
            assert body["disposition"] == "rejected" and not body.get("text")
        assert calls == ["Hello"]
        release.set()
        assert (await first).status_code == 200
        # The durable result is also bound, even though the original worker retired.
        again = await _turn(app, binding, "same-id", "room-2")
        assert again.status_code == 409 and json.loads(again.body)["code"] == "presence_event_identity_conflict"
        assert ctx.presence_turns.live() == []

    asyncio.run(scenario())


def test_many_slow_auth_probes_do_not_fill_the_default_executor(tmp_path, monkeypatch):
    app, _binding, ctx = _presence_app(tmp_path, _answer)
    entered, release = threading.Event(), threading.Event()
    lock, active, peak = threading.Lock(), [0], [0]
    original = ctx.authenticate_token_payload

    def slow_auth(token):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            if active[0] == 2:
                entered.set()
        try:
            assert release.wait(10)
            return original(token)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(ctx, "authenticate_token_payload", slow_auth)

    async def scenario():
        shared = ThreadPoolExecutor(max_workers=3, thread_name_prefix="auth-saturation")
        asyncio.get_running_loop().set_default_executor(shared)
        probes = [asyncio.create_task(host_service._api_identity(_request(app))) for _ in range(12)]
        try:
            await _until(entered.is_set)
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "admin-free"), 2) == "admin-free"
            assert peak[0] == 2 and not any(probe.done() for probe in probes)
        finally:
            release.set()
        assert all(response.status_code == 200 for response in await asyncio.wait_for(asyncio.gather(*probes), 10))
        shared.shutdown(wait=True)

    asyncio.run(scenario())


def test_slow_presence_preparation_cannot_fill_the_default_executor(tmp_path, monkeypatch):
    """Admission, staging and replay run before turn capacity; their workers still need a bound."""
    app, binding, ctx = _presence_app(tmp_path, _answer)
    entered, release = threading.Event(), threading.Event()
    lock, active, peak = threading.Lock(), [0], [0]
    original = host_service._admit_presence

    def slow_admission(*args):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            if active[0] == 2:
                entered.set()
        try:
            assert release.wait(10)
            return original(*args)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(host_service, "_admit_presence", slow_admission)

    async def scenario():
        shared = ThreadPoolExecutor(max_workers=3, thread_name_prefix="admission-saturation")
        asyncio.get_running_loop().set_default_executor(shared)
        requests = [asyncio.create_task(_turn(app, binding, f"admission-{i}")) for i in range(12)]
        try:
            await _until(entered.is_set)
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "owner-free"), 2) == "owner-free"
            assert peak[0] == 2 and not any(request.done() for request in requests)
        finally:
            release.set()
        assert all(response.status_code == 200 for response in await asyncio.wait_for(asyncio.gather(*requests), 10))
        assert ctx.presence_turns.live() == []
        shared.shutdown(wait=True)

    asyncio.run(scenario())
