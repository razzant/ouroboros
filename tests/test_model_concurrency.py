"""Shared capacity, live-ticket fairness and physical sender ownership; no providers."""
from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
import threading
import time

import pytest

from ouroboros import model_concurrency as mc

# Spawned processes race real file locks: run them outside xdist so host load cannot
# starve a child past its bounded wait.
pytestmark = pytest.mark.serial

ROUTE = {"provider": "test", "base_url": "https://provider.invalid/v1", "api_key": "fake-secret"}


@pytest.fixture(autouse=True)
def isolated_queue(monkeypatch, tmp_path):
    from ouroboros import config
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "30")
    mc.reset_for_tests()
    yield tmp_path
    mc.reset_for_tests()


def _records(root, suffix):
    return list((root / "state" / "model_concurrency").glob(f"*/*.{suffix}"))


def _until(predicate, timeout=30):
    until = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < until, "bounded condition did not become true"
        time.sleep(0.01)


def _worker(root, cap, name, entered, release, cancel=None, counts=None):
    from ouroboros import config
    config.SETTINGS_PATH = root / "settings.json"
    os.environ["OUROBOROS_DATA_DIR"] = str(root)
    os.environ["OUROBOROS_MODEL_MAX_CONCURRENCY"] = str(cap)
    os.environ["OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC"] = "60"
    try:
        with mc.model_call_slot("model", route=ROUTE, cancel_event=cancel):
            if counts is not None:
                with counts.get_lock():
                    counts[0] += 1
                    counts[1] = max(counts[0], counts[1])
            entered.put((name, "entered"))
            assert release.wait(60), "parent did not release owned child"
            if counts is not None:
                with counts.get_lock():
                    counts[0] -= 1
    except mc.ModelSlotCancelled:
        entered.put((name, "cancelled"))


@pytest.fixture
def processes():
    ctx = multiprocessing.get_context("spawn")
    owned = []

    def start(*args):
        process = ctx.Process(target=_worker, args=args)
        owned.append(process)
        process.start()
        return process

    yield ctx, start
    for process in owned:
        process.join(0.2)
        if process.is_alive():
            process.kill()
        process.join(5)
        assert not process.is_alive()
        process.close()


def test_processes_share_default_three_slots(isolated_queue, processes):
    ctx, start = processes
    entered, release = ctx.Queue(), ctx.Event()
    counts = ctx.Array("i", [0, 0])
    children = [start(isolated_queue, 3, i, entered, release, None, counts) for i in range(6)]
    first = [entered.get(timeout=30) for _ in range(3)]
    assert all(state == "entered" for _, state in first)
    _until(lambda: len(_records(isolated_queue, "ticket")) == 3)
    with pytest.raises(queue.Empty):
        entered.get(timeout=0.15)
    assert len(_records(isolated_queue, "slot")) == 3
    release.set()
    assert all(entered.get(timeout=30)[1] == "entered" for _ in range(3))
    for child in children:
        child.join(5)
        assert child.exitcode == 0
    assert counts[:] == [0, 3]
    assert not _records(isolated_queue, "ticket") + _records(isolated_queue, "slot")


def test_fifo_across_processes_and_dead_waiter(isolated_queue, processes):
    ctx, start = processes
    entered = ctx.Queue()
    releases = [ctx.Event() for _ in range(5)]
    holder = start(isolated_queue, 1, 0, entered, releases[0])
    assert entered.get(timeout=30) == (0, "entered")
    waiting = []
    for index in range(1, 5):
        waiting.append(start(isolated_queue, 1, index, entered, releases[index]))
        _until(lambda: len(_records(isolated_queue, "ticket")) == index)
    waiting[1].kill()
    waiting[1].join(5)
    releases[0].set()
    for index in [1, 3, 4]:
        assert entered.get(timeout=30) == (index, "entered")
        releases[index].set()
    holder.join(5)
    for child in (waiting[0], waiting[2], waiting[3]):
        child.join(5)
        assert child.exitcode == 0
    assert not _records(isolated_queue, "ticket") + _records(isolated_queue, "slot")


def test_killed_holder_reclaimed_but_live_holder_never_expires(isolated_queue, processes):
    ctx, start = processes
    entered, release = ctx.Queue(), ctx.Event()
    # A killed process may leave a multiprocessing.Condition wait unacknowledged;
    # do not later notify its test-only Event, shared with no surviving child.
    holder_release = ctx.Event()
    holder = start(isolated_queue, 1, "holder", entered, holder_release)
    assert entered.get(timeout=30) == ("holder", "entered")
    slot = _records(isolated_queue, "slot")[0]
    os.utime(slot, (1, 1))
    waiter = start(isolated_queue, 1, "waiter", entered, release)
    _until(lambda: len(_records(isolated_queue, "ticket")) == 1)
    with pytest.raises(queue.Empty):
        entered.get(timeout=0.15)
    holder.kill()
    holder.join(5)
    assert entered.get(timeout=30) == ("waiter", "entered")
    assert len(_records(isolated_queue, "slot")) == 1
    release.set()
    waiter.join(5)
    assert waiter.exitcode == 0
    assert not _records(isolated_queue, "slot")


def test_waiter_cancellation_releases_its_ticket(isolated_queue, processes):
    ctx, start = processes
    entered, release, cancel = ctx.Queue(), ctx.Event(), ctx.Event()
    holder = start(isolated_queue, 1, "holder", entered, release)
    assert entered.get(timeout=30) == ("holder", "entered")
    waiter = start(isolated_queue, 1, "waiter", entered, release, cancel)
    _until(lambda: len(_records(isolated_queue, "ticket")) == 1)
    cancel.set()
    assert entered.get(timeout=30) == ("waiter", "cancelled")
    waiter.join(5)
    assert waiter.exitcode == 0
    assert not _records(isolated_queue, "ticket")
    assert len(_records(isolated_queue, "slot")) == 1
    release.set()
    holder.join(5)
    with mc.model_call_slot("model", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1


@pytest.mark.parametrize("deadline", ["expired", "wait_ceiling"])
def test_timeout_never_runs_body_and_retry_can_enter(monkeypatch, isolated_queue, deadline):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    with mc.model_call_slot("model", route=ROUTE):
        monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "1")
        bound = time.time() - 1 if deadline == "expired" else None
        with pytest.raises(mc.ModelSlotTimeout):
            with mc.model_call_slot("model", deadline_ts=bound, route=ROUTE):
                pytest.fail("timeout bypassed capacity")
        assert not _records(isolated_queue, "ticket")
        assert len(_records(isolated_queue, "slot")) == 1
    with mc.model_call_slot("model", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1


def test_slot_released_on_exception(isolated_queue, monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    with pytest.raises(ValueError):
        with mc.model_call_slot("model", route=ROUTE):
            raise ValueError("fake provider failure")
    with mc.model_call_slot("model", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1
    assert not _records(isolated_queue, "slot")


def test_disabled_is_passthrough(monkeypatch, isolated_queue):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "0")
    assert not mc.enabled()
    with mc.model_call_slot("model", route=ROUTE, deadline_ts=0):
        assert not (isolated_queue / "state").exists()
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    assert mc.enabled()
    with mc.model_call_slot("model", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1


def test_one_model_shares_its_route_and_every_other_identity_separates(monkeypatch, isolated_queue):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "1")
    with mc.model_call_slot("main", route=ROUTE):
        with pytest.raises(mc.ModelSlotTimeout):
            with mc.model_call_slot("main", route={**ROUTE, "base_url": ROUTE["base_url"] + "/"}):
                pytest.fail("the same model on the same route escaped by an endpoint spelling")
        for model, key, value in [("checker", "provider", "test"), ("main", "api_key", "other-fake"),
                                  ("main", "base_url", "https://other.invalid/v1"), ("main", "provider", "other")]:
            with mc.model_call_slot(model, route={**ROUTE, key: value}):
                assert len(_records(isolated_queue, "slot")) == 2
    for path in (isolated_queue / "state").rglob("*"):
        assert "fake-secret" not in str(path)
        if path.is_file():
            assert b"fake-secret" not in path.read_bytes()


def test_managed_engine_routes_are_not_queued_by_the_host(monkeypatch, isolated_queue):
    """Claudexor's one daemon arbitrates its accounts: the host cap never shrinks that parallel work."""
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC", "1")
    managed = {"provider": "claudexor", "source": "managed", "credential_profile_id": ""}
    with mc.model_call_slot("main", route=managed):
        with mc.model_call_slot("main", route=managed):
            assert _records(isolated_queue, "slot") == []
    # A direct route at the same cap still queues (the guard is not off).
    with mc.model_call_slot("main", route=ROUTE):
        with pytest.raises(mc.ModelSlotTimeout):
            with mc.model_call_slot("main", route=ROUTE):
                pytest.fail("a direct route escaped its cap")


def test_unreadable_settings_fall_back_to_the_process_cap(monkeypatch, isolated_queue):
    from ouroboros import settings_integrity
    monkeypatch.setattr(settings_integrity, "read_settings_json_verified",
                        lambda path: (_ for _ in ()).throw(ValueError("unverified")))
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "2")
    assert mc._cap() == 2
    with mc.model_call_slot("main", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1


def test_cap_changes_drain_existing_holders_and_wake_old_waiters(isolated_queue, processes):
    from ouroboros.utils import atomic_write_json
    ctx, start = processes
    entered = ctx.Queue()
    releases = [ctx.Event() for _ in range(4)]
    children = [start(isolated_queue, 3, i, entered, releases[i]) for i in range(3)]
    assert {entered.get(timeout=30)[0] for _ in range(3)} == {0, 1, 2}
    settings = isolated_queue / "settings.json"
    atomic_write_json(settings, {"OUROBOROS_MODEL_MAX_CONCURRENCY": 1})
    # This process inherited cap=3, so only the shared setting can narrow it.
    children.append(start(isolated_queue, 3, 3, entered, releases[3]))
    _until(lambda: len(_records(isolated_queue, "ticket")) == 1)
    releases[0].set()
    children[0].join(5)
    releases[1].set()
    children[1].join(5)
    with pytest.raises(queue.Empty):
        entered.get(timeout=0.15)
    atomic_write_json(settings, {"OUROBOROS_MODEL_MAX_CONCURRENCY": 2})
    assert entered.get(timeout=30) == (3, "entered")
    releases[2].set()
    releases[3].set()
    for child in children:
        child.join(5)
        assert child.exitcode == 0


def test_threads_keep_fifo_and_capacity(monkeypatch, isolated_queue):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    order = []
    errors = []

    def run(number):
        try:
            with mc.model_call_slot("model", route=ROUTE):
                order.append(number)
        except BaseException as error:
            errors.append(error)

    threads = []
    with mc.model_call_slot("model", route=ROUTE):
        for number in range(4):
            thread = threading.Thread(target=run, args=(number,))
            threads.append(thread)
            thread.start()
            _until(lambda: len(_records(isolated_queue, "ticket")) == number + 1)
        assert not order
    for thread in threads:
        thread.join(35)
        assert not thread.is_alive()
    assert not errors
    assert order == [0, 1, 2, 3]


def test_a_turn_the_owner_awaits_goes_ahead_of_background_calls_not_yet_started(monkeypatch, isolated_queue):
    """A call of a turn the owner awaits takes the next free slot before background calls still waiting; the call
    holding the slot keeps it, and awaited and background calls each keep their arrival order."""
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    monkeypatch.setattr(mc, "_owner_awaits", lambda: threading.current_thread().name.startswith("owner"))
    order, errors = [], []

    def run():
        try:
            with mc.model_call_slot("model", route=ROUTE):
                order.append(threading.current_thread().name)
        except BaseException as error:
            errors.append(error)

    threads = []
    with mc.model_call_slot("model", route=ROUTE):  # a background call already holds the one slot
        for count, name in enumerate(("background-1", "owner-1", "background-2", "owner-2"), start=1):
            thread = threading.Thread(target=run, name=name)
            threads.append(thread)
            thread.start()
            _until(lambda: len(_records(isolated_queue, "ticket")) == count)
        assert not order
    for thread in threads:
        thread.join(35)
        assert not thread.is_alive()
    assert not errors
    assert order == ["owner-1", "owner-2", "background-1", "background-2"]


def test_only_a_direct_turn_the_owner_awaits_counts_as_awaited():
    """The queue's predicate is the turn receipt's: a direct turn the owner's door stamped; a wake, a Presence
    correspondent's turn and a promoted root task the owner also awaits all queue as background calls."""
    from types import SimpleNamespace

    from ouroboros import model_wait

    ref = {"chat_id": 1, "client_message_id": "c1", "ts": "2026-10-26T12:00:00+00:00"}
    cases = [({"_is_direct_chat": True, "metadata": {"origin_message_ref": ref}}, True),
             ({"_is_direct_chat": True, "metadata": {"source": "consciousness"}}, False),
             ({"_is_direct_chat": True, "metadata": {"origin_message_ref": ref, "presence": {"room": "r"}}}, False),
             ({"type": "task", "metadata": {"origin_message_ref": ref}}, False)]
    for task, awaited in cases:
        token = model_wait._CURRENT.set(SimpleNamespace(task=task))
        try:
            assert model_wait.owner_awaits_current_call() is awaited, task
            assert mc._owner_awaits() is awaited
        finally:
            model_wait._CURRENT.reset(token)
    assert model_wait.owner_awaits_current_call() is False  # no bound task: a background call


def test_the_same_turn_s_work_after_its_answer_queues_as_background(tmp_path):
    """After the answer, the turn's post-task worker (synthesis, memory drafts, scratchpad consolidation) keeps the
    turn's wait owner but no longer prepares a reply: a new owner turn must not wait behind it."""
    from types import SimpleNamespace

    from ouroboros import model_wait
    from ouroboros.usage_accounting import UsageScope, usage_scope

    ref = {"chat_id": 1, "client_message_id": "c1", "ts": "2026-10-26T12:00:00+00:00"}
    token = model_wait._CURRENT.set(SimpleNamespace(task={"_is_direct_chat": True, "metadata": {"origin_message_ref": ref}}))
    try:
        for category, awaited in (("task", True), ("post_task", False), ("consolidation", False), ("web_search", True)):
            with usage_scope(UsageScope(drive_root=tmp_path, category=category, source="probe")):
                assert model_wait.owner_awaits_current_call() is awaited, category
    finally:
        model_wait._CURRENT.reset(token)


def test_async_cancellation_has_no_orphan_ticket(monkeypatch, isolated_queue):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")

    async def run():
        async def wait():
            async with mc.model_call_slot_async("model", route=ROUTE):
                pytest.fail("cancelled waiter sent")
        with mc.model_call_slot("model", route=ROUTE):
            task = asyncio.create_task(wait())
            for _ in range(100):
                if _records(isolated_queue, "ticket"):
                    break
                await asyncio.sleep(0.01)
            assert _records(isolated_queue, "ticket")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not _records(isolated_queue, "ticket")
        async with mc.model_call_slot_async("model", route=ROUTE):
            assert len(_records(isolated_queue, "slot")) == 1
    asyncio.run(run())


def test_scope_and_backoff_hold_no_slot_and_each_physical_attempt_does(monkeypatch, isolated_queue):
    monkeypatch.setenv("OUROBOROS_MODEL_MAX_CONCURRENCY", "1")
    from ouroboros import llm_attempt
    monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", lambda: None)
    observed = []

    def send():
        observed.append(len(_records(isolated_queue, "slot")))
        if len(observed) == 1:
            raise ValueError("fake 429")
        return "response"

    with mc.model_call_scope("model", deadline_ts=time.time() + 3):
        assert not _records(isolated_queue, "slot")
        wrapped = mc.limit_physical_send(ROUTE, send)
        with pytest.raises(ValueError):
            wrapped()
        assert not _records(isolated_queue, "slot")  # retry/backoff/nested check
        assert wrapped() == "response"
    assert observed == [1, 1]
    assert not _records(isolated_queue, "slot")


def test_queue_lock_failure_refuses_but_recovery_runs(monkeypatch, isolated_queue):
    acquire = mc.platform_layer.acquire_exclusive_file_lock
    monkeypatch.setattr(mc.platform_layer, "acquire_exclusive_file_lock", lambda *a, **k: None)
    with pytest.raises(mc.ModelSlotUnavailable):
        with mc.model_call_slot("model", route=ROUTE):
            pytest.fail("unlocked queue sent")
    monkeypatch.setattr(mc.platform_layer, "acquire_exclusive_file_lock", acquire)
    with mc.model_call_slot("model", route=ROUTE):
        assert len(_records(isolated_queue, "slot")) == 1
