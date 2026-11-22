"""Physical-attempt attribution for local capability checks."""

from __future__ import annotations

import sys
import types
import time
from tests._usage_store_testing import ledger_rows


def _queue_records(root, suffix):
    return list((root / "state" / "model_concurrency").glob(f"*/*.{suffix}"))


def _queue_slots(root, route):
    """How many slots the built-in queue this route names holds, at any moment inside a probe."""
    from ouroboros import model_concurrency as mc

    path = root / "state" / "model_concurrency" / mc._route_digest("local-model", True, route)
    return len(list(path.glob("*.slot"))) if path.exists() else 0


def test_local_capability_probes_take_the_builtin_servers_slot_and_keep_its_count_out(monkeypatch, tmp_path):
    """The Settings/onboarding Test's two completions are generations of the built-in model, not a side channel
    beside its queue. With no number chosen each probe holds the one slot the server computes with while it sends
    — the server's own count declines there instead of opening a connection beside it — and both leave no slot or
    ticket behind. An owner's number keeps the queue at that number and the count outside it."""
    from ouroboros import config, llm_attempt
    from ouroboros import model_concurrency as mc
    from ouroboros.local_model import LocalModelManager
    from ouroboros.utils import atomic_write_json

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "10")
    monkeypatch.delenv("OUROBOROS_MODEL_MAX_CONCURRENCY", raising=False)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", lambda: None)
    mc.reset_for_tests()
    held, counted = [], []

    class Completions:
        def create(self, **payload):
            held.append(len(_queue_records(tmp_path, "slot")))
            with mc.measurement_slot(wait=False) as admitted:  # the server's own count, now
                counted.append(admitted)
            if payload.get("tools"):
                message = types.SimpleNamespace(content="", tool_calls=[object()])
            else:
                message = types.SimpleNamespace(content="hello", tool_calls=[])
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)],
                usage=types.SimpleNamespace(completion_tokens=1),
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    manager = LocalModelManager()

    atomic_write_json(tmp_path / "settings.json", {})  # no number chosen: the server's one slot
    assert manager.test_tool_calling()["success"] is True
    assert held == [1, 1], "each probe holds the server's one slot while it sends"
    assert counted == [False, False], "the server's own count never opens beside a probe"
    assert not _queue_records(tmp_path, "slot") and not _queue_records(tmp_path, "ticket")

    held.clear()
    counted.clear()
    atomic_write_json(tmp_path / "settings.json",
                      {"OUROBOROS_MODEL_MAX_CONCURRENCY": 2, "_model_cap_format": 1})
    assert manager.test_tool_calling()["success"] is True
    assert held == [1, 1], "the owner's number is the queue's cap, one probe at a time"
    assert counted == [True, True], "an owner's number keeps the count outside the queue"
    assert not _queue_records(tmp_path, "slot") and not _queue_records(tmp_path, "ticket")
    mc.reset_for_tests()


def test_local_model_capability_checks_are_accounted_and_inherit_task_scope(
    monkeypatch,
    tmp_path,
):
    from ouroboros.local_model import LocalModelManager
    from ouroboros.usage_accounting import UsageScope, current_usage_scope, usage_scope

    provider_calls = []

    class Completions:
        def create(self, **payload):
            provider_calls.append(payload)
            if payload.get("tools"):
                message = types.SimpleNamespace(content="", tool_calls=[object()])
            else:
                message = types.SimpleNamespace(content="hello", tool_calls=[])
            usage = types.SimpleNamespace(completion_tokens=1)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)],
                usage=usage,
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "10")

    manager = LocalModelManager()
    system_result = manager.test_tool_calling()

    assert system_result["success"] is True
    assert len(provider_calls) == 2

    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, 'local-probe-root', 'running', billing_group={
        'billing_group_id': 'local-probe-root', 'billing_group_limit_usd': 2.0,
        'billing_group_limit_source': 'initial_task_admission', 'billing_group_limit_revision': 'fixture'})
    task_scope = UsageScope(
        drive_root=tmp_path,
        task_id="local-probe-task",
        root_task_id="local-probe-root",
        parent_task_id="local-probe-parent",
        category="task",
        source="main_loop",
        global_limit_usd=11.0,
        root_limit_usd=2.0,
    )
    provider_calls.clear()
    with usage_scope(task_scope):
        task_result = manager.test_tool_calling()

    assert task_result["success"] is True
    assert len(provider_calls) == 2
    assert current_usage_scope() is None

    rows = ledger_rows(tmp_path)
    settled = [row for row in rows if row.get("state") == "settled"]
    assert len(settled) == 4
    for row in settled[:2]:
        assert row["task_id"] == "system:capability_probe"
        assert row["root_task_id"] == "system:capability_probe"
        assert row["category"] == "capability_probe"
        assert row["source"] == "capability_probe.local_model"
        assert row["cost_usd"] == 0.0
        assert row["cost_final"] is True
    for row in settled[2:]:
        assert row["task_id"] == "local-probe-task"
        assert row["root_task_id"] == "local-probe-root"
        assert row["parent_task_id"] == "local-probe-parent"
        assert row["category"] == "task"
        assert row["source"] == "capability_probe.local_model"


def test_the_settings_probe_queues_at_the_endpoint_its_own_client_reaches(monkeypatch, tmp_path):
    """A server started on another port is another server. The Test's slot belongs to the endpoint its client
    sends to (``start_server``'s own port), so the probe neither joins the configured server's queue nor blocks
    it: while this server's one slot is held, that other queue's generation and its own count stay free.

    The probe reads that port once, so the address its own SDK calls reach and the queue it holds cannot name
    different servers: a host that moves the port between the two reads leaves the probe on the endpoint it
    sends to, and a probe that held the moved queue fails here — at once, never by waiting on it."""
    from ouroboros import config, llm_attempt
    from ouroboros import model_concurrency as mc
    from ouroboros.local_model import LocalModelManager
    from ouroboros.utils import atomic_write_json

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "10")
    monkeypatch.setenv("LOCAL_MODEL_PORT", "9101")  # the lane's configured server: a different process
    monkeypatch.delenv("OUROBOROS_MODEL_MAX_CONCURRENCY", raising=False)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(llm_attempt, "require_physical_dispatch_window", lambda: None)
    mc.reset_for_tests()
    atomic_write_json(tmp_path / "settings.json", {})  # no number chosen: each server computes one at a time
    own, moved = 9102, 9103  # the running server this probe reaches, and the port a host moves it to
    seen, addresses, keys, own_count, other_count, moved_held, other_entered = [], [], [], [], [], [], []
    sdk: dict = {}

    class Completions:
        def create(self, **payload):
            addresses.append(sdk.get("base_url"))  # the address of the client that actually sends this
            keys.append(sdk.get("api_key"))
            seen.append(_queue_slots(tmp_path, mc.built_in_route(own)))
            moved_held.append(_queue_slots(tmp_path, mc.built_in_route(moved)))
            with mc.measurement_slot(mc.built_in_route(own), wait=False) as admitted:
                own_count.append(admitted)
            with mc.measurement_slot(mc.built_in_route(), wait=False) as admitted:
                other_count.append(admitted)
            with mc.model_call_slot("local-model", True, route=mc.built_in_route(),
                                    deadline_ts=time.time() + 2.0):
                # Bounded: the probe took no slot of that other server's queue, so this enters; a probe that
                # held that queue instead ends here as ModelSlotTimeout, never as an unbounded wait.
                other_entered.append("entered")
            if payload.get("tools"):
                message = types.SimpleNamespace(content="", tool_calls=[object()])
            else:
                message = types.SimpleNamespace(content="hello", tool_calls=[])
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)],
                usage=types.SimpleNamespace(completion_tokens=1),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            sdk.update(kwargs)
            self.chat = types.SimpleNamespace(completions=Completions())

    class MovedPortManager(LocalModelManager):
        """The host rewrites the port, as a restart may between a probe's address and its queue."""

        def __init__(self):
            self.port_reads = []
            super().__init__()

        @property
        def _port(self):
            self.port_reads.append(own if not self.port_reads else moved)
            return self.port_reads[-1]

        @_port.setter
        def _port(self, _value):
            pass

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    manager = MovedPortManager()
    assert manager.test_tool_calling()["success"] is True

    assert addresses == [f"http://127.0.0.1:{own}/v1"] * 2, "the SDK sends to the endpoint the probe read first"
    assert keys == ["local"] * 2, "the probe's client carries the built-in server's own key"
    assert manager.port_reads == [own], "one read: the address and the queue cannot name different endpoints"
    assert seen == [1, 1], "each probe holds the slot of the endpoint it sends to"
    assert moved_held == [0, 0], "the port the host moved to holds no queue of this probe"
    assert own_count == [False, False], "that server's own count declines while its one slot is held"
    assert other_count == [True, True], "the configured server's queue is another one, and it is free"
    assert other_entered == ["entered", "entered"]
    for route in (mc.built_in_route(own), mc.built_in_route(moved), mc.built_in_route()):
        assert _queue_slots(tmp_path, route) == 0, "a probe leaves no slot behind"
    assert not _queue_records(tmp_path, "slot") and not _queue_records(tmp_path, "ticket")
    mc.reset_for_tests()


def test_the_settings_test_handler_keeps_the_loop_and_its_attempt_alive(monkeypatch):
    """The handler's synchronous probe runs off the event loop: another coroutine progresses while the probe
    waits, its own outcome reaches the response, and a client that disconnects leaves the owned attempt to
    settle rather than abandoning it."""
    import asyncio
    import json
    import threading
    from ouroboros.gateway import models as gateway_models

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    outcome = {"success": True, "generated": True}

    class Manager:
        is_running = True

        def test_tool_calling(self):
            entered.set()
            release.wait(5)
            finished.set()
            return outcome

    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: Manager())

    async def served_client():
        handler = asyncio.create_task(gateway_models.api_local_model_test(object()))
        for _ in range(400):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set(), "the probe never started"
        progressed = []

        async def neighbour():
            await asyncio.sleep(0.01)
            progressed.append("ran")

        await asyncio.wait_for(neighbour(), 5)  # only a free loop finishes this while the probe waits
        assert progressed == ["ran"] and not handler.done(), "the handler's wait must not hold the loop"
        release.set()
        return await asyncio.wait_for(handler, 5)

    response = asyncio.run(served_client())
    assert json.loads(response.body) == outcome, "the probe's own outcome reaches the response"
    assert finished.is_set()

    entered.clear()
    release.clear()
    finished.clear()

    async def cancelled_client():
        handler = asyncio.create_task(gateway_models.api_local_model_test(object()))
        for _ in range(400):
            if entered.is_set():
                break
            await asyncio.sleep(0.005)
        assert entered.is_set(), "the probe never started"
        handler.cancel()
        for _ in range(50):  # bounded: an unshielded asyncio.to_thread settles the handler here at once
            if handler.done():
                break
            await asyncio.sleep(0.002)
        assert not handler.done(), ("the handler must stay owed until its attempt settles; an unshielded "
                                    "asyncio.to_thread would already be done here, its thread drained by the caller")
        assert not finished.is_set(), "the attempt must still be owed when its client goes away"
        release.set()
        try:
            await handler
        except asyncio.CancelledError:
            return "cancelled"
        return "finished"

    assert asyncio.run(cancelled_client()) == "cancelled"
    assert finished.is_set(), "the owned attempt settled instead of being abandoned"
