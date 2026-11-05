"""Real async extension runners keep their invocation until physical unwind."""
import asyncio
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause, extension_loader, owner_pause, task_store
from ouroboros.tools import extension_dispatch
from tests._extension_loader_shared import _prepare_extension
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401 -- autouse isolation
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run



@contextmanager
def extension_case(tmp_path, monkeypatch, handler):
    loaded, _, root = _prepare_extension(
        tmp_path, "async_lifetime", permissions=["tool"], plugin_body=(
            "async def invoke(ctx, value):\n"
            "    return await ctx._async_fixture(value)\n"
            "def register(api):\n"
            "    api.register_tool('invoke', invoke, description='fixture',\n"
            "        schema={'type': 'object', 'properties': {'value': {'type': 'string'}}},\n"
            "        timeout_sec=1)\n"
        ),
    )
    registry, _, _ = _registry(root, monkeypatch)
    registry._ctx._async_fixture = handler
    registry._ctx.task_attempt = 3
    assert extension_loader.load_extension(
        loaded, lambda: {}, drive_root=root, _force_in_process=True) is None
    name = extension_loader.extension_surface_name(loaded.name, "invoke")
    assert registry.get_schema_by_name(name) is not None
    threads, writes = [], []

    class Runner(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

    # Keep actual threads; replace only this module's constructor for collection.
    monkeypatch.setattr(extension_dispatch, "threading", SimpleNamespace(Thread=Runner))
    patch = task_store.patch_handoff

    def recorded(*args, **kwargs):
        if args[2] == "launch_handoffs":
            writes.append(args[3])
        return patch(*args, **kwargs)

    monkeypatch.setattr(task_store, "patch_handoff", recorded)
    try:
        yield SimpleNamespace(registry=registry, root=root, name=name, threads=threads,
                              writes=writes, runner_type=Runner)
    finally:
        for thread in threads:
            threading.Thread.join(thread, timeout=5)
        extension_loader.unload_extension(loaded.name)
        assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("text,code", [
    ("complete async output", "OK"),
    ('{"ok": false, "error": "ordinary business failure"}', "TOOL_REPORTED_FAILURE"),
])
def test_joined_async_extension_preserves_result_without_eager_claim(tmp_path, monkeypatch, text, code):
    calls = []

    async def handler(value):
        await asyncio.sleep(0)  # real coroutine execution, not construction
        calls.append(value)
        return text

    with extension_case(tmp_path, monkeypatch, handler) as case:
        with owner(case.registry, attempt=3):
            result = run(case.registry, name=case.name, args={"value": "once"}, timeout=5)
            assert result["tool_result"].code == code
            assert result["result"] == text
            assert result["is_error"] is (code != "OK")
            assert case.writes == [] and not claims(case.root)
            assert budget_pause.drain_local_tool_futures(case.registry._ctx, timeout_sec=0)["drained"]
        assert calls == ["once"] and len(case.threads) == 1


def test_cancel_resistant_async_runner_retains_same_invocation_until_finally(tmp_path, monkeypatch):
    started = asyncio.Event()
    cancelled, physical_end = threading.Event(), threading.Event()
    live, effects, observed_waits = {}, [], []

    async def handler(value):
        live.update(loop=asyncio.get_running_loop(), release=asyncio.Event())
        started.set()
        try:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await live["release"].wait()  # a real task suppresses cancellation
            effects.append(value)
            return "late value cannot replace the timeout"
        finally:
            physical_end.set()

    async def expired(tasks, *, timeout):
        await started.wait()
        observed_waits.append(timeout)
        return set(), set(tasks)

    # Only the physical timeout observation is accelerated. The dispatcher still
    # cancels the real asyncio.Task and awaits its cancellation-resistant unwind.
    monkeypatch.setattr(extension_dispatch, "asyncio", SimpleNamespace(
        run=asyncio.run, wait=expired, gather=asyncio.gather))
    with extension_case(tmp_path, monkeypatch, handler) as case:
        def ended_join(_thread, timeout=None):
            assert cancelled.wait(5)

        monkeypatch.setattr(case.runner_type, "join", ended_join)
        try:
            with owner(case.registry, attempt=3) as held:
                result = run(case.registry, name=case.name, args={"value": "once"}, timeout=5)
                typed = result["tool_result"]
                assert (typed.status, typed.code) == ("timeout", "EXTENSION_TIMEOUT")
                assert result["is_error"] and "handler exceeded timeout" in result["result"]
                returned = result["result"]
                invocation = result["invocation_id"]
                assert observed_waits == [1] and cancelled.is_set() and not physical_end.is_set()
                assert len(case.threads) == 1 and case.threads[0].is_alive()
                with budget_pause._TOOL_LOCK:
                    key = budget_pause.tool_scope_key(case.registry._ctx)
                    row = budget_pause._TOOL_FUTURES[key][invocation]
                    assert row["future"].done() and row["holds"] == 1
                assert not budget_pause.drain_local_tool_futures(case.registry._ctx, timeout_sec=0)["drained"]
                retained = claims(case.root)
                assert list(retained) == [invocation] and case.writes == [invocation]
                assert retained[invocation]["state"] == "claimed"
                assert retained[invocation]["local_owner"]["task_attempt"] == 3
                owner_pause.install_fence(case.root, "root", request_id="while-async-unwinds")
                with pytest.raises(owner_pause.OwnerPauseRefused):
                    with owner_pause.launch_admission(case.registry._ctx):
                        pytest.fail("new work bypassed Pause while the old runner remained")
                assert claims(case.root) == retained and effects == []
            # Closing the logical attempt must not forget a real old-attempt tail.
            assert held.closed and not row["settled"].is_set()
            assert claims(case.root) == retained
        finally:
            if live:
                live["loop"].call_soon_threadsafe(live["release"].set)
        threading.Thread.join(case.threads[0], timeout=5)
        assert physical_end.is_set() and not case.threads[0].is_alive()
        assert row["settled"].is_set() and not claims(case.root)
        assert case.writes == [invocation, invocation] and effects == ["once"]
        assert result["result"] == returned and result["tool_result"].code == "EXTENSION_TIMEOUT"
        with budget_pause._TOOL_LOCK:
            assert key not in budget_pause._TOOL_FUTURES
