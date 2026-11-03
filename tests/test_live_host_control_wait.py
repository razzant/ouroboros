"""Transient strict-control contention waits without admitting effects."""
import asyncio
import threading
from dataclasses import replace

import pytest

from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.store import StoreBusy, StoreError
from tests.test_live_conversation import host_world as host_world


def test_strict_control_contention_retries_before_tool_effect(host_world, monkeypatch):
    host = host_world.host
    owner = host.ctx.model_wait_context
    original = owner.control_reason
    probes, effects = [], []
    def probe(*, strict_current):
        probes.append(strict_current)
        assert not effects
        if len(probes) == 1:
            raise StoreBusy('fixture contention', reason='contention')
        return original(strict_current=strict_current)
    monkeypatch.setattr(owner, 'control_reason', probe)
    async def tool(port, arguments):
        effects.append('ran')
        return 'Language changed after a strict read.'
    host.call_tools['set_test_language'] = replace(host.call_tools['set_test_language'], handler=tool)
    async def run():
        rows = await host.tool_batch([{'id': 'after-contention', 'type': 'function',
            'function': {'name': 'set_test_language', 'arguments': '{}'}}])
        assert rows.silent == ('after-contention',)
        assert any(row.get('content') == 'Language changed after a strict read.' for row in rows)
    asyncio.run(run())
    assert len(probes) >= 2 and all(probes) and effects == ['ran']


def test_control_wait_keeps_typed_stops_budget_disable_and_other_errors(host_world, monkeypatch):
    from ouroboros import live_usage, usage_accounting
    host = host_world.host
    owner = host.ctx.model_wait_context
    for stop in ('owner_stop', 'deadline'):
        probes = []
        def probe(*, strict_current):
            probes.append(strict_current)
            if len(probes) == 1:
                raise StoreBusy('fixture contention', reason='contention')
            return stop
        monkeypatch.setattr(owner, 'control_reason', probe)
        with pytest.raises(RealtimeSetupError, match=stop):
            asyncio.run(host.check())
        assert probes == [True, True]
    def busy(*, strict_current):
        assert strict_current
        raise StoreBusy('fixture contention', reason='contention')
    monkeypatch.setattr(owner, 'control_reason', busy)
    def exhausted(**kwargs):
        raise usage_accounting.BudgetExceeded('fixture shared limit')
    with monkeypatch.context() as patch:
        patch.setattr(live_usage, 'check_live_budget', exhausted)
        with pytest.raises(RealtimeSetupError, match='budget_exhausted'):
            asyncio.run(host.check())
    async def disable():
        pending = asyncio.create_task(host.check())
        await asyncio.sleep(.01)
        host._end_reason = ('calls_disabled', 'Calls (beta) are off.')
        with pytest.raises(RealtimeSetupError, match='calls_disabled'):
            await pending
    asyncio.run(disable())
    host._end_reason = None
    def broken(*, strict_current):
        raise StoreError('fixture permanent store failure')
    monkeypatch.setattr(owner, 'control_reason', broken)
    with pytest.raises(StoreError, match='permanent store failure'):
        asyncio.run(host.check())


def test_cancelled_control_probe_joins_its_thread(host_world, monkeypatch):
    host = host_world.host
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    def probe(*, strict_current):
        entered.set()
        assert release.wait(5)
        exited.set()
        return None
    monkeypatch.setattr(host.ctx.model_wait_context, 'control_reason', probe)
    async def run():
        pending = asyncio.create_task(host.check())
        assert await asyncio.to_thread(entered.wait, 5)
        pending.cancel()
        await asyncio.sleep(.01)
        try:
            assert not pending.done(), 'cancel detached the still-running control read'
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert exited.is_set()
    asyncio.run(run())
