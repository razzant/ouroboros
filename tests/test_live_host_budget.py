"""Idle connection time obeys the same spending ceiling as model responses."""
import asyncio
import json

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.contracts.live_conversation import LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world
from tests.test_live_host_settlement import tariff


def test_idle_budget_check_keeps_closing_tail_and_duplicate_samples(host_world):
    host = host_world.host
    host.quote = tariff(input_cost_per_second=.01)
    def sample(session, seconds):
        session.observe('usage', json.dumps({'usage': {'connected_seconds': seconds}, 'raw_usage': {}}))
    async def consume(session):
        async def close():
            sample(session, 9)
            sample(session, 9)
            session.closed = True
        session.close = close
        sample(session, 4)
        await host.check()  # still below the shared limit
        sample(session, 7)
        sample(session, 7)
        await host.check()  # no response_started event ever arrives
    async def run():
        with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, global_limit_usd=.05)):
            with pytest.raises(RealtimeSetupError, match='budget_exhausted: The shared spending limit has been reached.'):
                await host.session(consume, LiveOptions())
    asyncio.run(run())
    rows = ua.read_usage_records(host.ctx.drive_root)
    assert [row['cost_usd'] for row in rows] == pytest.approx([.04, .03, .02])
    assert all(row['source'] == 'live_connection_usage' for row in rows)
    assert host_world.sessions[-1].closed and not host.responses
