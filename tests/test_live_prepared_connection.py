"""Per-connection custody beneath the shared prepare-ahead driver."""
import asyncio
import json

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.contracts.live_conversation import LiveHost, LiveOptions
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world
from tests.test_live_host_settlement import tariff


def test_old_host_port_default_keeps_sequential_sessions():
    assert asyncio.run(LiveHost.prepare_session(object(), LiveOptions())) is None
    assert LiveHost.tool_results_delivered(object(), ()) is None


@pytest.mark.parametrize('close_error', [False, True])
def test_retired_connection_settles_its_quote_and_journal_after_switch(host_world, close_error):
    host = host_world.host
    host.quote = tariff(input_cost_per_second=.01)
    async def run():
        entered, retire, close_entered, close_release = (asyncio.Event() for _ in range(4))
        def sample(session, seconds):
            session.observe('usage', json.dumps({'usage': {'connected_seconds': seconds}, 'raw_usage': {}}))
        async def old(session):
            sample(session, 4)
            async def close():
                close_entered.set()
                await close_release.wait()
                sample(session, 9)
                session.closed = True
                if close_error:
                    raise RuntimeError('Synthetic retired close error')
            session.close = close
            entered.set()
            await retire.wait()
        with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id)):
            first = asyncio.create_task(host.session(old, LiveOptions()))
            await entered.wait()
            first_physical = host._physical_context
            host.quote = tariff(input_cost_per_second=.02)
            next_connection = await host.prepare_session(LiveOptions())
            assert host.connection_number == 1 and host._physical_context is first_physical
            assert host._connection_quote['entry']['input_cost_per_second'] == .01
            retire.set()
            await first
            await close_entered.wait()
            async def new(session):
                assert host.connection_number == 2
                sample(session, 3)
                close_release.set()
                if close_error:
                    with pytest.raises(RuntimeError, match='Synthetic retired close error'):
                        await host._connection_tasks[0]
                else:
                    await host._connection_tasks[0]
                assert host._connection_accounting and host._connection_quote['entry']['input_cost_per_second'] == .02
            await next_connection.run(new)
            await next_connection.task
            await host.finish()
        rows = ua.read_usage_records(host.ctx.drive_root)
        assert [r['cost_usd'] for r in rows] == pytest.approx([.04, .06, .05])
        host.wire.flush()
        samples = [(r['connection'], json.loads(r['raw'])['usage']['connected_seconds'])
                   for r in map(json.loads, host.wire_path.read_text().splitlines()) if r['direction'] == 'usage']
        assert samples == [(1, 4), (2, 3), (1, 9)]
        assert ('cleanup_failure' in host.timing['connections'][0]) is close_error
    asyncio.run(run())


def test_changed_canonical_prefix_rejects_prepared_projection_without_overwriting_it(host_world):
    host = host_world.host
    async def run():
        lease = await host.prepare_session(LiveOptions())
        host.messages[0]['content'] += '\nExact new owner-authored prefix.'
        expected = host.messages[0]['content']
        with pytest.raises(RealtimeSetupError, match='prepared_stale'):
            await lease.run(lambda session: asyncio.sleep(0))
        await lease.close()
        assert host.messages[0]['content'] == expected
        assert host_world.sessions[-1].closed
    asyncio.run(run())
