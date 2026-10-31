"""A received receipt settles its refused start, even after admission ends."""
import asyncio

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world
from tests.test_live_host_settlement import tariff
from tests.test_live_usage import token_usage


def test_received_done_settles_same_refused_attempt_before_call_finish(host_world):
    host = host_world.host
    host.quote = tariff(input_cost_per_token=.001, output_cost_per_token=.002)
    with ua.usage_scope(ua.UsageScope(drive_root=host.ctx.drive_root, task_id=host.ctx.task_id, global_limit_usd=0)):
        with pytest.raises(RealtimeSetupError, match='budget_exhausted'):
            host.response_started('already-received')
        pending, = ua.read_usage_records(host.ctx.drive_root)
        assert pending['state'] == 'dispatched'
        host.quote['entry']['input_cost_per_token'] = 100
        for _ in range(2):
            host.response_done('already-received', usage=token_usage(text_input_tokens=12),
                raw_usage={'input_tokens': 12}, status='completed')
        asyncio.run(host.finish())
        asyncio.run(host.finish())
    settled, = ua.read_usage_records(host.ctx.drive_root)
    assert settled['attempt_id'] == pending['attempt_id']
    assert settled['state'] == 'settled' and settled['cost_usd'] == pytest.approx(.012)
    assert settled['cost_evidence']['budget_refusal'] == 'budget_exhausted'
    assert host.responses == {} and 'already-received' in host.completed
