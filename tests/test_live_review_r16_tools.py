"""R16: argument failures and unavailable pixels are facts, not call teardown."""
import asyncio
import json

import pytest

from ouroboros.contracts.realtime import CatalogModel
from skills.live_conversation.attention import Attention, attend_tool
from tests._live_world import host_world as host_world_fixture

host_world = host_world_fixture


@pytest.mark.parametrize('arguments', [{'see': 'invalid'}, {'look': 'yes'}, {'revisit_after_s': -1}])
def test_bad_attend_is_a_tool_row_and_the_next_valid_call_still_runs(host_world, arguments):
    async def scenario():
        host = host_world.host
        host.call_tools['attend'] = attend_tool()
        host.state['attention'] = Attention(host)
        for cid, args in [('bad', arguments), ('good', {'see': 'asked'})]:
            rows = await host.tool_batch([{'id': cid, 'type': 'function', 'function': {
                'name': 'attend', 'arguments': json.dumps(args)}}])
            text = next(r['content'] for r in rows if r.get('tool_call_id') == cid)
            assert rows.silent == (cid,)
            if cid == 'bad':
                assert text.startswith('Call tool argument error: ')
            else:
                assert json.loads(text) == {'mode': 'asked', 'changed': True}
            assert not host.hub.closed
    asyncio.run(scenario())


def test_a_default_thinking_level_must_be_one_of_the_models_levels():
    from ouroboros.contracts.realtime import RealtimeSetupError
    named = CatalogModel('m', 'M', thinking_levels=('minimal', 'low'), honours=frozenset({'thinking'}), thinking_default='low')
    assert named.thinking_default == 'low'
    with pytest.raises(RealtimeSetupError, match='default thinking level'):
        CatalogModel('m', 'M', thinking_levels=('minimal', 'low'), honours=frozenset({'thinking'}), thinking_default='high')
