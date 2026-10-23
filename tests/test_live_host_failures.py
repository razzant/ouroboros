"""Call failures keep diagnostic evidence separate from the owner's sentence."""
import json

import pytest

from ouroboros.contracts.realtime import RealtimeSetupError
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('unexpected', [False, True])
def test_call_failure_preserves_sanitized_origin_and_stack(host_world, monkeypatch, unexpected):
    from ouroboros import call_peers
    host = host_world.host
    endings = []
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda *a, **kw: endings.append((a, kw)))
    async def failing_tool(port):
        if unexpected:
            raise RuntimeError('language tool failed: password="test-credential"')
        raise RealtimeSetupError('fixture_refusal', 'The fixture refused this call.')
    monkeypatch.setitem(host.turn.feature, 'handler', failing_tool)
    sentence, metadata, trace = host.turn.run(host.ctx, host.messages, host.tools,
                                             task=host.task, initial_effort='')
    events = [json.loads(json.loads(line)['raw']) for line in host.wire_path.read_text(encoding="utf-8").splitlines()]
    ended = next(event for event in events if event.get('type') == 'call.ended')
    if unexpected:
        assert metadata['reason_code'] == 'call_failed'
        failure = ended['failure']
        assert trace['failure'] == metadata['failure'] == failure
        assert failure['class'] == 'RuntimeError'
        assert 'failing_tool' in failure['origin']
        assert any(frame['function'] == 'failing_tool' for frame in failure['stack'])
        assert 'language tool failed' in failure['message']
        assert 'test-credential' not in json.dumps(failure)
        assert 'RuntimeError' not in sentence + endings[0][1]['detail']
        assert 'test-credential' not in sentence + endings[0][1]['detail']
    else:
        assert metadata['reason_code'] == 'fixture_refusal'
        assert 'failure' not in ended and 'failure' not in trace
        assert endings[0][1]['detail'] == 'The fixture refused this call.'
