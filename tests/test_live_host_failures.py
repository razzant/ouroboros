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


def test_an_engine_that_closes_its_socket_hands_its_own_words_to_the_owner(host_world, monkeypatch):
    """Gemini closed the stand's call with 1011 'Your prepayment credits are depleted' (27.10); the owner
    saw only a bare code. The close code and the engine's sentence are the ending's words."""
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    from ouroboros import call_peers
    host = host_world.host
    endings = []
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda *a, **kw: endings.append((a, kw)))
    async def closing_engine(port):
        raise ConnectionClosedError(Close(1011, 'Your prepayment credits are depleted. Please go to AI Studio.'), None)
    monkeypatch.setitem(host.turn.feature, 'handler', closing_engine)
    sentence, metadata, trace = host.turn.run(host.ctx, host.messages, host.tools, task=host.task, initial_effort='')
    assert metadata['reason_code'] == 'provider_closed' and metadata['execution_status'] == 'infra_failed'
    assert endings[0][1]['detail'] == '1011 Your prepayment credits are depleted. Please go to AI Studio.'
    assert sentence == 'Call ended: 1011 Your prepayment credits are depleted. Please go to AI Studio. Its transcript is in this room.'
    assert 'failure' not in metadata and 'failure' not in trace


def test_a_call_nobody_joined_is_cancelled_as_one_notice(host_world, monkeypatch):
    """A call whose client never attached (27.10: an admission 500) ended `peers_left` as a failure and left a
    Failed card, an empty call card and a transcript notice; nobody joining is one short notice, not a failure."""
    from ouroboros import call_peers
    host = host_world.host
    endings = []
    monkeypatch.setattr(call_peers, 'get_hub', lambda _: host.hub)
    monkeypatch.setattr(call_peers, 'close_hub', lambda *a, **kw: endings.append((a, kw)))
    async def nobody(port):
        raise RealtimeSetupError('not_connected', 'no one joined the call')
    monkeypatch.setitem(host.turn.feature, 'handler', nobody)
    sentence, metadata, trace = host.turn.run(host.ctx, host.messages, host.tools, task=host.task, initial_effort='')
    assert sentence == 'The call did not start: no one joined the call.'
    assert (metadata['reason_code'], metadata['execution_status']) == ('not_connected', 'cancelled')
    assert 'failure' not in metadata and endings[0][0][1] == 'not_connected'
