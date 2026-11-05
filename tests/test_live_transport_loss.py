"""Transport termination is distinct from a provider refusal; old update callers still answer."""
import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from ouroboros.contracts.live_conversation import ContextUpdate
from ouroboros.realtime_session import transport_loss


@pytest.mark.parametrize('code,reason,expected', [
    (1000, '', True), (1001, 'Going away', True), (1012, 'Service restart', True),
])
def test_close_codes_do_not_turn_a_provider_refusal_into_a_transport_retry(code, reason, expected):
    assert transport_loss(ConnectionClosedError(Close(code, reason), None)) is expected


def test_network_loss_without_a_close_frame_is_transport():
    assert transport_loss(ConnectionClosedError(None, None))
    assert transport_loss(ConnectionResetError())


def test_context_update_addition_preserves_old_callers():
    assert ContextUpdate(({'role': 'user', 'content': 'Old caller'},)).respond is True
    assert ContextUpdate((), respond=False).respond is False


def test_silent_catchup_capability_is_absent_by_default_for_old_providers():
    from ouroboros.contracts.live_conversation import LiveSession
    assert LiveSession.silent_context_updates is False
