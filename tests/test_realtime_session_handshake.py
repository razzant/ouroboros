"""A refused websocket handshake reaches the owner as a typed reason, not as 'the call could not start'."""
import asyncio
from types import SimpleNamespace

import pytest

from ouroboros.contracts.realtime import RealtimeSetupError
from ouroboros.realtime_session import JsonSession


class Handshake(Exception):
    def __init__(self, status):
        super().__init__(status)
        if status is not None:
            self.status_code = status


class Connection:
    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc


def _session(exc):
    return JsonSession(SimpleNamespace(model="m"), None, connect=lambda _model: Connection(exc), observe=lambda *_: None)


@pytest.mark.parametrize("status, code", [(401, "authentication"), (403, "authentication"),
                                          (429, "rate_limit"), (503, "engine_unavailable")])
def test_handshake_status_codes_become_typed_refusals(status, code):
    with pytest.raises(RealtimeSetupError) as refused:
        asyncio.run(_session(Handshake(status)).connect_wire())
    assert refused.value.code == code and str(status) in str(refused.value)


def test_an_unexplained_handshake_failure_keeps_its_own_type():
    with pytest.raises(Handshake):
        asyncio.run(_session(Handshake(None)).connect_wire())
