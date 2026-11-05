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


def _refused_handshake(shape, status):
    """The refusal as each websockets generation raises it, its headers and reason carrying a credential canary."""
    if shape == "attribute":
        return Handshake(status)
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus, InvalidStatusCode
    from websockets.http11 import Response

    headers = Headers({"Authorization": "synthetic-secret-canary"})
    if shape == "legacy":
        return InvalidStatusCode(status, headers)
    return InvalidStatus(Response(status, "synthetic-secret-canary", headers))  # the current shape: .response


@pytest.mark.parametrize("shape, status, code", [
    ("attribute", 401, "authentication"), ("attribute", 403, "authentication"), ("attribute", 429, "rate_limit"),
    ("response", 401, "authentication"), ("legacy", 403, "authentication"), ("response", 429, "rate_limit"),
    ("legacy", 500, "engine_unavailable"), ("response", 503, "engine_unavailable"),
    ("response", 599, "engine_unavailable"),
])
def test_handshake_status_codes_become_typed_refusals(shape, status, code):
    with pytest.raises(RealtimeSetupError) as refused:
        asyncio.run(_session(_refused_handshake(shape, status)).connect_wire())
    assert refused.value.code == code and str(status) in str(refused.value)
    assert "synthetic-secret-canary" not in str(refused.value)  # never the response's headers or reason


def test_an_unexplained_handshake_failure_keeps_its_own_type():
    with pytest.raises(Handshake):
        asyncio.run(_session(Handshake(None)).connect_wire())
