"""WS4 — WebSocket server-side hardening (v6.34.0): every client is sent to by its
own writer (no per-client head-of-line); chat-history jsonl parsing runs off the
event loop. Server-side only (no frontend change)."""

from __future__ import annotations

import asyncio


def test_broadcast_is_concurrent_not_head_of_line():
    """A slow / half-open client must NOT delay the broadcast (and the heartbeat)
    to every other client: one writer per socket, not a sequential await loop."""
    from ouroboros.gateway import ws as wsmod

    order: list = []

    class _Client:
        def __init__(self, name, delay):
            self.name = name
            self.delay = delay

        async def send_text(self, data):
            await asyncio.sleep(self.delay)
            order.append(self.name)

    async def _run():
        slow = wsmod._register(_Client("slow", 0.20))  # slow registered FIRST
        fast = wsmod._register(_Client("fast", 0.0))
        await wsmod.broadcast_ws({"type": "x"})
        while len(order) < 2:  # each socket's own writer sends; none waits for another
            await asyncio.sleep(0.01)
        for connection in (slow, fast):
            wsmod._unregister(connection)

    try:
        asyncio.run(_run())
        # The fast client completes before the slow one despite being second:
        # proof the sends ran concurrently, not sequentially head-of-line.
        assert order == ["fast", "slow"]
    finally:
        wsmod._ws_clients[:] = []


def test_broadcast_drops_only_failed_clients():
    """A client whose send raises is dropped; healthy clients still receive."""
    from ouroboros.gateway import ws as wsmod

    received: list = []

    class _Good:
        async def send_text(self, data):
            received.append("good")

    class _Bad:
        async def send_text(self, data):
            raise RuntimeError("broken pipe")

    async def _run():
        bad, good = wsmod._register(_Bad()), wsmod._register(_Good())
        await wsmod.broadcast_ws({"type": "y"})
        await bad.writer  # retired by its failed send
        while not received:
            await asyncio.sleep(0.01)
        registered = list(wsmod._ws_clients)
        wsmod._unregister(good)
        return good, bad, registered

    try:
        good, bad, registered = asyncio.run(_run())
        assert received == ["good"]
        # The broken client was removed from the registry; the good one stays.
        assert bad not in registered
        assert good in registered
    finally:
        wsmod._ws_clients[:] = []
