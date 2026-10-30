"""Mechanical JSON websocket lifetime shared by in-process provider packages.

No provider grammar, context selection, resumption policy, pricing or credential
lookup lives here. The host leases this entire object through close, including
a connection that was interrupted before it finished opening.
"""
from __future__ import annotations

import asyncio
import copy
import json
import string

from ouroboros.contracts.realtime import RealtimeSetupError, canonical_json


def connect_provider(uri, *, headers):
    """Share flow control for data, keepalive and automatic control frames.

    The asyncio connection owns one drain waiter per writer and removes it on
    cancellation. Legacy's single waiter races with ping/pong even when our JSON
    sends are locked. Keep native heartbeat detection and the prior direct route.
    """
    from websockets.asyncio.client import connect

    # A dead-socket bound for the opening handshake, not a timing judgment: on 28.10 a loaded Mac
    # needed more than 10 s and a call failed to start (owner 3A: wait up to 30 s).
    return connect(uri, additional_headers=headers, proxy=None,
                   open_timeout=30, close_timeout=3, max_size=None,
                   ping_interval=20, ping_timeout=20)


async def joined(awaitable):
    """Settle owned work, then propagate cancellation of its waiter."""
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    result = task.result()
    if cancelled is not None:
        raise cancelled
    return result


# A refused websocket handshake, typed for the owner: a wrong or revoked key, a quota, an outage.
_HANDSHAKE_REFUSALS = {401: "authentication", 403: "authentication", 429: "rate_limit"}


def transport_loss(error):
    """Wire termination, not a provider refusal. RFC 6455 / IANA close codes.

    No received close frame means the network/heartbeat failed. 1011 alone is
    ambiguous (Google also uses it for depleted credit), so only its exact
    neutral internal-error/expiry sentences qualify; never match keywords in
    arbitrary provider text. Error events and failed responses stay typed.
    """
    received = getattr(error, "rcvd", None)
    if received is None:
        return True
    return received.code in {1000, 1001, 1005, 1006, 1012, 1013, 1014} or (
        received.code == 1011 and (not received.reason.strip()
            or received.reason.strip().rstrip(string.punctuation).casefold() in {
                "internal error encountered", "deadline expired", "deadline expired before operation could complete"}))


class JsonSession:
    # Providers may hold native transcription/boundary work after generation
    # stops. The feature waits for that work before replacing a connection.
    renewal_ready = True

    def __init__(self, selected, options, *, connect, observe):
        self.selected, self.options = selected, options
        self.connect, self.observe = connect, observe
        self.ws = self.connection = None
        self._send_lock = asyncio.Lock()
        self.held_context_rows = json.loads(getattr(selected, "messages_json", "[]"))
        self.response_context_rows = {}

    def hold_context(self, rows):
        """Adapter delivery boundary, after a successful native write, never on queueing."""
        self.held_context_rows.extend(copy.deepcopy(list(rows)))

    def capture_response_context(self, response_id):
        # Snapshot before a turn-complete handler releases queued updates. Those
        # later writes cannot borrow the answer that just ended as exposure.
        self.response_context_rows[response_id] = list(self.held_context_rows)

    def hold_tool_call(self, call_id, name, arguments):
        self.hold_context([{'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}]}])

    async def connect_wire(self):
        self.connection = self.connect(self.selected.model)
        try:
            self.ws = await self.connection.__aenter__()
        except Exception as exc:  # noqa: BLE001 - only the handshake status is read; anything else keeps its type
            status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            code = _HANDSHAKE_REFUSALS.get(status) or ("engine_unavailable" if type(status) is int and status >= 500 else None)
            if code is None:
                raise
            raise RealtimeSetupError(code, f"the voice service answered the connection with HTTP {status}") from exc

    async def write(self, payload):
        raw = canonical_json(payload)
        async with self._send_lock:
            self.observe("send", raw)
            await self.ws.send(raw)

    async def read(self):
        raw = await self.ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        self.observe("receive", raw)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RealtimeSetupError("wire", "provider event must be an object")
        if "error" in value and not self.is_benign_error(value):
            # Raw evidence is retained in the task. Provider text may echo credentials;
            # it never crosses the owner UI as a formatted exception.
            raise RealtimeSetupError("provider_error", "provider refused a live event; see the task evidence")
        return value

    async def timed_read(self, timeout):
        """A metered read owns its child even when completion races cancellation.

        Python 3.10 wait_for can return a completed child's value after its
        caller was cancelled. A connection reader must instead finish teardown.
        """
        pending = asyncio.create_task(self.read())
        try:
            done, _ = await asyncio.wait({pending}, timeout=timeout)
            if not done:
                raise asyncio.TimeoutError
            return pending.result()
        finally:
            pending.cancel()
            await joined(asyncio.gather(pending, return_exceptions=True))

    def is_benign_error(self, event):
        """Only a provider can recognize an expected error for its own request."""
        return False

    def final_captions(self):
        """Known but unfinished text at actual teardown; no new wire work."""
        return []

    async def close(self):
        connection, self.connection = self.connection, None
        try:
            if connection is not None:
                await joined(connection.__aexit__(None, None, None))
        finally:
            self.ws = None
