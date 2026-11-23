"""No-network live adapter socket and feature tool-result doubles."""
import asyncio
from contextlib import asynccontextmanager
import json

from ouroboros.contracts.realtime import RealtimeInput, canonical_json


class Wire:
    def __init__(self):
        self.sent, self.incoming = [], asyncio.Queue()
        self.closed = False

    def put(self, event):
        self.incoming.put_nowait(event)

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if "setup" in event:
            self.put({"setupComplete": {}})

    async def recv(self):
        event = await self.incoming.get()
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    @asynccontextmanager
    async def connect(self, model):
        try:
            yield self
        finally:
            self.closed = True


def selected(model="gemini-3.8-live", tools=(), messages=None):
    return RealtimeInput(model, canonical_json(messages or [
        {"role": "system", "content": "Host system rules."},
        {"role": "user", "content": "Original owner history."},
    ]), canonical_json(tools), 256, "nano", "core", "view")


class Batch(list):
    """The additive ToolBatch port shape; trunk owns its contract class."""
    def __init__(self, rows=(), *, silent=()):
        super().__init__(rows)
        self.silent = tuple(silent)
