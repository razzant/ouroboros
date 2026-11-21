"""Shared stream-wire fixtures: the chat.completion chunk dict, the SSE encoding and the fake wire response.

Moved from ``tests/test_transport_b_stream_deadlines.py`` so test modules do not import each other
(Development 02, "Documentation contract").
"""
import json


def chunk(delta=None, finish=None, *, index=0, **extra):
    return {"id": "gen-test", "object": "chat.completion.chunk", "model": "vendor/test-stream",
            "choices": [{"index": index, "delta": delta or {}, "finish_reason": finish}], **extra}


def sse(*events, done=True):
    wire = b""
    for event in events:
        if isinstance(event, tuple):
            kind, body = event
            wire += f"event: {kind}\r\n".encode()
        else:
            body = event
        wire += ("data: " + json.dumps(body, ensure_ascii=False) + "\r\n\r\n").encode()
    return wire + (b"data: [DONE]\r\n\r\n" if done else b"")


class WireResponse:
    status_code = 200
    reason = "OK"
    url = "https://provider.invalid/v1/messages"

    def __init__(self, wire, *, failure=None, step=None, chunk_size=13):
        self.wire, self.failure, self.step, self.chunk_size = wire, failure, step, chunk_size
        self.headers = {"x-generation-id": "header-generation"}
        self.closed = False

    def iter_bytes(self):
        # Fragment through UTF-8, CRLF and JSON boundaries (13 bytes by default).
        for offset in range(0, len(self.wire), self.chunk_size):
            if self.step:
                self.step()
            yield self.wire[offset:offset + self.chunk_size]
        if self.failure:
            raise self.failure

    def iter_content(self, **kwargs):
        return self.iter_bytes()

    async def aiter_bytes(self):
        for item in self.iter_bytes():
            yield item

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()
