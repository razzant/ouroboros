"""Offline wire contract for the shared UI model fixture."""
from __future__ import annotations

import io
import json
from email.parser import BytesParser
from types import SimpleNamespace

import pytest

from tests import fixtures_mock_llm as mock_llm


USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


@pytest.fixture(autouse=True)
def _no_hold(monkeypatch):
    monkeypatch.setattr(mock_llm, "HOLD_SECONDS", 0.0)


def _post(payload, *, path="/v1/chat/completions"):
    raw = json.dumps(payload).encode("utf-8")
    # Bypass the socket-owning constructor, not the handler's HTTP response code.
    handler = object.__new__(mock_llm._Handler)
    handler.path = path
    handler.command = "POST"
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.do_POST()
    head, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
    status, header_bytes = head.split(b"\r\n", 1)
    headers = BytesParser().parsebytes(header_bytes + b"\r\n\r\n")
    return status.split()[1], headers, body


@pytest.mark.parametrize("options", [{}, {"stream": False}], ids=["omitted", "false"])
def test_non_stream_completion_is_unchanged(options):
    status, headers, body = _post({"model": "fixture-model", **options})
    assert status == b"200"
    assert headers["Content-Type"] == "application/json"
    assert int(headers["Content-Length"]) == len(body)
    assert json.loads(body) == {
        "id": "mock-chat",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "OK"}}],
        "usage": USAGE,
    }


@pytest.mark.parametrize(
    "options", [{}, {"stream_options": {"include_usage": True}}],
    ids=["plain", "usage-requested"],
)
def test_stream_completion_has_content_terminal_usage_and_done(options):
    status, headers, body = _post({
        "model": "fixture-model", "stream": True, **options,
    })
    assert status == b"200"
    assert headers["Content-Type"] == "text/event-stream"
    assert int(headers["Content-Length"]) == len(body)
    events = body.decode("utf-8").split("\n\n")
    assert events.pop() == "", "the last SSE event must be terminated"
    assert events.pop() == "data: [DONE]"
    assert events and all(event.startswith("data: ") for event in events)
    chunks = [json.loads(event[len("data: "):]) for event in events]
    for chunk in chunks:
        assert chunk["id"] == "mock-chat"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "fixture-model"
        assert isinstance(chunk["created"], int)
    choices = [choice for chunk in chunks for choice in chunk["choices"]]
    assert choices and all(choice["index"] == 0 for choice in choices)
    assert choices[0]["delta"]["role"] == "assistant"
    assert "".join(choice["delta"].get("content", "") for choice in choices) == "OK"
    assert all(choice["finish_reason"] is None for choice in choices[:-1])
    assert choices[-1]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == USAGE


@pytest.mark.parametrize("stream", [False, True])
def test_both_completion_modes_use_the_shared_hold(monkeypatch, stream):
    waits = []
    monkeypatch.setattr(mock_llm, "HOLD_SECONDS", 2.5)
    monkeypatch.setattr(mock_llm, "HOLD_RELEASE", SimpleNamespace(wait=waits.append))
    status, _, _ = _post({"stream": stream})
    assert status == b"200"
    assert waits == [2.5]


def test_other_endpoint_still_returns_not_found():
    status, _, _ = _post({"stream": True}, path="/not-chat")
    assert status == b"404"
