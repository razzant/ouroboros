"""The one synthetic streamed OpenAI-compatible reply, shared by transport-level checks.

Owned here so a test module does not import another test module to complete a send:
``tests/test_opencode_go_chat.py`` drives the OpenCode Go conversation paths and
``tests/test_wrapup_real_send_parity.py`` meets the real wrap-up send at the executor
seam; both need the same well-formed SSE chat-completion body (content ``\"4\"`` plus usage,
or one tool call) with no network. Nothing here selects a model or route."""

from __future__ import annotations

import json

import httpx


def streamed_reply(*, tool_call=None):
    common = {"id": "reply", "object": "chat.completion.chunk", "created": 0, "model": "example-model"}
    delta = {"role": "assistant", **({"tool_calls": [{"index": 0, **tool_call}]} if tool_call else {"content": "4"})}
    frames = [{**common, "choices": [{"index": 0, "delta": delta, "finish_reason": "tool_calls" if tool_call else "stop"}]},
              {**common, "choices": [], "usage": {"prompt_tokens": 23, "completion_tokens": 4, "cost": 0}}]
    content = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})
