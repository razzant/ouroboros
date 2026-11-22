"""Append-only notices retain supported local and GigaChat request shapes."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.llm import LLMClient
from ouroboros.loop_messages import _append_or_merge_user_content


@pytest.mark.parametrize("notice", ["notice", [{"type": "text", "text": "notice"}]])
def test_appended_notice_preserves_local_and_gigachat_tool_adjacency(monkeypatch, notice):
    from gigachat.models import Chat

    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    messages = [
        {"role": "system", "content": "root"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "read-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "read-1", "content": "proof"},
        {"role": "user", "content": "follow-up"},
    ]
    sent = copy.deepcopy(messages)
    _append_or_merge_user_content(messages, notice)
    assert messages[:len(sent)] == sent and len(messages) == len(sent) + 1
    canonical = copy.deepcopy(messages)

    client = LLMClient()
    _, local = client._build_local_candidate(messages, None, 64, "auto")
    giga = client._gigachat_messages(messages)

    assert messages == canonical
    assert [m["role"] for m in local["messages"]] == ["system", "user", "assistant", "tool", "user", "user"]
    assert [m["role"] for m in giga] == ["system", "user", "assistant", "function", "user", "user"]
    assert local["messages"][3]["tool_call_id"] == "read-1"
    assert local["messages"][3]["content"] == "proof"
    assert giga[3]["name"] == "read_file" and json.loads(giga[3]["content"]) == {"result": "proof"}
    for rows in (local["messages"], giga):
        assert [m["content"] for m in rows[-2:]] == ["follow-up", "notice"]
    # SDK shape validation only: no authentication, request, or model generation.
    assert len(Chat(model="GigaChat-test", messages=giga, max_tokens=64).messages) == len(messages)


def test_local_assistant_history_preserves_ordered_text_and_refusal_bytes(monkeypatch):
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    history = [
        {"role": "user", "content": [{"type": "text", "text": "First"}, {"type": "text", "text": "second"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "  Ob"},
            {"type": "refusal", "refusal": "servation"},
            {"type": "text", "text": ".\nNext line.  "},
        ]},
    ]
    canonical = copy.deepcopy(history)
    _, payload = LLMClient()._build_local_candidate(history, None, 64, "auto")
    assert payload["messages"] == [
        {"role": "user", "content": "First\n\nsecond"},
        {"role": "assistant", "content": "  Observation.\nNext line.  "},
    ]
    assert history == canonical
