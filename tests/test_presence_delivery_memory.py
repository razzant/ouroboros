"""Every memory reader retains where a reported message went and what is known."""
from __future__ import annotations

import pytest
import json

from ouroboros.chronicle_sources import format_source_row
from ouroboros.memory import Memory
from ouroboros.utils import append_jsonl


def _row(state):
    return {
        "ts": "2026-01-02T03:04:05Z", "direction": "out", "chat_id": 73,
        "text": "The exact message", "source": "presence:chat-provider",
        "transport": {"provider": "chat-provider", "account_id": "account-1",
                      "conversation_id": "direct-7", "thread_id": "thread-2",
                      "delivery": {"state": state, "delivery_id": "request-3", "part_id": "0"}},
    }


@pytest.mark.parametrize("state,label", [
    ("delivered", "delivery=delivered"),
    ("authored", "delivery=authored (delivery unconfirmed)"),
    ("accepted", "delivery=accepted (provider acceptance only)"),
])
def test_outgoing_destination_and_state_survive_all_memory_views(tmp_path, state, label):
    row = _row(state)
    memory = Memory(tmp_path)
    append_jsonl(tmp_path / "logs/chat.jsonl", row)
    for view in (memory.summarize_chat([row]), memory.chat_history(count=10)):
        assert "The exact message" in view
        assert "provider=chat-provider" in view
        assert "account=account-1" in view
        assert "conversation=direct-7" in view
        assert "thread=thread-2" in view
        assert label in view
    view = format_source_row(row)
    metadata = json.loads(view.splitlines()[1])
    assert metadata["direction"] == "out" and metadata["transport"] == row["transport"]
    assert "; Ouroboros; source_row_id=" in view and view.endswith("The exact message")


@pytest.mark.parametrize("state", ["failed", "uncertain"])
def test_failed_send_is_a_system_fact_not_confirmed_speech(state):
    row = {**_row(state), "direction": "system", "type": "presence_delivery"}
    for view in (Memory._format_chat_line(row, compact=True),):
        assert "delivery=" + state in view
        assert "conversation=direct-7" in view
        assert "delivery=delivered" not in view
    assert Memory._format_chat_line(row, compact=True).startswith("📋")
    source = format_source_row(row)
    metadata = json.loads(source.splitlines()[1])
    assert metadata["direction"] == "system" and metadata["type"] == "presence_delivery"
    assert metadata["transport"]["delivery"]["state"] == state
    assert source.endswith("The exact message") and "delivery=delivered" not in source


def test_ordinary_owner_reply_format_is_unchanged():
    row = {"direction": "out", "ts": "2026-01-02T03:04:05Z", "text": "Hello", "source": "web", "transport": {}}
    assert Memory._format_chat_line(row, compact=True) == "→ 03:04 Hello"
    assert Memory._format_chat_line(row, compact=False) == "→ [2026-01-02T03:04] Hello"
    source = format_source_row(row)
    assert source.startswith("[2026-01-02T03:04:05Z; Ouroboros; source_row_id=")
    assert json.loads(source.splitlines()[1])["direction"] == "out" and source.endswith("\nHello")


def test_attachment_and_mail_receipt_facts_do_not_disappear_from_memory(tmp_path):
    row = {**_row("accepted"), "type": "presence_delivery", "text": ""}
    row["transport"]["message"] = {
        "attachments": [{"filename": "report.pdf", "mime_type": "application/pdf"}],
        "recipients": ["reader@example.org"], "subject": "Requested report",
        "provider_message_id": "provider-42",
    }
    memory = Memory(tmp_path)
    append_jsonl(tmp_path / "logs/chat.jsonl", row)
    for view in (memory.summarize_chat([row]), memory.chat_history(count=10)):
        assert "Delivery details:" in view and "report.pdf" in view
        assert "reader@example.org" in view and "Requested report" in view
        assert "provider acceptance only" in view
    source = format_source_row(row)
    assert "Delivery details:" in source and "report.pdf" in source
    assert "reader@example.org" in source and "Requested report" in source
    assert json.loads(source.splitlines()[1])["transport"] == row["transport"]
    assert "report.pdf" in memory.chat_history(count=10, search="report.pdf")
    legacy = {**row, "type": ""}
    assert "Delivery details:" not in Memory._format_chat_line(legacy, compact=True)
