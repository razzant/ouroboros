"""Every memory reader retains where a reported message went and what is known."""
from __future__ import annotations

import pytest

from ouroboros.dialogue_provenance import row_author
from ouroboros.memory import Memory
from ouroboros.utils import append_jsonl


def _chronicle_view(row):
    """One chat row as ``memory_read`` renders it — the successor of the retired block formatter."""
    from ouroboros import chat_chain
    from ouroboros.tools.chronicle import _row_line

    header, text = _row_line(chat_chain.row_address(row), row, 0, {})
    return f"{header} {text}"


def _view_line(row):
    """The line the memory view prints for an open row of lane 1 (``memory_view._row_line``)."""
    from ouroboros import chat_chain
    from ouroboros.dialogue_provenance import render_memory_row

    return render_memory_row(chat_chain.row_address(row), row, author=row_author(row), indent="  ")


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
    for view in (_view_line(row), memory.chat_history(count=10), _chronicle_view(row)):
        assert "The exact message" in view
        assert "provider=chat-provider" in view
        assert "account=account-1" in view
        assert "conversation=direct-7" in view
        assert "thread=thread-2" in view
        assert label in view


@pytest.mark.parametrize("state", ["failed", "uncertain"])
def test_failed_send_is_a_system_fact_not_confirmed_speech(state):
    row = {**_row(state), "direction": "system", "type": "presence_delivery"}
    for view in (Memory._format_chat_line(row, compact=True), _chronicle_view(row)):
        assert "delivery=" + state in view
        assert "conversation=direct-7" in view
        assert "delivery=delivered" not in view
    assert Memory._format_chat_line(row, compact=True).startswith("📋")
    author = row_author(row)
    assert author["kind"] == "host" and "Ouroboros" not in author["label"]
    # In the view it is a typed fact of its task's lane-2 line: the state and the address, no JSON.
    from ouroboros import memory_view
    from ouroboros.memory_inventory import row_meta

    assert memory_view._typed_fact(row_meta(row)) == "delivery " + state
    assert memory_view._typed_fact(row_meta({**row, "type": "task_summary"})) == ""


def test_ordinary_owner_reply_format_is_unchanged():
    row = {"direction": "out", "ts": "2026-01-02T03:04:05Z", "text": "Hello", "source": "web", "transport": {}}
    assert Memory._format_chat_line(row, compact=True) == "→ 03:04 Hello"
    assert Memory._format_chat_line(row, compact=False) == "→ [2026-01-02T03:04] Hello"
    assert row_author(row)["label"] == "Ouroboros"  # no transport facts, no suffix


def test_attachment_and_mail_receipt_facts_do_not_disappear_from_memory(tmp_path):
    row = {**_row("accepted"), "type": "presence_delivery", "text": ""}
    row["transport"]["message"] = {
        "attachments": [{"filename": "report.pdf", "mime_type": "application/pdf"}],
        "recipients": ["reader@example.org"], "subject": "Requested report",
        "provider_message_id": "provider-42",
    }
    memory = Memory(tmp_path)
    append_jsonl(tmp_path / "logs/chat.jsonl", row)
    for view in (_view_line(row), memory.chat_history(count=10), _chronicle_view(row)):
        assert "Delivery details:" in view and "report.pdf" in view
        assert "reader@example.org" in view and "Requested report" in view
        assert "provider acceptance only" in view
    # Memory text (the view's line and memory_read) carries the details as words, never JSON;
    # chat_history keeps its raw details.
    for view in (_view_line(row), _chronicle_view(row)):
        assert "{" not in view and "[Delivery details: attachments filename report.pdf, mime_type application/pdf; " \
            "provider_message_id provider-42; recipients reader@example.org; subject Requested report]" in view
    assert '"filename": "report.pdf"' in memory.chat_history(count=10)
    assert "report.pdf" in memory.chat_history(count=10, search="report.pdf")
    legacy = {**row, "type": ""}
    assert "Delivery details:" not in Memory._format_chat_line(legacy, compact=True)
