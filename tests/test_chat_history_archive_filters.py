"""Focused coverage for archive-aware, exact-provenance chat recall."""

from __future__ import annotations

import json
import re

from ouroboros.memory import Memory
from ouroboros.tools.control import get_tools


def _row(ts: str, text: str, *, actor: str = "u-1", **transport: str) -> dict:
    return {
        "ts": ts,
        "direction": "in",
        "text": text,
        "sender_label": "Alex",
        "sender_session_id": actor,
        "transport": {
            "provider": transport.get("provider", "telegram"),
            "account_id": transport.get("account_id", "bot-1"),
            "conversation_id": transport.get("conversation_id", "room-1"),
            "thread_id": transport.get("thread_id", "topic-1"),
            "actor": {"platform_actor_id": actor, "username": "alex"},
        },
    }


def _write(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_chat_history_combines_archive_and_live_with_shared_provenance(tmp_path):
    _write(tmp_path / "archive" / "chat_20260820T000000.jsonl", [
        _row("2026-08-20T09:00:00Z", "archived hello"),
    ])
    _write(tmp_path / "logs" / "chat.jsonl", [
        _row("2026-08-21T09:00:00Z", "live hello"),
    ])

    history = Memory(tmp_path).chat_history(count=2)

    assert history.index("archived hello") < history.index("live hello")
    assert "Alex [provider=telegram; account=bot-1; conversation=room-1; thread=topic-1]" in history


def test_chat_history_exact_provenance_and_inclusive_date_filters(tmp_path):
    _write(tmp_path / "archive" / "chat_20260820T000000.jsonl", [
        _row("2026-08-20T08:59:59Z", "too early"),
        _row("2026-08-20T09:00:00Z", "matching archived"),
    ])
    _write(tmp_path / "logs" / "chat.jsonl", [
        _row("2026-08-20T10:00:00+00:00", "matching live"),
        _row("2026-08-20T10:00:00Z", "wrong actor", actor="u-2"),
        _row("2026-08-20T10:00:00Z", "wrong room", conversation_id="room-2"),
        _row("2026-08-20T10:00:01Z", "too late"),
    ])

    history = Memory(tmp_path).chat_history(
        count=20,
        provider="telegram",
        account_id="bot-1",
        conversation_id="room-1",
        thread_id="topic-1",
        actor_id="u-1",
        date_from="2026-08-20T09:00:00Z",
        date_to="2026-08-20T10:00:00Z",
    )

    assert "matching archived" in history and "matching live" in history
    assert "too early" not in history and "too late" not in history
    assert "wrong actor" not in history and "wrong room" not in history


def test_chat_history_old_call_keeps_search_offset_count_result(tmp_path):
    _write(tmp_path / "logs" / "chat.jsonl", [
        {"ts": "2026-08-21T09:00:00Z", "direction": "in", "text": "match one"},
        {"ts": "2026-08-21T09:01:00Z", "direction": "in", "text": "ignore"},
        {"ts": "2026-08-21T09:02:00Z", "direction": "in", "text": "match two"},
        {"ts": "2026-08-21T09:03:00Z", "direction": "in", "text": "match three"},
    ])

    legacy = Memory(tmp_path).chat_history(count=1, offset=1, search="MATCH")
    assert "Showing 1 of 3 messages; 1 older remain." in legacy
    assert "live offset" in legacy and "shift" in legacy
    assert "← [2026-08-21T09:02] [User] match two" in legacy
    exhausted = Memory(tmp_path).chat_history(count=1, offset=99, search="MATCH")
    assert exhausted.startswith("Showing 0 of 3 messages; matching history is exhausted at offset=99.")
    assert "no messages matching query" not in exhausted


def test_chat_history_snapshot_refuses_append_between_pages(tmp_path):
    _write(tmp_path / "logs" / "chat.jsonl", [
        _row(f"2026-08-21T09:0{index}:00Z", text)
        for index, text in enumerate(("match a", "match b", "match c"))
    ])
    memory = Memory(tmp_path)

    first = memory.chat_history(count=2, search="MATCH")
    snapshot = re.search(r"snapshot=([0-9a-f]{64})", first)
    assert snapshot is not None
    assert "match b" in first and "match c" in first

    with (tmp_path / "logs" / "chat.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row("2026-08-21T09:03:00Z", "match d")) + "\n")

    second = memory.chat_history(
        count=2, offset=2, search="match", snapshot=snapshot.group(1),
    )
    assert second.startswith("CHAT_HISTORY_SNAPSHOT_CHANGED:")
    assert "restart with offset=0" in second
    assert "match a" not in second and "match b" not in second


def test_chat_history_snapshot_pages_when_source_is_unchanged(tmp_path):
    _write(tmp_path / "logs" / "chat.jsonl", [
        _row(f"2026-08-21T09:0{index}:00Z", text)
        for index, text in enumerate(("match a", "match b", "match c"))
    ])
    memory = Memory(tmp_path)

    first = memory.chat_history(count=2, search="match")
    snapshot = re.search(r"snapshot=([0-9a-f]{64})", first)
    assert snapshot is not None
    second = memory.chat_history(
        count=2, offset=2, search="MATCH", snapshot=snapshot.group(1),
    )

    assert "match a" in second
    assert "match b" not in second and "match c" not in second
    assert f"snapshot={snapshot.group(1)}" in second
    mismatch = memory.chat_history(
        count=2, offset=2, search="different query", snapshot=snapshot.group(1),
    )
    assert mismatch.startswith("CHAT_HISTORY_SNAPSHOT_CHANGED:")


def test_chat_history_snapshot_retries_append_during_capture(tmp_path, monkeypatch):
    _write(tmp_path / "logs" / "chat.jsonl", [_row("2026-08-21T09:00:00Z", "first")])
    memory = Memory(tmp_path)
    original = memory._read_chat_generation
    appended = False

    def read_then_append(path, **kwargs):
        nonlocal appended
        rows, gaps = original(path, **kwargs)
        if not appended:
            appended = True
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_row("2026-08-21T09:01:00Z", "second")) + "\n")
        return rows, gaps

    monkeypatch.setattr(memory, "_read_chat_generation", read_then_append)
    entries, coverage = memory.read_chat_generations()

    assert [row["text"] for row in entries] == ["first", "second"]
    assert coverage["capture_attempts"] == 2
    assert coverage["snapshot_stable"] is True


def test_chat_history_gap_never_claims_complete_or_exhausted(tmp_path):
    path = tmp_path / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(
        b'{"ts":"2026-08-21T09:00:00Z","direction":"in","text":"match valid"}\n'
        b'{"direction":"in","text":"match hidden"\n'
    )

    first = Memory(tmp_path).chat_history(count=1, search="match")
    exhausted = Memory(tmp_path).chat_history(count=1, offset=99, search="match")

    assert "1 observed messages" in first
    assert "completeness unknown" in first
    assert "0 older remain" not in first
    assert "no further observed matches" in exhausted
    assert "matching history is exhausted" not in exhausted
    assert "jsonl_malformed" in exhausted


def test_explicit_chat_history_surfaces_missing_consolidation_generation(tmp_path):
    _write(tmp_path / "logs" / "chat.jsonl", [_row("2026-08-21T09:00:00Z", "survivor")])
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "dialogue_meta.json").write_text(json.dumps({
        "last_consolidated_offset": 50,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 999},
    }), encoding="utf-8")

    result = Memory(tmp_path).chat_history(count=20)

    assert "survivor" in result
    assert "consolidation_cursor_generation_missing" in result
    assert "completeness unknown" in result

    (tmp_path / "logs" / "chat.jsonl").unlink()
    no_survivors = Memory(tmp_path).chat_history(count=20)
    assert "consolidation_cursor_generation_missing" in no_survivors
    assert "completeness unknown" in no_survivors


def test_chat_history_keeps_durable_gap_after_consolidator_rebases_cursor(tmp_path):
    from ouroboros.consolidator import consolidate, should_consolidate
    from ouroboros.chronicle_store import ChronicleStore

    _write(tmp_path / "logs" / "chat.jsonl", [_row("2026-08-21T09:00:00Z", "survivor")])
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    blocks_path = memory_dir / "dialogue_blocks.json"
    meta_path = memory_dir / "dialogue_meta.json"
    meta_path.write_text(json.dumps({
        "last_consolidated_offset": 50,
        "chat_log_signature": {"first_line_sha256": "f" * 64, "size": 999},
    }), encoding="utf-8")
    memory = Memory(tmp_path)

    first = memory.chat_history(count=20)
    old_snapshot = re.search(r"snapshot=([0-9a-f]{64})", first)
    assert old_snapshot is not None
    assert "consolidation_cursor_generation_missing" in first
    assert should_consolidate(meta_path, tmp_path / "logs" / "chat.jsonl") is True

    class NoLlmExpected:
        def chat(self, **_kwargs):
            raise AssertionError("open raw rows must not be summarized during gap rebasing")

    assert consolidate(
        chat_path=tmp_path / "logs" / "chat.jsonl",
        blocks_path=blocks_path,
        meta_path=meta_path,
        llm_client=NoLlmExpected(),
    )["_blocks_written"] == 0
    store = ChronicleStore(tmp_path)
    gaps = store.records(kinds=["gap"])
    assert len(gaps) == 1 and gaps[0]["id"].startswith("gap:")
    assert "[MEMORY GAP]" in gaps[0]["text"]
    assert store.scan_state()["last_consolidated_offset"] == 0
    assert store.scan_state()["chat_log_signature"]["first_line_sha256"] != "f" * 64
    assert json.loads(meta_path.read_text(encoding="utf-8"))["last_consolidated_offset"] == 50
    assert not blocks_path.exists()
    consolidate(
        chat_path=tmp_path / "logs" / "chat.jsonl", blocks_path=blocks_path,
        meta_path=meta_path, llm_client=NoLlmExpected(),
    )
    assert len(store.records(kinds=["gap"])) == 1

    second = memory.chat_history(count=20)
    stale_page = memory.chat_history(count=20, snapshot=old_snapshot.group(1))
    assert "durable_consolidation_gap" in second
    assert "completeness unknown" in second
    assert stale_page.startswith("CHAT_HISTORY_SNAPSHOT_CHANGED:")


def test_chat_history_recognizes_legacy_memory_gap_block_without_gap_id(tmp_path):
    _write(tmp_path / "logs" / "chat.jsonl", [_row("2026-08-21T09:00:00Z", "survivor")])
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": "2026-08-21T10:00:00Z",
        "type": "summary",
        "range": "unknown",
        "message_count": 0,
        "content": "[MEMORY GAP] Legacy durable discontinuity.",
    }]), encoding="utf-8")

    result = Memory(tmp_path).chat_history(count=20)

    assert "durable_consolidation_gap" in result
    assert "completeness unknown" in result


def test_chat_history_tool_exposes_only_exact_filter_fields():
    tool = next(entry for entry in get_tools() if entry.name == "chat_history")
    assert set(tool.schema["parameters"]["properties"]) == {
        "count", "offset", "search", "provider", "account_id", "conversation_id",
        "thread_id", "actor_id", "date_from", "date_to", "snapshot",
    }
    assert tool.schema["parameters"]["additionalProperties"] is False
