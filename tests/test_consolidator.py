"""The public consolidation entry activates one writer and preserves legacy sources."""
import json
from unittest.mock import MagicMock

import pytest

from ouroboros.consolidator import consolidate, should_consolidate, _load_meta
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.chronicle_sources import format_source_row
from ouroboros.utils import atomic_write_json
from tests.test_consolidator_context_fit import fit as _fit

fit = _fit


@pytest.fixture
def tmp_paths(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "memory").mkdir()
    return tmp_path / "logs/chat.jsonl", tmp_path / "memory/dialogue_blocks.json", tmp_path / "memory/dialogue_meta.json"


def _write_chat_entries(path, count):
    rows = [{"ts": f"2026-02-25T{10 + i // 60:02d}:{i % 60:02d}:00Z", "chat_id": 1,
             "task_id": "fixture", "direction": "in" if i % 2 == 0 else "out", "text": f"Message {i}"}
            for i in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def test_should_consolidate_no_sources(tmp_paths):
    chat, _, meta = tmp_paths
    assert should_consolidate(meta, chat) is False


@pytest.mark.parametrize("count", [1, 5, 105])
def test_unseen_source_is_not_gated_by_a_global_message_count(tmp_paths, fit, count):
    chat, blocks, meta = tmp_paths
    _write_chat_entries(chat, count)
    assert should_consolidate(meta, chat)
    model = MagicMock()
    model.chat.return_value = ({"content": "The actor understood the completed conversation."},
                               {"prompt_tokens": 100, "completion_tokens": 50, "cost": .001})
    raw = chat.read_bytes()
    usage = consolidate(chat, blocks, meta, model, completed_task={"id": "fixture"})
    store = ChronicleStore(meta.parent.parent)
    assert store.activation()["kind"] == "activation"
    assert len(store.records(kinds=["episode"])) == len(store.records(kinds=["revision"])) == 1
    assert store.scan_state()["last_consolidated_offset"] == count
    assert usage["cost"] == pytest.approx(.002) and model.chat.call_count == 2
    assert chat.read_bytes() == raw and not blocks.exists() and not meta.exists()
    assert not should_consolidate(meta, chat)
    consolidate(chat, blocks, meta, model, completed_task={"id": "fixture"})
    assert model.chat.call_count == 2  # a represented source is not purchased again


def test_an_open_room_is_retained_without_an_invented_completed_producer(tmp_paths, fit):
    chat, blocks, meta = tmp_paths
    _write_chat_entries(chat, 105)
    model = MagicMock()
    usage = consolidate(chat, blocks, meta, model)
    assert usage["_blocks_written"] == 0
    model.chat.assert_not_called()
    assert not ChronicleStore(meta.parent.parent).records(kinds=["episode"])


def test_activation_failure_is_typed_and_never_reenters_the_old_writer(tmp_paths, monkeypatch):
    chat, blocks, meta = tmp_paths
    _write_chat_entries(chat, 105)
    blocks.write_text('[{"content":"retained old memory"}]', encoding="utf-8")
    before = blocks.read_bytes()
    monkeypatch.setattr(ChronicleStore, "import_legacy", lambda self, **kw: {"kind": "import_pending", "reason": "source busy"})
    model = MagicMock()
    usage = consolidate(chat, blocks, meta, model, completed_task={"id": "fixture"})
    assert usage["_consolidation_errors"][0]["reason"] == "chronicle_activation_incomplete"
    assert usage["_blocks_written"] == 0 and blocks.read_bytes() == before
    model.chat.assert_not_called()
    events = [json.loads(row) for row in (chat.parent / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "consolidation_skipped_source"


def test_legacy_cursor_and_flat_summary_remain_readable_and_byte_identical(tmp_paths, fit):
    chat, blocks, meta = tmp_paths
    rows = _write_chat_entries(chat, 5)
    atomic_write_json(meta, {"last_consolidated_offset": 0})
    atomic_write_json(blocks, [{"type": "summary", "range": "old", "content": "A retained earlier account."}])
    flat = meta.parent / "dialogue_summary.md"
    flat.write_text("A still older account.", encoding="utf-8")
    originals = {path: path.read_bytes() for path in (blocks, meta, flat)}
    model = MagicMock()
    model.chat.return_value = ({"content": "A new understood episode."}, {"cost": .001})
    consolidate(chat, blocks, meta, model, completed_task={"id": "fixture"})
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert _load_meta(meta) == {"last_consolidated_offset": 0}
    store = ChronicleStore(meta.parent.parent)
    assert {r["text"] for r in store.records(kinds=["legacy"])} == {"A retained earlier account.", "A still older account."}
    assert store.records(kinds=["episode"])[0]["metadata"]["message_count"] == len(rows)


def test_read_chat_entries_includes_project_rows_full_awareness(tmp_path):
    from ouroboros.consolidator import _read_chat_entries
    from ouroboros.projects_registry import create_project

    (tmp_path / "logs").mkdir()
    project_chat = int(create_project(tmp_path, "racer")["chat_id"])
    chat = tmp_path / "logs/chat.jsonl"
    rows = [{"chat_id": 1, "direction": "in", "text": "main-1"},
            {"chat_id": project_chat, "direction": "in", "text": "project-visible"},
            {"chat_id": 1, "direction": "out", "text": "main-2"},
            {"chat_id": -1001, "direction": "out", "text": "a2a-noise"}]
    chat.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert [row["text"] for row in _read_chat_entries(chat)] == ["main-1", "project-visible", "main-2"]


def test_current_formatter_keeps_speakers_in_retained_source():
    entries = [{"ts": "2026-02-25T10:00:00Z", "direction": "in", "text": "Hello"},
               {"ts": "2026-02-25T10:01:00Z", "direction": "out", "text": "Hi there"}]
    incoming, outgoing = map(format_source_row, entries)
    assert "; User;" in incoming and incoming.endswith("\nHello")
    assert "; Ouroboros;" in outgoing and outgoing.endswith("\nHi there")
    assert json.loads(incoming.splitlines()[1])["direction"] == "in"
    assert json.loads(outgoing.splitlines()[1])["direction"] == "out"
