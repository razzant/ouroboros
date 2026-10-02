"""Corrupt legacy sources and downgrade writes never become silent memory loss."""
import json
import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.chronicle_view import capture_chronicle, render_memory
from ouroboros.memory import Memory


def retained(root, ref):
    return read_actor_source_bytes(root, ref["task_id"], ref)


def terminal(task_id):
    return {"chat_id": 1, "direction": "system", "type": "task_summary", "task_id": task_id,
            "summary_kind": "terminal_root_projection", "outcome_authority": "canonical_task_result_after_finalization",
            "outcome_final": True, "outcome_phase": "done", "status": "completed", "text": "Finished task."}


@pytest.mark.parametrize("bad", [b"{broken", b'{}', b'null', b'\xff\xfe'])
def test_bad_blocks_are_copied_but_valid_flat_and_new_authorship_are_visible(tmp_path, bad):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "dialogue_blocks.json").write_bytes(bad)
    (memory / "dialogue_summary.md").write_text("A surviving earlier life.", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    receipt = store.import_legacy()
    assert receipt["kind"] == "activation"
    assert retained(tmp_path, receipt["metadata"]["source_refs"]["blocks"]) == bad
    assert (memory / "dialogue_blocks.json").read_bytes() == bad
    store.append_episode("1", "I learned something new after activation.", [], {"kind": "mind"})
    snapshot = json.loads(capture_chronicle(Memory(tmp_path, tmp_path), {"id": "read", "chat_id": 1}))
    text, _facts = render_memory(snapshot)
    assert "A surviving earlier life." in text
    assert "I learned something new after activation." in text
    assert "[MEMORY GAP]" in text
    assert not list(memory.glob("*.corrupt*"))


def test_invalid_member_does_not_discard_other_valid_blocks(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    blocks = [{"content": "First valid memory."}, 17, {"content": "Last valid memory."}]
    source = json.dumps(blocks).encode()
    (memory / "dialogue_blocks.json").write_bytes(source)
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    rows = store.room_records("legacy")
    assert {"First valid memory.", "Last valid memory."} <= {row["text"] for row in rows}
    gap = next(row for row in rows if row["metadata"].get("legacy_type") == "gap")
    assert gap["metadata"]["location"] == "1"
    assert retained(tmp_path, gap["source_refs"][0]) == source


@pytest.mark.parametrize("cursor", [b"{broken", b"[]", b'{"last_consolidated_offset":-1}',
                                     b'{"last_consolidated_offset":"100"}'])
def test_unknown_cursor_captures_boundary_without_paid_historical_rebuild(tmp_path, monkeypatch, cursor):
    from ouroboros import consolidator as c
    from tests.test_chronicle_consolidation import Helper
    from ouroboros.tools.registry import ToolContext

    memory, logs = tmp_path / "memory", tmp_path / "logs"
    memory.mkdir()
    logs.mkdir()
    blocks, meta, chat = memory / "dialogue_blocks.json", memory / "dialogue_meta.json", logs / "chat.jsonl"
    blocks.write_text('[{"content":"The surviving old interpretation."}]', encoding="utf-8")
    meta.write_bytes(cursor)
    old = {"chat_id": 1, "task_id": "old-task", "direction": "in", "ts": "2026-01-01", "text": "Old raw history."}
    chat.write_text(json.dumps(old) + "\n" + json.dumps(terminal("old-task")) + "\n", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    store.import_legacy()
    assert helper.calls == []
    assert store.scan_state()["last_consolidated_offset"] == 2
    assert any(row["metadata"].get("legacy_type") == "cursor_gap" for row in store.room_records("legacy"))
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="new-experience")
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx)
    assert helper.calls == []
    with chat.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**old, "task_id": "new-experience", "ts": "2026-09-30", "text": "A genuinely new experience."}) + "\n")
    c.consolidate(chat, blocks, meta, None, knowledge_context=ctx, completed_task={"id": "new-experience"})
    assert len(helper.calls) == 2
    assert "Old raw history." not in helper.calls[0][0]
    assert "A genuinely new experience." in helper.calls[0][0]
    assert meta.read_bytes() == cursor


def test_unknown_cursor_with_empty_live_generation_anchors_last_archive(tmp_path, monkeypatch):
    from ouroboros import consolidator as c
    from ouroboros.tools.registry import ToolContext
    from tests.test_chronicle_consolidation import Helper

    memory, logs, archive = (tmp_path / name for name in ("memory", "logs", "archive"))
    for directory in (memory, logs, archive):
        directory.mkdir()
    (memory / "dialogue_meta.json").write_text("{broken", encoding="utf-8")
    old = {"chat_id": 1, "task_id": "old-task", "text": "Old archived event."}
    (archive / "chat_20260101.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(terminal("old-task")) + "\n", encoding="utf-8")
    chat = logs / "chat.jsonl"
    chat.write_text("", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    store.import_legacy()
    helper = Helper()
    monkeypatch.setattr(c, "_light_call", lambda *_args: helper)
    chat.write_text(json.dumps({"chat_id": 1, "task_id": "new", "text": "New live event."}) + "\n", encoding="utf-8")
    c.consolidate(chat, memory / "dialogue_blocks.json", memory / "dialogue_meta.json", None,
        knowledge_context=ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="new"), completed_task={"id": "new"})
    assert len(helper.calls) == 2
    assert "Old archived event." not in helper.calls[0][0]


def test_reupgrade_captures_changed_legacy_without_resetting_new_memory_or_frontier(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    blocks, meta = memory / "dialogue_blocks.json", memory / "dialogue_meta.json"
    before = b'[{"content":"Original legacy understanding."}]'
    blocks.write_bytes(before)
    meta.write_text("{}", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    receipt = store.import_legacy()
    authored = store.append_episode("1", "A later authored understanding.", [], {"kind": "mind"})
    store.publish([], scan_state={"last_consolidated_offset": 77, "new_state": "keep"})
    after = b'[{"content":"Changed during downgrade."}]'
    blocks.write_bytes(after)
    meta.write_text('{"last_consolidated_offset":999}', encoding="utf-8")
    assert store.import_legacy() == receipt
    assert store.scan_state() == {"last_consolidated_offset": 77, "new_state": "keep"}
    assert store.get(authored["id"])["text"] == "A later authored understanding."
    gap = next(row for row in store.room_records("legacy") if row["metadata"].get("legacy_type") == "reconciliation_gap")
    preserved = [retained(tmp_path, ref) for ref in gap["source_refs"]]
    assert before in preserved and after in preserved
    assert gap["metadata"]["coverage"] == "unknown"
    unchanged = store.log_path.read_bytes()
    assert store.import_legacy() == receipt
    assert store.log_path.read_bytes() == unchanged
    assert blocks.read_bytes() == after


def test_failed_activation_still_copies_all_readable_legacy_sources(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    inputs = {"dialogue_blocks.json": b"{broken", "dialogue_meta.json": b"[]", "dialogue_summary.md": b"Valid flat text."}
    for name, data in inputs.items():
        (memory / name).write_bytes(data)
    store = ChronicleStore(tmp_path)
    def reject(*_args, **_kwargs):
        raise OSError("publication unavailable")
    monkeypatch.setattr(store, "publish", reject)
    with pytest.raises(OSError, match="publication unavailable"):
        store.import_legacy()
    assert store.activation() is None
    copied = list((tmp_path / "task_results/artifacts/chronicle-import/source_handles/context_checkpoints").glob("*"))
    assert set(inputs.values()) <= {path.read_bytes() for path in copied}
    assert {name: (memory / name).read_bytes() for name in inputs} == inputs


def test_busy_legacy_reconciliation_keeps_existing_active_biography(tmp_path):
    import os
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock
    memory = tmp_path / "memory"
    memory.mkdir()
    path = memory / "dialogue_blocks.json"
    path.write_text('[{"content":"Original interpretation."}]', encoding="utf-8")
    store = ChronicleStore(tmp_path)
    active = store.import_legacy()
    authored = store.append_episode("1", "A new authored account.", [], {"kind": "mind"})
    fd = os.open(str(memory / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    file_lock_exclusive_nb(fd)
    try:
        path.write_text('[{"content":"Older writer is still publishing."}]', encoding="utf-8")
        assert store.import_legacy() == active
        assert store.get(authored["id"])["text"] == "A new authored account."
        assert not store.records(kinds=["legacy_reconciliation"])
    finally:
        file_unlock(fd)
        os.close(fd)
    assert store.import_legacy() == active
    assert store.records(kinds=["legacy_reconciliation"])
