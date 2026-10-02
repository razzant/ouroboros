"""Quiet-room source reads reuse physical locators without a global byte cutoff."""
import json
from pathlib import Path

from ouroboros.chronicle_sources import capture_room
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory import Memory


def _line(row, *, escaped=False):
    return (json.dumps(row, ensure_ascii=escaped) + "\n").encode("utf-8")


def _write(root, rows, *, escaped=False):
    path = root / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_line(row, escaped=escaped) for row in rows))
    return path


def test_small_focus_survives_large_foreign_history_and_hot_reads_only_its_bytes(tmp_path, monkeypatch):
    from ouroboros import chronicle_sources
    own = [{"chat_id": 7, "text": f"my line {n}", "direction": "in", "ts": str(n)} for n in range(3)]
    foreign = [{"chat_id": 50 + n % 150, "text": "foreign" * 100, "ts": str(n)} for n in range(2000)]
    _write(tmp_path, own + foreign)
    memory = Memory(tmp_path)
    rows, coverage = capture_room(memory, "7", rendered_chars_budget=2000)
    assert rows == own and coverage["complete"]
    assert coverage["matched_rows"] == 3
    read_bytes = []
    original = chronicle_sources.JsonlChainSnapshot._read
    def counted(self, start, end):
        read_bytes.append(end - start)
        return original(self, start, end)
    monkeypatch.setattr(chronicle_sources.JsonlChainSnapshot, "_read", counted)
    rows, hot = capture_room(memory, "7", rendered_chars_budget=2000)
    assert rows == own and hot["complete"]
    assert sum(read_bytes) == hot["matched_physical_bytes"]
    assert len(read_bytes) == 3
    assert all(summary["new_index_bytes"] == 0 for summary in hot["generations"])


def test_budget_uses_unescaped_rendered_characters_and_reports_exact_omission(tmp_path):
    from ouroboros.chronicle_sources import format_source_row
    row = {"chat_id": 7, "text": "Привет мир" * 100, "direction": "in"}
    _write(tmp_path, [row], escaped=True)
    rows, coverage = capture_room(Memory(tmp_path), "7", rendered_chars_budget=2000)
    assert rows == [row]
    assert coverage["matched_physical_bytes"] > 5000
    assert coverage["matched_rendered_chars"] < 2000
    assert coverage["matched_rendered_chars"] == len(format_source_row(row)) + 2
    omitted, reduced = capture_room(Memory(tmp_path), "7", rendered_chars_budget=10)
    assert omitted == []
    assert reduced["omission"] == "matched_room_exceeds_supplied_budget"
    locator = reduced["row_locators"][0]
    raw = Path(locator["path"]).read_bytes()[locator["start_byte"]:locator["end_byte"]]
    assert json.loads(raw) == row
    assert locator["line"] == 1
    # Full cognitive/review content and falsy recorded status survive compact
    # presentation; only routine transport/UI bookkeeping is by reference.
    row.update(session_id="transport-session", client_surface={"width": 800},
               outcome_final=False, review_projection={"verdict": "FAIL", "finding": "Not approved"})
    view = format_source_row(row)
    assert row["text"] in view and '"outcome_final":false' in view and "Not approved" in view
    assert "transport-session" not in view and '"width"' not in view


def test_late_review_evidence_is_rendered_once_with_its_authority():
    from ouroboros.chronicle_sources import format_source_row

    row = {"type": "acceptance_late_settlement", "text": "Late review arrived.",
           "late_evidence": {"verdict": "FAIL", "finding": "Approval was never given."},
           "status": "completed", "outcome_final": False}
    text = format_source_row(row)
    assert text.count("Approval was never given.") == 1 and "Late review evidence:" in text
    assert '"outcome_final":false' in text and '"status":"completed"' in text
    other = format_source_row({**row, "type": "other_fact"})
    assert other.count("Approval was never given.") == 1
    assert '"late_evidence"' in other


def test_append_and_rotation_read_only_new_source_suffix(tmp_path, monkeypatch):
    from ouroboros import chronicle_sources
    first = {"chat_id": 7, "text": "before rotation"}
    path = _write(tmp_path, [first])
    memory = Memory(tmp_path)
    assert capture_room(memory, "7")[0] == [first]
    archive = tmp_path / "archive"
    archive.mkdir()
    path.rename(archive / "chat_20260101T000000.jsonl")
    second = {"chat_id": 7, "text": "after rotation"}
    path.write_bytes(_line(second))
    third = {"chat_id": 999, "text": "foreign append"}
    with path.open("ab") as stream:
        stream.write(_line(third))
    rows, coverage = capture_room(memory, "7")
    assert rows == [first, second]
    assert coverage["generations"][0]["new_index_bytes"] == 0
    assert coverage["generations"][1]["new_index_bytes"] == len(_line(second)) + len(_line(third))
    assert coverage["complete"]
    read_bytes = []
    original = chronicle_sources.JsonlChainSnapshot._read
    monkeypatch.setattr(chronicle_sources.JsonlChainSnapshot, "_read",
                        lambda self, start, end: (read_bytes.append(end-start), original(self, start, end))[1])
    assert capture_room(memory, "7")[0] == rows
    assert sum(read_bytes) == len(_line(first)) + len(_line(second))
    ChronicleStore(tmp_path).index_path.unlink()
    assert capture_room(memory, "7")[0] == rows


def test_late_project_binding_uses_original_locators_without_reindex(tmp_path, monkeypatch):
    import ouroboros.projects_registry as projects
    import ouroboros.project_dialogue as dialogue
    first = {"chat_id": 1, "task_id": "old-main-task", "text": "original Main words"}
    _write(tmp_path, [first, {"chat_id": 8, "text": "unrelated"}])
    bindings = {}
    monkeypatch.setattr(projects, "all_task_bindings", lambda _root: dict(bindings))
    monkeypatch.setattr(projects, "reserved_project_chat_ids", lambda _root: {777})
    monkeypatch.setattr(dialogue, "source_refs_for_project", lambda *_args: [])
    memory = Memory(tmp_path)
    assert capture_room(memory, "777")[0] == []
    bindings["old-main-task"] = 777
    rows, coverage = capture_room(memory, "777")
    assert rows == [first]
    assert rows[0]["chat_id"] == 1
    assert coverage["generations"][0]["new_index_bytes"] == 0
    assert first not in capture_room(memory, "1")[0]


def test_incomplete_tail_retries_after_completion_without_skipping_row(tmp_path):
    row = {"chat_id": 7, "text": "complete after next write"}
    raw = _line(row)
    path = _write(tmp_path, [])
    path.write_bytes(raw[:20])
    rows, coverage = capture_room(Memory(tmp_path), "7")
    assert rows == [] and not coverage["complete"]
    with path.open("ab") as stream:
        stream.write(raw[20:])
    rows, coverage = capture_room(Memory(tmp_path), "7")
    assert rows == [row] and coverage["complete"]


def test_gap_records_are_visible_in_indexed_cover_and_exact_reads(tmp_path):
    store = ChronicleStore(tmp_path)
    store.publish([{"id": "gap", "kind": "gap", "room_id": "7", "text": "Older source unavailable",
                    "author": {"kind": "host"}, "source_refs": [{"path": "missing"}]}])
    assert store.room_ids() == ["7"]
    assert store.room_records("7")[0]["kind"] == "gap"
    assert store.room_cover("7")[0]["text"] == "Older source unavailable"
    store.revise("gap", "Source recovered; original missing-state fact retained", {"kind": "mind"})
    store.index_path.unlink()
    assert store.room_cover("7")[0]["current_text"].startswith("Source recovered")
    assert store.get("gap")["text"] == "Older source unavailable"


def test_complete_archived_last_record_needs_no_newline(tmp_path):
    row = {"chat_id": 7, "text": "surviving archived last line"}
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "chat_20260101T000000.jsonl").write_bytes(_line(row).rstrip(b"\n"))
    _write(tmp_path, [])
    rows, coverage = capture_room(Memory(tmp_path), "7")
    assert rows == [row] and coverage["complete"]


def test_project_owner_source_reference_matches_without_copying_text_to_index(tmp_path, monkeypatch):
    import ouroboros.projects_registry as projects
    import ouroboros.project_dialogue as dialogue
    row = {"chat_id": 1, "direction": "in", "text": "original instruction", "ts": "2026-01-01", "client_message_id": "msg-a"}
    _write(tmp_path, [row])
    reference = dialogue.build_owner_message_ref(chat_id=1, client_message_id="msg-a", ts="2026-01-01", text=row["text"])
    monkeypatch.setattr(projects, "reserved_project_chat_ids", lambda _root: {777})
    monkeypatch.setattr(projects, "all_task_bindings", lambda _root: {})
    monkeypatch.setattr(dialogue, "source_refs_for_project", lambda *_args: [reference])
    rows, coverage = capture_room(Memory(tmp_path), "777")
    assert rows == [row] and coverage["complete"]


def test_pending_index_reads_only_unrepresented_suffix_and_covered_open_focus(tmp_path, monkeypatch):
    from ouroboros import chronicle_sources, consolidator
    from ouroboros.chronicle_store import source_row_id
    from ouroboros.chronicle_view import capture_chronicle

    closed = [{"chat_id": 1, "text": "retained legacy words " * 200, "ts": str(n)} for n in range(30)]
    represented = {"chat_id": 1, "text": "already authored tail", "task_id": "done"}
    pending = [{"chat_id": 1, "text": "new owner words", "task_id": "open"},
               {"chat_id": 7, "text": "other room open words", "task_id": "other"}]
    path = _write(tmp_path, closed)
    with path.open("ab") as stream:
        stream.write(b"\nnot-json\n" + _line({"chat_id": -2, "text": "excluded A2A"}))
        stream.write(_line(represented) + b"\n" + b"".join(_line(row) for row in pending))
    memory, store = Memory(tmp_path), ChronicleStore(tmp_path)
    store.import_legacy()
    store.publish([], scan_state={"last_consolidated_offset": len(closed),
                                  "chat_log_signature": consolidator._chat_log_signature(path)})
    store.append_episode("1", "Retained tail", [], {"kind": "mind"},
                         metadata={"source_row_ids": [source_row_id(represented)]})
    capture_room(memory, "1", rendered_chars_budget=1)  # Build the existing index once.
    calls, original = [], chronicle_sources.JsonlChainSnapshot._read
    def counted(self, start, end):
        calls.append(end - start)
        return original(self, start, end)
    monkeypatch.setattr(chronicle_sources.JsonlChainSnapshot, "_read", counted)
    def no_full_replay(*_args, **_kwargs):
        raise AssertionError("a warm model start must not reparse the raw generation")
    monkeypatch.setattr(consolidator, "_capture_generation_window", no_full_replay)
    snap = json.loads(capture_chronicle(memory, {"id": "warm", "chat_id": 1}, rendered_chars_budget=1))
    visible = snap["open_focus"] + [row for room in snap["other_open_rooms"] for row in room["rows"]]
    # A source-bound account is not proof that the represented task finished.
    assert visible == [represented, *pending]
    assert sum(calls) == sum(len(_line(row)) for row in [represented, *pending])
    assert any("invalid_chat_row" in gap["kind"] for gap in snap["coverage"]["gaps"])
    assert store.scan_state()["last_consolidated_offset"] == len(closed)


def test_pending_index_preserves_rotated_tail_and_reports_missing_or_torn_generation(tmp_path):
    from ouroboros import consolidator
    from ouroboros.chronicle_sources import capture_pending_rows

    first, tail, later = ({"chat_id": 1, "text": word} for word in ("old covered", "archived tail", "new generation"))
    path = _write(tmp_path, [first, tail])
    memory, store = Memory(tmp_path), ChronicleStore(tmp_path)
    store.import_legacy()
    scan = {"last_consolidated_offset": 1, "chat_log_signature": consolidator._chat_log_signature(path)}
    store.publish([], scan_state=scan)
    capture_room(memory, "1", rendered_chars_budget=1)
    archive = tmp_path / "archive"
    archive.mkdir()
    saved = archive / "chat_20260101T000000.jsonl"
    path.rename(saved)
    path.write_bytes(_line(later) + b'{"text":"unfinished')
    rows, gaps = capture_pending_rows(memory, store, set())
    assert rows == [tail, later]
    assert any(gap["kind"] == "incomplete_chat_line" for gap in gaps)
    assert store.scan_state() == scan
    saved.unlink()
    rows, gaps = capture_pending_rows(memory, store, set())
    assert rows is None and gaps[0]["cause"] == "cursor_generation_missing"
    assert store.scan_state() == scan
