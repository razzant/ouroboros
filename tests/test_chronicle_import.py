"""Legacy dialogue memory enters the chronicle once, by position, with no model call.

The import (``chronicle_import.ensure_activated``) retains each legacy file before
reading it, writes one ``legacy`` record per room section, aligns each block to the
chat stream by position only when the counts meet the old cursor, turns pending
nominations into global marks, and never reads the legacy files again once an
activation exists. Every rule is pinned in both directions: it acts where it must
and stays quiet where it must not.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from ouroboros import chat_chain
from ouroboros import chronicle_import as ci
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.utils import jsonl_generation_signature

REPO = pathlib.Path(__file__).resolve().parents[1]
MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
ARCHIVE = "chat_20260901T010000.jsonl"
COPIES = pathlib.Path("task_results/artifacts/chronicle-import/source_handles/context_checkpoints")
LEGACY_FILES = ("dialogue_blocks.json", "dialogue_meta.json", "dialogue_summary.md")


def _row(i, **extra):
    hour = "00" if i < 4 else "02"
    return {"chat_id": 1, "direction": "in", "ts": f"2026-09-01T{hour}:00:{i:02d}+00:00", "text": f"row {i}", **extra}


def _chat(root, *, live=3, lineage_at=None):
    """Four archived rows (stream 0-3) and ``live`` live rows; an A2A row and a blank line take no position."""
    rows = [_row(i) for i in range(4 + live)]
    if lineage_at is not None:
        rows[lineage_at] = {**rows[lineage_at], "direction": "out", "subagent_task_id": "child-1",
                            "delegation_role": "subagent", "parent_task_id": "root-1"}
    (root / "archive").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    a2a = json.dumps({"chat_id": -5, "direction": "in", "ts": "2026-09-01T00:00:00+00:00", "text": "agent"})
    archived = [json.dumps(rows[0]), a2a, "", *(json.dumps(r) for r in rows[1:4])]
    (root / "archive" / ARCHIVE).write_text("\n".join(archived) + "\n", encoding="utf-8")
    (root / "logs" / "chat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[4:]), encoding="utf-8")
    return rows


def _blocks(sizes=(3, 2)):
    first = {"ts": "2026-09-01T01:00:00+00:00", "type": "summary", "range": "2026-09-01 00:00 - 00:02",
             "message_count": sizes[0], "content": "### Block one",
             "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "I talked with the owner."},
                       {"room_id": "7", "label": "Project seven [chat_id=7]", "message_count": 1,
                        "content": "Project seven started."}]}
    era = {"ts": "2026-09-01T03:00:00+00:00", "type": "era", "range": "2026-09-01 to 2026-09-01",
           "message_count": sizes[1], "content": "### Era", "source_ref": {"kind": "task_source", "sha256": "e" * 64,
                                                                           "path": "source_handles/old.json"},
           "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "An era of Main."}]}
    return [first, era]


def _meta(root, offset=1, **extra):
    signature = jsonl_generation_signature(root / "logs" / "chat.jsonl")
    return {"chat_log_signature": signature, "last_consolidated_offset": offset, **extra}


def _world(root, *, blocks=None, meta=None, lineage_at=None, live=3):
    rows = _chat(root, live=live, lineage_at=lineage_at)
    memory = root / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "dialogue_blocks.json").write_text(json.dumps(_blocks() if blocks is None else blocks), encoding="utf-8")
    meta_value = _meta(root) if meta is None else meta
    if isinstance(meta_value, bytes):
        (memory / "dialogue_meta.json").write_bytes(meta_value)
    else:
        (memory / "dialogue_meta.json").write_text(json.dumps(meta_value), encoding="utf-8")
    return rows


def _shas(root):
    return {name: hashlib.sha256((root / "memory" / name).read_bytes()).hexdigest()
            for name in LEGACY_FILES if (root / "memory" / name).exists()}


def _legacy(store, room=None):
    return [r for r in store.records(room, kinds=["legacy"])]


def _by_type(store, legacy_type):
    return [r for r in _legacy(store) if r["metadata"].get("legacy_type") == legacy_type]


def _pos(root, address):
    return chat_chain.stream_position_of(root, address)


def _copies(root):
    directory = root / COPIES
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


# --- one record per room section, aligned by position ---------------------------------------------

def test_each_room_section_is_one_legacy_record_aligned_to_the_stream_by_position(tmp_path):
    _world(tmp_path)
    before = _shas(tmp_path)
    receipt = ChronicleStore(tmp_path).ensure_activated()
    store = ChronicleStore(tmp_path)
    assert receipt["kind"] == "activation"
    records = {r["id"]: r for r in _legacy(store)}
    assert set(records) == {"legacy-b00-r1", "legacy-b00-r7", "legacy-b01-r1"}
    main = records["legacy-b00-r1"]
    assert main["text"] == "I talked with the owner." and main["room_id"] == "1"
    assert main["author"]["kind"] == "legacy_helper"
    assert main["metadata"]["legacy_block"] == 0 and main["metadata"]["room_message_count"] == 2
    assert main["metadata"]["legacy_written_at"] == "2026-09-01T01:00:00+00:00"
    first, era = main["covers"]["raw_range"], records["legacy-b01-r1"]["covers"]["raw_range"]
    assert first["status"] == "exact" and first["pos"] == [0, 3]
    assert [_pos(tmp_path, first["first"]), _pos(tmp_path, first["last"])] == [0, 2]
    assert first["ts_span"] == {"start": "2026-09-01T00:00:00+00:00", "end": "2026-09-01T00:00:02+00:00",
                                "incomplete": False}
    # The era's range crosses the rotation; both ends resolve by address, not by path.
    assert era["pos"] == [3, 5] and [_pos(tmp_path, era["first"]), _pos(tmp_path, era["last"])] == [3, 4]
    # An era is one record: its nested retelling is addressed, never unfolded into children.
    assert records["legacy-b01-r1"]["metadata"]["legacy_source_ref"]["sha256"] == "e" * 64
    assert len(store.records()) == 4  # three sections and the receipt
    frontier = ci.legacy_frontier(store)
    assert store.scan_state() == {"legacy_frontier": frontier}
    assert frontier["status"] == "exact" and frontier["pos"] == 5 and frontier["offset"] == 1
    assert _pos(tmp_path, frontier["last_covered"]) == 4
    meta = receipt["metadata"]
    assert meta["paid_calls"] == 0 and meta["legacy_files_unchanged"] is True and meta["imported_records"] == 3
    assert meta["chain_end"]["pos"] == 6 and _pos(tmp_path, meta["chain_end"]["address"]) == 6
    for name, filename in (("blocks", "dialogue_blocks.json"), ("meta", "dialogue_meta.json")):
        ref = meta["source_refs"][name]
        assert read_actor_source_bytes(tmp_path, ref["task_id"], ref) == (tmp_path / "memory" / filename).read_bytes()
    assert _shas(tmp_path) == before


@pytest.mark.parametrize("change, exact", [
    (None, True),
    ("count", False),          # the counts no longer meet the cursor
    ("gap", False),            # a room section begins with the gap marker
    ("gap_block", False),      # an old gap block: its content begins with the marker, after whitespace
    ("gap_id", False),         # the old writer's typed facts
    ("gap_type", False),
    ("gap_mentioned", True),   # a retelling that discusses the marker mid-text is not a gap
    ("not_int", False),        # a count that is not an integer
    ("cursor_live_end", False),  # the same counts, a cursor further on
])
def test_raw_range_is_exact_only_when_the_counts_meet_the_cursor_without_gaps(tmp_path, change, exact):
    blocks = _blocks((3, 1) if change == "count" else (3, 2))
    if change == "gap":
        blocks[1]["rooms"][0]["content"] = "[MEMORY GAP] lost hours"
    elif change == "gap_block":
        blocks[1]["content"] = "\n  [MEMORY GAP] Legacy durable discontinuity."
    elif change == "gap_id":
        blocks[1]["gap_id"] = "g1"
    elif change == "gap_type":
        blocks[1]["type"] = "gap"
    elif change == "gap_mentioned":
        blocks[0]["content"] = "### Block one: the owner asked why a [MEMORY GAP] line appears"
        blocks[1]["rooms"][0]["content"] = "I explained the marker: [MEMORY GAP] opens a record of lost rows."
    elif change == "not_int":
        blocks[0]["message_count"] = "3"
    _chat(tmp_path)
    meta = _meta(tmp_path, offset=2 if change == "cursor_live_end" else 1)
    _world(tmp_path, blocks=blocks, meta=meta)
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    ranges = {r["covers"]["raw_range"]["status"] for r in _legacy(store)}
    assert ranges == {"exact" if exact else "unknown"}
    if not exact:
        unknown = {"status": "unknown", "pos": None, "first": None, "last": None, "ts_span": None}
        assert all(r["covers"]["raw_range"] == unknown for r in _legacy(store))
    # Only the block that IS a gap is typed one; mentioning the marker types nothing.
    is_gap = change in ("gap", "gap_block", "gap_id", "gap_type")
    assert {r["id"] for r in _by_type(store, "gap")} == ({"legacy-b01-r1"} if is_gap else set())
    # The frontier follows the readable cursor whatever the blocks say.
    frontier = ci.legacy_frontier(store)
    assert frontier["status"] == "exact" and frontier["pos"] == (6 if change == "cursor_live_end" else 5)


def test_lineage_epoch_is_the_first_row_carrying_delegation_lineage_else_the_chain_end(tmp_path):
    _world(tmp_path, lineage_at=5)
    epoch = ChronicleStore(tmp_path).ensure_activated()["metadata"]["lineage_epoch"]
    assert epoch["pos"] == 5 and epoch["ts"] == "2026-09-01T02:00:05+00:00" and epoch["basis"] == "first_lineage_row"
    assert _pos(tmp_path, epoch["address"]) == 5
    # No row proves that lineage was recorded: every row written so far precedes the epoch.
    plain = tmp_path / "plain"
    rows = _world(plain)
    assert ChronicleStore(plain).ensure_activated()["metadata"]["lineage_epoch"] == {
        "pos": len(rows), "ts": None, "address": None, "basis": "chain_end"}
    # An install with no chat row at activation has no earlier period at all.
    empty = tmp_path / "empty"
    (empty / "logs").mkdir(parents=True)
    assert ChronicleStore(empty).ensure_activated()["metadata"]["lineage_epoch"] is None


# --- idempotent, and never re-read after activation ------------------------------------------------

def test_a_second_activation_leaves_the_journal_byte_identical(tmp_path):
    _world(tmp_path)
    store = ChronicleStore(tmp_path)
    receipt = ci.ensure_activated(store)
    journal, copies = store.log_path.read_bytes(), _copies(tmp_path)
    assert store.ensure_activated() == receipt == ci.ensure_activated(ChronicleStore(tmp_path))
    assert store.log_path.read_bytes() == journal and _copies(tmp_path) == copies


def test_changed_legacy_files_after_activation_write_nothing_but_import_where_none_exists(tmp_path):
    _world(tmp_path)
    store = ChronicleStore(tmp_path)
    receipt = store.ensure_activated()
    journal, copies = store.log_path.read_bytes(), _copies(tmp_path)
    changed = _blocks()
    changed[0]["rooms"][0]["content"] = "Rewritten during a downgrade."
    (tmp_path / "memory" / "dialogue_blocks.json").write_text(json.dumps(changed), encoding="utf-8")
    assert store.ensure_activated() == receipt
    assert store.log_path.read_bytes() == journal and _copies(tmp_path) == copies
    assert store.get("legacy-b00-r1")["text"] == "I talked with the owner."
    # Without an activation the same changed files are imported, with their own copy.
    fresh = tmp_path / "fresh"
    _world(fresh, blocks=changed)
    fresh_store = ChronicleStore(fresh)
    assert fresh_store.ensure_activated()["kind"] == "activation"
    assert fresh_store.get("legacy-b00-r1")["text"] == "Rewritten during a downgrade."
    assert len(_copies(fresh)) == 2


# --- corrupt sources become located gaps beside intact evidence -------------------------------------

def test_an_invalid_block_member_becomes_a_located_gap_and_its_neighbours_stay(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    source = json.dumps([{"content": "First valid memory."}, 17, {"content": "Last valid memory."}]).encode()
    (memory / "dialogue_blocks.json").write_bytes(source)
    store = ChronicleStore(tmp_path)
    assert store.ensure_activated()["kind"] == "activation"
    texts = {r["text"] for r in store.room_records("legacy")}
    assert {"First valid memory.", "Last valid memory."} <= texts
    assert {r["id"] for r in _legacy(store)} >= {"legacy-b00-rlegacy", "legacy-b02-rlegacy"}
    gap = _by_type(store, "gap")
    assert len(gap) == 1 and gap[0]["metadata"]["location"] == "1" and gap[0]["author"]["kind"] == "host"
    assert read_actor_source_bytes(tmp_path, gap[0]["source_refs"][0]["task_id"], gap[0]["source_refs"][0]) == source
    assert {r["covers"]["raw_range"]["status"] for r in _legacy(store)} == {"unknown"}
    # Without an invalid member there is no gap.
    clean = tmp_path / "clean"
    (clean / "memory").mkdir(parents=True)
    (clean / "memory" / "dialogue_blocks.json").write_text(json.dumps([{"content": "Only valid."}]), encoding="utf-8")
    clean_store = ChronicleStore(clean)
    clean_store.ensure_activated()
    assert not _by_type(clean_store, "gap")


@pytest.mark.parametrize("bad", [b"{broken", b"{}", b"null", b"\xff\xfe"])
def test_an_unreadable_blocks_file_is_retained_and_the_flat_summary_still_imports(tmp_path, bad):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "dialogue_blocks.json").write_bytes(bad)
    (memory / "dialogue_summary.md").write_text("A surviving earlier life.", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    receipt = store.ensure_activated()
    ref = receipt["metadata"]["source_refs"]["blocks"]
    assert read_actor_source_bytes(tmp_path, ref["task_id"], ref) == bad
    assert (memory / "dialogue_blocks.json").read_bytes() == bad
    rows = store.room_records("legacy")
    assert [r["metadata"]["legacy_type"] for r in rows] == ["flat", "gap"]
    assert rows[0]["text"] == "A surviving earlier life." and rows[0]["author"]["kind"] == "legacy_helper"
    assert rows[1]["metadata"]["coverage"] == "unknown" and rows[1]["source_refs"]
    assert not list(memory.glob("*.corrupt*"))


def test_a_flat_summary_that_repeats_a_section_is_not_imported_twice(tmp_path):
    _world(tmp_path)
    (tmp_path / "memory" / "dialogue_summary.md").write_text("Project seven started.", encoding="utf-8")
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    assert not _by_type(store, "flat")
    other = tmp_path / "other"
    _world(other)
    (other / "memory" / "dialogue_summary.md").write_text("A different flat life.", encoding="utf-8")
    other_store = ChronicleStore(other)
    other_store.ensure_activated()
    assert [r["text"] for r in _by_type(other_store, "flat")] == ["A different flat life."]


@pytest.mark.parametrize("cursor", [
    b"{broken", b"[]", b'{"last_consolidated_offset":-1}', b'{"last_consolidated_offset":"100"}',
    b'{"last_consolidated_offset":1,"last_consolidated_offset":2}',
    b'{"pending_knowledge_nominations":[{"topic":"no id"}]}',
    "missing_generation", "past_the_rows",
])
def test_an_unreadable_or_unresolvable_cursor_makes_the_frontier_unknown_at_the_chain_end(tmp_path, cursor):
    rows = _chat(tmp_path)
    if cursor == "missing_generation":
        cursor = json.dumps({"chat_log_signature": {"first_line_sha256": "f" * 64}, "last_consolidated_offset": 1})
    elif cursor == "past_the_rows":
        cursor = json.dumps(_meta(tmp_path, offset=9))
    _world(tmp_path, meta=cursor if isinstance(cursor, bytes) else cursor.encode())
    before = _shas(tmp_path)
    store = ChronicleStore(tmp_path)
    receipt = store.ensure_activated()
    frontier = ci.legacy_frontier(store)
    assert frontier["status"] == "unknown" and frontier["pos"] == len(rows) and frontier["reason"]
    assert _pos(tmp_path, frontier["last_covered"]) == len(rows) - 1
    gap = _by_type(store, "cursor_gap")
    assert len(gap) == 1 and gap[0]["metadata"]["chain_end"] == receipt["metadata"]["chain_end"]
    assert receipt["metadata"]["paid_calls"] == 0
    assert {r["covers"]["raw_range"]["status"] for r in _legacy(store) if r["metadata"]["legacy_type"] == "summary"
            } == {"unknown"}
    assert _shas(tmp_path) == before
    assert set(store.scan_state()) == {"legacy_frontier"}


def test_a_readable_cursor_has_no_cursor_gap(tmp_path):
    _world(tmp_path)
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    assert ci.legacy_frontier(store)["status"] == "exact" and not _by_type(store, "cursor_gap")


def test_unknown_cursor_with_an_empty_live_generation_ends_at_the_last_archived_row(tmp_path):
    rows = _world(tmp_path, live=0, meta=b"{broken")
    store = ChronicleStore(tmp_path)
    end = store.ensure_activated()["metadata"]["chain_end"]
    assert end["pos"] == len(rows) - 1 == 3
    assert end["address"]["hint"]["gen"] == jsonl_generation_signature(tmp_path / "archive" / ARCHIVE)[
        "first_line_sha256"]
    assert ci.legacy_frontier(store)["pos"] == 4


def test_a_missing_cursor_beside_legacy_blocks_is_unknown_and_an_empty_install_is_exact_at_zero(tmp_path):
    _world(tmp_path)
    (tmp_path / "memory" / "dialogue_meta.json").unlink()
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    assert ci.legacy_frontier(store)["status"] == "unknown" and _by_type(store, "cursor_gap")
    empty = tmp_path / "empty"
    _chat(empty)
    empty_store = ChronicleStore(empty)
    receipt = empty_store.ensure_activated()
    assert receipt["kind"] == "activation" and receipt["metadata"]["imported_records"] == 0
    assert ci.legacy_frontier(empty_store) == {"status": "exact", "pos": 0, "last_covered": None,
                                               "chat_log_signature": None, "offset": 0}


# --- nominations become marks the mind releases -----------------------------------------------------

def test_pending_nominations_become_global_marks_that_the_mind_releases(tmp_path):
    _chat(tmp_path)
    nominations = [{"id": "a" * 64 + ":0:1", "scope": "global", "topic": "overview", "reason": "revision_required"},
                   {"id": "b" * 64 + ":2:0", "scope": "project", "topic": "deploys", "reason": "revision_conflict"}]
    last = {"entry_id": "c" * 64, "failed": 1, "total": 1}
    _world(tmp_path, meta=_meta(tmp_path, pending_knowledge_nominations=nominations,
                                last_unpublished_nominations=last))
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    marks = store.active_marks("1")
    assert marks == store.active_marks("some-other-room") and len(marks) == 3
    assert all(m["scope"] == "global" and m["author"]["kind"] == "legacy_helper" and m["room_id"] == "legacy"
               for m in marks)
    locations = [m["target_ref"]["location"] for m in marks]
    assert locations == ["pending_knowledge_nominations/0", "pending_knowledge_nominations/1",
                         "last_unpublished_nominations"]
    assert "overview" in marks[0]["text"] and "a" * 64 in marks[0]["text"] and ":0:1" not in marks[0]["text"]
    assert "c" * 64 in marks[2]["text"]
    assert set(store.scan_state()) == {"legacy_frontier"}
    assert "pending_knowledge_nominations" not in json.dumps(store.scan_state())
    released = store.release_mark(marks[0]["id"], MIND, "Published the overview revision by hand.")
    assert released.ok
    assert [m["id"] for m in store.active_marks("1")] == [marks[1]["id"], marks[2]["id"]]


def test_a_fully_published_last_batch_leaves_no_mark(tmp_path):
    _chat(tmp_path)
    _world(tmp_path, meta=_meta(tmp_path, last_unpublished_nominations={"entry_id": "c" * 64, "failed": 0,
                                                                         "total": 2}))
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    assert store.active_marks("1") == []


# --- one importer at a time, and failures keep the copies -------------------------------------------

def test_a_busy_legacy_lock_defers_the_import_until_it_is_free(tmp_path):
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock

    _world(tmp_path)
    store = ChronicleStore(tmp_path)
    fd = os.open(str(tmp_path / "memory" / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    file_lock_exclusive_nb(fd)
    try:
        assert store.ensure_activated() == {"kind": "import_pending", "reason": "legacy_memory_lock_busy"}
        assert store.activation() is None and not store.log_path.exists()
    finally:
        file_unlock(fd)
        os.close(fd)
    assert store.ensure_activated()["kind"] == "activation"


def test_a_failed_publication_still_retains_every_readable_source(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    memory.mkdir()
    inputs = {"dialogue_blocks.json": b"{broken", "dialogue_meta.json": b"[]", "dialogue_summary.md": b"Valid flat."}
    for name, data in inputs.items():
        (memory / name).write_bytes(data)
    store = ChronicleStore(tmp_path)

    def reject(*_args, **_kwargs):
        raise OSError("publication unavailable")

    monkeypatch.setattr(store, "publish", reject)
    with pytest.raises(OSError, match="publication unavailable"):
        store.ensure_activated()
    assert store.activation() is None
    assert set(inputs.values()) <= {(tmp_path / COPIES / name).read_bytes() for name in _copies(tmp_path)}
    assert {name: (memory / name).read_bytes() for name in inputs} == inputs


def test_a_refused_publication_is_typed_and_publishes_nothing(tmp_path):
    _world(tmp_path)
    store = ChronicleStore(tmp_path)
    squatter = {"id": "legacy-b00-r1", "kind": "legacy", "room_id": "1", "text": "someone else's",
                "author": {"kind": "legacy_helper"}}
    assert store.publish([squatter]).ok
    journal = store.log_path.read_bytes()
    refused = store.ensure_activated()
    assert refused["kind"] == "import_refused" and refused["reason"] == "invalid"
    assert refused["conflict_ids"] == ["legacy-b00-r1"]
    assert store.activation() is None and store.log_path.read_bytes() == journal


def test_one_room_twice_in_a_block_keeps_both_sections(tmp_path):
    blocks = _blocks()
    blocks[0]["rooms"].append({"room_id": "1", "label": "Main", "message_count": 0, "content": "Main again."})
    _world(tmp_path, blocks=blocks)
    store = ChronicleStore(tmp_path)
    assert store.ensure_activated()["kind"] == "activation"
    assert {r["id"]: r["text"] for r in _legacy(store, "1")} == {
        "legacy-b00-r1": "I talked with the owner.", "legacy-b00-r1-2": "Main again.",
        "legacy-b01-r1": "An era of Main."}


def test_room_sections_lives_in_the_import_and_never_guesses_a_room():
    """The old writer's ``room_sections`` moved here with its pseudo-room when the writer went:
    typed ``rooms`` become one section each; a block without them is one legacy mixed section."""
    typed = {"message_count": 3, "content": "both", "rooms": [
        {"room_id": "1", "label": "Main", "message_count": 2, "content": "Main part."},
        {"room_id": "1500", "message_count": 1, "content": "Project part."}]}
    assert ci.room_sections(typed) == [
        {"room_id": "1", "label": "Main", "message_count": 2, "content": "Main part."},
        {"room_id": "1500", "label": "1500", "message_count": 1, "content": "Project part."}]
    for untyped in ({"message_count": 4, "content": "An era."},
                    {"message_count": 4, "content": "An era.", "rooms": [{"room_id": 1, "content": "bad id"}]}):
        assert ci.room_sections(untyped) == [{"room_id": ci.LEGACY_ROOM_ID, "label": ci.LEGACY_ROOM_LABEL,
                                              "message_count": 4, "content": "An era."}]
    assert not (REPO / "ouroboros" / "room_consolidation.py").exists()


# --- no model, no model client ----------------------------------------------------------------------

def test_the_import_loads_no_model_client(tmp_path):
    _world(tmp_path)
    probe = textwrap.dedent(f"""
        import json, sys
        from ouroboros.chronicle_store import ChronicleStore
        receipt = ChronicleStore({str(tmp_path)!r}).ensure_activated()
        loaded = sorted(name for name in sys.modules if name.startswith("ouroboros.llm"))
        import ouroboros.llm  # the detector sees the client once something does load it
        after = "ouroboros.llm" in sys.modules
        print(json.dumps({{"kind": receipt["kind"], "loaded": loaded, "after": after}}))
    """)
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"kind": "activation", "loaded": [], "after": True}
