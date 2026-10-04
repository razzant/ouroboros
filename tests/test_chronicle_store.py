"""Chronicle store: append-only authority, set-based page coverage, folds, heads and typed refusals."""
import ast
import json
import pathlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.chronicle_store import ChronicleStore, PublishResult

REPO = pathlib.Path(__file__).resolve().parent.parent
MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
LIGHT = {"kind": "helper", "route": "configured-light"}
LEGACY = {"kind": "legacy_helper", "writer": "old_consolidator"}


def covers(*refs, at=0):
    return {"mode": "range", "rows": list(refs), "stream_span": [at, at + max(len(refs) - 1, 0)]}


def note(store, text="kept", room="r", task="t1"):
    result = store.write_note(room_id=room, task_id=task, text=text, author=MIND)
    assert result.ok, result
    return result.record


def page(store, *refs, room="r", at=0, author=MIND, text=None, **kwargs):
    return store.publish_page(room_id=room, text=text or f"page over {refs}", covers=covers(*refs, at=at),
                              author=author, **kwargs)


def body(record):
    return {k: v for k, v in record.items() if k != "sequence"}


def legacy(store, block, room="r"):
    result = store.publish([{"id": f"legacy-b{block:02d}-r{room}", "kind": "legacy", "room_id": room,
                             "text": f"retelling {block}", "author": LEGACY,
                             "metadata": {"legacy_type": "era", "legacy_block": block}}])
    assert result.ok, result
    return result.record


# --- the journal is the authority (carried over from the candidate store) ---------------------------

def test_records_and_scan_publish_atomically_and_replay_after_index_loss(tmp_path):
    store = ChronicleStore(tmp_path)
    original = note(store, "first")
    saved = store.publish([{"id": "next", "kind": "legacy", "room_id": "r", "text": "next", "author": LEGACY}],
                          scan_state={"last_offset": 8})
    assert saved.ok and saved.reason == "saved" and saved.record["sequence"] == original["sequence"] + 1
    refused = store.publish([{**body(original), "text": "collision"}], scan_state={"last_offset": 999})
    assert (refused.ok, refused.reason, refused.conflict_ids) == (False, "invalid", (original["id"],))
    assert "identity collision" in refused.detail and refused.record is None
    store.index_path.unlink()
    assert store.get(original["id"])["text"] == "first"
    assert store.get("next")["text"] == "next"
    assert store.scan_state() == {"last_offset": 8}


def test_failed_index_commit_replays_logged_transaction(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.records()
    original = store._project

    def fail(db, tx):
        raise OSError("simulated interruption after durable append")

    monkeypatch.setattr(store, "_project", fail)
    with pytest.raises(OSError):
        store.publish([{"id": "durable", "kind": "note", "room_id": "r", "task_id": "t1", "text": "kept",
                        "author": MIND}], scan_state={"offset": 7})
    monkeypatch.setattr(store, "_project", original)
    assert store.get("durable")["text"] == "kept"
    assert store.scan_state() == {"offset": 7}


def test_duplicate_publication_replays_and_collision_is_refused(tmp_path):
    store = ChronicleStore(tmp_path)
    record = {"id": "stable", "kind": "note", "room_id": "r", "task_id": "t1", "text": "once", "author": MIND}
    first = store.publish([record])
    before = store.log_path.read_bytes()
    again = store.publish([record])
    assert (first.reason, again.reason) == ("saved", "unchanged") and again.record == first.record
    assert store.log_path.read_bytes() == before
    refused = store.publish([{**record, "text": "not once"}])
    assert refused.reason == "invalid" and "collision" in refused.detail
    assert store.get("stable")["text"] == "once"
    assert store.log_path.read_bytes() == before


def test_concurrent_writers_and_index_corruption_preserve_records(tmp_path):
    def write(n):
        return note(ChronicleStore(tmp_path), str(n))
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(write, range(16)))
    store = ChronicleStore(tmp_path)
    assert len(store.room_records("r")) == 16
    store.index_path.write_bytes(b"not a sqlite database")
    assert {r["id"] for r in store.room_records("r")} == {r["id"] for r in records}


def test_marks_release_is_explicit_and_room_projection_does_not_change_sources(tmp_path):
    store = ChronicleStore(tmp_path)
    source = {"kind": "raw_chat", "id": "original-main-row", "chat_id": 1}
    mark = store.mark(source, "Keep this decision", MIND, room_id="r", scope="global", quote="exact").record
    assert store.active_marks("other")[0]["target_ref"] == source
    refused = store.release_mark(mark["id"], MIND, "")
    assert refused.reason == "invalid" and store.active_marks("r")
    assert store.release_mark(mark["id"], MIND, "Superseded by owner's later choice").ok
    assert store.active_marks("r") == []
    again = store.release_mark(mark["id"], MIND, "Released twice")
    assert (again.reason, again.conflict_ids) == ("target_missing", (mark["id"],))
    assert store.get(mark["id"])["target_ref"] == source


def test_active_marks_without_a_room_are_every_rooms_and_with_one_its_own_and_the_global_ones(tmp_path):
    store = ChronicleStore(tmp_path)
    source = {"kind": "task", "task_id": "t1"}
    room_a = store.mark(source, "in room a", MIND, room_id="a").record["id"]
    room_b = store.mark(source, "in room b", MIND, room_id="b").record["id"]
    shared = store.mark(source, "for all", MIND, room_id="a", scope="global").record["id"]
    assert [mark["id"] for mark in store.active_marks()] == [room_a, room_b, shared]
    assert [mark["id"] for mark in store.active_marks(None)] == [room_a, room_b, shared]
    assert [mark["id"] for mark in store.active_marks("b")] == [room_b, shared]  # a room: its own and the global
    assert [mark["id"] for mark in store.active_marks("b", include_global=False)] == [room_b]
    assert store.release_mark(room_b, MIND, "done").ok
    assert [mark["id"] for mark in store.active_marks()] == [room_a, shared]


def test_hot_lookup_reads_only_unindexed_log_suffix(tmp_path, monkeypatch):
    store = ChronicleStore(tmp_path)
    store.publish([{"id": "x", "kind": "note", "room_id": "r", "task_id": "t1", "text": "hello", "author": MIND}])
    original = store._project

    def fail_on_replay(db, tx):
        pytest.fail("hot lookup replayed an already indexed transaction")

    monkeypatch.setattr(store, "_project", fail_on_replay)
    assert store.get("x")["text"] == "hello"
    monkeypatch.setattr(store, "_project", original)


def test_batch_identity_collision_cannot_poison_log(tmp_path):
    store = ChronicleStore(tmp_path)
    refused = store.publish([{"id": "same", "kind": "legacy", "room_id": "r", "text": "a", "author": LEGACY},
                             {"id": "same", "kind": "legacy", "room_id": "r", "text": "b", "author": LEGACY}])
    assert refused.reason == "invalid" and "collision" in refused.detail
    assert not store.log_path.exists()
    assert store.records() == []


def test_page_rows_use_the_canonical_source_identity_of_the_raw_row(tmp_path):
    from ouroboros.chat_chain import source_row_id
    store = ChronicleStore(tmp_path)
    row = {"text": "Привет", "chat_id": 1, "task_id": "t"}
    assert page(store, source_row_id(row)).ok
    reordered = page(store, source_row_id(dict(reversed(list(row.items())))), at=5)
    assert reordered.reason == "already_sealed"
    assert page(store, source_row_id({**row, "text": "Привет!"}), at=6).ok


@pytest.mark.serial
def test_two_processes_publish_without_losing_frontiers(tmp_path):
    import subprocess
    import sys
    script = """
import sys
from ouroboros.chronicle_store import ChronicleStore
store = ChronicleStore(sys.argv[1])
room = sys.argv[2]
author = {'kind': 'mind', 'task_id': 't-' + room, 'focus': {'role': 'root'}}
for i in range(8):
    assert store.write_note(room_id=room, task_id='t-' + room, text=str(i), author=author).ok
"""
    children = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), room], cwd=str(REPO),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for room in ("a", "b")]
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, (stdout, stderr)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
    store = ChronicleStore(tmp_path)
    assert len(store.room_records("a")) == len(store.room_records("b")) == 8
    assert all([row["text"] for row in store.room_records(room)] == [str(i) for i in range(8)]
               for room in ("a", "b"))


def test_torn_transaction_keeps_prior_memory_gap_and_allows_new_publication(tmp_path):
    store = ChronicleStore(tmp_path)
    first = store.publish([{"kind": "note", "room_id": "1", "task_id": "t1", "text": "Completed memory",
                            "author": MIND}], scan_state={"offset": 3}).record
    original_prefix = store.log_path.read_bytes()
    torn = b'{"kind":"transaction","records":[{"id":"lost","kind":"note"'
    with store.log_path.open("ab") as stream:
        stream.write(torn)
    assert store.get(first["id"])["text"] == "Completed memory"
    gap = store.room_records("legacy")[0]
    assert gap["kind"] == "gap" and gap["author"]["kind"] == "host"
    source = gap["source_refs"][0]
    assert store.log_path.read_bytes()[source["start_byte"]:source["end_byte"]] == torn
    assert store.scan_state() == {"offset": 3}
    assert store.get("lost") is None
    second = store.publish([{"kind": "note", "room_id": "1", "task_id": "t1", "text": "New memory after interruption",
                             "author": MIND}], scan_state={"offset": 4}).record
    assert store.log_path.read_bytes().startswith(original_prefix + torn)
    assert b"\n" in store.log_path.read_bytes()[source["end_byte"]:]
    before = store.room_records("legacy")
    store.index_path.unlink()
    assert store.get(first["id"])["text"] == "Completed memory"
    assert store.get(second["id"])["text"] == "New memory after interruption"
    assert store.room_records("legacy") == before
    assert store.scan_state() == {"offset": 4}
    assert store.log_path.read_bytes().startswith(original_prefix + torn)


def test_invalid_complete_transaction_cannot_publish_its_first_record_or_scan(tmp_path):
    store = ChronicleStore(tmp_path)
    assert store.publish([{"kind": "note", "room_id": "1", "task_id": "t1", "text": "Committed", "author": MIND}],
                         scan_state={"offset": 5}).ok
    invalid = {"kind": "transaction", "records": [
        {"id": "phantom", "kind": "note", "room_id": "1", "task_id": "t1", "text": "never committed",
         "author": MIND}, {}], "scan_state": {"offset": 999}}
    with store.log_path.open("ab") as stream:
        stream.write((json.dumps(invalid) + "\n").encode())
    assert store.get("phantom") is None
    assert store.scan_state() == {"offset": 5}
    assert store.room_records("legacy")[0]["kind"] == "gap"


def test_room_records_are_story_records_paged_by_sequence(tmp_path):
    store = ChronicleStore(tmp_path)
    notes = [note(store, f"Meaning {n}", room="1") for n in range(3)]
    store.mark({"kind": "chronicle", "id": notes[0]["id"]}, "Matters", MIND, room_id="1")
    assert store.correct(notes[0]["id"], "Meaning 0, corrected", MIND).ok
    draft = page(store, "row-x", room="1", author=LIGHT).record
    assert store.decide(draft["id"], False, MIND, "Not what happened").ok
    first = store.room_records("1")
    assert [r["id"] for r in first] == [r["id"] for r in notes]
    assert first[0]["text"] == "Meaning 0" and first[0]["current_text"] == "Meaning 0, corrected"
    second = store.room_records("1", after_seq=first[1]["sequence"])
    assert [r["id"] for r in second] == [notes[2]["id"]]


# --- rewritten: only the mind corrects; a rejected draft stops acting ------------------------------

def test_correction_is_the_minds_and_keeps_the_original(tmp_path):
    store = ChronicleStore(tmp_path)
    original = note(store, "I chose X")
    before = store.log_path.read_bytes()
    for author in (LIGHT, {"kind": "host", "operation": "x"}):
        refused = store.correct(original["id"], "The mind chose Y", author)
        assert (refused.ok, refused.reason) == (False, "invalid")
    assert store.log_path.read_bytes() == before
    fixed = store.correct(original["id"], "The mind chose Y", MIND)
    assert fixed.ok and fixed.current_revision == fixed.record["id"]
    row = store.room_records("r")[0]
    assert (row["text"], row["current_text"], row["current_author"]) == ("I chose X", "The mind chose Y", MIND)
    assert row["revision"] == fixed.record["id"]
    assert store.get(original["id"])["text"] == "I chose X"
    # A helper is refused as such, not asked for a revision; the raw publish path keeps the same rule.
    assert store.correct(original["id"], "Light's view", LIGHT).reason == "invalid"
    raw = store.publish([{"kind": "correction", "room_id": "r", "target_id": original["id"], "text": "Light's view",
                          "author": LIGHT}])
    assert raw.reason == "invalid" and store.room_records("r")[0]["current_text"] == "The mind chose Y"


def test_rejected_light_draft_stops_acting_and_reopens_its_rows(tmp_path):
    store = ChronicleStore(tmp_path)
    draft = page(store, "a", "b", author=LIGHT, text="Light's retelling").record
    assert store.room_records("r")[0]["status"] == "draft"
    assert store.sealed_row_refs("r") == {"a", "b"}
    blocked = page(store, "b", "c", at=1)
    assert (blocked.reason, blocked.conflict_ids) == ("already_sealed", (draft["id"],))
    assert store.decide(draft["id"], False, LIGHT, "helpers do not decide").reason == "invalid"
    assert store.decide(draft["id"], False, MIND, " ").reason == "invalid"
    assert store.decide(draft["id"], False, MIND, "Not source-grounded").ok
    assert store.room_records("r") == [] and store.sealed_row_refs("r") == set()
    assert store.get(draft["id"])["text"] == "Light's retelling"
    assert store.decide(draft["id"], True, MIND, "changed my mind").reason == "invalid"
    mine = page(store, "a", "b", text="My page").record
    assert store.decide(mine["id"], True, MIND, "mine").reason == "invalid"
    accepted = page(store, "d", author=LIGHT, at=3).record
    assert store.decide(accepted["id"], True, MIND, "Matches the rows").ok
    statuses = {r["id"]: r["status"] for r in store.room_records("r")}
    assert statuses == {mine["id"]: "final", accepted["id"]: "accepted"}
    assert store.sealed_row_refs("r") == {"a", "b", "d"}


# --- new: coverage, folds, heads, quotes, marks, rebuild -------------------------------------------

def test_corrections_need_the_current_revision_and_refusals_carry_no_text(tmp_path):
    store = ChronicleStore(tmp_path)
    base = note(store, "original")
    first = store.correct(base["id"], "correction 1", MIND)
    assert first.ok
    required = store.correct(base["id"], "correction 2", MIND)
    assert (required.reason, required.current_revision, required.record) == (
        "revision_required", first.record["id"], None)
    assert "correction 1" not in repr(required)
    stale = store.correct(base["id"], "correction 2", MIND, expected_revision=base["id"])
    assert (stale.reason, stale.current_revision) == ("revision_conflict", first.record["id"])
    assert "correction 1" not in repr(stale)
    assert store.correct(base["id"], "correction 2", MIND, expected_revision=first.record["id"]).ok
    assert store.room_records("r")[0]["current_text"] == "correction 2"
    assert store.correct("no-such-record", "x", MIND).reason == "target_missing"


def test_a_page_seals_a_set_so_interleaved_task_rows_stay_open(tmp_path):
    store = ChronicleStore(tmp_path)
    t1 = store.publish_page(room_id="r", text="Task T1", covers={"mode": "tasks", "rows": ["r1", "r3", "r5"],
                                                                  "stream_span": [0, 4]}, author=MIND)
    assert t1.ok and store.sealed_row_refs("r") == {"r1", "r3", "r5"}
    t2 = store.publish_page(room_id="r", text="Task T2", covers={"mode": "tasks", "rows": ["r2", "r4"],
                                                                  "stream_span": [1, 3]}, author=MIND)
    assert t2.ok and store.sealed_row_refs("r") == {"r1", "r2", "r3", "r4", "r5"}
    overlap = page(store, "r4", "r6", at=3)
    assert (overlap.reason, overlap.conflict_ids, overlap.record) == ("already_sealed", (t2.record["id"],), None)
    both = page(store, "r3", "r2", at=1)
    assert both.conflict_ids == (t1.record["id"], t2.record["id"])
    assert page(store, "r1", room="other").ok


def test_a_note_is_covered_by_reference_once(tmp_path):
    store = ChronicleStore(tmp_path)
    kept = note(store, "Open question for later")
    ref = "note:" + kept["id"]
    assert page(store, "row-1", ref).ok
    assert ref in store.sealed_row_refs("r")
    again = page(store, ref, at=9)
    assert again.reason == "already_sealed"
    assert store.write_note(room_id="r", task_id="t1", text="x", author=LIGHT).reason == "invalid"
    assert store.write_note(room_id="r", task_id="", text="x", author=MIND).reason == "invalid"


def test_parts_fold_adjacent_records_of_one_level_once(tmp_path):
    store = ChronicleStore(tmp_path)
    p10 = page(store, "a10", at=10).record
    p20 = page(store, "a20", at=20).record
    p0 = page(store, "a0", at=0).record
    head = store.room_head("r")
    gap_in_order = store.publish_part(room_id="r", text="0 and 20", member_ids=[p0["id"], p20["id"]], author=MIND,
                                      expected_sequence=head)
    assert gap_in_order.reason == "invalid" and "adjacent" in gap_in_order.detail
    part = store.publish_part(room_id="r", text="0 and 10", member_ids=[p10["id"], p0["id"]], author=MIND,
                              expected_sequence=head)
    assert part.ok and part.record["covers"]["member_ids"] == [p0["id"], p10["id"]]
    assert part.record["covers"]["stream_span"] == [0, 10]
    refold = store.publish_part(room_id="r", text="10 and 20", member_ids=[p10["id"], p20["id"]], author=MIND,
                                expected_sequence=store.room_head("r"))
    assert (refold.reason, refold.conflict_ids) == ("already_folded", (part.record["id"],))
    kinds = store.publish_part(room_id="r", text="mixed", member_ids=[p20["id"], note(store)["id"]], author=MIND,
                               expected_sequence=store.room_head("r"))
    assert kinds.reason == "invalid"
    assert {r["id"]: r["folded_into"] for r in store.pages_of_room("r") if r["kind"] == "page"} == {
        p0["id"]: part.record["id"], p10["id"]: part.record["id"], p20["id"]: None}
    old = [legacy(store, block) for block in (2, 0, 1)]
    from_legacy = store.publish_part(room_id="r", text="blocks 0-1", member_ids=[old[1]["id"], old[2]["id"]],
                                     author=MIND, expected_sequence=store.room_head("r"))
    assert from_legacy.ok
    second = store.publish_part(room_id="r", text="20 alone", member_ids=[p20["id"]], author=MIND,
                                expected_sequence=store.room_head("r"))
    assert second.ok
    of_parts = store.publish_part(room_id="r", text="story so far", member_ids=[part.record["id"], second.record["id"]],
                                  author=MIND, expected_sequence=store.room_head("r"))
    assert of_parts.ok
    assert store.publish_part(room_id="r", text="again", member_ids=[second.record["id"]], author=MIND,
                              expected_sequence=store.room_head("r")).reason == "already_folded"


def test_rejected_drafts_cannot_be_members_and_a_rejected_part_unfolds(tmp_path):
    store = ChronicleStore(tmp_path)
    mine = page(store, "a", at=0).record
    draft = page(store, "b", at=1, author=LIGHT).record
    assert store.decide(draft["id"], False, MIND, "wrong").ok
    with_rejected = store.publish_part(room_id="r", text="x", member_ids=[mine["id"], draft["id"]], author=MIND,
                                       expected_sequence=store.room_head("r"))
    assert with_rejected.reason == "invalid" and "rejected" in with_rejected.detail
    other = page(store, "c", at=2).record
    helper_part = store.publish_part(room_id="r", text="Light's fold", member_ids=[mine["id"], other["id"]],
                                     author=LIGHT, expected_sequence=store.room_head("r"))
    assert helper_part.ok and store.room_records("r")[-1]["status"] == "draft"
    assert {r["folded_into"] for r in store.pages_of_room("r") if r["kind"] == "page"} == {helper_part.record["id"]}
    assert store.decide(helper_part.record["id"], False, MIND, "loses the owner's words").ok
    refolded = store.publish_part(room_id="r", text="My fold", member_ids=[mine["id"], other["id"]], author=MIND,
                                  expected_sequence=store.room_head("r"))
    assert refolded.ok


def test_a_folded_draft_is_rejected_only_after_its_part_and_an_unfolded_one_at_once(tmp_path):
    store = ChronicleStore(tmp_path)
    first = page(store, "r1", "r2", at=0, author=LIGHT).record
    second = page(store, "r3", at=2, author=LIGHT).record
    loose = page(store, "r9", at=9, author=LIGHT).record
    # An unfolded draft is rejected at once and its rows are open again.
    assert store.decide(loose["id"], False, MIND, "wrong arc").ok
    assert store.sealed_row_refs("r") == {"r1", "r2", "r3"}
    fold = store.publish_part(room_id="r", text="Light's fold", member_ids=[first["id"], second["id"]], author=LIGHT,
                              expected_sequence=store.room_head("r"))
    assert fold.ok
    # A folded draft would stop acting inside an acting part: refused with the part's id, rows still sealed.
    folded = store.decide(first["id"], False, MIND, "loses the owner's words")
    assert (folded.ok, folded.reason, folded.conflict_ids) == (False, "already_folded", (fold.record["id"],))
    assert store.sealed_row_refs("r") == {"r1", "r2", "r3"}
    assert store.publish_page(room_id="r", text="mine", covers=covers("r1", "r2"), author=MIND).reason == "already_sealed"
    # Rejecting the part first unfolds it; then the draft is rejected and its rows reopen.
    assert store.decide(fold.record["id"], False, MIND, "folds a wrong draft").ok
    assert store.decide(first["id"], False, MIND, "loses the owner's words").ok
    assert store.sealed_row_refs("r") == {"r3"}
    assert store.publish_page(room_id="r", text="mine", covers=covers("r1", "r2"), author=MIND).ok
    # Accepting a folded draft unfolds nothing and passes.
    mine = store.publish_part(room_id="r", text="my fold", member_ids=[second["id"]], author=MIND,
                              expected_sequence=store.room_head("r"))
    assert mine.ok and store.decide(second["id"], True, MIND, "faithful").ok


def test_a_part_carries_its_members_period_from_pages_and_from_legacy_raw_ranges(tmp_path):
    store = ChronicleStore(tmp_path)

    def span(start, end):
        return {"start": f"2026-09-{start:02d}T00:00:00+00:00", "end": f"2026-09-{end:02d}T00:00:00+00:00",
                "incomplete": False}

    def section(block, raw_span, status="exact"):
        raw = {"status": status, "pos": [block * 10, block * 10 + 10] if raw_span else None, "first": None,
               "last": None, "ts_span": raw_span}
        result = store.publish([{"id": f"legacy-b{block:02d}-rr", "kind": "legacy", "room_id": "r",
                                 "text": f"retelling {block}", "author": LEGACY, "covers": {"room_id": "r", "raw_range": raw},
                                 "metadata": {"legacy_type": "summary", "legacy_block": block}}])
        assert result.ok, result
        return result.record["id"]

    # Legacy sections keep their period inside raw_range; the part reads it there.
    blocks = [section(0, span(1, 2)), section(1, span(3, 4))]
    of_legacy = store.publish_part(room_id="r", text="blocks 0-1", member_ids=blocks, author=MIND,
                                   expected_sequence=store.room_head("r"))
    assert of_legacy.ok and of_legacy.record["covers"]["stream_span"] == [0, 20]
    assert of_legacy.record["covers"]["ts_span"] == {"start": "2026-09-01T00:00:00+00:00",
                                                     "end": "2026-09-04T00:00:00+00:00", "incomplete": False}
    # A section whose range is unknown leaves the part's period incomplete, never invented.
    unknown = [section(2, span(5, 6)), section(3, None, status="unknown")]
    partial = store.publish_part(room_id="r", text="blocks 2-3", member_ids=unknown, author=MIND,
                                 expected_sequence=store.room_head("r"))
    assert partial.ok and partial.record["covers"]["ts_span"] == {
        "start": "2026-09-05T00:00:00+00:00", "end": "2026-09-06T00:00:00+00:00", "incomplete": True}
    # Pages keep their period in covers, as before.
    pages = [store.publish_page(room_id="r", text=f"p{n}", covers={**covers(f"a{n}", at=40 + n), "ts_span": span(n, n)},
                                author=MIND).record["id"] for n in (7, 8)]
    of_pages = store.publish_part(room_id="r", text="pages", member_ids=pages, author=MIND,
                                  expected_sequence=store.room_head("r"))
    assert of_pages.ok and of_pages.record["covers"]["ts_span"] == {
        "start": "2026-09-07T00:00:00+00:00", "end": "2026-09-08T00:00:00+00:00", "incomplete": False}


def test_room_head_guards_publication_without_a_lock_across_reasoning(tmp_path):
    store = ChronicleStore(tmp_path)
    seen = store.room_head("r")
    assert seen == 0
    b = page(store, "b1", text="Writer B's page")
    assert b.ok and b.current_head == b.record["sequence"] == store.room_head("r")
    stale = page(store, "a1", at=5, expected_sequence=seen)
    assert (stale.reason, stale.current_head, stale.conflict_ids, stale.record) == (
        "revision_conflict", b.record["sequence"], (b.record["id"],), None)
    assert "Writer B" not in repr(stale)
    fresh = page(store, "a1", at=5, expected_sequence=stale.current_head)
    assert fresh.ok and fresh.current_head == fresh.record["sequence"]
    assert page(store, "c1", at=7).ok
    head = store.room_head("r")
    no_base = store.publish_part(room_id="r", text="fold", member_ids=[b.record["id"]], author=MIND,
                                 expected_sequence=None)
    assert (no_base.reason, no_base.current_head) == ("revision_required", head)
    old_base = store.publish_part(room_id="r", text="fold", member_ids=[b.record["id"]], author=MIND,
                                  expected_sequence=seen)
    assert old_base.reason == "revision_conflict" and old_base.current_head == head
    stale_fix = store.correct(b.record["id"], "B, corrected", MIND, expected_sequence=seen)
    assert stale_fix.reason == "revision_conflict" and stale_fix.current_head == head
    assert store.correct(b.record["id"], "B, corrected", MIND, expected_sequence=head).ok
    assert store.publish_part(room_id="r", text="fold", member_ids=[b.record["id"]], author=MIND,
                              expected_sequence=store.room_head("r")).ok


def test_marks_do_not_move_the_head_but_member_corrections_and_decisions_do(tmp_path):
    store = ChronicleStore(tmp_path)
    member = page(store, "m1").record
    head = store.room_head("r")
    mark = store.mark({"kind": "chronicle", "id": member["id"]}, "Remember", MIND, room_id="r").record
    assert store.set_mark_view(mark["id"], "meaning", MIND, "Keep only the meaning").ok
    assert store.release_mark(mark["id"], MIND, "Done").ok
    assert store.room_head("r") == head
    assert page(store, "m2", at=1, expected_sequence=head).ok
    head = store.room_head("r")
    fix = store.correct(member["id"], "Member, corrected", MIND)
    assert fix.ok and fix.current_head == store.room_head("r") > head
    moved = page(store, "m3", at=2, expected_sequence=head)
    assert moved.conflict_ids == (fix.record["id"],)
    head = store.room_head("r")
    draft = page(store, "m4", at=3, author=LIGHT).record
    head_after_draft = store.room_head("r")
    decision = store.decide(draft["id"], True, MIND, "Right")
    assert decision.ok and store.room_head("r") > head_after_draft > head
    assert page(store, "m5", at=4, expected_sequence=head_after_draft).conflict_ids == (decision.record["id"],)


def test_quotes_are_verified_through_the_injected_resolver_before_the_lock(tmp_path):
    store = ChronicleStore(tmp_path)
    rows = {"row:1@t1#aaaaaaaaaaaa": ("The owner said: ship it on Friday", "human"),
            "row:1@t2#bbbbbbbbbbbb": ("I will ship it on Friday", "ouroboros")}

    def resolver(address):
        assert not store.lock_path.exists(), "the resolver read chat rows under the publication lock"
        return rows.get(address)

    owner = {"address": "row:1@t1#aaaaaaaaaaaa", "text": "ship it on Friday", "speaker": "human"}
    good = page(store, "q1", quotes=[owner], quote_resolver=resolver)
    assert good.ok and good.record["quotes"] == [owner]
    for bad in ({**owner, "text": "ship it on Monday"},
                {"address": "row:1@t2#bbbbbbbbbbbb", "text": "ship it", "speaker": "human"},
                {**owner, "address": "row:1@t9#cccccccccccc"}):
        refused = page(store, "q2", at=1, quotes=[owner, bad], quote_resolver=resolver)
        assert refused.reason == "quote_mismatch" and refused.detail.startswith("quotes[1]")
    assert page(store, "q2", at=1, quotes=[owner]).reason == "quote_mismatch"
    smuggled = store.publish([{"kind": "page", "room_id": "r", "text": "x", "author": MIND, "quotes": [owner],
                               "covers": covers("q3", at=3)}])
    assert smuggled.reason == "quote_mismatch"
    assert page(store, "q2", at=1).ok
    assert store.log_path.read_bytes().count(b"\n") == 2


def test_a_quote_is_exact_up_to_emphasis_markers_and_a_changed_word_is_refused(tmp_path):
    """A helper that copied a row's words without its bold is not refused; any other change still is,
    with the same refusal. The markers are set aside on both sides; underscores inside a word are no marker."""
    store = ChronicleStore(tmp_path)
    rows = {"row:1@t1#aaaaaaaaaaaa": ("This removes the product fork, **but not** the red checks; call memory_read.",
                                      "ouroboros"),
            "row:1@t2#bbbbbbbbbbbb": ("This removes the product fork, but not the red checks.", "ouroboros")}
    resolver = rows.get

    def quote(address, text):
        return {"address": address, "text": text, "speaker": "ouroboros"}

    bold, plain = "row:1@t1#aaaaaaaaaaaa", "row:1@t2#bbbbbbbbbbbb"
    for kept in (quote(bold, "the product fork, **but not** the red"), quote(bold, "the product fork, but not the red"),
                 quote(bold, "fork, *but not* the"), quote(plain, "the product fork, **but not** the red"),
                 quote(bold, "call memory_read.")):
        assert store_verify(kept, resolver) is None, kept
    for changed in (quote(bold, "the product fork, but now the red"), quote(bold, "the product fork but not the red"),
                    quote(bold, "the red checks, but not"), quote(bold, "call memoryread."),
                    quote(plain, "the product fork, **but not** the green")):
        refused = store_verify(changed, resolver)
        assert (refused.reason, refused.detail) == ("quote_mismatch",
                                                    "quotes[0]: the text is not an exact substring of that row"), changed
    published = page(store, "q1", quotes=[quote(bold, "the product fork, but not the red")], quote_resolver=resolver)
    assert published.ok and published.record["quotes"][0]["text"] == "the product fork, but not the red"


def store_verify(quote, resolver):
    from ouroboros.chronicle_store import verify_quotes

    return verify_quotes([quote], resolver)


def test_mark_targets_are_checked_inside_the_publication(tmp_path):
    store = ChronicleStore(tmp_path)
    kept = note(store, "The owner chose the narrow fix")
    missing = store.mark({"kind": "chronicle", "id": "gone"}, "x", MIND, room_id="r")
    assert (missing.reason, missing.conflict_ids) == ("target_missing", ("gone",))
    wrong = store.mark({"kind": "chronicle", "id": kept["id"]}, "x", MIND, room_id="r", quote="wide fix")
    assert wrong.reason == "quote_mismatch"
    right = store.mark({"kind": "chronicle", "id": kept["id"]}, "Why", MIND, room_id="r", quote="narrow fix")
    assert right.ok
    assert store.mark({"kind": "chronicle", "id": kept["id"]}, "Why", MIND, room_id="r", scope="world").reason == "invalid"
    assert store.set_mark_view("unknown", "meaning", MIND, "x").reason == "target_missing"
    assert store.set_mark_view(right.record["id"], "loud", MIND, "x").reason == "invalid"
    assert store.set_mark_view(right.record["id"], "meaning", MIND, "Meaning is enough").ok
    assert store.active_marks("r")[0]["visibility"] == "meaning"


def test_rebuilt_index_has_the_same_rows_folds_sequences_and_page_order(tmp_path):
    store = ChronicleStore(tmp_path)
    late = page(store, "x5", "x6", at=5).record
    early = page(store, "x0", at=0).record
    kept = note(store, "note")
    page(store, "x9", "note:" + kept["id"], at=9)
    draft = page(store, "x3", at=3, author=LIGHT).record
    store.decide(draft["id"], False, MIND, "no")
    store.publish_part(room_id="r", text="fold", member_ids=[early["id"], late["id"]], author=MIND,
                       expected_sequence=store.room_head("r"))
    store.correct(late["id"], "late, corrected", MIND)

    def snapshot():
        derived = ([(r["id"], r["sequence"]) for r in store.records()],
                   [r["id"] for r in store.pages_of_room("r")], store.sealed_row_refs("r"), store.room_head("r"))
        with sqlite3.connect(store.index_path) as db:
            tables = {name: sorted(db.execute(f"SELECT * FROM {name}"))
                      for name in ("page_rows", "folded", "active_marks")}
        return tables, derived

    before = snapshot()
    assert [r["id"] for r in store.pages_of_room("r") if r["kind"] == "page"][:2] == [early["id"], late["id"]]
    store.index_path.unlink()
    assert snapshot() == before


def test_index_schema_is_version_one_and_a_foreign_version_replays(tmp_path):
    store = ChronicleStore(tmp_path)
    kept = note(store, "survives")
    with sqlite3.connect(store.index_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        names = {name for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"records", "state", "active_marks", "page_rows", "folded"} <= names
        db.execute("PRAGMA user_version=7")
    assert store.get(kept["id"])["sequence"] == kept["sequence"]


def test_scan_state_merges_by_top_level_key(tmp_path):
    store = ChronicleStore(tmp_path)
    assert store.publish([], scan_state={"legacy_frontier": {"pos": 12400, "status": "exact"}}).ok
    assert store.publish([], scan_state={"fallback_refusals": [{"room_id": "1"}]}).ok
    merged = {"legacy_frontier": {"pos": 12400, "status": "exact"}, "fallback_refusals": [{"room_id": "1"}]}
    assert store.scan_state() == merged
    before = store.log_path.read_bytes()
    assert store.publish([], scan_state={"legacy_frontier": {"pos": 12400, "status": "exact"}}).reason == "unchanged"
    assert store.log_path.read_bytes() == before
    store.index_path.unlink()
    assert store.scan_state() == merged


def test_refusals_are_typed_results_and_append_nothing(tmp_path):
    store = ChronicleStore(tmp_path)
    good = note(store, "valid")
    before = store.log_path.read_bytes()
    cases = [
        [{"kind": "episode", "room_id": "r", "text": "x", "author": MIND}],
        [{"kind": "note", "room_id": "r", "task_id": "t1", "text": "x", "author": {"kind": "mind", "task_id": "t1"}}],
        [{"kind": "note", "room_id": "r", "task_id": "t1", "text": "x", "author": {"kind": "stranger"}}],
        [{"kind": "page", "room_id": "r", "text": "x", "author": MIND, "covers": {"rows": ["z"]}}],
        [{"kind": "page", "room_id": "r", "text": "x", "author": MIND, "covers": {"rows": ["z", "z"],
                                                                                    "stream_span": [0, 1]}}],
        [{"kind": "page", "room_id": "r", "text": " ", "author": MIND, "covers": covers("z")}],
        [{"kind": "page", "room_id": "r", "text": "x", "author": LEGACY, "covers": covers("z")}],
    ]
    for records in cases:
        result = store.publish(records)
        assert isinstance(result, PublishResult) and not result.ok and result.reason == "invalid", records
    assert page(store, "z", expected_sequence="7").reason == "invalid"
    assert store.log_path.read_bytes() == before
    assert store.publish([{"kind": "page", "room_id": "r", "text": "x", "author": MIND,
                           "covers": covers("z")}]).ok
    assert store.get(good["id"])["text"] == "valid"


def test_repeated_activation_returns_the_existing_receipt(tmp_path):
    store = ChronicleStore(tmp_path)
    receipt = {"id": "legacy-import-1", "kind": "activation", "room_id": "", "author": {"kind": "host"}}
    first = store.publish([receipt], scan_state={"legacy_frontier": {"pos": 1}})
    again = store.publish([{**receipt, "id": "legacy-import-2"}], scan_state={"legacy_frontier": {"pos": 9}})
    assert first.reason == "saved" and again.reason == "unchanged" and again.record["id"] == "legacy-import-1"
    assert store.activation()["id"] == "legacy-import-1"
    assert store.scan_state() == {"legacy_frontier": {"pos": 1}}
    assert store.room_head("") == 0


def test_store_imports_no_writer_view_or_model_code():
    tree = ast.parse((REPO / "ouroboros" / "chronicle_store.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = ("ouroboros.consolidator", "ouroboros.room_consolidation", "ouroboros.chronicle_view", "ouroboros.llm")
    assert not [name for name in imported if name.startswith(forbidden)]
    assert {"ouroboros.platform_layer", "ouroboros.utils"} <= imported
