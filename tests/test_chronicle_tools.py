"""The memory tools: chronicle_write, memory_read, memory_mark.

chronicle_write: a page names a range or a set of tasks and the host publishes
the exact row set, stamped and with checked quotes; a stale room head is refused
with ids, never text. memory_read: text with one header line per record or row,
every mode bounded by ``tool_result_limit("memory_read")`` with its continuation
named on the second line, and nothing written to disk once the chronicle is
active. memory_mark: one target, an exact quote, view and release. Each tool's
first call activates the chronicle (the one legacy import); later calls take the
fast path. No test calls a model or the network.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import pathlib
import re

import pytest

from ouroboros import chat_chain
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tool_capabilities import tool_result_limit
from ouroboros.tools.chronicle import (
    _chronicle_write, _memory_mark, _memory_read, check_quotes, host_stamp, page_covers,
)
from ouroboros.tools.registry import ToolContext

REPO = pathlib.Path(__file__).resolve().parents[1]
LIMIT = tool_result_limit("memory_read")
LEGACY = {"kind": "legacy_helper", "writer": "old_consolidator", "attribution": "retelling by Light, not lived"}
LIGHT = {"kind": "helper", "route": "configured-light"}


def ctx_for(root: pathlib.Path, task_id: str = "root0001", **fields) -> ToolContext:
    fields.setdefault("current_chat_id", 1)
    return ToolContext(repo_dir=root, drive_root=root, task_id=task_id, **fields)


def ts(n: int) -> str:
    return f"2026-10-01T{n // 3600 % 24:02d}:{n // 60 % 60:02d}:{n % 60:02d}+00:00"


def chat(root: pathlib.Path, rows) -> list:
    path = root / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return list(rows)


def addr(row) -> str:
    return chat_chain.format_address(chat_chain.row_address(row))


def sha(row) -> str:
    return chat_chain.source_row_id(row)


def write(ctx, **args) -> dict:
    return json.loads(_chronicle_write(ctx, **args))


def snapshot(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def second_line(text: str) -> str:
    return text.split("\n", 2)[1]


# --- chronicle_write: pages ---------------------------------------------------------------------

def project_chat(root: pathlib.Path, name: str) -> int:
    from ouroboros.projects_registry import create_project

    return int(create_project(root, name)["chat_id"])


def test_range_page_publishes_its_room_rows_as_a_set_with_stamp_and_source_facts(tmp_path):
    side = project_chat(tmp_path, "side-room")
    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Please check the build", "client_message_id": "m1"},
        {"chat_id": side, "direction": "in", "ts": ts(2), "text": "another room"},
        {"chat_id": 1, "direction": "out", "ts": ts(3), "task_id": "taskA001", "text": "The build is red."},
        {"chat_id": 1, "direction": "system", "ts": ts(4), "type": "task_summary", "task_id": "taskA001",
         "summary_kind": "terminal_root_projection", "status": "failed", "outcome": "Failed",
         "outcome_phase": "error", "reason_detail": "Reviewers rejected it.", "text": "Failed. Root task taskA001.",
         "result_ref": {"kind": "task_result", "task_id": "taskA001", "reader": "get_task_result"}},
    ])
    from ouroboros.task_results import write_task_result

    # No row carries lineage yet, so the epoch is the chain end: my final is mine by its task result.
    write_task_result(tmp_path, "taskA001", "failed", result="The build is red.")
    ctx = ctx_for(tmp_path)
    reply = write(ctx, kind="page", text="I found the build red; the reviewers rejected my fix.",
                  covers={"from": addr(rows[0]), "to": addr(rows[3])})
    assert reply["ok"] and reply["reason"] == "saved" and reply["kind"] == "page" and reply["room_id"] == "1"
    assert reply["sequence"] == reply["room_head"] and reply["rows"] == 3
    assert reply["first"] == addr(rows[0]) and reply["last"] == addr(rows[3])
    assert reply["coverage_facts"]["by_author"] == {"host": 1, "human": 1, "ouroboros": 1}
    assert reply["stamp"] == {"failed": 1}
    page = ChronicleStore(tmp_path).get(reply["node_id"])
    # The other room's row between the bounds is not this room's and is not sealed.
    assert page["covers"]["rows"] == [sha(rows[0]), sha(rows[2]), sha(rows[3])]
    assert page["covers"]["mode"] == "range" and page["covers"]["stream_span"] == [0, 3]
    assert page["covers"]["task_ids"] == ["taskA001"]
    assert page["host_stamp"]["tasks"] == [{
        "task_id": "taskA001", "status": "failed", "outcome": "Failed", "outcome_phase": "error",
        "source": "terminal_root_projection", "review_verdict": "Reviewers rejected it.",
        "result_ref": {"kind": "task_result", "task_id": "taskA001", "reader": "get_task_result"},
        "source_address": addr(rows[3])}]
    assert page["author"]["kind"] == "mind" and page["author"]["focus"]["role"] == "root"
    assert page["author"]["task_id"] == "root0001"
    assert ChronicleStore(tmp_path).sealed_row_refs(str(side)) == set()


def test_task_page_takes_task_rows_and_bound_owner_words_and_leaves_an_interleaved_task_open(tmp_path):
    ask_a = {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Fix the login", "client_message_id": "mA"}
    ask_b = {"chat_id": 1, "direction": "in", "ts": ts(2), "text": "Draft the release note", "client_message_id": "mB"}
    unbound = {"chat_id": 1, "direction": "in", "ts": ts(3), "text": "ok thanks", "client_message_id": "mC"}
    origin = {"chat_id": 1, "client_message_id": "mA", "ts": ts(1),
              "text_sha256": hashlib.sha256(b"Fix the login").hexdigest()}
    rows = chat(tmp_path, [
        ask_a, ask_b, unbound,
        {"chat_id": 1, "direction": "out", "ts": ts(4), "task_id": "taskA001", "text": "Login fixed.",
         "origin_message_ref": origin},
        {"chat_id": 1, "direction": "out", "ts": ts(5), "task_id": "taskB001", "text": "Release note drafted."},
        {"chat_id": 1, "direction": "out", "ts": ts(6), "task_id": "kidA0001", "root_task_id": "taskA001",
         "parent_task_id": "taskA001", "subagent_task_id": "kidA0001", "text": "Child report."},
    ])
    annotations = tmp_path / "logs" / "chat_annotations.jsonl"
    annotations.write_text(json.dumps({"ts": ts(2), "type": "chat_annotation", "client_message_id": "mB",
                                       "action": "promote_chat_to_task", "target": "taskB001",
                                       "status": "scheduled"}) + "\n", encoding="utf-8")
    from ouroboros.task_results import write_task_result

    # The first tool call activates the chronicle, and its lineage epoch is this chain's
    # first child row: the root's earlier final is attributed by its recorded task result.
    write_task_result(tmp_path, "taskA001", "completed", result="Login fixed.")
    ctx = ctx_for(tmp_path)
    covered = page_covers(tmp_path, "1", task_ids=["taskA001"])
    assert [row for _address, row in covered["rows"]] == [rows[0], rows[3], rows[5]]
    first = write(ctx, kind="page", text="I fixed the login; my child reported.", covers={"task_ids": ["taskA001"]})
    assert first["ok"] and first["rows"] == 3
    assert first["coverage_facts"]["by_author"] == {"child": 1, "human": 1, "ouroboros": 1}
    # The interleaved task's page passes: its rows were never sealed by the first page.
    second = write(ctx, kind="page", text="I drafted the release note.", covers={"task_ids": ["taskB001"]})
    assert second["ok"] and second["rows"] == 2  # promote annotation binds the owner's words
    sealed = ChronicleStore(tmp_path).sealed_row_refs("1")
    assert sha(unbound) not in sealed and sha(ask_a) in sealed and sha(ask_b) in sealed
    # A range over rows the task pages hold is refused with their ids, never their text.
    again = json.loads(_chronicle_write(ctx, kind="page", text="dup", covers={"from": addr(rows[0]),
                                                                              "to": addr(rows[1])}))
    assert again["ok"] is False and again["reason"] == "already_sealed"
    assert again["conflict_ids"] == [first["node_id"], second["node_id"]]
    assert "login" not in json.dumps(again).lower()


def test_stale_expected_sequence_returns_the_head_and_ids_without_text_and_a_current_one_saves(tmp_path):
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"owner words {n}"}
                           for n in range(1, 5)])
    ctx = ctx_for(tmp_path)
    head = int(re.match(r"room 1; head (\d+)", _memory_read(ctx)).group(1))
    assert head == 0
    other = write(ctx, kind="note", text="SECRET-NOTE-TEXT for my future self")
    assert other["ok"] and other["room_head"] == other["sequence"]
    stale = json.loads(_chronicle_write(ctx, kind="page", text="based on an old head",
                                        covers={"from": addr(rows[0]), "to": addr(rows[1])},
                                        expected_sequence=head))
    assert stale["ok"] is False and stale["reason"] == "revision_conflict"
    assert stale["current_head"] == other["sequence"] and stale["conflict_ids"] == [other["node_id"]]
    assert "SECRET-NOTE-TEXT" not in json.dumps(stale)
    assert ChronicleStore(tmp_path).records("1", kinds=["page"]) == []
    fresh = write(ctx, kind="page", text="based on the current head",
                  covers={"from": addr(rows[0]), "to": addr(rows[1])}, expected_sequence=other["sequence"])
    assert fresh["ok"] and fresh["sequence"] == fresh["room_head"] > other["sequence"]
    # Without expected_sequence a page of another arc is not held back by the head.
    free = write(ctx, kind="page", text="another arc", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    assert free["ok"]


def test_a_quote_of_a_bold_row_holds_without_the_markers_and_a_changed_word_does_not(tmp_path):
    """On the real resolver (the one the Light helper's draft is checked with): the row's words
    copied without its ``**`` pass, with them too; a changed word is refused."""
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(1),
                            "text": "Merge it, **but not** before the red checks are green."}])
    owner = {"address": addr(rows[0]), "speaker": "human"}
    for text in ("Merge it, but not before", "Merge it, **but not** before", "red checks are green."):
        assert check_quotes(tmp_path, [{**owner, "text": text}]) == (True, None), text
    for text in ("Merge it, but now before", "Merge it but not before", "before the green checks"):
        assert check_quotes(tmp_path, [{**owner, "text": text}]) == (False, 0), text


def test_forged_quote_is_refused_and_an_exact_quote_by_its_speaker_passes(tmp_path):
    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Ship it only after the tests pass."},
        {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "taskA001", "text": "I will run them first."},
    ])
    ctx = ctx_for(tmp_path)
    covers = {"from": addr(rows[0]), "to": addr(rows[1])}
    owner = {"address": addr(rows[0]), "text": "only after the tests pass", "speaker": "human"}
    for forged in ({**owner, "speaker": "ouroboros"}, {**owner, "text": "ship it now"},
                   {**owner, "address": addr(rows[1])}):
        refused = json.loads(_chronicle_write(ctx, kind="page", text="t", covers=covers, quotes=[forged]))
        assert refused["ok"] is False and refused["reason"] == "quote_mismatch"
        assert check_quotes(tmp_path, [owner, forged]) == (False, 1)
    assert ChronicleStore(tmp_path).records("1", kinds=["page"]) == []
    mine = {"address": addr(rows[1]), "text": "I will run them first.", "speaker": "ouroboros"}
    # No row carries lineage, so my reply precedes the epoch: without a task result it is not mine.
    assert check_quotes(tmp_path, [owner, mine]) == (False, 1)
    assert check_quotes(tmp_path, [owner, {**mine, "speaker": "unattributed"}]) == (True, None)
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "taskA001", "completed", result="Tests first.")
    assert check_quotes(tmp_path, [owner, mine]) == (True, None)
    saved = write(ctx, kind="page", text="The owner set a condition; I promised to test first.", covers=covers,
                  quotes=[owner, mine])
    assert saved["ok"]
    assert ChronicleStore(tmp_path).get(saved["node_id"])["quotes"] == [owner, mine]
    # A page without quotes is published as well: the host inserts no words itself.
    assert write(ctx, kind="note", text="n")["ok"]


def test_note_is_written_and_a_later_page_of_its_task_covers_it_once(tmp_path):
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "out", "ts": ts(n), "task_id": "root0001",
                            "text": f"step {n}"} for n in range(1, 5)])
    ctx = ctx_for(tmp_path)
    note = write(ctx, kind="note", text="Open thread: the owner still owes a decision on pricing.")
    assert note["ok"] and note["kind"] == "note"
    stored = ChronicleStore(tmp_path).get(note["node_id"])
    assert stored["task_id"] == "root0001" and stored["room_id"] == "1" and stored["author"]["kind"] == "mind"
    first = write(ctx, kind="page", text="first arc", covers={"from": addr(rows[0]), "to": addr(rows[1])})
    assert first["notes"] == 1
    assert f"note:{note['node_id']}" in ChronicleStore(tmp_path).sealed_row_refs("1")
    # The same task's later arc does not take the already sealed note again.
    later = write(ctx, kind="page", text="second arc", covers={"from": addr(rows[2]), "to": addr(rows[3])})
    assert later["ok"] and later["notes"] == 0


def test_default_room_is_own_room_and_chat_id_zero_is_an_address(tmp_path):
    assert write(ctx_for(tmp_path, current_chat_id=0), kind="note", text="hidden")["room_id"] == "0"
    meta_only = ctx_for(tmp_path, current_chat_id=None, task_metadata={"chat_id": 0})
    assert write(meta_only, kind="note", text="hidden too")["room_id"] == "0"
    assert write(meta_only, kind="note", text="explicit", room_id="12")["room_id"] == "12"
    assert _memory_read(ctx_for(tmp_path, current_chat_id=0)).startswith("room 0; head ")
    nowhere = ctx_for(tmp_path, current_chat_id=None)
    refused = _chronicle_write(nowhere, kind="note", text="lost")
    assert "TOOL_ARG_ERROR" in refused and "room_id is required" in refused
    assert write(nowhere, kind="note", text="addressed", room_id="7")["room_id"] == "7"


def test_a_room_name_is_refused_with_the_repair_and_never_read_or_written_as_a_room(tmp_path):
    """Regression: ``room_id='Main'`` read as an empty room and wrote a note
    into a phantom room; now each mode refuses it naming the repair, while a chat id (negative
    or zero included) and ``legacy`` still address rooms."""
    ctx = ctx_for(tmp_path)
    for refused in (_chronicle_write(ctx, kind="note", text="lost", room_id="Main"),
                    _chronicle_write(ctx, kind="page", text="lost", room_id="Main", covers={"task_ids": ["t1"]}),
                    _memory_read(ctx, room_id="Main"), _memory_read(ctx, room_id="Main", rows=True)):
        assert "TOOL_ARG_ERROR" in refused and "room_id 'Main' is not a room address" in refused
    assert ChronicleStore(tmp_path).room_records("Main") == []
    assert write(ctx, kind="note", text="kept", room_id="-1001")["room_id"] == "-1001"
    assert _memory_read(ctx, room_id="legacy").startswith("room legacy; head ")


def test_part_correction_and_decision_go_through_the_store_rules(tmp_path):
    store, ctx = ChronicleStore(tmp_path), ctx_for(tmp_path)
    for block in (0, 1):
        assert store.publish([{"id": f"legacy-b{block:02d}-r5", "kind": "legacy", "room_id": "5",
                               "text": f"retelling {block}", "author": LEGACY,
                               "metadata": {"legacy_type": "era", "legacy_block": block}}]).ok
    head = store.room_head("5")
    unbased = json.loads(_chronicle_write(ctx, kind="part", text="fold", member_ids=["legacy-b00-r5"]))
    assert unbased["reason"] == "revision_required" and unbased["current_head"] == head
    # The members' own room is the default, not this task's room.
    part = write(ctx, kind="part", text="I lived 30.07-25.09 like this.", member_ids=["legacy-b00-r5"],
                 expected_sequence=head)
    assert part["ok"] and part["room_id"] == "5"
    assert json.loads(_chronicle_write(ctx, kind="part", text="again", member_ids=["legacy-b00-r5"],
                                       expected_sequence=part["room_head"]))["reason"] == "already_folded"
    first = write(ctx, kind="correction", target_id="legacy-b01-r5", text="It was 26.09, not 25.09.")
    assert first["ok"] and first["revision"] == first["node_id"]
    again = json.loads(_chronicle_write(ctx, kind="correction", target_id="legacy-b01-r5", text="second"))
    assert again["reason"] == "revision_required" and again["current_revision"] == first["node_id"]
    assert write(ctx, kind="correction", target_id="legacy-b01-r5", text="second",
                 expected_revision=first["node_id"])["ok"]
    draft = store.publish_page(room_id="5", text="Light's draft", author=LIGHT,
                               covers={"mode": "range", "rows": ["r1"], "stream_span": [3, 3]})
    assert draft.ok
    decided = write(ctx, kind="decision", target_id=draft.record["id"], accepted=False, reason="wrong period")
    assert decided["ok"] and decided["kind"] == "decision"
    assert all(record["id"] != draft.record["id"] for record in store.room_records("5"))
    assert "TOOL_ARG_ERROR" in _chronicle_write(ctx, kind="decision", target_id=draft.record["id"])
    assert "TOOL_ARG_ERROR" in _chronicle_write(ctx, kind="essay", text="x")


# --- memory_read ---------------------------------------------------------------------------------

def test_room_read_is_text_with_the_head_line_and_one_header_per_record(tmp_path):
    rows = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(1), "text": "owner words"}])
    ctx = ctx_for(tmp_path)
    page = write(ctx, kind="page", text="PAGE TEXT\nsecond line", covers={"from": addr(rows[0]), "to": addr(rows[0])})
    mark = json.loads(_memory_mark(ctx, text="MARK TEXT", node_id=page["node_id"], quote="PAGE TEXT"))
    text = _memory_read(ctx)
    lines = text.split("\n")
    assert lines[0] == f"room 1; head {page['sequence']}"
    assert lines[1].startswith(f"complete: no records after seq {mark['sequence']}")
    assert lines[2].startswith(f"[page {page['node_id']}; room 1; mind (root root0001); final; covers ")
    assert lines[2].endswith(f"seq {page['sequence']}]") and lines[3:5] == ["PAGE TEXT", "second line"]
    assert lines[5].startswith(f"[mark {mark['mark_id']}; room 1;") and lines[6:] == ["MARK TEXT", "quote: PAGE TEXT"]
    with pytest.raises(ValueError):
        json.loads(text)
    assert _memory_read(ctx, after_seq=mark["sequence"]).split("\n")[1].startswith("complete: no records after")


def test_rows_read_attributes_each_row_by_its_source_fields_as_text(tmp_path):
    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "system", "type": "quiz_answer", "ts": ts(1),
         "quiz": {"quiz_id": "q1", "question": "Which?", "options": [{"label": "Alpha"}, {"label": "Beta"}],
                  "answered_index": 1}, "client_message_id": "quiz_answer:q1"},
        {"chat_id": 1, "direction": "system", "type": "task_summary", "summary_kind": "host_task_facts", "ts": ts(2),
         "task_id": "taskA001", "status": "completed", "outcome": "Done", "outcome_phase": "done", "text": "",
         "result_ref": {"kind": "task_result", "task_id": "taskA001", "reader": "get_task_result"}},
        {"chat_id": 1, "direction": "out", "ts": ts(3), "task_id": "kid00001", "subagent_task_id": "kid00001",
         "parent_task_id": "taskA001", "text": "child report"},
        {"chat_id": 1, "direction": "out", "ts": ts(4), "task_id": "taskA001", "text": "my answer"},
    ])
    text = _memory_read(ctx_for(tmp_path), rows=True)
    lines = text.split("\n")
    assert lines[0] == "room 1; rows" and lines[1].startswith("complete: no further rows")
    assert lines[2].startswith(f"[{ts(1)}; Owner; {addr(rows[0])}] ") and "Beta" in lines[2]
    assert lines[3] == (f"[{ts(2)}; host; {addr(rows[1])}] host facts for taskA001: status=completed; outcome=Done; "
                        "phase=done; result: get_task_result(task_id=taskA001)")
    assert lines[4].startswith(f"[{ts(3)}; child kid00001 of taskA001; ") and lines[4].endswith("child report")
    assert lines[5] == f"[{ts(4)}; Ouroboros; {addr(rows[3])}] my answer"
    assert "Ouroboros" not in "".join(lines[2:5])
    only = _memory_read(ctx_for(tmp_path), rows=True, task_id="kid00001").split("\n")
    assert only[0] == "room 1; rows; task_id kid00001" and len(only) == 3
    bounded = _memory_read(ctx_for(tmp_path), rows=True, **{"from": addr(rows[1]), "to": addr(rows[2])})
    assert len(bounded.split("\n")) == 4
    missing = _memory_read(ctx_for(tmp_path), rows=True, **{"from": "row:1@" + ts(9) + "#" + "a" * 12})
    assert "TOOL_ARG_ERROR" in missing and "row_missing" in missing


def test_an_install_without_any_lineage_row_attributes_old_outgoing_rows_by_task_results_only(tmp_path):
    """An update from a version that never recorded a child's lineage: no row proves which old
    outgoing row was mine, so the task results decide and the rest is unattributed. Rows written
    after activation carry lineage when a child wrote them, so a lineage-free one is mine."""
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION
    from ouroboros.task_results import task_result_path

    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Review the patch."},
        {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "kid00001",
         "text": "## Summary Child review: the patch is unsafe."},
        {"chat_id": 1, "direction": "out", "ts": ts(3), "task_id": "kid00002", "text": "## Summary Second look."},
        {"chat_id": 1, "direction": "out", "ts": ts(4), "task_id": "root0001", "text": "Done."},
    ])
    for task_id, fields in (("root0001", {}), ("kid00002", {"parent_task_id": "root0001", "root_task_id": "root0001",
                                                             "delegation_role": "subagent"})):
        task_result_path(tmp_path, task_id).write_text(json.dumps({
            SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": task_id, "status": "completed", **fields}),
            encoding="utf-8")
    ctx = ctx_for(tmp_path)
    lines = _memory_read(ctx, rows=True).split("\n")[2:]
    assert lines[1].startswith(f"[{ts(2)}; outgoing, author not recorded; {addr(rows[1])}]")
    assert lines[2].startswith(f"[{ts(3)}; child kid00002 of root0001; ")
    assert lines[3] == f"[{ts(4)}; Ouroboros; {addr(rows[3])}] Done."
    assert ChronicleStore(tmp_path).activation()["metadata"]["lineage_epoch"]["pos"] == 4
    later = chat(tmp_path, [{"chat_id": 1, "direction": "out", "ts": ts(5), "task_id": "root0002", "text": "Next."}])
    assert _memory_read(ctx, rows=True).split("\n")[-1] == f"[{ts(5)}; Ouroboros; {addr(later[0])}] Next."


def test_node_read_shows_stamp_original_and_corrections_with_the_acting_revision(tmp_path):
    chat(tmp_path, [{"chat_id": 1, "direction": "out", "ts": ts(1), "task_id": "taskA001", "text": "done"}])
    ctx = ctx_for(tmp_path)
    page = write(ctx, kind="page", text="ORIGINAL", covers={"task_ids": ["taskA001"]})
    fix = write(ctx, kind="correction", target_id=page["node_id"], text="CORRECTED")
    text = _memory_read(ctx, node_id=page["node_id"])
    lines = text.split("\n")
    assert lines[0].startswith(f"[page {page['node_id']}; room 1;")
    assert f"revision {fix['node_id']} (corrected by mind (root root0001))" in lines[0]
    assert "stamp: taskA001=not_recorded;" in lines[0] and lines[1].endswith("complete")
    assert "stamp taskA001: status=not_recorded" in text
    assert "text:\nORIGINAL" in text and f"correction {fix['node_id']} by mind (root root0001)" in text
    assert text.rstrip().endswith("CORRECTED")
    assert "TOOL_ARG_ERROR" in _memory_read(ctx, node_id="no-such-node")
    assert "TOOL_ARG_ERROR" in _memory_read(ctx, node_id=page["node_id"], rows=True)


def _parse_window(text: str):
    match = re.match(r"chars (\d+)–(\d+) of (\d+); (?:next_start=(\d+)|complete)", second_line(text))
    assert match, second_line(text)
    return int(match[1]), int(match[2]), int(match[3]), match[4]


def test_every_read_mode_stays_within_the_limit_names_its_continuation_and_writes_nothing(tmp_path):
    """Acceptance: a 10 000-row room, a record larger than a page, Main-sized legacy records,
    a row larger than a page and a retained source, read to the end page by page."""
    room = [{"chat_id": 1, "direction": "in" if n % 2 else "out", "ts": ts(n), "task_id": f"t{n // 50:04d}",
             "text": f"row {n} " + "words " * (n % 40)} for n in range(10_000)]
    side = project_chat(tmp_path, "giant-room")
    giant = {"chat_id": side, "direction": "out", "ts": ts(1), "text": "G" * (LIMIT * 2 + 17)}
    after_giant = {"chat_id": side, "direction": "in", "ts": ts(2), "text": "after the giant"}
    chat(tmp_path, [*room, giant, after_giant])
    store = ChronicleStore(tmp_path)
    sizes = [61_000, 47_500, 33_000, 21_000, 12_310, 9_000, 4_000, 2_000, 1_000]  # 190 810 chars of Main
    for block, size in enumerate(sizes):
        assert store.publish([{"id": f"legacy-b{block:02d}-r1", "kind": "legacy", "room_id": "1", "author": LEGACY,
                               "text": f"[{block}]" + "x" * (size - 4),
                               "metadata": {"legacy_type": "era", "legacy_block": block}}]).ok
    huge_note = "N" * (LIMIT + 5_000)
    ctx = ctx_for(tmp_path)
    note = write(ctx, kind="note", text=huge_note)
    source = chat_chain.retain_memory_source(ctx, "probe", ("S" * (LIMIT + 123)).encode("utf-8"), "md")
    store.records()  # the disposable index is current before the snapshot
    before = snapshot(tmp_path)

    seen, frm, pages = [], None, 0  # rows: 10 000 rows of Main, older to newer
    while True:
        text = _memory_read(ctx, rows=True, **({"from": frm} if frm else {}))
        pages += 1
        assert len(text) <= LIMIT and text.split("\n")[0] == "room 1; rows" + (f"; from {frm}" if frm else "")
        seen += [line for line in text.split("\n")[2:] if line.startswith("[")]
        cont = second_line(text)
        if cont.startswith("complete"):
            break
        frm = re.match(r"next: from=(\S+) ", cont).group(1)
    assert pages > 5 and len(seen) == 10_000
    assert [line.split("; ")[2].rstrip("]").split("] ")[0] for line in seen][:2] == [addr(room[0]), addr(room[1])]
    assert seen[-1].endswith(room[-1]["text"])

    gathered, start = "", 0  # one row larger than a page, by character window
    while True:
        text = _memory_read(ctx, rows=True, room_id=str(side), **({"from": addr(giant)}), start=start)
        assert len(text) <= LIMIT
        gathered += text.split("\n")[2].split(") ", 1)[1]
        cont = second_line(text)
        found = re.match(r"next: from=(\S+) start=(\d+)", cont)
        if not found:  # the giant's rest fits: the page goes on to the next row
            assert cont.startswith("complete") and text.endswith(after_giant["text"])
            gathered = gathered.split("\n", 1)[0]
            break
        start = int(found.group(2))
    assert gathered == giant["text"]

    listed, after = [], 0  # room records: Main-sized legacy and a note larger than a page
    while True:
        text = _memory_read(ctx, room_id="1", after_seq=after)
        assert len(text) <= LIMIT and text.startswith("room 1; head ")
        listed += re.findall(r"^\[(?:legacy|note) (\S+);", text, flags=re.M)
        cont = second_line(text)
        if cont.startswith("complete"):
            break
        after = int(re.match(r"next_after_seq=(\d+):", cont).group(1))
    assert listed == [f"legacy-b{b:02d}-r1" for b in range(len(sizes))] + [note["node_id"]]
    stub = _memory_read(ctx, room_id="1", after_seq=store.get("legacy-b08-r1")["sequence"])
    assert f"memory_read(node_id={note['node_id']}) pages it" in stub and huge_note[:100] not in stub

    for read, original in ((lambda s: _memory_read(ctx, node_id=note["node_id"], start=s), huge_note),
                           (lambda s: _memory_read(ctx, source_ref=source, start=s), "S" * (LIMIT + 123))):
        body, start = "", 0
        while True:
            text = read(start)
            assert len(text) <= LIMIT
            first, end, total, nxt = _parse_window(text)
            body += text.split("\n", 2)[2]
            if nxt is None:
                break
            start = int(nxt)
        assert original in body
    assert snapshot(tmp_path) == before  # reading wrote nothing


def test_reading_on_a_fresh_install_activates_once_then_reads_write_nothing(tmp_path):
    chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(1), "text": "hello"}])
    ctx = ctx_for(tmp_path)
    assert not (tmp_path / "memory" / "chronicle").exists()
    assert _memory_read(ctx).split("\n")[:2] == ["room 1; head 0", "complete: no records after seq 0 (older to newer)"]
    store = ChronicleStore(tmp_path)
    receipt = store.activation()
    assert receipt and receipt["metadata"]["imported_records"] == 0 and receipt["metadata"]["paid_calls"] == 0
    before = snapshot(tmp_path)
    assert "hello" in _memory_read(ctx, rows=True)
    assert _memory_read(ctx).startswith("room 1; head 0")
    assert snapshot(tmp_path) == before  # after activation, reading writes nothing


def _legacy_install(root):
    """Legacy dialogue memory the old writer left: two chat rows, one block over both, its cursor."""
    rows = chat(root, [{"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Never push on Fridays."},
                       {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "taskA001", "text": "Understood."}])
    (root / "memory").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "dialogue_blocks.json").write_text(json.dumps([{
        "ts": ts(3), "type": "summary", "range": "r", "message_count": 2,
        "rooms": [{"room_id": "1", "label": "Main", "message_count": 2, "content": "I learned the Friday rule."}],
        "content": "I learned the Friday rule."}]), encoding="utf-8")
    (root / "memory" / "dialogue_meta.json").write_text(json.dumps({
        "last_consolidated_offset": 2,
        "chat_log_signature": chat_chain._chat_log_signature(root / "logs" / "chat.jsonl")}), encoding="utf-8")
    return rows


@pytest.mark.parametrize("first", ["chronicle_write", "memory_read", "memory_mark"])
def test_the_first_memory_tool_call_imports_once_and_every_later_call_takes_the_fast_path(tmp_path, monkeypatch, first):
    """Each of the three tools starts by activating the chronicle. Whichever comes first
    imports the legacy memory (no model call); a repeat of any tool finds the receipt and
    imports nothing again."""
    from ouroboros import chronicle_import

    rows = _legacy_install(tmp_path)
    imports = []
    real_import = chronicle_import._import
    monkeypatch.setattr(chronicle_import, "_import", lambda store: (imports.append(1), real_import(store))[1])
    ctx = ctx_for(tmp_path)
    calls = {"chronicle_write": lambda: write(ctx, kind="note", text="A note for later."),
             "memory_read": lambda: _memory_read(ctx),
             "memory_mark": lambda: json.loads(_memory_mark(ctx, text="the rule", address=addr(rows[0])))}
    assert not (tmp_path / "memory" / "chronicle").exists()
    calls[first]()
    store = ChronicleStore(tmp_path)
    receipt = store.activation()
    assert imports == [1] and receipt and receipt["metadata"]["paid_calls"] == 0
    assert store.get("legacy-b00-r1")["text"] == "I learned the Friday rule."
    for name in ("chronicle_write", "memory_read", "memory_mark"):
        calls[name]()
    assert imports == [1] and store.activation() == receipt
    kinds = [record["kind"] for line in (tmp_path / "memory" / "chronicle" / "records.jsonl").read_text(
        encoding="utf-8").splitlines() for record in json.loads(line)["records"]]
    assert kinds.count("activation") == 1 and kinds.count("legacy") == 1


def _hold_import_lock(root):
    from ouroboros.platform_layer import file_lock_exclusive_nb

    path = root / "memory" / ".consolidation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o644)
    file_lock_exclusive_nb(fd)
    return fd


def test_a_tool_meeting_another_importer_waits_and_reads_the_imported_memory_with_its_epoch(tmp_path, monkeypatch):
    """Two parallel first reads after an update: the second meets the first's import lock. It waits
    and then reads the imported room, with pre-epoch rows unattributed, never "complete" over an
    empty journal and never signing an old outgoing row as mine."""
    from ouroboros import chronicle_import
    from ouroboros.platform_layer import file_unlock

    rows = _legacy_install(tmp_path)
    chat(tmp_path, [{"chat_id": 1, "direction": "out", "ts": ts(3), "task_id": "kid00001", "subagent_task_id": "kid00001",
                     "parent_task_id": "taskA001", "text": "child report"}])
    holder = _hold_import_lock(tmp_path)
    real_wait = chronicle_import.file_lock_exclusive

    def other_importer_finishes(fd):
        # The holder (the other first call) completes its import and releases; only then is the lock free.
        chronicle_import._import(ChronicleStore(tmp_path))
        file_unlock(holder)
        os.close(holder)
        real_wait(fd)

    monkeypatch.setattr(chronicle_import, "file_lock_exclusive", other_importer_finishes)
    listing = _memory_read(ctx_for(tmp_path))
    assert listing.split("\n")[0] != "room 1; head 0" and "legacy legacy-b00-r1" in listing
    assert "complete: no records after seq 0" not in listing
    read = _memory_read(ctx_for(tmp_path), rows=True)
    assert f"[{ts(2)}; outgoing, author not recorded; {addr(rows[1])}] Understood." in read
    assert "; Ouroboros;" not in read
    store = ChronicleStore(tmp_path)
    assert [r["kind"] for r in store.records(kinds=["activation"])] == ["activation"]


def test_a_tool_whose_import_has_not_completed_reads_and_writes_nothing_and_says_so(tmp_path, monkeypatch):
    from ouroboros import chronicle_import

    rows = _legacy_install(tmp_path)
    monkeypatch.setattr(chronicle_import, "_import", lambda store: {
        "kind": "import_refused", "reason": "invalid", "detail": "identity collision", "conflict_ids": ["legacy-b00-r1"]})
    ctx = ctx_for(tmp_path)
    replies = [json.loads(_memory_read(ctx)), json.loads(_memory_read(ctx, rows=True)),
               json.loads(_chronicle_write(ctx, kind="note", text="for later")),
               json.loads(_memory_mark(ctx, text="the rule", address=addr(rows[0])))]
    for reply in replies:
        assert reply["ok"] is False and reply["reason"] == "memory_not_activated"
        assert "import_refused" in reply["detail"] and reply["conflict_ids"] == ["legacy-b00-r1"]
    assert not ChronicleStore(tmp_path).log_path.exists()
    # Without the refusal the same calls read and write as usual.
    monkeypatch.undo()
    assert _memory_read(ctx).startswith("room 1; head ")
    assert write(ctx, kind="note", text="for later")["ok"]
    # A journal lost under its index is the store's own refusal, not an argument error.
    ChronicleStore(tmp_path).log_path.unlink()
    lost = json.loads(_memory_read(ctx))
    assert lost["reason"] == "memory_not_activated" and "import_failed" in lost["detail"]
    assert "authority missing" in lost["detail"]


# --- memory_mark ---------------------------------------------------------------------------------

def test_mark_targets_take_exact_quotes_and_refuse_invented_ones(tmp_path):
    rows = chat(tmp_path, [
        {"chat_id": 1, "direction": "in", "ts": ts(1), "text": "Never push on Fridays."},
        {"chat_id": 1, "direction": "out", "ts": ts(2), "task_id": "taskA001", "text": "progress"},
        {"chat_id": 1, "direction": "out", "ts": ts(3), "task_id": "taskA001", "text": "Final: shipped Monday."},
    ])
    ctx = ctx_for(tmp_path)
    note = write(ctx, kind="note", text="The owner's rule about Fridays.")
    source = chat_chain.retain_memory_source(ctx, "probe", json.dumps({"rows": [{"text": "line one\nline two"}]})
                                             .encode("utf-8"), "json")
    targets = [({"node_id": note["node_id"]}, "rule about Fridays"), ({"address": addr(rows[0])}, "push on Fridays"),
               ({"task_id": "taskA001"}, "shipped Monday"), ({"source_ref": source}, "line one\nline two")]
    for target, quote in targets:
        bad = json.loads(_memory_mark(ctx, text="why it matters", quote="invented words", **target))
        assert bad["ok"] is False and bad["reason"] == "quote_mismatch"
        good = json.loads(_memory_mark(ctx, text="why it matters", quote=quote, **target))
        assert good["ok"] and good["operation"] == "mark" and "why it matters" not in json.dumps(good)
    marks = ChronicleStore(tmp_path).active_marks("1")
    assert [m["quote"] for m in marks] == [quote for _t, quote in targets]
    assert marks[1]["target_ref"]["row_sha256"] == sha(rows[0]) and marks[2]["target_ref"] == {
        "kind": "task", "task_id": "taskA001"}
    assert all(m["author"]["focus"]["role"] == "root" for m in marks)
    # The task's earlier progress row is not its final words.
    assert json.loads(_memory_mark(ctx, text="x", task_id="taskA001", quote="progress"))["reason"] == "quote_mismatch"
    assert "TOOL_ARG_ERROR" in _memory_mark(ctx, text="x", node_id=note["node_id"], task_id="taskA001")
    assert "TOOL_ARG_ERROR" in _memory_mark(ctx, text="x")
    missing = json.loads(_memory_mark(ctx, text="x", address="row:1@" + ts(9) + "#" + "b" * 12))
    assert missing["ok"] is False and missing["reason"] == "row_missing"


def test_mark_view_and_release_including_an_imported_nomination(tmp_path):
    store, ctx = ChronicleStore(tmp_path), ctx_for(tmp_path)
    assert store.publish([{"id": "legacy-nomination-0123456789abcdef", "kind": "mark", "room_id": "legacy",
                           "scope": "global", "author": LEGACY, "text": "Nominated: people/rowan",
                           "target_ref": {"kind": "task_source", "path": "x", "location": "pending/0"},
                           "visibility": "full", "quote": None}]).ok
    note = write(ctx, kind="note", text="verbatim words")
    mark = json.loads(_memory_mark(ctx, text="keep", node_id=note["node_id"], quote="verbatim words"))
    assert json.loads(_memory_mark(ctx, mark_id=mark["mark_id"], visibility="meaning"))["reason"] == "invalid"
    view = json.loads(_memory_mark(ctx, mark_id=mark["mark_id"], visibility="meaning", reason="room for work"))
    assert view["ok"] and view["operation"] == "mark_view" and view["mark_id"] == mark["mark_id"]
    listed = _memory_read(ctx)
    assert "visibility meaning" in listed and "quote: verbatim words" not in listed
    assert "legacy-nomination-0123456789abcdef" in listed  # a global mark is in every room
    released = json.loads(_memory_mark(ctx, release_id="legacy-nomination-0123456789abcdef",
                                       reason="published the note myself"))
    assert released["ok"] and released["operation"] == "mark_release"
    assert [m["id"] for m in store.active_marks("1")] == [mark["mark_id"]]
    assert json.loads(_memory_mark(ctx, release_id=mark["mark_id"]))["reason"] == "invalid"  # a reason is required


# --- host facts helpers and boundaries ------------------------------------------------------------

def test_host_stamp_prefers_terminal_then_host_facts_then_task_result(tmp_path):
    from ouroboros.task_result_schema import SCHEMA_VERSION_KEY, TASK_RESULT_SCHEMA_VERSION
    from ouroboros.task_results import task_result_path

    path = task_result_path(tmp_path, "fromfile", create=True)
    path.write_text(json.dumps({SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION, "task_id": "fromfile",
                                "status": "cancelled"}), encoding="utf-8")
    facts = {"type": "task_summary", "summary_kind": "host_task_facts", "task_id": "both", "status": "completed",
             "outcome": "Done", "outcome_phase": "done"}
    terminal = {**facts, "summary_kind": "terminal_root_projection", "status": "failed", "outcome_phase": "error"}
    stamp = host_stamp(tmp_path, ["both", "facts", "fromfile", "nowhere"],
                       rows=[({}, terminal), ({}, facts), ({}, {**facts, "task_id": "facts"})])
    by_task = {entry["task_id"]: entry for entry in stamp["tasks"]}
    assert by_task["both"]["source"] == "terminal_root_projection" and by_task["both"]["status"] == "failed"
    assert by_task["facts"]["source"] == "host_task_facts"
    assert by_task["fromfile"]["source"] == "task_results" and by_task["fromfile"]["status"] == "cancelled"
    assert by_task["nowhere"] == {"task_id": "nowhere", "status": "not_recorded"}
    assert path.exists()  # the strict read never moves a result


def test_tools_module_imports_no_retired_memory_machinery_and_lazy_domains_stay_lazy():
    tree = ast.parse((REPO / "ouroboros" / "tools" / "chronicle.py").read_text(encoding="utf-8"))
    top, nested = set(), set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module not in top:
            nested.add(node.module)
    forbidden = {"ouroboros.chronicle_view", "ouroboros.chronicle_sources", "ouroboros.room_consolidation",
                 "ouroboros.consolidator", "ouroboros.llm", "ouroboros.memory_guidance"}
    assert not (top | nested) & forbidden
    lazy = {"ouroboros.dialogue_evidence", "ouroboros.project_dialogue", "ouroboros.projects_registry",
            "ouroboros.terminal_projection", "ouroboros.artifacts"}
    assert lazy <= nested and not lazy & top


# --- through the registry (root task) --------------------------------------------------------------

def test_registry_root_writes_to_the_canonical_data_root_and_reads_back(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    canonical, own = tmp_path / "canonical", tmp_path / "own-drive"
    canonical.mkdir()
    own.mkdir()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=own)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=own, task_id="root0001", current_chat_id=1,
                                     task_metadata={"budget_drive_root": str(canonical)}))
    for name in ("chronicle_write", "memory_read", "memory_mark"):
        schema = registry.get_schema_by_name(name)
        assert schema is not None and "author" not in schema["function"]["parameters"]["properties"]
    note = json.loads(registry.execute("chronicle_write", {"kind": "note", "text": "kept in the canonical root"}))
    assert note["ok"]
    assert json.loads(registry.execute("memory_mark", {"text": "remember", "node_id": note["node_id"]}))["ok"]
    assert ChronicleStore(canonical).get(note["node_id"])["text"] == "kept in the canonical root"
    assert not (own / "memory" / "chronicle").exists()
    result = registry.execute_result("memory_read", {})
    assert result.status == "ok" and result.text.startswith("room 1; head ")
    assert "kept in the canonical root" in result.text
    rows = chat(canonical, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"words {n}"} for n in (1, 2)])
    ranged = registry.execute_result("memory_read", {"rows": True, "from": addr(rows[1]), "to": addr(rows[1])})
    assert ranged.status == "ok" and ranged.text.split("\n")[2:] == [f"[{ts(2)}; User; {addr(rows[1])}] words 2"]


# --- delegated children (both child sets; pages and parts only as the child's drafts) --------------

CHILD_META = {"delegation_role": "subagent", "parent_task_id": "root0001", "root_task_id": "root0001"}


def test_both_child_sets_read_write_knowledge_mark_and_draft_but_neither_writes_identity():
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES, COGNITIVE_MEMORY_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    memory_names = COGNITIVE_MEMORY_TOOL_NAMES | {"chronicle_write"}
    child_memory = {"knowledge_read", "knowledge_list", "knowledge_write", "memory_read", "memory_mark", "chat_history",
                    "chronicle_write"}
    # Parity: a read-only and an acting child hold the same memory tools; chat_history is new for acting.
    # chronicle_write publishes only the child's drafts (tests/test_child_chronicle_drafts.py).
    assert LOCAL_READONLY_SUBAGENT_TOOL_NAMES & memory_names == child_memory
    assert ACTING_SUBAGENT_TOOL_NAMES & memory_names == child_memory
    for name in ("update_identity", "update_scratchpad"):
        assert name not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES and name not in ACTING_SUBAGENT_TOOL_NAMES


def test_readonly_child_registry_writes_signed_memory_and_is_refused_notes_identity_and_scratchpad(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry

    note = write(ctx_for(tmp_path), kind="note", text="the parent's note")
    assert note["ok"]
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(
        repo_dir=tmp_path, drive_root=tmp_path, task_id="kid00001", current_chat_id=1,
        task_metadata=dict(CHILD_META),
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False)))
    for name in ("knowledge_write", "memory_mark", "memory_read", "chronicle_write"):
        assert registry.get_schema_by_name(name) is not None
    for name in ("update_identity", "update_scratchpad"):
        assert registry.get_schema_by_name(name) is None
    # Allowed: a mark and a knowledge note, both signed with the child's focus.
    mark = json.loads(registry.execute("memory_mark", {"text": "the decision", "node_id": note["node_id"]}))
    assert mark["ok"]
    focus = ChronicleStore(tmp_path).get(mark["mark_id"])["author"]["focus"]
    assert focus["role"] == "child" and focus["parent_task_id"] == "root0001"
    saved = registry.execute("knowledge_write", {"topic": "people/rowan", "content": "Short reports.", "mode": "append"})
    assert "saved" in saved and "BLOCKED" not in saved
    # Refused: a chronicle note is the integrating mind's (a child only drafts pages and parts),
    # identity and scratchpad stay with the integrating parent.
    identity, scratchpad = tmp_path / "memory" / "identity.md", tmp_path / "memory" / "scratchpad.md"
    for path in (identity, scratchpad):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unchanged", encoding="utf-8")
    assert json.loads(registry.execute("chronicle_write", {"kind": "note", "text": "must not land"}))["reason"] == (
        "not_integrator")
    for name, args in (("update_identity", {"content": "must not land"}),
                       ("update_scratchpad", {"content": "must not land"})):
        assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute(name, args)
    notes = [record["id"] for record in ChronicleStore(tmp_path).room_records("1") if record["kind"] == "note"]
    assert notes == [note["node_id"]]
    assert identity.read_text(encoding="utf-8") == scratchpad.read_text(encoding="utf-8") == "unchanged"


def test_acting_child_registry_reads_chat_history_and_writes_memory_but_no_note(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    repo, data, worktree = tmp_path / "repo", tmp_path / "data", tmp_path / "wt"
    for path in (repo, data, worktree):
        path.mkdir()
    chat(data, [{"chat_id": 1, "direction": "in", "ts": ts(1), "text": "the owner's words"}])
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(
        repo_dir=repo, drive_root=data, task_id="kid00002", current_chat_id=1,
        task_metadata=dict(CHILD_META), workspace_root=str(worktree), workspace_mode="self_worktree",
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(worktree))))
    for name in ("chat_history", "knowledge_write", "memory_mark", "memory_read"):
        assert registry.get_schema_by_name(name) is not None
    history = registry.execute("chat_history", {"count": 5})
    assert "the owner's words" in history and "ACTING_SUBAGENT_BLOCKED" not in history
    assert registry.get_schema_by_name("chronicle_write") is not None
    refused = registry.execute("chronicle_write", {"kind": "note", "text": "must not land"})
    assert json.loads(refused)["reason"] == "not_integrator" and "ACTING_SUBAGENT_BLOCKED" not in refused
    assert not (data / "memory" / "chronicle" / "records.jsonl").exists()
