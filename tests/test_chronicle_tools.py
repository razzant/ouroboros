"""Memory authoring tools bind exact sources and preserve capability classes."""
import json

import pytest
from types import SimpleNamespace

from ouroboros.chronicle_store import ChronicleStore, source_row_id
from ouroboros.tools.chronicle import _chronicle_write, _memory_mark, _memory_read


def context(tmp_path):
    return SimpleNamespace(drive_root=tmp_path / "child", budget_drive_root=str(tmp_path),
                           task_metadata={"chat_id": 1}, current_chat_id=1, task_id="author",
                           _accumulated_usage={"_observed_route": {"provider": "actual", "model": "served"}})


def test_authored_episode_has_host_actor_and_no_invented_coverage(tmp_path):
    ctx = context(tmp_path)
    result = json.loads(_chronicle_write(ctx, text="I chose to investigate"))
    assert result["author"]["task_id"] == "author"
    assert result["author"]["route"] == ctx._accumulated_usage["_observed_route"]
    assert result["room_id"] == "1"
    assert result["metadata"]["coverage"] == "authored_without_source_range"
    assert result["metadata"]["task_ids"] == ["author"]
    assert ChronicleStore(tmp_path).scan_state() == {}
    assert not (ctx.drive_root / "memory" / "chronicle").exists()


@pytest.mark.parametrize("source_backed", [False, True])
def test_bound_writer_cannot_adopt_an_explicit_foreign_room(tmp_path, source_backed):
    from ouroboros.chronicle_view import capture_chronicle
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project, bind_task_to_project

    ctx, store = context(tmp_path), ChronicleStore(tmp_path)
    store.import_legacy()
    own = create_project(tmp_path, "writer-room")
    foreign = create_project(tmp_path, "source-room")
    bind_task_to_project(tmp_path, ctx.task_id, own["id"], origin={"absent": "system"})
    ctx.current_chat_id = ctx.task_metadata["chat_id"] = own["chat_id"]
    row = {"chat_id": foreign["chat_id"], "direction": "in", "text": "Owner has not chosen."}
    path = tmp_path / "logs/chat.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    page = json.loads(_memory_read(ctx, room_id=str(foreign["chat_id"]), raw_room=True))
    episode = json.loads(_chronicle_write(ctx, room_id=str(foreign["chat_id"]),
        text="I remember the other room's open choice.",
        **({"source_ref": page["source_ref"]} if source_backed else {})))
    assert episode["author"]["task_id"] == ctx.task_id
    assert episode["metadata"]["task_ids"] == []
    assert episode["metadata"]["source_row_ids"] == ([source_row_id(row)] if source_backed else [])
    mark = json.loads(_memory_mark(ctx, node_id=episode["id"], text="Still open"))
    journal = store.log_path.read_bytes()
    assert store.records_for_tasks([ctx.task_id]) == []
    store.index_path.unlink()  # Rebuilding the index must not invent writer membership.
    assert store.records_for_tasks([ctx.task_id]) == []
    for focus in (own["chat_id"], foreign["chat_id"]):
        snap = json.loads(capture_chronicle(Memory(tmp_path), {"id": "inspect", "chat_id": focus}))
        destinations = [room["id"] for room in snap["rooms"]
                        if any(record["id"] == episode["id"] for record in room["records"])]
        assert destinations == [str(foreign["chat_id"])]
        assert (mark["id"] in {m["id"] for m in snap["marks"]}) == (focus == foreign["chat_id"])
    assert store.log_path.read_bytes() == journal


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("source_backed", [False, True])
def test_own_main_episode_follows_its_source_task_promotion(tmp_path, explicit, source_backed):
    from ouroboros.chronicle_view import capture_chronicle
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project, bind_task_to_project

    ctx, store = context(tmp_path), ChronicleStore(tmp_path)
    store.import_legacy()
    source_task = "source-task" if source_backed else ctx.task_id
    kwargs = {"room_id": "1"} if explicit else {}
    if source_backed:
        path = tmp_path / "logs/chat.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"chat_id": 1, "task_id": source_task, "text": "Our decision"}) + "\n",
                        encoding="utf-8")
        kwargs["source_ref"] = json.loads(_memory_read(ctx, raw_room=True))["source_ref"]
    episode = json.loads(_chronicle_write(ctx, text="I remember our decision.", **kwargs))
    assert episode["metadata"]["task_ids"] == [source_task]
    assert episode["author"]["task_id"] == ctx.task_id
    room = create_project(tmp_path, "promoted")
    bind_task_to_project(tmp_path, source_task, room["id"], origin={"absent": "system"})
    snap = json.loads(capture_chronicle(Memory(tmp_path), {"id": "inspect", "chat_id": room["chat_id"]}))
    assert any(r["id"] == str(room["chat_id"]) and any(e["id"] == episode["id"] for e in r["records"])
               for r in snap["rooms"])
    assert store.get(episode["id"])["room_id"] == "1"  # Adoption never rewrites the original.


def test_explicit_episode_destination_needs_no_current_address(tmp_path):
    ctx = context(tmp_path)
    ctx.task_metadata = {}
    ctx.current_chat_id = None
    episode = json.loads(_chronicle_write(ctx, room_id="7", text="An explicitly addressed memory"))
    assert episode["room_id"] == "7" and episode["author"]["task_id"] == ctx.task_id
    assert episode["metadata"]["task_ids"] == []


def test_raw_room_read_retains_exact_page_and_write_binds_same_rows(tmp_path):
    ctx = context(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"chat_id": 7, "text": "owner first", "direction": "in", "ts": "2026-09-30T00:00:00Z"},
            {"chat_id": 2, "text": "another room", "direction": "in", "ts": "2026-09-30T00:00:01Z"},
            {"chat_id": 7, "text": "answer\nexact", "direction": "out", "ts": "2026-09-30T00:00:02Z"}]
    (logs / "chat.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = json.loads(_memory_read(ctx, room_id="7", raw_room=True, start=1, end=2))
    assert result["rows"] == [rows[2]]
    assert result["range"] == {"start": 1, "end": 2, "total": 2, "unit": "matching_rows"}
    assert result["page_complete"] is False
    assert result["source_row_ids"] == [source_row_id(rows[2])]
    episode = json.loads(_chronicle_write(ctx, text="I answered", source_ref=result["source_ref"]))
    assert episode["metadata"]["source_row_ids"] == result["source_row_ids"]
    assert episode["metadata"]["source_range"] == result["range"]
    assert episode["metadata"]["source_span"] == {
        "start": "2026-09-30T00:00:02+00:00", "end": "2026-09-30T00:00:02+00:00", "incomplete": False}
    mark = json.loads(_memory_mark(ctx, text="Remember exact answer", source_ref=result["source_ref"], quote="answer\nexact"))
    assert mark["quote"] == "answer\nexact"
    assert json.loads(_memory_mark(ctx, text="Bad", source_ref=result["source_ref"], quote="invented"))["error"]


def test_revision_read_and_rejection_keep_original(tmp_path):
    ctx = context(tmp_path)
    episode = json.loads(_chronicle_write(ctx, text="Original"))
    store = ChronicleStore(tmp_path)
    correction = store.revise(episode["id"], "Helper's correction", {"kind": "helper"})
    before = json.loads(_memory_read(ctx, node_id=episode["id"]))
    assert before["original"]["text"] == "Original"
    assert before["current"]["current_text"] == "Helper's correction"
    decision = json.loads(_chronicle_write(ctx, revision_id=correction["id"], decision="reject", reason="Source says otherwise"))
    assert decision["accepted"] is False
    after = json.loads(_memory_read(ctx, node_id=episode["id"]))
    assert after["current"]["current_text"] == "Original"
    assert after["current"]["revisions"][-1]["decision"]["reason"] == "Source says otherwise"


def test_mark_release_and_pagination_are_explicit(tmp_path):
    ctx = context(tmp_path)
    first = json.loads(_chronicle_write(ctx, text="first"))
    json.loads(_chronicle_write(ctx, text="second"))
    mark = json.loads(_memory_mark(ctx, text="Important", node_id=first["id"], scope="global", quote="first"))
    page = json.loads(_memory_read(ctx, limit=1))
    assert page["has_more"] is True
    assert len(page["records"]) == 1
    assert page["active_marks"][0]["id"] == mark["id"]
    next_page = json.loads(_memory_read(ctx, limit=1, after_seq=page["next_after_seq"]))
    assert next_page["records"][0]["text"] == "second"
    assert json.loads(_memory_mark(ctx, release_id=mark["id"]))["error"]
    json.loads(_memory_mark(ctx, release_id=mark["id"], reason="No longer current"))
    assert ChronicleStore(tmp_path).active_marks("1") == []


def test_catalog_and_readonly_capability_parity():
    from ouroboros.tools.knowledge import get_tools
    from ouroboros.tool_capabilities import (COGNITIVE_MEMORY_TOOL_NAMES, LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
                                             ACTING_SUBAGENT_TOOL_NAMES, UNTRUNCATED_TOOL_RESULTS)
    entries = {e.name: e for e in get_tools()}
    assert {"chronicle_write", "memory_read", "memory_mark"} <= entries.keys()
    assert {"chronicle_write", "memory_read", "memory_mark"} <= COGNITIVE_MEMORY_TOOL_NAMES
    assert "memory_read" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert not {"chronicle_write", "memory_mark"} & LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert ("chronicle_write" in ACTING_SUBAGENT_TOOL_NAMES) == ("knowledge_write" in ACTING_SUBAGENT_TOOL_NAMES)
    assert "memory_read" in UNTRUNCATED_TOOL_RESULTS
    for name in ("chronicle_write", "memory_mark"):
        assert "author" not in entries[name].schema["parameters"]["properties"]


def test_mind_can_change_mark_view_without_releasing_or_erasing_quote(tmp_path):
    ctx = context(tmp_path)
    episode = json.loads(_chronicle_write(ctx, text="verbatim source"))
    mark = json.loads(_memory_mark(ctx, text="The decision remains important", node_id=episode["id"], quote="verbatim source"))
    assert json.loads(_memory_mark(ctx, mark_id=mark["id"], visibility="meaning"))["error"]
    json.loads(_memory_mark(ctx, mark_id=mark["id"], visibility="meaning", reason="Need space for this task"))
    projected = ChronicleStore(tmp_path).active_marks("1")[0]
    assert projected["visibility"] == "meaning"
    assert projected["quote"] == "verbatim source"
    assert projected["text"] == "The decision remains important"
    assert projected["view_decision"]["author"]["task_id"] == "author"
    ChronicleStore(tmp_path).index_path.unlink()
    assert ChronicleStore(tmp_path).active_marks("1")[0]["visibility"] == "meaning"
    json.loads(_memory_mark(ctx, mark_id=mark["id"], visibility="full", reason="Return to the exact words"))
    assert ChronicleStore(tmp_path).active_marks("1")[0]["visibility"] == "full"
    assert ChronicleStore(tmp_path).get(mark["id"])["quote"] == "verbatim source"


def test_readonly_registry_exposes_read_and_denies_actual_memory_writes(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_chat_id=1,
        task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False)))
    assert registry.get_schema_by_name("memory_read") is not None
    for name in ("chronicle_write", "memory_mark"):
        assert registry.get_schema_by_name(name) is None
        assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in registry.execute(name, {"text": "must not land"})
    assert not (tmp_path / "memory" / "chronicle" / "records.jsonl").exists()


@pytest.mark.parametrize("source_format", ["list", "chunks", "locators"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_retained_source_formats_page_exact_rows_and_credit_only_that_page(tmp_path, monkeypatch, source_format, newline):
    from ouroboros.chronicle_sources import capture_room, retain_room_source
    from ouroboros.consolidator import retain_memory_source
    from ouroboros.memory import Memory
    from ouroboros import chronicle_sources

    ctx = context(tmp_path)
    rows = [{"chat_id": 7, "text": "first original words", "task_id": "first"},
            {"chat_id": 7, "text": "second original words", "task_id": "second"}]
    chat = tmp_path / "logs/chat.jsonl"
    chat.parent.mkdir()
    source_lines = [(json.dumps(row) + newline).encode("utf-8") for row in rows]
    chat.write_bytes(b"".join(source_lines))
    if source_format == "list":
        ref = retain_memory_source(SimpleNamespace(drive_root=tmp_path, task_id=ctx.task_id),
                                  "episode", json.dumps(rows).encode("utf-8"), "json")
    else:
        memory = Memory(tmp_path)
        captured, coverage = capture_room(memory, "7", rendered_chars_budget=1 if source_format == "locators" else None)
        locators = coverage.pop("row_locators")
        if source_format == "locators":
            # Even a misleading stored path cannot redirect the source reader.
            for locator in locators:
                locator["path"] = str(tmp_path / "not-a-chat-source.json")
        ref = retain_room_source(memory, ctx.task_id, captured, locators, coverage)
    archive = tmp_path / "archive"
    archive.mkdir()
    chat.rename(archive / "chat_20260101T000000.jsonl")
    chat.write_text(json.dumps({"chat_id": 7, "text": "later uncaptured words"}) + "\n", encoding="utf-8")
    read_sizes = []
    original = chronicle_sources.JsonlChainSnapshot._read
    def counted(self, start, end):
        read_sizes.append(end - start)
        return original(self, start, end)
    monkeypatch.setattr(chronicle_sources.JsonlChainSnapshot, "_read", counted)
    page = json.loads(_memory_read(ctx, source_ref=ref, start=1, end=2))
    assert page["rows"] == rows[1:] and page["source_row_ids"] == [source_row_id(rows[1])]
    assert page["range"] == {"start": 1, "end": 2, "total": 2, "unit": "matching_rows"}
    assert page["range_complete"] and not page["page_complete"]
    assert page["source_ref"] != ref
    assert "first original words" not in json.dumps(page) and "later uncaptured words" not in json.dumps(page)
    if source_format == "locators":
        # Locators address physical bytes, including the stored line ending.
        assert sum(read_sizes) == len(source_lines[1])
    full = json.loads(_memory_read(ctx, source_ref=ref))
    assert full["rows"] == rows and full["page_complete"] and full["range_complete"]
    assert page["parent_source_ref"] == ref
    episode = json.loads(_chronicle_write(ctx, text="I recalled only the second source.", source_ref=page["source_ref"]))
    assert episode["metadata"]["source_row_ids"] == [source_row_id(rows[1])]
    assert "first" not in episode["metadata"]["task_ids"]
    assert episode["metadata"]["source_range"] == page["range"]
    (archive / "chat_20260101T000000.jsonl").unlink()
    assert json.loads(_memory_read(ctx, source_ref=page["source_ref"]))["rows"] == rows[1:]
    assert json.loads(_memory_mark(ctx, text="Exact page", room_id="7", source_ref=page["source_ref"],
                                   quote="second original words"))["quote"] == "second original words"


@pytest.mark.parametrize("damage", ["missing", "rewritten", "identity"])
def test_locator_source_gaps_do_not_claim_complete_or_unread_source_credit(tmp_path, damage):
    from ouroboros.chronicle_sources import capture_room, retain_room_source
    from ouroboros.memory import Memory

    ctx, memory = context(tmp_path), Memory(tmp_path)
    rows = [{"chat_id": 7, "text": "original A"}, {"chat_id": 7, "text": "original B"}]
    chat = tmp_path / "logs/chat.jsonl"
    chat.parent.mkdir()
    raw = "".join(json.dumps(row) + "\n" for row in rows)
    chat.write_text(raw, encoding="utf-8")
    captured, coverage = capture_room(memory, "7", rendered_chars_budget=1)
    locators = coverage.pop("row_locators")
    if damage == "identity":
        locators[0]["source_row_id"] = "not-this-row"
    ref = retain_room_source(memory, ctx.task_id, captured, locators, coverage)
    if damage == "missing":
        chat.unlink()
    elif damage == "rewritten":
        chat.write_text(raw.replace("original A", "rewrittenA"), encoding="utf-8")
    page = json.loads(_memory_read(ctx, source_ref=ref))
    assert page["rows"] == ([] if damage == "missing" else rows[1:])
    assert not page["page_complete"] and not page["range_complete"]
    assert page["missing"] and page["coverage"]["complete"] is False
    assert page["range"]["total"] == 2
    assert "rewrittenA" not in json.dumps(page)
    episode = json.loads(_chronicle_write(ctx, text="My source has an explicit gap.", source_ref=page["source_ref"]))
    assert episode["metadata"]["source_row_ids"] == [source_row_id(row) for row in page["rows"]]
    assert episode["metadata"]["source_coverage"]["complete"] is False
    reread = json.loads(_memory_read(ctx, source_ref=page["source_ref"]))
    assert reread["rows"] == page["rows"] and not reread["range_complete"]


def test_mind_interim_covers_hundreds_of_unfinished_foreign_rooms_through_registry(tmp_path):
    from ouroboros.chronicle_view import capture_chronicle, render_memory
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    store = ChronicleStore(tmp_path)
    store.import_legacy()
    rooms = [create_project(tmp_path, f"unfinished-{n}", name=f"Unfinished project {n}") for n in range(200)]
    rows = [{"chat_id": room["chat_id"], "task_id": f"unfinished-task-{n}", "direction": "in",
             "status": "waiting_owner", "ts": "2000-01-01T00:00:00Z",
             "text": f"Question {n} is still unanswered. " + "Original reasoning and alternatives. " * 100}
            for n, room in enumerate(rooms)]
    chat = tmp_path / "logs/chat.jsonl"
    chat.parent.mkdir(exist_ok=True)
    original = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    chat.write_bytes(original)
    state = tmp_path / "state/queue_snapshot.json"
    state.parent.mkdir(exist_ok=True)
    statuses = json.dumps({row["task_id"]: "waiting_owner" for row in rows}).encode("utf-8")
    state.write_bytes(statuses)
    memory = Memory(tmp_path)
    before = json.loads(capture_chronicle(memory, {"id": "before", "chat_id": 1}))
    assert sum(len(room["rows"]) for room in before["other_open_rooms"]) == 200
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_chat_id=1, task_id="remember"))
    assert registry.get_schema_by_name("chronicle_write") is not None
    for n, room in enumerate(rooms):
        page = json.loads(registry.execute("memory_read", {"room_id": str(room["chat_id"]), "raw_room": True}))
        assert page["rows"] == [rows[n]]
        episode = json.loads(registry.execute("chronicle_write", {
            "room_id": str(room["chat_id"]), "source_ref": page["source_ref"],
            "text": f"Project {n}: I compared alternatives; the owner has not chosen. Work remains unfinished."}))
        assert episode["author"]["kind"] == "mind"
        assert episode["metadata"]["source_row_ids"] == [source_row_id(rows[n])]
    after = json.loads(capture_chronicle(memory, {"id": "after", "chat_id": 1}))
    compact, _ = render_memory(after, 0, shared_out={})
    assert after["other_open_rooms"] == []
    for n in range(200):
        assert f"Project {n}: I compared alternatives; the owner has not chosen." in compact
        assert rows[n]["text"] not in compact
    assert len(compact) < len(render_memory(before)[0]) / 4
    assert chat.read_bytes() == original and state.read_bytes() == statuses
    assert len(store.records(kinds=["episode"])) == 200
    assert not store.records(kinds=["digest", "revision"])


@pytest.mark.parametrize("finished", [False, True], ids=["still-open", "cancelled"])
def test_interim_coverage_keeps_current_open_arc_raw_without_reopening_cancelled_task(tmp_path, finished):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.chronicle_view import capture_chronicle, refresh_chronicle_snapshot, render_memory
    from ouroboros.memory import Memory
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    store = ChronicleStore(tmp_path)
    store.import_legacy()
    room = create_project(tmp_path, "interim", name="Interim project")
    chat_id = room["chat_id"]
    original = {"chat_id": chat_id, "task_id": "interim-task", "direction": "in",
                "status": "waiting_owner", "text": "Owner's exact still-unanswered question. " * 100}
    rows = [original]
    if finished:
        rows.append({"chat_id": chat_id, "task_id": "interim-task", "direction": "system",
                     "type": "task_summary", "summary_kind": "terminal_root_projection",
                     "outcome_authority": "canonical_task_result_after_finalization", "outcome_final": True,
                     "outcome_phase": "done", "status": "cancelled", "text": "Task was cancelled; no approval granted."})
    chat = tmp_path / "logs/chat.jsonl"
    chat.parent.mkdir(exist_ok=True)
    source_bytes = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    chat.write_bytes(source_bytes)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_chat_id=1, task_id="remember"))
    page = json.loads(registry.execute("memory_read", {"room_id": str(chat_id), "raw_room": True}))
    meaning = "I retain the cancellation; no approval was granted." if finished else "I have not received the owner's choice; the question remains open."
    registry.execute("chronicle_write", {"room_id": str(chat_id), "source_ref": page["source_ref"], "text": meaning})
    memory = Memory(tmp_path)
    for source_budget in (None, 1):
        captured = capture_chronicle(memory, {"id": "return", "chat_id": chat_id}, rendered_chars_budget=source_budget)
        for snapshot in (json.loads(captured), json.loads(refresh_chronicle_snapshot(captured, tmp_path))):
            text, facts = render_memory(snapshot, 0, shared_out={})
            assert (original["text"] in text) is not finished
            assert meaning in text and facts["target_miss"]
            assert bool(snapshot["open_focus"]) is not finished
            if source_budget == 1:
                assert snapshot["raw_focus"] == []
                retained = json.loads(read_actor_source_bytes(tmp_path, "return", snapshot["source_ref"]))
                assert "row_locators" in retained and "source_chunks" not in retained
    returned = json.loads(capture_chronicle(memory, {"id": "roomy-return", "chat_id": chat_id}))
    assert original["text"] in render_memory(returned, None)[0]
    assert json.loads(registry.execute("memory_read", {"source_ref": page["source_ref"]}))["rows"] == rows
    assert chat.read_bytes() == source_bytes


@pytest.fixture
def raw_room_registry(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (data / "logs").mkdir(parents=True)
    rows = [{"chat_id": 7, "direction": "in", "task_id": f"original-{index}",
             "text": f"Original owner message {index}",
             "ts": f"2026-10-01T12:{index // 60:02}:{index % 60:02}Z"} for index in range(75)]
    foreign = {"chat_id": 99, "direction": "in", "text": "Another room"}
    (data / "logs/chat.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in [foreign, *rows]), encoding="utf-8")
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data, current_chat_id=7,
                                    task_id="raw-page-reader", task_metadata={"chat_id": 7}))
    return registry, rows


def test_raw_room_limit_pages_real_registry_and_binds_only_returned_sources(raw_room_registry):
    registry, rows = raw_room_registry
    seen, identifiers = [], []
    for start in (0, 20, 40, 60):
        result = registry.execute_result("memory_read", {"raw_room": True, "start": start, "limit": 20})
        assert result.status == "ok"
        page = json.loads(result.text)
        end = min(start + 20, len(rows))
        assert page["rows"] == rows[start:end]
        assert page["range"] == {"start": start, "end": end, "total": 75, "unit": "matching_rows"}
        assert page["page_complete"] is False
        expected_ids = [source_row_id(row) for row in rows[start:end]]
        assert page["source_row_ids"] == expected_ids
        retained = json.loads(registry.execute("memory_read", {"source_ref": page["source_ref"]}))
        assert retained["rows"] == page["rows"] and retained["source_row_ids"] == expected_ids
        episode = json.loads(registry.execute("chronicle_write", {
            "text": "I inspected only this page.", "source_ref": page["source_ref"]}))
        assert episode["metadata"]["source_row_ids"] == expected_ids
        assert episode["metadata"]["source_range"] == page["range"]
        assert set(episode["metadata"]["task_ids"]) == {row["task_id"] for row in rows[start:end]}
        assert episode["author"]["task_id"] == "raw-page-reader"
        seen.extend(page["rows"])
        identifiers.extend(expected_ids)
    assert seen == rows and len(set(identifiers)) == 75


@pytest.mark.parametrize("arguments,start,end", [
    ({"start": 10, "end": 35, "limit": 20}, 10, 30),
    ({"start": 10, "end": 15, "limit": 20}, 10, 15),
    ({"start": 10, "end": 35}, 10, 35),
    ({"start": 70}, 70, 75),
    ({}, 0, 75),
    ({"limit": 100}, 0, 75),
])
def test_raw_room_limit_respects_end_and_keeps_unbounded_reads(raw_room_registry, arguments, start, end):
    registry, rows = raw_room_registry
    result = registry.execute_result("memory_read", {"raw_room": True, **arguments})
    assert result.status == "ok"
    page = json.loads(result.text)
    assert page["rows"] == rows[start:end]
    assert page["range"] == {"start": start, "end": end, "total": 75, "unit": "matching_rows"}
    assert page["page_complete"] is (start == 0 and end == 75)


@pytest.mark.parametrize("room", [0, 1, 42017], ids=["hidden", "main", "project"])
def test_default_room_tools_preserve_hidden_main_and_project_addresses(tmp_path, room):
    ctx = context(tmp_path)
    ctx.task_metadata = {"chat_id": room}
    ctx.current_chat_id = room
    logs = tmp_path / "logs"
    logs.mkdir()
    row = {"chat_id": room, "direction": "in", "text": "Exact source words"}
    (logs / "chat.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    page = json.loads(_memory_read(ctx, raw_room=True, limit=1))
    assert page["room_id"] == str(room) and page["rows"] == [row]
    episode = json.loads(_chronicle_write(ctx, text="I understood the source", source_ref=page["source_ref"]))
    assert episode["room_id"] == str(room)
    readback = json.loads(_memory_read(ctx))
    assert readback["room_id"] == str(room) and readback["records"][0]["id"] == episode["id"]
    mark = json.loads(_memory_mark(ctx, text="Keep this source", source_ref=page["source_ref"], quote="Exact source words"))
    assert mark["room_id"] == str(room) and mark["quote"] == row["text"]


def test_default_room_uses_first_present_address_not_first_truthy_address(tmp_path):
    from ouroboros.tools.chronicle import _room

    ctx = context(tmp_path)
    ctx.task_metadata = {"chat_id": 0}
    ctx.current_chat_id = 7
    ctx.chat_id = 9
    assert _room(ctx) == "0"
    assert _room(ctx, "12") == "12"
    ctx.task_metadata = {"chat_id": None}
    ctx.current_chat_id = 0
    assert _room(ctx) == "0"
    ctx.current_chat_id = None
    assert _room(ctx) == "9"
    ctx.chat_id = None
    with pytest.raises(ValueError, match="room_id is required"):
        _room(ctx)
