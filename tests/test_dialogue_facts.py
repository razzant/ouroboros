"""The five writer/import facts the dialogue readers address (unit Q1c): the chain readers'
counts (ordinals and stream positions), exact row ends and the skipped intervals, every legacy
``(gen, line)`` alias with an indexed reverse lookup, the ``subagent_task_id`` probe, and a room
revision that continues from the project's ``visible_revision``. Each fact is asserted against
the old readers or the chain files themselves, in both directions, and the probes are pinned to
their indexes (no scan of rows or addresses)."""
from __future__ import annotations

import contextlib
import json
import pathlib
import sqlite3
from typing import Any, Dict, List, NamedTuple, Optional

import pytest

from ouroboros import chat_chain, dialogue_import as di, dialogue_positions as dp, dialogue_store as ds, store
from ouroboros.utils import jsonl_generation_signature
from tests._dialogue import seed


def owner(cmid, *, chat_id=1, ts="2026-10-17T00:00:01+00:00", **extra):
    return {"ts": ts, "direction": "in", "chat_id": chat_id, "client_message_id": cmid, "text": f"words {cmid}", **extra}


def progress(task_id, text="step", **extra):
    return {"ts": "2026-10-17T00:00:02+00:00", "type": "send_message", "is_progress": True, "direction": "out",
            "task_id": task_id, "text": text, **extra}


def encoded(row, **kwargs):
    return json.dumps(row, ensure_ascii=False, **kwargs).encode("utf-8")


class Expected(NamedTuple):
    ordinal: int
    pos: Optional[int]
    end: int
    gen: str
    line: int


def chain_oracle(root: pathlib.Path, stream: str) -> Dict[int, Expected]:
    """What the chain files say about every stored row: its counts as the JSONL readers counted
    them (every parsed object; chat rows outside A2A), its exact end, its generation and line."""
    archive = root / "archive"
    paths = sorted(archive.glob(f"{stream}_*.jsonl"), key=lambda p: p.name) + [root / "logs" / f"{stream}.jsonl"]
    expected, seq, ordinal, pos = {}, 0, 0, 0
    for path in paths:
        if not path.exists():
            continue
        gen = str(jsonl_generation_signature(path).get("first_line_sha256") or "")
        for number, raw in enumerate(path.read_bytes().splitlines(keepends=True), 1):
            try:
                row = json.loads(raw.decode("utf-8").strip()) if raw.strip() else None
            except ValueError:
                row = None
            if isinstance(row, dict):
                counted = stream == "chat" and not chat_chain.is_a2a_chat_id(row.get("chat_id", 1))
                expected[seq] = Expected(ordinal, pos if counted else None, seq + len(raw), gen, number)
                ordinal, pos = ordinal + 1, pos + int(counted)
            seq += len(raw)
    return expected


A2A_ROW = {"ts": "2026-10-17T00:00:01+00:00", "chat_id": -5, "direction": "out", "text": "machine traffic"}
COMPACT = encoded(owner("a2", chat_id=2), separators=(",", ":")) + b"\r\n"  # a compact CRLF line
CHILD_ROW = {"ts": "2026-10-17T00:00:03+00:00", "direction": "out", "chat_id": 1, "task_id": "t1",
             "subagent_task_id": "child-1", "text": "child words"}
UNTERMINATED = encoded(owner("live-3"))  # the chain's last line, without its newline


def chain_root(root: pathlib.Path, *, non_object: bool = True) -> None:
    """Two chat archives plus a live file with A2A, skipped, blank, CRLF and unterminated lines;
    a progress archive with a row that carries only ``subagent_task_id``. (``chat_chain``'s own
    generation reader cannot read past a non-object JSON line; an oracle built on it leaves it out.)"""
    seed(root, archives={
        "chat_20261001T000000.jsonl": [owner("a1"), A2A_ROW, b"{not json\n", *([b"[1, 2]\n"] if non_object else []), b"\n",
                                       COMPACT, owner("a3", chat_id=2, task_id="t1")],
        "chat_20261002T000000.jsonl": [CHILD_ROW, owner("b2")],
        "progress_20261001T000000.jsonl": [progress("t1", "step 1"), b"garbage\n",
                                           {"ts": "2026-10-17T00:00:02+00:00", "subagent_task_id": "child-1",
                                            "text": "child step"}, progress("t2")],
    }, live={"chat": [owner("live-1"), {**A2A_ROW, "chat_id": "-7"}, UNTERMINATED],
             "progress": [progress("t1", "live step")]})


@contextlib.contextmanager
def traced(monkeypatch):
    """The expanded SQL statements and VM steps of every store transaction in the block."""
    counts: Dict[str, Any] = {"statements": [], "steps": 0}
    transaction = ds.transaction

    @contextlib.contextmanager
    def wrapped(*args, **kwargs):
        with transaction(*args, **kwargs) as conn:
            conn.set_trace_callback(counts["statements"].append)
            conn.set_progress_handler(lambda: counts.__setitem__("steps", counts["steps"] + 1) or 0, 1)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
                conn.set_progress_handler(None, 0)

    with monkeypatch.context() as patched:
        patched.setattr(ds, "transaction", wrapped)
        yield counts


def plans(root, statements: List[str]) -> List[str]:
    """The query plan details of the traced statements (the probes replayed with their values)."""
    details = []
    with ds.transaction(root, write=False) as conn:
        for sql in statements:
            if sql.lstrip().upper().startswith(("SELECT", "UPDATE", "INSERT")):
                details.extend(str(record[3]) for record in conn.execute(f"EXPLAIN QUERY PLAN {sql}"))
    assert details, statements
    return details


def gen_of(root, name) -> str:
    return jsonl_generation_signature(root / "archive" / name)["first_line_sha256"]


# ---- fact 1: durable ordinals -------------------------------------------------------------------

def test_ordinals_are_the_chain_readers_counts_and_unknown_until_the_archive_is_complete(tmp_path):
    chain_root(tmp_path)
    oracle = {stream: chain_oracle(tmp_path, stream) for stream in ds.STREAMS}
    old = [e.pos for e in oracle["chat"].values() if e.pos is not None]
    assert old == list(range(7)), "independent frozen-file oracle, including A2A exclusion"
    live_gen = jsonl_generation_signature(tmp_path / "logs" / "chat.jsonl")["first_line_sha256"]
    di.cutover(tmp_path)
    live_seq = ds.rows_by_client_id(tmp_path, "live-1")[0].seq
    for unsettled in (lambda: dp.ordinal_at(tmp_path, "chat", live_seq), lambda: dp.row_at_position(tmp_path, 0),
                      lambda: dp.row_at_ordinal(tmp_path, "progress", 0), lambda: dp.generation_row(tmp_path, live_gen, 2),
                      lambda: dp.generation_row(tmp_path, gen_of(tmp_path, "chat_20261001T000000.jsonl"), 0)):
        with pytest.raises(ds.StoreMigrating, match="still importing"):
            unsettled()
    assert dp.generation_row(tmp_path, live_gen, 0).row == owner("live-1"), "inside an imported unit the cursor resolves"
    assert dp.generation_row(tmp_path, live_gen, 1).row == owner("live-3"), "A2A rows take no position"
    di.run_dialogue_import(tmp_path)
    for stream in ds.STREAMS:
        rows = ds.newest(tmp_path, stream, limit=1000)
        assert {row.seq for row in rows} == set(oracle[stream])
        counts = dp.ordinals_of(tmp_path, stream, [row.seq for row in rows])
        for row in rows:
            expected = oracle[stream][row.seq]
            assert counts[row.seq] == (expected.ordinal, expected.pos) == dp.ordinal_at(tmp_path, stream, row.seq)
            assert dp.row_at_ordinal(tmp_path, stream, expected.ordinal).seq == row.seq
            if expected.pos is not None:
                assert dp.row_at_position(tmp_path, expected.pos).seq == row.seq
        assert dp.row_at_ordinal(tmp_path, stream, len(rows)) is None and dp.ordinals_of(tmp_path, stream, [1 << 40]) == {}
    assert dp.row_at_position(tmp_path, len(old)) is None
    assert dp.ordinal_at(tmp_path, "chat", ds.probe_rows(tmp_path, "chat", [("chat_id", -5)])[0].seq).pos is None


def test_rows_appended_after_the_cutover_continue_the_same_counting(tmp_path):
    chain_root(tmp_path)
    di.cutover(tmp_path)
    during = [ds.append(tmp_path, "chat", owner("during-1")), ds.append(tmp_path, "chat", {**A2A_ROW, "chat_id": -1}),
              ds.append(tmp_path, "chat", owner("during-2"))]
    with pytest.raises(ds.StoreMigrating):
        dp.ordinal_at(tmp_path, "chat", during[0].seq)
    di.run_dialogue_import(tmp_path)
    objects, positions = len(chain_oracle(tmp_path, "chat")), 7
    assert dp.ordinal_at(tmp_path, "chat", during[0].seq) == (objects, positions)
    assert dp.ordinal_at(tmp_path, "chat", during[1].seq) == (objects + 1, None)
    assert dp.ordinal_at(tmp_path, "chat", during[2].seq) == (objects + 2, positions + 1)
    assert dp.row_at_position(tmp_path, positions + 1).seq == during[2].seq
    later = ds.append(tmp_path, "progress", progress("t3"))
    assert dp.ordinal_at(tmp_path, "progress", later.seq) == (len(chain_oracle(tmp_path, "progress")), None)
    assert dp.row_at_ordinal(tmp_path, "progress", 4).seq == later.seq
    fresh = tmp_path / "fresh"
    first = ds.append(fresh, "chat", owner("first"))
    assert dp.ordinal_at(fresh, "chat", first.seq) == (0, 0), "a root without chain files counts from zero at once"
    assert dp.row_at_position(fresh, 1) is None
    child = tmp_path / "child"
    seed(child, live={"chat": [A2A_ROW, owner("only-live")]})  # a child root: its own small live file, no archive
    appended = ds.append(child, "chat", owner("child-new"))
    assert dp.ordinal_at(child, "chat", appended.seq) == (2, 1) and dp.row_at_position(child, 0).row == owner("only-live")


def test_generation_cursors_resolve_as_the_old_consolidation_reader_did(tmp_path):
    chain_root(tmp_path, non_object=False)
    live = tmp_path / "logs" / "chat.jsonl"
    paths = sorted((tmp_path / "archive").glob("chat_*.jsonl")) + [live]

    def legacy_rows(path):
        result = []
        for raw in path.read_bytes().splitlines():
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            if isinstance(row, dict) and not chat_chain.is_a2a_chat_id(row.get("chat_id", 1)):
                result.append(row)
        return result

    cases = []
    for index, path in enumerate(paths):
        signature = jsonl_generation_signature(path)
        pending = [entry for segment in paths[index:] for entry in legacy_rows(segment)]
        for offset in range(len(legacy_rows(path)) + 2):
            cases.append((signature["first_line_sha256"], offset, pending[offset] if offset < len(pending) else None))
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    for gen, offset, expected in cases:
        found = dp.generation_row(tmp_path, gen, offset)
        assert (None if found is None else found.row) == expected, (gen, offset)
    assert dp.generation_row(tmp_path, "0" * 64, 0) is None and dp.generation_row(tmp_path, "", 0) is None
    assert dp.generation_row(tmp_path, gen_of(tmp_path, "chat_20261002T000000.jsonl"), -1) is None


def test_a_segment_that_joins_after_the_cutover_counts_at_its_chain_place(tmp_path):
    chain_root(tmp_path)
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    exported = ds.append(tmp_path, "chat", owner("new"))
    assert dp.ordinal_at(tmp_path, "chat", exported.seq) == (9, 7)
    di.export_for_older_body(tmp_path)
    seed(tmp_path, live={"chat": [owner("old-1"), {**A2A_ROW, "chat_id": -2}, owner("old-2")]})  # an older body appended
    joined_gen = jsonl_generation_signature(tmp_path / "logs" / "chat.jsonl")["first_line_sha256"]
    report = di.cutover(tmp_path)
    assert report["chat"]["imported"] == {"status": "done", "rows": 4, "bad": 0}
    old_1, old_2 = (ds.rows_by_client_id(tmp_path, cmid)[0] for cmid in ("old-1", "old-2"))
    assert dp.ordinal_at(tmp_path, "chat", exported.seq) == (9, 7), "a row the export placed keeps its counts"
    assert dp.ordinal_at(tmp_path, "chat", old_1.seq) == (10, 8) and dp.ordinal_at(tmp_path, "chat", old_2.seq) == (12, 9)
    unit = ds.legacy_generation(tmp_path, joined_gen)
    assert (unit["rows"], unit["stream_rows"], unit["ordinal_base"], unit["pos_base"]) == (4, 3, 9, 7)
    assert dp.generation_row(tmp_path, joined_gen, 1).seq == old_1.seq and dp.generation_row(tmp_path, joined_gen, 2).seq == old_2.seq
    assert ds.legacy_address(tmp_path, old_2.seq) == (joined_gen, 4)
    after = ds.append(tmp_path, "chat", owner("after"))
    assert dp.ordinal_at(tmp_path, "chat", after.seq) == (13, 10) and dp.row_at_ordinal(tmp_path, "chat", 11).row["chat_id"] == -2


def test_a_segment_that_cannot_be_imported_makes_the_counts_a_typed_error(tmp_path):
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [owner("archived")]}, live={"chat": [owner("live")]})
    di.cutover(tmp_path)
    live = ds.rows_by_client_id(tmp_path, "live")[0]
    with pytest.raises(ds.StoreMigrating):
        dp.ordinals_of(tmp_path, "chat", [live.seq])
    (tmp_path / "archive" / "chat_20261001T000000.jsonl").write_text("changed after the cutover\n", encoding="utf-8")
    di.run_dialogue_import(tmp_path)
    with pytest.raises(store.StoreError, match="could not be imported"):
        dp.ordinals_of(tmp_path, "chat", [live.seq])
    with pytest.raises(store.StoreError, match="could not be imported"):
        dp.row_at_position(tmp_path, 0)


# ---- fact 2: exact row ends and the skipped intervals -----------------------------------------

def test_rows_keep_their_exact_physical_length_and_skipped_lines_are_recorded_intervals(tmp_path):
    chain_root(tmp_path)
    oracle = chain_oracle(tmp_path, "chat")
    raw = (tmp_path / "archive" / "chat_20261001T000000.jsonl").read_bytes()
    bad, non_object = raw.index(b"{not json"), raw.index(b"[1, 2]")
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    for row in ds.newest(tmp_path, "chat", limit=1000):
        assert row.end == row.seq + row.length == oracle[row.seq].end, "seq + length is the exact end"
    compact = ds.rows_by_client_id(tmp_path, "a2")[0]
    assert compact.length == len(COMPACT) and compact.end != compact.seq + len(ds.encode(compact.row)) + 1, "CRLF and compact JSON are not re-encoded away"
    last = ds.rows_by_client_id(tmp_path, "live-3")[0]
    assert last.length == len(UNTERMINATED) and last.end == ds.stream_end(tmp_path, "chat"), "the unterminated last line ends where the chain ends"
    assert dp.skipped_between(tmp_path, "chat") == [dp.Skipped("chat", bad, len(b"{not json\n"), "invalid"),
                                                     dp.Skipped("chat", non_object, len(b"[1, 2]\n"), "non_object")]
    assert dp.skipped_between(tmp_path, "chat")[0].end == non_object and dp.skipped_between(tmp_path, "chat", lower=non_object + 1) == []
    assert dp.skipped_between(tmp_path, "progress")[0].kind == "invalid" and len(dp.skipped_between(tmp_path, "progress")) == 1
    assert ds.range_coverage(tmp_path, "chat") == {"imported": True, "pending": 0, "mismatch": 0, "gaps": 2}, "a skipped line is a gap, not coverage"
    assert ds.range_coverage(tmp_path, "chat", lower=bad, upper=non_object)["gaps"] == 1
    assert ds.range_coverage(tmp_path, "chat", lower=non_object + 1)["gaps"] == 0
    assert di.coverage(tmp_path, "chat")["gaps"] == 2 and di.coverage(tmp_path, "chat", bad)["gaps"] == 0
    with ds.transaction(tmp_path, write=False) as conn:
        assert conn.execute("SELECT bad FROM import_ledger WHERE name='chat_20261001T000000.jsonl'").fetchone()["bad"] == 2, "the blank line is neither a row nor a gap"
    receipt = ds.append(tmp_path, "chat", owner("appended", text="ünïcode"))
    assert receipt.length == len(ds.encode(receipt.row).encode("utf-8")) + 1 and receipt.end == ds.stream_end(tmp_path, "chat")
    assert ds.row_at(tmp_path, "chat", receipt.seq).end == receipt.end
    assert ds.append(tmp_path, "chat", owner("appended")).end == receipt.end, "a duplicate answers the existing row's end"


# ---- fact 3: reverse legacy addresses, every alias ---------------------------------------------

def imported_rooms(root, rooms):
    rows = [owner(f"foreign-{room}-{index}", chat_id=room + 10) for room in range(rooms) for index in range(10)] + [owner("selected")]
    seed(root, archives={"chat_20261016T000000.jsonl": rows})
    di.cutover(root)
    di.run_dialogue_import(root)
    return ds.rows_by_client_id(root, "selected")[0]


def test_reverse_legacy_lookup_is_one_indexed_probe_whatever_the_other_rooms_imported(tmp_path, monkeypatch):
    results, statements = {}, []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        row = imported_rooms(root, rooms)
        with traced(monkeypatch) as counts:
            assert ds.legacy_address(root, row.seq) == (gen_of(root, "chat_20261016T000000.jsonl"), 10 * rooms + 1)
        results[rooms] = (len(counts["statements"]), counts["steps"])
        statements = counts["statements"]
    assert results[1] == results[30] and results[1][1] > 0
    details = plans(root, statements)
    assert any("legacy_seq" in detail for detail in details) and not any(detail.startswith("SCAN") for detail in details), details


def test_repeated_generation_signatures_keep_every_alias(tmp_path):
    shared = owner("shared", text="the same first line twice")
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [shared, owner("first-second")],
                             "chat_20261002T000000.jsonl": [shared, owner("second-second")]})
    gen = gen_of(tmp_path, "chat_20261001T000000.jsonl")
    assert gen == gen_of(tmp_path, "chat_20261002T000000.jsonl")
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    first = [row.seq for row in ds.legacy_candidates(tmp_path, gen, 1)]
    assert [ds.row_at(tmp_path, "chat", seq).row for seq in first] == [shared, shared] and first == sorted(first)
    second = ds.legacy_candidates(tmp_path, gen, 2)
    assert [row.row["client_message_id"] for row in second] == ["first-second", "second-second"], "chain order"
    assert ds.resolve_legacy(tmp_path, gen, 2).row == second[0].row and ds.legacy_candidates(tmp_path, gen, 3) == []
    assert ds.legacy_address(tmp_path, second[1].seq) == (gen, 2) and ds.legacy_address(tmp_path, first[1]) == (gen, 1)
    assert ds.legacy_generation(tmp_path, gen)["name"] == "chat_20261001T000000.jsonl", "the first generation, as the chain reader chose"
    for row in second:  # the old reader told the aliases apart by the row's own hash; the candidates let a caller do the same
        found, resolution = chat_chain.resolve_row(tmp_path, chat_chain.row_address(row.row, gen=gen, line=2))
        assert resolution["status"] == "ok" and found == row.row
    assert ds.legacy_address(tmp_path, ds.append(tmp_path, "chat", owner("appended")).seq) is None


@pytest.mark.parametrize("first", [
    pytest.param(encoded(owner("s1")) + b"\n", id="ordinary"),
    pytest.param(b"\r\n" + encoded(owner("s1")) + b"\r\n", id="blank_line_then_crlf"),
    pytest.param(encoded(owner("s1", text="pasted\u2028from a PDF")) + b"\n", id="u2028"),
    pytest.param(encoded(owner("s1", text="one paragraph\u2029the next")) + b"\n", id="u2029"),
    pytest.param(encoded(owner("s1", text="cp1252 read as latin-1\x85")) + b"\n", id="u0085"),
])
def test_a_generation_is_signed_as_its_old_cursors_signed_it(tmp_path, first):
    """``append_jsonl`` writes U+2028, U+2029 and U+0085 raw (``ensure_ascii=False``) and the text-mode
    reader of ``jsonl_generation_signature`` ends a line only at ``\\n``/``\\r``: the import signs a
    generation the same way, or every old cursor naming it (the frozen consolidation cursor, a parked
    Presence cursor, a wake boundary) stops resolving after the cutover."""
    name = "chat_20261001T000000.jsonl"
    seed(tmp_path, archives={name: [first, owner("s2")]}, live={"chat": [owner("live-1")]})
    gen = gen_of(tmp_path, name)
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    assert (ds.legacy_generation(tmp_path, gen) or {}).get("name") == name
    assert ds.legacy_address(tmp_path, ds.rows_by_client_id(tmp_path, "s2")[0].seq) == (gen, first.count(b"\n") + 1)
    assert chat_chain.generation_cursor(tmp_path, {"chat_log_signature": {"first_line_sha256": gen},
                                                   "last_consolidated_offset": 1}) == (1, False)


# ---- fact 4: task aliases --------------------------------------------------------------------------

def test_rows_whose_only_task_identifier_is_subagent_task_id_are_found_by_the_task_probes(tmp_path, monkeypatch):
    chain_root(tmp_path)
    di.cutover(tmp_path)
    di.run_dialogue_import(tmp_path)
    alias = ds.append(tmp_path, "progress", {"subagent_task_id": "selected", "text": "visible in the timeline"})
    ds.append(tmp_path, "progress", {"subagent_task_id": "other", "text": "another child"})
    own = ds.append(tmp_path, "progress", progress("selected", "the task's own row"))
    assert [row.seq for row in ds.task_rows(tmp_path, ["selected"], stream="progress")] == [alias.seq, own.seq]
    assert [row.seq for row in ds.task_rows(tmp_path, ["selected"], stream="progress", after=alias.seq)] == [own.seq]
    assert ds.task_rows(tmp_path, ["absent"], stream="progress") == []
    assert [row.row["text"] for row in ds.task_rows(tmp_path, ["child-1"], stream="progress")] == ["child step"], "populated by the import"
    assert [row.row["text"] for row in ds.task_rows(tmp_path, ["child-1"], stream="chat")] == ["child words"]
    assert ds.facts(alias.row)["subagent_task_id"] == "selected" and ds.facts(own.row)["subagent_task_id"] is None
    with traced(monkeypatch) as counts:
        assert [row.seq for row in ds.probe_rows(tmp_path, "progress", [("subagent_task_id", "selected")])] == [alias.seq]
    details = plans(tmp_path, counts["statements"])
    # The only scans allowed are over the probe's own value list (``json_each``) and the coordinates it selected (a
    # one-step page's subquery), never the dialogue table.
    assert any("dialogue_subagent" in detail for detail in details) and not any(
        detail.startswith("SCAN") and "json_each" not in detail and not detail.startswith("SCAN (subquery")
        for detail in details), details


# ---- fact 5: room revision continuity ------------------------------------------------------------

def test_room_revision_continues_from_the_projects_visible_revision_at_every_planning_cutover(tmp_path):
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "unread-proof", name="Unread proof")
    quiet = create_project(tmp_path, "quiet", name="Quiet")
    chat_id = project["chat_id"]
    def legacy_revision(value):
        path = tmp_path / "state/projects.json"
        registry = json.loads(path.read_text(encoding="utf-8"))
        next(row for row in registry["projects"] if row["id"] == "unread-proof")["visible_revision"] = value
        path.write_text(json.dumps(registry), encoding="utf-8")

    legacy_revision(3)
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner("archived", chat_id=chat_id)]},
         live={"chat": [owner("live", chat_id=quiet["chat_id"])]})
    report = di.cutover(tmp_path)
    assert ds.room(tmp_path, chat_id)["revision"] == 3 and report["rooms"] == {chat_id: 3}
    assert ds.room(tmp_path, quiet["chat_id"])["revision"] == 0, "a project without unread arrivals keeps 0"
    di.run_dialogue_import(tmp_path)
    assert ds.room(tmp_path, chat_id)["revision"] == 3, "imported rows do not count as unread"
    ds.append(tmp_path, "chat", {"direction": "out", "chat_id": chat_id, "text": "new arrival"}, feed=True)
    assert ds.room(tmp_path, chat_id)["revision"] == 4
    assert ds.mark_room_read(tmp_path, chat_id, 4)["read_revision"] == 4
    legacy_revision(5)  # an older body advanced the frozen registry counter
    seed(tmp_path, live={"chat": [owner("that body's row", chat_id=quiet["chat_id"])]})  # and appended
    assert di.cutover(tmp_path)["rooms"] == {chat_id: 5} and ds.room(tmp_path, chat_id)["revision"] == 5
    ds.append(tmp_path, "chat", {"direction": "out", "chat_id": chat_id, "text": "another"}, feed=True)
    seed(tmp_path, live={"chat": [owner("and again", chat_id=quiet["chat_id"])]})
    assert di.cutover(tmp_path)["rooms"] == {chat_id: 5} and ds.room(tmp_path, chat_id)["revision"] == 6, "never backwards"
    assert ds.room(tmp_path, chat_id)["read_revision"] == 4


# ---- the schema gate and the probes' plans -------------------------------------------------------

def test_a_store_of_another_schema_version_is_refused_and_never_replaced(tmp_path):
    ds.append(tmp_path, "chat", owner("kept"))
    store.close_connections()
    with contextlib.closing(sqlite3.connect(ds.path(tmp_path))) as raw:
        raw.execute("PRAGMA user_version = 1")
    before = ds.path(tmp_path).read_bytes()
    with pytest.raises(ds.StoreLost, match="schema 1"):
        ds.newest(tmp_path, "chat", limit=1)
    assert ds.path(tmp_path).read_bytes() == before and ds.present(tmp_path)
    with contextlib.closing(sqlite3.connect(ds.path(tmp_path))) as raw:
        raw.execute(f"PRAGMA user_version = {ds.SCHEMA_VERSION}")
    store.close_connections()
    assert ds.newest(tmp_path, "chat", limit=1)[0].row == owner("kept") and ds.meta(tmp_path, "schema_version") == ds.SCHEMA_VERSION


@pytest.mark.parametrize("read", [
    lambda root, row: dp.ordinal_at(root, "chat", row.seq),
    lambda root, row: dp.row_at_position(root, dp.ordinal_at(root, "chat", row.seq).pos),
    lambda root, row: dp.row_at_ordinal(root, "chat", dp.ordinal_at(root, "chat", row.seq).ordinal),
    lambda root, row: dp.generation_row(root, gen_of(root, "chat_20261016T000000.jsonl"), 0),
    lambda root, row: dp.skipped_between(root, "chat", lower=row.seq),
    lambda root, row: ds.resolve_legacy(root, gen_of(root, "chat_20261016T000000.jsonl"), 1),
])
def test_count_and_address_probes_are_indexed_and_cost_the_same_with_thirty_rooms(tmp_path, monkeypatch, read):
    results, statements = {}, []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        row = imported_rooms(root, rooms)
        warm = read(root, row)
        with traced(monkeypatch) as counts:
            assert read(root, row) == warm
        results[rooms] = (len(counts["statements"]), counts["steps"])
        statements = counts["statements"]
    assert results[1] == results[30] and results[1][1] > 0, results
    details = plans(root, statements)
    assert not any(detail.startswith("SCAN") for detail in details), details


def _row(text, stamp="2026-10-01T00:00:00Z"):
    return {"ts": stamp, "direction": "in", "chat_id": 1, "text": text}


def test_a_steady_boot_plans_the_chains_again_only_when_another_body_may_have_written(tmp_path, monkeypatch):
    seed(tmp_path, archives={f"chat_2026100{i}T000000.jsonl": [_row(f"archived {i}")] for i in range(1, 4)},
         live={"chat": [_row("live")]})
    assert di.cutover(tmp_path)["first"] is True
    real, planned = di.segment_plan, []
    monkeypatch.setattr(di, "segment_plan", lambda root, stream: planned.append(stream) or real(root, stream))
    assert di.cutover(tmp_path)["replanned"] is False and planned == [], "a steady boot reads no segment"
    seed(tmp_path, live={"chat": [_row("an older body appended")]})
    di.cutover(tmp_path)
    assert planned and any(entry.row.get("text") == "an older body appended"
                           for entry in ds.newest(tmp_path, "chat", limit=10))
    planned.clear()
    di.cutover(tmp_path)
    assert planned == [], "the joined rows are stored; the next boot is steady again"
    seed(tmp_path, archives={"chat_20261020T000000.jsonl": [_row("rotated by an older body")]})
    di.cutover(tmp_path)
    assert planned, "a segment the last plan did not see: the boot plans again"


@pytest.mark.parametrize("pending", [1, 40])
def test_the_import_check_is_one_probe_whatever_is_still_pending(tmp_path, monkeypatch, pending):
    from tests.test_dialogue_addressed_readers import metered

    seed(tmp_path, archives={f"chat_2026{index:06d}T000000.jsonl": [_row(f"old {index}")] for index in range(pending)})
    di.cutover(tmp_path)
    with metered(monkeypatch) as counts:
        with pytest.raises(ds.StoreMigrating):
            ds.require_imported(tmp_path, "chat")
    assert counts["statements"] <= 4 and counts["steps"] <= 200, counts


def test_only_the_segment_whose_bytes_changed_is_refused_never_the_ones_it_moved(tmp_path):
    """An archive that grew after its import is refused (``mismatch``); the segments after it moved
    in the chain but hold the bytes the ledger recorded, so their imports stay done and their
    ranges stay covered (they were the cascade of false gaps before)."""
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [owner("a")], "chat_20261002T000000.jsonl": [owner("b")]},
         live={"chat": [owner("c")]})
    di.run_dialogue_import(tmp_path)

    def ledger():
        with ds.transaction(tmp_path, write=False) as conn:
            return {row["name"]: dict(row) for row in conn.execute("SELECT * FROM import_ledger WHERE stream='chat'")}

    before = ledger()
    assert set(row["state"] for row in before.values()) <= {"done", "live"}
    grown = tmp_path / "archive" / "chat_20261001T000000.jsonl"
    grown.write_bytes(grown.read_bytes() + encoded(owner("late")) + b"\n")
    seed(tmp_path, live={"chat": [owner("d")]})  # another body wrote: the next cutover plans the chain again
    di.cutover(tmp_path)
    after = ledger()
    assert after[grown.name]["state"] == "mismatch", "a segment whose bytes changed is never vouched for"
    # The later archive and the segment the first cutover rotated out of the live file.
    moved = {name: row for name, row in after.items() if name in before and name.startswith("chat_") and name != grown.name}
    assert len(moved) == 2 and all((row["state"], row["base"]) == (before[name]["state"], before[name]["base"])
                                   for name, row in moved.items())
    for row in moved.values():
        assert ds.range_coverage(tmp_path, "chat", lower=row["base"], upper=row["base"] + row["size"]) == {
            "imported": True, "pending": 0, "mismatch": 0, "gaps": 0}


def test_a_segment_that_sorts_before_imported_ones_is_stored_after_them_never_lost(tmp_path):
    """A segment that appears after the cutover with an older name (a clock set back, a copy from a
    backup) used to be planned onto coordinates the import had already given away: its rows were
    dropped while the ledger and coverage said "imported". It is placed after every stored row."""
    def row(tag, index):
        return {"ts": "2026-09-01T00:00:00Z", "chat_id": 1, "direction": "in", "text": f"{tag}-{index}"}

    seed(tmp_path, archives={"chat_20260901T000000.jsonl": [row("A", index) for index in range(5)]})
    di.run_dialogue_import(tmp_path)
    seed(tmp_path, archives={"chat_20260801T000000.jsonl": [row("B", index) for index in range(5)]})
    di.run_dialogue_import(tmp_path)
    texts = [entry.row["text"] for entry in ds.newest(tmp_path, "chat", limit=100)]
    assert sorted(texts) == sorted([f"A-{index}" for index in range(5)] + [f"B-{index}" for index in range(5)])
    assert ds.range_coverage(tmp_path, "chat") == {"imported": True, "pending": 0, "mismatch": 0, "gaps": 0}



def test_a_steady_boot_lists_the_archive_only_after_its_stamp_moved(tmp_path, monkeypatch):
    """Both directions. The cutover records the archive directory's stamp beside the tails it saw; a
    steady boot stats the directory and lists nothing while the stamp holds. A file of another stream
    landing in the archive moves the stamp: that boot lists once, finds the same tails, records the
    stamp and plans nothing, and the next boot lists nothing again. A chain segment an older body
    rotated in moves it too: that boot lists once and plans again."""
    import os

    seed(tmp_path, archives={f"chat_2026100{i}T000000.jsonl": [_row(f"archived {i}")] for i in range(1, 3)},
         live={"chat": [_row("live")]})
    archive, listings, planned = tmp_path / "archive", [], []
    scandir, real_plan = os.scandir, di.segment_plan

    def counted(path=".", *args, **kwargs):
        if pathlib.Path(path) == archive:
            listings.append(1)
        return scandir(path, *args, **kwargs)

    monkeypatch.setattr(di.os, "scandir", counted)
    monkeypatch.setattr(di, "segment_plan", lambda root, stream: planned.append(stream) or real_plan(root, stream))
    assert di.cutover(tmp_path)["first"] is True
    assert ds.meta(tmp_path, "archive_stamp") == [archive.stat().st_ino, archive.stat().st_mtime_ns]
    listings.clear(), planned.clear()
    for _ in range(2):
        assert di.cutover(tmp_path)["replanned"] is False
    assert (listings, planned) == ([], []), "a steady boot listed the archive"

    def moved(write):
        recorded = ds.meta(tmp_path, "archive_stamp")
        write()
        if archive.stat().st_mtime_ns == recorded[1]:  # a coarse clock: the entry still moved the directory
            os.utime(archive, ns=(recorded[1] + 1, recorded[1] + 1))

    moved(lambda: (archive / "events_20261023T000000.jsonl").write_text('{"kind":"rotated"}\n', encoding="utf-8"))
    report = di.cutover(tmp_path)
    assert report["replanned"] is False and (listings, planned) == ([1], []), "another stream's segment planned the chains"
    assert ds.meta(tmp_path, "archive_stamp") == [archive.stat().st_ino, archive.stat().st_mtime_ns]
    listings.clear()
    assert di.cutover(tmp_path)["replanned"] is False and listings == [], "the moved stamp was not recorded"

    moved(lambda: seed(tmp_path, archives={"chat_20261020T000000.jsonl": [_row("rotated by an older body")]}))
    assert "chat" in di.cutover(tmp_path) and listings and planned, "a segment the last plan did not see"
    assert di.coverage(tmp_path, "chat")["pending"] >= 1  # the plan itself lists as it must: the segment is planned
    listings.clear(), planned.clear()
    assert di.cutover(tmp_path)["replanned"] is False and (listings, planned) == ([], [])
