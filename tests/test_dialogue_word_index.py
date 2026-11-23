"""Disposable word postings (``ouroboros/dialogue_word_index.py``) and their reader path."""
from __future__ import annotations

import json
import re
import sqlite3
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_import as di, dialogue_recall, dialogue_store as ds, dialogue_word_index as dwi
from ouroboros import maintenance
from ouroboros.tools.control_runtime import _chat_history
from tests import _dialogue as dialogue_test
from tests._dialogue import chat_rows, seed

needs_fts5 = pytest.mark.skipif(not dwi.fts5_available(), reason="this SQLite has no FTS5")

LONG = "x" * 70 + "tail"


def _read(root, **kwargs) -> str:
    return str(_chat_history(SimpleNamespace(drive_root=root, task_metadata={}, pending_events=[], task_id="t"),
                             **kwargs))


def _lines(answer: str):
    """The answer without its header: what scan and postings must give alike."""
    return answer.split("\n\n", 1)[1] if "\n\n" in answer else ""


def _total(answer: str) -> str:
    return re.search(r"(total=\d+|observed=\d+; total_unknown)", answer).group(1)


def _all(conn, sql, *args):
    return [tuple(record) for record in conn.execute(sql, args).fetchall()]


def _say(root, ts, text, chat_id=1, direction="in", **extra):
    return ds.append(root, "chat", {"ts": ts, "direction": direction, "chat_id": chat_id, "text": text, **extra})


def _corpus(root):
    _say(root, "2026-11-01T00:00:00Z", "zero row first decision")  # rowid 0
    for n in range(1, 40):
        _say(root, f"2026-11-01T00:{n:02d}:00Z", f"решили сокет {n} red fox" if n % 3 == 0
             else f"обсуждение {n} brown fox red" if n % 3 == 1 else f"chat_history memory_view.py {n}",
             chat_id=(1, 7, 9)[n % 3])
    _say(root, "2026-11-02T00:00:00Z", f"very long {LONG} word")
    ds.append(root, "chat", {"ts": "2026-11-03T00:00:00Z", "type": "quiz_answer", "direction": "system",
                             "chat_id": 1, "task_id": "asker", "text": "[answer]",
                             "quiz": {"quiz_id": "q", "question": "alpha topic", "options": ["one", "two"],
                                      "answered_index": 1, "comment": "beta reply"}})


QUERIES = [["реш сок"], ['"red fox"'], ["fox", "zero"], ["history"], ['"chat_history"'], ["alpha beta"],
           ["alpha"], ["x" * 66], ['"' + LONG + '"'], ["zero decis"], ["missing"]]


@needs_fts5
def test_scan_and_postings_give_the_same_answer(tmp_path):
    _corpus(tmp_path)
    scanned = {json.dumps(q): _read(tmp_path, words=q) for q in QUERIES}
    assert all("Read by scan of every row" in answer for answer in scanned.values())
    assert dwi.run(tmp_path, lambda: False)["state"] == "done"
    for q in QUERIES:
        posted = _read(tmp_path, words=q)
        assert "Read by postings" in posted, q
        assert _lines(posted) == _lines(scanned[json.dumps(q)]) and _total(posted) == _total(scanned[json.dumps(q)]), q
    assert "total=0" in _read(tmp_path, words=["alpha beta"])  # the postings propose it; one field decides
    assert "row:1@2026-11-01T00:00:00Z" in _read(tmp_path, words=["zero"])


@needs_fts5
def test_a_scan_only_read_installs_nothing_and_an_append_writes_no_posting(tmp_path):
    _say(tmp_path, "2026-11-01T00:00:00Z", "plain words")
    _read(tmp_path, words=["plain"])
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn) is None
    dwi.run(tmp_path, lambda: False)
    with ds.transaction(tmp_path, write=False) as conn:
        before = _all(conn, "SELECT next_ord FROM dw_mark")
    _say(tmp_path, "2026-11-01T00:01:00Z", "freshly appended")
    with ds.transaction(tmp_path, write=False) as conn:
        assert _all(conn, "SELECT next_ord FROM dw_mark") == before
    fresh = _read(tmp_path, words=["fresh"])
    assert "total=1" in fresh and "1 not yet posted" in fresh  # findable at once: the uncovered row is read
    assert not dwi.due(tmp_path, {}, {}, False)  # one row starts no child: the reader re-reads it


@needs_fts5
def test_an_edit_is_found_by_its_new_words_and_not_its_old_ones(tmp_path):
    _corpus(tmp_path)
    receipt = _say(tmp_path, "2026-11-04T00:00:00Z", "kumquat original")
    dwi.run(tmp_path, lambda: False)
    ds.revise_text(tmp_path, "chat", receipt.seq, "persimmon corrected")
    assert "total=0" in _read(tmp_path, words=["kumquat"])
    found = _read(tmp_path, words=["persimmon"])
    assert "total=1" in found and "1 edited" in found and "Read by postings" in found
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn)["sound"]
        assert _all(conn, "SELECT seq FROM dw_dirty") == [(receipt.seq,)]
    ds.revise_text(tmp_path, "chat", receipt.seq, "kumquat again")  # and back: the dirty row is re-read
    assert "total=1" in _read(tmp_path, words=["kumquat"]) and "total=0" in _read(tmp_path, words=["persimmon"])


@needs_fts5
def test_a_missed_edit_latches_the_generation_and_a_rebuild_restores_it(tmp_path):
    _corpus(tmp_path)
    receipt = _say(tmp_path, "2026-11-04T00:00:00Z", "kumquat original")
    dwi.run(tmp_path, lambda: False)
    with ds.transaction(tmp_path, write=True) as conn:  # a counter step no witnessed edit explains
        ds._set_meta(conn, "chat_text_revision", int(ds._meta(conn, "chat_text_revision") or 0) + 1)
    ds.revise_text(tmp_path, "chat", receipt.seq, "persimmon corrected")  # the next good edit must not certify it
    with ds.transaction(tmp_path, write=False) as conn:
        facts = dwi.generation(conn)
        assert not facts["sound"] and facts["reason"] == "latched unusable"
    answer = _read(tmp_path, words=["persimmon"])
    assert "Read by scan of every row" in answer and "total=1" in answer
    assert dwi.due(tmp_path, {}, {}, False)
    dwi.run(tmp_path, lambda: False)
    rebuilt = _read(tmp_path, words=["persimmon"])
    assert "Read by postings" in rebuilt and "total=1" in rebuilt and "0 edited" in rebuilt


@needs_fts5
def test_an_edit_without_a_counter_step_is_a_discontinuity(tmp_path):
    receipt = _say(tmp_path, "2026-11-04T00:00:00Z", "kumquat original")
    dwi.run(tmp_path, lambda: False)
    with ds.transaction(tmp_path, write=True) as conn:
        conn.execute("UPDATE dialogue SET body=? WHERE stream='chat' AND seq=?",
                     (ds.encode({**receipt.row, "text": "silent change"}), receipt.seq))
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn)["reason"].startswith("witness")
    assert "Read by scan" in _read(tmp_path, words=["silent"])


@needs_fts5
def test_a_rolled_back_edit_leaves_no_trace(tmp_path):
    receipt = _say(tmp_path, "2026-11-04T00:00:00Z", "kumquat original")
    dwi.run(tmp_path, lambda: False)
    with pytest.raises(RuntimeError):
        with ds.transaction(tmp_path, write=True) as conn:
            conn.execute("UPDATE dialogue SET body=? WHERE stream='chat' AND seq=?",
                         (ds.encode({**receipt.row, "text": "never"}), receipt.seq))
            raise RuntimeError("rolled back")
    with ds.transaction(tmp_path, write=False) as conn:
        assert conn.execute("SELECT count(*) FROM dw_dirty").fetchone()[0] == 0
        assert dwi.generation(conn)["sound"]


def _two_archives(root):
    seed(root, archives={
        "chat_20261001T000000.jsonl": [{"ts": "2026-10-01T00:00:00Z", "direction": "in", "chat_id": 1,
                                        "text": "earlier archive quince"}],
        "chat_20261002T000000.jsonl": [{"ts": "2026-10-02T00:00:00Z", "direction": "in", "chat_id": 1,
                                        "text": "later archive medlar"}]},
        live={"chat": [{"ts": "2026-10-03T00:00:00Z", "direction": "in", "chat_id": 1, "text": "live loquat"}]})
    di.cutover(root)


@needs_fts5
def test_an_earlier_segment_completed_after_a_later_one_is_never_lost(tmp_path):
    _two_archives(tmp_path)
    di.import_segment(tmp_path, "chat", "chat_20261002T000000.jsonl")  # the later file first
    assert dwi.run(tmp_path, lambda: False)["state"] in ("done", "pending")
    di.import_segment(tmp_path, "chat", "chat_20261001T000000.jsonl")  # the earlier one lands below the marks
    with ds.transaction(tmp_path, write=False) as conn:
        facts = dwi.generation(conn)
        assert facts["import_epoch"] != facts["built_epoch"]
    assert dwi.due(tmp_path, {}, {}, False)
    found = _read(tmp_path, words=["quince"])
    assert "Read by postings" in found and "earlier archive quince" in found
    dwi.run(tmp_path, lambda: False)
    assert not dwi.due(tmp_path, {}, {}, False)
    again = _read(tmp_path, words=["quince"])
    assert "earlier archive quince" in again and "0 not yet posted" in again
    assert {row["text"] for row in chat_rows(tmp_path)} >= {"earlier archive quince", "later archive medlar"}


@needs_fts5
def test_a_stopped_pass_keeps_its_progress_and_resumes(tmp_path, monkeypatch):
    _corpus(tmp_path)
    monkeypatch.setattr(dwi, "CHUNK_ROWS", 5)
    calls = iter([True])
    assert dwi.run(tmp_path, lambda: next(calls, False))["state"] == "pending"
    with ds.transaction(tmp_path, write=False) as conn:
        assert _all(conn, "SELECT next_ord FROM dw_mark") == [(5,)]
    assert dwi.due(tmp_path, {}, {}, False)
    assert dwi.run(tmp_path, lambda: False)["state"] == "done"
    assert "0 not yet posted" in _read(tmp_path, words=["fox"])


@needs_fts5
def test_two_workers_cannot_post_one_interval_twice(tmp_path):
    _corpus(tmp_path)
    name = dwi._install(tmp_path)
    with ds.transaction(tmp_path, write=False) as conn:
        units = dwi._segments(conn)
    totals = {"rows": 0, "source_bytes": 0, "token_bytes": 0, "slowest_ms": 0.0}
    segment, mark, upper = units[0]
    assert dwi._post_segment(tmp_path, name, segment, mark, upper, lambda: False, totals)
    # a second worker that read the same mark loses the compare and posts nothing
    assert not dwi._post_segment(tmp_path, name, segment, mark, upper, lambda: False, totals)
    with ds.transaction(tmp_path, write=False) as conn:
        assert conn.execute("SELECT next_ord FROM dw_mark WHERE segment=?", (segment,)).fetchone()[0] == upper


@needs_fts5
def test_a_busy_writer_leaves_the_chunk_pending(tmp_path, monkeypatch):
    _corpus(tmp_path)
    dwi._install(tmp_path)
    real = ds.transaction

    def busy(root, *, write, wait_sec=ds.WAIT_SEC):
        if write:
            raise ds.StoreBusy("held")
        return real(root, write=write, wait_sec=wait_sec)

    monkeypatch.setattr(ds, "transaction", busy)
    with pytest.raises(ds.StoreBusy):
        dwi.run(tmp_path, lambda: False)
    monkeypatch.setattr(ds, "transaction", real)
    assert dwi.run(tmp_path, lambda: False)["state"] == "done"
    logged = [json.loads(line) for line in (tmp_path / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = [entry for entry in logged if entry.get("job") == "dialogue_words"]
    assert chunks[-1]["outcome"] == "committed" and {"hold_ms", "wal_bytes", "dirty_rows", "source_bytes",
                                                     "token_bytes", "rows_read"} <= set(chunks[-1])


@needs_fts5
def test_without_fts5_the_reader_scans_and_the_job_rests(tmp_path, monkeypatch):
    _corpus(tmp_path)
    dwi.run(tmp_path, lambda: False)
    monkeypatch.setattr(dwi, "_FTS5", False)
    answer = _read(tmp_path, words=["fox"])
    assert "Read by scan of every row" in answer
    assert not dwi.due(tmp_path, {}, {}, False)
    receipt = _say(tmp_path, "2026-11-05T00:00:00Z", "edited later")
    ds.revise_text(tmp_path, "chat", receipt.seq, "edited without the module")  # the trigger needs no FTS5
    retired = dwi.retire(tmp_path)
    assert retired["state"] == "fts5_required"
    with ds.transaction(tmp_path, write=False) as conn:
        assert not dwi.generation(conn)["sound"]
    monkeypatch.setattr(dwi, "_FTS5", True)
    assert "Read by scan" in _read(tmp_path, words=["fox"])  # latched, never silently trusted again


@needs_fts5
def test_an_older_body_opens_appends_edits_and_exports_byte_exactly(tmp_path):
    plain, posted = tmp_path / "plain", tmp_path / "posted"
    for root in (plain, posted):
        _corpus(root)
    dwi.run(posted, lambda: False)
    for root in (plain, posted):  # the statements of a body that knows nothing of the postings
        receipt = _say(root, "2026-11-06T00:00:00Z", "older body row")
        ds.revise_text(root, "chat", receipt.seq, "older body edit")
        di.export_for_older_body(root)
    for name in ("chat.jsonl", "progress.jsonl"):
        a, b = plain / "logs" / name, posted / "logs" / name
        assert a.exists() == b.exists()
        if a.exists():
            assert a.read_bytes() == b.read_bytes()
    with sqlite3.connect(ds.path(posted)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] in (ds.SCHEMA_VERSION, ds.SHARED_VERSION)


def test_the_trigger_needs_no_fts5_module(tmp_path):
    """The edit trigger and every plain table work on a file whose connection never touches the virtual
    table: a SQLite without FTS5 edits through them (a build without the module was not available here)."""
    receipt = _say(tmp_path, "2026-11-04T00:00:00Z", "kumquat")
    with ds.transaction(tmp_path, write=True) as conn:
        for statement in dwi._DDL:
            if "VIRTUAL TABLE" not in statement:
                conn.execute(statement)
        conn.execute("INSERT INTO dw_meta (id, format, generation, seen_rev, usable) VALUES (1, 1, 'g', 0, 1)")
    ds.revise_text(tmp_path, "chat", receipt.seq, "edited")
    with ds.transaction(tmp_path, write=False) as conn:
        assert _all(conn, "SELECT seq, rev FROM dw_dirty") == [(receipt.seq, 1)]
        assert tuple(conn.execute("SELECT seen_rev, usable FROM dw_meta").fetchone()) == (1, 1)


@needs_fts5
def test_retirement_frees_pages_and_leaves_the_rows(tmp_path):
    assert dwi.retire(tmp_path)["state"] == "absent"
    _corpus(tmp_path)
    rows = chat_rows(tmp_path)
    assert dwi.retire(tmp_path)["state"] == "nothing_installed"
    dwi.run(tmp_path, lambda: False)
    size = ds.path(tmp_path).stat().st_size
    retired = dwi.retire(tmp_path)
    assert retired["state"] == "retired" and retired["pages_freed"] > 0 and retired["file_shrinks"] is False
    assert ds.path(tmp_path).stat().st_size >= size
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn) is None
        assert not _all(conn, "SELECT name FROM sqlite_master WHERE name LIKE 'dw_%'")
    assert chat_rows(tmp_path) == rows
    assert "Read by scan" in _read(tmp_path, words=["fox"])
    assert dwi.main(["--retire", "--data-root", str(tmp_path)]) == 0


@needs_fts5
def test_the_maintenance_duty_works_then_rests_and_wakes_on_new_rows(tmp_path):
    assert "dialogue_words" in maintenance.JOB_NAMES
    assert maintenance.JOB_NAMES.index("dialogue_words") > maintenance.JOB_NAMES.index("dialogue_import")
    _corpus(tmp_path)
    assert maintenance.run_job(tmp_path, "dialogue_words")["state"] == "done"
    assert maintenance.run_job(tmp_path, "dialogue_words") is None  # at rest
    for n in range(dwi.CHUNK_ROWS - 1):
        _say(tmp_path, "2026-11-07T00:00:00Z", f"new row {n}")
    assert maintenance.run_job(tmp_path, "dialogue_words") is None  # less than a chunk: at rest
    _say(tmp_path, "2026-11-07T00:00:00Z", "the chunk's last row")
    assert maintenance.run_job(tmp_path, "dialogue_words")["state"] == "done"  # a whole chunk: work


@needs_fts5
def test_past_the_bound_the_scan_answers_whole_and_says_why(tmp_path, monkeypatch):
    for n in range(6):
        _say(tmp_path, f"2026-11-0{n + 1}T00:00:00Z", f"plum {n}")
    dwi.run(tmp_path, lambda: False)
    monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", 6)  # met, not exceeded: the postings answer
    within = _read(tmp_path, words=["plum"])
    assert "total=6" in within and "Read by postings" in within
    monkeypatch.setattr(dialogue_recall, "RECHECK_BOUND", 4)
    broad = _read(tmp_path, words=["plum"])
    assert "total=6" in broad and "plum 5" in broad and "plum 0" in broad
    assert "Read by scan of every row: more posting candidates than the re-check bound of 4" in broad
    assert "recheck_bound" not in broad and not re.search(r"snapshot=[0-9a-f]{64}:", broad)


@needs_fts5
def test_no_read_transaction_spans_the_fetch_of_the_candidates(tmp_path, monkeypatch):
    for n in range(5):
        _say(tmp_path, f"2026-11-0{n + 1}T00:00:00Z", f"plum {n}")
    dwi.run(tmp_path, lambda: False)
    real_transaction, real_page = ds.transaction, ds._page
    opened, current, pages = [0], [None], []

    import contextlib

    @contextlib.contextmanager
    def counted(root, *, write, **kwargs):
        opened[0] += 1
        current[0] = opened[0]
        try:
            with real_transaction(root, write=write, **kwargs) as conn:
                yield conn
        finally:
            current[0] = None

    def page(conn, where, params):
        pages.append(current[0])
        return real_page(conn, where, params)

    monkeypatch.setattr(ds, "transaction", counted)
    monkeypatch.setattr(ds, "_page", page)
    monkeypatch.setattr(dialogue_recall, "_CHUNK", 2)
    assert "total=5" in _read(tmp_path, words=["plum"])
    assert len(pages) == 3 and len(set(pages)) == 3  # five bodies, three pages, three short reads
    pages.clear()
    monkeypatch.setattr(dialogue_recall, "_CHUNK", 500)
    assert "total=5" in _read(tmp_path, words=["plum"])
    assert len(pages) == 1  # a small set is one page, one read


@needs_fts5
def test_postings_built_under_other_unicode_data_are_bypassed(tmp_path, monkeypatch):
    import unicodedata

    _say(tmp_path, "2026-11-01T00:00:00Z", "plum")
    dwi.run(tmp_path, lambda: False)
    assert "Read by postings" in _read(tmp_path, words=["plum"])
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn)["generation"].split(":")[1] == unicodedata.unidata_version
    monkeypatch.setattr(unicodedata, "unidata_version", "0.0.0")
    with ds.transaction(tmp_path, write=False) as conn:
        assert dwi.generation(conn)["reason"].startswith("unicode data")
    assert "Read by scan" in _read(tmp_path, words=["plum"]) and dwi.due(tmp_path, {}, {}, False)


@needs_fts5
def test_the_uncovered_rows_are_found_by_the_ordinal_index_whatever_the_history(tmp_path):
    """With every row posted, finding the rows no posting covers costs one index probe a unit, never a walk of
    the chat history: the plan names ``dialogue_ordinal`` and the work does not grow with the rows."""
    from ouroboros.dialogue_words import parse

    small, large = tmp_path / "small", tmp_path / "large"
    for root, rows in ((small, 50), (large, 1500)):
        dialogue_test.seed(root, live={"chat": [{"ts": "2026-11-01T00:00:00Z", "direction": "in", "chat_id": 1,
                                                  "text": f"row {n}"} for n in range(rows)]})
        dialogue_test.import_all(root)
        dwi.run(root, lambda: False)
    steps = {}
    for root in (small, large):
        with ds.transaction(root, write=False) as conn:
            plan = " ".join(str(tuple(row)) for row in conn.execute("EXPLAIN QUERY PLAN " + dwi._UNCOVERED, (1 << 62,)))
            assert "dialogue_ordinal" in plan and "PRIMARY KEY (stream=? AND seq<?)" not in plan, plan
            assert dwi.candidates(conn, parse(["zzzz"]), upper=1 << 62).uncovered == 0
            counted = [0]
            conn.set_progress_handler(lambda: counted.__setitem__(0, counted[0] + 1) or 0, 1)
            assert conn.execute(dwi._UNCOVERED, (1 << 62,)).fetchall() == []
            conn.set_progress_handler(None, 0)
            steps[root.name] = counted[0]
    assert steps["large"] == steps["small"], steps  # thirty times the rows, the same work


@needs_fts5
def test_a_generation_still_on_its_first_pass_leaves_the_read_to_the_scan(tmp_path, monkeypatch):
    _corpus(tmp_path)
    monkeypatch.setattr(dwi, "CHUNK_ROWS", 5)
    assert dwi.run(tmp_path, lambda: True)["state"] == "pending"  # one chunk posted, the pass stopped
    partial = _read(tmp_path, words=["fox"])
    assert "Read by scan of every row" in partial and re.search(r"total=\d+;", partial)
    assert dwi.run(tmp_path, lambda: False)["state"] == "done"
    built = _read(tmp_path, words=["fox"])
    assert "Read by postings" in built and _total(built) == _total(partial)


@needs_fts5
def test_copies_of_one_row_are_one_line_on_the_postings_path(tmp_path):
    row = {"ts": "2026-11-01T10:00:00Z", "direction": "in", "chat_id": 1, "text": "twice stored plum"}
    ds.append(tmp_path, "chat", dict(row))
    ds.append(tmp_path, "chat", dict(row))  # a crash-retry copy: same source, same time
    dwi.run(tmp_path, lambda: False)
    answer = _read(tmp_path, words=["plum"])
    assert "Read by postings" in answer and "total=1" in answer and answer.count("twice stored plum") == 1


def test_a_runtime_without_fts5_beside_a_postings_table_keeps_every_dialogue_path(tmp_path, monkeypatch):
    """The stand-in for a SQLite without FTS5 on any runner: ``dw_post`` names a module this library lacks and
    FTS5 is reported absent. The read scans, append and an edit through the trigger work, the legacy search
    works, the job rests, the export is byte-exact, and retirement answers that it needs FTS5, latched."""
    plain, held = tmp_path / "plain", tmp_path / "held"
    for root in (plain, held):
        _say(root, "2026-11-01T00:00:00Z", "plum before")
    with ds.transaction(held, write=True) as conn:
        for statement in dwi._DDL:
            if "VIRTUAL TABLE" not in statement:
                conn.execute(statement)
        conn.execute("INSERT INTO dw_meta (id, format, generation, seen_rev, usable, built_epoch) "
                     "VALUES (1, ?, ?, 0, 1, 0)", (dwi.FORMAT, f"{dwi.FORMAT}:x:g"))
    raw = sqlite3.connect(ds.path(held))
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute("INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES "
                "('table', 'dw_post', 'dw_post', 0, 'CREATE VIRTUAL TABLE dw_post USING no_such_fts(tokens)')")
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.commit()
    raw.close()
    monkeypatch.setattr(dwi, "_FTS5", False)
    for root in (plain, held):
        receipt = _say(root, "2026-11-02T00:00:00Z", "plum appended")
        ds.revise_text(root, "chat", receipt.seq, "plum edited")
    with ds.transaction(held, write=False) as conn:
        assert _all(conn, "SELECT seq FROM dw_dirty") and conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    answer = _read(held, words=["plum"])
    assert "total=2" in answer and "Read by scan of every row" in answer
    assert _read(held, search="plum").startswith("Showing 2 of 2 messages")
    assert not dwi.due(held, {}, {}, False) and dwi.run(held, lambda: False)["state"] == "done"
    for root in (plain, held):
        di.export_for_older_body(root)
    assert (plain / "logs" / "chat.jsonl").read_bytes() == (held / "logs" / "chat.jsonl").read_bytes()
    assert dwi.retire(held)["state"] == "fts5_required"
    with ds.transaction(held, write=False) as conn:
        assert not dwi.generation(conn)["usable"]




@needs_fts5
def test_collection_stops_one_past_the_bound(tmp_path):
    from ouroboros.dialogue_words import parse

    for n in range(9):
        _say(tmp_path, f"2026-11-01T00:0{n}:00Z", f"plum {n}")
    dwi.run(tmp_path, lambda: False)
    with ds.transaction(tmp_path, write=False) as conn:
        capped = dwi.candidates(conn, parse(["plum"]), upper=1 << 62, limit=4)
        whole = dwi.candidates(conn, parse(["plum"]), upper=1 << 62, limit=9)
    assert capped.more and capped.seqs == [] and capped.posted == 5  # never more than bound + 1 ids
    assert not whole.more and len(whole.seqs) == 9 and whole.posted == 9
