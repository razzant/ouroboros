"""A project's workpad through workpad_write's modes: its history beside the page (rows by change), the blind
overwrite's warning, the size every receipt states, one critical section, byte-exact pages and the root they share."""
from __future__ import annotations

import hashlib
import json
import types

import pytest

from ouroboros.tools.tool_result import _TOOL_RESULT_ATTR

PID = "racer"


def _ctx(root, task="t-1", project=PID):
    ctx = types.SimpleNamespace(project_id=project, task_id=task, drive_root=root, task_metadata={})
    setattr(ctx, _TOOL_RESULT_ATTR, None)  # the typed result a handler publishes lands here
    return ctx


def _code(ctx):
    return getattr(ctx, _TOOL_RESULT_ATTR).code


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    from ouroboros.project_facts import project_workpad_history_path, project_workpad_path

    return project_workpad_path(PID), project_workpad_history_path(PID)


def _rows(journal):
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]


PLAN = "# Plan\n\n## Now\n\n- wheels\n- brakes\n\n## Later\n\n- paint\n"


def test_writes_keep_a_history_of_changes_beside_the_page_and_every_version_reads(tmp_path, store):
    from ouroboros import knowledge_versions
    from ouroboros.memory_texts import workpad_source
    from ouroboros.tools.project_journal import _workpad_write

    page, journal = store
    ctx = _ctx(tmp_path)
    created = _workpad_write(ctx, PLAN, reason="first page")
    assert created.startswith("OK: workpad created (overwrite") and _code(ctx) == "OK", created
    assert f"Size: {len(PLAN)} chars, about {len(PLAN) // 4} tokens (chars/4); it enters every request of project " \
           f"{PID} whole." in created
    v1 = _sha(page.read_bytes())
    edited = _workpad_write(ctx, mode="edit", expected_revision=v1, reason="brakes done",
                            edits=[{"old_str": "- brakes\n", "new_str": ""}, {"old_str": "- paint", "new_str": "- paint red"}])
    assert edited.startswith("OK: workpad updated (edit") and "Size: " in edited, edited
    v2 = _sha(page.read_bytes())
    assert f"memory_read(source='workpad:{PID}', version='{v1}')" in edited
    assert f"workpad_write(project_id='{PID}', mode='revert', version='{v1}', expected_revision='{v2}')" in edited
    reverted = _workpad_write(ctx, mode="revert", version=v1, expected_revision=v2)
    assert reverted.startswith("OK: workpad updated (revert") and page.read_bytes() == PLAN.encode(), reverted
    rows = _rows(journal)
    assert [row["mode"] for row in rows] == ["overwrite", "edit", "revert"] and all(row["topic"] == "workpad" for row in rows)
    assert all("old_preview" not in row and "new_preview" not in row and row["delta"] for row in rows)
    assert "base" in rows[1] and "change" in rows[1] and "new_content" not in rows[1]  # stored by change
    assert rows[2]["reverted_from"] == v1 and rows[1]["reason"] == "brakes done"
    for revision in (v1, v2):
        text, row = knowledge_versions.find_version(workpad_source(PID), revision)
        assert _sha(text.encode("utf-8")) == revision and row["new_sha256"] == revision


def test_a_stale_revision_gets_the_changes_since_from_change_rows(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_write

    page, _journal = store
    _workpad_write(_ctx(tmp_path), PLAN)
    held = _sha(page.read_bytes())
    _workpad_write(_ctx(tmp_path, "t-2"), mode="edit", old_str="- wheels", content="- wheels, new", expected_revision=held)
    stale = _workpad_write(_ctx(tmp_path), mode="edit", old_str="- paint", content="- paint blue", expected_revision=held)
    assert "revision_conflict" in stale and "\n- - wheels\n+ - wheels, new\n" in stale and "1 block(s)" in stale, stale
    assert page.read_text(encoding="utf-8").count("blue") == 0


def test_a_blind_overwrite_of_another_tasks_version_lands_with_a_warning(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_write

    page, _journal = store
    mine, other = _ctx(tmp_path, "t-1"), _ctx(tmp_path, "t-2")
    assert _workpad_write(mine, "mine\n").startswith("OK:") and _code(mine) == "OK"
    again = _workpad_write(mine, "mine, again\n")  # its own version: no warning
    assert again.startswith("OK: workpad updated (overwrite") and _code(mine) == "OK", again
    replaced = _sha(page.read_bytes())
    blind = _workpad_write(other, "theirs\n")
    assert blind.startswith("⚠️ BLIND_OVERWRITE: this overwrite replaced a workpad version this task did not write "
                            "(written ") and "by focus root task_id=t-1" in blind, blind
    assert _code(other) == "LEGACY_WARNING" and page.read_bytes() == b"theirs\n"  # the write landed
    assert f"Replaced {replaced}" in blind and f"version='{replaced}'" in blind
    page.write_bytes(b"written outside the writer\n")  # no row names this version: unknown provenance warns too
    unknown = _workpad_write(other, "theirs again\n")
    assert "no history row wrote it" in unknown and _code(other) == "LEGACY_WARNING", unknown
    checked = _workpad_write(mine, "checked\n", expected_revision=_sha(page.read_bytes()))
    assert checked.startswith("OK:") and _code(mine) == "OK"  # a revision-checked overwrite is never blind


def test_an_empty_page_stays_legal_and_a_missing_page_is_created_even_empty(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_read, _workpad_write

    page, journal = store
    ctx = _ctx(tmp_path)
    assert "is empty) workpad_write creates it" in _workpad_read(ctx)
    for mode, extra in (("edit", {"old_str": "a", "content": "b"}), ("revert", {"version": "1" * 64})):
        refused = _workpad_write(ctx, mode=mode, expected_revision="", **extra)
        assert refused.startswith(f"⚠️ TOOL_ERROR: {mode}_source_missing") and _code(ctx) == "TOOL_REPORTED_FAILURE"
    created = _workpad_write(ctx, "")  # absence and sha256(b"") are different states
    assert created.startswith("OK: workpad created") and page.read_bytes() == b"" and len(_rows(journal)) == 1, created
    unchanged = _workpad_write(ctx, "", expected_revision=_sha(b""))
    assert unchanged.startswith("OK: workpad unchanged") and "Size: 0 chars" in unchanged and len(_rows(journal)) == 1
    assert _workpad_write(ctx, "x", expected_revision=_sha(b"")).startswith("OK:")
    emptied = _workpad_write(ctx, mode="edit", old_str="x", content="", expected_revision=_sha(b"x"))
    assert emptied.startswith("OK: workpad updated (edit") and page.read_bytes() == b"", emptied


def test_no_size_refuses_a_page_and_every_receipt_states_it(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_write

    page, _journal = store
    large = "# Large\n" + "a line of the page\n" * 20_000  # past the old 256 KB refusal
    reply = _workpad_write(_ctx(tmp_path), large)
    assert reply.startswith("OK:") and page.stat().st_size > 256 * 1024 and f"Size: {len(large)} chars" in reply


def test_the_row_lands_first_and_a_failed_append_leaves_the_page_untouched(tmp_path, store, monkeypatch):
    from ouroboros import memory_publication
    from ouroboros.tools.project_journal import _workpad_write

    page, journal = store
    ctx = _ctx(tmp_path)
    _workpad_write(ctx, PLAN)
    before, rows = page.read_bytes(), len(_rows(journal))
    with monkeypatch.context() as patched:
        patched.setattr(memory_publication, "append_jsonl", lambda *args, **kwargs: False)
        refused = _workpad_write(ctx, "Replaced.\n", expected_revision=_sha(before))
    assert "history_unavailable" in refused and page.read_bytes() == before and len(_rows(journal)) == rows
    order, real_append, real_write = [], memory_publication.append_jsonl, memory_publication.write_bytes_atomic
    with monkeypatch.context() as patched:
        patched.setattr(memory_publication, "append_jsonl",
                        lambda *a, **k: order.append(("row", page.read_bytes())) or real_append(*a, **k))
        patched.setattr(memory_publication, "write_bytes_atomic",
                        lambda *a, **k: order.append(("page",)) or real_write(*a, **k))
        assert _workpad_write(ctx, "Replaced.\n", expected_revision=_sha(before)).startswith("OK:")
    assert order == [("row", before), ("page",)]


def test_two_writers_from_one_revision_land_exactly_one(tmp_path, store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from ouroboros.tools.project_journal import _workpad_write

    page, journal = store
    _workpad_write(_ctx(tmp_path), PLAN)
    revision, start = _sha(page.read_bytes()), Barrier(2)

    def write(number):
        start.wait()
        return _workpad_write(_ctx(tmp_path, f"w{number}"), mode="edit", old_str="- wheels",
                              content=f"- wheels {number}", expected_revision=revision)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, (0, 1)))
    assert sum(text.startswith("OK:") for text in results) == 1 and sum("revision_conflict" in t for t in results) == 1
    assert _rows(journal)[-1]["new_sha256"] == _sha(page.read_bytes()) and len(_rows(journal)) == 2


def test_a_torn_tail_and_a_failed_replace_cost_no_version(tmp_path, store, monkeypatch):
    from ouroboros import knowledge_versions, memory_publication
    from ouroboros.memory_texts import workpad_source
    from ouroboros.tools.project_journal import _workpad_write

    page, journal = store
    ctx = _ctx(tmp_path)
    _workpad_write(ctx, PLAN)
    v1 = _sha(page.read_bytes())
    with journal.open("ab") as stream:
        stream.write(b'{"topic": "workpad", "torn')  # a crashed append: no line end
    with monkeypatch.context() as patched:
        patched.setattr(memory_publication, "write_bytes_atomic",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(5, "EIO")))
        failed = _workpad_write(ctx, mode="edit", old_str="- paint", content="- paint red", expected_revision=v1)
    assert failed.startswith("⚠️ TOOL_ERROR: replace_failed") and page.read_bytes() == PLAN.encode(), failed
    retried = _workpad_write(ctx, mode="edit", old_str="- paint", content="- paint red", expected_revision=v1)
    assert retried.startswith("OK:") and "Replaced " + v1 in retried, retried
    lines = journal.read_bytes().split(b"\n")
    failed_row, landed_row = json.loads(lines[2]), json.loads(lines[3])  # the failed capture, then the retry's row
    assert lines[1] == b'{"topic": "workpad", "torn' and failed_row["new_sha256"] == landed_row["new_sha256"]
    assert landed_row["old_sha256"] == v1 and landed_row["base"]["sha256"] == v1  # its base: the row that wrote v1
    for revision in (v1, _sha(page.read_bytes())):
        assert _sha(knowledge_versions.find_version(workpad_source(PID), revision)[0].encode()) == revision


def test_a_crlf_page_reads_and_edits_byte_for_byte(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_read, _workpad_write

    page, _journal = store
    page.parent.mkdir(parents=True)
    page.write_bytes(b"# Plan\r\n\r\n- one\r\n- two\r\n")
    ctx = _ctx(tmp_path)
    read = _workpad_read(ctx)
    revision = _sha(page.read_bytes())
    assert read.startswith(f"revision={revision} (project {PID}'s workpad, 24 chars") and read.endswith(
        "\n\n# Plan\r\n\r\n- one\r\n- two\r\n"), repr(read)
    reply = _workpad_write(ctx, mode="edit", old_str="- one\r\n", content="- one, done\r\n", expected_revision=revision)
    assert reply.startswith("OK:") and page.read_bytes() == b"# Plan\r\n\r\n- one, done\r\n- two\r\n", reply


def test_a_forked_drive_writes_the_page_and_its_history_on_the_canonical_store(tmp_path, store):
    from ouroboros.tools.project_journal import _workpad_write

    page, journal = store
    fork = tmp_path / "fork"
    fork.mkdir()
    ctx = _ctx(fork)
    ctx.task_metadata = {"budget_drive_root": str(tmp_path / "data"), "root_task_id": "t-1"}
    assert _workpad_write(ctx, PLAN).startswith("OK:")
    assert page.read_bytes() == PLAN.encode() and journal.exists() and page.parent == journal.parent
    assert not (fork / "projects").exists()


def test_update_focus_retains_the_page_bytes_not_the_reads_revision_line(tmp_path, store, monkeypatch):
    from ouroboros.tools import project_journal as pj

    page, _journal = store
    page.parent.mkdir(parents=True)
    page.write_bytes(b"line one\r\nline two\r\n")
    ctx = _ctx(tmp_path)
    monkeypatch.setattr("ouroboros.tools.registry_guards._capability_resource_guard_result", lambda *a, **k: None)
    body, refusal = pj._read_focus_source(ctx, {"reader": "workpad_read", "project_id": PID})
    assert refusal == "" and body == "line one\r\nline two\r\n"
    page.write_bytes(b"x" * (pj._FOCUS_SOURCE_MAX_BYTES + 1))
    body, refusal = pj._read_focus_source(ctx, {"reader": "workpad_read", "project_id": PID})
    assert body == "" and "the workpad has no narrower page" in refusal and "offset/snapshot" not in refusal


def test_the_resident_page_carries_the_revision_of_the_bytes_shown(tmp_path, store):
    from ouroboros.context_sections import build_knowledge_sections

    page, _journal = store
    page.parent.mkdir(parents=True)
    raw = "# План\r\n\r\n- колёса\r\n".encode("utf-8")
    page.write_bytes(raw)
    env = types.SimpleNamespace(drive_path=lambda name: tmp_path / "data" / name)
    sections = [str(section) for section in build_knowledge_sections(env, project_id=PID)]
    pad = next(section for section in sections if section.startswith(f"## Project workpad ({PID})"))
    assert pad == f"## Project workpad ({PID})\n\n" + raw.decode("utf-8") + \
        f"\nRevision of this workpad: {_sha(raw)}.\n"


def test_a_presence_workpad_read_is_a_pure_read_not_possible_speech():
    """SPEC §10, verified on the base first: ``may_speak`` counted a workpad read as words for the place."""
    from ouroboros.place_reply import may_speak
    from ouroboros.tool_capabilities import PURE_READ_TOOLS, READ_ONLY_PARALLEL_TOOLS

    ctx = types.SimpleNamespace(task_metadata={})
    assert may_speak(ctx, {"function": {"name": "workpad_read", "arguments": "{}"}}) is False
    assert "workpad_read" in PURE_READ_TOOLS and "workpad_read" not in READ_ONLY_PARALLEL_TOOLS  # no new parallelism
    assert may_speak(ctx, {"function": {"name": "workpad_write", "arguments": "{}"}}) is False  # a receipt write, as before


def test_scope_refusals_carry_the_resolvers_own_code(tmp_path, store):
    """The project scope's refusal kind comes from its resolver (``_project_scope``), never from its text."""
    from ouroboros.tools.project_journal import _project_scope, _resolve_project_id, _workpad_write

    placed = _ctx(tmp_path, "root", project="mine")
    foreign = _workpad_write(placed, "x", "racer")
    assert foreign.startswith("⚠️ TOOL_FORBIDDEN") and _code(placed) == "ACCESS_BLOCKED", foreign
    malformed = _workpad_write(placed, "x", "a/b")
    assert malformed.startswith("⚠️ TOOL_ARG_ERROR") and _code(placed) == "TOOL_ARG_ERROR", malformed
    unscoped = _ctx(tmp_path, project="")
    assert "no project scope" in _workpad_write(unscoped, "x") and _code(unscoped) == "TOOL_ARG_ERROR"
    assert _project_scope(placed, "racer", write=True)[2] == "ACCESS_BLOCKED"
    assert _resolve_project_id(placed, "racer", write=True) == _project_scope(placed, "racer", write=True)[:2]
