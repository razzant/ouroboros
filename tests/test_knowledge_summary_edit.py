"""An authored summary is revised where the index reads it, never as body text.

``knowledge_write(mode="edit", summary=...)`` revises the resident summary alone or
together with one exact ``old_str`` replacement, through the same revision-checked
note writer, history and index as every other write. The receipt states whether the
body bytes and the resident summary changed; it says nothing about what was learned.
Every test writes only a synthetic data root.
"""

from __future__ import annotations

import inspect
import json
import threading

import pytest

from ouroboros import consolidator, context
from ouroboros import knowledge as store
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests import test_book_context_capture as book

OLD = "Prefers long written reports."
NEW = "Prefers short answers first, detail on request."
NOTE = (f"---\ntype: person\ntitle: Ada\nsummary: {OLD}\ncustom:\n  since: 2026-09-01\n---\n"
        "# Ada\r\nСтарый факт.\r\nKeep this line.\r\n")


def _history(address):
    path = address.shelf.parent / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _files(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _body(note):
    return note.raw[note.source.body_span.start_byte:]


def _receipt(reply):
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


def _setup(tmp_path, topic="people/ada", raw=NOTE):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="summary-edit")
    address = store.resolve_knowledge_address(tmp_path, topic, "global")
    return ctx, address, store.write_knowledge_note(address, raw).current


@pytest.mark.parametrize("scope", ["global", "project:demo"])
def test_registry_summary_edit_reaches_the_next_context_and_not_a_captured_one(tmp_path, monkeypatch, scope):
    project = scope.partition(":")[2]
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "data")
    env, _core, task = book._capture(tmp_path, {"project_id": project} if project else None)
    address = store.resolve_knowledge_address(env.drive_root, "people/ada", scope)
    original = store.write_knowledge_note(address, NOTE).current
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    captured = context._capture_context_core(env, memory, task, None, None)
    earlier_plan = book._plan(env, captured, task, "max")
    earlier_messages = json.dumps(earlier_plan.messages_for("max"), sort_keys=True)
    assert OLD in book._text(earlier_plan, "max")

    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    registry.set_context(ToolContext(repo_dir=env.repo_dir, drive_root=env.drive_root, task_id="summary-edit",
                                     budget_drive_root=str(env.drive_root), project_id=project))
    # Models fill every key: the empty old_str and content of a summary-only call ask for nothing.
    receipt = _receipt(registry.execute("knowledge_write", {
        "topic": "people/ada", "scope": scope, "mode": "edit", "old_str": "", "content": "",
        "summary": NEW, "expected_revision": original.revision}))

    current = store.read_knowledge_note(address)
    assert current.metadata == {**original.metadata, "summary": NEW}
    assert _body(current) == _body(original)  # every body byte, CRLF and Cyrillic included
    delta = receipt["knowledge_delta"]
    assert (delta["body_changed"], delta["summary_changed"]) == (False, True)
    assert receipt["knowledge_source"]["revision"] == current.revision
    row = _history(address)[-1]
    assert (row["mode"], row["writer"], row["task_id"], row["summary"]) == ("edit", "turn", "summary-edit", NEW)
    assert (row["old_content"], row["new_content"], row["delta"]) == (original.text, current.text, delta)
    assert "edits" not in row
    for inventory in ((address.shelf / store.INDEX_FILE).read_text(encoding="utf-8"),
                      registry.execute("knowledge_list", {"scope": scope})):
        assert NEW in inventory and OLD not in inventory

    fresh = context._capture_context_core(env, memory, task, None, None)
    fresh_text = book._text(book._plan(env, fresh, task, "max"), "max")
    assert NEW in fresh_text and OLD not in fresh_text
    assert (f"## Project knowledge ({project})" if project else "## Knowledge base") in fresh_text
    # A context captured before the write is a snapshot: neither its plan nor a new plan from it moves.
    assert json.dumps(earlier_plan.messages_for("max"), sort_keys=True) == earlier_messages
    rebuilt = book._text(book._plan(env, captured, task, "max"), "max")
    assert OLD in rebuilt and NEW not in rebuilt


def test_one_body_replacement_and_its_summary_land_in_one_revision(tmp_path):
    ctx, address, original = _setup(tmp_path)
    receipt = _receipt(tools._knowledge_write(
        ctx, "people/ada", "Новый факт.", mode="edit", scope="global", old_str="Старый факт.",
        summary="Line one.\nLine two.", expected_revision=original.revision))
    current = store.read_knowledge_note(address)
    assert _body(current) == _body(original).replace("Старый факт.".encode(), "Новый факт.".encode())
    assert current.metadata == {**original.metadata, "summary": "Line one.\nLine two."}
    assert (receipt["knowledge_delta"]["body_changed"], receipt["knowledge_delta"]["summary_changed"]) == (True, True)
    rows = _history(address)
    assert len(rows) == 2 and rows[-1]["summary"] == "Line one.\nLine two."
    assert (rows[-1]["old_content"], rows[-1]["new_content"]) == (original.text, current.text)
    assert "  Line one.\n  Line two." in (address.shelf / store.INDEX_FILE).read_text(encoding="utf-8")


def test_summary_text_written_into_the_body_is_reported_as_an_unchanged_summary(tmp_path):
    ctx, address, _original = _setup(tmp_path)
    receipt = _receipt(tools._knowledge_write(ctx, "people/ada", f"\nsummary: {NEW}\n", mode="append", scope="global"))
    assert (receipt["knowledge_delta"]["body_changed"], receipt["knowledge_delta"]["summary_changed"]) == (True, False)
    assert store.read_knowledge_note(address).summary == OLD
    index = (address.shelf / store.INDEX_FILE).read_text(encoding="utf-8")
    assert f"  {OLD}" in index and NEW not in index


@pytest.mark.parametrize("mode", ["overwrite", "append", "edit"])
def test_legacy_calls_and_an_empty_summary_publish_identical_bytes(tmp_path, mode):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="legacy")
    written = []
    for topic, extra in (("plain", {}), ("filled", {"summary": ""})):
        original = store.write_knowledge_note(store.resolve_knowledge_address(tmp_path, topic, "global"), NOTE).current
        args = ({"mode": "edit", "old_str": "Старый факт.", "expected_revision": original.revision}
                if mode == "edit" else {"mode": mode, "expected_revision": original.revision if mode == "overwrite" else None})
        _receipt(tools._knowledge_write(ctx, topic, "Новый факт.", scope="global", **args, **extra))
        written.append((original, store.read_knowledge_note(original.address)))
    (plain_before, plain), (_filled_before, filled) = written
    assert plain.raw == filled.raw
    if mode == "edit":  # an old_str edit without a summary keeps every preamble byte
        assert plain.raw == plain_before.raw.replace("Старый факт.".encode(), "Новый факт.".encode())


REFUSALS = [
    ({"mode": "overwrite", "content": "Body.", "summary": NEW}, "summary is used only with mode=edit"),
    ({"mode": "append", "content": "Body.", "summary": NEW}, "summary is used only with mode=edit"),
    ({"mode": "edit", "summary": "   "}, "is not summary text"),
    ({"mode": "edit", "summary": 7}, "is not summary text"),
    ({"mode": "edit", "summary": NEW, "content": "Orphan replacement."}, "has no old_str to replace"),
    ({"mode": "edit", "summary": NEW, "old_str": "Keep this line."}, "requires content"),
    ({"mode": "edit", "old_str": "Keep this line."}, "requires content"),
    ({"mode": "edit", "content": "Replacement."}, "non-empty old_str, or a summary"),
    ({"mode": "overwrite"}, "mode=overwrite requires content"),
    ({"mode": "append"}, "mode=append requires content"),
]


@pytest.mark.parametrize("args,message", REFUSALS)
def test_ambiguous_or_invalid_calls_refuse_before_any_write(tmp_path, args, message):
    ctx, _address, original = _setup(tmp_path)
    before = _files(tmp_path)
    reply = tools._knowledge_write(ctx, "people/ada", scope="global", expected_revision=original.revision, **args)
    assert reply.startswith("⚠️ TOOL_ARG_ERROR") and message in reply
    assert _files(tmp_path) == before


@pytest.mark.parametrize("old_str", ["Absent line.", "e"])  # missing, then ambiguous
def test_a_refused_body_anchor_publishes_no_summary_either(tmp_path, old_str):
    ctx, address, original = _setup(tmp_path)
    before = _files(tmp_path)
    reply = tools._knowledge_write(ctx, "people/ada", "x", mode="edit", scope="global", old_str=old_str,
                                   summary=NEW, expected_revision=original.revision)
    assert "not completed: invalid_note" in reply and original.text in reply
    assert _files(tmp_path) == before and store.read_knowledge_note(address).summary == OLD


def test_summary_edit_needs_the_current_revision_of_an_existing_readable_note(tmp_path):
    ctx, address, original = _setup(tmp_path)
    before = _files(tmp_path)
    for revision, reason in ((None, "revision_required"), ("stale", "revision_conflict"), ("", "revision_conflict")):
        reply = tools._knowledge_write(ctx, "people/ada", mode="edit", scope="global", summary=NEW,
                                       expected_revision=revision)
        assert f"not completed: {reason}" in reply and original.text in reply
    assert _files(tmp_path) == before
    missing = tools._knowledge_write(ctx, "people/nobody", mode="edit", scope="global", summary=NEW, expected_revision="")
    assert "not completed: edit_source_missing" in missing
    assert not store.resolve_knowledge_address(tmp_path, "people/nobody", "global").path.exists()
    broken = store.resolve_knowledge_address(tmp_path, "people/broken", "global")
    broken.path.write_bytes(b"---\ncustom: [unfinished\n---\nOriginal evidence.\n")
    revision = store.read_knowledge_note(broken).revision
    reply = tools._knowledge_write(ctx, "people/broken", mode="edit", scope="global", summary=NEW,
                                   expected_revision=revision)
    assert "invalid_note: edit requires an existing readable note" in reply
    assert broken.path.read_bytes() == b"---\ncustom: [unfinished\n---\nOriginal evidence.\n"


def test_two_summary_revisions_from_one_read_cannot_both_land(tmp_path):
    _ctx, address, original = _setup(tmp_path)
    start, results = threading.Barrier(3), []

    def write(summary):
        start.wait()
        results.append((summary, store.write_knowledge_note(address, "", "edit", original.revision, summary=summary)))

    workers = [threading.Thread(target=write, args=(text,)) for text in ("View A.", "View B.")]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(3)
    assert all(not worker.is_alive() for worker in workers)
    assert sorted(result.ok for _, result in results) == [False, True]
    winner = next(summary for summary, result in results if result.ok)
    assert next(result.reason for _, result in results if not result.ok) == "revision_conflict"
    assert store.read_knowledge_note(address).summary == winner == _history(address)[-1]["summary"]


def test_a_plain_legacy_note_gains_only_summary_frontmatter(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    address = store.resolve_knowledge_address(tmp_path, "legacy", "global")
    address.path.parent.mkdir(parents=True)
    address.path.write_bytes("# Река\r\nПростой текст.\r\n".encode("utf-8"))
    original = store.read_knowledge_note(address)
    receipt = _receipt(tools._knowledge_write(ctx, "legacy", mode="edit", scope="global", summary="Река.",
                                              expected_revision=original.revision))
    current = store.read_knowledge_note(address)
    assert current.metadata == {"summary": "Река.", "type": "note"}
    assert _body(current) == original.raw
    assert (receipt["knowledge_delta"]["body_changed"], receipt["knowledge_delta"]["summary_changed"]) == (False, True)


def test_crlf_frontmatter_is_rerendered_while_the_crlf_body_stays_exact(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    address = store.resolve_knowledge_address(tmp_path, "windows", "global")
    address.path.parent.mkdir(parents=True)
    address.path.write_bytes(b"---\r\ntype: note\r\nsummary: Old.\r\nkept: [1, 2]\r\n---\r\nBody one.\r\nBody two.\r\n")
    original = store.read_knowledge_note(address)
    _receipt(tools._knowledge_write(ctx, "windows", "Body 2.", mode="edit", scope="global", old_str="Body two.",
                                    summary="New.", expected_revision=original.revision))
    current = store.read_knowledge_note(address)
    assert current.metadata == {"type": "note", "summary": "New.", "kept": [1, 2]}
    assert _body(current) == b"Body one.\r\nBody 2.\r\n"


def test_restating_the_current_summary_writes_nothing(tmp_path):
    ctx, _address, original = _setup(tmp_path)
    before = _files(tmp_path)
    receipt = _receipt(tools._knowledge_write(ctx, "people/ada", mode="edit", scope="global", summary=OLD,
                                              expected_revision=original.revision))
    assert receipt["knowledge_write_reason"] == "unchanged"
    assert (receipt["knowledge_delta"]["body_changed"], receipt["knowledge_delta"]["summary_changed"]) == (False, False)
    assert _files(tmp_path) == before


def test_summary_revision_keeps_a_recursive_yaml_field(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    address = store.resolve_knowledge_address(tmp_path, "recursive", "global")
    address.path.parent.mkdir(parents=True)
    address.path.write_bytes(b"---\ntype: note\ncustom: &loop [*loop]\nsummary: Old.\n---\nBody.\n")
    original = store.read_knowledge_note(address)
    assert original.source is not None and not original.parse_error
    _receipt(tools._knowledge_write(ctx, "recursive", mode="edit", scope="global", summary="New.",
                                    expected_revision=original.revision))
    current = store.read_knowledge_note(address)
    assert current.summary == "New." and current.metadata["custom"][0] is current.metadata["custom"]
    assert _body(current) == b"Body.\n"


def test_unparseable_source_leaves_both_change_facts_unknown(tmp_path):
    address = store.resolve_knowledge_address(tmp_path, "malformed", "global")
    address.path.parent.mkdir(parents=True)
    address.path.write_bytes(b"---\ncustom: [unfinished\n---\n# Old heading\n")
    appended = store.write_knowledge_note(address, "New evidence.\n", mode="append")
    assert appended.ok and (appended.delta["body_changed"], appended.delta["summary_changed"]) == (None, None)


def test_backlog_summary_is_refused_rather_than_ignored(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    item = "### ibl-1\n- summary: Trim the index preview.\n- category: memory\n"
    for args in ({"mode": "edit", "summary": NEW}, {"mode": "overwrite", "content": item, "summary": NEW}):
        assert tools._knowledge_write(ctx, tools.BACKLOG_TOPIC, **args).startswith("⚠️ TOOL_ARG_ERROR")
    assert not (tmp_path / "memory").exists()


def test_automatic_nomination_keeps_its_edits_form_and_gains_the_same_change_facts(tmp_path):
    ctx, address, original = _setup(tmp_path)
    reads = consolidator.KnowledgeReadContext(ctx)
    reads.reads[(address.scope, address.topic)] = original.revision
    edit = {"old_text": "Старый факт.", "new_text": "Новый факт.", "basis": "The episode moved it."}
    outcome = consolidator._write_knowledge_entries(address.shelf, reads.bind_entries([
        {"topic": "people/ada", "scope": "global", "edits": [edit], "summary": NEW}]), context=ctx)[0]
    assert outcome["ok"]
    row = _history(address)[-1]
    assert (row["edits"], row["summary"]) == ([edit], NEW)
    assert (row["delta"]["body_changed"], row["delta"]["summary_changed"]) == (True, True)


def test_schema_adds_summary_without_breaking_older_calls(tmp_path):
    schema = next(entry.schema for entry in tools.get_tools() if entry.name == "knowledge_write")
    props = schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["topic"]
    assert props["summary"]["type"] == "string" and "mode=edit only" in props["summary"]["description"]
    assert {"topic", "scope", "content", "mode", "old_str", "expected_revision"} < set(props)
    assert props["mode"]["enum"] == ["overwrite", "append", "edit", "archive", "restore"]
    assert set(props) == set(inspect.signature(tools._knowledge_write).parameters) - {"ctx"}
    assert "body text never replaces it" in schema["description"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    assert tools._knowledge_write(ctx, "positional", "Body.").startswith("✅")  # the oldest call shape
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    # Omitted content is the handler's typed refusal, not a signature-binding failure.
    assert "mode=overwrite requires content" in registry.execute("knowledge_write", {"topic": "no-content"})
    assert not store.resolve_knowledge_address(tmp_path, "no-content", "global").path.exists()
