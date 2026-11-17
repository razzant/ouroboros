"""An authored summary is revised where the index reads it, never as body text.

``knowledge_write(mode="edit", summary=...)`` revises the resident summary alone or
together with one exact ``old_str`` replacement, and ``mode="append"`` together with
the appended text, through the same revision-checked note writer, history and index
as every other write. The receipt states whether the
body bytes and the resident summary changed; it says nothing about what was learned.
Every test writes only a synthetic data root.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import threading

import pytest

from ouroboros import context
from ouroboros import knowledge as store
from ouroboros import knowledge_versions as versions
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests import test_book_context_capture as book
from tests._cases import case_folders

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
    from tests._knowledge_history_shared import published, replaced
    assert (replaced(address.canonical_root, row), published(address.canonical_root, row), row["delta"]) == (
        original.text, current.text, delta)
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
    from tests._knowledge_history_shared import published, replaced
    assert (replaced(address.canonical_root, rows[-1]), published(address.canonical_root, rows[-1])) == (
        original.text, current.text)
    assert "  Line one.\n  Line two." in (address.shelf / store.INDEX_FILE).read_text(encoding="utf-8")


def test_an_append_and_its_summary_land_in_one_revision(tmp_path):
    ctx, address, original = _setup(tmp_path)
    plain = _receipt(tools._knowledge_write(ctx, "people/ada", "First fact.", mode="append", scope="global"))
    kept = store.read_knowledge_note(address)  # adding text alone stays blind and keeps the summary
    assert kept.summary == OLD and plain["knowledge_delta"]["summary_changed"] is False
    before = _files(tmp_path)  # replacing the summary beside it needs the current revision
    reply = tools._knowledge_write(ctx, "people/ada", "Second fact.", mode="append", scope="global", summary=NEW)
    assert "not completed: revision_required" in reply and _files(tmp_path) == before
    receipt = _receipt(tools._knowledge_write(ctx, "people/ada", "Second fact.", mode="append", scope="global",
                                              summary=NEW, expected_revision=kept.revision))
    current = store.read_knowledge_note(address)
    assert current.metadata == {**original.metadata, "summary": NEW}
    assert _body(current) == _body(kept) + b"\nSecond fact."
    assert (receipt["knowledge_delta"]["body_changed"], receipt["knowledge_delta"]["summary_changed"]) == (True, True)
    rows = _history(address)
    assert len(rows) == 3 and (rows[-1]["mode"], rows[-1]["summary"]) == ("append", NEW)
    index = (address.shelf / store.INDEX_FILE).read_text(encoding="utf-8")
    assert f"  {NEW}" in index and OLD not in index
    _receipt(tools._knowledge_write(ctx, "people/new", "Met today.", mode="append", scope="global", summary=NEW))
    assert store.read_knowledge_note(store.resolve_knowledge_address(tmp_path, "people/new", "global")).summary == NEW


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
    ({"mode": "restore", "summary": NEW}, "used only with mode=overwrite, append or edit, not mode=restore"),
    ({"mode": "edit", "summary": "   "}, "is not summary text"),
    ({"mode": "edit", "summary": 7}, "is not summary text"),
    ({"mode": "edit", "summary": NEW, "content": "Orphan replacement."}, "has no old_str to replace"),
    ({"mode": "edit", "summary": NEW, "old_str": "Keep this line."}, "requires content"),
    ({"mode": "edit", "old_str": "Keep this line."}, "requires content"),
    ({"mode": "edit", "content": "Replacement."}, "non-empty old_str, or a summary"),
    ({"mode": "overwrite"}, "mode=overwrite requires content"),
    ({"mode": "append"}, "mode=append requires content"),
    ({"summary": NEW}, "mode is missing or null"),  # a summary revision still names its edit
]


def test_ambiguous_or_invalid_calls_refuse_before_any_write(subtests, tmp_path):
    case_paths = case_folders(tmp_path)
    for args, message in REFUSALS:
        tmp_path = next(case_paths)
        with subtests.test(args=args, message=message):
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


def test_the_backlog_summary_is_an_ordinary_revision_checked_edit(tmp_path):
    """The backlog has no merge writer of its own: its summary changes like any note's."""
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="summary-edit")
    item = "- Trim the index preview.\n"
    assert tools._knowledge_write(ctx, tools.BACKLOG_TOPIC, mode="restore", summary=NEW).startswith(
        "⚠️ TOOL_ARG_ERROR")
    assert not (tmp_path / "memory").exists()
    _receipt(tools._knowledge_write(ctx, tools.BACKLOG_TOPIC, item, mode="overwrite"))
    address = store.resolve_knowledge_address(tmp_path, tools.BACKLOG_TOPIC, "global")
    created = store.read_knowledge_note(address)
    stale = tools._knowledge_write(ctx, tools.BACKLOG_TOPIC, mode="edit", summary=NEW)
    assert "revision_required" in stale and store.read_knowledge_note(address).raw == created.raw
    receipt = _receipt(tools._knowledge_write(ctx, tools.BACKLOG_TOPIC, mode="edit", summary=NEW,
                                              expected_revision=created.revision))
    assert receipt["knowledge_delta"]["summary_changed"] is True and receipt["knowledge_delta"]["body_changed"] is False
    current = store.read_knowledge_note(address)
    assert current.summary == NEW and _body(current) == _body(created)
    assert _history(address)[-1]["summary"] == NEW


def test_schema_adds_summary_without_breaking_older_calls(tmp_path):
    schema = next(entry.schema for entry in tools.get_tools() if entry.name == "knowledge_write")
    props = schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["topic", "mode"]
    assert props["summary"]["type"] == "string" and "when to open this note" in props["summary"]["description"]
    assert "overwrite, append or edit" in props["summary"]["description"]
    assert {"topic", "scope", "content", "mode", "old_str", "expected_revision"} < set(props)
    assert props["mode"]["enum"] == ["overwrite", "append", "edit", "revert", "archive", "restore"]
    assert set(props) == set(inspect.signature(tools._knowledge_write).parameters) - {"ctx"}
    assert "body text never replaces it" in schema["description"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    # The oldest call shape keeps its positions (topic, content, mode); since mode became required, it names one.
    assert tools._knowledge_write(ctx, "positional", "Body.", "overwrite").startswith("✅")
    assert tools._knowledge_write(ctx, "unnamed", "Body.").startswith("⚠️ TOOL_ARG_ERROR: mode is missing or null.")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    # An omitted mode or content is the handler's typed refusal, not a signature-binding failure.
    for arguments, refusal in (({}, "⚠️ TOOL_ARG_ERROR: mode is missing or null."),
                               ({"mode": "overwrite"}, "mode=overwrite requires content")):
        reply = registry.execute("knowledge_write", {"topic": "no-content", **arguments})
        assert refusal in reply and "invalid arguments for knowledge_write" not in reply, reply
    for topic in ("unnamed", "no-content"):
        assert not store.resolve_knowledge_address(tmp_path, topic, "global").path.exists()


def test_an_overwrite_takes_its_summary_after_its_own_merge(tmp_path):
    ctx, address, original = _setup(tmp_path)
    content = "---\nsummary: From the content.\n---\n# Ada\n\nRewritten.\n"
    assert "revision_required" in tools._knowledge_write(ctx, "people/ada", content, mode="overwrite", scope="global",
                                                         summary=NEW)
    assert store.read_knowledge_note(address).raw == original.raw
    receipt = _receipt(tools._knowledge_write(ctx, "people/ada", content, mode="overwrite", scope="global", summary=NEW,
                                              expected_revision=original.revision))
    current = store.read_knowledge_note(address)
    assert current.summary == NEW  # the parameter wins over the content's own frontmatter
    assert {key: current.metadata[key] for key in ("type", "title", "custom")} == {
        key: original.metadata[key] for key in ("type", "title", "custom")}
    assert _body(current) == b"# Ada\n\nRewritten.\n" and receipt["knowledge_delta"]["summary_changed"] is True
    assert (_history(address)[-1]["mode"], _history(address)[-1]["summary"]) == ("overwrite", NEW)
    plain = _receipt(tools._knowledge_write(ctx, "people/ada", content, mode="overwrite", scope="global",
                                            expected_revision=current.revision))
    assert store.read_knowledge_note(address).summary == "From the content." and plain["knowledge_delta"]["summary_changed"]


def test_a_bare_append_is_answered_with_the_outline_and_says_earlier_text_stood(tmp_path):
    ctx, address, _original = _setup(tmp_path)
    reply = tools._knowledge_write(ctx, "people/ada", "\n## Later\nNew fact.\n", mode="append", scope="global")
    assert reply.count("\n") == 1  # the meta holds this outline: nothing follows the JSON line
    receipt = _receipt(reply)
    assert receipt["knowledge_note"] == "Earlier text was not revised by this write."
    assert receipt["knowledge_outline"].startswith("Outline of 'people/ada' (global)")
    assert "# Ada — line 8" in receipt["knowledge_outline"] and "## Later — line" in receipt["knowledge_outline"]
    assert "New fact." not in json.dumps(receipt, ensure_ascii=False)  # the appended text is not echoed
    assert "revision_required" in tools._knowledge_write(ctx, "people/ada", "More.", mode="append", scope="global",
                                                         summary=NEW)
    current = store.read_knowledge_note(address)
    revised = _receipt(tools._knowledge_write(ctx, "people/ada", "More.", mode="append", scope="global", summary=NEW,
                                              expected_revision=current.revision))
    assert "knowledge_outline" not in revised and "knowledge_note" not in revised


def test_a_bare_append_whose_outline_the_receipt_cannot_hold_carries_it_after_the_json_line(tmp_path):
    raw = "---\ntype: note\n---\n# Журнал\n\n" + "".join(f"## Запись номер {i:03d}\n\nТекст.\n\n" for i in range(120))
    ctx, address, _original = _setup(tmp_path, "journal/long", raw)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(ctx)
    reply = tools._knowledge_write(ctx, "journal/long", "\n## Позже\n\nНовое.\n", mode="append", scope="global")
    line, _, carried = reply.split("\n", 1)[1].partition("\n\n")
    receipt = json.loads(line)
    assert receipt["knowledge_outline"]["omitted"] is True  # the meta cannot hold it ...
    assert carried == versions.render_outline(store.read_knowledge_note(address))  # ... so it follows, whole
    assert receipt["knowledge_outline"]["sha256"] == hashlib.sha256(carried.encode("utf-8")).hexdigest()
    assert "## Позже — line" in carried and receipt["knowledge_note"] == "Earlier text was not revised by this write."
    assert receipt["knowledge_delta"]["added_headings"] == ["## Позже"]  # the outline gave way first, as the largest
    pointer = receipt["knowledge_outline"]["read"]
    assert registry.execute(pointer["tool"], pointer["arguments"]).endswith(carried)  # the meta's pointer reads it too
