"""A reader opens the archive receipt through the ``read_file`` tool it advertises.

This complements archive lifecycle tests: the historical source still names the
pre-archive bytes after edits and restore, while current reading stays current, and
a reader on the canonical drive finds it before a forked task's copyback.
"""
import hashlib
import json
from pathlib import Path

import pytest

from ouroboros.tools.core_file_tools import _render_line_slice, delivered_source_prefix
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _write(registry, **arguments):
    reply = registry.execute("knowledge_write", arguments)
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


@pytest.mark.parametrize("scope", ["global", "project:reader-check"])
def test_a_reader_reads_exact_previous_source_after_restore(tmp_path, scope):
    canonical = tmp_path / "canonical"
    execution = tmp_path / "execution"
    canonical.mkdir()
    execution.mkdir()
    ctx = ToolContext(repo_dir=Path.cwd(), drive_root=execution,
                      budget_drive_root=str(canonical), task_id="reader-consumer",
                      project_id="reader-check")
    registry = ToolRegistry(repo_dir=Path.cwd(), drive_root=execution)
    registry.set_context(ctx)
    original = "---\r\ntype: note\r\nsummary: Earlier meaning\r\n---\r\n# Берег\r\nEarlier facts.\r\n"
    created = _write(registry, topic="reader-source", scope=scope, mode="overwrite", content=original)
    revision = created["knowledge_source"]["revision"]
    archived = _write(registry, topic="reader-source", scope=scope, mode="archive",
                      expected_revision=revision, reason="Retained in an ordinary linked overview.")
    ref = archived["knowledge_previous_source"]
    changed = _write(registry, topic="reader-source", scope=scope, mode="edit",
                     expected_revision=archived["knowledge_source"]["revision"],
                     old_str="Earlier facts.", content="Later corrected facts.")
    _write(registry, topic="reader-source", scope=scope, mode="restore",
           expected_revision=changed["knowledge_source"]["revision"])
    assert ref["read"]["tool"] == "read_file"
    shown = registry.execute(ref["read"]["tool"], ref["read"]["arguments"])
    # A numbered display; the reader's own recorded view maps it back to source.
    payload = json.loads(delivered_source_prefix(registry._ctx.last_read_view, shown, len(shown)))
    raw = payload[ref["field"]].encode("utf-8")
    assert payload["revision"] == revision == hashlib.sha256(raw).hexdigest()
    assert b"Earlier facts." in raw and b"Later corrected facts." not in raw
    assert b"\r\n" in raw
    current = registry.execute("knowledge_read", {"topic": "reader-source", "scope": scope})
    assert '"state": "active"' in current and "Later corrected facts." in current


@pytest.mark.parametrize(("mode", "scope"), [("archive", "global"), ("restore", "project:reader-check")])
def test_a_canonical_reader_reads_previous_source_before_copyback(tmp_path, scope, mode):
    from ouroboros.artifacts import read_actor_source_bytes
    from tests.test_knowledge_archive import setup, transition

    ctx, registry, address, note = setup(tmp_path, scope, fork=True)
    if mode == "restore":
        _write(registry, topic=address.topic, scope=scope, mode="archive",
               expected_revision=note.revision, reason="Retain the original.")
        from ouroboros.knowledge import read_knowledge_note
        note = read_knowledge_note(address)
    original = note.raw
    reply = transition(registry, note, mode)
    assert reply.startswith("✅"), reply
    ref = json.loads(reply.split("\n", 1)[1])["knowledge_previous_source"]
    # The actor's existing reader is still usable as soon as it gets the handle.
    exact = read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref)
    _write(registry, topic=address.topic, scope=scope, mode="edit",
           expected_revision=json.loads(reply.split("\n", 1)[1])["knowledge_source"]["revision"],
           old_str="Старый берег.", content="LATER CURRENT BODY")
    # A reader on the CANONICAL drive (the forked task has not copied back yet).
    canonical = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    canonical.set_context(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=ctx.task_id))
    delivered = canonical.execute(ref["read"]["tool"], ref["read"]["arguments"])
    assert "old_content" in delivered
    view = {}  # the delivered numbered display, mapped by the renderer that produced it
    _render_line_slice(ref["path"], exact.decode("utf-8"), extent=view)
    view["body_start"] = delivered.find("\n") + 1
    payload = json.loads(delivered_source_prefix(view, delivered, len(delivered)))
    assert payload["old_content"].encode("utf-8") == original
    assert payload["revision"] == hashlib.sha256(original).hexdigest()
    assert "LATER CURRENT BODY" not in delivered
    assert read_actor_source_bytes(tmp_path, ctx.task_id, ref) == exact


@pytest.mark.parametrize(("failing_root", "scope"), [("canonical", "global"), ("execution", "project:reader-check")])
def test_archive_requires_previous_source_in_both_reader_roots(tmp_path, monkeypatch, scope, failing_root):
    from ouroboros import artifacts
    from tests.test_knowledge_archive import setup, transition

    ctx, registry, address, note = setup(tmp_path, scope, fork=True)
    target = tmp_path if failing_root == "canonical" else ctx.drive_root
    before = {p: p.read_bytes() for p in address.shelf.parent.rglob("*")
              if p.is_file() and p.suffix != ".lock"}
    persist = artifacts.persist_exact_text_source

    def fail_selected_reader(root, *args, **kwargs):
        if Path(root) == target:
            return "", {}, {"reason": "reader source store unavailable"}
        return persist(root, *args, **kwargs)

    monkeypatch.setattr(artifacts, "persist_exact_text_source", fail_selected_reader)
    result = transition(registry, note)
    assert "source_capture_unavailable" in result
    assert "knowledge_previous_source" not in result
    assert {p: p.read_bytes() for p in address.shelf.parent.rglob("*")
            if p.is_file() and p.suffix != ".lock"} == before
