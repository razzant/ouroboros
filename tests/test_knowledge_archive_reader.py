"""Reflection must read the archive receipt through its own file tool.

This complements archive lifecycle tests: the historical source still names the
pre-archive bytes after edits and restore, while current reading stays current.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.consolidator import KnowledgeReadContext
from ouroboros.tools.registry import ToolContext, ToolRegistry


def _write(registry, **arguments):
    reply = registry.execute("knowledge_write", arguments)
    assert reply.startswith("✅"), reply
    return json.loads(reply.split("\n", 1)[1])


@pytest.mark.parametrize("scope", ["global", "project:reader-check"])
def test_reflection_reads_exact_previous_source_after_restore(tmp_path, scope):
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
    created = _write(registry, topic="reader-source", scope=scope, content=original)
    revision = created["knowledge_source"]["revision"]
    archived = _write(registry, topic="reader-source", scope=scope, mode="archive",
                      expected_revision=revision, reason="Retained in an ordinary linked overview.")
    ref = archived["knowledge_previous_source"]
    changed = _write(registry, topic="reader-source", scope=scope, mode="edit",
                     expected_revision=archived["knowledge_source"]["revision"],
                     old_str="Earlier facts.", content="Later corrected facts.")
    _write(registry, topic="reader-source", scope=scope, mode="restore",
           expected_revision=changed["knowledge_source"]["revision"])
    reader = KnowledgeReadContext(ctx, "reflection")
    allowed = {item["function"]["name"] for item in reader.tools}
    assert ref["read"]["tool"] == "read_file" and "read_file" in allowed
    observed = reader.read_call({"id": "historical-source", "function": {
        "name": ref["read"]["tool"], "arguments": json.dumps(ref["read"]["arguments"])}})
    assert observed["status"] == "ok", observed
    payload = json.loads(observed["result"].split("\n", 1)[1])
    raw = payload[ref["field"]].encode("utf-8")
    assert payload["revision"] == revision == hashlib.sha256(raw).hexdigest()
    assert b"Earlier facts." in raw and b"Later corrected facts." not in raw
    assert b"\r\n" in raw
    current = registry.execute("knowledge_read", {"topic": "reader-source", "scope": scope})
    assert '"state": "active"' in current and "Later corrected facts." in current


@pytest.mark.parametrize("scope", ["global", "project:reader-check"])
@pytest.mark.parametrize("mode", ["archive", "restore"])
def test_run_reflection_reads_previous_source_before_copyback(tmp_path, monkeypatch, scope, mode):
    from ouroboros import consolidator, post_task_synthesis
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
    monkeypatch.setattr(consolidator, "_consolidation_route", lambda: ("test/model", False))
    observed = []

    class Reader:
        def chat(self, *, messages, **kwargs):
            import copy
            observed.append(copy.deepcopy(messages))
            if len(observed) == 1:
                return {"tool_calls": [{"id": "previous-source", "type": "function", "function": {
                    "name": ref["read"]["tool"], "arguments": json.dumps(ref["read"]["arguments"]),
                }}]}, {}
            return {"content": "The recorded previous bytes are available.\nMEMORY_ACTIONS_JSON: []"}, {}

    task = {"id": ctx.task_id, "type": "task", "text": "Archive and preserve the earlier source.",
            "budget_drive_root": str(tmp_path), "project_id": ctx.project_id}
    trace = {"tool_calls": [{"tool": "knowledge_write", "status": "ok", "result": reply}]}
    entry = post_task_synthesis._run_reflection(
        SimpleNamespace(repo_dir=tmp_path, drive_root=ctx.drive_root), Reader(), task,
        {"rounds": 15}, trace, {"task_id": ctx.task_id, "has_evidence": True,
                                "tool_trajectory": trace["tool_calls"]})
    assert entry and len(observed) == 2, entry
    assert ref["path"] in json.dumps(observed[0]), "the real prompt advertises the previous reader"
    delivered = [row["content"] for row in observed[1] if row.get("role") == "tool"]
    assert len(delivered) == 1 and "old_content" in delivered[0], delivered
    payload = json.loads(delivered[0].split("\n", 1)[1])
    assert payload["old_content"].encode("utf-8") == original
    assert payload["revision"] == hashlib.sha256(original).hexdigest()
    assert "LATER CURRENT BODY" not in delivered[0]
    assert read_actor_source_bytes(tmp_path, ctx.task_id, ref) == exact


@pytest.mark.parametrize("scope", ["global", "project:reader-check"])
@pytest.mark.parametrize("failing_root", ["canonical", "execution"])
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
