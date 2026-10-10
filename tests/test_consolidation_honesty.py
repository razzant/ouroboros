"""What stays honest after the old dialogue writer is retired.

The shared knowledge-maintenance prompt still tells Light to author summaries; a
nominated note with a summary still becomes resident in the index; Health keeps its
own-memory lines for identity and scratchpad and no longer reports the retired
writer's state, whatever the frozen ``dialogue_meta.json`` holds (its pending
nominations are imported as marks); and the frozen cursor is still read strictly, so
an unreadable file is never taken for an empty one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_health
from ouroboros.memory_nomination_receipts import DialogueMetaUnreadable, parse_meta
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json


def _health_env(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                           repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)


def test_light_is_told_to_author_summaries_and_keep_explicit_requests_explicit():
    # Light never sees the knowledge_write schema, so the maintenance prompt is its only carrier.
    assert "YAML summary" in c.KNOWLEDGE_MAINTENANCE_PROMPT
    assert "resident in the index" in c.KNOWLEDGE_MAINTENANCE_PROMPT
    assert "explicit standing request stays explicit" in c.KNOWLEDGE_MAINTENANCE_PROMPT


def test_a_nominated_note_with_a_summary_becomes_resident_in_the_index(tmp_path):
    from ouroboros.knowledge import inventory_knowledge, render_knowledge_index, resolve_knowledge_address
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path), task_id="t")
    entry = {"topic": "people/alex", "scope": "global", "expected_revision": None,
             "content": "---\ntype: understanding\nsummary: Alex asks for brevity; an interpretation to test.\n---\nEvidence."}
    outcomes = c._write_knowledge_entries(tmp_path / "memory" / "knowledge", [entry], context=ctx)
    assert outcomes and outcomes[0]["ok"]
    rendered = render_knowledge_index(inventory_knowledge(resolve_knowledge_address(tmp_path, "overview", "global")))
    assert "Alex asks for brevity" in rendered


def test_memory_health_lines_still_carry_identity_and_scratchpad(tmp_path):
    """The extracted helper keeps the existing own-memory checks it was split from."""
    env = _health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("thin", encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("x", encoding="utf-8")
    lines = context_health._memory_health_lines(env)
    assert any("THIN IDENTITY" in line for line in lines)
    assert any("EMPTY SCRATCHPAD" in line for line in lines)


@pytest.mark.parametrize("meta", [
    {"pending_knowledge_nominations": [{"id": "src:0:0", "scope": "global", "topic": "t", "reason": "x"}],
     "last_unpublished_nominations": {"entry_id": "abc", "failed": 1, "total": 1},
     "last_consolidation_error": {"kind": "provider_failed", "cursor_offset": 0},
     "era_retry": {"0" * 64: {"route": {"model": "m"}}}},
    b'{"pending_knowledge_nominations":[{"id":"old"}]',
])
def test_health_carries_no_retired_dialogue_writer_state(tmp_path, meta):
    env = _health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros. " * 20, encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("working notes " * 10, encoding="utf-8")
    path = tmp_path / "memory" / "dialogue_meta.json"
    if isinstance(meta, bytes):
        path.write_bytes(meta)
    else:
        atomic_write_json(path, meta)
    lines = context_health._memory_health_lines(env)
    assert not any("DIALOGUE" in line for line in lines)
    assert "OK: identity.md recent" in lines and any(line.startswith("OK: scratchpad size") for line in lines)


def test_frozen_cursor_reads_strictly(tmp_path):
    path = tmp_path / "dialogue_meta.json"
    atomic_write_json(path, {"last_consolidated_offset": 42, "pending_knowledge_nominations": [{"id": "a:0:0"}]})
    assert parse_meta(path.read_bytes())["last_consolidated_offset"] == 42


@pytest.mark.parametrize("bad_bytes", [b'{"pending_knowledge_nominations":[{"id":"old"}]',
                                       b'["wrong top-level type"]',
                                       b'{"pending_knowledge_nominations":[],"pending_knowledge_nominations":[]}',
                                       b'{"pending_knowledge_nominations":{"not":"a list"}}',
                                       b'{"pending_knowledge_nominations":[{"id":"a"},{"id":"a"}]}'])
def test_unreadable_frozen_cursor_is_never_read_as_empty(bad_bytes):
    with pytest.raises(DialogueMetaUnreadable):
        parse_meta(bad_bytes)


def test_the_import_reads_the_cursor_through_the_same_strict_parser(monkeypatch):
    """One strict reader: the chronicle import refuses exactly what ``parse_meta`` refuses."""
    from ouroboros import chronicle_import, memory_nomination_receipts

    seen = []
    real = memory_nomination_receipts.parse_meta
    monkeypatch.setattr(memory_nomination_receipts, "parse_meta", lambda raw: (seen.append(raw), real(raw))[1])
    good = b'{"last_consolidated_offset": 3, "pending_knowledge_nominations": [{"id": "a:0:0"}]}'
    errors = {}
    assert chronicle_import._cursor({"meta": good}, errors)["last_consolidated_offset"] == 3 and not errors
    bad = b'{"pending_knowledge_nominations":[{"id":"a"},{"id":"a"}]}'
    assert chronicle_import._cursor({"meta": bad}, errors) is None and "Duplicate" in errors["meta"]
    assert seen == [good, bad]
