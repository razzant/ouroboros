"""What stays honest after the old dialogue writer is retired.

An authored note with a summary still becomes resident in the index (Light no
longer nominates notes at all); Health states when identity.md was last written as a
fact, passes no verdict on its length or the scratchpad's, and no longer reports the
retired writer's state, whatever the frozen ``dialogue_meta.json`` holds (its pending
nominations are imported as marks); and the frozen cursor is still read strictly, so an
unreadable file is never taken for an empty one.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from ouroboros import context_health
from ouroboros.memory_nomination_receipts import DialogueMetaUnreadable, parse_meta
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json


def _health_env(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                           repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)


def test_an_authored_note_with_a_summary_becomes_resident_in_the_index(tmp_path):
    from ouroboros.knowledge import inventory_knowledge, render_knowledge_index, resolve_knowledge_address
    from ouroboros.tools.knowledge import _knowledge_write

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path), task_id="t")
    content = "---\ntype: understanding\nsummary: Alex asks for brevity; an interpretation to test.\n---\nEvidence."
    assert _knowledge_write(ctx, "people/alex", content, scope="global").startswith("✅")
    rendered = render_knowledge_index(inventory_knowledge(resolve_knowledge_address(tmp_path, "overview", "global")))
    assert "Alex asks for brevity" in rendered


def test_identity_age_is_a_plain_fact_and_memory_sizes_get_no_verdict(tmp_path):
    """A four-character identity and a one-character scratchpad: the block says when identity.md
    was written and nothing else about either file (the STALE/THIN/EMPTY/BLOATED verdicts and their
    OK lines left with the owner's rule that no threshold stands in for judgment)."""
    env = _health_env(tmp_path)
    (tmp_path / "memory" / "identity.md").write_text("thin", encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("x", encoding="utf-8")
    text = context_health.build_health_invariants(env)
    assert re.search(r"^- identity\.md last updated \d+\.\d h ago$", text, re.MULTILINE)
    for verdict in ("STALE IDENTITY", "THIN IDENTITY", "EMPTY SCRATCHPAD", "BLOATED SCRATCHPAD", "OK:", "scratchpad"):
        assert verdict not in text, verdict
    (tmp_path / "memory" / "identity.md").unlink()
    assert "identity.md" not in context_health.build_health_invariants(env)  # absent: no fact, never a verdict


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
    text = context_health.build_health_invariants(env)
    assert "DIALOGUE" not in text and "OK:" not in text and "identity.md last updated 0.0 h ago" in text


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
