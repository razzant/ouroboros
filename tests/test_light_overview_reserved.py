"""The overview is the acting mind's own words; a Light helper never writes it.

The overview note is "what I hold true now", loaded into every context. No Light
operation writes knowledge any more: the post-task scratchpad upkeep and reflection
that nominated notes are retired (#918: the mind folds its own scratchpad). The one
Light helper left, the fallback memory writer over the Light transport, drafts
a page even when its answer still nominates the overview (and a neighbour beside it),
and writes neither. The mind's own ``knowledge_write`` of the overview is untouched.
"""

from __future__ import annotations

import json

from ouroboros import knowledge as store
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext

_MIND_OVERVIEW = "---\nsummary: What I hold true now.\n---\n# Orientation\n\nThe mind's own words.\n"
_OVERVIEW_EDIT = {"topic": "overview", "scope": "global", "edits": [
    {"old_text": "The mind's own words.", "new_text": "A helper's rewrite.", "basis": "This episode."}]}
_NEIGHBOUR = {"topic": "lessons/neighbour", "scope": "global", "content": "# Neighbour\n\nA durable lesson.\n"}


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _overview(root):
    return store.resolve_knowledge_address(root, store.OVERVIEW_TOPIC, "global")


def _neighbour(root):
    return store.resolve_knowledge_address(root, "lessons/neighbour", "global")


def _mind_writes_the_overview(root) -> bytes:
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="mind-turn")
    assert "✅" in knowledge_tools._knowledge_write(ctx, "overview", _MIND_OVERVIEW, mode="overwrite")
    return _overview(root).path.read_bytes()


def test_a_light_draft_nominating_the_overview_and_a_neighbour_writes_neither(tmp_path, monkeypatch):
    from tests import _memory_inventory_shared as shared
    from tests.test_memory_fallback import _Light, _consciousness, _run, light_route

    shared.world(tmp_path)
    _consciousness(monkeypatch, False)
    light_route(monkeypatch)
    before = _mind_writes_the_overview(tmp_path)
    answer = json.dumps({"text": "The owner asked; Ouroboros answered.", "quotes": [],
                         "knowledge_entries": [_OVERVIEW_EDIT, _NEIGHBOUR]})
    llm = _Light(answer)

    assert _run(tmp_path, llm).outcome == "published"  # the draft acts; its nominations are ignored

    prompt = _normalized(llm.prompts[0])
    assert len(llm.prompts) == 1 and "knowledge_entries" not in prompt and "nominat" not in prompt
    assert _overview(tmp_path).path.read_bytes() == before
    assert not _neighbour(tmp_path).path.exists()


def test_the_mind_still_writes_and_revises_the_overview(tmp_path):
    first = _mind_writes_the_overview(tmp_path)
    assert b"The mind's own words." in first
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mind-turn-2")
    current = store.read_knowledge_note(_overview(tmp_path))
    assert "✅" in knowledge_tools._knowledge_write(
        ctx, "overview", "# Orientation\n\nWhat changed is now true.\n", mode="overwrite",
        expected_revision=current.revision)
    assert "What changed is now true." in store.read_knowledge_note(_overview(tmp_path)).text
    history = [json.loads(line) for line in
               (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["writer"] for row in history if row.get("topic") == "overview"
            and row.get("publication") == "source_capture"] == ["turn", "turn"]
