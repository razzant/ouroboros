"""The overview is the acting mind's own words; a Light helper never writes it.

The overview note is "what I hold true now", loaded into every context. No Light
operation writes knowledge any more: the post-task scratchpad upkeep that last carried
this check is retired (#918: the mind folds its own scratchpad), and so is the post-task
reflection with its shared nomination writer. An answer of the shared Light transport
that still nominates the overview (and a neighbour beside it) after reading it whole
writes neither. The mind's own ``knowledge_write`` of the overview is untouched.
"""

from __future__ import annotations

import json

from ouroboros import consolidator as c
from ouroboros import knowledge as store
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit

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
    assert "✅" in knowledge_tools._knowledge_write(ctx, "overview", _MIND_OVERVIEW)
    return _overview(root).path.read_bytes()


class ReadsTheOverviewFirst:
    """A Light actor that reads the whole current overview, then answers."""

    def __init__(self, answer: str):
        self.answer, self.prompts = answer, []

    def chat(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}
        if len(self.prompts) == 1:
            return {"content": "", "tool_calls": [{"id": "read-overview", "type": "function", "function": {
                "name": "knowledge_read", "arguments": json.dumps({"topic": "overview", "scope": "global"})}}]}, usage
        return {"content": self.answer}, usage


def test_a_light_operation_writes_neither_the_overview_nor_its_neighbour(tmp_path, fit):
    before = _mind_writes_the_overview(tmp_path)
    answer = json.dumps({"knowledge_entries": [_OVERVIEW_EDIT, _NEIGHBOUR], "compressed_block": "compressed"})
    llm = ReadsTheOverviewFirst(answer)
    reads = c.KnowledgeReadContext(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="light-operation"))

    content, _usage = c._call_consolidation_llm(llm, "Summarize this episode for my working memory.",
                                                "Light memory operation", knowledge=reads)

    prompt = _normalized(llm.prompts[0])
    assert "knowledge_entries" not in prompt and "nominat" not in prompt
    assert len(llm.prompts) == 2 and content == answer  # read whole, then answered as text
    assert _overview(tmp_path).path.read_bytes() == before
    assert not _neighbour(tmp_path).path.exists()


def test_the_mind_still_writes_and_revises_the_overview(tmp_path):
    first = _mind_writes_the_overview(tmp_path)
    assert b"The mind's own words." in first
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="mind-turn-2")
    current = store.read_knowledge_note(_overview(tmp_path))
    assert "✅" in knowledge_tools._knowledge_write(
        ctx, "overview", "# Orientation\n\nWhat changed is now true.\n", expected_revision=current.revision)
    assert "What changed is now true." in store.read_knowledge_note(_overview(tmp_path)).text
    history = [json.loads(line) for line in
               (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["writer"] for row in history if row.get("topic") == "overview"
            and row.get("publication") == "source_capture"] == ["turn", "turn"]
