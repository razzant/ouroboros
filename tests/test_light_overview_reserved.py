"""The overview is the acting mind's own words; a Light helper never writes it.

The overview note is "what I hold true now", loaded into every context. Both Light
operations that nominate knowledge, scratchpad consolidation and post-task
reflection, share one prompt and one writer (``_write_knowledge_entries``). The
prompt now tells Light to name a stale passage in its own text instead of
nominating the overview, and the writer refuses a nominated overview with
``overview_is_mind_authored`` while the rest of the batch lands as before. The
mind's own ``knowledge_write`` of the overview is untouched.

Every refusal is pinned together with the case that still works, and the
nominations here are ones the writer would publish without the refusal: Light
reads the whole current overview first, so its anchored edit is bound to that
exact revision.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import consolidator as c
from ouroboros import knowledge as store
from ouroboros import reflection
from ouroboros.memory import Memory
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


def test_the_shared_prompt_leaves_the_overview_to_the_acting_mind():
    prompt = _normalized(c.KNOWLEDGE_MAINTENANCE_PROMPT)
    assert "The note overview (scope global) is written by the acting mind" in prompt
    assert "name the stale passage in the text you return" in prompt
    assert "do not nominate overview" in prompt
    # The words that told Light to keep the overview current, or create it, are gone.
    assert "keep it current" not in prompt
    assert "create it after reading the index" not in prompt


def test_scratchpad_upkeep_cannot_write_the_overview_but_writes_its_neighbour(tmp_path, fit):
    before = _mind_writes_the_overview(tmp_path)
    memory = Memory(tmp_path)
    for index in range(4):
        memory.append_scratchpad_block(f"block-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")
    llm = ReadsTheOverviewFirst(json.dumps({"knowledge_entries": [_OVERVIEW_EDIT, _NEIGHBOUR],
                                            "compressed_block": "compressed"}))

    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", llm)

    assert "written by the acting mind" in _normalized(llm.prompts[0])
    assert _overview(tmp_path).path.read_bytes() == before
    assert "A durable lesson." in store.read_knowledge_note(_neighbour(tmp_path)).text
    writes = memory.load_scratchpad_blocks()[0]["metadata"]["knowledge_writes"]
    assert [(row["topic"], row["scope"], row["ok"], row["reason"]) for row in writes] == [
        ("overview", "global", False, "overview_is_mind_authored"),
        ("lessons/neighbour", "global", True, "saved")]


def test_reflection_cannot_write_the_overview_but_writes_its_neighbour(tmp_path, fit):
    before = _mind_writes_the_overview(tmp_path)
    llm = ReadsTheOverviewFirst("Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps([
        {"type": "knowledge_write", **_OVERVIEW_EDIT}, {"type": "knowledge_write", **_NEIGHBOUR}]))
    entry = reflection.generate_reflection(
        {"id": "reflection-task", "text": "The episode.", "drive_root": str(tmp_path)},
        {}, "trace", llm, {"rounds": 2, "cost": 0.1})
    assert "written by the acting mind" in _normalized(llm.prompts[0])
    overview_action = next(a for a in entry["memory_actions"] if a["topic"] == "overview")
    assert overview_action["expected_revision"] == store.read_knowledge_note(_overview(tmp_path)).revision

    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, entry["memory_actions"]) == 1

    assert _overview(tmp_path).path.read_bytes() == before
    assert "A durable lesson." in store.read_knowledge_note(_neighbour(tmp_path)).text
    history = [json.loads(line) for line in
               (tmp_path / "memory" / "knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()]
    (incomplete,) = [row for row in history if row.get("type") == "reflection_knowledge_write_incomplete"]
    assert incomplete["proposal"]["topic"] == "overview"
    assert [(row["ok"], row["reason"]) for row in incomplete["outcomes"]] == [(False, "overview_is_mind_authored")]


def test_a_call_without_a_stamp_refuses_the_overview_from_any_room(tmp_path):
    room = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, budget_drive_root=str(tmp_path),
                       project_id="demo", task_id="t1")
    outcomes = c._write_knowledge_entries(tmp_path / "projects" / "demo" / "knowledge", [
        {"topic": " overview ", "content": "# Orientation\n\nA helper's overview.\n"},
        {"topic": "overview", "scope": "project:demo", "content": "# Orientation\n\nAgain.\n"},
        {"topic": "room-notes", "content": "# Room\n\nProject detail.\n"},
    ], context=room)
    assert [(row["topic"], row["scope"], row["ok"], row["reason"]) for row in outcomes] == [
        ("overview", "global", False, "overview_is_mind_authored"),
        ("overview", "global", False, "overview_is_mind_authored"),
        ("room-notes", "project:demo", True, "saved")]
    assert not _overview(tmp_path).path.exists()
    assert not (tmp_path / "projects" / "demo" / "knowledge" / "overview.md").exists()


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
