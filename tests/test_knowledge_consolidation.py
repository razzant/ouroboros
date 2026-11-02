"""One Light memory operation reads current knowledge whole; it writes none.

The shared Light transport's read tools deliver the complete current note (or a retained
partial view of an oversized one) inside the same call. The automatic reflection, dialogue,
pressure and scratchpad writers are retired (the last by #918: the mind folds its own
scratchpad), and so are their knowledge nominations: an answer that still nominates a note
is returned as text and leaves the note as its author wrote it.
"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from ouroboros import consolidator as c, knowledge as k
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit


def _address(root, topic="people/alex"):
    return k.resolve_knowledge_address(root, topic, "global")


def _call(topic="people/alex"):
    return {"id": "read-current", "type": "function", "function": {
        "name": "knowledge_read", "arguments": json.dumps({"topic": topic, "scope": "global"}),
    }}


class MemoryLLM:
    def __init__(self, answer, *, before_answer=None):
        self.answer, self.before_answer = answer, before_answer
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if len(self.calls) == 1:
            return {"content": "", "tool_calls": [_call()]}, {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}
        if self.before_answer:
            self.before_answer()
        return {"content": self.answer}, {
            "prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.02}


def _initial(root):
    return k.write_knowledge_note(_address(root),
        "---\ntype: understanding\nsummary: Brevity may depend on context.\ncustom: retained\n---\n"
        "# Alex\n\nHe asked for brevity while hurried.\nDECISIVE ORIGINAL TAIL.\n").current


def test_same_light_operation_reads_the_complete_current_note(tmp_path, fit, monkeypatch):
    original = _initial(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="memory-operation")
    reads = c.KnowledgeReadContext(ctx)
    monkeypatch.setattr(c, "_consolidation_route", lambda: ("claudexor::codex=test", False))
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"light":"kept-account"}')
    llm = MemoryLLM("Completed interpretation.")
    content, usage = c._call_consolidation_llm(llm, "ORIGINAL EPISODE: today he asked for depth.",
                                              "test memory", knowledge=reads)
    assert content == "Completed interpretation."
    assert usage["cost"] == pytest.approx(0.03)
    assert usage["prompt_tokens"] == 30 and usage["completion_tokens"] == 15
    assert len(llm.calls) == 2
    assert llm.calls[0]["model_turn_state"] is llm.calls[1]["model_turn_state"]
    assert all(call["model_account_override"] == "kept-account" for call in llm.calls)
    assert all(call["model_role"] == "light" for call in llm.calls)
    second = llm.calls[1]["messages"]
    assert "ORIGINAL EPISODE" in second[0]["content"]
    assert second[-1]["content"].endswith(original.text)
    assert "DECISIVE ORIGINAL TAIL." in second[-1]["content"]
    assert k.read_knowledge_note(original.address).raw == original.raw


_EPISODES = "\n\n".join(f"ORIGINAL EPISODE {index}: asked for depth. " + "x" * 11000 for index in range(3))


@pytest.mark.parametrize("concurrent", [False, True])
def test_a_light_source_stays_durable_and_a_nominated_revision_writes_nothing(tmp_path, fit, concurrent):
    """Formerly carried by the post-task scratchpad pass (retired, #918): the same operation shape
    through the shared transport, whose answer nominates an edit while an authored revision races it."""
    original = _initial(tmp_path)
    update = "Two original episodes, context differs."
    edits = [{"old_text": "He asked for brevity while hurried.", "new_text": update,
              "basis": "The original episodes establish context-specific requests."}]
    answer = json.dumps({"knowledge_entries": [{"topic": "people/alex", "edits": edits,
                                                "summary": "Brevity while hurried; depth while exploring."}],
                         "compressed_block": "I learned that context matters; keep both episodes."})
    competing = lambda: k.write_knowledge_note(original.address, "A simultaneous newer observation.",
                                               expected_revision=original.revision)
    llm = MemoryLLM(answer, before_answer=competing if concurrent else None)
    reads = c.KnowledgeReadContext(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="light-operation"))
    content, usage = c._call_consolidation_llm(llm, _EPISODES, "Light memory operation", knowledge=reads)
    assert usage["cost"] == pytest.approx(0.03)
    assert content == answer  # returned as text: nothing parses or applies the nomination
    assert "ORIGINAL EPISODE 0" in llm.calls[1]["messages"][0]["content"]
    assert "DECISIVE ORIGINAL TAIL." in llm.calls[1]["messages"][-1]["content"]
    # The nominated edit and summary never land; a concurrent authored revision stays.
    current = k.read_knowledge_note(original.address)
    if concurrent:
        assert current.text.endswith("A simultaneous newer observation.")
    else:
        assert current.raw == original.raw
    assert update not in current.text


def test_an_oversized_requested_note_is_retained_and_delivered_as_a_partial_view(tmp_path, fit):
    from ouroboros.artifacts import read_actor_source_bytes

    original = _initial(tmp_path)
    current = k.write_knowledge_note(original.address, "Large source. " * 10000, expected_revision=original.revision).current
    fit.window = 24000
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    llm = MemoryLLM("The delivered source is partial; do not rewrite it yet.")
    reads = c.KnowledgeReadContext(ctx)
    content, usage = c._call_consolidation_llm(llm, "Original episode.", "test memory",
                                              knowledge=reads)
    assert content and len(llm.calls) == 2
    shown = llm.calls[-1]["messages"][-1]["content"]
    notice = json.loads(shown.split("\n[Tool result source view]\n", 1)[1])
    assert read_actor_source_bytes(tmp_path, "consolidation", notice["source_ref"]).decode().endswith(current.text)
    assert notice["delivered_range"][1] < notice["complete_chars"]
    assert k.read_knowledge_note(current.address).raw == current.raw
    assert usage["cost"] == pytest.approx(0.03)


