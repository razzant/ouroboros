"""Light memory operations keep their sources whole.

A Light operation whose prompt cannot fit is retained first and read completely in
ranges; an answer given before the retained source was read is refused with its
response kept. (These tests outlived the measured-pressure batch they used to sit
beside, the post-task reflection they were first written for, and the post-task
scratchpad pass that last carried them: #918 retired it. Its check that blocks written
while its Light call ran are never dropped lives on at the mind's fold:
``test_scratchpad_fold.py::test_a_concurrent_change_or_append_is_never_lost``.)
"""
import hashlib
import json

from ouroboros import consolidator as c
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.memory import Memory
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit


def call(name, args, ident):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class SourceReader:
    """Fake actor advances through real read_file results and requests authored views."""
    def __init__(self, root, window, answer=None):
        self.root, self.window, self.answer = root, window, answer
        self.calls, self.sources, self.received, self.operation_turns = [], [], [], []
        self.ref, self.position, self.stage = None, 0, ""

    def finish(self, prompt):
        if self.answer:
            return self.answer
        return "I remember the complete episode including its final unresolved decision."

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        assert estimate_context_prompt_tokens(kwargs["messages"], kwargs["tools"]) + kwargs["max_tokens"] <= self.window
        assert {t["function"]["name"] for t in kwargs["tools"]} >= {"read_file", "compact_context", "knowledge_read"}
        first = kwargs["messages"][0]["content"]
        if len(kwargs["messages"]) == 1:
            self.operation_turns.append(kwargs["model_turn_state"])
            if not first.startswith("Complete source and instructions"):
                return {"content": self.finish(first)}, {"cost": 0.01}
            self.ref = json.loads(first.split("\n", 1)[1])
            self.position, self.stage = 0, "read"
            self.sources.append(self.ref)
            self.received.append("")
            assert (self.root / self.ref["read"]["arguments"]["path"]).read_bytes()
        else:
            assert kwargs["model_turn_state"] is self.operation_turns[-1]
            if self.stage == "read":
                result = kwargs["messages"][-1]["content"]
                source = (self.root / self.ref["read"]["arguments"]["path"]).read_text(encoding="utf-8")
                from ouroboros.tools.core_file_tools import _render_line_slice, delivered_source_prefix

                view = {}
                _render_line_slice("source", source, start_char=self.position, extent=view)
                view["body_start"] = result.find("\n") + 1
                visible = result.split("\n[Tool result source view]\n", 1)[0]
                body = delivered_source_prefix(view, visible, len(visible))
                assert body == source[self.position:self.position + len(body)]
                assert body
                self.position += len(body)
                self.received[-1] += body
                if self.position == len(source):
                    return {"content": self.finish(self.received[-1])}, {"cost": 0.01}
                self.stage = "compact"
                return {"tool_calls": [call("compact_context", {
                    "working_note": f"I read through character {self.position}. The beginning remains part of the chronology; continue the original source and preserve its final unresolved decision.",
                    "keep_unit_ids": []}, f"compact-{len(self.calls)}")]}, {"cost": 0.01}
            receipt = json.loads(kwargs["messages"][-1]["content"])
            assert receipt["context_view"]["status"] == "applied"
            self.stage = "read"
        return {"tool_calls": [call("read_file", {
            **self.ref["read"]["arguments"], "max_lines": 2000, "start_char": self.position},
            f"read-{len(self.calls)}")]}, {"cost": 0.01}


def setup_memory(root):
    memory = Memory(root, root)
    memory.identity_path().parent.mkdir(parents=True, exist_ok=True)
    memory.identity_path().write_text("# Identity\nThe same unmodified identity.\n")
    return memory, ToolContext(repo_dir=root, drive_root=root, task_id="source-task")


def _oversized_source():
    text = "BEGINNING original event.\n" + "Decision and counterexample matter. " * 30000
    return text + "\n" + text + "\nDECISIVE FINAL EVENT.\nA later episode."


def test_a_two_megabyte_source_is_retained_before_first_fit_and_read_completely(tmp_path, fit):
    fit.window = 50000
    _memory, ctx = setup_memory(tmp_path)
    source = _oversized_source()
    assert len(source.encode()) > 2_000_000
    actor = SourceReader(tmp_path, fit.window, "I retain the original beginning and DECISIVE FINAL EVENT.")
    content, usage = c._call_consolidation_llm(actor, source, "Light source retention",
                                              knowledge=c.KnowledgeReadContext(ctx))
    assert not usage.get("_consolidation_errors"), (usage.get("_consolidation_errors"), len(actor.calls), actor.position)
    [ref] = actor.sources
    stored = (tmp_path / ref["read"]["arguments"]["path"]).read_bytes()
    assert ref["sha256"] == hashlib.sha256(stored).hexdigest()
    assert stored.decode() == source
    assert actor.received == [stored.decode()]
    assert len(actor.calls) > 5
    assert len(actor.calls[0]["messages"][0]["content"]) < 3000
    assert all(ca["model_role"] == "light" for ca in actor.calls)
    assert content == "I retain the original beginning and DECISIVE FINAL EVENT."


def test_an_answer_before_the_retained_source_was_read_is_refused(tmp_path, fit):
    fit.window = 24000
    _memory, ctx = setup_memory(tmp_path)

    class Unread:
        def chat(self, **_kwargs):
            return {"content": '{"compressed_block": "Unread claim"}'}, {"cost": 0.03}

    content, usage = c._call_consolidation_llm(Unread(), _oversized_source(), "Light source retention",
                                              knowledge=c.KnowledgeReadContext(ctx))
    [error] = usage["_consolidation_errors"]
    assert content == "" and error["kind"] == "source_incomplete" and error["response_ref"]
    assert (tmp_path / error["source_ref"]["read"]["arguments"]["path"]).exists()
