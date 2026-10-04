"""Light memory operations keep their sources whole.

A reflection whose prompt cannot fit is retained first and read completely in
ranges; an answer given before the retained source was read authorizes nothing;
a scratchpad pass never drops blocks written while its Light call ran. (These
tests outlived the measured-pressure batch they used to sit beside.)
"""
import hashlib
import json

import pytest

from ouroboros import consolidator as c, reflection
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
                body = result.split("\n[Tool result source view]\n", 1)[0].split("\n", 1)[1]
                source = (self.root / self.ref["read"]["arguments"]["path"]).read_text(encoding="utf-8")
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


def test_two_megabyte_reflection_retained_before_first_fit_and_read_completely(tmp_path, fit):
    fit.window = 50000
    text = "BEGINNING original event.\n" + "Decision and counterexample matter. " * 60000 + "\nDECISIVE FINAL EVENT."
    assert len(text.encode()) > 2_000_000
    actor = SourceReader(tmp_path, fit.window, "I retain the original beginning and DECISIVE FINAL EVENT.\nMEMORY_ACTIONS_JSON: []")
    entry = reflection.generate_reflection({"id": "big-reflection", "text": text, "drive_root": str(tmp_path)},
                                           {}, "trace", actor, {"rounds": 20, "cost": 1})
    assert not entry.get("memory_operation_errors"), (entry.get("memory_operation_errors"), len(actor.calls), actor.position, actor.stage)
    assert "DECISIVE FINAL EVENT" in entry["reflection"]
    ref = entry["source_ref"]
    stored = (tmp_path / ref["read"]["arguments"]["path"]).read_bytes()
    assert ref["sha256"] == hashlib.sha256(stored).hexdigest()
    assert text in stored.decode()
    assert actor.received == [stored.decode()]
    assert len(actor.calls) > 5
    assert len(actor.calls[0]["messages"][0]["content"]) < 3000
    assert all(ca["model_role"] == "light" for ca in actor.calls)


def test_unread_initial_source_cannot_authorize_a_reflection_action(tmp_path, fit):
    fit.window = 24000
    class Unread:
        def chat(self, **_kwargs):
            return {"content": 'Reflection.\nMEMORY_ACTIONS_JSON: [{"type":"knowledge_write","topic":"new","content":"Unread claim"}]'}, {"cost": 0.03}
    entry = reflection.generate_reflection({"id": "unread", "text": "large " * 100000, "drive_root": str(tmp_path)},
                                           {}, "trace", Unread(), {"rounds": 20})
    assert entry["memory_actions"] == []
    assert entry["memory_operation_errors"][0]["kind"] == "source_incomplete"
    assert entry["memory_operation_errors"][0]["response_ref"]
    assert (tmp_path / entry["source_ref"]["read"]["arguments"]["path"]).exists()


@pytest.mark.parametrize("change_same_source", [False, True])
def test_scratchpad_pressure_cas_preserves_concurrent_sources(tmp_path, fit, change_same_source):
    memory, ctx = setup_memory(tmp_path)
    original = {"ts": "same", "source": "task", "content": "old" * 2000}
    memory.mutate_scratchpad_blocks(lambda _: [original])
    newer = {**original, "content": "new concurrent content"}
    appended = {"ts": "later", "source": "task", "content": "A new episode"}
    class Concurrent:
        def chat(self, **_kwargs):
            memory.mutate_scratchpad_blocks(lambda rows: [newer] if change_same_source else rows + [appended])
            return {"content": '{"knowledge_entries":[],"compressed_block":"Short retained understanding."}'}, {"cost": 0.01}
    c.consolidate_scratchpad(memory, tmp_path / "memory/knowledge", Concurrent(), pressure=True, knowledge_context=ctx)
    saved = memory.load_scratchpad_blocks()
    assert saved == [newer] if change_same_source else saved[1:] == [appended]
