"""Existing Light memory operations relieve measured pressure without losing sources."""
import hashlib
import json

import pytest

from ouroboros import consolidator as c, reflection, knowledge as k
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
        if prompt.startswith("Compare this draft memory"):
            return "I retain the beginning, middle and last event, checked against the complete source."
        if "scratchpad working memory has" in prompt:
            return json.dumps({"knowledge_entries": [], "compressed_block": "I retain the beginning, middle and last event, including unresolved questions."})
        if prompt.startswith("Compress these older memory blocks"):
            return "### Era\nI retain the beginning, middle and last event across the complete dated span."
        return "### Block\nI remember the complete episode including its final unresolved decision."

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
    return memory, ToolContext(repo_dir=root, drive_root=root, task_id="pressure-task")


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


@pytest.mark.parametrize("legacy_gap", [False, True])
def test_pressure_reduces_whole_chronicle_and_one_huge_block_before_normal_send(tmp_path, fit, legacy_gap):
    fit.window = 50000
    memory, ctx = setup_memory(tmp_path)
    identity_before = memory.identity_path().read_bytes()
    blocks = [{"ts": "2024-01-01", "range": "2024-01-01", "type": "summary", "message_count": 100,
               "content": "BEGINNING. " + "History before gap. " * 9000},
              {"gap_id": "known-gap", "content": "[MEMORY GAP] An authentic discontinuity", "range": "2025-01-01"},
              {"ts": "2026-01-01", "range": "2026-01-01", "type": "summary", "message_count": 100,
               "content": "Middle. " + "History after gap. " * 9000 + "LAST EVENT."}]
    if legacy_gap:
        blocks[1].pop("gap_id")
    path = tmp_path / "memory/dialogue_blocks.json"
    c.atomic_write_json(path, blocks)
    scratch = {"ts": "2026-09-01", "source": "task", "content": "Active complete source. " * 15000 + "FINAL QUESTION."}
    memory.mutate_scratchpad_blocks(lambda _current: [scratch])
    def fits():
        messages = [{"role": "system", "content": memory.identity_path().read_text(encoding="utf-8") + path.read_text(encoding="utf-8") + memory.scratchpad_path().read_text(encoding="utf-8")}]
        return estimate_context_prompt_tokens(messages) < 2000
    assert not fits() and not c.should_consolidate_scratchpad(memory)
    actor = SourceReader(tmp_path, fit.window)
    result = c.maintain_memory_pressure(memory, actor, ctx, fits=fits, current_topic="CURRENT GOAL: resolve the outstanding research question.")
    assert result["status"] == "fitting", result
    assert fits() and memory.identity_path().read_bytes() == identity_before
    assert result["changed_sources"]
    assert result["usage"]["cost"] == pytest.approx(len(actor.calls) * 0.01)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[1] == blocks[1] and len(saved) == 3
    for original, compressed in ((blocks[0], saved[0]), (blocks[2], saved[2])):
        ref = compressed["source_ref"]
        assert json.loads((tmp_path / ref["read"]["arguments"]["path"]).read_text(encoding="utf-8")) == [original]
    journal = [json.loads(line) for line in memory.journal_path().read_text(encoding="utf-8").splitlines()]
    assert next(row for row in journal if row["type"] == "blocks_consolidated")["source_blocks"] == [scratch]
    # Two contiguous runs: each is compressed and then corrected against its complete
    # sections through the retained-source route, plus the scratchpad source.
    assert len(actor.sources) == 5 and all(actor.received)
    assert all("CURRENT GOAL: resolve the outstanding research question." in source for source in actor.received)
    # The caller can now construct its normal first request; maintenance has
    # not changed the identity or truncated any original source to achieve fit.
    assert estimate_context_prompt_tokens([{"role": "system", "content": path.read_text(encoding="utf-8") + memory.load_scratchpad()}]) < 2000


def test_force_tail_is_explicit_and_advances_a_huge_short_dialogue_once(tmp_path, fit):
    memory, ctx = setup_memory(tmp_path)
    chat = tmp_path / "logs/chat.jsonl"
    chat.parent.mkdir()
    rows = [{"text": "Huge single message. " * 10000, "chat_id": 1, "ts": "2026-09-13T01:00:00Z"}]
    chat.write_text(json.dumps(rows[0]) + "\n")
    blocks, meta = tmp_path / "memory/dialogue_blocks.json", tmp_path / "memory/dialogue_meta.json"
    assert not c.should_consolidate(meta, chat)
    actor = SourceReader(tmp_path, fit.window)
    assert c.consolidate(chat, blocks, meta, actor, knowledge_context=ctx) is None
    assert not actor.calls
    before = chat.read_bytes()
    result = c.maintain_memory_pressure(memory, actor, ctx,
        fits=lambda: meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("last_consolidated_offset") == 1)
    assert result["status"] == "fitting"
    assert chat.read_bytes() == before
    assert len(actor.calls) == 2  # one draft and its correction; the tail now fits, so no era call
    assert sum(row["message_count"] for row in json.loads(blocks.read_text(encoding="utf-8"))) == 1
    assert not c.should_consolidate(meta, chat)


def test_pressure_uses_read_revision_to_rewrite_the_authored_overview(tmp_path, fit):
    memory, ctx = setup_memory(tmp_path)
    address = k.resolve_knowledge_address(tmp_path, "overview", "global")
    old = k.write_knowledge_note(address, "---\nsummary: Full authored orientation.\n---\n" + "Detailed understanding. " * 500).current
    class Overview:
        calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"tool_calls": [call("knowledge_read", {"topic": "overview", "scope": "global"}, "read-overview")]}, {"cost": 0.01}
            assert old.text in kwargs["messages"][-1]["content"]
            # Pressure shortening is an explicit edit of the whole long span it replaces.
            return {"content": json.dumps({"knowledge_entries": [{"topic": "overview", "scope": "global",
                "summary": "Authored compact orientation.",
                "edits": [{"old_text": "Detailed understanding. " * 500,
                           "new_text": "Full scope retained with [detail](detail.md).",
                           "basis": "The complete current source is retained in the detailed note."}]}]})}, {"cost": 0.02}
    result = c.maintain_memory_pressure(memory, Overview(), ctx, fits=lambda: len(address.path.read_bytes()) < 1000)
    assert result["status"] == "fitting"
    assert result["actions"][0]["writes"][0]["ok"]
    current = k.read_knowledge_note(address)
    assert current.metadata == {"summary": "Authored compact orientation.", "type": "note"}
    assert current.text.endswith("---\nFull scope retained with [detail](detail.md).")
    history = [json.loads(line) for line in (tmp_path / "memory/knowledge_history.jsonl").read_text(encoding="utf-8").splitlines()]
    change = next(row for row in history if row.get("old_content") == old.text)
    assert change["writer"] == "knowledge_maintenance" and change["edits"][0]["basis"]
    assert change["summary"] == "Authored compact orientation."


def test_irreducible_identity_is_preserved_with_no_progress(tmp_path, fit):
    memory, ctx = setup_memory(tmp_path)
    raw = b"identity " * 20000
    memory.identity_path().write_bytes(raw)
    class NoCall:
        def chat(self, **_kwargs):
            raise AssertionError("No existing mutable memory source can relieve this core")
    result = c.maintain_memory_pressure(memory, NoCall(), ctx, fits=lambda: False)
    assert result["status"] == "no_progress" and not result["changed_sources"]
    assert memory.identity_path().read_bytes() == raw
    assert not result["actions"]


def test_failed_scratchpad_output_keeps_paid_usage_and_source(tmp_path, fit):
    memory, ctx = setup_memory(tmp_path)
    original = {"ts": "same", "source": "task", "content": "old" * 2000}
    memory.mutate_scratchpad_blocks(lambda _: [original])
    class Invalid:
        calls = 0
        def chat(self, **_kwargs):
            self.calls += 1
            return {"content": "This is not the requested JSON."}, {"cost": 0.25}
    actor = Invalid()
    result = c.maintain_memory_pressure(memory, actor, ctx, fits=lambda: False)
    assert actor.calls == 1 and result["usage"]["cost"] == 0.25
    assert result["usage"]["_consolidation_errors"][0]["kind"] == "scratchpad_consolidation_failed"
    assert memory.load_scratchpad_blocks() == [original]
    assert result["status"] == "no_progress"


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
