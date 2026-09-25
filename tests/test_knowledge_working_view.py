"""The existing Light operation reads complete notes across authored working views."""

import json
from copy import deepcopy

import pytest

from ouroboros import consolidator as c, knowledge as k
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

fit = fit_helpers.fit


def _call(name, args, ident="call"):
    return {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _setup(tmp_path, text):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="memory-view")
    address = k.resolve_knowledge_address(tmp_path, "large", "global")
    note = k.write_knowledge_note(address, text).current
    return ctx, note, c.KnowledgeReadContext(ctx)


def _read(reads, start, end, ident="read"):
    return reads.read_call(_call("knowledge_read", {"topic": "large", "scope": "global",
                                                    "start_char": start, "end_char": end}, ident))


def test_only_gap_free_delivered_union_of_one_revision_can_bind_replacement(tmp_path):
    ctx, original, reads = _setup(tmp_path, "x" * 100)
    total = len(original.text)
    rows = [_read(reads, 0, 40), _read(reads, 50, total)]
    assert not reads.reads  # producer reads are not model delivery
    reads.pending_delivery = rows
    reads.accept_delivery()
    reads.pending_delivery = [_read(reads, 0, 40)]
    reads.accept_delivery()
    assert not reads.reads  # repeat does not fill the 40:50 gap
    reads.pending_delivery = [_read(reads, 40, 50)]
    reads.accept_delivery()
    assert reads.reads == {("global", "large"): original.revision}
    newer = k.write_knowledge_note(original.address, "y" * 100, expected_revision=original.revision).current
    reads.pending_delivery = [_read(reads, 0, 40)]
    reads.accept_delivery()
    assert not reads.reads
    reads.pending_delivery = [_read(reads, 40, len(newer.text))]
    reads.accept_delivery()
    assert reads.reads == {("global", "large"): newer.revision}


def test_tool_result_projection_never_credits_undelivered_body_or_header(tmp_path):
    ctx, original, reads = _setup(tmp_path, "x" * 100)
    row = _read(reads, 0, len(original.text))
    header = row["result_meta"]["knowledge_body_start"]
    row.update(result_partial=True, result_source_view={"delivered_range": [0, header - 1]})
    reads.pending_delivery = [row]
    reads.accept_delivery()
    assert not reads.read_ranges
    row["result_source_view"]["delivered_range"][1] = header + 20
    reads.pending_delivery = [row]
    reads.accept_delivery()
    assert reads.read_ranges[("global", "large", original.revision)][1] == [(0, 20)]
    assert not reads.reads


@pytest.mark.parametrize("room_correction", [False, True])
def test_light_reads_large_note_in_multiple_windows_then_publishes_its_revision(tmp_path, fit, room_correction):
    fit.window = 50000
    ctx, original, reads = _setup(tmp_path, "Original account of events. " * 7000 + "DECISIVE LAST EVENT.")
    chunk = 40000
    total = len(original.text)
    assert estimate_context_prompt_tokens([{"role": "user", "content": original.text}], reads.tools) + 16384 > fit.window
    episode = "ORIGINAL EPISODE: reconsider the complete account."
    revised = "A coherent revised account preserving the decisive last event."
    nomination = [{"topic": "large", "scope": "global", "edits": [{
        "old_text": "DECISIVE LAST EVENT.", "new_text": revised,
        "basis": "The complete source and new episode correct the final event."}]}]
    answer = "I read the full account and retained what changed."
    if room_correction:
        answer += "\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(nomination)

    class Reader:
        def __init__(self):
            self.calls, self.next_start, self.stage, self.revisions = [], 0, "read", []

        def chat(self, **kwargs):
            self.calls.append(deepcopy(kwargs))
            assert estimate_context_prompt_tokens(kwargs["messages"], kwargs["tools"]) + kwargs["max_tokens"] <= fit.window
            assert episode in kwargs["messages"][0]["content"]
            if room_correction and not kwargs["messages"][0]["content"].startswith("Compare this draft memory"):
                # No draft read credit: the correction must earn its own, across views.
                return {"content": "Draft.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps(nomination)}, {"cost": 0.01}
            if self.stage == "read":
                if self.next_start >= total:
                    assert "DECISIVE LAST EVENT." in kwargs["messages"][-1]["content"]
                    return {"content": answer}, {"cost": 0.01}
                start, end = self.next_start, min(total, self.next_start + chunk)
                self.next_start, self.stage = end, "inspect"
                call = _call("knowledge_read", {"topic": "large", "scope": "global", "start_char": start, "end_char": end}, f"read-{start}")
            elif self.stage == "inspect":
                if self.next_start >= total:
                    assert "DECISIVE LAST EVENT." in kwargs["messages"][-1]["content"]
                    return {"content": answer}, {"cost": 0.01}
                call, self.stage = _call("compact_context", {"inspect": True}, f"inspect-{self.next_start}"), "compact"
            else:
                observed = json.loads(kwargs["messages"][-1]["content"])
                self.revisions.append(observed["view_revision"])
                call = _call("compact_context", {"expected_view_revision": observed["view_revision"],
                    "working_note": f"I have read source characters 0 through {self.next_start}; retain the event chronology and continue the exact same revision.",
                    "keep_unit_ids": []}, f"compact-{self.next_start}")
                self.stage = "read"
            return {"content": "", "tool_calls": [call]}, {"cost": 0.01}

    llm = Reader()
    if room_correction:
        from ouroboros import room_consolidation as rc
        room = rc.RoomSource("1", "Main", [{"text": episode}], episode)
        content, usage = rc.summarize_block(c._light_call(llm, ctx, {}), [room],
            first_ts="2026-09-01T10:00:00Z", last_ts="2026-09-01T10:01:00Z",
            knowledge_instruction=c.KNOWLEDGE_MAINTENANCE_PROMPT)
        bound = usage["_knowledge_entries"]
    else:
        content, usage = c._call_consolidation_llm(llm, episode, "multiwindow", knowledge=reads)
        bound = reads.bind_entries(nomination)
    assert content, usage
    assert len(llm.revisions) > 2
    assert bound[0]["expected_revision"] == original.revision
    assert c._write_knowledge_entries(original.address.shelf, bound, context=ctx)[0]["ok"]
    assert k.read_knowledge_note(original.address).text.endswith("decisive last event.")
    assert list((tmp_path / "task_results" / "artifacts" / "memory-view" / "source_handles").rglob("*.json"))


def test_first_full_pressure_read_retains_source_when_inspection_cannot_fit(tmp_path, fit):
    """Joint-fit follow-up: a greedily filled view may have no room for inspect."""
    from ouroboros.artifacts import read_actor_source_bytes

    fit.window = 24000
    ctx, original, reads = _setup(tmp_path, "Large complete experience. " * 10000)

    class FullThenInspect:
        calls = 0
        source = None

        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"tool_calls": [_call("knowledge_read", {"topic": "large", "scope": "global"})]}, {"cost": 0.01}
            if self.calls == 2:
                partial = kwargs["messages"][-1]["content"]
                self.source = json.loads(partial.split("\n[Tool result source view]\n", 1)[1])["source_ref"]
                return {"tool_calls": [_call("compact_context", {"inspect": True}, "inspect-full")]}, {"cost": 0.01}
            raise AssertionError("No complete fitting next request was established")

    llm = FullThenInspect()
    content, usage = c._call_consolidation_llm(llm, "Original episode.", "full pressure", knowledge=reads)
    assert not content and llm.calls == 2
    assert usage["_consolidation_errors"][-1]["kind"] == "knowledge_source_unfit"
    assert "source locators" in usage["_consolidation_errors"][-1]["message"]
    assert read_actor_source_bytes(tmp_path, "memory-view", llm.source).decode().endswith(original.text)
    assert not reads.reads
    assert k.read_knowledge_note(original.address).raw == original.raw


def test_full_pressure_read_can_compact_directly_from_its_causal_send(tmp_path, fit):
    from ouroboros.artifacts import read_actor_source_bytes

    fit.window = 24000
    ctx, original, reads = _setup(tmp_path, "Large complete experience. " * 10000)
    class FullThenCompact:
        calls = 0
        source = None
        def chat(self, **kwargs):
            self.calls += 1
            assert estimate_context_prompt_tokens(kwargs["messages"], kwargs["tools"]) + kwargs["max_tokens"] <= fit.window
            if self.calls == 1:
                return {"tool_calls": [_call("knowledge_read", {"topic": "large", "scope": "global"})]}, {"cost": 0.01}
            if self.calls == 2:
                partial = kwargs["messages"][-1]["content"]
                self.source = json.loads(partial.split("\n[Tool result source view]\n", 1)[1])["source_ref"]
                return {"tool_calls": [_call("compact_context", {
                    "working_note": "I have only read a partial source. Keep its exact source reference and continue reading ranges.",
                    "keep_unit_ids": [],
                }, "compact-direct")]}, {"cost": 0.01}
            assert any("partial source" in str(m.get("content")) for m in kwargs["messages"])
            return {"content": "The working view is usable; the full note remains unread and unchanged."}, {"cost": 0.01}
    llm = FullThenCompact()
    content, usage = c._call_consolidation_llm(llm, "Original episode.", "causal compact", knowledge=reads)
    assert content and llm.calls == 3, usage
    assert not reads.reads
    assert k.read_knowledge_note(original.address).raw == original.raw
    assert read_actor_source_bytes(tmp_path, "memory-view", llm.source).decode().endswith(original.text)
