"""Nothing in memory maintenance leaves silently (TZ-3 PR-1, invariant I4).

Every host decision that used to vanish is a typed fact, each proven in both
directions: a consolidation skipped on the lock, an era withheld because it was
not shorter (with its source-bound receipt), a scratchpad pass and its outcome,
a reflection lesson the host declined, and the ``writer``/``route``/
``writer_input_ref``/``old_chars``/``new_chars`` stamp on every
``source_capture`` history row. Published digests may be coarsened again while exact sources and gaps remain. The reader-less ``knowledge_journal.jsonl`` writer is gone.
"""

from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import context_health
from ouroboros import knowledge as store
from ouroboros import reflection
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json
from tests import test_consolidator_context_fit as fit_helpers
from tests.test_consolidator_context_fit import _LLM, _paths, _write_chat

fit = fit_helpers.fit


def _events(root, kind):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == kind]


def _history(root):
    path = root / "memory" / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_blocks(count, start=0):
    return [{"ts": "2026-01-01T00:00:00Z", "type": "summary", "range": f"2026-01-01 {i:02d}:00 - {i:02d}:59",
             "message_count": 1, "content": f"block-{i} " + "x" * 40} for i in range(start, start + count)]


# --- the consolidation lock ---------------------------------------------------------


def test_a_lock_skip_is_a_typed_event_and_a_free_run_is_not(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    _write_chat(chat, text_size=0)
    meta.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(str(meta.parent / ".consolidation.lock"), os.O_CREAT | os.O_WRONLY, 0o644)
    c._lock_nb(holder)
    try:
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="held")
        llm = _LLM()
        outcome = c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx, represented_only=True)
        assert outcome["_blocks_written"] == 0 and not llm.calls
        assert [(row["kind"], row["reason"]) for row in outcome["_consolidation_errors"]] == [
            ("temporarily_unavailable", "consolidation_lock_held")]
    finally:
        c._unlock(holder)
        os.close(holder)
    skipped = _events(tmp_path, "consolidation_skipped_locked")
    assert len(skipped) == 1 and skipped[0]["task_id"] == "held"
    assert skipped[0]["lock_path"].endswith(".consolidation.lock")
    assert not blocks.exists()  # the holder owned the run; nothing was consolidated twice

    assert c.consolidate(chat, blocks, meta, _LLM(), completed_task={"id": "fixture"})["_blocks_written"] == 1
    assert len(_events(tmp_path, "consolidation_skipped_locked")) == 1


# --- room digests retain sources, gaps and truthful retry evidence -------------------


def _digest_store(root, rooms=("a",)):
    from ouroboros.chronicle_store import ChronicleStore
    chronicle = ChronicleStore(root)
    chronicle.import_legacy()
    original = [chronicle.append_episode(room, f"Room {room} history. " * 50, [], {"kind": "mind"}) for room in rooms]
    return chronicle, original


def _digest_run(root, llm, *, fits=lambda: False, fitting_demand=None):
    chat, blocks, meta = _paths(root)
    return c.consolidate(chat, blocks, meta, llm, knowledge_context=ToolContext(
        repo_dir=root, drive_root=root, task_id="digest-maintenance"), compact_chronicle=True, pressure_fits=fits, fitting_demand=fitting_demand)


def _long_digest():
    return _LLM(effect=lambda llm, _prompt: ({"content": "Longer than the source. " * 1000}, dict(llm.usage)))


def test_existing_digest_can_be_coarsened_without_changing_its_original_sources(tmp_path, fit):
    chronicle, (original,) = _digest_store(tmp_path)
    prior = chronicle.append_episode("a", "A prior digest. " * 20, [], {"kind": "helper"}, kind="digest",
        metadata={"covers_record_ids": [original["id"]]})
    llm = _LLM()
    _digest_run(tmp_path, llm)
    _digest_run(tmp_path, llm, fitting_demand={"memory_budget_tokens": 100})
    assert not llm.calls  # a changed leftover allowance is not a new semantic request
    _digest_run(tmp_path, llm, fitting_demand={"purpose": "explicit_revision", "memory_budget_tokens": 100})
    assert not llm.calls  # A new purpose label alone does not authorize a rebuy either.
    _digest_run(tmp_path, llm, fitting_demand={"purpose": "owner_mode", "rendered_mode": "nano",
                                             "requirement_tokens": 85000, "memory_budget_tokens": 100})
    latest = chronicle.records(kinds=["digest"])[-1]
    assert len(llm.calls) == 1 and latest["metadata"]["covers_record_ids"] == [original["id"]]
    request = str(llm.calls[0]["messages"])
    assert original["text"] in request and prior["text"] not in request
    assert chronicle.get(original["id"])["text"] == original["text"]
    assert chronicle.get(prior["id"])["text"] == prior["text"]
    assert latest["source_refs"] and latest["author"]["kind"] == "helper"
    assert len(latest["text"]) < len(prior["text"])


def test_fitting_projection_never_buys_an_automatic_digest(tmp_path, fit):
    chronicle, originals = _digest_store(tmp_path, ("a", "b"))
    llm = _LLM()
    _digest_run(tmp_path, llm, fits=lambda: True)
    assert llm.calls == [] and chronicle.records(kinds=["digest"]) == []
    assert [chronicle.get(row["id"])["text"] for row in originals] == [row["text"] for row in originals]


def test_imported_gap_remains_visible_after_digest_even_if_helper_omits_it(tmp_path, fit):
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.chronicle_view import capture_chronicle, render_memory
    memory = tmp_path / "memory"
    memory.mkdir()
    original = [*_summary_blocks(3), {"type": "gap", "gap_id": "durable-hole", "content": "[MEMORY GAP]"}]
    path = memory / "dialogue_blocks.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    before = path.read_bytes()
    chronicle = ChronicleStore(tmp_path)
    chronicle.import_legacy()
    _digest_run(tmp_path, _LLM())
    text, _facts = render_memory(json.loads(capture_chronicle(Memory(tmp_path), {"id": "view", "chat_id": 1})))
    assert "durable-hole" in text and "gap" in text.lower()
    assert path.read_bytes() == before
    gaps, _identities = Memory(tmp_path)._durable_dialogue_gaps()
    assert any(row["gap_id"] == "durable-hole" for row in gaps)


def test_not_shorter_digest_is_recorded_and_same_inputs_are_not_rebought(tmp_path, fit):
    chronicle, (original,) = _digest_store(tmp_path)
    first = _long_digest()
    assert _digest_run(tmp_path, first)["cost"] == 0.01
    assert len(first.calls) == 1 and chronicle.records(kinds=["digest"]) == []
    (receipt,) = [r for r in chronicle.records(kinds=["maintenance"]) if r.get("status") == "not_shorter"]
    assert receipt["source_keys"] == [[original["id"], original["id"]]]
    second = _long_digest()
    assert _digest_run(tmp_path, second)["cost"] == 0
    assert second.calls == []
    assert chronicle.get(original["id"])["text"] == original["text"]


def test_each_room_keeps_its_own_refusal_when_another_room_changes_and_succeeds(tmp_path, fit):
    chronicle, originals = _digest_store(tmp_path, ("a", "b"))
    first = _long_digest()
    _digest_run(tmp_path, first)
    maintenance = chronicle.records(kinds=["maintenance"])
    receipts = [row for row in maintenance if row.get("status") == "not_shorter"]
    assert len(receipts) == 2 and {row["room_id"] for row in receipts} == {"a", "b"}
    measured = [row for row in maintenance if row["id"].startswith("digest-fit:")]
    assert len(measured) == 2 and {row["target_id"] for row in measured} == {row["id"] for row in receipts}
    assert all(row["published_progress"] is False and row["target_fits"] is False for row in measured)
    chronicle.revise(originals[0]["id"], originals[0]["text"] + "New cause.", {"kind": "mind"})
    next_model = _LLM()
    _digest_run(tmp_path, next_model)
    assert len(next_model.calls) == 1
    assert [row["room_id"] for row in chronicle.records(kinds=["digest"])] == ["a"]
    assert all(chronicle.get(row["id"]) for row in receipts)  # history was not erased by unrelated success
    unchanged = _long_digest()
    _digest_run(tmp_path, unchanged)
    assert all("Room b history" not in call["messages"][0]["content"] for call in unchanged.calls)


def test_digest_retry_uses_effective_account_and_model_binding(tmp_path, fit, monkeypatch):
    from contextlib import nullcontext
    from ouroboros import model_slots, model_wait
    chronicle, (original,) = _digest_store(tmp_path)
    calls = []
    def run(demand=None):
        llm = _long_digest()
        usage = _digest_run(tmp_path, llm, fitting_demand=demand)
        calls.extend(llm.calls)
        return usage
    run()
    run()
    assert len(calls) == 1
    monkeypatch.setattr(model_slots, "model_role_option", lambda key, role, **_kw: "acct-B" if role == "light" else "")
    run()
    assert len(calls) == 1  # moving completed source work to another account does not buy it again
    demand = {"purpose": "owner_mode", "requirement_tokens": 250000, "rendered_mode": "low"}
    usage = run(demand)
    run({**demand, "requirement_tokens": 250009})
    assert len(calls) == 2 and calls[-1]["model_account_override"] == "acct-B"
    assert usage["_light_dispatch_binding"]["model_account_override"] == "acct-B"
    override = {"model": "override/model", "use_local": False, "model_account_override": "acct-B"}
    waiter = SimpleNamespace(overrides={"light": override}, register_reprepare=lambda *_: nullcontext())
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    run(demand)
    assert len(calls) == 2  # a model change alone is not another source requirement either
    revision = chronicle.revise(original["id"], original["text"] + "New source meaning.", {"kind": "mind"})
    usage = run(demand)
    run(demand)
    assert len(calls) == 3 and {key: calls[-1][key] for key in override} == override
    assert usage["_light_dispatch_binding"] == override and "New source meaning." in str(calls[-1]["messages"])
    receipts = [row for row in chronicle.records(kinds=["maintenance"]) if row.get("status") == "not_shorter"]
    assert len(receipts) == 3 and receipts[-1]["source_keys"] == [[original["id"], revision["id"]]]
    assert chronicle.get(original["id"])["text"] == original["text"]


def test_digest_retry_is_bound_to_the_request_after_model_wait_reprepare(tmp_path, fit, monkeypatch):
    from contextlib import nullcontext
    import hashlib
    from ouroboros import model_wait
    from ouroboros.memory_guidance import remembering_guidance
    chronicle, (original,) = _digest_store(tmp_path)
    callbacks = {}
    def register(role, callback):
        callbacks[role] = callback
        return nullcontext()
    waiter = SimpleNamespace(overrides={}, register_reprepare=register)
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: waiter)
    route_b = {"model": "switched/model", "use_local": False, "model_account_override": "acct-B"}
    long_text = "A non-shrinking interpretation. " * 1000
    def rebound(model, _prompt):
        # Exercise the actual reprepare callback, not a settings-only change.
        waiter.overrides["light"] = dict(route_b)
        callbacks["light"]({**model.calls[-1], **route_b})
        return {"content": long_text}, {**model.usage, "provider": "openrouter", "resolved_model": "switched/model"}
    first = _LLM(effect=rebound)
    usage = _digest_run(tmp_path, first)
    assert len(first.calls) == 1 and usage["_light_dispatch_binding"] == route_b
    (receipt,) = [row for row in chronicle.records(kinds=["maintenance"]) if row.get("status") == "not_shorter"]
    fit_receipt = chronicle.get(receipt["id"].replace("digest-attempt:", "digest-fit:", 1))
    assert fit_receipt["target_id"] == receipt["id"] and fit_receipt["target_fits"] is False
    assert receipt["source_keys"] == [[original["id"], original["id"]]]
    assert receipt["guidance_sha256"] == usage["_remembering_guidance_sha256"] == hashlib.sha256(
        remembering_guidance(tmp_path).encode("utf-8")).hexdigest()
    still_b = _long_digest()
    _digest_run(tmp_path, still_b)
    assert still_b.calls == []
    waiter.overrides.clear()
    on_a = _long_digest()
    _digest_run(tmp_path, on_a)
    assert on_a.calls == []  # the completed judgment is source-bound, even after returning to A
    _digest_run(tmp_path, on_a, fitting_demand={"purpose": "explicit_revision"})
    assert on_a.calls == []  # Rewording the same demand does not undo the reprepare receipt.
    next_usage = _digest_run(tmp_path, on_a, fitting_demand={"purpose": "owner_mode", "rendered_mode": "nano",
                                                           "requirement_tokens": 85000})
    assert len(on_a.calls) == 1 and on_a.calls[0]["model"] == "test/model"
    assert next_usage["_light_dispatch_binding"]["model"] == "test/model"
    assert original["text"] in str(on_a.calls[0]["messages"])
    assert len([row for row in chronicle.records(kinds=["maintenance"]) if row.get("status") == "not_shorter"]) == 2


def test_a_legacy_single_era_retry_record_is_read_as_one_run(tmp_path):
    legacy = {"era_retry": {"source_sha256": "abc", "route": {"model": "m", "use_local": False}}}
    assert c._era_retry_runs(legacy) == {"abc": {"route": {"model": "m", "use_local": False}}}
    assert c._era_retry_runs({"era_retry": "garbage"}) == {} and c._era_retry_runs({}) == {}


def test_health_names_a_withheld_era_without_a_timestamp(tmp_path):
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path,
                          repo_path=lambda p: tmp_path / p, drive_path=lambda p: tmp_path / p)
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {"last_consolidated_offset": 100})
    assert not any("ERA COMPRESSION" in line for line in context_health._memory_health_lines(env))
    atomic_write_json(tmp_path / "memory" / "dialogue_meta.json", {
        "era_retry": {"abcdef0123456789": {"route": {"model": "light/model", "use_local": False}},
                      "0123456789abcdef": {"route": "unknown"}}})
    rows = [line for line in context_health._memory_health_lines(env) if "ERA COMPRESSION WITHHELD" in line]
    assert len(rows) == 2 and "abcdef012345" in rows[0] and "light/model" in rows[0] and "2026-" not in rows[0]
    assert "0123456789ab" in rows[1]



# --- every scratchpad pass names its outcome ------------------------------------------


class _Scratch:
    def __init__(self, content):
        self.content = content

    def chat(self, **_kwargs):
        return {"content": self.content}, {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.02,
                                           "provider": "openrouter", "resolved_model": "light/served"}


def _scratchpad(tmp_path, count=4):
    memory = Memory(tmp_path)
    for index in range(count):
        memory.append_scratchpad_block(f"block-{index}-" + (chr(97 + index) * 8_000), source=f"source-{index}")
    return memory


def test_a_scratchpad_replacement_reports_its_counts_and_source(tmp_path):
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge",
                             _Scratch(json.dumps({"knowledge_entries": [], "compressed_block": "compressed"})))
    events = _events(tmp_path, "scratchpad_consolidation")
    assert len(events) == 1
    event = events[0]
    assert event["outcome"] == "replaced" and event["pressure"] is False
    assert (event["blocks_before"], event["compressed_blocks"], event["blocks_after"]) == (4, 2, 3)
    assert event["chars_before"] > event["chars_after"] > 0
    assert event["source_entry_id"] == memory.load_scratchpad_blocks()[0]["metadata"]["source_ref"]["entry_id"]
    assert event["knowledge_writes"] == {"ok": 0, "failed": 0}
    assert event["last_error_kind"] is None and event["accounted_upper_bound_usd"] == 0.02


@pytest.mark.parametrize("content, outcome", [
    ("not the requested JSON", "failed"),
    (json.dumps({"knowledge_entries": [], "compressed_block": "   "}), "empty_block"),
])
def test_a_refused_scratchpad_pass_names_why_and_keeps_the_blocks(tmp_path, content, outcome):
    memory = _scratchpad(tmp_path)
    before = memory.load_scratchpad_blocks()
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(content))
    assert memory.load_scratchpad_blocks() == before
    events = _events(tmp_path, "scratchpad_consolidation")
    assert len(events) == 1 and events[0]["outcome"] == outcome
    assert events[0]["blocks_after"] == 4 and events[0]["source_entry_id"] == ""
    assert events[0]["last_error_kind"] == ("scratchpad_consolidation_failed" if outcome == "failed" else None)


def test_no_scratchpad_pass_means_no_event(tmp_path):
    memory = Memory(tmp_path)
    memory.append_scratchpad_block("small", source="task")
    assert c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch("unused")) is None
    assert _events(tmp_path, "scratchpad_consolidation") == []


# --- a declined reflection lesson is a fact --------------------------------------------


def test_project_scoped_reflection_skips_are_typed_events(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    applied = reflection.apply_memory_actions(env, [
        {"type": "scratchpad_append", "content": "a lesson", "task_id": "t1"},
        {"type": "identity_update_candidate", "content": "a trait", "task_id": "t1"},
        {"type": "knowledge_write", "content": "topic-less", "task_id": "t1"},
    ], project_id="proj_x")
    assert applied == 0
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"]) for e in events] == [
        ("scratchpad_append", "project_scoped_task"), ("identity_update_candidate", "project_scoped_task"),
        ("knowledge_write", "missing_topic")]
    assert all(e["project_id"] == "proj_x" and e["task_id"] == "t1" and e["content_chars"] > 0 for e in events)
    assert events[0]["input_ref"] == {"status": "source_unavailable", "project_id": "proj_x"}
    assert not (tmp_path / "memory" / "scratchpad_blocks.json").exists()


def _reflect(tmp_path, llm, task_id="t-skip", project_id="proj_x"):
    return reflection.generate_reflection(
        {"id": task_id, "text": "the episode", "drive_root": str(tmp_path), "project_id": project_id},
        {}, "trace", llm, {"rounds": 1, "cost": 0.1})


_REJECTED_RAW = [
    {"type": "knowledge_write", "topic": "people/alex", "content": "   "},
    {"type": "knowledge_write", "content": "topic-less"},
    {"type": "delete_everything", "content": "nope"},
    {"type": "scratchpad_append", "content": "a kept lesson"},
]


def test_rejected_raw_reflection_actions_are_typed_events_on_the_real_path(tmp_path, fit):
    # Round 2 (critical 3): production drops empty and topic-less actions in the
    # validator, before apply_memory_actions ever runs; the event must fire there.
    from tests.test_knowledge_consolidation import MemoryLLM

    entry = _reflect(tmp_path, MemoryLLM("Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps(_REJECTED_RAW)))
    assert entry["reflection"] == "Reflection." and [a["type"] for a in entry["memory_actions"]] == ["scratchpad_append"]
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"], e["content_chars"]) for e in events] == [
        ("knowledge_write", "empty_content", 3), ("knowledge_write", "missing_topic", len("topic-less")),
        ("delete_everything", "unsupported_type", len("nope"))]
    assert all((e["task_id"], e["project_id"]) == ("t-skip", "proj_x") for e in events)
    # The retained exact task input the model answered is what the validator seam
    # kept; the rejected reflection text itself is not retained.
    assert entry["source_ref"]["kind"] == "task_source"
    assert all(e["input_ref"] == entry["source_ref"] for e in events)


def test_a_failed_validator_skip_event_cannot_abort_the_reflection(tmp_path, fit, monkeypatch, caplog):
    from tests.test_knowledge_consolidation import MemoryLLM

    original = reflection.append_jsonl

    def broken_event(path, row, **kwargs):
        if row.get("type") == "reflection_memory_action_skipped":
            raise OSError("event store unavailable")
        return original(path, row, **kwargs)

    monkeypatch.setattr(reflection, "append_jsonl", broken_event)
    entry = _reflect(tmp_path, MemoryLLM("Reflection.\nMEMORY_ACTIONS_JSON: " + json.dumps(_REJECTED_RAW)))
    assert entry["reflection"] == "Reflection."  # not "(reflection generation failed ...)"
    assert [a["content"] for a in entry["memory_actions"]] == ["a kept lesson"]
    assert caplog.text.count("Reflection memory skip event could not be written") == 3
    assert _events(tmp_path, "reflection_memory_action_skipped") == []


def test_failed_skip_event_cannot_discard_later_reflection_lessons(tmp_path, monkeypatch, caplog):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    original = reflection.append_jsonl

    def broken_event(path, row, **kwargs):
        if row.get("type") == "reflection_memory_action_skipped":
            raise OSError("event store unavailable")
        return original(path, row, **kwargs)

    monkeypatch.setattr(reflection, "append_jsonl", broken_event)
    assert reflection.apply_memory_actions(env, [
        {"type": "knowledge_write", "content": "topic-less", "task_id": "t1"},
        {"type": "scratchpad_append", "content": "a real lesson", "task_id": "t1"},
    ], project_id="") == 1
    assert "Reflection memory skip event could not be written" in caplog.text
    assert "a real lesson" in Memory(tmp_path).load_scratchpad()


def test_applied_reflection_actions_emit_no_skip(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, [
        {"type": "scratchpad_append", "content": "a lesson", "task_id": "t1"},
        {"type": "identity_update_candidate", "content": "a trait", "task_id": "t1"},
        {"type": "scratchpad_append", "content": "   ", "task_id": "t1"},
    ]) == 2
    events = _events(tmp_path, "reflection_memory_action_skipped")
    assert [(e["action_type"], e["reason"], e["project_id"]) for e in events] == [
        ("scratchpad_append", "empty_content", "")]


# --- the host stamp on every source_capture row --------------------------------------


def _address(root, topic="people/alex"):
    return store.resolve_knowledge_address(root, topic, "global")


def test_an_unnamed_writer_and_a_named_one_both_stamp_the_capture_row(tmp_path):
    target = _address(tmp_path)
    assert store.write_knowledge_note(target, "# Alex\n\nFirst.").ok
    legacy_shaped = _history(tmp_path)[-1]
    assert legacy_shaped["publication"] == "source_capture"
    assert (legacy_shaped["writer"], legacy_shaped["route"], legacy_shaped["writer_input_ref"]) == (
        store.UNKNOWN_STAMP, store.UNKNOWN_STAMP, store.UNKNOWN_STAMP)
    assert (legacy_shaped["old_chars"], legacy_shaped["new_chars"]) == (0, len(legacy_shaped["new_content"]))

    current = store.read_knowledge_note(target)
    result = store.write_knowledge_note(target, "# Alex\n\nFirst. Second.", expected_revision=current.revision,
                                        writer="turn", route={"model": "m"}, writer_input_ref={"chat": 1})
    assert result.ok
    row = _history(tmp_path)[-1]
    assert (row["writer"], row["route"], row["writer_input_ref"]) == ("turn", {"model": "m"}, {"chat": 1})
    assert row["old_chars"] == len(current.text) and row["new_chars"] == len(result.current.text)
    assert row["delta"]["old_chars"] == row["old_chars"] and row["source_ref"] == result.current.source_ref()
    assert "writer" not in result.current.text  # the stamp lives on the history row, never in the note


def test_the_route_stamp_is_what_answered_never_the_configuration():
    # Review F3: a model-wait override or account rotation changes what ANSWERED;
    # the configured Light route cannot say which model wrote the note.
    assert store.observed_route_stamp({"cost": 0.01}) == store.UNKNOWN_STAMP  # no physical fact at all
    assert store.observed_route_stamp(None) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"provider": "openrouter", "resolved_model": "openai/gpt-x"}) == {
        "provider": "openrouter", "model": "openai/gpt-x"}
    # Round 2 (advisory 5): a partial physical fact leaves the missing field unknown,
    # never the caller's configured model; the local lane stamps its own resolved_model.
    assert store.observed_route_stamp({"provider": "openrouter"}) == {"provider": "openrouter", "model": "unknown"}
    assert store.observed_route_stamp({"resolved_model": "local-model"}) == {"provider": "unknown", "model": "local-model"}
    assert store.observed_route_stamp({"provider": "local", "resolved_model": "local-model"}) == {
        "provider": "local", "model": "local-model"}
    # The production-shaped served route names the account as credentialProfileId.
    served = {"provider": "claudexor", "resolved_model": "claude-fable", "claudexor": {"route": {
        "source": "claude", "model": "claude-fable", "credentialProfileId": "acct-B", "accountFingerprint": "fp-B"}}}
    assert store.observed_route_stamp(served) == {
        "provider": "claudexor", "model": "claude-fable", "source": "claude", "account": "acct-B",
        "account_fingerprint": "fp-B"}
    rotated = {**served, "claudexor": {"route": {**served["claudexor"]["route"], "credentialProfileId": "acct-C"}}}
    assert store.observed_route_stamp(rotated)["account"] == "acct-C"  # rotation is visible in the stamp
    # A merged consolidation usage forwards the LAST physical route of the unit.
    merged = c._merge_consolidation_usage({"cost": 0.01, "provider": "openrouter", "resolved_model": "a"},
                                          {"cost": 0.01, "provider": "openrouter", "resolved_model": "b"})
    assert merged["_observed_route"] == {"provider": "openrouter", "model": "b"}
    assert store.observed_route_stamp(merged) == {"provider": "openrouter", "model": "b"}
    assert "_observed_route" not in c._merge_consolidation_usage({"cost": 0.01})


def test_the_route_stamp_never_reads_a_configured_route_from_an_empty_usage():
    # Round 2 (advisory 5): the former ``model``/``use_local`` fallback turned an
    # empty usage into the configured route. Only physical facts are read now.
    import inspect

    assert list(inspect.signature(store.observed_route_stamp).parameters) == ["usage"]
    assert store.observed_route_stamp({}) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"cost": 0.0, "prompt_tokens": 0}) == store.UNKNOWN_STAMP
    assert store.observed_route_stamp({"_observed_route": "unknown"}) == store.UNKNOWN_STAMP  # not a dict stamp


def test_a_merge_never_lets_an_earlier_stamp_masquerade_as_the_final_call():
    # Round 2 (advisory 5): the final call of a unit answered without a physical
    # fact (a released send, an exception's usage); an earlier known stamp must not
    # be forwarded as if that call had produced it.
    stamped = {"cost": 0.01, "provider": "openrouter", "resolved_model": "a"}
    merged = c._merge_consolidation_usage(stamped, {"cost": 0.01})
    assert "_observed_route" not in merged and store.observed_route_stamp(merged) == store.UNKNOWN_STAMP
    # A forwarded merged stamp counts as the (known) last call when it IS the last usage ...
    again = c._merge_consolidation_usage({"cost": 0.01}, c._merge_consolidation_usage(stamped))
    assert again["_observed_route"] == {"provider": "openrouter", "model": "a"}
    # ... and an unknown-last merged unit stays unknown through a further merge.
    nested = c._merge_consolidation_usage(stamped, c._merge_consolidation_usage(stamped, {"cost": 0.01}))
    assert store.observed_route_stamp(nested) == store.UNKNOWN_STAMP


def test_a_direct_turn_stamps_itself_and_its_observed_route_when_the_loop_recorded_one(tmp_path):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="turn-1")
    assert "✅" in knowledge_tools._knowledge_write(ctx, "notes/a", "Plain observation.")
    first = _history(tmp_path)[-1]
    assert (first["writer"], first["route"], first["task_id"]) == ("turn", store.UNKNOWN_STAMP, "turn-1")

    # The loop records what answered its last round on every lane, not only Claudexor.
    ctx._accumulated_usage = {"_observed_route": {"provider": "openrouter", "model": "openai/gpt-x"}}
    assert "✅" in knowledge_tools._knowledge_write(ctx, "notes/b", "Another observation.")
    second = _history(tmp_path)[-1]
    assert second["writer"] == "turn" and second["route"] == {"provider": "openrouter", "model": "openai/gpt-x"}


class _ThreeRoomNominating:
    """One block of three rooms; each room's correction answers on its own route.

    Review N2 / round 2 (advisory 6): provenance is per nomination. Room A's
    correction has no physical route fact (unknown), rooms B and C answer on two
    different accounts, and every correction also tries to forge the host stamp.
    """
    topics = {"A": "people/alex", "B": "people/bob", "C": "people/cara"}
    accounts = {"A": None, "B": "acct-1", "C": "acct-2"}

    def __init__(self):
        self.corrections = []

    @staticmethod
    def _served(account):
        return {"provider": "claudexor", "resolved_model": "light/served",
                "claudexor": {"route": {"source": "codex", "credentialProfileId": account}}}

    def chat(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if prompt.startswith("Compare this draft memory"):
            room = next(name for name in "ABC" if f"## Draft memory\nEpisode {name}." in prompt)
            self.corrections.append(room)
            account = self.accounts[room]
            usage = {"cost": 0.01, **({} if account is None else self._served(account))}
            return {"content": f"Episode {room}, checked.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
                {"topic": self.topics[room], "content": f"Understanding {room}.", "_nomination_route": _FORGED}])}, usage
        room = "A" if "entry-0 " in prompt else "B" if "entry-34 " in prompt else "C"
        return {"content": f"Episode {room}.\nKNOWLEDGE_ENTRIES_JSON: " + json.dumps([
            {"topic": self.topics[room], "content": f"Understanding {room}."}])}, {"cost": 0.01, **self._served("acct-draft")}


def test_dialogue_consolidation_stamps_each_nomination_with_its_own_correction_route(tmp_path, fit):
    chat, blocks, meta = _paths(tmp_path)
    chat.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"ts": f"2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z", "direction": "in",
             "text": f"entry-{index} ", "task_id": "consolidate", "chat_id": 1 if index < 34 else 2 if index < 67 else 3} for index in range(100)]
    chat.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="consolidate")
    llm = _ThreeRoomNominating()
    usage = c.consolidate(chat, blocks, meta, llm, knowledge_context=ctx, completed_task={"id": "consolidate"})
    assert llm.corrections == ["A", "B", "C"] and usage["_blocks_written"] == 3
    # The block-wide stamp is the LAST call's known route (room C's correction) ...
    assert c._route_stamp(usage)["account"] == "acct-2"
    captures = {row["topic"]: row for row in _history(tmp_path) if row.get("publication") == "source_capture"}
    assert set(captures) == {"people/alex", "people/bob", "people/cara"}
    assert all(row["writer"] == "chronicle_correction" for row in captures.values())
    # ... yet each nomination carries the route of the correction that released IT:
    # an explicit unknown outranks the known block stamp, two known routes stay
    # distinct within one block, and the forged model stamp reached none of them.
    assert captures["people/alex"]["route"] == store.UNKNOWN_STAMP
    assert captures["people/bob"]["route"] == {
        "provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-1"}
    assert captures["people/cara"]["route"] == {
        "provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-2"}
    from ouroboros.chronicle_store import ChronicleStore
    chronicle = ChronicleStore(tmp_path)
    for topic, capture in captures.items():
        ref = capture["writer_input_ref"]
        revision = chronicle.get(ref["record_id"])
        assert ref["kind"] == "chronicle" and revision["metadata"]["source_refs"] == ref["source_refs"]
        (entry,) = revision["metadata"]["knowledge_entries"]
        assert entry["topic"] == topic and "_nomination_route" not in entry
    nominations = [row for row in _history(tmp_path) if row.get("type") == "dialogue_knowledge_nominations"]
    assert len(nominations) == 3
    assert {entry["topic"] for row in nominations for entry in row["nominations"]} == set(captures)
    assert not any("_nomination_route" in entry for row in nominations for entry in row["nominations"])


def test_scratchpad_consolidation_stamps_its_journal_source(tmp_path):
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(json.dumps({
        "knowledge_entries": [{"topic": "lessons/one", "content": "A durable lesson."}],
        "compressed_block": "compressed"})))
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["writer"] == "scratchpad_consolidation"
    assert capture["route"] == {"provider": "openrouter", "model": "light/served"}
    assert capture["writer_input_ref"] == memory.load_scratchpad_blocks()[0]["metadata"]["source_ref"]
    assert _events(tmp_path, "scratchpad_consolidation")[0]["knowledge_writes"] == {"ok": 1, "failed": 0}


_FORGED = {"provider": "forged", "model": "forged/model", "account": "forged-acct"}


def test_a_model_supplied_nomination_route_never_reaches_history_from_scratchpad(tmp_path):
    # Round 2 (critical 1): the scratchpad producer binds every key the model wrote
    # before the writer ran; a forged host stamp must be dropped at that binding.
    memory = _scratchpad(tmp_path)
    c.consolidate_scratchpad(memory, tmp_path / "memory" / "knowledge", _Scratch(json.dumps({
        "knowledge_entries": [{"topic": "lessons/one", "content": "A durable lesson.", "_nomination_route": _FORGED,
                               "_expected_revision": "forged", "_task_id": "forged"}],
        "compressed_block": "compressed"})))
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["topic"] == "lessons/one" and capture["writer"] == "scratchpad_consolidation"
    assert capture["route"] == {"provider": "openrouter", "model": "light/served"}  # the host's observed stamp
    assert _events(tmp_path, "scratchpad_consolidation")[0]["knowledge_writes"] == {"ok": 1, "failed": 0}
    journal = [json.loads(line) for line in memory.journal_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    (bound,) = next(row for row in journal if row.get("type") == "blocks_consolidated")["knowledge_entries"]
    assert not [key for key in bound if key.startswith("_")]  # nothing model-written survives as a host key


def test_a_model_supplied_nomination_route_never_reaches_history_from_knowledge_maintenance(tmp_path, fit):
    from tests.test_memory_pressure_maintenance import setup_memory

    memory, ctx = setup_memory(tmp_path)
    assert store.write_knowledge_note(store.resolve_knowledge_address(tmp_path, "overview", "global"),
                                      "---\nsummary: Orientation.\n---\n" + "Detailed understanding. " * 200).ok

    class Forging:
        def chat(self, **_kwargs):
            return {"content": json.dumps({"knowledge_entries": [
                {"topic": "lessons/forged", "scope": "global", "content": "A shorter detail note.",
                 "_nomination_route": _FORGED}]})}, {
                "cost": 0.02, "provider": "claudexor", "resolved_model": "light/served",
                "claudexor": {"route": {"source": "codex", "credentialProfileId": "acct-real"}}}

    result = c.maintain_memory_pressure(memory, Forging(), ctx, fits=lambda: False)
    action = next(row for row in result["actions"] if row["owner"] == "knowledge_maintenance")
    assert [(row["topic"], row["scope"], row["ok"]) for row in action["writes"]] == [("lessons/forged", "global", True)]
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture"
                   and row["topic"] == "lessons/forged")
    assert capture["writer"] == "knowledge_maintenance"
    assert capture["route"] == {"provider": "claudexor", "model": "light/served", "source": "codex", "account": "acct-real"}


def test_bind_entries_keeps_model_fields_and_drops_every_host_key(tmp_path):
    reads = c.KnowledgeReadContext(ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="op-1"))
    (bound,) = reads.bind_entries([{"topic": "people/alex", "content": "Observed.", "scope": "global",
                                    "_nomination_route": _FORGED, "_anything": 1, "extra": "kept"}])
    assert bound == {"topic": "people/alex", "content": "Observed.", "scope": "global", "extra": "kept",
                     "expected_revision": None, "canonical_root": str(tmp_path), "task_id": "op-1"}


def test_project_reflection_action_uses_actor_readable_exact_source(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    env = SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path, repo_dir=tmp_path)
    entry = {"task_id": "t3", "ts": "2026-01-01T00:00:00Z", "route": {"provider": "claudexor", "model": "claude-fable"},
             "memory_actions": [
                 {"type": "knowledge_write", "topic": "lessons/project", "content": "Grounded.", "task_id": "t3"}]}
    reflection.append_reflection_routed(env, {"id": "t3", "project_id": "proj_x",
                                              "budget_drive_root": str(tmp_path)}, entry)
    action = entry["memory_actions"][0]
    source = action["_reflection_source_ref"]
    assert source["kind"] == "task_source"
    assert json.loads(read_actor_source_bytes(tmp_path, "t3", source))["memory_actions"][0]["content"] == "Grounded."
    assert reflection.apply_memory_actions(env, entry["memory_actions"], project_id="proj_x") == 1
    history = tmp_path / "projects" / "proj_x" / "knowledge_history.jsonl"
    row = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert row["writer_input_ref"]["sha256"] == source["sha256"]
    assert row["writer_input_ref"]["task_id"] == "t3"
    assert row["route"] == {"provider": "claudexor", "model": "claude-fable"}  # the reflection's answering route


def test_reflection_stamps_the_reflection_row_it_came_from(tmp_path):
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    assert reflection.apply_memory_actions(env, [
        {"type": "knowledge_write", "topic": "lessons/two", "content": "Reusable fact.", "task_id": "t9"}]) == 1
    capture = next(row for row in _history(tmp_path) if row.get("publication") == "source_capture")
    assert capture["writer"] == "reflection" and capture["route"] == store.UNKNOWN_STAMP
    assert capture["writer_input_ref"]["task_id"] == "t9"
    assert capture["writer_input_ref"]["read"]["arguments"]["path"] == "logs/task_reflections.jsonl"


def test_the_reader_less_knowledge_journal_is_no_longer_written(tmp_path):
    assert store.write_knowledge_note(_address(tmp_path), "# Alex\n\nFirst.").ok
    assert not (tmp_path / "memory" / "knowledge_journal.jsonl").exists()
    assert (tmp_path / "memory" / "knowledge_history.jsonl").exists()
    assert "knowledge_journal" not in inspect.getsource(store)
