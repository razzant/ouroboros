"""Nothing in memory maintenance leaves silently (TZ-3 PR-1, invariant I4).

Every host decision that used to vanish is a typed fact, each proven in both
directions: a scratchpad pass and its outcome, a reflection lesson the host
declined, and the ``writer``/``route``/``writer_input_ref``/``old_chars``/
``new_chars`` stamp on every ``source_capture`` history row. The reader-less
``knowledge_journal.jsonl`` writer is gone, and so is the old dialogue writer
whose lock skips and withheld eras this file used to pin.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from ouroboros import consolidator as c
from ouroboros import knowledge as store
from ouroboros import reflection
from ouroboros.memory import Memory
from ouroboros.tools import knowledge as knowledge_tools
from ouroboros.tools.registry import ToolContext
from tests import test_consolidator_context_fit as fit_helpers

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
