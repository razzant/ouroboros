"""Host classification is canonical metadata, never a provider-visible speaker name."""
from copy import deepcopy
import json

from ouroboros import config, context_budget, peer_roster
from ouroboros.context_fit import estimate_context_prompt_tokens
from ouroboros.llm_attempt import _physical_candidate
from ouroboros.llm_claudexor import ModelTurnState, _request
from ouroboros.llm_messages import _MessageShapingMixin
from ouroboros.loop_messages import CONTEXT_FACTS_HEADER, CONTEXT_FACTS_NAME
from tests.test_main_authored_context import call, main_loop as _main_loop

main_loop = _main_loop


def _wire(monkeypatch, messages, tools=None, *, prospective=False, native=None):
    monkeypatch.setattr("ouroboros.llm_claudexor.owned_engine_version",
                        lambda: config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION)
    return _request({"provider": "claudexor", "source": "codex", "resolved_model": "fixture-model"},
                    messages, tools, {"model_account_override": "", "_no_account_preference": True,
                                      "prospective": prospective, "model_turn_state": ModelTurnState(native)})


def test_real_main_tool_batch_roster_and_facts_are_unnamed_on_both_wires(main_loop, monkeypatch):
    f = main_loop
    monkeypatch.setattr(peer_roster, "independent_roots", lambda root: {
        "roots": [{"task_id": "peer", "title": "Another active task", "status": "running", "chat_id": 2,
                   "project_id": None}],
        "incomplete": False})
    first = call("read_file", {"path": "evidence.txt"}, "read-one")
    first["tool_calls"] += call("read_file", {"path": "evidence.txt"}, "read-two")["tool_calls"]
    answer, _, _ = f.run([first, {"content": "done"}])
    assert answer == "done"
    canonical = f.inputs[1]["messages"]  # actual tool batch -> roster -> next-round facts
    hosts = [row for row in canonical if isinstance(row.get("content"), str)
             and row["content"].startswith((peer_roster.ROSTER_NOTE_HEADER, CONTEXT_FACTS_HEADER))]
    assert any(row["content"].startswith(peer_roster.ROSTER_NOTE_HEADER) for row in hosts)
    assert any(row["content"].startswith(CONTEXT_FACTS_HEADER) for row in hosts)
    # This assertion fails on 7e64 before accessing the newly added private-key constant.
    assert all("name" not in row for row in hosts)
    key = context_budget.HOST_CONTEXT_KIND_KEY
    assert {row[key] for row in hosts} >= {CONTEXT_FACTS_NAME, peer_roster.ROSTER_SNAPSHOT_NAME}
    assert {r["tool_call_id"] for r in canonical if r.get("role") == "tool"} >= {"read-one", "read-two"}
    before = deepcopy(canonical)
    prospective = _wire(monkeypatch, canonical, f.inputs[1]["tools"], prospective=True)
    sent = _wire(monkeypatch, canonical, f.inputs[1]["tools"])
    assert prospective == sent
    physical = _physical_candidate({"messages": canonical})["messages"]
    expected = [{"role": row["role"], "content": row["content"]} for row in hosts]
    host_texts = {row["content"] for row in hosts}
    for rows in (physical, sent["messages"]):
        shown = [row for row in rows if isinstance(row.get("content"), str) and row["content"] in host_texts]
        assert shown == expected  # exact text, role, ordering and no host metadata/name
    assert canonical == before
    from ouroboros import context_compaction as cc
    from ouroboros.model_send_seal import EXPOSURE_ATOMS_BASIS

    observed = f.ctx._last_context_observation
    host_units = [unit for unit in cc.context_units(canonical, scope="dialogue")
                  if canonical[unit.start].get(key)]
    assert observed["messages"] == canonical and len(host_units) == len(hosts)
    assert observed["physical_source_status"] == "observed_projection"
    assert observed["exposure_basis"] == EXPOSURE_ATOMS_BASIS
    exposed = {(row["unit_id"], row["raw_sha256"]) for row in observed["exposed_units"]}
    assert all((unit.unit_id, unit.raw_sha256) in exposed for unit in host_units)
    altered = deepcopy(physical)
    victim = host_units[0]
    altered[victim.start]["content"] = "Different physical text cannot prove those canonical facts."
    no_longer_exposed = {(row["unit_id"], row["raw_sha256"]) for row in cc.exposed_context_units(canonical, altered)}
    assert (victim.unit_id, victim.raw_sha256) not in no_longer_exposed
    assert all((unit.unit_id, unit.raw_sha256) in no_longer_exposed for unit in host_units[1:])


def test_private_host_key_never_erases_real_names_or_nested_native_fields(monkeypatch):
    key = context_budget.HOST_CONTEXT_KIND_KEY
    native = {"route": {"source": "codex", "model": "fixture-model"}, key: {"name": "opaque", key: "native value"}}
    tool = {"type": "function", "function": {"name": "reader", "parameters": {
        "type": "object", "properties": {key: {"type": "string"}, "name": {"type": "string"}}}}}
    messages = [
        {"role": "user", "name": peer_roster.ROSTER_SNAPSHOT_NAME, "content": "A genuinely named actor."},
        {"role": "user", key: CONTEXT_FACTS_NAME, "content": "Exact host facts."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "read", "type": "function",
            "function": {"name": "reader", "arguments": json.dumps({key: "argument", "name": "real"})}}]},
        {"role": "tool", "tool_call_id": "read", "name": "reader", "content": "Exact result."},
    ]
    payload = {"messages": messages, "tools": [tool], "nativeContinuation": native}
    before = deepcopy(payload)
    physical = _physical_candidate(payload)
    sent = _wire(monkeypatch, messages, [tool], native=native)
    for candidate in (physical, sent):
        assert all(key not in row for row in candidate["messages"])
        assert candidate["messages"][0]["name"] == peer_roster.ROSTER_SNAPSHOT_NAME
        assert candidate["messages"][-1]["name"] == "reader"
        assert candidate["messages"][2]["tool_calls"] == messages[2]["tool_calls"]
        assert candidate["tools"] == [tool] and candidate["nativeContinuation"] == native
    assert estimate_context_prompt_tokens(messages, [tool]) == estimate_context_prompt_tokens(physical["messages"], [tool])
    assert payload == before
    switched = _MessageShapingMixin.sanitize_reasoning_on_model_switch(messages, "first/model", "second/model")
    assert switched[1][key] == CONTEXT_FACTS_NAME  # checkpoint/model-switch history stays canonical


def test_only_private_producer_classification_marks_replaced_host_rows():
    from ouroboros.context_source_view import _obsolete_host_rows

    key = context_budget.HOST_CONTEXT_KIND_KEY
    rows = [{"role": "user", "name": peer_roster.ROSTER_SNAPSHOT_NAME, "content": "Named actor"},
            {"role": "user", key: peer_roster.ROSTER_SNAPSHOT_NAME, "content": "Old snapshot"},
            {"role": "user", key: peer_roster.ROSTER_UPDATE_NAME, "content": "Old update"},
            {"role": "user", key: peer_roster.ROSTER_SNAPSHOT_NAME, "content": "Current snapshot"},
            {"role": "user", key: CONTEXT_FACTS_NAME, "content": "Old facts"},
            {"role": "user", key: CONTEXT_FACTS_NAME, "content": "Current facts"}]
    before = deepcopy(rows)
    assert _obsolete_host_rows(rows) == {1, 2, 4}
    assert rows == before


def test_working_checkpoint_keeps_canonical_host_rows_exposure_and_selected_review(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import owner_wait, working_checkpoint as wc
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_history_view import publish_review_history_view, selected_review_history
    from ouroboros.task_results import load_task_result, write_task_result
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_review_history_view import fixture, persisted

    ctx, limit = _loop_ctx(tmp_path, "task-review")
    write_task_result(tmp_path, ctx.task_id, "running")
    history = fixture()
    pointer, reader, authored = persisted(tmp_path, history, {"reviewer_outputs[0].text"})
    publish_review_history_view(ctx, authored, {}, pointer=pointer)
    host = {"role": "user", context_budget.HOST_CONTEXT_KIND_KEY: CONTEXT_FACTS_NAME,
            "content": "Exact host facts before Pause."}
    native = {"type": "thinking", "thinking": "Retained native reasoning", "signature": "opaque-signature"}
    limit.messages[:0] = [host, {"role": "assistant", "content": [native]}, authored]
    ctx._last_context_observation = {"messages": deepcopy(limit.messages), "exposed_units": [
        {"unit_id": "already-exposed", "raw_sha256": "a" * 64}], "physical_source_status": "observed_projection"}
    ctx._inspected_context_view = {"view_revision": "b" * 64}
    ctx._pending_compaction = {"working_note": "The actor's pending note", "expected_view_revision": "b" * 64}
    ctx._historical_author_inputs = {"source_ref": pointer["source_ref"]}
    expected = owner_wait.continuation_state(ctx, limit.messages, limit.llm_trace,
                                            limit.accumulated_usage, 4, limit.tool_schemas, limit.owner_msg_seen)
    wc.save_round(limit, "pre_effect")
    handoff = wc.prepare_recovery(tmp_path, ctx.task_id, from_attempt=1, cause="restart")
    saved = wc.load_recovery(ctx, handoff)
    assert saved["messages"] == expected["messages"]
    assert saved["context_observations"] == expected["context_observations"]
    fresh, _ = _loop_ctx(tmp_path, "task-review", attempt=2)
    messages, trace, usage, seen = [], {}, {}, set()
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=fresh), saved, messages, trace, usage, seen)
    assert messages == expected["messages"]  # includes private host tags, native blocks and exact capsule
    assert fresh._last_context_observation == ctx._last_context_observation
    assert fresh._pending_compaction == ctx._pending_compaction
    assert wc.close_unanswered_calls(messages, "interruption") == ["call_b"]
    assert messages[-1]["tool_call_id"] == "call_b" and "UNKNOWN" in messages[-1]["content"]
    assert sum(row.get("tool_call_id") == "call_a" for row in messages) == 1
    physical = _physical_candidate({"messages": messages})["messages"]
    assert context_budget.HOST_CONTEXT_KIND_KEY not in physical[0]
    assert physical[0]["content"] == host["content"] and messages[0] == host
    assert physical[1]["content"] == [native]
    write_task_result(tmp_path, ctx.task_id, "running", task_attempt=2)
    assert load_task_result(tmp_path, ctx.task_id)["selected_review_history_view"] == pointer
    view = selected_review_history(history, drive_root=tmp_path, task_id=ctx.task_id)
    assert view["selection_status"] == "applied" and view["actor_account"]["capsule"] == authored
    assert read_actor_source_bytes(tmp_path, ctx.task_id, pointer["source_ref"]) == reader(pointer["source_ref"])
