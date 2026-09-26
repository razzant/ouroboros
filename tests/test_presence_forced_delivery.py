"""A Presence forced final keeps its internal record apart from what the conversation receives.

Owner Q4: host diagnostics are not sent automatically. The model may choose what
to say, including relevant limitations. The ONE forced model call declares its
outward delivery beside the internal record; without a valid declaration nothing
new is spoken, and a declared useful reply is spoken even when the run itself
failed. Deterministic fake-model replay; no transport sends anything.
"""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline as pipeline, loop
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.presence_runner import _cached_result
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolRegistry
from ouroboros.utils import append_jsonl
from tests.test_host_service_api import _seed_presence_behavior, _seed_token
from tests.test_presence_completion import _call
from tests.test_presence_runner import _admission

KEY = "telegram:bot-1:room-1:0"
RECORD = "Internal record: helper child-7 failed with provider 400; review not run; figures verified for Q1 only."


def _presence(binding="a" * 32, version=1):
    return {"binding_id": binding, "delivery_reporting_version": version,
            "event": {"conversation_key": KEY, "conversation_id": "room-1"}}


def _forced(outcome=None, message=None, **extra):
    body = {"delivery_control": "replace", "full_answer": RECORD, **extra}
    if outcome is not None:
        body["presence_finish"] = {"outcome": outcome, **({"message": message} if message is not None else {})}
    return json.dumps(body)


def _read():
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "read", "type": "function", "function": {"name": "chat_history", "arguments": "{}"},
    }]}


def _run(root, monkeypatch, forced, *, task=None, presence=True, handoff=None, first=None, ceiling=None,
         metadata=None, traces=None, rounds=1):
    """Round limit after one tool round, then the ONE forced call; real pipeline after it."""
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", str(rounds))
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    ctx = registry._ctx
    ctx.is_direct_chat = True
    ctx.task_metadata = {"inline_max_rounds": rounds, **({"presence": _presence()} if presence else {}),
                         **(metadata or {})}
    if presence if ceiling is None else ceiling:
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    if handoff:
        ctx._swarm_handoff_attempt = handoff
    registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
    calls, replies = [], iter([first or _read(), {"content": forced}])

    def respond(_llm, messages, *_a, **_k):
        calls.append([dict(row) for row in messages])
        return next(replies), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    task = task or {"id": "presence-loop", "type": "presence", "_presence_turn": True, "chat_id": 7,
                    "text": "Please help", "metadata": {"presence": _presence()}}
    task["_skip_post_task_synthesis"] = True
    text, usage, trace = loop.run_llm_loop(
        [{"role": "user", "content": "Please help"}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), root / "logs",
        lambda *_a, **_kw: None, queue.Queue(), task_id=task["id"], drive_root=root,
    )
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None,
                               events, task, text, usage, trace, 0.0, root / "logs", ctx=ctx)
    result = next((row for row in events if row["type"] == "presence_result"), None)
    if traces is not None:
        traces.append((trace, events))
    return result, load_task_result(root, task["id"]), calls, text


@pytest.mark.parametrize("forced,outcome,spoken,status", [
    (_forced("message", "Q1 figures are ready; Q2 is still coming."), "message",
     "Q1 figures are ready; Q2 is still coming.", "declared"),
    (_forced("message", "The review is delayed; Q1 figures are ready."), "message",
     "The review is delayed; Q1 figures are ready.", "declared"),
    (_forced("silent", ""), "silent", "", "declared"),
    (_forced("tool_delivered", "sent the table via the transport tool"), "tool_delivered", "", "declared"),
    (RECORD, "silent", "", "missing"),  # untyped internal prose is never speech
    (_forced("message", "   "), "silent", "", "invalid"),
    (_forced("maybe", "hi"), "silent", "", "invalid"),
    (_forced("silent", "but also this"), "silent", "", "invalid"),
    (_forced("deferred", "on it"), "silent", "", "invalid"),  # nothing was scheduled
    (json.dumps({"delivery_control": "replace", "full_answer": RECORD,
                 "presence_finish": {"outcome": "message", "message": "x", "to": "y"}}), "silent", "", "invalid"),
], ids=["message", "chosen_limitation", "silent", "tool_delivered", "prose", "blank_message", "unknown_outcome", "silent_with_text",
        "unscheduled_deferred", "extra_key"])
def test_forced_final_speaks_only_what_it_declares(tmp_path, monkeypatch, forced, outcome, spoken, status):
    result, stored, calls, text = _run(tmp_path, monkeypatch, forced)

    assert len(calls) == 2  # the one forced call; no repair or polishing round
    assert "[PRESENCE_DELIVERY]" in str(calls[-1][-1]["content"])
    assert "You decide what, if anything, to say" in str(calls[-1][-1]["content"])
    assert (result["outcome"], result["text"]) == (outcome, spoken)
    assert stored["reason_code"] == "round_limit"
    assert stored["outcome_axes"]["execution"]["status"] != "ok"  # speech is declared, not read off status
    assert stored["metadata"]["presence_declaration"]["status"] == status
    assert stored["metadata"]["presence_result_text"] == spoken
    assert _cached_result(tmp_path, "presence-loop").text == spoken
    assert text == RECORD and stored["result"].startswith(RECORD)  # the record survives beside it
    if outcome == "tool_delivered":
        assert result["message"] == "" and result["finish_note"] == "sent the table via the transport tool"


def test_a_malformed_control_body_speaks_nothing_and_keeps_the_host_fallback(tmp_path, monkeypatch):
    duplicate = ('{"delivery_control": "replace", "full_answer": "%s", "presence_finish": {"outcome": "silent"}, '
                 '"presence_finish": {"outcome": "message", "message": "dup"}}' % RECORD)
    result, stored, calls, _text = _run(tmp_path, monkeypatch, duplicate)
    assert len(calls) == 2
    assert (result["outcome"], result["text"]) == ("silent", "")
    assert stored["terminal_origin"] != "model_final"  # the host fallback is never spoken
    assert stored["metadata"]["presence_declaration"] == {"status": "invalid",
                                                          "reason": "the forced envelope repeats a key"}


def test_early_acknowledgement_is_shown_and_does_not_suppress_the_result(tmp_path, monkeypatch):
    logs = tmp_path / "logs"
    append_jsonl(logs / "chat.jsonl", {"task_id": "presence-loop", "direction": "in", "text": "Please help"})
    append_jsonl(logs / "chat.jsonl", {
        "task_id": "presence-loop", "type": "presence_delivery", "text": "Looking into it now.",
        "transport": {"conversation_key": KEY, "delivery": {"state": "delivered", "delivery_id": "d1", "part_id": "0"}},
    })
    append_jsonl(logs / "chat.jsonl", {
        "task_id": "presence-loop", "type": "presence_delivery", "text": "Partial table",
        "transport": {"conversation_key": KEY, "delivery": {"state": "uncertain", "delivery_id": "d2", "part_id": "0"}},
    })
    result, _stored, calls, _text = _run(tmp_path, monkeypatch, _forced("message", "Here are the Q1 figures."))

    prompt = str(calls[-1][-1]["content"])
    assert '"Looking into it now."' in prompt and "1 more part(s) have an uncertain outcome" in prompt
    assert "An early acknowledgement is not the promised result" in prompt
    assert (result["outcome"], result["text"]) == ("message", "Here are the Q1 figures.")


def test_declared_useful_partial_survives_failure_and_owed_child_stays_pollable(tmp_path, monkeypatch):
    handoff = {"status": "scheduled", "task_id": "later-work"}
    result, stored, calls, _text = _run(tmp_path, monkeypatch, _forced("deferred", "Started the full audit."),
                                        handoff=handoff)
    assert "deferred = acknowledge work that was actually scheduled (it was)" in str(calls[-1][-1]["content"])
    assert (result["outcome"], result["text"], result["work_ref"]) == ("deferred", "Started the full audit.", "later-work")
    silent, _stored, _calls, _text = _run(tmp_path / "silent", monkeypatch, RECORD, handoff=handoff)
    # No declaration: nothing new is said, but the admitted child is still owed.
    assert (silent["outcome"], silent["text"], silent["work_ref"]) == ("deferred", "", "later-work")


def test_ordinary_forced_final_is_unchanged(tmp_path, monkeypatch):
    task = {"id": "owner-task", "type": "task", "chat_id": 1, "text": "Please help"}
    _result, stored, calls, text = _run(tmp_path, monkeypatch, RECORD, task=task, presence=False)
    assert "[PRESENCE_DELIVERY]" not in str(calls[-1][-1]["content"])
    assert text == RECORD and stored["result"].startswith(RECORD)
    assert "presence_declaration" not in (stored.get("metadata") or {})


def test_promoted_work_result_reaches_the_work_endpoint_as_declared(tmp_path, monkeypatch):
    _seed_token(tmp_path, skill="telegram-bot", token="presence-token",
                permissions=["presence"], manifest_permissions=["presence"])
    binding = _seed_presence_behavior(tmp_path)
    task = {"id": "promoted-9", "type": "task", "chat_id": 7, "text": "Compile", "delegation_role": "root",
            "root_task_id": "promoted-9", "metadata": {"presence": _presence(binding)}}
    _run(tmp_path, monkeypatch, _forced("message", "The figures you asked for: 41 and 43."), task=task)
    with TestClient(create_host_service_app(tmp_path)) as client:
        body = client.get("/presence/work/promoted-9", params={"binding_id": binding},
                          headers={"X-Skill-Token": "presence-token"}).json()
    assert (body["outcome"], body["text"]) == ("message", "The figures you asked for: 41 and 43.")
    assert RECORD not in json.dumps(body)


@pytest.mark.parametrize("feedback", [True, False])
def test_invalidated_finish_is_named_void_with_this_tasks_confirmed_sends(tmp_path, monkeypatch, feedback):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    registry._ctx.task_metadata = {"presence": _presence()}
    registry._ctx.is_direct_chat = True
    logs = tmp_path / "logs"
    append_jsonl(logs / "chat.jsonl", {"task_id": "parent1", "direction": "in", "text": "Please help"})
    append_jsonl(logs / "chat.jsonl", {
        "task_id": "parent1", "type": "presence_delivery", "text": "The full answer, sent by tool.",
        "transport": {"conversation_key": KEY, "delivery": {"state": "delivered", "delivery_id": "d1", "part_id": "0"}},
    })
    reviews, calls = [], []

    def review(**kwargs):
        reviews.append(kwargs["content"])
        if feedback and len(reviews) == 1:  # the panel's feedback is new transcript content
            kwargs["messages"].append({"role": "user", "content": "[REVIEW] Also confirm the Q2 figures."})
        return len(reviews) == 1  # the first finish is held for another pass

    def respond(_llm, messages, *_a, **_k):
        calls.append([dict(row) for row in messages])
        return ([_call("tool_delivered", ""), _call("tool_delivered", "")][len(calls) - 1]), 0.0

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", review)
    monkeypatch.setattr(loop, "call_llm_with_retry", respond)
    _text, usage, _trace = loop.run_llm_loop(
        [{"role": "user", "content": "Please help"}], registry,
        SimpleNamespace(default_model=lambda: "test-model"), logs,
        lambda *_a, **_kw: None, queue.Queue(), task_id="parent1", drive_root=tmp_path,
    )

    assert len(calls) == 2
    notes = [row for row in calls[1] if "[PRESENCE_FINISH_NOT_ACCEPTED]" in str(row.get("content"))]
    assert "[PRESENCE_FINISH_NOT_ACCEPTED]" not in json.dumps(calls[0])
    assert usage["presence_completion_outcome"] == "tool_delivered"  # the fresh finish is accepted
    if not feedback:  # nothing new was said: the turn parks as before, with no added reminder
        assert notes == []
        return
    assert len(notes) == 1
    assert '"The full answer, sent by tool."' in str(notes[0]["content"])
    assert "you decide whether any of their facts matter" in str(notes[0]["content"])


# --- repair pass: arming identity, duplicate evidence, internal notes --------------

def test_a_child_inheriting_only_the_ceiling_keeps_its_ordinary_forced_final(tmp_path, monkeypatch):
    task = {"id": "child-3", "type": "task", "chat_id": 7, "text": "Check Q1", "delegation_role": "subagent",
            "parent_task_id": "promoted-9", "root_task_id": "promoted-9"}
    traces = []
    result, stored, calls, text = _run(tmp_path, monkeypatch, RECORD, task=task, presence=False, ceiling=True,
                                       metadata={"delegation_role": "subagent", "parent_task_id": "promoted-9"},
                                       traces=traces)
    assert "[PRESENCE_DELIVERY]" not in str(calls[-1][-1]["content"])  # a child answers its parent
    assert result is None and text == RECORD and stored["result"].startswith(RECORD)
    assert "presence_declaration" not in (stored.get("metadata") or {})
    assert [event["type"] for event in traces[0][1]].count("presence_result") == 0


def test_a_child_holding_the_inherited_binding_authority_still_answers_only_its_parent(tmp_path, monkeypatch):
    authority = {"presence_binding_authority": {"binding_id": "a" * 32}}  # it acts for the binding, never speaks
    task = {"id": "child-4", "type": "task", "chat_id": 7, "text": "Check Q1", "delegation_role": "subagent",
            "parent_task_id": "presence-loop", "root_task_id": "presence-loop", "metadata": dict(authority)}
    traces = []
    result, stored, calls, text = _run(tmp_path, monkeypatch, _forced("message", "Hello room"), task=task,
                                       presence=False, ceiling=True, traces=traces,
                                       metadata={"delegation_role": "subagent", **authority})
    assert "[PRESENCE_DELIVERY]" not in str(calls[-1][-1]["content"])
    assert result is None and "presence_declaration" not in (stored.get("metadata") or {})
    assert [event["type"] for event in traces[0][1]].count("presence_result") == 0


def test_duplicate_subject_evidence_survives_the_presence_arm_and_declares_nothing(tmp_path, monkeypatch):
    duplicated = ('{"delivery_control": "replace", "full_answer": "%s", "acceptance_subject": '
                  '{"owner_source_sha256": "aaa", "owner_source_sha256": "bbb"}, '
                  '"presence_finish": {"outcome": "message", "message": "Here are the figures."}}' % RECORD)
    traces = []
    result, stored, calls, text = _run(tmp_path, monkeypatch, duplicated, traces=traces)

    assert len(calls) == 2
    # The resolver saw the original bytes: the ambiguous subject is refused as a duplicate,
    # not silently collapsed to its last value and judged as a different source.
    assert traces[0][0]["forced_acceptance_subject"] == {
        "applied": False, "reason": "acceptance_subject requires an exact owner source and optional criteria/tool indices"}
    assert (result["outcome"], result["text"]) == ("silent", "")
    assert stored["metadata"]["presence_declaration"] == {"status": "invalid",
                                                          "reason": "the forced envelope repeats a key"}
    assert text == RECORD  # the record keeps its ordinary degraded-subject handling


def test_a_duplicate_inside_the_declaration_voids_only_the_declaration(tmp_path, monkeypatch):
    body = ('{"delivery_control": "replace", "full_answer": "%s", '
            '"presence_finish": {"outcome": "message", "message": "hi", "outcome": "silent"}}' % RECORD)
    result, stored, _calls, text = _run(tmp_path, monkeypatch, body)
    assert (result["outcome"], result["text"]) == ("silent", "")
    assert stored["metadata"]["presence_declaration"]["status"] == "invalid"
    assert text == RECORD and stored["terminal_origin"] == "model_final"


def test_a_tool_delivered_note_is_never_speech_even_when_owed_work_defers_the_turn(tmp_path, monkeypatch):
    note = "sent the table via the transport tool; helper child-7 failed"
    handoff = {"status": "scheduled", "task_id": "later-work"}
    result, stored, _calls, _text = _run(tmp_path, monkeypatch, _forced("tool_delivered", note), handoff=handoff)

    assert (result["outcome"], result["text"], result["work_ref"]) == ("deferred", "", "later-work")
    assert result["message"] == "" and result["finish_note"] == note  # context, never speech
    assert stored["metadata"]["presence_result_text"] == ""
    replay = _cached_result(tmp_path, "presence-loop")
    assert (replay.outcome, replay.text, replay.work_ref) == ("deferred", "", "later-work")
    assert note not in json.dumps(stored["metadata"])


@pytest.mark.parametrize("prepared_fits,reply,outcome,spoken,declaration", [
    # The priced candidate still fits twice: ordinary work continues and its reply is speech.
    ((True, True), "Here are the Q1 figures.", "message", "Here are the Q1 figures.", None),
    # The priced candidate confirms the last-fit stop: the committed forced call stays armed.
    ((True, False), RECORD, "silent", "", "missing"),
], ids=["repriced_fall_through", "confirmed_stop"])
def test_a_budget_repricing_arms_presence_only_for_the_forced_call_it_commits(
        tmp_path, monkeypatch, prepared_fits, reply, outcome, spoken, declaration):
    from ouroboros import task_pacing
    from ouroboros.contracts.task_contract import normalize_budget_profile

    ceiling = task_pacing.resolve_cost_ceiling(None, normalize_budget_profile(None), root_cap_usd=50.0)
    monkeypatch.setattr(loop, "_resolve_task_cost_ceiling", lambda *_a: ceiling)
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_k: {"accounted_usd": 20.0})
    # proxy last-fit -> exact probe last-fit -> the prepared candidate decides
    answers = iter((True, False, True, False, *prepared_fits))
    monkeypatch.setattr(task_pacing, "wrapup_reservation_fits", lambda **_k: next(answers, True))
    monkeypatch.setattr(task_pacing, "prospective_wrapup_attempt_request", lambda **_k: object())
    monkeypatch.setattr(task_pacing, "prepared_wrapup_candidate",
                        lambda _ctx, messages, **_k: (object(), messages))
    prepared, prepare = [], loop._prepare_forced_prompt
    monkeypatch.setattr(loop, "_prepare_forced_prompt", lambda *args: prepared.append(1) or prepare(*args))
    post_tool = loop._prepare_post_tool_budget_context

    def measured(tools, limit_ctx, *args):  # the fake route records no context-fit measurement
        limit_ctx.accumulated_usage["_context_prompt_estimate"] = 4_000
        return post_tool(tools, limit_ctx, *args)

    monkeypatch.setattr(loop, "_prepare_post_tool_budget_context", measured)

    result, stored, calls, text = _run(tmp_path, monkeypatch, reply, rounds=3)

    assert prepared == [1] and len(calls) == 2  # the real preparer priced the forced prompt once
    assert (result["outcome"], result["text"]) == (outcome, spoken)
    assert (stored["metadata"].get("presence_declaration") or {}).get("status") == declaration
    assert _cached_result(tmp_path, "presence-loop").text == spoken
    if declaration is None:
        assert "[PRESENCE_DELIVERY]" not in json.dumps(calls[-1])  # priced on a copy, never sent
        assert text == reply and stored["terminal_origin"] == "model_final"
    else:
        assert "[PRESENCE_DELIVERY]" in str(calls[-1][-1]["content"])
        assert stored["reason_code"] == "budget_exhausted" and text == RECORD


def _receipt(task_id, text, delivery_id, *, state="delivered", key=KEY):
    return {"task_id": task_id, "type": "presence_delivery", "text": text, "transport": {
        "conversation_key": key, "delivery": {"state": state, "delivery_id": delivery_id, "part_id": "0"}}}


def test_a_promoted_roots_observed_sends_reach_its_forced_final_without_an_inbound_row(tmp_path, monkeypatch):
    # A promoted root logs no inbound row of its own: only its tool sends carry its id.
    for row in (_receipt("promoted-9", "First table sent.", "d1"),
                _receipt("promoted-9", "Second part", "d2", state="uncertain"),
                _receipt("promoted-9", "Sent to another room", "d3", key="telegram:bot-1:room-2:0"),
                _receipt("presence-turn", "The turn's own early reply", "d4")):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", row)
    task = {"id": "promoted-9", "type": "task", "chat_id": 7, "text": "Compile", "delegation_role": "root",
            "root_task_id": "promoted-9", "metadata": {"presence": _presence()}}

    result, _stored, calls, _text = _run(tmp_path, monkeypatch, _forced("tool_delivered", ""), task=task)

    prompt = str(calls[-1][-1]["content"])
    assert 'Sends confirmed for this task so far: "First table sent." (live chat log only;' in prompt
    assert "so there may be more); 1 more part(s) have an uncertain outcome" in prompt
    assert "another room" not in prompt and "early reply" not in prompt
    assert (result["outcome"], result["text"]) == ("tool_delivered", "")


def test_send_facts_mark_only_an_uncovered_task_as_partial(tmp_path):
    from ouroboros.presence_context import presence_send_facts

    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"task_id": "turn", "direction": "in", "text": "x"})
    for task_id in ("turn", "promoted"):
        append_jsonl(tmp_path / "logs" / "chat.jsonl", _receipt(task_id, f"sent by {task_id}", task_id))

    assert presence_send_facts(tmp_path, "turn", _presence()) == '"sent by turn"'  # its inbound row covers it
    assert presence_send_facts(tmp_path, "promoted", _presence()).startswith('"sent by promoted" (live chat log only;')
    assert presence_send_facts(tmp_path, "silent-root", _presence()).startswith("none (live chat log only;")
    assert presence_send_facts(tmp_path, "promoted", _presence(version=0)).startswith("unknown (this transport")


def test_forced_prompt_and_facts_read_the_canonical_root_on_a_forked_drive(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    append_jsonl(canonical / "logs" / "chat.jsonl", {"task_id": "presence-loop", "direction": "in", "text": "x"})
    append_jsonl(canonical / "logs" / "chat.jsonl", {
        "task_id": "presence-loop", "type": "presence_delivery", "text": "Canonical receipt",
        "transport": {"conversation_key": KEY, "delivery": {"state": "delivered", "delivery_id": "d1", "part_id": "0"}},
    })
    append_jsonl(tmp_path / "logs" / "chat.jsonl", {"task_id": "presence-loop", "direction": "in", "text": "x"})
    _result, _stored, calls, _text = _run(tmp_path, monkeypatch, _forced("silent", ""),
                                          metadata={"budget_drive_root": str(canonical)})
    prompt = str(calls[-1][-1]["content"])
    assert 'Sends confirmed for this task so far: "Canonical receipt"' in prompt


EARLIER = "Earlier record: Q1 figures verified; Q2 not yet checked."


@pytest.mark.parametrize("body,record,outcome,spoken,declaration", [
    # A valid keep retains the earlier answer as the record and still speaks its declaration.
    ({"delivery_control": "keep"}, EARLIER, "message", "Fresh reply", {"status": "declared"}),
    ({"delivery_control": "replace", "full_answer": RECORD}, RECORD, "message", "Fresh reply", {"status": "declared"}),
    # A rejected envelope keeps the earlier answer as the record: its fresh reply is never spoken beside it.
    ({"delivery_control": "replace", "full_answer": ""}, EARLIER, "silent", "",
     {"status": "invalid", "reason": "the delivery-control envelope was rejected"}),
    ({"delivery_control": "revise", "full_answer": RECORD}, EARLIER, "silent", "",
     {"status": "invalid", "reason": "the delivery-control envelope was rejected"}),
    ({"full_answer": RECORD}, EARLIER, "silent", "",
     {"status": "invalid", "reason": "the delivery-control envelope was rejected"}),
], ids=["keep", "replace", "empty_replace_rejected", "unknown_verb_rejected", "missing_verb_rejected"])
def test_a_declaration_speaks_only_beside_the_answer_its_own_envelope_authorized(
        tmp_path, monkeypatch, body, record, outcome, spoken, declaration):
    from tests.test_delivery_forced_finalization import _arm_latch_with_candidate, _forced_test_context

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    forced_loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    ctx = registry._ctx
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    ctx.task_metadata = {**ctx.task_metadata, "presence": _presence()}
    _arm_latch_with_candidate(forced_loop, registry, limit_ctx, trace, text=EARLIER)  # a live earlier answer
    reply = json.dumps({**body, "presence_finish": {"outcome": "message", "message": "Fresh reply"}})
    monkeypatch.setattr(forced_loop, "call_llm_with_retry",
                        lambda *_a, **_k: ({"role": "assistant", "content": reply}, 0.0))

    text, usage, trace = forced_loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit")
    task = {"id": "parent1", "type": "presence", "_presence_turn": True, "chat_id": 7, "text": "Please help",
            "metadata": {"presence": _presence()}, "_skip_post_task_synthesis": True}
    events = []
    pipeline.emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None,
                               events, task, text, usage, trace, 0.0, tmp_path / "logs", ctx=ctx)
    result = next(row for row in events if row["type"] == "presence_result")
    stored = load_task_result(tmp_path, "parent1")

    assert text.startswith(record) and stored["result"].startswith(record)
    assert (result["outcome"], result["text"]) == (outcome, spoken)
    assert stored["metadata"]["presence_declaration"] == declaration
    assert stored["metadata"]["presence_result_text"] == spoken
