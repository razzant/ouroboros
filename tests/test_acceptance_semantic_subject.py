"""Main's explicit subject decision separates source acknowledgement from meaning.

The owner-source observation and acknowledgement (``loop_messages``) and the root
admission seal's generation checks (``delivery_seal``) are the shared completion
protocol of ``finish_task`` / ``presence_finish``. A task-acceptance verdict once
rode beside them as an authoritative binding; that machinery is gone, so a
retained answer's binding is never authoritative and a review run in the trace is
evidence only.
"""
from __future__ import annotations

import copy
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import loop
from ouroboros.delivery_seal import (
    _end_task_acceptance_fence,
    _owner_generation_changed,
    _release_for_owner_followup,
)
from ouroboros.loop_delivery import (
    _current_delivery_candidate,
    _no_tool_final_answer,
    _replace_delivery_candidate,
    completion_observation,
    consume_completion_request,
    apply_delivery_subject_decision,
    delivery_subject_hash,
    delivery_subject_projection,
)
from ouroboros.loop_messages import (
    _record_owner_directive,
    acceptance_observation_prompt,
    acknowledge_acceptance_observation,
    capture_acceptance_observation,
    owner_source_sha256,
)


@pytest.fixture
def case(tmp_path):
    agent = SimpleNamespace(_owner_message_generation=0, _accepting_owner_messages=True)
    tool_ctx = SimpleNamespace(
        task_id="root", task_attempt=1, drive_root=tmp_path, budget_drive_root=tmp_path,
        task_metadata={"root_task_id": "root"}, task_contract={"expected_output": "Complete report"},
        _owner_directives=[], _loop_mailbox_seen_ids=set(), owner_message_admission_agent=agent,
        owner_message_admission_lock=threading.RLock(),
    )
    _record_owner_directive(tool_ctx, source="initial_user", content="Prepare the full report.", msg_id="first")
    tools = SimpleNamespace(_ctx=tool_ctx)
    trace = {"tool_calls": [], "review_runs": [], "reasoning_notes": []}
    ctx = SimpleNamespace(
        tools=tools, task_id="root", root_task_id="root", drive_root=tmp_path, status_drive_root=tmp_path,
        drive_logs=tmp_path / "logs", messages=[], incoming_messages=queue.Queue(),
        accumulated_usage={}, event_queue=None, round_idx=2, max_rounds=20, task_type="task",
    )
    capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    candidate = _replace_delivery_candidate(tools, ctx, trace, "The complete report.", control="candidate")
    run = {"authority": "host_root", "candidate_hash": candidate.content_sha256,
           "panel_id": "panel", "binding_hash": "binding", "aggregate_signal": "PASS"}
    trace["review_runs"].append(run)
    trace["review_decision"] = {"panel_id": "panel", "binding_hash": "binding"}
    return tool_ctx, tools, ctx, trace, candidate, run


def _followup(tool_ctx, text):
    tool_ctx.owner_message_admission_agent._owner_message_generation += 1
    _record_owner_directive(tool_ctx, source="direct_incoming", content=text, msg_id="followup")


def _select_retained(tools, ctx, trace, **subject):
    from ouroboros.tools.control_runtime import _finish_task

    tool_ctx = tools._ctx
    tool_ctx._completion_observation = completion_observation(tool_ctx, trace)
    payload = json.loads(_finish_task(tool_ctx, action="finish",
        answer_sha256=tool_ctx._delivery_candidate.content_sha256, acceptance_subject=subject or None))
    assert payload["status"] == "completion_requested"
    assert consume_completion_request(tools, ctx, trace)
    return tool_ctx._delivery_candidate.full_text


@pytest.mark.parametrize("text", ["Как дела?", "То есть мне нужен тот же полный отчёт."])
def test_consumed_status_or_rephrasing_keeps_subject_and_review_evidence(case, text):
    tool_ctx, tools, ctx, trace, candidate, run = case
    before = delivery_subject_hash(tool_ctx, trace)
    original = copy.deepcopy(tool_ctx._owner_directives)
    _followup(tool_ctx, text)
    assert _owner_generation_changed(tool_ctx)
    assert delivery_subject_hash(tool_ctx, trace) == before
    assert _current_delivery_candidate(ctx, trace) is None
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert observed["owner_source_sha256"] in acceptance_observation_prompt(tool_ctx, observed)
    tool_ctx._delivery_control_required = True
    answer = _select_retained(tools, ctx, trace, owner_source_sha256=observed["owner_source_sha256"])
    assert answer == "The complete report."
    assert delivery_subject_hash(tool_ctx, trace) == before
    assert not run.get("superseded_by_revision")
    assert _current_delivery_candidate(ctx, trace) is candidate
    assert tool_ctx._owner_directives[:1] == original
    assert tool_ctx._owner_directives[-1]["content"] == text


def test_changed_criteria_keeps_answer_but_creates_new_subject(case):
    tool_ctx, tools, ctx, trace, candidate, _run = case
    before, text_hash = delivery_subject_hash(tool_ctx, trace), candidate.content_sha256
    _followup(tool_ctx, "Include the budget as a separate section.")
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    tool_ctx._delivery_control_required = True
    answer = _select_retained(tools, ctx, trace,
        owner_source_sha256=observed["owner_source_sha256"],
        effective_criteria="Complete report, including a separate budget section.")
    assert answer == candidate.full_text
    assert candidate.content_sha256 == text_hash
    assert delivery_subject_hash(tool_ctx, trace) != before
    assert candidate.acceptance_binding["authoritative"] is False
    assert trace["delivery_candidate"]["effective_criteria"].endswith("budget section.")


def test_selected_read_observation_is_material_and_not_classified_by_tool_name(case):
    tool_ctx, tools, ctx, trace, _candidate, _run = case
    before = delivery_subject_hash(tool_ctx, trace)
    trace["tool_calls"].append({"tool": "read_file", "args": {"path": "report.json"},
                               "status": "ok", "result": '{"valid": false}'})
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    ok, error = apply_delivery_subject_decision(tools, ctx, trace, {
        "owner_source_sha256": observed["owner_source_sha256"], "material_tool_indices": [0],
    })
    assert ok and not error
    selected = delivery_subject_hash(tool_ctx, trace)
    assert selected != before
    trace["tool_calls"][0]["duration_sec"] = 200
    assert delivery_subject_hash(tool_ctx, trace) == selected
    trace["tool_calls"][0]["result"] = '{"valid": true}'
    assert delivery_subject_hash(tool_ctx, trace) != selected
    assert _current_delivery_candidate(ctx, trace) is None


def test_memory_narration_and_context_view_do_not_change_subject(case):
    tool_ctx, _tools, ctx, trace, _candidate, _run = case
    before = delivery_subject_hash(tool_ctx, trace)
    trace["reasoning_notes"].append("The current report is complete.")
    trace["tool_calls"].append({"tool": "knowledge_write", "args": {"topic": "lesson"},
                               "status": "ok", "result": "Knowledge recorded"})
    ctx.messages.clear()  # Rebuilding a model view cannot lose the retained owner source.
    assert delivery_subject_hash(tool_ctx, trace) == before
    assert "Prepare the full report." in json.dumps(delivery_subject_projection(tool_ctx, trace))


@pytest.mark.parametrize("arrival", ["direct_generation", "delivered_source", "incoming_queue", "mailbox"])
def test_unread_or_newer_owner_input_cannot_be_acknowledged(case, arrival):
    tool_ctx, tools, ctx, trace, candidate, _run = case
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    before = candidate.owner_source_sha256
    if arrival == "direct_generation":
        tool_ctx.owner_message_admission_agent._owner_message_generation += 1
    elif arrival == "delivered_source":
        _record_owner_directive(tool_ctx, source="direct_incoming", content="New requirement")
    elif arrival == "incoming_queue":
        ctx.incoming_messages.put("Still unread")
    else:
        from ouroboros.owner_mailbox import write_owner_message
        assert write_owner_message(tool_ctx.drive_root, "Still unread", "root", msg_id="queued")
    ok, error = apply_delivery_subject_decision(tools, ctx, trace, {
        "owner_source_sha256": observed["owner_source_sha256"], "effective_criteria": "Changed",
    })
    # The refusal names its typed cause: unread input is not the same fact as a changed source.
    cause = "owner_input_unread" if arrival in {"incoming_queue", "mailbox"} else "owner_source_changed"
    assert not ok and error.startswith(cause), error
    assert candidate.owner_source_sha256 == before
    assert candidate.effective_criteria != "Changed"


def test_queue_generation_cas_and_retained_source_both_prevent_blind_seal(case):
    tool_ctx, _tools, ctx, trace, _candidate, _run = case
    state = {"owner_message_generation": 0}
    calls = []
    tool_ctx._task_acceptance_fence_token = "queue-fence"
    tool_ctx.inspect_acceptance_fence = lambda **_kwargs: dict(state)
    tool_ctx.end_acceptance_fence = lambda **kwargs: calls.append(kwargs) or {"ok": True, "status": "released"}
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])
    _followup(tool_ctx, "Actually add another requirement.")
    assert _end_task_acceptance_fence(tool_ctx, outcome="author_stop")
    assert calls[-1]["outcome"] == "revision"
    assert tool_ctx._task_acceptance_fence_generation_mismatch is True


def test_queue_arrival_after_snapshot_refuses_source_ack(case):
    tool_ctx, _tools, ctx, trace, _candidate, _run = case
    state = {"owner_message_generation": 0}
    tool_ctx._task_acceptance_fence_token = "queue-fence"
    tool_ctx.inspect_acceptance_fence = lambda **_kwargs: dict(state)
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    state["owner_message_generation"] = 1
    assert not acknowledge_acceptance_observation(tool_ctx, observed["owner_source_sha256"])


def test_arrival_retains_old_paid_review_instead_of_declaring_semantic_change(case):
    """An owner follow-up reopens a sealed root for Main: the selected completion is
    void and the seal is released as a revision, while the paid review and its
    decision stay exactly as they were — arrival is unread input, not a judgment."""
    tool_ctx, _tools, _ctx, trace, _candidate, run = case
    trace["task_completion"] = {"action": "stop", "rationale": "Waiting for the owner.", "answer_sha256": "a" * 64}
    tool_ctx._task_acceptance_sealed_fence_token = "sealed-root"
    calls = []
    tool_ctx.end_acceptance_fence = lambda **kwargs: calls.append(kwargs) or {"ok": True, "status": "released"}
    _followup(tool_ctx, "Status?")
    assert _release_for_owner_followup(tool_ctx, trace)
    assert [call["outcome"] for call in calls] == ["revision"]
    assert tool_ctx._task_acceptance_fence_generation_mismatch is False
    assert "task_completion" not in trace
    assert not run.get("superseded_by_revision")
    assert trace["review_decision"] == {"panel_id": "panel", "binding_hash": "binding"}
    assert "acceptance_decision" not in trace


def test_invalid_subject_is_atomic_and_cannot_select_future_tool_results(case):
    tool_ctx, tools, ctx, trace, candidate, _run = case
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    for delta in [{"material_tool_indices": [0]}, {"effective_criteria": ""}, {"extra": True}]:
        ok, error = apply_delivery_subject_decision(tools, ctx, trace, {
            "owner_source_sha256": observed["owner_source_sha256"], **delta,
        })
        assert not ok and error
    assert candidate.effective_criteria["task_contract"]["expected_output"] == "Complete report"


def test_before_send_observation_is_not_itself_owner_ack(case):
    tool_ctx, _tools, ctx, trace, _candidate, _run = case
    previous = tool_ctx._acceptance_ack_source_sha256
    _followup(tool_ctx, "Status?")
    observed = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert observed["owner_source_sha256"] != previous
    assert tool_ctx._acceptance_ack_source_sha256 == previous
    assert owner_source_sha256(tool_ctx) == observed["owner_source_sha256"]


def test_a_selected_finish_delivers_without_opening_the_seal_and_closes_ingress(case, monkeypatch):
    """A plain finish never opens the root admission seal (only an author stop does);
    delivery finalizes services, passes the result review step, and closes owner
    ingress under the admission lock."""
    tool_ctx, tools, ctx, trace, candidate, _run = case
    tool_ctx._delivery_control_required = True
    seen = []
    monkeypatch.setattr(loop, "_project_child_result_dispositions", lambda *_args: None)
    monkeypatch.setattr(loop, "_enforce_plan_gate", lambda *_args: False)
    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_args: "")
    monkeypatch.setattr(loop, "_maybe_enforce_child_absorption_gate", lambda *_args: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(loop, "_finalize_task_services", lambda *_args: seen.append("services") or False)
    monkeypatch.setattr(loop, "_plan_gate_disclosure", lambda *_args: "")
    monkeypatch.setattr(loop, "_forced_orphan_note", lambda *_args, **_kwargs: "")
    fences = []
    tool_ctx.begin_acceptance_fence = lambda **kwargs: fences.append(kwargs) or {"token": "never", "owner_message_generation": 0}
    tool_ctx.end_acceptance_fence = lambda **kwargs: fences.append(kwargs) or {"ok": True, "status": "sealed"}

    def review(_ctx, content, _trace, _hold):
        seen.append(content)
        return False

    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review)
    monkeypatch.setattr(loop, "_handle_text_response", lambda text, *_args: (text, {}, trace))
    selected = _select_retained(tools, ctx, trace)
    returned = _no_tool_final_answer(
        selected, ctx, trace, tools, ctx.incoming_messages,
        set(), lambda *_args: None, explicit_candidate=True,
    )
    assert seen == ["services", candidate.full_text]
    assert returned[0] == candidate.full_text
    assert fences == []
    assert not getattr(tool_ctx, "_task_acceptance_sealed_fence_token", None)
    assert tool_ctx.owner_message_admission_agent._accepting_owner_messages is False


def test_pending_ingress_cannot_be_hidden_by_a_new_final_fence(case):
    tool_ctx, _tools, ctx, _trace, _candidate, _run = case
    tool_ctx._task_acceptance_fence_token = "final"
    tool_ctx._task_acceptance_owner_generation = 1
    tool_ctx.owner_message_admission_agent._owner_message_generation = 1
    called = []
    tool_ctx.end_acceptance_fence = lambda **kwargs: called.append(kwargs) or {"ok": True, "status": "released"}
    ctx.incoming_messages.put("Not yet delivered to Main")
    assert _end_task_acceptance_fence(tool_ctx, outcome="author_stop")
    assert called[-1]["outcome"] == "revision"
    assert tool_ctx._task_acceptance_fence_generation_mismatch is True


def test_observation_prompt_is_available_before_any_retained_answer(case):
    tool_ctx, _tools, ctx, trace, _candidate, _run = case
    del tool_ctx._delivery_candidate
    observation = capture_acceptance_observation(tool_ctx, trace, ctx.incoming_messages)
    assert observation["owner_source_sha256"] in acceptance_observation_prompt(tool_ctx, observation)
