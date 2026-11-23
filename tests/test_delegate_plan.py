"""Plan leaves use ordinary durable starts and terminal artifact answer turns."""
from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_current as current
from ouroboros.delegate_interactions import _delegate_answer
from ouroboros.delegate_reply import normalize_reply_to, validate_reply_target
from ouroboros.subagents import delegated_run_shape
from tests import _delegate_plan_shared

plan_setup = _delegate_plan_shared.plan_setup


@pytest.mark.parametrize("access", ["readonly", "workspace_write", "full"])
def test_explicit_modes_lower_without_erasing_access_derived_default(access):
    default = delegated_run_shape(True, access)
    assert default.mode == ("ask" if access == "readonly" else "agent")
    for mode in ("ask", "plan"):
        lowered = delegated_run_shape(True, access, mode)
        assert lowered.mode == mode and lowered.access == "readonly" and not lowered.delegated
    assert delegated_run_shape(True, access, "agent").mode == "agent"
    with pytest.raises(ValueError):
        delegated_run_shape(True, "unsupported", "plan")
    with pytest.raises(ValueError):
        delegated_run_shape(True, access, "unknown")


@pytest.mark.parametrize("unsupported", ["old_engine", "unknown_engine", "no_plan"])
def test_plan_capability_refuses_before_any_start_row_but_ordinary_starts_survive(plan_setup, unsupported):
    p = plan_setup
    if unsupported == "old_engine": p.engine.engine_version = "4.4.9"
    if unsupported == "unknown_engine": p.engine.engine_version = ""
    if unsupported == "no_plan": p.engine.modes = ["ask", "agent"]
    if unsupported == "route_no_plan": p.engine.route_modes = ["ask", "agent"]
    result = p.start(mode="plan")
    assert result["reason"] == "plan_thread_unsupported", result
    assert not p.engine.calls
    assert not [r for r in custody.custody_rows(p.root) if r["type"] in (custody.START_REQUESTED, custody.STARTED)]
    assert p.start(mode="ask")["status"] == "started"
    assert p.engine.calls[-1][0] == "/v2/runs"


def test_plan_start_holds_thread_creation_during_an_unanswered_removal(plan_setup):
    p = plan_setup
    current.mark_removal(p.root, "existing-project", {}, "custody")
    assert p.start(mode="plan")["reason"] == "registration_unconfirmed"
    assert p.engine.calls == [] and not custody.pending_invocations(p.root)


@pytest.mark.parametrize("phase", ["attachments", "processing"])
def test_plan_preparation_refusal_precedes_thread_creation(plan_setup, monkeypatch, phase):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.tools import delegate

    p, steps = plan_setup, []

    def registration(root):
        steps.append("registration")
        return "existing-project"

    def prepare(name, original):
        def step(*args, **kwargs):
            steps.append(name)
            if name == phase:
                raise ClaudexorUnavailable("fixture_preparation_refused", name)
            return original(*args, **kwargs)
        return step

    monkeypatch.setattr(p.engine, "find_project_id", registration)
    monkeypatch.setattr(delegate, "attach_selected_documents", prepare("attachments", delegate.attach_selected_documents))
    monkeypatch.setattr(delegate, "_processing_start_request", prepare("processing", delegate._processing_start_request))
    refused = p.start(mode="plan")
    assert refused["reason"] == "fixture_preparation_refused" and refused["definitely_unrun"] is True
    assert steps == ["registration", "attachments"] + (["processing"] if phase == "processing" else [])
    assert p.engine.calls == [] and not custody.pending_invocations(p.root)


def test_plan_start_records_ceiling_and_readonly_thread_before_send(plan_setup):
    p = plan_setup
    first = p.start(mode="plan")
    assert first["status"] == "started", first
    create, turn = p.engine.calls
    assert create[0] == "/v2/threads" and create[1]["mode"] == "plan"
    assert create[1]["scope"]["root"] == p.ctx.workspace_root
    assert create[1]["workspaceRoot"] != p.ctx.workspace_root
    assert turn[0] == f'/v2/threads/{first["thread_id"]}/turns'
    assert turn[1]["mode"] == "plan" and turn[1]["access"] == "readonly"
    assert "scope" not in turn[1] and "_thread_id" not in turn[1]
    custody._CUSTODY.clear()
    entry = current.run(p.root, first["run_id"])
    assert entry.execution_mode == "plan" and entry.effective_access == "readonly"
    assert entry.daemon_store_id == "store-plan", "the thread turn's own receipt binds its store"
    assert entry.authority_ceiling == first["authority_ceiling"] == {
        "task_authority_fingerprint": entry.authority_fingerprint,
        "allowed_access": "workspace_write", "target_root": p.ctx.workspace_root}
    requested = custody.invocation_record(p.root, entry.invocation_id)
    assert requested["authority_ceiling"] == entry.authority_ceiling
    second = p.start(mode="plan")
    assert second["run_id"] != first["run_id"] and second["thread_id"] != first["thread_id"]
    assert current.run(p.root, second["run_id"]).snapshot_id != entry.snapshot_id


def test_terminal_plan_questions_are_retained_addressed_artifacts_not_live_interactions(plan_setup):
    p = plan_setup
    started = p.start(mode="plan")
    entry, result = p.finish(started["run_id"])
    plan = result["plan"]
    assert plan["readiness"] == "needs_answers" and plan["questions"] == p.engine.questions["questions"]
    assert plan["plan_sha256"] == hashlib.sha256(p.engine.plan).hexdigest()
    assert plan["plan_ref"]["sha256"] == plan["plan_sha256"]
    assert plan["questions_ref"]["sha256"] and not plan["source_gaps"]
    assert not result.get("pending_interactions")
    assert not current.collection_pending(entry) and not custody.undisposed_patches(p.root)
    assert entry.snapshot_id not in custody.open_snapshot_ids(p.root)
    address = plan["reply_to"]
    assert normalize_reply_to(address) == address
    assert validate_reply_target(p.ctx, address, recipient=p.ctx.task_id) == address
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result
    assert retained_terminal_result(p.ctx, entry.run_id)["plan"] == plan


@pytest.mark.parametrize("questions", [b"not JSON", {"questions": []}, {"parse": "found", "questions": [{"id": "x"}, {"id": "x"}]}])
def test_malformed_plan_questions_stay_unverified_with_source_ref(plan_setup, questions):
    p = plan_setup
    started = p.start(mode="plan")
    p.engine.questions = questions
    _entry, result = p.finish(started["run_id"])
    assert result["plan"]["readiness"] == "unverified"
    assert result["plan"]["source_gaps"] and result["plan"]["questions_ref"]


def test_answer_turn_and_clarification_stay_on_same_thread(plan_setup):
    p = plan_setup
    first = p.start(mode="plan")
    entry, receipt = p.finish(first["run_id"])
    address = receipt["plan"]["reply_to"]
    for bad in ({**address, "plan_run_id": "wrong"}, {**address, "question_ids": ["wrong"]}):
        refused = json.loads(_delegate_answer(p.ctx, entry.run_id,
            answers=[{"question_id": "q1", "free_text": "README.md"}], reply_to=bad).text)
        assert refused["reason"] == "reply_to_mismatch"
    duplicate = json.loads(_delegate_answer(p.ctx, entry.run_id,
        answers=[{"question_id": "q1", "free_text": "README.md"}] * 2, reply_to=address).text)
    assert duplicate["reason"] == "reply_to_mismatch"
    assert len(p.engine.calls) == 2
    answer = json.loads(_delegate_answer(p.ctx, entry.run_id,
        answers=[{"question_id": "q1", "free_text": "README.md"}], reply_to=address).text)
    assert answer["status"] == "started", answer
    path, body, _key = p.engine.calls[-1]
    assert path == f'/v2/threads/{entry.thread_id}/turns'
    assert body["mode"] == "plan" and body["answersPlanRunId"] == entry.run_id
    assert "README.md" in body["prompt"] and "continueFrom" not in body
    next_entry = current.run(p.root, answer["run_id"])
    assert next_entry.authority_ceiling == entry.authority_ceiling
    assert next_entry.snapshot_id == entry.snapshot_id
    p.finish(answer["run_id"])
    clarified = p.start("Clarify the rollback step", continue_from=answer["run_id"])
    assert clarified["status"] == "started" and clarified["mode"] == "plan", clarified
    assert p.engine.calls[-1][0] == path and "answersPlanRunId" not in p.engine.calls[-1][1]


def test_a_plan_threads_workspace_fact_survives_its_saved_record_retry_and_continue(plan_setup):
    """A plan's turn body names no execution folder; its thread works in the private copy it was created on. That one
    workspace fact rides the START row and the saved invocation, a retry of a lost answer claims with it again, and a
    Continue on the same thread records the same folder; body, access and route are unchanged."""
    p = plan_setup

    def requested(invocation_id):
        return [row for row in custody.custody_rows(p.root)
                if row["type"] == custody.START_REQUESTED and row["invocation_id"] == invocation_id]

    p.engine.fail_next = True  # the turn is accepted, its answer lost
    lost = p.start(mode="plan")
    token = lost["pending_invocation_id"]
    [created] = [body for path, body, _key in p.engine.calls if path == "/v2/threads"]
    folder = created["workspaceRoot"]
    saved = custody.invocation_record(p.root, token)
    assert folder and saved["registration_workspace"] == folder
    assert "workspaceRoot" not in (saved["request"].get("execution") or {}) and saved["request"]["access"] == "readonly"
    retried = p.start("", retry_of=token)
    assert retried["status"] == "started", retried
    assert [row["registration_workspace"] for row in requested(token)] == [folder, folder]
    p.finish(retried["run_id"])
    clarified = p.start("Clarify the rollback step", continue_from=retried["run_id"])
    assert clarified["status"] == "started" and clarified["mode"] == "plan", clarified
    [follow] = requested(current.run(p.root, clarified["run_id"]).invocation_id)
    assert follow["registration_workspace"] == folder


def test_plan_thread_clarification_carries_its_named_note_in_the_wire_turn_prompt(plan_setup):
    from ouroboros.knowledge import resolve_knowledge_address

    p = plan_setup
    first = p.start(mode="plan")
    p.finish(first["run_id"])
    address = resolve_knowledge_address(p.root, "plan-evidence", "global")
    address.path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Order notes\n\n## Scope\nSynthetic evidence for this turn.\n"
    address.path.write_bytes(text.encode("utf-8"))
    successor = p.start("", continue_from=first["run_id"], knowledge_notes=[{"topic": "plan-evidence"}])
    assert successor["status"] == "started", successor
    path, body, _key = p.engine.calls[-1]
    assert path == f'/v2/threads/{first["thread_id"]}/turns'
    assert body["prompt"].count(text) == 1 and "## Named knowledge notes" not in body["instructions"]
    assert successor["named_notes"] == {"sent": [{"scope": "global", "topic": "plan-evidence",
        "revision": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}]}


def test_wrong_starter_and_interaction_selector_cannot_answer_plan(plan_setup):
    p = plan_setup
    first = p.start(mode="plan")
    p.finish(first["run_id"])
    args = {"answers": [{"question_id": "q1", "free_text": "yes"}]}
    wrong_kind = json.loads(_delegate_answer(p.ctx, first["run_id"], interaction_id="live", **args).text)
    assert wrong_kind["reason"] == "plan_answer_conflict"
    p.ctx.task_id = "peer"
    refused = json.loads(_delegate_answer(p.ctx, first["run_id"], **args).text)
    assert refused["status"] == "refused"
    assert len(p.engine.calls) == 2
