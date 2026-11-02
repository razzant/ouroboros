"""Plan custody stays immutable across legacy rows, task cards and resource bindings."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ouroboros import delegate_collection, delegate_custody as custody, delegate_custody_current as current
from ouroboros.delegate_continuation import ContinuationStart, NO_CONTINUATION, REFUSAL_SUPERSEDED
from ouroboros.delegate_plan import bind_start
from ouroboros.subagent_runtime import exact_start
from ouroboros.subagents import DelegationRoute, delegated_run_shape
from tests import _delegate_plan_shared
from tests.test_delegate_plan_promotion import ready_plan

plan_setup = _delegate_plan_shared.plan_setup


def test_plan_hash_drift_refuses_before_send_then_unchanged_plan_implements(plan_setup):
    p = plan_setup
    prior = ready_plan(p)
    original = p.engine.plan
    p.engine.plan = b"# Different plan\n"
    refused = p.start("Implement", continue_from=prior.run_id, mode="agent")
    assert refused["reason"] == "plan_hash_mismatch"
    assert len(p.engine.calls) == 2
    p.engine.plan = original
    assert p.start("Implement", continue_from=prior.run_id, mode="agent")["status"] == "started"


@pytest.mark.parametrize("options,reason", [({"mode": "agent", "continue_carrier": "packet"}, "plan_carrier_conflict"),
                                         ({"mode": "ask"}, "plan_mode_required")])
def test_plan_turn_refuses_carrier_or_mode_substitution(plan_setup, options, reason):
    p = plan_setup
    prior = ready_plan(p)
    assert p.start("Continue", continue_from=prior.run_id, **options)["reason"] == reason
    assert len(p.engine.calls) == 2
    assert p.start("Implement", continue_from=prior.run_id, mode="agent")["status"] == "started"


def test_missing_plan_artifact_never_implements_a_bare_prompt(plan_setup, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    p = plan_setup
    p.engine.questions = {"parse": "found", "questions": []}
    started = p.start(mode="plan")
    original = p.engine.get_run_artifact

    def missing(run_id, path):
        if path == "final/plan.md":
            raise ClaudexorUnavailable("artifact_not_found", path, status_code=404)
        return original(run_id, path)

    monkeypatch.setattr(p.engine, "get_run_artifact", missing)
    prior, receipt = p.finish(started["run_id"])
    assert not receipt["plan"].get("plan_sha256")
    assert p.start("Implement", continue_from=prior.run_id, mode="agent")["reason"] == "plan_artifact_unavailable"
    assert len(p.engine.calls) == 2
    monkeypatch.setattr(p.engine, "get_run_artifact", original)
    assert p.start("Implement", continue_from=prior.run_id, mode="agent")["status"] == "started"


def test_readiness_comes_from_server_projection_not_question_count(plan_setup):
    from ouroboros.delegate_plan import terminal_plan

    p = plan_setup
    prior = ready_plan(p)
    detail = p.engine.get_run(prior.run_id)
    detail["planReadiness"] = None
    unknown = terminal_plan(prior, p.engine, detail)
    assert unknown["readiness"] == "unverified" and unknown["source_gaps"]
    detail["planReadiness"] = {"state": "unverified", "questionCount": 0}
    assert terminal_plan(prior, p.engine, detail)["readiness"] == "unverified"
    detail["planReadiness"]["state"] = "ready"
    assert terminal_plan(prior, p.engine, detail)["readiness"] == "ready"


def test_override_does_not_attach_to_an_ordinary_continuation(plan_setup):
    p = plan_setup
    ordinary = p.start(mode="ask")
    p.finish(ordinary["run_id"])
    refused = p.start("Continue", continue_from=ordinary["run_id"], mode="agent", access="readonly",
                      override_plan_readiness=True)
    assert refused["reason"] == "plan_override_conflict", refused
    plan = ready_plan(p)
    assert p.start("Implement", continue_from=plan.run_id, mode="agent",
                   override_plan_readiness=True)["status"] == "started"


def test_legacy_ceiling_cannot_be_filled_by_duplicate_start_or_clarification(plan_setup):
    p = plan_setup
    prior = ready_plan(p)
    legacy = replace(prior, run_id="legacy", thread_id="", authority_ceiling={},
                     execution_mode="", effective_access="", promotion={})
    assert custody.record_started(p.root, legacy, shape={"access": "readonly", "mode": "plan"})
    assert custody.record_started(p.root, replace(legacy, thread_id="invented", authority_ceiling=prior.authority_ceiling),
                                  shape={"access": "workspace_write", "mode": "agent"})
    restored = current.run(p.root, "legacy")
    assert restored.authority_ceiling == {} and restored.thread_id == "" and restored.mode == "plan"
    request = {"scope": {"root": prior.target_root}, "mode": "plan", "access": "readonly"}
    args = (p.engine, request, DelegationRoute("some-route"), "fresh-invocation")
    tail = (delegated_run_shape(True, "workspace_write", "plan"), delegated_run_shape(True),
            {"authority_fingerprint": "current"}, prior.target_root, "")
    clarified = bind_start(*args, ContinuationStart(custody={"thread_id": prior.thread_id, "authority_ceiling": {}}), *tail)
    assert clarified.custody["authority_ceiling"] == {}
    fresh = bind_start(*args, NO_CONTINUATION, *tail)
    assert fresh.custody["authority_ceiling"]["allowed_access"] == "workspace_write"


def test_readonly_plan_heads_advance_without_patch_duties(plan_setup):
    p = plan_setup
    p.snapshot["access"] = "readonly"
    prior = ready_plan(p)
    successor = p.start("Clarify", continue_from=prior.run_id, mode="plan")
    assert successor["status"] == "started", successor
    restored = current.run(p.root, prior.run_id)
    assert restored.superseded_by == successor["run_id"] and not restored.patch_disposed
    p.engine.threads[prior.thread_id]["thread"]["headRunId"] = prior.run_id
    assert p.start("Stale clarification", continue_from=prior.run_id)["reason"] == REFUSAL_SUPERSEDED


def test_authorization_card_names_the_starter_act_after_durable_record(plan_setup):
    p, frames = plan_setup, []
    p.ctx.emit_progress_fn = lambda text, **meta: frames.append((text, meta))
    prior = ready_plan(p)
    assert not frames
    result = p.start("Implement", mode="agent", continue_from=prior.run_id)
    assert result["status"] == "started", result
    assert len(frames) == 1 and "This task authorized plan implementation" in frames[0][0]
    assert frames[0][1] == {"card_row": "timeline", "card_row_id": f'plan-implementation:{result["invocation_id"]}'}
    assert current.run(p.root, result["run_id"]).promotion["authorized_by_task_id"] == p.ctx.task_id


@pytest.mark.parametrize("args,reason", [({"mode": "wrong"}, "mode_invalid"),
    ({"mode": "plan", "override_plan_readiness": True}, "plan_override_conflict"),
    ({"mode": "agent", "override_plan_readiness": "yes"}, "mode_invalid")])
def test_invalid_mode_and_override_are_definite_no_run(plan_setup, args, reason):
    result = plan_setup.start(**args)
    assert result["reason"] == reason and result["definitely_unrun"]
    assert not plan_setup.engine.calls
    assert plan_setup.start(mode="plan")["status"] == "started"


def test_plain_folder_plan_implements_in_place_only_with_directory_support(plan_setup, monkeypatch):
    p = plan_setup
    folder = p.root.parent / (p.root.name + "-folder")
    folder.mkdir()
    p.ctx.workspace_root = str(folder)
    p.ctx.task_constraint = replace(p.ctx.task_constraint, write_root=str(folder))
    geometry = p.start(mode="plan", directory_strategy="copy", scope_paths=["."])
    assert geometry["reason"] == "directory_execution_unavailable"
    assert "choose directory geometry" in geometry["detail"]
    prior = ready_plan(p)
    assert prior.target_root == str(folder) and not prior.snapshot_id
    assert p.engine.calls[0][1]["workspace"] == "in_place"
    refused = p.start(mode="agent", continue_from=prior.run_id)
    assert refused["reason"] == "directory_execution_unavailable" and len(p.engine.calls) == 2
    catalog = p.engine.agent_capabilities()
    monkeypatch.setattr(p.engine, "agent_capabilities", lambda: {**catalog, "mutability": {"workspaceKinds": ["directory"]}})
    copied = p.start(mode="agent", continue_from=prior.run_id, directory_strategy="copy", scope_paths=["."])
    assert copied["reason"] == "same_session_promotion_unsupported" and len(p.engine.calls) == 2
    result = p.start(mode="agent", continue_from=prior.run_id)
    assert result["status"] == "started", result
    request = p.engine.calls[-1][1]
    assert request["execution"] == {"workspaceKind": "directory", "isolation": "live"}
    assert request["planRunId"] == prior.run_id and request["access"] == "workspace_write"
    assert current.run(p.root, result["run_id"]).resource_ref["strategy"] == "direct"
    assert not (folder / ".git").exists()


def test_plan_workspace_is_retained_without_capture_or_apply_duty(plan_setup, monkeypatch):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    p = plan_setup
    prior = ready_plan(p)
    assert capture_terminal_patch_for_drive(p.root, prior, gateway=p.engine) is None
    writable = replace(prior, mode="agent", execution_mode="agent", access="workspace_write",
                       authority_source="acting_constraint", delegated=True)
    assert capture_terminal_patch_for_drive(p.root, writable, gateway=p.engine) is not None
    assert custody.candidate_apply_possible(writable)
    assert not custody.candidate_apply_possible(replace(writable, mode="plan", execution_mode="plan"))
    monkeypatch.setattr(current, "runs", lambda *a, **kw: {prior.run_id: prior})
    assert prior.snapshot_id not in custody.open_snapshot_ids(p.root)
    prior.mode = "agent"
    assert prior.snapshot_id in custody.open_snapshot_ids(p.root)


def test_skill_plan_answer_and_implementation_keep_exact_payload_baseline(plan_setup, monkeypatch):
    from tests.test_delegated_skill_payload import _payload_ctx, _seed_skill
    from ouroboros.delegate_interactions import _delegate_answer

    p = plan_setup
    ctx = _payload_ctx(p.root, monkeypatch)
    _seed_skill(ctx.drive_root)
    p.engine.root = ctx.drive_root
    selector = {"root": "skill_payload", "bucket": "external", "skill_name": "alpha"}
    first = json.loads(exact_start(ctx, "Plan the skill edit", {"snapshot": ctx._payload_subagent_snapshot,
        "mode": "plan", **selector}).text)
    assert first["status"] == "started", first
    rid = first["run_id"]

    def finish(run_id):
        p.engine.runs[run_id]["summary"]["state"] = "succeeded"
        result = delegate_collection.collect_terminal(ctx.drive_root, run_id,
            detail=p.engine.get_run(run_id), gateway=p.engine)
        assert result["status"] == "collected", result

    finish(rid)
    original = current.run(ctx.drive_root, rid)
    assert original.resource_ref["payload_hash"]
    answer = json.loads(_delegate_answer(ctx, rid, answers=[{"question_id": "q1", "free_text": "Edit the skill"}]).text)
    assert answer["status"] == "started", answer
    p.engine.questions = {"parse": "found", "questions": []}
    finish(answer["run_id"])
    result = json.loads(exact_start(ctx, "Implement", {"snapshot": ctx._payload_subagent_snapshot,
        "mode": "agent", "continue_from": answer["run_id"], **selector}).text)
    assert result["status"] == "started", result
    implemented = current.run(ctx.drive_root, result["run_id"])
    assert implemented.resource_ref == original.resource_ref and implemented.snapshot_id == original.snapshot_id
    assert Path(implemented.execution_root).is_dir()
