"""Only the starter's recorded, still-current ceiling authorizes plan implementation."""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from ouroboros import delegate_continuation as continuation, delegate_custody as custody, delegate_custody_current as current
from tests import _delegate_plan_shared

plan_setup = _delegate_plan_shared.plan_setup


def ready_plan(p):
    p.engine.questions = {"parse": "found", "questions": []}
    result = p.start(mode="plan")
    assert result["status"] == "started", result
    return p.finish(result["run_id"])[0]


def test_starter_implements_frozen_plan_with_authorization_before_send(plan_setup):
    p = plan_setup
    prior = ready_plan(p)
    result = p.start("Implement it", continue_from=prior.run_id, mode="agent")
    assert result["status"] == "started", result
    path, request, key = p.engine.calls[-1]
    assert path == f"/v2/threads/{prior.thread_id}/turns"
    assert request["mode"] == "agent" and request["access"] == "workspace_write"
    assert request["planRunId"] == prior.run_id
    assert hashlib.sha256(p.engine.plan).hexdigest() == result["promotion"]["plan_sha256"]
    assert request["overridePlanReadiness"] is False and "continueFrom" not in request
    custody._CUSTODY.clear()
    promoted = current.run(p.root, result["run_id"])
    assert promoted.authority_ceiling == prior.authority_ceiling
    assert promoted.snapshot_id == prior.snapshot_id and promoted.target_root == prior.target_root
    assert current.run(p.root, prior.run_id).superseded_by == promoted.run_id
    assert promoted.promotion == custody.invocation_record(p.root, key)["promotion"] == {
        "from_run_id": prior.run_id, "from_mode": "plan", "to_mode": "agent",
        "from_access": "readonly", "to_access": "workspace_write", "authorized_by_task_id": p.ctx.task_id,
        "authority_fingerprint": prior.authority_fingerprint, "invocation_id": key,
        "plan_sha256": hashlib.sha256(p.engine.plan).hexdigest(), "override_plan_readiness": False}
    assert custody.candidate_apply_possible(promoted)
    assert not custody.candidate_apply_possible(prior)


@pytest.mark.parametrize("readiness", ["needs_answers", "unverified"])
def test_unready_requires_explicit_recorded_override(plan_setup, readiness):
    p = plan_setup
    p.engine.questions["parse"] = "none_found" if readiness == "unverified" else "found"
    result = p.start(mode="plan")
    p.finish(result["run_id"])
    refused = p.start("Implement", continue_from=result["run_id"], mode="agent")
    assert refused["reason"] == "plan_readiness_required", refused
    assert len(p.engine.calls) == 2
    implemented = p.start("Implement", continue_from=result["run_id"], mode="agent", override_plan_readiness=True)
    assert implemented["status"] == "started", implemented
    assert implemented["promotion"]["override_plan_readiness"] is True
    assert p.engine.calls[-1][1]["overridePlanReadiness"] is True


@pytest.mark.parametrize("blocked", ["leaf", "peer", "unrecorded_starter", "legacy", "beyond_ceiling", "changed_authority", "changed_target"])
def test_promotion_floor_refuses_without_ceiling_authority_and_keeps_ordinary_floor(plan_setup, monkeypatch, blocked):
    p = plan_setup
    prior = ready_plan(p)
    actor = {"authority_fingerprint": prior.authority_fingerprint}
    ctx = copy.copy(p.ctx)
    access, target = "workspace_write", prior.target_root
    if blocked in ("leaf", "peer"): ctx.task_id = blocked
    if blocked == "unrecorded_starter": prior.task_id = ctx.task_id = ""
    if blocked == "legacy": prior.authority_ceiling = {}
    if blocked == "beyond_ceiling": access = "full"
    if blocked == "changed_authority": actor["authority_fingerprint"] = "changed"
    if blocked == "changed_target": target += "-other"
    monkeypatch.setattr(current, "run", lambda *a, **kw: prior)
    facts, reason, _detail = continuation.bind_continuation(ctx, p.root, prior.run_id,
        actor=actor, route=SimpleNamespace(route_id=prior.route_id),
        authority=SimpleNamespace(access=access), target_root=target, plan_promotion=True)
    assert not facts and reason == continuation.REFUSAL_AUTHORITY_WIDENED
    assert len(p.engine.calls) == 2
    # Explicit mode selection is the exception; ordinary continuations never widen.
    prior.task_id = p.ctx.task_id
    _facts, reason, _detail = continuation.bind_continuation(p.ctx, p.root, prior.run_id,
        actor={"authority_fingerprint": prior.authority_fingerprint}, route=SimpleNamespace(route_id=prior.route_id),
        authority=SimpleNamespace(access="workspace_write"), target_root=prior.target_root)
    assert reason == continuation.REFUSAL_AUTHORITY_WIDENED


@pytest.mark.parametrize("problem", ["engine", "head", "workspace", "route"])
def test_implementation_contract_and_thread_binding_refuse_before_new_intent(plan_setup, problem):
    p = plan_setup
    prior = ready_plan(p)
    if problem == "engine": p.engine.engine_version = "4.4.9"
    if problem == "head": p.engine.threads[prior.thread_id]["thread"]["headRunId"] = "other-run"
    if problem == "workspace":
        from pathlib import Path
        Path(prior.execution_root).rename(prior.execution_root + ".retired")
    if problem == "route": p.snapshot["route"]["credential_profile_id"] = "another-account"
    before = len(list(custody.custody_rows(p.root)))
    result = p.start("Implement", continue_from=prior.run_id, mode="agent")
    expected = {"engine": "same_session_promotion_unsupported", "head": "plan_thread_head_mismatch",
                "workspace": continuation.REFUSAL_SNAPSHOT_MISSING, "route": "plan_thread_route_mismatch"}[problem]
    assert result["reason"] == expected, result
    assert len(p.engine.calls) == 2
    assert not [r for r in list(custody.custody_rows(p.root))[before:]
                if r["type"] in (custody.START_REQUESTED, custody.STARTED)]


@pytest.mark.parametrize("recover", ["explicit", "background"])
def test_ambiguous_promotion_replays_one_authorized_turn(plan_setup, recover):
    p = plan_setup
    prior = ready_plan(p)
    p.engine.fail_next = True
    unknown = p.start("Implement", continue_from=prior.run_id, mode="agent")
    key = unknown["pending_invocation_id"]
    record = custody.invocation_record(p.root, key)
    assert record["promotion"]["authorized_by_task_id"] == p.ctx.task_id
    original = copy.deepcopy(p.engine.calls[-1])
    custody._CUSTODY.clear()
    if recover == "explicit":
        recovered = p.start("", retry_of=key)
        assert recovered["status"] == "started", recovered
        assert recovered["promotion"] == record["promotion"]
        rid = recovered["run_id"]
    else:
        from ouroboros.delegate_custody_reconcile import _recover_pending_invocation
        pending = next(r for r in custody.pending_invocations(p.root) if r["invocation_id"] == key)
        recovered = _recover_pending_invocation(p.root, p.engine, pending)
        assert recovered["action"] == "left_live", recovered
        rid = recovered["run_id"]
    assert p.engine.calls[-1] == original
    assert len(p.engine.accepted) == 2 and len(p.engine.threads) == 1
    entry = current.run(p.root, rid)
    assert entry.thread_id == prior.thread_id and entry.promotion == record["promotion"]
    assert entry.authority_ceiling == prior.authority_ceiling and entry.execution_mode == "agent"


def test_second_promotion_cannot_race_an_unknown_first_send(plan_setup):
    p = plan_setup
    prior = ready_plan(p)
    p.engine.fail_next = True
    unknown = p.start("Implement", continue_from=prior.run_id, mode="agent")
    assert unknown["pending_invocation_id"]
    # An engine whose head observation is stale must still meet the host's pending handover fence.
    p.engine.threads[prior.thread_id]["thread"]["headRunId"] = prior.run_id
    other = p.start("Another implementation", continue_from=prior.run_id, mode="agent")
    assert other["reason"] == continuation.REFUSAL_SUPERSEDED, other
    assert len(p.engine.accepted) == 2


def test_job_identity_implements_the_thread_head_run(plan_setup):
    p = plan_setup
    prior = ready_plan(p)
    p.engine.runs[prior.run_id]["summary"].update(runId="physical-run", jobId=prior.run_id)
    p.engine.threads[prior.thread_id]["thread"]["headRunId"] = "physical-run"
    result = p.start("Implement", continue_from=prior.run_id, mode="agent")
    assert result["status"] == "started", result
    assert p.engine.calls[-1][1]["planRunId"] == "physical-run"


def test_readonly_ceiling_can_implement_readonly_but_not_gain_write_access(plan_setup):
    p = plan_setup
    p.snapshot["access"] = "readonly"
    prior = ready_plan(p)
    assert not prior.snapshot_id
    p.snapshot["access"] = "workspace_write"
    assert p.start(continue_from=prior.run_id, mode="agent")["reason"] == continuation.REFUSAL_AUTHORITY_WIDENED
    allowed = p.start(continue_from=prior.run_id, mode="agent", access="readonly")
    assert allowed["status"] == "started" and allowed["effective_access"] == "readonly", allowed
