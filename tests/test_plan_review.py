"""Tests for the plan_task (plan_review.py) pre-implementation design review tool.

Tests cover:
- Tool is registered and callable
- Input validation (missing plan, missing goal)
- Budget gate fires when prompt is oversized
- _get_review_models fallback when OUROBOROS_REVIEW_MODELS not set
- _load_plan_checklist returns non-empty text (section exists in CHECKLISTS.md)
- _format_output aggregate signal logic (GREEN / REVIEW_REQUIRED / REVISE_PLAN)
- Output structure: all reviewer sections present
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import queue
import unittest
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_plan_gate_state(kind: str) -> dict:
    fingerprint = "a" * 64
    state = {
        "schema_version": 1,
        "current_attempt": {},
        "latest_review_fingerprint": "",
        "waves": [],
    }
    if kind == "absent":
        return state
    state["current_attempt"] = {
        "fingerprint": fingerprint,
        "status": (
            "unavailable" if kind == "unavailable" else
            "rail_degraded" if kind == "rail_degraded" else
            "open"
        ),
        "reason": (
            "reviewer unavailable" if kind == "unavailable" else
            "plan_task_deadline" if kind == "rail_degraded" else
            ""
        ),
    }
    if kind in {"open", "closed"}:
        outcome = "GREEN" if kind == "closed" else "REVIEW_REQUIRED"
        state["latest_review_fingerprint"] = fingerprint
        state["waves"] = [{
            "request_fingerprint": fingerprint,
            "phase": "reviewed",
            "review_evidence_status": "integrated",
            "review": {
                "aggregate_signal": outcome,
                "closed": kind == "closed",
            },
        }]
    return state


@pytest.mark.parametrize(
    ("kind", "enforcement", "allowed", "status"),
    [
        ("absent", "advisory", False, "absent"),
        ("absent", "blocking", False, "absent"),
        ("open", "advisory", True, "advisory_open"),
        ("open", "blocking", False, "reviewed"),
        ("unavailable", "advisory", True, "advisory_open"),
        ("unavailable", "blocking", False, "unavailable"),
        ("rail_degraded", "advisory", True, "rail_degraded"),
        ("rail_degraded", "blocking", True, "rail_degraded"),
        ("closed", "advisory", True, "closed"),
        ("closed", "blocking", True, "closed"),
    ],
)
def test_force_plan_gate_projection_matrix(kind, enforcement, allowed, status):
    from ouroboros.task_results import plan_review_gate_projection

    decision = plan_review_gate_projection(_force_plan_gate_state(kind), enforcement)

    assert decision["allow"] is allowed
    assert decision["status"] == status


@pytest.mark.parametrize("state", [
    None,
    _force_plan_gate_state("absent"),
    _force_plan_gate_state("open"),
    _force_plan_gate_state("unavailable"),
])
def test_force_plan_gate_real_hard_rail_always_releases(state):
    from ouroboros.task_results import plan_review_gate_projection

    decision = plan_review_gate_projection(state, "blocking", hard_rail="round_limit")

    assert decision["allow"] is True
    assert decision["status"] == "rail_degraded"
    assert decision["reason"] == "round_limit"


def test_new_canonical_attempt_does_not_fall_back_to_old_closed_review():
    from ouroboros.task_results import plan_review_gate_projection

    state = _force_plan_gate_state("closed")
    state["current_attempt"] = {
        "fingerprint": "b" * 64,
        "status": "open",
        "reason": "",
    }

    decision = plan_review_gate_projection(state, "blocking")

    assert decision["allow"] is False
    assert decision["status"] == "open"


def test_closed_plan_review_wave_resolution():
    from ouroboros.task_results import closed_plan_review_wave

    closed = _force_plan_gate_state("closed")
    closed["waves"][0]["acceptance_claims"] = ["game boots", "score persists"]
    wave = closed_plan_review_wave(closed)
    assert wave is not None
    assert wave["acceptance_claims"] == ["game boots", "score persists"]

    # Open (non-closed) review yields no authority.
    assert closed_plan_review_wave(_force_plan_gate_state("open")) is None
    assert closed_plan_review_wave(_force_plan_gate_state("absent")) is None
    assert closed_plan_review_wave(None) is None

    # A newer canonical attempt supersedes the old closed wave (no stale-GREEN revival).
    superseded = _force_plan_gate_state("closed")
    superseded["current_attempt"] = {"fingerprint": "b" * 64, "status": "open", "reason": ""}
    assert closed_plan_review_wave(superseded) is None

    # Disposition-closed REVIEW_REQUIRED counts as closed authority too.
    disposed = _force_plan_gate_state("closed")
    disposed["waves"][0]["review"]["aggregate_signal"] = "REVIEW_REQUIRED"
    assert closed_plan_review_wave(disposed) is not None

    # Pending evidence is not integrated authority.
    pending = _force_plan_gate_state("closed")
    pending["waves"][0]["review_evidence_status"] = "pending"
    assert closed_plan_review_wave(pending) is None


def test_wave_freezes_bounded_acceptance_claims(tmp_path):
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        reserve_plan_review_wave,
        write_task_result,
    )

    write_task_result(tmp_path, "root1", STATUS_RUNNING, result="running")
    fingerprint = "c" * 64
    long_claim = "x" * 2_000
    wave, created = reserve_plan_review_wave(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        plan_text_hash="d" * 64,
        scout_roles=[],
        cutoff_at="2026-08-08T00:00:00+00:00",
        acceptance_claims=["game boots", "  ", long_claim] + [f"claim {i}" for i in range(30)],
    )
    assert created
    stored = wave["acceptance_claims"]
    assert stored[0] == "game boots"
    assert "OMISSION NOTE" in stored[1]  # long claim bounded with a disclosed marker
    assert len(stored) == 24
    assert wave["acceptance_claims_omitted"] == 8  # 32 non-blank - 24 cap
    # The persisted state round-trips through the validator.
    state = load_plan_review_state(tmp_path, "root1")
    assert state["waves"][0]["acceptance_claims"] == stored

    # Vacuous claims stay ABSENT on the wave (only-when-set).
    wave2, _ = reserve_plan_review_wave(
        tmp_path,
        "root1",
        fingerprint="e" * 64,
        plan_text_hash="d" * 64,
        scout_roles=[],
        cutoff_at="2026-08-08T00:00:00+00:00",
        acceptance_claims=["", "   "],
    )
    assert "acceptance_claims" not in wave2
    assert "acceptance_claims_omitted" not in wave2


def test_plan_review_state_validator_rejects_malformed_wave_claims(tmp_path):
    from ouroboros.task_results import _validated_plan_review_state

    def _state(**wave_extra):
        return {
            "schema_version": 1,
            "current_attempt": {},
            "latest_review_fingerprint": "",
            "waves": [{
                "request_fingerprint": "a" * 64,
                "plan_text_hash": "b" * 64,
                "created_at": "2026-08-08T00:00:00+00:00",
                "scout_cutoff_at": "2026-08-08T00:00:00+00:00",
                "phase": "scheduling",
                "intended_scouts": [],
                "included_task_ids": [],
                "omissions": [],
                "consumed_task_ids": [],
                "disposition_warnings": [],
                **wave_extra,
            }],
        }

    assert _validated_plan_review_state(_state(acceptance_claims=["ok"]))
    for bad in ([], [""], ["x" * 900], "not-a-list", ["ok"] * 25):
        with pytest.raises(ValueError):
            _validated_plan_review_state(_state(acceptance_claims=bad))
    with pytest.raises(ValueError):
        _validated_plan_review_state(_state(acceptance_claims=["ok"], acceptance_claims_omitted=-1))
    with pytest.raises(ValueError):
        _validated_plan_review_state(_state(acceptance_claims=["ok"], acceptance_claims_omitted=True))


def test_invalid_new_plan_attempts_do_not_reuse_old_green(tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        plan_review_gate_projection,
        record_plan_review_attempt,
        record_plan_review_collection,
        record_plan_review_result,
        reserve_plan_review_wave,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    write_task_result(tmp_path, "root1", STATUS_RUNNING, result="running")
    old_fingerprint = "a" * 64
    record_plan_review_attempt(tmp_path, "root1", fingerprint=old_fingerprint)
    reserve_plan_review_wave(
        tmp_path,
        "root1",
        fingerprint=old_fingerprint,
        plan_text_hash=pr.plan_text_fingerprint("old plan"),
        scout_roles=[],
        cutoff_at="2026-08-03T00:00:00+00:00",
    )
    record_plan_review_collection(
        tmp_path,
        "root1",
        fingerprint=old_fingerprint,
        included_task_ids=[],
        omissions=[],
        stop_reason="complete",
    )
    record_plan_review_result(
        tmp_path,
        "root1",
        fingerprint=old_fingerprint,
        review={
            "schema_version": 1,
            "request_fingerprint": old_fingerprint,
            "plan_text_hash": pr.plan_text_fingerprint("old plan"),
            "aggregate_signal": "GREEN",
            "findings": [],
            "closed": True,
            "reviewer_slots_degraded": False,
        },
        reviewed_result_hashes={},
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root1"
    ctx.system_repo_dir = tmp_path
    ctx.emit_progress_fn = lambda _message: None
    cases = [
        (_plan_request("new invalid plan", "G", files_to_touch=["../escape.py"], context_level="minimal"),
         "ERROR: PLAN_SUBJECT_ROOT_INVALID:"),
        (_plan_request("", "G", context_level="minimal"), "ERROR: plan parameter is required"),
        (_plan_request("P", "G", context_level="minimal", scope="not-an-object"),
         "ERROR: PLAN_SCOPE_INVALID:"),
    ]
    for new_request, error_prefix in cases:
        record_plan_review_attempt(tmp_path, "root1", fingerprint=old_fingerprint)
        output = pr._handle_plan_task(ctx, **vars(new_request))
        assert output.startswith(error_prefix)
        state = load_plan_review_state(tmp_path, "root1")
        assert state["current_attempt"]["fingerprint"] != old_fingerprint
        decision = plan_review_gate_projection(state, "blocking")
        assert decision["allow"] is False
        assert decision["status"] == "open"

    record_plan_review_attempt(tmp_path, "root1", fingerprint=old_fingerprint)
    with pytest.raises(TypeError):
        pr._handle_plan_task(ctx, plan="P", goal="G", files_to_touch=123)
    state = load_plan_review_state(tmp_path, "root1")
    assert state["current_attempt"]["fingerprint"] != old_fingerprint
    assert plan_review_gate_projection(state, "blocking")["allow"] is False


def test_rail_degraded_attempt_overrides_same_fingerprint_open_review():
    from ouroboros.task_results import plan_review_gate_projection

    state = _force_plan_gate_state("open")
    state["current_attempt"].update({
        "status": "rail_degraded",
        "reason": "plan_task_deadline",
    })

    decision = plan_review_gate_projection(state, "blocking")

    assert decision["allow"] is True
    assert decision["status"] == "rail_degraded"
    assert decision["outcome"] == "REVIEW_REQUIRED"
    assert decision["reason"] == "plan_task_deadline"


def test_current_plan_attempt_survives_reload(tmp_path):
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        record_plan_review_attempt,
        write_task_result,
    )

    write_task_result(tmp_path, "root1", STATUS_RUNNING, result="running")
    record_plan_review_attempt(tmp_path, "root1", fingerprint="c" * 64)

    state = load_plan_review_state(tmp_path, "root1")
    assert state["current_attempt"] == {
        "fingerprint": "c" * 64,
        "status": "open",
        "reason": "",
    }


def test_reviewer_unavailability_stays_retryable_not_durable_verdict(tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        plan_review_wave_handoffs,
        record_plan_review_collection,
        record_plan_review_attempt,
        record_plan_review_scout,
        reserve_plan_review_wave,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    request = _plan_request("P", "G", context_level="minimal")
    fingerprint = pr._plan_request_fingerprint(
        plan=request.plan,
        goal=request.goal,
        files_to_touch=request.files_to_touch,
        context_level=request.context_level,
        context_notes=request.context_notes,
        plan_class=request.plan_class,
        scope=request.scope,
        include_tests=request.include_tests,
    )
    write_task_result(tmp_path, "root1", STATUS_RUNNING, result="running")
    record_plan_review_attempt(tmp_path, "root1", fingerprint=fingerprint)
    wave, _ = reserve_plan_review_wave(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        plan_text_hash=pr.plan_text_fingerprint(request.plan),
        scout_roles=["planning-scout-1"],
        cutoff_at="2026-08-03T00:00:00+00:00",
    )
    wave = record_plan_review_scout(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        role="planning-scout-1",
        schedule_status="started",
        task_ids=["scout1"],
        reason="scheduled scout1",
    )
    write_task_result(
        tmp_path,
        "scout1",
        "completed",
        parent_task_id="root1",
        root_task_id="root1",
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: original reviewer input",
    )
    wave = record_plan_review_collection(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        included_task_ids=["scout1"],
        omissions=[],
        stop_reason="complete",
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root1"

    planning_handoffs = plan_review_wave_handoffs(wave)
    planning_handoffs["wait"] = {
        "tasks": {
            "scout1": {
                "status": "completed",
                "role": "planning-scout-1",
                "result": "summary: original reviewer input",
            },
        },
        "all_terminal": True,
    }
    pr._persist_planning_handoffs(ctx, planning_handoffs)
    reviewed_result_hashes = pr._reviewed_handoff_hashes(planning_handoffs)
    assert "error" not in pr._persist_planning_snapshot(ctx, planning_handoffs)
    output = pr._finalize_plan_review_output(ctx, pr._PlanReviewFinalization(
        request=request,
        raw_results=[{"model": "m", "text": "", "error": "timeout"}],
        models=["m"],
        estimated_tokens=1,
        subject_repo=tmp_path,
        governance_repo=tmp_path,
        planning_handoffs=planning_handoffs,
        state_root=tmp_path,
        state_task_id="root1",
        request_fingerprint=fingerprint,
        degraded_scout_note="",
        reviewed_result_hashes=reviewed_result_hashes,
    ))

    state = load_plan_review_state(tmp_path, "root1")
    assert state["current_attempt"]["status"] == "unavailable"
    assert state["waves"][0]["review"]["closed"] is False
    assert state["waves"][0]["review"]["reviewer_slots_degraded"] is True
    assert state["waves"][0]["reviewed_result_hashes"] == reviewed_result_hashes
    assert "not a durable review verdict" in output
    assert pr._reuse_or_disposition_plan_review(
        ctx,
        fingerprint,
        None,
        pr.plan_text_fingerprint(request.plan),
    ) is None
    other_handoffs = {
        "schema_version": 1,
        "request_fingerprint": "b" * 64,
        "task_ids": [],
        "included_task_ids": [],
        "wait": {"tasks": {}, "all_terminal": True},
    }
    pr._persist_planning_handoffs(ctx, other_handoffs)
    assert "error" not in pr._persist_planning_snapshot(ctx, other_handoffs)
    assert json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))[
        "request_fingerprint"
    ] == "b" * 64
    write_task_result(
        tmp_path,
        "scout1",
        "completed",
        parent_task_id="root1",
        root_task_id="root1",
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: rewritten after reviewer outage",
    )
    retry = pr._start_planning_swarm(ctx, request, fingerprint)
    assert retry["started"] is True and retry["resumed"] is True
    assert (
        retry["handoffs"]["wait"]["tasks"]["scout1"]["result"]
        == "summary: original reviewer input"
    )
    resumed = pr._start_planning_swarm(ctx, request, fingerprint)
    assert resumed["started"] is True
    assert resumed["resumed"] is True
    assert len(load_plan_review_state(tmp_path, "root1")["waves"]) == 1


def test_reviewer_unavailability_retries_zero_included_snapshot(tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        plan_review_wave_handoffs,
        record_plan_review_collection,
        record_plan_review_attempt,
        reserve_plan_review_wave,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    request = _plan_request("P", "G", context_level="minimal")
    fingerprint = pr._plan_request_fingerprint(
        plan=request.plan,
        goal=request.goal,
        files_to_touch=request.files_to_touch,
        context_level=request.context_level,
        context_notes=request.context_notes,
        plan_class=request.plan_class,
        scope=request.scope,
        include_tests=request.include_tests,
    )
    write_task_result(tmp_path, "root-zero", STATUS_RUNNING, result="running")
    record_plan_review_attempt(tmp_path, "root-zero", fingerprint=fingerprint)
    reserve_plan_review_wave(
        tmp_path,
        "root-zero",
        fingerprint=fingerprint,
        plan_text_hash=pr.plan_text_fingerprint(request.plan),
        scout_roles=[],
        cutoff_at="2026-08-03T00:00:00+00:00",
    )
    wave = record_plan_review_collection(
        tmp_path,
        "root-zero",
        fingerprint=fingerprint,
        included_task_ids=[],
        omissions=[],
        stop_reason="complete",
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root-zero"
    handoffs = plan_review_wave_handoffs(wave)
    handoffs["wait"] = {"tasks": {}, "all_terminal": True}
    pr._persist_planning_handoffs(ctx, handoffs)
    assert "error" not in pr._persist_planning_snapshot(ctx, handoffs)

    pr._finalize_plan_review_output(ctx, pr._PlanReviewFinalization(
        request=request,
        raw_results=[{"model": "m", "text": "", "error": "timeout"}],
        models=["m"],
        estimated_tokens=1,
        subject_repo=tmp_path,
        governance_repo=tmp_path,
        planning_handoffs=handoffs,
        state_root=tmp_path,
        state_task_id="root-zero",
        request_fingerprint=fingerprint,
        degraded_scout_note="",
        reviewed_result_hashes={},
    ))

    state = load_plan_review_state(tmp_path, "root-zero")
    assert state["current_attempt"]["status"] == "unavailable"
    retry = pr._start_planning_swarm(ctx, request, fingerprint)
    assert retry["started"] is True
    assert retry["resumed"] is True
    assert retry["handoffs"]["included_task_ids"] == []
    assert retry["handoffs"]["wait"] == {"tasks": {}, "all_terminal": True}
    assert "error" not in retry["handoffs"]["artifact"]


def test_exception_after_snapshot_retries_exact_frozen_evidence(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import write_task_result

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-post-freeze-exception"
    request = _plan_request("P", "G", context_level="minimal")
    real_start = pr._start_planning_swarm
    starts = 0

    def first_then_real(ctx_arg, request_arg, fingerprint):
        nonlocal starts
        starts += 1
        if starts == 1:
            return _completed_planning_swarm(ctx_arg, request_arg, fingerprint)
        return real_start(ctx_arg, request_arg, fingerprint)

    panel_inputs = []

    async def fail_then_pass(_ctx, models, _system_prompt, user_content, user_stable_len=0, slot_ids=None):
        panel_inputs.append(user_content)
        if len(panel_inputs) == 1:
            raise RuntimeError("injected post-freeze reviewer failure")
        return [{
            "model": model,
            "text": _review_text("GREEN"),
            "error": None,
        } for model in models]

    monkeypatch.setattr(pr, "_start_planning_swarm", first_then_real)
    monkeypatch.setattr(pr, "_run_plan_review_slots", fail_then_pass)
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: "")
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setattr("ouroboros.config.get_review_models", lambda: ["m1", "m2"])

    with pytest.raises(RuntimeError, match="post-freeze reviewer failure"):
        asyncio.run(pr._run_plan_review_async(ctx, request))
    assert "error" not in pr._plan_unavailable(ctx, "review failed", "review_failed").lower()
    write_task_result(
        tmp_path,
        "scout1",
        "completed",
        parent_task_id=ctx.task_id,
        root_task_id=ctx.task_id,
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: changed after exception",
    )

    output = asyncio.run(pr._run_plan_review_async(ctx, request))

    assert "PLAN_REVIEW_OUTCOME: GREEN" in output
    assert "summary: ok" in panel_inputs[1]
    assert "summary: changed after exception" not in panel_inputs[1]


def test_green_with_addressable_findings_is_substantive_review_required():
    import ouroboros.tools.plan_review as pr

    summary = pr._summarize_plan_review_results([{
        "model": "m",
        "text": _review_text("GREEN", [{
            "id": "real-risk",
            "level": "RISK",
            "summary": "The plan misses a real seam.",
            "recommendation": "Use the existing reducer.",
        }]),
        "error": None,
    }])

    assert summary["signals"] == ["REVIEW_REQUIRED"]
    assert summary["aggregate_signal"] == "REVIEW_REQUIRED"
    assert summary["degraded_count"] == 0


def test_substantive_finding_survives_peer_outage_and_retries_same_wave(tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        STATUS_RUNNING,
        load_plan_review_state,
        plan_review_wave_handoffs,
        record_plan_review_attempt,
        record_plan_review_collection,
        record_plan_review_scout,
        reserve_plan_review_wave,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    request = _plan_request("P", "G", context_level="minimal")
    fingerprint = pr._plan_request_fingerprint(
        plan=request.plan,
        goal=request.goal,
        files_to_touch=request.files_to_touch,
        context_level=request.context_level,
        context_notes=request.context_notes,
        plan_class=request.plan_class,
        scope=request.scope,
        include_tests=request.include_tests,
    )
    write_task_result(tmp_path, "root1", STATUS_RUNNING, result="running")
    record_plan_review_attempt(tmp_path, "root1", fingerprint=fingerprint)
    wave, _ = reserve_plan_review_wave(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        plan_text_hash=pr.plan_text_fingerprint(request.plan),
        scout_roles=["planning-scout-1"],
        cutoff_at="2026-08-03T00:00:00+00:00",
    )
    wave = record_plan_review_scout(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        role="planning-scout-1",
        schedule_status="started",
        task_ids=["scout1"],
        reason="scheduled scout1",
    )
    write_task_result(
        tmp_path,
        "scout1",
        "completed",
        parent_task_id="root1",
        root_task_id="root1",
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: use the existing reducer",
    )
    wave = record_plan_review_collection(
        tmp_path,
        "root1",
        fingerprint=fingerprint,
        included_task_ids=["scout1"],
        omissions=[],
        stop_reason="complete",
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root1"
    planning_handoffs = plan_review_wave_handoffs(wave)
    planning_handoffs["wait"] = {
        "tasks": {
            "scout1": {
                "status": "completed",
                "role": "planning-scout-1",
                "result": "summary: use the existing reducer",
            },
        },
        "all_terminal": True,
    }
    reviewed_result_hashes = pr._reviewed_handoff_hashes(planning_handoffs)
    assert "error" not in pr._persist_planning_snapshot(ctx, planning_handoffs)

    output = pr._finalize_plan_review_output(ctx, pr._PlanReviewFinalization(
        request=request,
        raw_results=[
            {
                "model": "m1",
                "text": _review_text("REVIEW_REQUIRED", [{
                    "id": "real-risk",
                    "level": "RISK",
                    "summary": "The plan misses a real seam.",
                    "recommendation": "Use the existing reducer.",
                }]),
                "error": None,
            },
            {"model": "m2", "text": "", "error": "timeout"},
        ],
        models=["m1", "m2"],
        estimated_tokens=1,
        subject_repo=tmp_path,
        governance_repo=tmp_path,
        planning_handoffs=planning_handoffs,
        state_root=tmp_path,
        state_task_id="root1",
        request_fingerprint=fingerprint,
        degraded_scout_note="",
        reviewed_result_hashes=reviewed_result_hashes,
    ))

    state = load_plan_review_state(tmp_path, "root1")
    review = state["waves"][0]["review"]
    assert review["reviewer_slots_degraded"] is True
    assert any(item["finding_id"].endswith(":real-risk") for item in review["findings"])
    assert "substantive findings were stored" in output
    assert "not a durable review verdict" not in output
    from ouroboros.task_results import plan_review_gate_projection
    from ouroboros.loop import _force_plan_reminder

    decision = plan_review_gate_projection(state, "blocking")
    assert decision["reviewer_slots_degraded"] is True
    assert "no review_disposition" in _force_plan_reminder(decision)
    assert pr._reuse_or_disposition_plan_review(
        ctx,
        fingerprint,
        None,
        pr.plan_text_fingerprint(request.plan),
    ) is None
    write_task_result(
        tmp_path,
        "scout1",
        "completed",
        parent_task_id="root1",
        root_task_id="root1",
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: a later process rewrote this completed result",
    )
    retry = pr._start_planning_swarm(ctx, request, fingerprint)
    assert retry["started"] is True and retry["resumed"] is True
    assert (
        retry["handoffs"]["wait"]["tasks"]["scout1"]["result"]
        == "summary: use the existing reducer"
    )
    assert pr._reviewed_handoff_hashes(retry["handoffs"]) == reviewed_result_hashes
    assert len(load_plan_review_state(tmp_path, "root1")["waves"]) == 1
    rejected = pr._reuse_or_disposition_plan_review(
        ctx,
        fingerprint,
        {"review_fingerprint": fingerprint, "items": []},
        pr.plan_text_fingerprint(request.plan),
    )
    assert rejected.startswith("ERROR: PLAN_REVIEW_RETRY_REQUIRED")

    recovered = pr._finalize_plan_review_output(ctx, pr._PlanReviewFinalization(
        request=request,
        raw_results=[
            {"model": "m1", "text": _review_text("GREEN", []), "error": None},
            {"model": "m2", "text": _review_text("GREEN", []), "error": None},
        ],
        models=["m1", "m2"],
        estimated_tokens=1,
        subject_repo=tmp_path,
        governance_repo=tmp_path,
        planning_handoffs=retry["handoffs"],
        state_root=tmp_path,
        state_task_id="root1",
        request_fingerprint=fingerprint,
        degraded_scout_note="",
        reviewed_result_hashes=pr._reviewed_handoff_hashes(retry["handoffs"]),
    ))

    state = load_plan_review_state(tmp_path, "root1")
    wave = state["waves"][0]
    assert "PLAN_REVIEW_OUTCOME: GREEN" in recovered
    assert wave["review"]["closed"] is True
    assert wave["review_evidence_status"] == "integrated"
    assert wave["consumed_task_ids"] == ["scout1"]


def _make_ctx(tmp_path: pathlib.Path | None = None) -> MagicMock:
    import tempfile

    repo_root = tmp_path or pathlib.Path(".")
    tempdir = tempfile.TemporaryDirectory() if tmp_path is None else None
    drive_root = tmp_path or pathlib.Path(tempdir.name)
    ctx = MagicMock()
    ctx.repo_dir = repo_root
    ctx.drive_root = drive_root
    ctx.budget_drive_root = str(drive_root)
    ctx.task_id = "plan-review-test"
    ctx.task_metadata = {}
    ctx.task_contract = {}
    ctx.project_id = ""
    ctx.drive_logs.return_value = drive_root / "logs"
    ctx.emit_progress_fn = MagicMock()
    ctx._test_tempdir = tempdir
    return ctx


def _review_text(signal: str, findings: list[dict] | None = None) -> str:
    return (
        "## PROPOSALS\n\nConcrete review.\n\n"
        "PLAN_FINDINGS_JSON:\n"
        + json.dumps(findings if findings is not None else [], ensure_ascii=False)
        + f"\nAGGREGATE: {signal}"
    )


def _plan_request(
    plan: str,
    goal: str,
    files_to_touch: list | None = None,
    **kwargs,
):
    from ouroboros.tools.plan_review import _PlanReviewRequest

    return _PlanReviewRequest(
        plan=plan,
        goal=goal,
        files_to_touch=list(files_to_touch or []),
        **kwargs,
    )


def test_planning_state_requires_real_task_id(tmp_path):
    import pytest

    from ouroboros.tools.plan_review import _planning_state_location
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    with pytest.raises(ValueError, match="PLAN_REVIEW_TASK_ID_REQUIRED"):
        _planning_state_location(ctx)
    assert not (tmp_path / "task_results" / "plan_review.json").exists()


def test_disposition_without_task_id_returns_typed_state_error(tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    result = pr._handle_plan_task(
        ctx,
        review_disposition={"review_fingerprint": "a" * 64, "items": []},
    )

    assert result.startswith("ERROR: PLAN_REVIEW_STATE_INVALID:")
    assert "PLAN_REVIEW_TASK_ID_REQUIRED" in result
    assert not (tmp_path / "task_results").exists()


def test_malformed_reviewer_slots_block_plan_review_before_any_dispatch(tmp_path, monkeypatch):
    """#116: a malformed OUROBOROS_REVIEWER_SLOTS must refuse plan review loudly
    (typed retryable unavailability, precise parse error in the message) BEFORE
    any scout or reviewer dispatch — never run the panel on the silently
    projected default models."""
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", "{broken")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "plan-slot-config"
    with (
        patch.object(pr, "_get_review_models",
                     side_effect=AssertionError("no reviewer dispatch")),
        patch.object(pr, "_start_planning_swarm",
                     side_effect=AssertionError("no scout dispatch")),
    ):
        result = pr._handle_plan_task(ctx, plan="P", goal="G", context_level="minimal")

    assert "Invalid reviewer-slot configuration blocks plan review" in result
    assert "not valid JSON" in result
    # The pointer names a place that EXISTS: D-10 moved these rows out of the
    # Models tab and renamed the section, so the old wording sent the owner
    # looking for a heading no tab carries any more.
    assert "Fix Review lanes on the Agents tab in Settings" in result


def _contract_review_text(signal: str) -> str:
    findings = [] if signal == "GREEN" else [{
        "id": "material-finding",
        "level": "FAIL" if signal == "REVISE_PLAN" else "RISK",
        "summary": "A material planning issue needs disposition.",
        "recommendation": "Address the concrete planning issue.",
    }]
    return _review_text(signal, findings)


def _start_swarm(ctx, request):
    """Test seam: production computes the BINDING fingerprint (agent-passed values
    only) before calling the swarm, so the wave is keyed by an identity the agent can
    reproduce. Mirrors that call shape for the direct unit tests."""
    from ouroboros.tools.plan_review import _plan_request_fingerprint, _start_planning_swarm

    return _start_planning_swarm(ctx, request, _plan_request_fingerprint(
        plan=request.plan, goal=request.goal, files_to_touch=request.files_to_touch,
        context_level=request.context_level, context_notes=request.context_notes,
        plan_class=request.plan_class, scope=request.scope,
        include_tests=request.include_tests,
    ))

def test_planning_swarm_cutoff_omissions_reach_panel(monkeypatch, tmp_path):
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "0")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "0.25")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "parent1", "session_id": "sess1"}

    result = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="focused"),
    )

    assert result["started"] is True
    assert result["degraded_evidence"] is True
    assert result["task_ids"]
    assert result["handoffs"]["omissions"][0]["reason"].startswith(
        "not_terminal_at_review_cutoff:"
    )
    assert (tmp_path / "task_results" / "artifacts" / "parent1" / "plan_task_handoffs.json").exists()


def test_planning_swarm_resumes_existing_handoffs_without_rescheduling(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "0")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "0.25")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "parent1", "session_id": "sess1"}
    scheduled = {"count": 0}

    def fake_schedule(ctx_arg, _internal=None, **kwargs):
        scheduled["count"] += 1
        ctx_arg._last_scheduled_subagents = [{"task_ids": ["scout-resume"]}]
        return "scheduled scout-resume"

    wait_results = [
        {"timed_out": True, "tasks": {"scout-resume": {"status": "running", "result": ""}}},
        {"timed_out": True, "tasks": {"scout-resume": {"status": "running", "result": ""}}},
        {"timed_out": False, "tasks": {"scout-resume": {"status": "completed", "role": "planning-scout-1", "result": "summary: resumed"}}},
    ]

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)

    def fake_wait(*_args, **kwargs):
        result = wait_results.pop(0)
        if result.get("timed_out"):
            __import__("time").sleep(float(kwargs.get("timeout_sec") or 0))
        return result

    monkeypatch.setattr(pr, "wait_for_effective_tasks", fake_wait)

    first = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="minimal"),
    )
    second = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="minimal"),
    )

    assert first["started"] is True
    assert first["degraded_evidence"] is True
    assert second["started"] is True
    assert second["resumed"] is True
    assert scheduled["count"] == 1


def test_planning_swarm_persists_wave_before_wait(monkeypatch, tmp_path):
    import pytest

    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-persist-before-wait"
    ctx.event_queue = queue.Queue()
    scheduled = {"count": 0}
    collected = {"count": 0}

    def fake_schedule(ctx_arg, _internal=None, **_kwargs):
        scheduled["count"] += 1
        ctx_arg._last_scheduled_subagents = [{"task_ids": ["scout-durable"]}]
        return "scheduled scout-durable"

    def fake_collect(ctx_arg, **kwargs):
        from ouroboros.task_results import load_plan_review_state, plan_review_wave

        collected["count"] += 1
        state = load_plan_review_state(tmp_path, str(ctx_arg.task_id))
        stored = plan_review_wave(state, kwargs["fingerprint"])
        assert stored is not None
        assert stored["intended_scouts"][0]["task_ids"] == ["scout-durable"]
        assert stored["scout_cutoff_at"]
        if collected["count"] == 1:
            raise RuntimeError("simulated wait crash")
        return {
            "schema_version": 1,
            "request_fingerprint": kwargs["fingerprint"],
            "task_ids": ["scout-durable"],
            "schedule_outputs": ["scheduled scout-durable"],
            "wait": {"tasks": {"scout-durable": {
                "status": "completed", "role": "planning-scout-1", "result": "ready",
            }}},
            "included_task_ids": ["scout-durable"],
            "omissions": [],
            "artifact": {"path": str(pr._planning_handoff_path(ctx_arg))},
        }

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)
    monkeypatch.setattr(pr, "_collect_planning_handoffs", fake_collect)
    request = _plan_request("Do the work", "Ship a fix", context_level="minimal")
    with pytest.raises(RuntimeError, match="simulated wait crash"):
        _start_swarm(ctx, request)
    resumed = _start_swarm(ctx, request)

    assert resumed["started"] is True
    assert resumed["resumed"] is True
    assert scheduled["count"] == 1


def test_planning_scout_recovers_durable_id_when_side_channel_is_missing(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    from ouroboros.task_results import load_plan_review_state, write_task_result
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "0.25")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-durable-id"
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    # A POSITIVE consumable window: with max_wait 0 the wave now refuses to launch scouts at
    # all (v6.79.0 scout-window admission), which is a different path from id recovery.

    def fake_schedule(_ctx, _internal=None, **kwargs):
        write_task_result(
            tmp_path,
            "scout-issued",
            "requested",
            parent_task_id=ctx.task_id,
            root_task_id=ctx.task_id,
            delegation_role="subagent",
            role=kwargs["role"],
            result="queued before side-channel update",
        )
        return "scheduled without side channel"

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)
    result = _start_swarm(
        ctx,
        _plan_request("P", "G", context_level="minimal"),
    )

    assert result["started"] is True
    assert result["task_ids"] == ["scout-issued"]
    attempt = load_plan_review_state(tmp_path, ctx.task_id)["waves"][0]["intended_scouts"][0]
    assert attempt["schedule_status"] == "started"
    assert attempt["task_ids"] == ["scout-issued"]


def test_resumed_pending_scout_recovers_previously_issued_durable_id(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        load_plan_review_state,
        reserve_plan_review_wave,
        write_task_result,
    )
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-resume-issued"
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    # v6.80.0: the binding fingerprint uses the AGENT-passed plan_class (empty here,
    # so it is omitted from the payload) — not the host-resolved "self_mod".
    fingerprint = pr._plan_request_fingerprint(
        plan="P",
        goal="G",
        files_to_touch=[],
        context_level="minimal",
        context_notes="",
        plan_class="",
        scope=None,
        include_tests=False,
    )
    reserve_plan_review_wave(
        tmp_path,
        ctx.task_id,
        fingerprint=fingerprint,
        plan_text_hash=pr.plan_text_fingerprint("P"),
        scout_roles=["planning-scout-1"],
        cutoff_at="2000-01-01T00:00:00+00:00",
    )
    write_task_result(
        tmp_path,
        "scout-before-crash",
        "requested",
        parent_task_id=ctx.task_id,
        root_task_id=ctx.task_id,
        delegation_role="subagent",
        role="planning-scout-1",
        result="durable request",
    )
    monkeypatch.setattr(
        control,
        "_schedule_task",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not reschedule")),
    )

    result = _start_swarm(
        ctx,
        _plan_request("P", "G", context_level="minimal"),
    )

    assert result["started"] is True
    assert result["resumed"] is True
    assert result["task_ids"] == ["scout-before-crash"]
    attempt = load_plan_review_state(tmp_path, ctx.task_id)["waves"][0]["intended_scouts"][0]
    assert attempt["schedule_status"] == "started"
    assert "recovered durable issued child id" in attempt["schedule_reason"]


def test_corrupt_parent_task_result_fails_closed_without_overwrite(tmp_path):
    import pytest

    from ouroboros.task_results import load_plan_review_state, reserve_plan_review_wave

    result_path = tmp_path / "task_results" / "parent-corrupt.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="PLAN_REVIEW_STATE_INVALID"):
        load_plan_review_state(tmp_path, "parent-corrupt")
    with pytest.raises(ValueError, match="PLAN_REVIEW_STATE_INVALID"):
        reserve_plan_review_wave(
            tmp_path,
            "parent-corrupt",
            fingerprint="f" * 64,
            plan_text_hash="a" * 64,
            scout_roles=[],
            cutoff_at="2099-01-01T00:00:00+00:00",
        )

    assert result_path.read_text(encoding="utf-8") == "{broken"


def test_expired_explicit_deadline_skips_before_scout_or_reviewer(monkeypatch, tmp_path):
    import asyncio
    from datetime import datetime, timedelta, timezone

    import ouroboros.tools.plan_review as pr
    import ouroboros.loop as loop_mod
    from ouroboros.task_results import load_plan_review_state

    ctx = _make_ctx(tmp_path)
    ctx.task_metadata = {
        "deadline_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        "force_plan": True,
    }
    ctx.is_ephemeral_turn = False
    monkeypatch.setattr(
        pr,
        "_start_planning_swarm",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("expired deadline must skip")),
    )

    out = asyncio.run(pr._run_plan_review_async(
        ctx,
        _plan_request(
            plan="P",
            goal="G",
            files_to_touch=[],
            context_level="minimal",
            plan_class="external",
        ),
    ))

    assert out.startswith("PLAN_TASK_SKIPPED_DEADLINE: the task deadline has expired")
    assert pr._planning_swarm_timing(ctx)[1] == 0.0
    attempt = load_plan_review_state(tmp_path, ctx.task_id)["current_attempt"]
    assert attempt["status"] == "rail_degraded"
    assert attempt["reason"] == "plan_task_deadline"

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    decision = loop_mod._force_plan_decision(ctx, {})
    assert decision["allow"] is True
    assert decision["status"] == "rail_degraded"
    assert "plan_task_deadline" in loop_mod._force_plan_disclosure(ctx, {}, forced_reason="")


def test_expired_deadline_releases_existing_open_wave(monkeypatch, tmp_path):
    import asyncio
    from datetime import datetime, timedelta, timezone

    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        load_plan_review_state,
        record_plan_review_attempt,
        reserve_plan_review_wave,
    )

    request = _plan_request(
        plan="P",
        goal="G",
        files_to_touch=[],
        context_level="minimal",
        plan_class="external",
    )
    fingerprint = pr._plan_request_fingerprint(
        plan=request.plan,
        goal=request.goal,
        files_to_touch=request.files_to_touch,
        context_level=request.context_level,
        context_notes=request.context_notes,
        plan_class=request.plan_class,
        scope=request.scope,
        include_tests=request.include_tests,
    )
    ctx = _make_ctx(tmp_path)
    ctx.task_metadata = {
        "deadline_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        "force_plan": True,
    }
    record_plan_review_attempt(tmp_path, ctx.task_id, fingerprint=fingerprint)
    reserve_plan_review_wave(
        tmp_path,
        ctx.task_id,
        fingerprint=fingerprint,
        plan_text_hash=pr.plan_text_fingerprint(request.plan),
        scout_roles=[],
        cutoff_at="2099-01-01T00:00:00+00:00",
    )
    from ouroboros.tools import plan_review_setup as pr_setup
    monkeypatch.setattr(pr_setup, "_resolve_plan_roots", lambda *_a, **_k: (tmp_path, tmp_path))

    out = asyncio.run(pr._run_plan_review_async(ctx, request))

    assert out.startswith("PLAN_TASK_SKIPPED_DEADLINE:")
    attempt = load_plan_review_state(tmp_path, ctx.task_id)["current_attempt"]
    assert attempt["status"] == "rail_degraded"
    assert attempt["reason"] == "plan_task_deadline"


def test_rail_degraded_prepanel_scouts_only_stop_quiescence():
    from ouroboros.task_results import (
        plan_review_audit_only_task_ids,
        plan_review_recorded_panel_task_ids,
    )

    fingerprint = "f" * 64
    state = {
        "current_attempt": {
            "fingerprint": fingerprint,
            "status": "rail_degraded",
            "reason": "plan_task_deadline",
        },
        "waves": [{
            "request_fingerprint": fingerprint,
            "intended_scouts": [{"task_ids": ["scout-rail"]}],
        }],
    }

    assert plan_review_audit_only_task_ids(state) == ["scout-rail"]
    assert plan_review_recorded_panel_task_ids(state) == []

    state["waves"][0]["review"] = {"aggregate_signal": "REVIEW_REQUIRED"}
    assert plan_review_recorded_panel_task_ids(state) == ["scout-rail"]


def test_prepanel_unavailable_scout_remains_visible_to_absorption(monkeypatch, tmp_path):
    import ouroboros.task_results as task_results
    import ouroboros.task_status as task_status
    from ouroboros.loop import _load_direct_child_results

    fingerprint = "e" * 64
    state = {
        "current_attempt": {
            "fingerprint": fingerprint,
            "status": "unavailable",
            "reason": "review_budget_unavailable",
        },
        "waves": [{
            "request_fingerprint": fingerprint,
            "intended_scouts": [{"task_ids": ["scout-paid"]}],
        }],
    }
    monkeypatch.setattr(task_results, "load_plan_review_state", lambda *_args: state)
    monkeypatch.setattr(task_status, "find_child_tasks", lambda *_args, **_kwargs: [{
        "task_id": "scout-paid",
        "parent_task_id": "root",
        "root_task_id": "root",
        "status": "completed",
        "result": "paid planning evidence",
    }])

    rows = _load_direct_child_results(tmp_path, "root", "root")

    assert [row["task_id"] for row in rows] == ["scout-paid"]


def test_missing_deadline_is_not_treated_as_expired(tmp_path):
    import ouroboros.tools.plan_review as pr

    ctx = _make_ctx(tmp_path)
    ctx.task_metadata = {}
    assert pr._plan_deadline_skip(ctx) == ""


def test_non_live_queue_clamps_default_swarm_timing_to_subsecond(monkeypatch, tmp_path):
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "120")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "900")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)

    assert pr._planning_swarm_timing(ctx) == (0.25, 0.25)


def test_planning_swarm_persists_intents_before_launch_and_partial_failure(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import load_plan_review_state, plan_review_wave
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-partial-launch"
    ctx.event_queue = queue.Queue()
    calls = {"count": 0, "fingerprint": ""}
    leaked = "abcdefghijklmnop-secret-value"

    def fake_schedule(ctx_arg, _internal=None, **_kwargs):
        # Planning scouts use the generic scheduler default: a configured live
        # harness is selected there, with its existing loud native fallback.
        assert _kwargs.get("executor", "auto") == "auto"
        calls["count"] += 1
        state = load_plan_review_state(tmp_path, str(ctx_arg.task_id))
        assert len(state["waves"]) == 1
        wave = state["waves"][0]
        calls["fingerprint"] = wave["request_fingerprint"]
        assert wave["scout_cutoff_at"]
        if calls["count"] == 1:
            assert [row["schedule_status"] for row in wave["intended_scouts"]] == [
                "pending", "pending",
            ]
            ctx_arg._last_scheduled_subagents = [{"task_ids": ["scout-ready"]}]
            return "scheduled scout-ready"
        assert [row["schedule_status"] for row in wave["intended_scouts"]] == [
            "started", "pending",
        ]
        return f"ERROR: provider rejected api_key={leaked}"

    def fake_wait(*_args, **_kwargs):
        state = load_plan_review_state(tmp_path, str(ctx.task_id))
        wave = plan_review_wave(state, calls["fingerprint"])
        assert wave is not None
        assert [row["schedule_status"] for row in wave["intended_scouts"]] == [
            "started", "failed",
        ]
        return {
            "all_terminal": True,
            "tasks": {
                "scout-ready": {
                    "status": "completed",
                    "role": "planning-scout-1",
                    "result": "summary: usable",
                }
            },
        }

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)
    monkeypatch.setattr(pr, "wait_for_effective_tasks", fake_wait)
    result = _start_swarm(
        ctx,
        _plan_request(
            "Do the work", "Ship a fix",
            ["a.py", "b.py", "c.py", "d.py"],
            context_level="minimal",
        ),
    )

    assert result["started"] is True
    assert result["task_ids"] == ["scout-ready"]
    assert result["handoffs"]["included_task_ids"] == ["scout-ready"]
    assert len(result["handoffs"]["omissions"]) == 1
    omission = result["handoffs"]["omissions"][0]
    assert omission["role"] == "planning-scout-2"
    assert omission["reason"] == "schedule_failed"
    assert "***REDACTED***" in omission["detail"]
    assert leaked not in json.dumps(result["handoffs"], ensure_ascii=False)
    audit = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
    assert audit["audit_only"] is True
    assert audit["authoritative"] is False


def test_plan_review_audit_wait_merge_is_fingerprint_and_lineage_bound(tmp_path):
    from ouroboros.task_results import persist_plan_review_handoffs

    fingerprint = "a" * 64
    task_id = "parent-audit-merge"
    incoming = {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "task_ids": ["scout-1"],
        "wait": {},
    }
    artifact = persist_plan_review_handoffs(tmp_path, task_id, incoming)
    path = pathlib.Path(artifact["path"])
    assert json.loads(path.read_text(encoding="utf-8"))["wait"] == {}

    original_wait = {
        "tasks": {"scout-1": {"status": "completed", "result": "full handoff"}},
        "all_terminal": True,
    }
    safe_prior = {
        **incoming,
        "wait": original_wait,
        "audit_only": True,
        "authoritative": False,
    }
    path.write_text(json.dumps(safe_prior), encoding="utf-8")
    persist_plan_review_handoffs(tmp_path, task_id, incoming)
    assert json.loads(path.read_text(encoding="utf-8"))["wait"] == original_wait

    replacement_wait = {"tasks": {"scout-1": {"status": "running"}}}
    persist_plan_review_handoffs(tmp_path, task_id, {**incoming, "wait": replacement_wait})
    assert json.loads(path.read_text(encoding="utf-8"))["wait"] == replacement_wait

    unsafe_priors = [
        {**safe_prior, "request_fingerprint": "b" * 64},
        {**safe_prior, "schema_version": 2},
        {**safe_prior, "audit_only": False},
        {**safe_prior, "authoritative": True},
        {**safe_prior, "wait": []},
        {**safe_prior, "wait": {"tasks": []}},
        {**safe_prior, "wait": {"tasks": {}}},
        {**safe_prior, "wait": {"tasks": {"scout-1": "forged"}}},
        {**safe_prior, "wait": {"tasks": {"forged-scout": {"status": "completed"}}}},
    ]
    for prior in unsafe_priors:
        path.write_text(json.dumps(prior), encoding="utf-8")
        persist_plan_review_handoffs(tmp_path, task_id, incoming)
        assert json.loads(path.read_text(encoding="utf-8"))["wait"] == {}

    path.write_text("{malformed", encoding="utf-8")
    artifact = persist_plan_review_handoffs(tmp_path, task_id, incoming)
    assert "error" not in artifact
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["wait"] == {}
    assert stored["audit_only"] is True
    assert stored["authoritative"] is False


def test_consumed_handoff_intermediate_persist_preserves_wait_audit(tmp_path, monkeypatch):
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    fingerprint = "d" * 64
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-consumed-audit"
    original_wait = {
        "tasks": {"scout-1": {"status": "completed", "result": "full handoff"}},
        "all_terminal": True,
    }
    wave = {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "plan_text_hash": "e" * 64,
        "intended_scouts": [{
            "role": "planning-scout-1",
            "schedule_status": "started",
            "task_ids": ["scout-1"],
            "schedule_reason": "scheduled scout-1",
        }],
        "included_task_ids": [],
        "consumed_task_ids": [],
        "omissions": [{
            "role": "planning-scout-1",
            "task_id": "scout-1",
            "reason": "not_terminal_at_review_cutoff",
        }],
        "reviewed_result_hashes": {},
        "review_evidence_status": "pending",
    }
    initial = pr.plan_review_wave_handoffs(wave)
    initial["wait"] = original_wait
    assert "error" not in pr._persist_planning_handoffs(ctx, initial)
    monkeypatch.setattr(pr, "record_plan_review_consumed", lambda *args, **kwargs: wave)

    pr._mark_planning_handoffs_consumed(ctx, dict(wave))

    stored = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
    assert stored["wait"] == original_wait


def test_plan_review_wave_reservation_is_atomic_per_task_result(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from ouroboros.task_results import load_plan_review_state, reserve_plan_review_wave

    def reserve(_index):
        _wave, created = reserve_plan_review_wave(
            tmp_path,
            "parent-concurrent-reserve",
            fingerprint="f" * 64,
            plan_text_hash="a" * 64,
            scout_roles=["planning-scout-1"],
            cutoff_at="2099-01-01T00:00:00+00:00",
        )
        return created

    with ThreadPoolExecutor(max_workers=8) as pool:
        created = list(pool.map(reserve, range(8)))

    assert sum(created) == 1
    state = load_plan_review_state(tmp_path, "parent-concurrent-reserve")
    assert len(state["waves"]) == 1


def test_planning_swarm_does_not_duplicate_wave_after_terminal_empty_handoff(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "0")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "0.25")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "parent1", "session_id": "sess1"}
    scheduled = {"count": 0}

    def fake_schedule(ctx_arg, _internal=None, **kwargs):
        scheduled["count"] += 1
        tid = f"scout-{scheduled['count']}"
        records = list(getattr(ctx_arg, "_last_scheduled_subagents", []) or [])
        records.append({"task_ids": [tid]})
        ctx_arg._last_scheduled_subagents = records
        return f"scheduled {tid}"

    wait_results = [
        {"timed_out": False, "tasks": {"scout-1": {"status": "failed", "result": ""}}},
        {"timed_out": False, "tasks": {"scout-1": {"status": "failed", "result": ""}}},
        {"timed_out": False, "tasks": {"scout-2": {"status": "completed", "role": "planning-scout-1", "result": "summary: fresh"}}},
    ]

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)
    monkeypatch.setattr(pr, "wait_for_effective_tasks", lambda *_args, **_kwargs: wait_results.pop(0))

    first = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="minimal"),
    )
    second = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="minimal"),
    )

    assert first["started"] is True
    assert second["started"] is True
    assert first["degraded_evidence"] is True
    assert second["degraded_evidence"] is True
    assert second["task_ids"] == ["scout-1"]
    assert second["handoffs"]["omissions"][0]["reason"] == "terminal_without_usable_handoff:failed"
    assert scheduled["count"] == 1


def test_planning_swarm_capacity_becomes_panel_omission(monkeypatch, tmp_path):
    import ouroboros.tools.control as control
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "1")
    monkeypatch.setattr(
        control,
        "_schedule_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not schedule")),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "parent1", "session_id": "sess1"}

    result = _start_swarm(
        ctx,
        _plan_request("Do the work", "Ship a fix", context_level="focused"),
    )

    assert result["started"] is True
    assert result["degraded_evidence"] is True
    assert result["task_ids"] == []
    assert result["handoffs"]["omissions"][0]["reason"] == "schedule_failed"
    assert "no spare worker capacity" in result["handoffs"]["omissions"][0]["detail"]


def test_capacity_cutoff_and_schedule_failure_all_become_panel_omissions(monkeypatch, tmp_path):
    """Every intended scout gets one omission; none invokes a second model lane."""
    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr
    from ouroboros.tools.registry import ToolContext

    # <2 workers → capacity.
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "1")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "parent1", "session_id": "sess1"}
    result = _start_swarm(
        ctx, _plan_request("P", "G", context_level="minimal"),
    )
    assert result["started"] is True
    assert result["handoffs"]["omissions"][0]["reason"] == "schedule_failed"

    # Saturated pool (scouts scheduled, none completed, timed out) → capacity.
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "0")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "0.25")

    def fake_schedule(ctx_arg, _internal=None, **kwargs):
        ctx_arg._last_scheduled_subagents = [{"task_ids": ["scout-sat"]}]
        return "scheduled scout-sat"

    monkeypatch.setattr(control, "_schedule_task", fake_schedule)
    def saturated_wait(*_a, **kwargs):
        __import__("time").sleep(float(kwargs.get("timeout_sec") or 0))
        return {"timed_out": True, "tasks": {"scout-sat": {"status": "running", "result": ""}}}

    monkeypatch.setattr(pr, "wait_for_effective_tasks", saturated_wait)
    ctx2 = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx2.task_id = "parent2"
    ctx2.task_depth = 0
    ctx2.current_chat_id = 1
    ctx2.event_queue = queue.Queue()
    ctx2.task_metadata = {"root_task_id": "parent2", "session_id": "sess1"}
    saturated = _start_swarm(
        ctx2, _plan_request("P", "G", context_level="minimal"),
    )
    assert saturated["started"] is True
    assert saturated["handoffs"]["omissions"][0]["reason"] == (
        "not_terminal_at_review_cutoff:ceiling"
    )

    # Scheduling failure (no scout started at all) is still explicit panel evidence.
    monkeypatch.setattr(control, "_schedule_task", lambda ctx_arg, _internal=None, **kwargs: "ERROR: refused")
    ctx3 = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx3.task_id = "parent3"
    ctx3.task_depth = 0
    ctx3.current_chat_id = 1
    ctx3.event_queue = queue.Queue()
    ctx3.task_metadata = {"root_task_id": "parent3", "session_id": "sess1"}
    ctx3._last_scheduled_subagents = []
    infra = _start_swarm(
        ctx3, _plan_request("P", "G", context_level="minimal"),
    )
    assert infra["started"] is True
    assert infra["handoffs"]["omissions"][0]["reason"] == "schedule_failed"


def test_all_schedule_failures_still_reach_reviewer_panel(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.control as control
    import ouroboros.tools.plan_review as pr

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "parent-all-schedule-failed"
    # A REAL queue, not _make_ctx's MagicMock attribute: _planning_swarm_timing
    # treats a mock event_queue as non-live and clamps the scout window to 0.25s
    # regardless of the env below, reintroducing the slow-runner race.
    ctx.event_queue = queue.Queue()
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    # A generous window, not 0 and not 0.25: a zero-length window is refused before
    # launch (this test is about a schedule FAILURE reaching the panel, not window
    # admission), and 0.25s raced slow Windows runners — the durable wave write ate
    # the whole window, so the pre-launch plan refused BEFORE _schedule_task ran and
    # the "queue refused" detail never existed. Every schedule fails, so the collect
    # loop still exits immediately (no scouts to wait for).
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "30")
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m1,m2")
    monkeypatch.setattr(control, "_schedule_task", lambda *_a, **_k: "ERROR: queue refused")
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: ("", frozenset()))
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    captured = {}

    async def fake_slots(_ctx, models, _system_prompt, user_content, user_stable_len=0, slot_ids=None):
        captured["user_content"] = user_content
        return [{
            "model": model,
            "text": _review_text("GREEN"),
            "error": None,
            "tokens_in": 1,
            "tokens_out": 1,
            "cost": 0.0,
        } for model in models]

    monkeypatch.setattr(pr, "_run_plan_review_slots", fake_slots)
    out = asyncio.run(pr._run_plan_review_async(
        ctx, _plan_request(plan="P", goal="G", files_to_touch=[], context_level="minimal"),
    ))

    assert "schedule_failed" in captured["user_content"]
    assert "queue refused" in captured["user_content"]
    assert "PLAN_REVIEW_OUTCOME: GREEN" in out


def test_cutoff_omissions_go_directly_to_reviewer_without_inline_model(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.plan_review as pr

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "parent-cap"

    omitted = [
        {
            "task_id": "s1", "role": "planning-scout-1", "status": "running",
            "reason": "not_terminal_at_review_cutoff:ceiling",
        },
        {
            "task_id": "s2", "role": "planning-scout-2", "status": "failed",
            "reason": "terminal_without_usable_handoff:failed",
        },
    ]
    def capacity_swarm(ctx_arg, request, fingerprint=""):
        handoffs = _seed_mock_planning_wave(
            ctx_arg,
            request,
            scouts=[
                {"role": "planning-scout-1", "task_ids": ["s1"], "reason": "scheduled s1"},
                {"role": "planning-scout-2", "task_ids": ["s2"], "reason": "scheduled s2"},
            ],
            included_task_ids=[],
            omissions=omitted,
            stop_reason="ceiling",
            fingerprint=fingerprint,
        )
        handoffs["wait"] = {
            "tasks": {"s1": {"status": "running"}, "s2": {"status": "failed"}}
        }
        return {
            "started": True,
            "degraded_evidence": True,
            "task_ids": ["s1", "s2"],
            "handoffs": handoffs,
        }

    monkeypatch.setattr(pr, "_start_planning_swarm", capacity_swarm)
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: ("", frozenset()))
    captured = {}

    async def fake_slots(_ctx, models, system_prompt, user_content, user_stable_len=0, slot_ids=None):
        captured["user_content"] = user_content
        return [{
            "model": m,
            "text": "PLAN_FINDINGS_JSON: []\nAGGREGATE: GREEN",
            "error": None,
            "tokens_in": 1,
            "tokens_out": 1,
            "cost": 0.0,
        } for m in models]

    monkeypatch.setattr(pr, "_run_plan_review_slots", fake_slots)
    monkeypatch.setattr(pr, "wait_for_effective_tasks", lambda *_a, **_k: {
        "tasks": {
            "s1": {"status": "completed", "result": "late one"},
            "s2": {"status": "completed", "result": "late two"},
        },
        "all_terminal": True,
    })
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m1,m2")

    out = asyncio.run(pr._run_plan_review_async(
        ctx, _plan_request(plan="P", goal="G", files_to_touch=[], context_level="minimal"),
    ))
    assert "DEGRADED PLANNING EVIDENCE" in out
    assert "Planning Critique" not in captured["user_content"]
    assert "inline" not in captured["user_content"].lower()
    assert "not_terminal_at_review_cutoff:ceiling" in captured["user_content"]
    assert "terminal_without_usable_handoff:failed" in captured["user_content"]
    stored = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
    assert stored["omissions"] == omitted
    assert stored["late_audit"]["affects_review"] is False
    assert stored["review"]["aggregate_signal"] == "GREEN"
    assert stored["review"]["closed"] is True

    # Strict host-state/artifact failures still fail closed before reviewer calls.
    monkeypatch.setattr(
        pr, "_start_planning_swarm",
        lambda *_a, **_k: {"started": False, "error": "ERROR: artifact save failed"},
    )
    out_infra = asyncio.run(pr._run_plan_review_async(
        ctx,
        _plan_request(
            plan="P changed again", goal="G", files_to_touch=[], context_level="minimal",
        ),
    ))
    assert out_infra == "ERROR: artifact save failed"


def test_scout_change_after_prompt_is_audit_only_without_paid_review_replay(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.plan_review as pr
    from ouroboros.loop import _direct_child_results
    from ouroboros.task_results import load_plan_review_state, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _current_child_result_disposition

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "parent-stale-after-prompt"
    reviewed_snapshot = {
        "status": "completed",
        "role": "planning-scout-1",
        "result": "reviewed version",
    }

    def stale_swarm(ctx_arg, request, fingerprint=""):
        handoffs = _seed_mock_planning_wave(
            ctx_arg,
            request,
            scouts=[{
                "role": "planning-scout-1",
                "task_ids": ["scout-stale"],
                "reason": "scheduled scout-stale",
            }],
            included_task_ids=["scout-stale"],
            omissions=[],
            fingerprint=fingerprint,
        )
        root, parent_id = pr._planning_state_location(ctx_arg)
        write_task_result(
            root,
            "scout-stale",
            "completed",
            parent_task_id=parent_id,
            root_task_id=parent_id,
            delegation_role="subagent",
            role="planning-scout-1",
            result=reviewed_snapshot["result"],
        )
        handoffs["wait"] = {"tasks": {"scout-stale": dict(reviewed_snapshot)}}
        handoffs["artifact"] = {"path": str(pr._planning_handoff_path(ctx_arg))}
        return {
            "started": True,
            "task_ids": ["scout-stale"],
            "handoffs": handoffs,
        }

    calls = {"panel": 0}

    async def mutate_then_review(_ctx, models, _system_prompt, _user_content, user_stable_len=0, slot_ids=None):
        calls["panel"] += 1
        write_task_result(tmp_path, "scout-stale", "completed", result="changed after prompt")
        return [{
            "model": model,
            "text": _review_text("GREEN"),
            "error": None,
            "tokens_in": 1,
            "tokens_out": 1,
            "cost": 0.0,
        } for model in models]

    monkeypatch.setattr(pr, "_start_planning_swarm", stale_swarm)
    monkeypatch.setattr(pr, "_run_plan_review_slots", mutate_then_review)
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: ("", frozenset()))
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m1,m2")

    first = asyncio.run(pr._run_plan_review_async(
        ctx, _plan_request(plan="P", goal="G", files_to_touch=[], context_level="minimal"),
    ))
    second = asyncio.run(pr._run_plan_review_async(
        ctx, _plan_request(plan="P", goal="G", files_to_touch=[], context_level="minimal"),
    ))

    assert "PLAN_REVIEW_OUTCOME: GREEN" in first
    assert "Cached exact review:** True" in second
    assert "PLANNING SCOUT SNAPSHOT CHANGED" in first
    assert "PLANNING SCOUT SNAPSHOT CHANGED" in second
    assert calls["panel"] == 1
    state = load_plan_review_state(tmp_path, ctx.task_id)
    wave = state["waves"][0]
    assert wave["consumed_task_ids"] == ["scout-stale"]
    assert wave["disposition_warnings"][0]["code"] == "CHILD_RESULT_STALE"
    stored = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
    assert stored["disposition_warnings"] == wave["disposition_warnings"]
    current = load_effective_task_result(tmp_path, "scout-stale")
    assert _current_child_result_disposition(current) == ""
    loop_ctx = SimpleNamespace(
        status_drive_root=tmp_path,
        drive_root=tmp_path,
        drive_logs=tmp_path / "logs",
        task_id=ctx.task_id,
        root_task_id=ctx.task_id,
    )
    assert _direct_child_results(loop_ctx) == []


def test_paid_review_resumes_evidence_integration_without_panel_replay(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import load_plan_review_state, plan_review_wave

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "parent-paid-review-resume"
    calls = {"panel": 0, "consumed": 0}

    async def fake_slots(_ctx, models, _system_prompt, _user_content, user_stable_len=0, slot_ids=None):
        calls["panel"] += 1
        return [{
            "model": model,
            "text": _review_text("GREEN"),
            "error": None,
            "tokens_in": 1,
            "tokens_out": 1,
            "cost": 0.0,
        } for model in models]

    real_record_consumed = pr.record_plan_review_consumed

    def fail_first_consumed(*args, **kwargs):
        calls["consumed"] += 1
        if calls["consumed"] == 1:
            raise TimeoutError("injected post-panel persistence failure")
        return real_record_consumed(*args, **kwargs)

    monkeypatch.setattr(pr, "_start_planning_swarm", _completed_planning_swarm)
    monkeypatch.setattr(pr, "_run_plan_review_slots", fake_slots)
    monkeypatch.setattr(pr, "record_plan_review_consumed", fail_first_consumed)
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: ("", frozenset()))
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m1,m2")

    kwargs = dict(plan="P", goal="G", files_to_touch=[], context_level="minimal")
    first = asyncio.run(pr._run_plan_review_async(ctx, _plan_request(**kwargs)))
    assert "PLAN_REVIEW_STATE_PERSIST_FAILED" in first
    state = load_plan_review_state(tmp_path, ctx.task_id)
    fingerprint = state["waves"][0]["request_fingerprint"]
    pending = plan_review_wave(state, fingerprint)
    assert pending["review"]["aggregate_signal"] == "GREEN"
    assert pending["review_evidence_status"] == "pending"
    assert set(pending["reviewed_result_hashes"]) == {"scout1"}
    assert state["latest_review_fingerprint"] == ""

    second = asyncio.run(pr._run_plan_review_async(ctx, _plan_request(**kwargs)))
    assert "PLAN_REVIEW_OUTCOME: GREEN" in second
    assert "Cached exact review:** True" in second
    assert calls["panel"] == 1
    integrated_state = load_plan_review_state(tmp_path, ctx.task_id)
    integrated = plan_review_wave(integrated_state, fingerprint)
    assert integrated["review_evidence_status"] == "integrated"
    assert integrated["consumed_task_ids"] == ["scout1"]
    assert integrated_state["latest_review_fingerprint"] == fingerprint


def test_terminal_zero_ready_scout_wave_still_reaches_reviewer_panel(monkeypatch, tmp_path):
    import asyncio

    import ouroboros.tools.plan_review as pr

    ctx = _make_ctx(tmp_path)
    ctx.task_id = "parent-terminal-empty"
    omissions = [
        {
            "task_id": "s1", "role": "planning-scout-1", "status": "failed",
            "reason": "terminal_without_usable_handoff:failed",
        },
        {
            "task_id": "s2", "role": "planning-scout-2", "status": "completed",
            "reason": "completed_without_nonempty_handoff",
        },
    ]
    def terminal_empty_swarm(ctx_arg, request, fingerprint=""):
        handoffs = _seed_mock_planning_wave(
            ctx_arg,
            request,
            scouts=[
                {"role": "planning-scout-1", "task_ids": ["s1"], "reason": "scheduled s1"},
                {"role": "planning-scout-2", "task_ids": ["s2"], "reason": "scheduled s2"},
            ],
            included_task_ids=[],
            omissions=omissions,
            fingerprint=fingerprint,
        )
        handoffs["wait"] = {"tasks": {
            "s1": {"status": "failed", "result": ""},
            "s2": {"status": "completed", "result": ""},
        }}
        return {
            "started": True,
            "degraded_evidence": True,
            "task_ids": ["s1", "s2"],
            "handoffs": handoffs,
        }

    monkeypatch.setattr(pr, "_start_planning_swarm", terminal_empty_swarm)
    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "doc")
    monkeypatch.setattr(pr, "build_head_snapshot_section", lambda *_a, **_k: ("", frozenset()))
    captured = {}

    async def fake_slots(_ctx, models, _system_prompt, user_content, user_stable_len=0, slot_ids=None):
        captured["user_content"] = user_content
        return [{
            "model": model, "text": _review_text("GREEN"), "error": None,
            "tokens_in": 1, "tokens_out": 1, "cost": 0.0,
        } for model in models]

    monkeypatch.setattr(pr, "_run_plan_review_slots", fake_slots)
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["m1", "m2"])
    monkeypatch.setattr(pr, "wait_for_effective_tasks", lambda *_a, **_k: {
        "tasks": {"s1": {"status": "failed"}, "s2": {"status": "completed", "result": ""}},
        "all_terminal": True,
    })
    monkeypatch.setenv("OUROBOROS_REVIEW_MODELS", "m1,m2")

    out = asyncio.run(pr._run_plan_review_async(
        ctx, _plan_request(plan="P", goal="G", files_to_touch=[], context_level="minimal"),
    ))

    assert "DEGRADED PLANNING EVIDENCE" in out
    assert "terminal_without_usable_handoff:failed" in captured["user_content"]
    assert "completed_without_nonempty_handoff" in captured["user_content"]
    assert "PLAN_REVIEW_OUTCOME: GREEN" in out


def _seed_mock_planning_wave(
    ctx,
    request,
    *,
    scouts: list[dict],
    included_task_ids: list[str],
    omissions: list[dict],
    stop_reason: str = "",
    fingerprint: str = "",
) -> dict:
    """Create authoritative host state for tests that mock only scout execution."""
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import (
        load_plan_review_state,
        plan_review_wave,
        plan_review_wave_handoffs,
        record_plan_review_collection,
        record_plan_review_scout,
        reserve_plan_review_wave,
    )

    # The BINDING fingerprint is computed by the caller from the AGENT-passed
    # envelope; recomputing it here from a host-RESOLVED request would key the wave
    # under an identity production never uses.
    fingerprint = fingerprint or pr._plan_request_fingerprint(
        plan=str(request.plan or ""),
        goal=str(request.goal or ""),
        files_to_touch=list(request.files_to_touch or []),
        context_level=str(request.context_level or ""),
        context_notes=str(request.context_notes or ""),
        plan_class=str(request.plan_class or "self_mod"),
        scope=request.scope,
        include_tests=bool(request.include_tests),
    )
    root, task_id = pr._planning_state_location(ctx)
    wave = plan_review_wave(load_plan_review_state(root, task_id), fingerprint)
    if wave is None:
        wave, _ = reserve_plan_review_wave(
            root,
            task_id,
            fingerprint=fingerprint,
            plan_text_hash=pr.plan_text_fingerprint(str(request.plan or "")),
            scout_roles=[str(item["role"]) for item in scouts],
            cutoff_at="2099-01-01T00:00:00+00:00",
        )
        for scout in scouts:
            wave = record_plan_review_scout(
                root,
                task_id,
                fingerprint=fingerprint,
                role=str(scout["role"]),
                schedule_status=str(scout.get("schedule_status") or "started"),
                task_ids=list(scout.get("task_ids") or []),
                reason=str(scout.get("reason") or "scheduled"),
            )
        wave = record_plan_review_collection(
            root,
            task_id,
            fingerprint=fingerprint,
            included_task_ids=included_task_ids,
            omissions=omissions,
            stop_reason=stop_reason,
        )
    return plan_review_wave_handoffs(wave)


def _completed_planning_swarm(ctx, request, fingerprint="") -> dict:
    """Return a mocked completed swarm without bypassing host planning authority."""
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.plan_review import _planning_state_location

    handoffs = _seed_mock_planning_wave(
        ctx,
        request,
        scouts=[{
            "role": "planning-scout-1",
            "schedule_status": "started",
            "task_ids": ["scout1"],
            "reason": "scheduled scout1",
        }],
        included_task_ids=["scout1"],
        omissions=[],
        fingerprint=fingerprint,
    )
    handoffs.update({
        "wait": {
            "tasks": {
                "scout1": {
                    "status": "completed",
                    "role": "planning-scout-1",
                    "result": "summary: ok",
                }
            }
        },
        "artifact": {"path": "/tmp/plan_task_handoffs.json"},
    })
    root, parent_id = _planning_state_location(ctx)
    write_task_result(
        root,
        "scout1",
        "completed",
        parent_task_id=parent_id,
        root_task_id=parent_id,
        delegation_role="subagent",
        role="planning-scout-1",
        result="summary: ok",
    )
    return {
        "started": True,
        "task_ids": ["scout1"],
        "handoffs": handoffs,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPlanReviewInputValidation(unittest.TestCase):
    def setUp(self):
        from ouroboros.tools.plan_review import _handle_plan_task
        self.handler = _handle_plan_task
        self.ctx = _make_ctx()

    def test_missing_plan_returns_error(self):
        result = self.handler(self.ctx, plan="", goal="some goal")
        self.assertIn("ERROR", result)
        self.assertIn("plan", result.lower())

    def test_missing_goal_returns_error(self):
        result = self.handler(self.ctx, plan="some plan", goal="")
        self.assertIn("ERROR", result)
        self.assertIn("goal", result.lower())

    def test_whitespace_plan_returns_error(self):
        result = self.handler(self.ctx, plan="   ", goal="some goal")
        self.assertIn("ERROR", result)

    def test_whitespace_goal_returns_error(self):
        result = self.handler(self.ctx, plan="some plan", goal="   ")
        self.assertIn("ERROR", result)


class TestPlanReviewModels(unittest.TestCase):
    def test_falls_back_to_config_default_when_env_is_empty(self):
        """Empty OUROBOROS_REVIEW_MODELS → use the shipped SETTINGS_DEFAULTS.

        Post-v4.33.1: plan_task delegates to ``config.get_review_models``,
        which returns the shipped triad default when the env is empty — the
        same behavior as the commit triad. This keeps plan_task and commit
        review in lockstep instead of plan_task silently collapsing to
        ``[main] * 3`` on an unconfigured instance.

        Hermetic: explicitly clears all provider env vars AND
        ``OPENAI_BASE_URL`` so this test does not depend on shell/CI
        environment. An ambient ANTHROPIC_API_KEY (or any direct-provider
        key) would flip ``config.get_review_models`` into the exclusive
        direct-provider fallback path and break the assertion, and a
        non-empty ``OPENAI_BASE_URL`` is treated as a custom runtime
        configuration by ``_exclusive_direct_remote_provider_env`` which
        also alters the code path.
        """
        from ouroboros.tools.plan_review import _get_review_models
        env = {
            "OUROBOROS_REVIEW_MODELS": "",
            "OUROBOROS_MODEL": "test/model-x",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "",
            "OPENAI_COMPATIBLE_API_KEY": "",
            "CLOUDRU_FOUNDATION_MODELS_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            models = _get_review_models()
        self.assertEqual(len(models), 3)
        # The shipped default is the 3-model OpenRouter triad (GPT-5.4,
        # Gemini 3.1 Pro Preview, Claude Opus 4.7). Exact identities are
        # version-tracked in config.SETTINGS_DEFAULTS; we just assert the
        # size and that we did NOT silently collapse to [main] * 3.
        self.assertFalse(
            all(m == "test/model-x" for m in models),
            f"plan_task must not silently collapse to main × 3 when the default triad "
            f"is configured; got {models!r}",
        )

    def test_returns_configured_models(self):
        from ouroboros.tools.plan_review import _get_review_models
        configured = "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.6"
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": configured}, clear=False):
            models = _get_review_models()
        self.assertEqual(models, [
            "openai/gpt-5.5",
            "google/gemini-3.5-flash",
            "anthropic/claude-opus-4.6",
        ])

    def test_honors_arbitrary_model_count(self):
        """An arbitrary configured reviewer count is honored with no implicit cap
        — the owner chooses how many reviewer slots run (Decision D4)."""
        from ouroboros.tools.plan_review import _get_review_models
        many = "a/1,b/2,c/3,d/4,e/5"
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": many}, clear=False):
            models = _get_review_models()
        self.assertEqual(models, ["a/1", "b/2", "c/3", "d/4", "e/5"])

    def test_preserves_one_model_config(self):
        """One configured model stays one slot (plan_review then runs as a
        coordinative single reviewer — no implicit expansion, no hard error)."""
        from ouroboros.tools.plan_review import _get_review_models
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": "only/one"}, clear=False):
            models = _get_review_models()
        self.assertEqual(models, ["only/one"])

    def test_preserves_two_model_config(self):
        """Two configured models are two reviewer slots, not an implicit third."""
        from ouroboros.tools.plan_review import _get_review_models
        with patch.dict(os.environ, {"OUROBOROS_REVIEW_MODELS": "model/a,model/b"}, clear=False):
            models = _get_review_models()
        self.assertEqual(models, ["model/a", "model/b"])

    def test_delegates_to_config_get_review_models_for_direct_provider_fallback(self):
        """plan_task must use the same direct-provider fallback as the commit triad.

        Regression guard for v4.33.1 scope review finding
        ``plan_task_review_model_parity`` + v4.39.0 quorum-safe-fallback fix:
        ``config.get_review_models``'s OpenAI-only / Anthropic-only fallback
        now rewrites the list to ``[main, light, light]`` (3 slots)
        when the configured reviewers don't match the exclusive direct-
        provider prefix, and ``_get_review_models`` must see that shape
        unchanged. Duplicate model IDs are valid stochastic reviewer slots.
        """
        from ouroboros.tools.plan_review import _get_review_models
        # Simulate Anthropic-only direct setup: only ANTHROPIC key present,
        # main is anthropic::..., but the reviewer list is still the default
        # OpenRouter-style set (so none match the anthropic:: prefix).
        env = {
            "OUROBOROS_REVIEW_MODELS": "openai/gpt-5.5,google/gemini-3.5-flash,anthropic/claude-opus-4.6",
            "OUROBOROS_MODEL": "anthropic::claude-opus-4-6",
            "OUROBOROS_MODEL_LIGHT": "anthropic::claude-sonnet-4-6",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "",
            "OPENAI_BASE_URL": "",
            "OPENAI_COMPATIBLE_API_KEY": "",
            "CLOUDRU_FOUNDATION_MODELS_API_KEY": "",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env, clear=False):
            models = _get_review_models()
        # Expect the Anthropic-only fallback: `[main, light, light]`.
        self.assertEqual(len(models), 3)
        self.assertEqual(
            models,
            [
                "anthropic::claude-opus-4-6",
                "anthropic::claude-sonnet-4-6",
                "anthropic::claude-sonnet-4-6",
            ],
            f"expected [main, light, light] direct-provider fallback, got {models!r}",
        )


class TestPlanReviewChecklist(unittest.TestCase):
    def test_checklist_section_exists_and_non_empty(self):
        """Plan Review Checklist section must exist in CHECKLISTS.md."""
        from ouroboros.tools.plan_review import _load_plan_checklist
        checklist = _load_plan_checklist()
        self.assertIsInstance(checklist, str)
        self.assertGreater(len(checklist), 100)
        # Verify key items are present
        self.assertIn("completeness", checklist)
        self.assertIn("correctness", checklist)
        self.assertIn("minimalism", checklist)
        self.assertIn("bible_alignment", checklist)
        self.assertIn("PLAN_FINDINGS_JSON", checklist)
        self.assertIn("exactly one aggregate line", checklist)


class TestPlanReviewSystemPrompt(unittest.TestCase):
    def test_system_prompt_frames_reviewer_as_candidate_validator(self):
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("validating a concrete candidate plan", prompt)
        self.assertIn("not brainstorming from zero", prompt)

    def test_system_prompt_declares_generative_stance(self):
        """Review stance must explicitly frame the reviewer as a generative partner."""
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("## Review stance", prompt)
        self.assertIn("GENERATIVE", prompt)
        # Design PARTNER framing — not auditor
        self.assertIn("PARTNER", prompt)

    def test_system_prompt_requires_own_approach_and_proposals_sections(self):
        """Required output structure must include 'Your own approach' and ## PROPOSALS."""
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("Required output structure", prompt)
        self.assertIn("Your own approach", prompt)
        self.assertIn("## PROPOSALS", prompt)

    def test_system_prompt_forbids_commit_hygiene_penalty(self):
        """Reviewers must not penalise missing tests/VERSION/README — plan has no code yet."""
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("Do NOT penalise missing tests", prompt)

    def test_system_prompt_explains_adaptive_quorum_coordination(self):
        """The prompt must explain that REVISE_PLAN requires a quorum across the
        configured reviewer slot count (adaptive — arbitrary N, v6.36.0). The
        heading uses the same adaptive-quorum SSOT language as docs/CHECKLISTS.md."""
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("adaptive-quorum", prompt)
        self.assertIn("configured reviewer slots", prompt)
        self.assertIn("adaptive_quorum", prompt)
        self.assertNotIn("majority-vote", prompt)  # wording drift fixed (round-5 doc-sync)

    def test_system_prompt_preserves_aggregate_contract(self):
        from ouroboros.tools.plan_review import _build_system_prompt
        prompt = _build_system_prompt("checklist", "", "", "")
        self.assertIn("AGGREGATE: GREEN", prompt)
        self.assertIn("AGGREGATE: REVIEW_REQUIRED", prompt)
        self.assertIn("AGGREGATE: REVISE_PLAN", prompt)


class TestPlanReviewFormatOutput(unittest.TestCase):
    def _run(self, raw_results):
        from ouroboros.tools.plan_review import _format_output
        return _format_output(raw_results, ["model-a", "model-b", "model-c"], "test goal", 12345)

    def test_green_when_no_fails_or_risks(self):
        results = [
            {"model": "model-a", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        # Final verdict (bolded) must be GREEN, not REVISE_PLAN or REVIEW_REQUIRED
        self.assertIn("**GREEN**", aggregate_section)
        self.assertNotIn("**REVISE_PLAN**", aggregate_section)
        self.assertNotIn("**REVIEW_REQUIRED**", aggregate_section)

    def test_review_required_when_risk_present(self):
        results = [
            {"model": "model-a", "text": _contract_review_text("REVIEW_REQUIRED"), "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        self.assertIn("REVIEW_REQUIRED", out)

    def test_minority_revise_plan_becomes_review_required(self):
        """One reviewer flagging REVISE_PLAN while the others do not → REVIEW_REQUIRED.

        Majority-vote coordination: a lone dissenting REVISE_PLAN surfaces as a
        strong coordination signal (REVIEW_REQUIRED with dissent noted), not an
        automatic REVISE_PLAN. Replaces the pre-majority-vote behavior where any
        single REVISE_PLAN escalated the final verdict.
        """
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        # Final verdict should be REVIEW_REQUIRED, not REVISE_PLAN
        self.assertIn("REVIEW_REQUIRED", aggregate_section)
        self.assertNotIn("**REVISE_PLAN**", aggregate_section)
        # Dissent must be explicitly noted in the aggregate reasoning
        self.assertIn("dissent", aggregate_section.lower())

    def test_single_reviewer_plan_review_discloses_no_diversity(self):
        """v6.36.0 (Bible P3): a one-slot plan review surfaces a loud
        single_reviewer_no_diversity disclosure — never a silent one-slot pass."""
        out = self._run([{"model": "model-a", "text": _contract_review_text("GREEN"), "error": None}])
        assert "single_reviewer_no_diversity" in out
        # A multi-reviewer run does NOT carry the disclosure.
        multi = self._run([
            {"model": "model-a", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
        ])
        assert "single_reviewer_no_diversity" not in multi

    def test_single_reviewer_revise_plan_escalates(self):
        """A lone configured reviewer (1-slot setup) flagging REVISE_PLAN → REVISE_PLAN.

        The escalation quorum routes through config.adaptive_quorum (SSOT):
        adaptive_quorum(1) == 1, so the single reviewer's REVISE_PLAN is honored
        rather than downgraded — matching the system prompt's promise ("a single
        reviewer in a 1-slot setup"). Guards against the pre-SSOT hardcoded
        `revise_count >= 2` which silently downgraded N=1 to REVIEW_REQUIRED.
        """
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVISE_PLAN", aggregate_section)

    def test_majority_revise_plan_blocks(self):
        """Two reviewers flagging REVISE_PLAN → final verdict is REVISE_PLAN."""
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVISE_PLAN", aggregate_section)

    def test_unanimous_revise_plan_is_revise_plan(self):
        """Three reviewers flagging REVISE_PLAN → final verdict is REVISE_PLAN."""
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("REVISE_PLAN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVISE_PLAN", aggregate_section)

    def test_error_result_does_not_crash(self):
        results = [
            {"model": "model-a", "text": "", "error": "Timeout after 120s"},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        self.assertIn("ERROR", out)

    def test_single_revise_plus_error_is_review_required(self):
        """One REVISE_PLAN + one error + one GREEN → REVIEW_REQUIRED (no majority FAIL).

        Replaces the pre-majority-vote test that asserted REVISE_PLAN stayed final
        when a later reviewer errored. Majority-vote coordination requires TWO
        agreeing REVISE_PLAN reviewers; a single dissent plus a degraded reviewer
        does not clear the bar.
        """
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-b", "text": "", "error": "Timeout after 120s"},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVIEW_REQUIRED", aggregate_section)
        self.assertNotIn("**REVISE_PLAN**", aggregate_section)

    def test_aggregate_block_reports_per_reviewer_counts(self):
        """Aggregate block should surface per-reviewer signal counts for auditability."""
        results = [
            {"model": "model-a", "text": _contract_review_text("REVISE_PLAN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("REVIEW_REQUIRED"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        aggregate_section = out.split("## Aggregate")[1]
        self.assertIn("REVISE_PLAN=1", aggregate_section)
        self.assertIn("REVIEW_REQUIRED=1", aggregate_section)
        self.assertIn("GREEN=1", aggregate_section)

    def test_empty_reviewer_list_returns_explicit_review_required(self):
        """Empty per-reviewer list → explicit 'no responses' message, not misleading zero counts.

        Defensive path: `_run_plan_review_async` guarantees at least one reviewer,
        but if `_format_output` is ever called with an empty list the aggregate
        block must say so explicitly rather than rendering 'REVISE_PLAN=0, ...'
        which would read like a false clean-PASS aggregate.
        """
        out = self._run([])
        self.assertIn("## Aggregate", out)
        self.assertIn("REVIEW_REQUIRED", out)
        self.assertIn("No reviewer responses", out)
        # Must NOT render the zero-count line when there is no data at all
        self.assertNotIn("REVISE_PLAN=0", out)

    def test_missing_aggregate_line_yields_review_required(self):
        """A non-error response with no AGGREGATE: line → REVIEW_REQUIRED (not GREEN)."""
        results = [
            {"model": "model-a", "text": "Looks generally fine but some concerns.", "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        # model-a has no aggregate line → should pull aggregate down to REVIEW_REQUIRED
        self.assertIn("REVIEW_REQUIRED", out)
        self.assertNotIn("\n## Aggregate Signal: GREEN", out)

    def test_duplicate_aggregate_lines_fail_closed(self):
        results = [
            {"model": "model-a", "text": "AGGREGATE: GREEN\nAGGREGATE: GREEN", "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        aggregate_section = self._run(results).split("## Aggregate")[1]
        self.assertIn("**REVIEW_REQUIRED**", aggregate_section)
        self.assertNotIn("**GREEN**", aggregate_section)

    def test_all_reviewer_sections_present(self):
        results = [
            {"model": "model-a", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-b", "text": _contract_review_text("GREEN"), "error": None},
            {"model": "model-c", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        self.assertIn("Reviewer 1", out)
        self.assertIn("Reviewer 2", out)
        self.assertIn("Reviewer 3", out)

    def test_goal_and_token_estimate_in_output(self):
        results = [
            {"model": "model-a", "text": _contract_review_text("GREEN"), "error": None},
        ]
        out = self._run(results)
        self.assertIn("test goal", out)
        self.assertIn("12,345", out)


def test_oversized_planned_artifact_degrades_loudly_to_minimal(tmp_path, monkeypatch):
    """A touched file too large for both snapshot and Atlas remains a loud omission,
    while the same fingerprint/scout wave still receives a minimal panel review."""
    import asyncio
    import subprocess

    from ouroboros.tools import plan_review as pr

    repo = tmp_path / "repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts" / "huge.md").write_text("x" * 1_200_000, encoding="utf-8")
    (repo / "ok.py").write_text("print(1)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(repo), capture_output=True,
    )

    drive = tmp_path / "drive"
    drive.mkdir()
    ctx = _make_ctx(drive)
    ctx.repo_dir = repo

    dispatched = {"called": False}

    async def _fake_slots(_ctx, models, _system, _user, user_stable_len=0, slot_ids=None):
        dispatched["called"] = True
        return [
            {"model": str(m), "text": _review_text("GREEN"), "error": None}
            for m in models
        ]

    monkeypatch.setattr(pr, "_load_plan_checklist", lambda: "checklist")
    monkeypatch.setattr(pr, "load_governance_doc", lambda *_a, **_k: "")
    monkeypatch.setattr(pr, "_start_planning_swarm", _completed_planning_swarm)
    monkeypatch.setattr(pr, "review_wave_budget_gate", lambda *_a, **_k: None)
    monkeypatch.setattr("ouroboros.config.get_review_models", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(pr, "_get_review_models", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(pr, "_run_plan_review_slots", _fake_slots)

    result = asyncio.run(pr._run_plan_review_async(
        ctx,
        _plan_request(
            "my plan", "my goal", ["prompts/huge.md", "ok.py"],
            context_level="constitutional",
        ),
    ))

    assert "PLAN_CONTEXT_DEGRADED" in result
    assert "effective context_level=minimal" in result
    assert "prompts/huge.md" in result
    assert "PLAN_REVIEW_SKIPPED" not in result
    assert dispatched["called"] is True

    # Control: a small planned file keeps the requested context without degradation.
    repo_ok = tmp_path / "repo_ok"
    repo_ok.mkdir()
    (repo_ok / "ok.py").write_text("print(1)\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=str(repo_ok), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(repo_ok), capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=T", "commit", "-m", "init"],
        cwd=str(repo_ok), capture_output=True,
    )
    drive_ok = tmp_path / "drive_ok"
    drive_ok.mkdir()
    ctx_ok = _make_ctx(drive_ok)
    ctx_ok.repo_dir = repo_ok
    dispatched["called"] = False
    result_ok = asyncio.run(pr._run_plan_review_async(
        ctx_ok,
        _plan_request(
            "my plan", "another goal", ["ok.py"], context_level="constitutional",
        ),
    ))
    assert "PLAN_REVIEW_SKIPPED" not in result_ok
    assert "PLAN_CONTEXT_DEGRADED" not in result_ok
    assert dispatched["called"] is True


class TestPlanReviewBudgetGate(unittest.IsolatedAsyncioTestCase):
    async def test_declined_wave_does_not_freeze_pre_dispatch_snapshot(self):
        import tempfile

        from ouroboros.tools import plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            ctx = _make_ctx(pathlib.Path(raw))
            freeze = MagicMock(return_value={"path": "unused"})
            atlas = SimpleNamespace(text="small atlas", manifest={}, status="ok")
            with (
                patch.object(pr, "compile_review_context_atlas", return_value=atlas) as compile_atlas,
                patch.object(pr, "build_head_snapshot_section", return_value=""),
                patch.object(pr, "_load_plan_checklist", return_value="checklist"),
                patch.object(pr, "load_governance_doc", return_value="doc"),
                patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
                patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
                patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
                patch.object(pr, "_persist_planning_snapshot", freeze),
                patch.object(pr, "review_wave_budget_gate", return_value={
                    "estimated_wave_usd": 2.0,
                    "remaining_usd": 1.0,
                    "limit_usd": 1.0,
                }) as admission,
                patch.object(pr, "_run_plan_review_slots", new=AsyncMock()) as slots,
            ):
                result = await pr._run_plan_review_async(
                    ctx,
                    _plan_request("my plan", "my goal", [], context_level="localized"),
                )

        self.assertIn("PLAN_REVIEW_SKIPPED_BUDGET", result)
        self.assertNotIn("PLAN_CONTEXT_DEGRADED", result)
        compile_atlas.assert_called_once()
        admission.assert_called_once()
        slots.assert_not_awaited()
        freeze.assert_not_called()

    async def test_degraded_minimal_budget_refusal_keeps_loud_disclosure(self):
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        atlas = SimpleNamespace(
            text="",
            manifest={
                "unassembled_required": [{
                    "path": "huge.py", "reason": "required file exceeded the atlas hard budget",
                }],
            },
            status="required_artifact_omitted",
        )
        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
            patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "review_wave_budget_gate", return_value={
                "estimated_wave_usd": 2.0, "remaining_usd": 1.0, "limit_usd": 1.0,
            }),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock()) as slots,
        ):
            result = await pr._run_plan_review_async(
                ctx, _plan_request("P", "G", [], context_level="constitutional"),
            )

        self.assertIn("PLAN_CONTEXT_DEGRADED", result)
        self.assertIn("huge.py", result)
        self.assertIn("PLAN_REVIEW_SKIPPED_BUDGET", result)
        self.assertIn("effective review packet retained", result)
        self.assertNotIn("Reviewers received", result)
        slots.assert_not_awaited()

    async def test_quorum_oversize_degrades_same_call_to_minimal(self):
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")
        atlas = SimpleNamespace(text="x" * 1_000_000, manifest={}, status="budget_constrained")
        reviews = [
            {"model": model, "text": _review_text("GREEN"), "error": None}
            for model in ("model-a", "model-b")
        ]

        def sized(text):
            return 1_100_000 if len(text) > 500_000 else 10_000

        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "build_head_snapshot_section", return_value=("", frozenset())),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
            patch("ouroboros.config.get_review_models",
                  return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch("ouroboros.tools.plan_review.estimate_tokens", side_effect=sized),
            patch.object(pr, "review_wave_budget_gate", return_value=None),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock(return_value=reviews)) as slots,
        ):
            result = await pr._run_plan_review_async(
                ctx,
                _plan_request(
                    "my plan", "my goal", [], context_level="constitutional",
                ),
            )

        self.assertIn("PLAN_CONTEXT_DEGRADED", result)
        self.assertIn("requested context_level=constitutional", result)
        self.assertIn("effective context_level=minimal", result)
        self.assertNotIn("PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE", result)
        slots.assert_awaited_once()

    async def test_minimal_still_oversize_stops_after_one_fallback(self):
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        atlas = SimpleNamespace(text="small atlas", manifest={}, status="ok")
        swarm = MagicMock(side_effect=_completed_planning_swarm)
        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", swarm),
            patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "estimate_tokens", return_value=1_100_000),
            patch.object(
                pr, "review_wave_budget_gate", side_effect=AssertionError("fit must stop first"),
            ),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock()) as slots,
        ):
            result = await pr._run_plan_review_async(
                ctx, _plan_request("P", "G", [], context_level="constitutional"),
            )

        self.assertIn("PLAN_CONTEXT_DEGRADED", result)
        self.assertIn("effective context_level=minimal", result)
        self.assertIn("PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE", result)
        self.assertIn("effective review packet retained", result)
        self.assertNotIn("Reviewers received", result)
        swarm.assert_called_once()
        slots.assert_not_awaited()

    async def test_required_artifact_omission_degrades_loudly_and_persists_resolution(self):
        from ouroboros.tools import plan_review as pr
        from ouroboros.task_results import load_plan_review_state

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")
        atlas = SimpleNamespace(
            text="atlas without ouroboros/llm.py",
            manifest={
                "estimated_total_tokens": 500_000,
                "unassembled_required": [
                    {"path": "ouroboros/llm.py", "reason": "required file exceeded the atlas hard budget"}
                ],
            },
            status="required_artifact_omitted",
        )
        reviews = [
            {"model": model, "text": _review_text("GREEN"), "error": None}
            for model in ("model-a", "model-b")
        ]
        swarm = MagicMock(side_effect=_completed_planning_swarm)
        request = _plan_request("my plan", "my goal", [], context_level="constitutional")
        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "build_head_snapshot_section", return_value=("", frozenset())),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", swarm),
            patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch("ouroboros.tools.plan_review.estimate_tokens", return_value=10_000),
            patch.object(pr, "review_wave_budget_gate", return_value=None),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock(return_value=reviews)) as slots,
        ):
            result = await pr._run_plan_review_async(ctx, request)

        self.assertIn("PLAN_CONTEXT_DEGRADED", result)
        self.assertIn("ouroboros/llm.py", result)
        self.assertNotIn("PLAN_REVIEW_SKIPPED", result)
        swarm.assert_called_once()
        slots.assert_awaited_once()
        state = load_plan_review_state(pathlib.Path(ctx.drive_root), ctx.task_id)
        wave = state["waves"][0]
        self.assertEqual(wave["request_fingerprint"], swarm.call_args.args[2])
        self.assertEqual(wave["review"]["requested_context_level"], "constitutional")
        self.assertEqual(wave["review"]["effective_context_level"], "minimal")
        cached = pr._reuse_or_disposition_plan_review(
            ctx, wave["request_fingerprint"], None, pr.plan_text_fingerprint(request.plan),
        )
        self.assertIn("PLAN_CONTEXT_DEGRADED", cached)

    async def test_unexpected_atlas_compiler_exception_does_not_fallback(self):
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")
        with (
            patch.object(pr, "compile_review_context_atlas", side_effect=RuntimeError("boom")),
            patch.object(pr, "build_head_snapshot_section", return_value=("", frozenset())),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
            patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock()) as slots,
        ):
            result = await pr._run_plan_review_async(
                ctx, _plan_request("my plan", "my goal", [], context_level="localized"),
            )

        self.assertIn("Failed to build review context atlas: boom", result)
        self.assertNotIn("PLAN_CONTEXT_DEGRADED", result)
        slots.assert_not_awaited()

    async def test_mixed_assembly_failure_degrades_and_discloses_both_causes(self):
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")
        atlas = SimpleNamespace(
            text="",
            manifest={
                "status": "budget_exceeded",
                "estimated_total_tokens": 950_000,
                "unassembled_required": [
                    {"path": "ouroboros/llm.py", "reason": "required file exceeded the atlas hard budget"}
                ],
            },
            status="budget_exceeded",
        )
        reviews = [
            {"model": model, "text": _review_text("GREEN"), "error": None}
            for model in ("model-a", "model-b")
        ]
        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "build_head_snapshot_section", return_value=("", frozenset())),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
            patch("ouroboros.config.get_review_models", return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch("ouroboros.tools.plan_review.estimate_tokens", return_value=10_000),
            patch.object(pr, "review_wave_budget_gate", return_value=None),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock(return_value=reviews)) as slots,
        ):
            result = await pr._run_plan_review_async(
                ctx,
                _plan_request("my plan", "my goal", [], context_level="constitutional"),
            )

        self.assertIn("PLAN_CONTEXT_DEGRADED", result)
        self.assertNotIn("PLAN_REVIEW_SKIPPED", result)
        self.assertIn("ouroboros/llm.py", result)
        self.assertIn("exceeded hard budget", result)
        slots.assert_awaited_once()

    async def test_proceeds_when_within_budget(self):
        """When prompt is within budget, reviewers are called."""
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        ctx.repo_dir = pathlib.Path(".")

        mock_result = {
            "model": "model-a",
            "text": "All good.\nAGGREGATE: GREEN",
            "error": None,
            "tokens_in": 100,
            "tokens_out": 50,
        }
        atlas = SimpleNamespace(text="small atlas", manifest={}, status="ok")

        with (
            patch.object(pr, "compile_review_context_atlas", return_value=atlas),
            patch.object(pr, "build_head_snapshot_section", return_value=("", frozenset())),
            patch.object(pr, "_load_plan_checklist", return_value="checklist"),
            patch.object(pr, "load_governance_doc", return_value=""),
            patch.object(pr, "_start_planning_swarm", side_effect=_completed_planning_swarm),
            # Two distinct models so the quorum gate (v4.39.0) passes and we
            # actually reach the reviewer-call path under test. Patch both
            # `_cfg.get_review_models` and `pr._get_review_models` to stay
            # hermetic against developer `OUROBOROS_REVIEW_MODELS`.
            patch("ouroboros.config.get_review_models",
                  return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
            patch("ouroboros.tools.plan_review.estimate_tokens", return_value=10_000),
            patch.object(pr, "_run_plan_review_slots", new=AsyncMock(return_value=[mock_result, mock_result])),
        ):
            result = await pr._run_plan_review_async(
                ctx,
                _plan_request("my plan", "my goal", [], context_level="localized"),
            )

        self.assertIn("Plan Review Results", result)
        self.assertIn("GREEN", result)

    async def test_context_level_must_be_agent_chosen_explicitly(self):
        """plan_task must not use host-side auto heuristics for context selection."""
        from ouroboros.tools import plan_review as pr

        ctx = _make_ctx()
        with (
            patch("ouroboros.config.get_review_models",
                  return_value=["model-a", "model-b"]),
            patch.object(pr, "_get_review_models", return_value=["model-a", "model-b"]),
        ):
            result = await pr._run_plan_review_async(
                ctx,
                _plan_request("my plan", "my goal", []),
            )

        self.assertIn("ERROR", result)
        self.assertIn("explicit context_level", result)
        self.assertIn("host-side auto", result)


class TestParseAggregateSignal(unittest.TestCase):
    def setUp(self):
        from ouroboros.tools.plan_review import _parse_aggregate_signal
        self.parse = _parse_aggregate_signal

    def test_detects_green(self):
        self.assertEqual(self.parse("AGGREGATE: GREEN"), "GREEN")

    def test_detects_review_required(self):
        self.assertEqual(self.parse("AGGREGATE: REVIEW_REQUIRED"), "REVIEW_REQUIRED")

    def test_detects_revise_plan(self):
        self.assertEqual(self.parse("AGGREGATE: REVISE_PLAN"), "REVISE_PLAN")

    def test_case_insensitive(self):
        self.assertEqual(self.parse("aggregate: green"), "GREEN")

    def test_allows_leading_whitespace(self):
        self.assertEqual(self.parse("  AGGREGATE: REVISE_PLAN"), "REVISE_PLAN")

    def test_returns_empty_when_no_aggregate_line(self):
        text = "This is not a REVISE_PLAN case — the situation is fine.\nLooks GREEN to me overall."
        self.assertEqual(self.parse(text), "")

    def test_body_text_does_not_false_positive(self):
        """Reviewer explaining 'This would be REVISE_PLAN if X' should not trigger signal."""
        text = "Normally this would be REVISE_PLAN but in this case it is acceptable.\nAGGREGATE: REVIEW_REQUIRED"
        self.assertEqual(self.parse(text), "REVIEW_REQUIRED")

    def test_duplicate_aggregate_lines_are_invalid(self):
        text = "AGGREGATE: GREEN\nAGGREGATE: REVISE_PLAN"
        self.assertEqual(self.parse(text), "")

    def test_self_correction_must_not_emit_two_aggregate_controls(self):
        text = "AGGREGATE: REVIEW_REQUIRED\nAfter reconsideration:\nAGGREGATE: REVISE_PLAN"
        self.assertEqual(self.parse(text), "")


class TestPlanReviewToolRegistration(unittest.TestCase):
    def test_plan_task_schema_has_required_fields(self):
        from ouroboros.tools.plan_review import get_tools
        tool = next(t for t in get_tools() if t.name == "plan_task")
        params = tool.schema["parameters"]["properties"]
        self.assertIn("plan", params)
        self.assertIn("goal", params)
        self.assertIn("files_to_touch", params)
        self.assertIn("context_level", params)
        self.assertIn("scope", params)
        self.assertIn("review_disposition", params)
        disposition = params["review_disposition"]
        self.assertEqual(disposition["required"], ["review_fingerprint", "items"])
        decision = disposition["properties"]["items"]["items"]["properties"]["decision"]
        self.assertEqual(decision["enum"], ["accept", "reject", "defer"])
        # context_level is NOT schema-required (triad r2 self_consistency): the host
        # enforces explicit choice for self_mod while non-self_mod may omit it
        # (defaults to minimal) — an unconditional `required` contradicted that.
        self.assertEqual(tool.schema["parameters"]["required"], [])
        self.assertIn("review_disposition ONLY", tool.schema["description"])
        self.assertNotIn("auto", params["context_level"].get("enum", []))
        claims = params["scope"]["properties"]["acceptance_claims"]
        self.assertEqual(claims["type"], "array")
        self.assertEqual(claims["items"], {"type": "string"})
        # No min-constraints by design (v6.65.1/.2: they shape placeholder junk).
        self.assertNotIn("minItems", claims)
        self.assertNotIn("minLength", claims.get("items", {}))

    def test_vacuous_acceptance_claims_detection(self):
        from ouroboros.tools.review_synthesis import vacuous_acceptance_claims

        self.assertTrue(vacuous_acceptance_claims({"acceptance_claims": []}))
        self.assertTrue(vacuous_acceptance_claims({"acceptance_claims": ["", "  "]}))
        self.assertTrue(vacuous_acceptance_claims({"acceptance_claims": None}))
        self.assertFalse(vacuous_acceptance_claims({"acceptance_claims": ["game boots"]}))
        self.assertFalse(vacuous_acceptance_claims({"in_scope": ["x"]}))
        self.assertFalse(vacuous_acceptance_claims(None))
        # Shape errors are normalize_plan_scope's job, not the vacuous note's.
        self.assertFalse(vacuous_acceptance_claims({"acceptance_claims": "game boots"}))

    def test_public_registry_rejects_unknown_plan_task_arguments(self):
        import tempfile

        from ouroboros.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory() as raw_root:
            root = pathlib.Path(raw_root)
            registry = ToolRegistry(repo_dir=root, drive_root=root)
            registry.override_handler(
                "plan_task",
                lambda _ctx, **_params: "handler-ran",
            )
            result = registry.execute(
                "plan_task",
                {
                    "plan": "Do the work",
                    "goal": "Ship it",
                    "review_dispositon": {},
                },
            )

        self.assertIn("TOOL_ARG_ERROR (plan_task)", result)
        self.assertIn("review_disposition", result)
        self.assertNotIn("review_dispositon", result)
        self.assertNotIn("handler-ran", result)

    def test_plan_review_deduplicates_canonical_docs_from_repo_pack(self):
        import inspect
        import ouroboros.tools.plan_review as pr

        # The body, not the resource-scope wrapper: `_run_plan_review_async` now owns
        # the review's ExitStack (a remote review holds a materialized mirror) and
        # delegates the review itself to `_run_plan_review_body`.
        source = inspect.getsource(pr._run_plan_review_body)
        assert '"BIBLE.md"' in source
        assert '"docs/DEVELOPMENT.md"' in source
        assert '"docs/ARCHITECTURE.md"' in source
        assert '"docs/CHECKLISTS.md"' in source

    def test_plan_review_prompt_points_to_plan_checklist_section(self):
        from ouroboros.tools.plan_review import _build_system_prompt

        prompt = _build_system_prompt("", "", "", "", "## Plan Review Checklist\n\n- completeness\n")

        assert "Use the `## Plan Review Checklist` section" in prompt
        assert "## CHECKLISTS.md" in prompt

    def test_plan_task_description_mentions_pre_implementation(self):
        from ouroboros.tools.plan_review import get_tools
        tool = next(t for t in get_tools() if t.name == "plan_task")
        desc = tool.schema["description"].lower()
        self.assertIn("before", desc)
        self.assertIn("code", desc)
        self.assertIn("planning-scout", desc)

    def test_plan_task_contract_has_no_active_heartbeat_or_inline_fallback_knob(self):
        import inspect

        from ouroboros.config import SETTINGS_DEFAULTS
        import ouroboros.tools.plan_review as plan_review
        from ouroboros.tools.plan_review import get_tools

        tool = next(t for t in get_tools() if t.name == "plan_task")
        description = tool.schema["description"].lower()
        self.assertIn("one shared", description)
        self.assertNotIn("heartbeat", description)
        self.assertNotIn("inline", description)
        self.assertEqual(
            SETTINGS_DEFAULTS["OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC"],
            120,
        )
        self.assertNotIn(
            "OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
            inspect.getsource(plan_review._collect_planning_handoffs),
        )


class TestPlanReviewIntentAndDisposition(unittest.TestCase):
    def test_goal_and_scope_share_one_context_block(self):
        import ouroboros.tools.plan_review as pr

        scope = {
            "in_scope": ["planning contract"],
            "invariants": ["no new ledger"],
            "non_goals": ["widget work"],
            "selected_seam": "plan_task_handoffs.json",
            "rejected_expansions": ["new review endpoint"],
        }
        user, _stable_len = pr._build_user_content(
            _plan_request(
                "Implement it",
                "Improve planning",
                context_level="minimal",
                scope=scope,
            ),
            "",
            "",
            "",
        )
        scout = pr._planning_swarm_context(
            plan="Implement it",
            goal="Improve planning",
            files_to_touch=[],
            context_level="minimal",
            context_notes="",
            scope=scope,
        )
        self.assertIn("Goal and Scope", user)
        self.assertIn('"goal": "Improve planning"', user)
        self.assertIn('"selected_seam": "plan_task_handoffs.json"', user)
        self.assertIn("[GOAL_AND_SCOPE]", scout)
        self.assertNotIn("[GOAL]\n", scout)

    def test_empty_scope_still_names_every_intent_boundary(self):
        import ouroboros.tools.plan_review as pr

        user, _stable_len = pr._build_user_content(
            _plan_request("P", "G", context_level="minimal"), "", "", "",
        )
        scout = pr._planning_swarm_context(
            plan="P", goal="G", files_to_touch=[], context_level="minimal",
            context_notes="", scope=None,
        )
        for field in ("in_scope", "invariants", "non_goals", "selected_seam", "rejected_expansions"):
            self.assertIn(f'"{field}"', user)
            self.assertIn(f'"{field}"', scout)

    def test_scope_and_evidence_shape_change_fingerprint(self):
        import ouroboros.tools.plan_review as pr

        base = dict(
            plan="P",
            goal="G",
            files_to_touch=[],
            context_level="minimal",
            context_notes="",
            plan_class="self_mod",
        )
        fp = pr._plan_request_fingerprint(**base)
        scoped = pr._plan_request_fingerprint(
            **base, scope={"non_goals": ["x"]}
        )
        with_tests = pr._plan_request_fingerprint(**base, include_tests=True)
        self.assertNotEqual(fp, scoped)
        self.assertNotEqual(fp, with_tests)

    def test_acceptance_claims_normalize_only_when_set(self):
        """Vacuous claims == absent (v6.65.1/.2 lesson: no min-constraints), and the
        key enters the normalized scope — hence the fingerprint — only when non-empty
        (v6.61.0 plan_class only-when-set precedent: historical fingerprints stay valid)."""
        from ouroboros.tools.review_synthesis import normalize_plan_scope

        populated = normalize_plan_scope(
            {"acceptance_claims": [" game boots ", "", "   ", "score persists"]}
        )
        self.assertEqual(populated["acceptance_claims"], ["game boots", "score persists"])
        for vacuous in ({}, {"acceptance_claims": []}, {"acceptance_claims": ["", "  "]}):
            self.assertNotIn("acceptance_claims", normalize_plan_scope(vacuous))
        with self.assertRaises(ValueError):
            normalize_plan_scope({"acceptance_claims": "game boots"})
        with self.assertRaises(ValueError):
            normalize_plan_scope({"acceptance_claims": [{"claim": "x"}]})

    def test_acceptance_claims_change_fingerprint_only_when_set(self):
        import ouroboros.tools.plan_review as pr

        base = dict(
            plan="P",
            goal="G",
            files_to_touch=[],
            context_level="minimal",
            context_notes="",
            plan_class="self_mod",
        )
        fp = pr._plan_request_fingerprint(**base)
        vacuous = pr._plan_request_fingerprint(**base, scope={"acceptance_claims": []})
        claimed = pr._plan_request_fingerprint(
            **base, scope={"acceptance_claims": ["game boots"]}
        )
        self.assertEqual(fp, vacuous)
        self.assertNotEqual(fp, claimed)

    def test_reviewer_prompt_is_generative_without_numeric_issue_quota(self):
        import ouroboros.tools.plan_review as pr

        prompt = pr._build_system_prompt("check", "", "", "")
        self.assertIn("there is no issue quota", prompt)
        self.assertIn("Do not invent findings to fill a quota", prompt)
        self.assertIn("PLAN_FINDINGS_JSON", prompt)
        self.assertIn("concrete defect, duplicated authority or coupling", prompt)
        self.assertIn("Diff size, line count, and file count are not findings", prompt)
        self.assertNotIn("top 1-2 ideas", prompt)
        self.assertNotIn("2-5 sentences", prompt)
        self.assertNotIn("fewer lines changed", prompt)

    def test_addressable_findings_receive_stable_slot_prefixed_ids(self):
        import ouroboros.tools.plan_review as pr

        text = _review_text("REVIEW_REQUIRED", [
            {
                "id": "missing-seam",
                "level": "RISK",
                "summary": "Existing seam is not named.",
                "recommendation": "Name the current helper.",
            },
            {
                "id": "scope-gap",
                "level": "FAIL",
                "summary": "A required boundary is absent.",
                "recommendation": "Add the non-goal explicitly.",
            },
        ])
        findings, error = pr._addressable_plan_findings(
            {"model": "m", "text": text, "error": None},
            reviewer_index=2,
            signal="REVIEW_REQUIRED",
        )
        self.assertEqual(error, "")
        self.assertEqual(
            [item["finding_id"] for item in findings],
            ["plan-slot-2:missing-seam", "plan-slot-2:scope-gap"],
        )

    def test_green_without_required_findings_block_fails_closed(self):
        import ouroboros.tools.plan_review as pr

        summary = pr._summarize_plan_review_results([{
            "model": "m",
            "text": "## PROPOSALS\n\nNo issue.\nAGGREGATE: GREEN",
            "error": None,
        }])
        self.assertEqual(summary["signals"], ["DEGRADED"])
        self.assertEqual(summary["aggregate_signal"], "REVIEW_REQUIRED")
        self.assertIn("PLAN_FINDINGS_JSON is missing", summary["projection_errors"][1])
        self.assertEqual(summary["findings"][0]["finding_id"], "plan-slot-1:findings-contract")

    def test_green_with_findings_block_after_aggregate_fails_closed(self):
        import ouroboros.tools.plan_review as pr

        summary = pr._summarize_plan_review_results([{
            "model": "m",
            "text": "AGGREGATE: GREEN\nPLAN_FINDINGS_JSON:\n[]",
            "error": None,
        }])
        self.assertEqual(summary["signals"], ["DEGRADED"])
        self.assertEqual(summary["aggregate_signal"], "REVIEW_REQUIRED")
        self.assertIn("must precede AGGREGATE", summary["projection_errors"][1])

    def _write_review(self, tmp_path, *, aggregate="REVIEW_REQUIRED"):
        from ouroboros.task_results import (
            plan_review_wave_handoffs,
            record_plan_review_attempt,
            record_plan_review_collection,
            record_plan_review_result,
            reserve_plan_review_wave,
        )
        from ouroboros.tools.plan_review import _persist_planning_handoffs
        from ouroboros.tools.registry import ToolContext

        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
        ctx.task_id = "parent"
        fingerprint = "a" * 64
        review = {
            "schema_version": 1,
            "request_fingerprint": fingerprint,
            "plan_text_hash": hashlib.sha256(b"P").hexdigest(),
            "aggregate_signal": aggregate,
            "closed": aggregate == "GREEN",
            "findings": [] if aggregate == "GREEN" else [
                {"finding_id": "plan-slot-1:f1", "summary": "one"},
                {"finding_id": "plan-slot-2:f2", "summary": "two"},
            ],
        }
        record_plan_review_attempt(tmp_path, "parent", fingerprint=fingerprint)
        reserve_plan_review_wave(
            tmp_path,
            "parent",
            fingerprint=fingerprint,
            plan_text_hash=hashlib.sha256(b"P").hexdigest(),
            scout_roles=[],
            cutoff_at="2099-01-01T00:00:00+00:00",
        )
        record_plan_review_collection(
            tmp_path,
            "parent",
            fingerprint=fingerprint,
            included_task_ids=[],
            omissions=[],
            stop_reason="",
        )
        wave = record_plan_review_result(
            tmp_path, "parent", fingerprint=fingerprint, review=review
        )
        _persist_planning_handoffs(ctx, plan_review_wave_handoffs(wave))
        return ctx, fingerprint

    def test_cached_review_reuse_preserves_original_wait_tasks_audit(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            plan_review_wave_handoffs,
            record_plan_review_collection,
            record_plan_review_result,
            record_plan_review_scout,
            reserve_plan_review_wave,
        )
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent-cached-wait"
            fingerprint = "c" * 64
            plan_hash = pr.plan_text_fingerprint("P")
            reserve_plan_review_wave(
                root,
                ctx.task_id,
                fingerprint=fingerprint,
                plan_text_hash=plan_hash,
                scout_roles=["planning-scout-1"],
                cutoff_at="2099-01-01T00:00:00+00:00",
            )
            record_plan_review_scout(
                root,
                ctx.task_id,
                fingerprint=fingerprint,
                role="planning-scout-1",
                schedule_status="started",
                task_ids=["scout-1"],
                reason="scheduled scout-1",
            )
            record_plan_review_collection(
                root,
                ctx.task_id,
                fingerprint=fingerprint,
                included_task_ids=[],
                omissions=[{
                    "role": "planning-scout-1",
                    "task_id": "scout-1",
                    "reason": "not_terminal_at_review_cutoff",
                }],
                stop_reason="ceiling",
            )
            wave = record_plan_review_result(
                root,
                ctx.task_id,
                fingerprint=fingerprint,
                review={
                    "request_fingerprint": fingerprint,
                    "plan_text_hash": plan_hash,
                    "aggregate_signal": "GREEN",
                    "closed": True,
                    "findings": [],
                },
            )
            original_wait = {
                "tasks": {"scout-1": {"status": "running", "result": "full detail"}},
                "all_terminal": False,
            }
            audit = plan_review_wave_handoffs(wave)
            audit["wait"] = original_wait
            self.assertNotIn("error", pr._persist_planning_handoffs(ctx, audit))

            reused = pr._reuse_or_disposition_plan_review(
                ctx, fingerprint, None, plan_hash,
            )
            self.assertIn("PLAN_REVIEW_OUTCOME: GREEN", reused)
            stored = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
            self.assertEqual(stored["wait"], original_wait)

    def test_public_handoff_artifact_cannot_forge_green_or_scout_ids(self):
        import asyncio
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import load_plan_review_state
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent-forged-audit"
            fingerprint = pr._plan_request_fingerprint(
                plan="P",
                goal="G",
                files_to_touch=[],
                context_level="minimal",
                context_notes="",
                plan_class="external",
                scope={},
                include_tests=False,
            )
            path = pr._planning_handoff_path(ctx)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "schema_version": 1,
                "request_fingerprint": fingerprint,
                "task_ids": ["attacker-scout"],
                "included_task_ids": ["attacker-scout"],
                "review": {
                    "request_fingerprint": fingerprint,
                    "aggregate_signal": "GREEN",
                    "closed": True,
                },
            }), encoding="utf-8")
            with (
                patch("ouroboros.config.get_review_models", return_value=["m1", "m2"]),
                patch.object(pr, "_get_review_models", return_value=["m1", "m2"]),
                patch.object(
                    pr,
                    "_start_planning_swarm",
                    return_value={
                        "started": False,
                        "failure_class": "",
                        "error": "ERROR: fresh host wave required",
                    },
                ) as start,
            ):
                out = asyncio.run(pr._run_plan_review_async(
                    ctx,
                    _plan_request(
                        "P", "G", [], context_level="minimal", plan_class="external",
                    ),
                ))
            self.assertEqual(out, "ERROR: fresh host wave required")
            start.assert_called_once()
            self.assertEqual(load_plan_review_state(root, ctx.task_id)["waves"], [])

    def test_fingerprint_history_survives_a_b_a_without_duplicate_wave(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            load_plan_review_state,
            record_plan_review_collection,
            record_plan_review_result,
            reserve_plan_review_wave,
        )
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent-a-b-a"
            fingerprints = ("a" * 64, "b" * 64)
            plan_hashes = tuple(pr.plan_text_fingerprint(text) for text in ("A", "B"))
            for fingerprint, plan_hash in zip(fingerprints, plan_hashes):
                reserve_plan_review_wave(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    plan_text_hash=plan_hash,
                    scout_roles=[],
                    cutoff_at="2099-01-01T00:00:00+00:00",
                )
                record_plan_review_collection(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    included_task_ids=[],
                    omissions=[],
                    stop_reason="",
                )
                record_plan_review_result(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    review={
                        "request_fingerprint": fingerprint,
                        "plan_text_hash": plan_hash,
                        "aggregate_signal": "GREEN",
                        "closed": True,
                        "findings": [],
                    },
                )

            reused = pr._reuse_or_disposition_plan_review(
                ctx, fingerprints[0], None, plan_hashes[0]
            )
            self.assertIn("PLAN_REVIEW_OUTCOME: GREEN", reused)
            self.assertIn("Cached exact review:** True", reused)
            state = load_plan_review_state(root, ctx.task_id)
            self.assertEqual(
                [wave["request_fingerprint"] for wave in state["waves"]],
                list(fingerprints),
            )
            self.assertEqual(state["latest_review_fingerprint"], fingerprints[1])

    def test_open_a_b_a_is_represented_then_accepts_disposition(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            load_plan_review_state,
            record_plan_review_attempt,
            record_plan_review_collection,
            record_plan_review_result,
            reserve_plan_review_wave,
        )
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent-open-a-b-a"
            a_fp, b_fp = "a" * 64, "b" * 64
            a_hash, b_hash = pr.plan_text_fingerprint("A"), pr.plan_text_fingerprint("B")
            for fingerprint, plan_hash, aggregate in (
                (a_fp, a_hash, "REVIEW_REQUIRED"),
                (b_fp, b_hash, "GREEN"),
            ):
                reserve_plan_review_wave(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    plan_text_hash=plan_hash,
                    scout_roles=[],
                    cutoff_at="2099-01-01T00:00:00+00:00",
                )
                record_plan_review_collection(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    included_task_ids=[],
                    omissions=[],
                    stop_reason="",
                )
                record_plan_review_result(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    review={
                        "request_fingerprint": fingerprint,
                        "plan_text_hash": plan_hash,
                        "aggregate_signal": aggregate,
                        "closed": aggregate == "GREEN",
                        "findings": [] if aggregate == "GREEN" else [
                            {"finding_id": "plan-slot-1:f1", "summary": "fix A"}
                        ],
                    },
                )

            # A valid A re-presentation selects A as the current authority before
            # the free reference-only disposition call.
            record_plan_review_attempt(root, ctx.task_id, fingerprint=a_fp)
            represented = pr._reuse_or_disposition_plan_review(ctx, a_fp, None, a_hash)
            self.assertIn("PLAN_REVIEW_OUTCOME: REVIEW_REQUIRED", represented)
            self.assertIn("Cached exact review:** True", represented)
            self.assertEqual(
                load_plan_review_state(root, ctx.task_id)["latest_review_fingerprint"],
                a_fp,
            )
            closed = pr._reuse_or_disposition_plan_review(
                ctx,
                a_fp,
                {
                    "review_fingerprint": a_fp,
                    "items": [{
                        "finding_id": "plan-slot-1:f1",
                        "decision": "reject",
                        "rationale": "The proposed concern is outside the stated scope.",
                    }],
                },
                a_hash,
            )
            self.assertIn('"outcome":"REVIEW_REQUIRED","closed":true', closed)

    def test_complete_disposition_closes_same_fingerprint_without_review_call(self):
        import tempfile
        import ouroboros.tools.plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            out = pr._reuse_or_disposition_plan_review(
                ctx,
                fingerprint,
                {
                    "review_fingerprint": fingerprint,
                    "items": [
                        {
                            "finding_id": "plan-slot-1:f1",
                            "decision": "accept",
                            "rationale": "The adjustment is valid.",
                            "plan_revision": "Implementation step 2 uses the named seam.",
                        },
                        {
                            "finding_id": "plan-slot-2:f2",
                            "decision": "reject",
                            "rationale": "The cited path is outside the stated non-goals.",
                        },
                    ],
                },
            )
            self.assertIn("PLAN_REVIEW_OUTCOME: REVIEW_REQUIRED", out)
            self.assertIn(
                'PLAN_REVIEW_CONTROL_JSON: {"outcome":"REVIEW_REQUIRED","closed":true}',
                out,
            )
            self.assertNotIn("REVIEW_REQUIRED_CLOSED", out)
            stored = json.loads(pr._planning_handoff_path(ctx).read_text(encoding="utf-8"))
            self.assertTrue(stored["review"]["closed"])
            self.assertEqual(len(stored["review"]["disposition"]["items"]), 2)

    def test_legacy_integrated_review_without_status_accepts_disposition(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            PLAN_REVIEW_STATE_KEY,
            load_plan_review_state,
            task_result_path,
        )

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            result_path = task_result_path(root, ctx.task_id, create=False)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result[PLAN_REVIEW_STATE_KEY]["waves"][0].pop("review_evidence_status")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertNotIn(
                "review_evidence_status",
                load_plan_review_state(root, ctx.task_id)["waves"][0],
            )

            out = pr._handle_plan_task(ctx, review_disposition={
                "review_fingerprint": fingerprint,
                "items": [
                    {
                        "finding_id": "plan-slot-1:f1",
                        "decision": "reject",
                        "rationale": "The finding does not change the selected seam.",
                    },
                    {
                        "finding_id": "plan-slot-2:f2",
                        "decision": "reject",
                        "rationale": "The finding is outside the explicit non-goals.",
                    },
                ],
            })

            self.assertIn('"outcome":"REVIEW_REQUIRED","closed":true', out)
            stored = load_plan_review_state(root, ctx.task_id)["waves"][0]
            self.assertEqual(stored["review_evidence_status"], "integrated")

    def test_legacy_open_review_without_current_attempt_accepts_disposition(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            PLAN_REVIEW_STATE_KEY,
            load_plan_review_state,
            plan_review_gate_projection,
            task_result_path,
        )

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            result_path = task_result_path(root, ctx.task_id, create=False)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result[PLAN_REVIEW_STATE_KEY].pop("current_attempt")
            result_path.write_text(json.dumps(result), encoding="utf-8")

            legacy = load_plan_review_state(root, ctx.task_id)
            self.assertEqual(
                legacy["current_attempt"],
                {
                    "fingerprint": fingerprint,
                    "status": "open",
                    "reason": "legacy_latest_review",
                },
            )
            self.assertFalse(plan_review_gate_projection(legacy, "blocking")["allow"])

            disposition = {
                "review_fingerprint": fingerprint,
                "items": [
                    {
                        "finding_id": "plan-slot-1:f1",
                        "decision": "reject",
                        "rationale": "The finding does not change the selected seam.",
                    },
                    {
                        "finding_id": "plan-slot-2:f2",
                        "decision": "reject",
                        "rationale": "The finding is outside the explicit non-goals.",
                    },
                ],
            }
            out = pr._handle_plan_task(ctx, review_disposition=disposition)

            self.assertIn('"outcome":"REVIEW_REQUIRED","closed":true', out)
            stored = load_plan_review_state(root, ctx.task_id)
            self.assertEqual(stored["current_attempt"], legacy["current_attempt"])
            self.assertTrue(plan_review_gate_projection(stored, "blocking")["allow"])

            result = json.loads(result_path.read_text(encoding="utf-8"))
            result[PLAN_REVIEW_STATE_KEY].pop("current_attempt")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            replayed = pr._handle_plan_task(ctx, review_disposition=disposition)
            self.assertIn('"outcome":"REVIEW_REQUIRED","closed":true', replayed)

    def test_closed_disposition_replay_is_idempotent_and_contradiction_is_rejected(self):
        import tempfile

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import load_plan_review_state

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            disposition = {
                "review_fingerprint": fingerprint,
                "items": [
                    {
                        "finding_id": "plan-slot-1:f1",
                        "decision": "accept",
                        "rationale": "The adjustment is valid.",
                        "plan_revision": "Step 2 now names the existing seam.",
                    },
                    {
                        "finding_id": "plan-slot-2:f2",
                        "decision": "reject",
                        "rationale": "The cited path is outside the stated scope.",
                    },
                ],
            }
            pr._reuse_or_disposition_plan_review(ctx, fingerprint, disposition)
            before = load_plan_review_state(root, ctx.task_id)

            replayed = pr._reuse_or_disposition_plan_review(ctx, fingerprint, disposition)
            self.assertIn("Cached exact review:** True", replayed)
            self.assertEqual(load_plan_review_state(root, ctx.task_id), before)

            contradictory = {
                **disposition,
                "items": [
                    disposition["items"][0],
                    {
                        **disposition["items"][1],
                        "rationale": "A contradictory replacement rationale.",
                    },
                ],
            }
            rejected = pr._reuse_or_disposition_plan_review(
                ctx, fingerprint, contradictory,
            )
            self.assertIn("PLAN_REVIEW_DISPOSITION_IMMUTABLE", rejected)
            self.assertEqual(load_plan_review_state(root, ctx.task_id), before)

    def test_closed_disposition_is_also_immutable_inside_locked_state_update(self):
        import copy
        import tempfile

        import pytest

        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import (
            load_plan_review_state,
            plan_review_wave,
            record_plan_review_attempt,
            record_plan_review_result,
        )

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            disposition = {
                "review_fingerprint": fingerprint,
                "items": [
                    {
                        "finding_id": "plan-slot-1:f1",
                        "decision": "accept",
                        "rationale": "The adjustment is valid.",
                        "plan_revision": "Step 2 now names the existing seam.",
                    },
                    {
                        "finding_id": "plan-slot-2:f2",
                        "decision": "reject",
                        "rationale": "The cited path is outside the stated scope.",
                    },
                ],
            }
            pr._reuse_or_disposition_plan_review(ctx, fingerprint, disposition)
            stored = plan_review_wave(
                load_plan_review_state(root, ctx.task_id), fingerprint,
            )["review"]

            replay = copy.deepcopy(stored)
            replay["disposition"]["recorded_at"] = "2099-01-01T00:00:00+00:00"
            record_plan_review_result(
                root,
                ctx.task_id,
                fingerprint=fingerprint,
                review=replay,
                require_latest=True,
            )
            self.assertEqual(
                plan_review_wave(load_plan_review_state(root, ctx.task_id), fingerprint)["review"],
                stored,
            )

            contradictory = copy.deepcopy(stored)
            contradictory["disposition"]["items"][0]["rationale"] = "contradiction"
            with pytest.raises(ValueError, match="PLAN_REVIEW_DISPOSITION_IMMUTABLE"):
                record_plan_review_result(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    review=contradictory,
                    require_latest=True,
                )

            record_plan_review_attempt(
                root, ctx.task_id, fingerprint="b" * 64, reason="newer raw attempt",
            )
            before = load_plan_review_state(root, ctx.task_id)
            with pytest.raises(ValueError, match="PLAN_REVIEW_DISPOSITION_STALE"):
                record_plan_review_result(
                    root,
                    ctx.task_id,
                    fingerprint=fingerprint,
                    review=replay,
                    require_latest=True,
                )
            self.assertEqual(load_plan_review_state(root, ctx.task_id), before)

    def test_disposition_closes_near_deadline_without_configured_reviewers(self):
        import tempfile
        from datetime import datetime, timedelta, timezone

        import ouroboros.tools.plan_review as pr
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent"
            ctx.task_metadata = {
                "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=400)).isoformat()
            }
            fingerprint = pr._plan_request_fingerprint(
                plan="P", goal="G", files_to_touch=[], context_level="minimal",
                context_notes="", plan_class="external", scope={}, include_tests=False,
            )
            from ouroboros.task_results import (
                record_plan_review_attempt,
                record_plan_review_collection,
                record_plan_review_result,
                reserve_plan_review_wave,
            )

            record_plan_review_attempt(root, "parent", fingerprint=fingerprint)
            reserve_plan_review_wave(
                root,
                "parent",
                fingerprint=fingerprint,
                plan_text_hash=pr.plan_text_fingerprint("P"),
                scout_roles=[],
                cutoff_at="2099-01-01T00:00:00+00:00",
            )
            record_plan_review_collection(
                root,
                "parent",
                fingerprint=fingerprint,
                included_task_ids=[],
                omissions=[],
                stop_reason="",
            )
            record_plan_review_result(
                root,
                "parent",
                fingerprint=fingerprint,
                review={
                    "request_fingerprint": fingerprint,
                    "plan_text_hash": pr.plan_text_fingerprint("P"),
                    "aggregate_signal": "REVIEW_REQUIRED",
                    "closed": False,
                    "findings": [{"finding_id": "plan-slot-1:f1"}],
                },
            )
            disposition = {
                "review_fingerprint": fingerprint,
                "items": [{
                    "finding_id": "plan-slot-1:f1",
                    "decision": "reject",
                    "rationale": "The cited risk is outside the explicit non-goal.",
                }],
            }
            with (
                patch("ouroboros.config.get_review_models", return_value=[]),
                patch.object(pr, "_get_review_models", side_effect=AssertionError("no reviewer call")),
                patch.object(
                    pr, "_record_raw_plan_request_attempt",
                    side_effect=AssertionError("disposition-only must not create a raw attempt"),
                ),
            ):
                out = pr._handle_plan_task(ctx, review_disposition=disposition)
            self.assertIn('"outcome":"REVIEW_REQUIRED","closed":true', out)
            self.assertNotIn("PLAN_TASK_SKIPPED_DEADLINE", out)

    def test_malformed_disposition_fails_closed(self):
        import tempfile
        import ouroboros.tools.plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            ctx, fingerprint = self._write_review(pathlib.Path(raw))
            missing = pr._reuse_or_disposition_plan_review(
                ctx,
                fingerprint,
                {
                    "review_fingerprint": fingerprint,
                    "items": [{
                        "finding_id": "plan-slot-1:f1",
                        "decision": "accept",
                        "rationale": "yes",
                    }],
                },
            )
            self.assertIn("PLAN_REVIEW_DISPOSITION_INVALID", missing)
            self.assertIn("plan_revision", missing)
            stale = pr._reuse_or_disposition_plan_review(
                ctx,
                fingerprint,
                {"review_fingerprint": "b" * 64, "items": []},
            )
            self.assertIn("PLAN_REVIEW_DISPOSITION_STALE", stale)
            for bad_id in ("plan-slot-1:f1", "unknown"):
                items = [
                    {"finding_id": "plan-slot-1:f1", "decision": "reject", "rationale": "one"},
                    {"finding_id": bad_id, "decision": "reject", "rationale": "two"},
                ]
                invalid = pr._reuse_or_disposition_plan_review(
                    ctx, fingerprint, {"review_fingerprint": fingerprint, "items": items}
                )
                self.assertIn("PLAN_REVIEW_DISPOSITION_INVALID", invalid)

    def test_vacuous_disposition_is_absent_on_first_submission(self):
        # Models fill optional object params with empty defaults; an empty
        # disposition has no closing power, so it must behave like omission
        # (proceed to a fresh review wave), never PLAN_REVIEW_DISPOSITION_STALE.
        import tempfile
        import ouroboros.tools.plan_review as pr
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent"
            for vacuous in (
                {"review_fingerprint": "", "items": []},
                {"review_fingerprint": "   ", "items": []},
                {"items": []},
                {"review_fingerprint": ""},
                {},
            ):
                out = pr._reuse_or_disposition_plan_review(
                    ctx, "c" * 64, vacuous, hashlib.sha256(b"P").hexdigest()
                )
                self.assertIsNone(out, f"vacuous={vacuous!r} must mean absent")

    def test_vacuous_disposition_matches_none_path_after_review(self):
        import tempfile
        import ouroboros.tools.plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            ctx, fingerprint = self._write_review(pathlib.Path(raw))
            as_none = pr._reuse_or_disposition_plan_review(ctx, fingerprint, None)
            as_vacuous = pr._reuse_or_disposition_plan_review(
                ctx, fingerprint, {"review_fingerprint": "", "items": []}
            )
            self.assertEqual(as_none, as_vacuous)
            self.assertIn("PLAN_REVIEW_DISPOSITION_REQUIRED", as_vacuous)
            self.assertNotIn("PLAN_REVIEW_DISPOSITION_STALE", as_vacuous)

    def test_disposition_without_prior_review_is_stale_and_free(self):
        import tempfile
        import ouroboros.tools.plan_review as pr
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent"
            with patch.object(pr, "_run_plan_review_async") as run:
                out = pr._handle_plan_task(
                    ctx,
                    review_disposition={"review_fingerprint": "d" * 64, "items": []},
                )
            self.assertIn("PLAN_REVIEW_DISPOSITION_STALE", out)
            run.assert_not_called()
            self.assertFalse((root / "task_results" / "parent.json").exists())

    def test_mixed_disposition_and_plan_envelope_is_rejected_without_mutation(self):
        import tempfile
        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import load_plan_review_state

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            before = load_plan_review_state(root, ctx.task_id)
            disposition = {
                "review_fingerprint": fingerprint,
                "items": [
                    {"finding_id": "plan-slot-1:f1", "decision": "reject", "rationale": "one"},
                    {"finding_id": "plan-slot-2:f2", "decision": "reject", "rationale": "two"},
                ],
            }
            with patch.object(pr, "_record_raw_plan_request_attempt") as record, patch.object(
                pr, "_run_plan_review_async",
            ) as run:
                out = pr._handle_plan_task(
                    ctx, plan="P changed", goal="G", files_to_touch=["a.py", "b.py"],
                    review_disposition=disposition,
                )
            self.assertIn("PLAN_REVIEW_DISPOSITION_MIXED_ENVELOPE", out)
            record.assert_not_called()
            run.assert_not_called()
            self.assertEqual(load_plan_review_state(root, ctx.task_id), before)

    def test_state_lookup_failure_is_error_not_absence(self):
        # Consultation guard: an indeterminate state store must ERROR, never be
        # classified as "no review" (which would silently launch a paid wave).
        import tempfile
        import ouroboros.tools.plan_review as pr
        from ouroboros.tools.registry import ToolContext

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx = ToolContext(repo_dir=root, drive_root=root)
            ctx.task_id = "parent"
            results_dir = root / "task_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "parent.json").write_text("{corrupt", encoding="utf-8")
            out = pr._reuse_or_disposition_plan_review(
                ctx, "c" * 64,
                {"review_fingerprint": "d" * 64, "items": []},
                hashlib.sha256(b"P").hexdigest(),
            )
            self.assertIsNotNone(out)
            self.assertIn("PLAN_REVIEW_STATE_INVALID", out)
            self.assertNotIn("PLAN_REVIEW_DISPOSITION_UNBINDABLE", out)

    def test_disposition_cannot_revive_review_superseded_by_new_attempt(self):
        import tempfile
        import ouroboros.tools.plan_review as pr
        from ouroboros.task_results import load_plan_review_state, record_plan_review_attempt

        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ctx, fingerprint = self._write_review(root)
            record_plan_review_attempt(root, ctx.task_id, fingerprint="b" * 64)
            before = load_plan_review_state(root, ctx.task_id)
            with patch.object(pr, "_run_plan_review_async") as run:
                out = pr._handle_plan_task(ctx, review_disposition={
                    "review_fingerprint": fingerprint,
                    "items": [
                        {"finding_id": "plan-slot-1:f1", "decision": "reject", "rationale": "one"},
                        {"finding_id": "plan-slot-2:f2", "decision": "reject", "rationale": "two"},
                    ],
                })
            self.assertIn("PLAN_REVIEW_DISPOSITION_STALE", out)
            run.assert_not_called()
            self.assertEqual(load_plan_review_state(root, ctx.task_id), before)

    def test_vacuous_disposition_only_is_rejected_before_raw_attempt(self):
        import ouroboros.tools.plan_review as pr
        from ouroboros.tools.registry import ToolContext

        ctx = ToolContext(repo_dir=pathlib.Path("."), drive_root=pathlib.Path("."))
        ctx.task_id = "parent"
        with patch.object(pr, "_record_raw_plan_request_attempt") as record, patch.object(
            pr, "_run_plan_review_async",
        ) as run:
            out = pr._handle_plan_task(
                ctx, review_disposition={"review_fingerprint": "", "items": []},
            )
        self.assertIn("PLAN_REVIEW_DISPOSITION_EMPTY", out)
        self.assertIn("No plan attempt was recorded", out)
        record.assert_not_called()
        run.assert_not_called()

    def test_handle_plan_task_notes_ignored_vacuous_disposition(self):
        import ouroboros.tools.plan_review as pr
        from unittest.mock import patch
        from ouroboros.tools.registry import ToolContext

        async def _stub(ctx, request):
            assert request.review_disposition is None
            return "PLAN_REVIEW_OUTCOME: GREEN"

        ctx = ToolContext(repo_dir=pathlib.Path("."), drive_root=pathlib.Path("."))
        ctx.task_id = "parent"
        with patch.object(pr, "_record_raw_plan_request_attempt"), patch.object(
            pr, "_run_plan_review_async", _stub,
        ):
            out = pr._handle_plan_task(
                ctx,
                plan="P",
                goal="G",
                review_disposition={"review_fingerprint": "", "items": []},
            )
        self.assertIn("PLAN_REVIEW_OUTCOME: GREEN", out)
        self.assertIn("empty review_disposition was ignored", out)

    def test_revise_plan_cannot_be_overridden_by_disposition(self):
        import tempfile
        import ouroboros.tools.plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            ctx, fingerprint = self._write_review(
                pathlib.Path(raw), aggregate="REVISE_PLAN"
            )
            unchanged = pr._reuse_or_disposition_plan_review(ctx, fingerprint, None)
            self.assertIn("PLAN_REVIEW_REVISION_REQUIRED", unchanged)
            override = pr._reuse_or_disposition_plan_review(
                ctx,
                fingerprint,
                {"review_fingerprint": fingerprint, "items": []},
            )
            self.assertIn("PLAN_REVIEW_REVISION_REQUIRED", override)
            same_text = pr._reuse_or_disposition_plan_review(
                ctx, "b" * 64, None, pr.plan_text_fingerprint("P")
            )
            self.assertIn("requires changed plan text", same_text)
            changed_text = pr._reuse_or_disposition_plan_review(
                ctx, "b" * 64, None, pr.plan_text_fingerprint("P revised")
            )
            self.assertIsNone(changed_text)

    def test_green_exact_fingerprint_is_reused_without_new_wave(self):
        import tempfile
        import ouroboros.tools.plan_review as pr

        with tempfile.TemporaryDirectory() as raw:
            ctx, fingerprint = self._write_review(pathlib.Path(raw), aggregate="GREEN")
            out = pr._reuse_or_disposition_plan_review(ctx, fingerprint, None)
            self.assertIn("PLAN_REVIEW_OUTCOME: GREEN", out)
            self.assertIn("Cached exact review:** True", out)

    def test_contract_helpers_reuse_existing_review_synthesis_seam(self):
        import ouroboros.tools.plan_review as pr
        import ouroboros.tools.review_synthesis as synthesis

        self.assertIs(pr._format_output, synthesis.format_plan_review_output)
        self.assertIs(pr._summarize_plan_review_results, synthesis.summarize_plan_review_results)
        self.assertIs(pr._plan_request_fingerprint, synthesis.plan_review_fingerprint)
        self.assertIs(pr._format_planning_handoffs, synthesis.format_planning_handoffs)
        self.assertIs(pr._build_system_prompt, synthesis.build_plan_review_system_prompt)
        self.assertIs(pr._build_user_content, synthesis.build_plan_review_user_content)
        for outcome in ("GREEN", "REVIEW_REQUIRED", "REVISE_PLAN"):
            rendered = synthesis.render_plan_review_result({
                "request_fingerprint": "f" * 64,
                "aggregate_signal": outcome,
                "closed": outcome == "GREEN",
                "findings": [],
            })
            self.assertIn(f'"outcome":"{outcome}"', rendered)
            self.assertIn(f"PLAN_REVIEW_OUTCOME: {outcome}", rendered)
            self.assertNotIn("REVIEW_REQUIRED_CLOSED", rendered)

    def test_duplicate_plan_calls_use_existing_sequential_tool_lane(self):
        from ouroboros.loop_tool_execution import tool_calls_can_run_parallel

        calls = [
            {"function": {"name": "plan_task", "arguments": "{}"}},
            {"function": {"name": "plan_task", "arguments": "{}"}},
        ]
        self.assertFalse(tool_calls_can_run_parallel(calls))


class TestClassifyReviewerError(unittest.TestCase):
    """Tests for _classify_reviewer_error — readable error messages for reviewer failures."""

    def setUp(self):
        from ouroboros.tools.plan_review import _classify_reviewer_error
        self.classify = _classify_reviewer_error

    def test_json_decode_error_mentions_oversized_prompt(self):
        """JSONDecodeError → message explains the likely oversized-prompt root cause."""
        import json
        exc = json.JSONDecodeError("Expecting value", "", 0)
        msg = self.classify(exc, "openai/gpt-5.5")
        self.assertIn("non-JSON response body", msg)
        self.assertIn("oversized prompt", msg)
        self.assertIn("openai/gpt-5.5", msg)

    def test_json_decode_error_does_not_say_json_formatting_problem(self):
        """The user should not think it's a JSON format issue in our code."""
        import json
        exc = json.JSONDecodeError("Expecting value", "doc", 902)
        msg = self.classify(exc, "google/gemini-3.5-flash")
        # Should NOT say things like "JSON format" or "checklist formatting"
        self.assertNotIn("format", msg.lower().replace("non-JSON", ""))

    def test_json_decode_error_realistic_message(self):
        """Reproduces the exact JSONDecodeError seen in production logs."""
        import json
        # Exact args from the production failure:
        # "Expecting value: line 165 column 1 (char 902)"
        exc = json.JSONDecodeError("Expecting value", "line 165 column 1 (char 902)", 0)
        msg = self.classify(exc, "openai/gpt-5.5")
        self.assertIn("openai/gpt-5.5", msg)
        self.assertIn("non-JSON", msg)
        self.assertIn("oversized", msg)
        # The raw JSONDecodeError text should not be the ONLY content
        self.assertNotEqual(msg, str(exc))

    def test_generic_exception_preserves_type_and_message(self):
        """Unknown exception types fall back to 'TypeName: message' format."""
        exc = ValueError("something went wrong")
        msg = self.classify(exc, "some/model")
        self.assertIn("ValueError", msg)
        self.assertIn("something went wrong", msg)

    def test_timeout_error_fallback(self):
        """TimeoutError (if not caught before) is reported with its type."""
        exc = TimeoutError("connection timed out")
        msg = self.classify(exc, "my/model")
        self.assertIn("TimeoutError", msg)
        self.assertIn("connection timed out", msg)

    def test_model_name_always_included(self):
        """Model name should always appear in error message for traceability."""
        import json
        exc = json.JSONDecodeError("Expecting value", "", 0)
        msg = self.classify(exc, "very-specific/model-id-xyz")
        self.assertIn("very-specific/model-id-xyz", msg)

    def test_empty_exception_message_does_not_crash(self):
        """Exception with empty message string should not crash the helper."""
        exc = Exception("")
        msg = self.classify(exc, "test/model")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)


class TestPlanBudgetFollowsQuorum(unittest.TestCase):
    """The shared-prompt budget is the one a review QUORUM can read, not the global minimum.

    ``min(slot_limits.values())`` let one small-window slot dictate the Atlas for every reviewer:
    with caps [545K, 745K, 745K] and quorum 2 it threw away ~200K of context the two large slots
    could have read, and refused an irreducible 600K prompt those two would have accepted. The
    small slot must drop OUT of the quorum instead — and be RECORDED as not participating.
    """

    def setUp(self):
        from ouroboros.tools.review_synthesis import plan_slot_fit, quorum_input_token_limit

        self.limit = quorum_input_token_limit
        self.fit = plan_slot_fit
        self.models = ["small/545", "big/a", "big/b"]
        self.limits = {"small/545": 545_000, "big/a": 745_000, "big/b": 745_000}

    def test_reviewer_example_keeps_the_context_two_large_slots_can_read(self):
        self.assertEqual(self.limit(self.models, self.limits), 745_000)

    def test_600k_prompt_runs_on_the_quorum_and_records_the_slot_that_cannot_fit(self):
        callable_models, oversize, error = self.fit(self.models, self.limits, 600_000)
        self.assertEqual(callable_models, ["big/a", "big/b"])
        self.assertEqual(error, "")  # two of three IS the quorum — not a degradation
        self.assertEqual([rec["model"] for rec in oversize], ["small/545"])
        self.assertIn("preflight_oversize", oversize[0]["error"])
        self.assertEqual(oversize[0]["tokens_in"], 0)

    def test_quorum_preserving_exclusion_is_not_retryable_degradation(self):
        from ouroboros.tools.review_synthesis import summarize_plan_review_results

        _callable, oversize, _error = self.fit(self.models, self.limits, 600_000)
        raw = oversize + [
            {"model": model, "text": _review_text("GREEN"), "error": None}
            for model in ("big/a", "big/b")
        ]
        summary = summarize_plan_review_results(raw)

        self.assertEqual(summary["aggregate_signal"], "GREEN")
        self.assertEqual(summary["degraded_count"], 0)
        self.assertEqual(summary["preflight_excluded_count"], 1)

    def test_prompt_above_the_quorum_budget_still_fails_loudly(self):
        _callable, _oversize, error = self.fit(self.models, self.limits, 800_000)
        self.assertIn("PLAN_REVIEW_DEGRADED_PREFLIGHT_OVERSIZE", error)

    def test_unavailable_or_uncalibrated_slots_cannot_justify_a_bigger_prompt(self):
        # Only one slot has a cap at assembly time: the other two read 0, so the quorum-th
        # largest cap is 0 and nothing is assembled against a window nobody has.
        self.assertEqual(self.limit(self.models, {"big/a": 745_000}), 0)
        self.assertEqual(self.limit([], {}), 0)

    def test_small_configs_need_every_slot_so_the_budget_is_the_minimum(self):
        # adaptive_quorum: 1 slot -> 1, 2 slots -> both, 3+ -> 2 of N.
        self.assertEqual(self.limit(["a", "b"], {"a": 545_000, "b": 745_000}), 545_000)
        self.assertEqual(self.limit(["a"], {"a": 545_000}), 545_000)


if __name__ == "__main__":
    unittest.main()


def test_plan_task_at_zero_depth_finishes_on_degraded_evidence_without_a_wedge(monkeypatch, tmp_path):
    """v6.79.0 consequence of honouring depth=0 (owner Q26): planning scouts go through the
    SAME ``schedule_subagent`` gate, so a no-delegation run gets no scouts at all. plan_task
    must then complete on its existing ``degraded_evidence`` path — one refused attempt per
    intended scout, an explicit panel omission, a persisted audit artifact, and NO repeated
    scheduling on resume (the wave is not re-reserved and no scout is retried)."""
    import ouroboros.tools.plan_review as pr
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "0")
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_TIMEOUT_SEC", "0")
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_MAX_WAIT_SEC", "900")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent-zero-depth"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": ctx.task_id}
    request = _plan_request("P", "G", context_level="minimal")

    result = pr._start_planning_swarm(ctx, request, pr._plan_request_fingerprint(
        plan=request.plan, goal=request.goal, files_to_touch=request.files_to_touch,
        context_level=request.context_level, context_notes=request.context_notes,
        plan_class=request.plan_class, scope=request.scope,
        include_tests=request.include_tests,
    ))

    assert result["started"] is True
    assert result["degraded_evidence"] is True
    assert result["task_ids"] == []
    assert result["handoffs"]["omissions"]
    assert result["handoffs"]["omissions"][0]["reason"] == "schedule_failed"
    assert "depth limit (0) exceeded" in result["handoffs"]["omissions"][0]["detail"]
    assert (tmp_path / "task_results" / "artifacts" / ctx.task_id / "plan_task_handoffs.json").exists()

    scouts = load_plan_review_state(tmp_path, ctx.task_id)["waves"][0]["intended_scouts"]
    assert scouts and all(item["schedule_status"] == "failed" for item in scouts)

    # Resume of the same plan: still terminal, still no scheduling attempt, no extra wave.
    resumed = pr._start_planning_swarm(ctx, request, pr._plan_request_fingerprint(
        plan=request.plan, goal=request.goal, files_to_touch=request.files_to_touch,
        context_level=request.context_level, context_notes=request.context_notes,
        plan_class=request.plan_class, scope=request.scope,
        include_tests=request.include_tests,
    ))
    assert resumed["resumed"] is True and resumed["degraded_evidence"] is True
    assert len(load_plan_review_state(tmp_path, ctx.task_id)["waves"]) == 1
