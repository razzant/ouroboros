from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.review_evidence import build_task_acceptance_evidence
from ouroboros.review_substrate import (
    ReviewRequest,
    ReviewSlot,
    build_improvement_capsule,
    run_review_request,
)
from ouroboros import task_pacing
# The evidence horizon moved to `review_evidence`, whose whole job is assembling review
# evidence; `plan_review` was at its module ceiling and the function was in the wrong place.
from ouroboros.planning_evidence import planning_evidence_horizon as _planning_evidence_horizon
from ouroboros.tools.plan_review_setup import _resolve_plan_roots
from ouroboros.tools.registry import ToolContext
from ouroboros.usage_accounting import _claim_physical_dispatch
from ouroboros.utils import append_jsonl


def test_required_blocking_has_no_implicit_count_cap_but_explicit_cap_always_wins():
    snapshot = task_pacing.BudgetSnapshot(has_deadline=False)
    uncapped = normalize_budget_profile({})
    assert task_pacing.improvement_pass_allowed(
        snapshot, 999, uncapped, required_blocking=True,
    ) == (True, "")

    for policy in ("fixed", "adaptive", "until_deadline"):
        capped = normalize_budget_profile({
            "improvement_policy": policy,
            "max_improvement_passes": 6,
        })
        assert task_pacing.improvement_pass_allowed(
            snapshot, 6, capped, required_blocking=True,
        ) == (False, "improvement_passes_exhausted")


def test_system_prompt_describes_root_acceptance_as_evidence_only():
    import pathlib

    system = (
        pathlib.Path(__file__).resolve().parents[1] / "prompts" / "SYSTEM.md"
    ).read_text(encoding="utf-8")
    assert "For a root task in `task_review_mode=auto|required`" in system
    assert "this call is evidence-only" in system
    assert "single authoritative host panel" in system
    assert "Use `task_acceptance_review` for expensive independent critique" not in system


def test_acceptance_review_reserve_uses_existing_event_ewma(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "90")
    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    events = canonical / "logs" / "events.jsonl"
    append_jsonl(events, {"type": "task_acceptance_review_timing", "duration_sec": 100})
    append_jsonl(events, {"type": "task_acceptance_review_timing", "duration_sec": 400})
    ctx = SimpleNamespace(drive_root=str(child), budget_drive_root=str(canonical))
    # EWMA(alpha=.5) = 250; subsequent reserve = 1.5 * 250.
    assert task_pacing.acceptance_review_estimate_sec(ctx, passes_done=1) == 375.0
    assert task_pacing.acceptance_review_estimate_sec(ctx, passes_done=0) == 200.0
    assert task_pacing.acceptance_timing_events_path(ctx) == events
    assert not (child / "logs" / "events.jsonl").exists()


def test_acceptance_panel_persists_timing_to_canonical_root(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    import ouroboros.review_evidence as evidence_mod
    import ouroboros.review_substrate as substrate

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    tool_ctx = SimpleNamespace(
        drive_root=child,
        budget_drive_root=str(canonical),
        task_metadata={"budget_drive_root": str(canonical)},
    )
    monkeypatch.setattr(evidence_mod, "build_task_acceptance_evidence", lambda *_a, **_k: {})
    monkeypatch.setattr(substrate, "reviewer_slots", lambda **_k: [])
    monkeypatch.setattr(
        substrate,
        "run_review_request",
        lambda *_a, **_k: SimpleNamespace(aggregate_signal="PASS"),
    )
    ctx = loop._TaskAcceptanceContext(
        tools=SimpleNamespace(_ctx=tool_ctx),
        content="deliverable",
        task_id="root-timing",
        task_type="task",
        llm_trace={"tool_calls": []},
        drive_root=child,
        messages=[{"role": "system", "content": "policy"}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _text: None,
        mode="required",
        subtree_statuses=[],
        budget_profile={},
        passes_done=2,
    )

    loop._execute_task_acceptance_panel(ctx)

    rows = [json.loads(line) for line in (canonical / "logs" / "events.jsonl").read_text().splitlines()]
    assert rows[-1]["task_id"] == "root-timing"
    assert rows[-1]["pass_index"] == 2
    assert rows[-1]["aggregate_signal"] == "PASS"
    assert not (child / "logs" / "events.jsonl").exists()


def test_normalized_stall_default_does_not_emit_deprecation_noise(tmp_path):
    quiet = SimpleNamespace(
        drive_root=tmp_path,
        task_id="quiet",
        task_contract={"budget_profile": normalize_budget_profile({})},
    )
    task_pacing.resolve_budget_profile(quiet)
    events = tmp_path / "logs" / "events.jsonl"
    assert not events.exists()

    legacy = SimpleNamespace(
        drive_root=tmp_path,
        task_id="legacy",
        task_contract={"budget_profile": normalize_budget_profile({"stall_rounds_threshold": 2})},
    )
    task_pacing.resolve_budget_profile(legacy)
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["aliases"] == ["stall_rounds_threshold"]


def test_child_task_never_becomes_host_acceptance_authority():
    from ouroboros.loop import _task_acceptance_eligible

    assert _task_acceptance_eligible(
        "required", {"tool_calls": [{"tool": "write_file"}]}, False,
        is_root_task=False,
    ) == (False, "skipped_child_advisory")


def test_queue_owned_acceptance_fence_uses_only_optional_ctx_hooks():
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    calls = []

    def begin(**kwargs):
        calls.append(("begin", kwargs))
        return "fence-1"

    def end(**kwargs):
        calls.append(("end", kwargs))

    ctx = SimpleNamespace(
        task_metadata={"root_task_id": "root"},
        begin_acceptance_fence=begin,
        end_acceptance_fence=end,
    )
    assert _begin_task_acceptance_fence(ctx, "root") == (True, "fence-1")
    assert _end_task_acceptance_fence(ctx, outcome="revision") is True
    assert calls == [
        ("begin", {"root_task_id": "root", "task_id": "root"}),
        ("end", {"token": "fence-1", "outcome": "revision"}),
    ]


def test_acceptance_quiescence_does_not_treat_cancel_requested_as_settled(tmp_path, monkeypatch):
    from ouroboros.loop import _task_acceptance_subtree_snapshot
    import ouroboros.task_status as task_status

    monkeypatch.setattr(
        task_status,
        "find_child_tasks",
        lambda *_args, **_kwargs: [{
            "task_id": "child",
            "parent_task_id": "root",
            "status": "cancel_requested",
        }],
    )
    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_metadata={"root_task_id": "root"},
    )
    quiescent, rows = _task_acceptance_subtree_snapshot(ctx, tmp_path, "root")
    assert quiescent is False
    assert rows[0]["status"] == "cancel_requested"


def test_acceptance_subtree_uses_canonical_budget_root_for_split_drive(
    tmp_path, monkeypatch,
):
    from ouroboros.loop import _task_acceptance_subtree_snapshot
    from ouroboros.tools.join_ledger import _child_result_sha256
    import ouroboros.task_status as task_status

    canonical = tmp_path / "canonical-data"
    child = canonical / "state" / "headless_tasks" / "root" / "data"
    canonical.mkdir()
    child.mkdir(parents=True)
    captured = []

    child_result = {
        "task_id": "child",
        "parent_task_id": "root",
        "status": "completed",
    }

    def find_children(root, **_kwargs):
        captured.append(root)
        if pathlib.Path(root) != canonical:
            return []
        return [dict(child_result)]

    import pathlib
    monkeypatch.setattr(task_status, "find_child_tasks", find_children)
    ctx = SimpleNamespace(
        drive_root=child,
        budget_drive_root=str(canonical),
        task_metadata={
            "root_task_id": "root",
            "budget_drive_root": str(canonical),
        },
    )

    quiescent, rows = _task_acceptance_subtree_snapshot(ctx, child, "root")

    assert quiescent is True
    assert captured == [canonical]
    assert rows == [{
        "task_id": "child",
        "parent_task_id": "root",
        "status": "completed",
        "artifact_status": "",
        "child_result_sha256": _child_result_sha256(child_result),
    }]


def test_acceptance_quiescence_requires_empty_supervisor_snapshot(tmp_path, monkeypatch):
    from ouroboros.loop import _task_acceptance_subtree_snapshot
    import ouroboros.task_status as task_status

    monkeypatch.setattr(task_status, "find_child_tasks", lambda *_args, **_kwargs: [{
        "task_id": "child",
        "parent_task_id": "root",
        "status": "completed",
    }])
    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_metadata={"root_task_id": "root"},
        _task_acceptance_queue_descendants=[{"task_id": "child", "status": "running"}],
    )
    quiescent, rows = _task_acceptance_subtree_snapshot(ctx, tmp_path, "root")
    assert quiescent is False
    assert rows[-1] == {
        "task_id": "child",
        "parent_task_id": "",
        "status": "running",
        "artifact_status": "",
        "source": "supervisor_queue",
    }


def test_acceptance_quiescence_ignores_authoritative_plan_wave_scouts(
    tmp_path, monkeypatch,
):
    from ouroboros.loop import _task_acceptance_subtree_snapshot
    import ouroboros.task_results as task_results
    import ouroboros.task_status as task_status

    monkeypatch.setattr(task_results, "load_plan_review_state", lambda *_args: {})
    monkeypatch.setattr(
        task_results,
        "plan_review_audit_only_task_ids",
        lambda _state: ["planning-scout"],
    )
    monkeypatch.setattr(task_status, "find_child_tasks", lambda *_args, **_kwargs: [
        {
            "task_id": "planning-scout",
            "parent_task_id": "root",
            "status": "running",
        },
        {
            "task_id": "implementation-child",
            "parent_task_id": "root",
            "status": "completed",
        },
    ])
    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_metadata={"root_task_id": "root"},
        _task_acceptance_queue_descendants=[
            {"task_id": "planning-scout", "status": "running"},
        ],
    )

    quiescent, rows = _task_acceptance_subtree_snapshot(ctx, tmp_path, "root")

    assert quiescent is True
    assert [row["task_id"] for row in rows] == ["implementation-child"]


def test_acceptance_immutable_contract_is_never_silently_truncated(tmp_path):
    requirements = "owner requirement\n" * 30_000
    ctx = SimpleNamespace(
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={"requirements": requirements},
        repo_dir=tmp_path,
    )
    evidence = build_task_acceptance_evidence(
        ctx,
        task_id="root",
        canonical_subject="deliverable",
        subtree_statuses=[],
    )
    assert evidence["task_contract"]["requirements"] == requirements
    assert "__truncated__" not in evidence["task_contract"]
    assert evidence["__immutable_core_overflow__"]["reason"]
    assert isinstance(evidence["omissions_manifest"], list)
    assert evidence["aliases"]["root_task_id"] == "root"
    assert evidence["canonical_payload"]["source"] == "review_request.subject"


def test_acceptance_owner_corpus_preserves_followups_without_system_messages(tmp_path):
    ctx = SimpleNamespace(
        task_id="root",
        root_task_id="root",
        task_metadata={"root_task_id": "root"},
        task_contract={"objective": "Implement exactly the approved plan"},
        repo_dir=tmp_path,
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "Initial owner requirement"},
            {"role": "user", "content": "[SYSTEM REMINDER] internal"},
            {"role": "user", "content": "[Message from my human]: choose A"},
        ],
        _owner_directives=[
            {"source": "initial_user", "content": "Initial owner requirement"},
            {"source": "direct_incoming", "content": "choose A", "msg_id": "m1"},
        ],
    )
    evidence = build_task_acceptance_evidence(
        ctx,
        drive_root=tmp_path,
        task_id="root",
        canonical_subject="deliverable",
    )
    corpus = evidence["owner_requirements_and_decisions"]
    assert [row["content"] for row in corpus] == ["Initial owner requirement", "choose A"]
    assert corpus[1]["msg_id"] == "m1"
    assert "SYSTEM REMINDER" not in json.dumps(corpus)
    assert evidence["__provenance__"]["owner_requirements_and_decisions"] == "host_attested"


class _SplitVerdictLLM:
    def chat(self, **kwargs):
        verdict = "FAIL" if str(kwargs.get("model")) == "fail" else "PASS"
        findings = ([{
            "severity": "high",
            "item": "missing verification",
            "recommendation": "run the independent verification",
        }] if verdict == "FAIL" else [])
        return {"content": json.dumps({
            "verdict": verdict,
            "findings": findings,
            "summary": verdict,
        })}, {}


class _MinimalFailPanelLLM:
    def chat(self, **kwargs):
        if str(kwargs.get("model") or "") == "minimal-fail":
            return {"content": json.dumps({"verdict": "FAIL", "findings": []})}, {}
        return {"content": json.dumps({
            "verdict": "PASS",
            "outcome_tier": "solved",
            "completion_coach": "",
            "criteria_used": [{
                "criterion": "owner criterion",
                "status": "supported",
                "evidence_refs": ["verification_summary"],
            }],
            "findings": [],
            "summary": "PASS",
        })}, {}


class _SolvedFailPanelLLM:
    def chat(self, **kwargs):
        model = str(kwargs.get("model") or "")
        verdict = (
            "FAIL"
            if model.startswith(("actionless", "actionable", "coach", "tier"))
            else "PASS"
        )
        findings = []
        if model.startswith("actionable"):
            findings = [{
                "severity": "high",
                "item": "missing edge verification",
                "evidence": "edge receipt absent",
                "recommendation": "run the edge-case verification",
            }]
        return {"content": json.dumps({
            "verdict": verdict,
            "outcome_tier": "best_effort" if model.startswith("tier") else "solved",
            "completion_coach": (
                "run the independent edge verification" if model.startswith("coach") else ""
            ),
            "criteria_used": [{
                "criterion": "owner criterion",
                "status": "supported",
                "evidence_refs": ["verification_summary"],
            }],
            "findings": findings,
            "summary": verdict,
        })}, {}


def test_any_valid_task_acceptance_fail_vetoes_pass_quorum(tmp_path):
    slots = [
        ReviewSlot(slot_id="s1", model="pass-1"),
        ReviewSlot(slot_id="s2", model="pass-2"),
        ReviewSlot(slot_id="s3", model="fail"),
    ]
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={"min_successful_slots": 2},
            task_id="root",
        ),
        slots=slots,
        drive_root=tmp_path,
        llm=_SplitVerdictLLM(),
    )
    assert result.aggregate_signal == "FAIL"


def test_minimal_bare_fail_abstains_without_fabricated_improvement(tmp_path):
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={
                "min_successful_slots": 2,
                "classify_outcome_tier": True,
                "require_criterion_evidence": True,
            },
            task_id="root",
        ),
        slots=[
            ReviewSlot(slot_id="s1", model="pass-1"),
            ReviewSlot(slot_id="s2", model="pass-2"),
            ReviewSlot(slot_id="s3", model="minimal-fail"),
        ],
        drive_root=tmp_path,
        llm=_MinimalFailPanelLLM(),
    )
    assert result.aggregate_signal == "PASS"
    minimal = result.actors[2]
    assert minimal["parsed"] == {"verdict": "FAIL", "findings": []}
    assert minimal["signal"] == "DEGRADED"


def test_only_task_acceptance_fail_with_correction_rail_is_a_valid_veto(tmp_path):
    request = ReviewRequest(
        surface="task_acceptance",
        goal="g",
        policy={
            "min_successful_slots": 2,
            "classify_outcome_tier": True,
            "require_criterion_evidence": True,
        },
        task_id="root",
    )
    pass_quorum = run_review_request(
        request,
        slots=[
            ReviewSlot(slot_id="s1", model="pass-1"),
            ReviewSlot(slot_id="s2", model="pass-2"),
            ReviewSlot(slot_id="s3", model="actionless-fail"),
        ],
        drive_root=tmp_path,
        llm=_SolvedFailPanelLLM(),
    )
    assert pass_quorum.aggregate_signal == "PASS"
    contradictory = pass_quorum.actors[2]
    assert contradictory["signal"] == "DEGRADED"
    assert contradictory["parsed"]["verdict"] == "FAIL"  # raw claim stays auditable

    actionable_veto = run_review_request(
        request,
        slots=[
            ReviewSlot(slot_id="s1", model="pass-1"),
            ReviewSlot(slot_id="s2", model="pass-2"),
            ReviewSlot(slot_id="s3", model="actionable-fail"),
        ],
        drive_root=tmp_path,
        llm=_SolvedFailPanelLLM(),
    )
    assert actionable_veto.aggregate_signal == "FAIL"
    assert actionable_veto.actors[2]["signal"] == "FAIL"

    coach_veto = run_review_request(
        request,
        slots=[
            ReviewSlot(slot_id="s1", model="pass-1"),
            ReviewSlot(slot_id="s2", model="pass-2"),
            ReviewSlot(slot_id="s3", model="coach-fail"),
        ],
        drive_root=tmp_path,
        llm=_SolvedFailPanelLLM(),
    )
    assert coach_veto.aggregate_signal == "FAIL"
    assert coach_veto.actors[2]["signal"] == "FAIL"
    assert "run the independent edge verification" in build_improvement_capsule(coach_veto)

    tier_veto = run_review_request(
        request,
        slots=[
            ReviewSlot(slot_id="s1", model="pass-1"),
            ReviewSlot(slot_id="s2", model="pass-2"),
            ReviewSlot(slot_id="s3", model="tier-fail"),
        ],
        drive_root=tmp_path,
        llm=_SolvedFailPanelLLM(),
    )
    assert tier_veto.aggregate_signal == "FAIL"
    assert tier_veto.actors[2]["signal"] == "FAIL"
    assert "best_effort" in build_improvement_capsule(tier_veto)

    unanimous_minimal_fail = run_review_request(
        request,
        slots=[
            ReviewSlot(slot_id="s1", model="actionless-1"),
            ReviewSlot(slot_id="s2", model="actionless-2"),
            ReviewSlot(slot_id="s3", model="actionless-3"),
        ],
        drive_root=tmp_path,
        llm=_SolvedFailPanelLLM(),
    )
    assert unanimous_minimal_fail.aggregate_signal == "DEGRADED"
    assert unanimous_minimal_fail.degraded is True


class _ThreePhysicalSendsLLM:
    def chat(self, **_kwargs):
        for _ in range(3):
            _claim_physical_dispatch()
        raise AssertionError("third physical send must be rejected before provider dispatch")


def test_acceptance_actor_is_limited_to_two_physical_sends(tmp_path):
    result = run_review_request(
        ReviewRequest(
            surface="task_acceptance",
            goal="g",
            policy={"min_successful_slots": 1},
            task_id="root",
        ),
        slots=[ReviewSlot(slot_id="s1", model="m1")],
        drive_root=tmp_path,
        llm=_ThreePhysicalSendsLLM(),
    )
    assert result.aggregate_signal == "DEGRADED"
    assert result.actors[0]["status"] == "error"
    assert "physical attempt limit exhausted (2/2)" in result.actors[0]["error"]


class _CriterionLLM:
    def __init__(self, *, structured: bool, status: str = "supported"):
        self.structured = structured
        self.status = status

    def chat(self, **_kwargs):
        criteria = (
            [{"criterion": "works", "status": self.status, "evidence_refs": ["verification_summary"]}]
            if self.structured else ["works"]
        )
        return {"content": json.dumps({
            "verdict": "PASS",
            "outcome_tier": "solved",
            "completion_coach": "",
            "criteria_used": criteria,
            "findings": [],
            "summary": "ok",
        })}, {}


def test_clean_acceptance_requires_per_criterion_evidence(tmp_path):
    slots = [ReviewSlot(slot_id=f"s{i}", model=f"m{i}") for i in range(3)]
    request = ReviewRequest(
        surface="task_acceptance",
        goal="g",
        policy={
            "min_successful_slots": 2,
            "classify_outcome_tier": True,
            "require_criterion_evidence": True,
        },
        task_id="root",
    )
    degraded = run_review_request(
        request, slots=slots, drive_root=tmp_path, llm=_CriterionLLM(structured=False),
    )
    assert degraded.aggregate_signal == "DEGRADED"
    missing = run_review_request(
        request,
        slots=slots,
        drive_root=tmp_path,
        llm=_CriterionLLM(structured=True, status="missing"),
    )
    assert missing.aggregate_signal == "DEGRADED"
    clean = run_review_request(
        request, slots=slots, drive_root=tmp_path, llm=_CriterionLLM(structured=True),
    )
    assert clean.aggregate_signal == "PASS"


def test_plan_subject_root_is_distinct_and_escape_fails_loudly(tmp_path):
    system = tmp_path / "system"
    subject = tmp_path / "subject"
    drive = tmp_path / "drive"
    system.mkdir()
    subject.mkdir()
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        workspace_root=subject,
        workspace_mode="external",
        drive_root=drive,
    )
    assert _resolve_plan_roots(ctx, ["src/app.py"]) == (
        system.resolve(), subject.resolve(),
    )
    try:
        _resolve_plan_roots(ctx, [str(system / "BIBLE.md")])
    except ValueError as exc:
        assert "escapes active subject root" in str(exc)
    else:  # pragma: no cover - explicit fail-closed assertion
        raise AssertionError("mixed-root plan was accepted")


def test_planning_horizon_has_one_canonical_contract_and_forensic_refs(tmp_path):
    system = tmp_path / "system"
    drive = tmp_path / "drive"
    system.mkdir()
    (drive / "task_results" / "artifacts" / "root").mkdir(parents=True)
    (drive / "task_results" / "root.json").write_text("{}", encoding="utf-8")
    (drive / "task_results" / "artifacts" / "root" / "plan_task_handoffs.json").write_text(
        "{}", encoding="utf-8",
    )
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=drive,
        task_id="root",
        task_metadata={"root_task_id": "root", "parent_task_id": ""},
        task_contract={"objective": "verbatim owner objective"},
    )
    horizon = _planning_evidence_horizon(
        ctx, governance_repo=system, subject_repo=system,
    )
    assert horizon.count("verbatim owner objective") == 1
    assert "plan_task_handoffs.json" in horizon
    assert '"omissions_manifest": []' in horizon
