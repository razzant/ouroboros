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
from ouroboros.usage_accounting import _claim_physical_dispatch
from ouroboros.utils import append_jsonl


def test_required_blocking_binds_shared_cycle_cap_but_explicit_cap_always_wins(monkeypatch):
    # Owner decisions D10/D20 (2026-08-15): the shared OUROBOROS_REVIEW_MAX_CYCLES
    # binds Required+Blocking too (passes = cycles - 1); ``unlimited`` restores
    # the former unbounded local count. The pre-D10 pin ("999 passes allowed")
    # asserted the replaced behavior and was removed.
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES", raising=False)
    snapshot = task_pacing.BudgetSnapshot(has_deadline=False)
    uncapped = normalize_budget_profile({})
    assert task_pacing.improvement_pass_allowed(
        snapshot, 0, uncapped, required_blocking=True,
    ) == (True, "")
    assert task_pacing.improvement_pass_allowed(
        snapshot, 1, uncapped, required_blocking=True,
    ) == (False, "review_cycles_exhausted")  # the SHARED cap under blocking: typed (D27)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "unlimited")
    assert task_pacing.improvement_pass_allowed(
        snapshot, 999, uncapped, required_blocking=True,
    ) == (True, "")
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)

    for policy in ("fixed", "adaptive"):
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


def test_review_capacity_is_unavailable_when_task_review_is_off(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.task_pacing import project_task_acceptance_review_capacity

    monkeypatch.setattr(config, "get_task_review_mode", lambda: "off")
    ctx = SimpleNamespace(
        task_id="root-off", drive_root=tmp_path,
        task_metadata={"root_task_id": "root-off", "delegation_role": "root"},
    )
    projection = project_task_acceptance_review_capacity(ctx)
    assert projection["state"] == "unavailable"
    assert projection["reason"] == "task_review_mode_off"
    assert projection["remaining_cycles"] is None


def test_review_capacity_discloses_corrupt_cancellation_projection(tmp_path):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_pacing import project_task_acceptance_review_capacity
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    contract = build_task_contract({"budget_profile": {"max_improvement_passes": 0}})
    write_task_result(
        tmp_path, "root-corrupt-cancel", STATUS_RUNNING,
        root_task_id="root-corrupt-cancel", delegation_role="root",
        task_contract=contract,
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "cancel_intents.json").write_text("{", encoding="utf-8")
    ctx = SimpleNamespace(
        task_id="root-corrupt-cancel", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_contract=contract,
        task_metadata={
            "root_task_id": "root-corrupt-cancel", "delegation_role": "root",
            "budget_drive_root": str(tmp_path), "task_contract": contract,
        },
    )
    projection = project_task_acceptance_review_capacity(ctx)
    assert projection["state"] == "unknown"
    assert projection["reason"] == (
        "cancellation_state_unknown:CancelIntentProjectionCorrupt"
    )


def test_acceptance_timing_rows_go_to_the_canonical_split_drive_stream(tmp_path, monkeypatch):
    """The timing row a paid panel writes belongs to the CANONICAL drive of a
    split task, not the child's own logs. Its content is telemetry: since owner
    R52 no gate reads it back."""
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "90")
    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    events = canonical / "logs" / "events.jsonl"
    append_jsonl(events, {"type": "task_acceptance_review_timing", "duration_sec": 100})
    ctx = SimpleNamespace(drive_root=str(child), budget_drive_root=str(canonical))
    assert task_pacing.acceptance_timing_events_path(ctx) == events
    assert not (child / "logs" / "events.jsonl").exists()


def test_acceptance_panel_persists_timing_to_canonical_root(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    import ouroboros.review_evidence as evidence_mod
    import ouroboros.review_substrate as substrate
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools import review_helpers

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    contract = build_task_contract({})
    tool_ctx = SimpleNamespace(
        task_id="root-timing",
        drive_root=child,
        budget_drive_root=str(canonical),
        task_contract=contract,
        task_metadata={
            "root_task_id": "root-timing",
            "delegation_role": "root",
            "budget_drive_root": str(canonical),
            "task_contract": contract,
        },
    )
    write_task_result(
        canonical, "root-timing", STATUS_RUNNING,
        root_task_id="root-timing", delegation_role="root",
        task_contract=contract,
    )
    monkeypatch.setattr(evidence_mod, "build_task_acceptance_evidence", lambda *_a, **_k: {})
    monkeypatch.setattr(
        substrate, "triad_delivery_slots",
        lambda **_k: [SimpleNamespace(model="test-reviewer")],
    )
    monkeypatch.setattr(review_helpers, "review_wave_budget_gate", lambda *_a, **_k: None)
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
        emit_progress=lambda _text, *, incident=None: None,
        mode="required",
        subtree_statuses=[],
        budget_profile={},
        passes_done=2,
        review_binding=substrate.build_review_binding(
            candidate="deliverable", evidence={}, fence_token_or_state="timing-test",
        ),
    )

    loop._execute_task_acceptance_panel(ctx)

    rows = [json.loads(line) for line in (canonical / "logs" / "events.jsonl").read_text().splitlines()]
    assert rows[-1]["task_id"] == "root-timing"
    assert rows[-1]["pass_index"] == 2
    assert rows[-1]["aggregate_signal"] == "PASS"
    assert not (child / "logs" / "events.jsonl").exists()


def _root_acceptance_context(tmp_path, evidence):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    contract = build_task_contract({"budget_profile": {"max_improvement_passes": 0}})
    tool_ctx = SimpleNamespace(
        task_id="root-wallet-gate",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        task_contract=contract,
        task_metadata={
            "root_task_id": "root-wallet-gate",
            "delegation_role": "root",
            "budget_drive_root": str(tmp_path),
            "task_contract": contract,
        },
    )
    write_task_result(
        tmp_path,
        "root-wallet-gate",
        STATUS_RUNNING,
        root_task_id="root-wallet-gate",
        delegation_role="root",
        task_contract=contract,
    )
    return loop._TaskAcceptanceContext(
        tools=SimpleNamespace(_ctx=tool_ctx),
        content="deliverable",
        task_id="root-wallet-gate",
        task_type="task",
        llm_trace={"tool_calls": []},
        drive_root=tmp_path,
        messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "goal"},
        ],
        emit_progress=lambda _text, *, incident=None: None,
        mode="required",
        subtree_statuses=[],
        budget_profile=contract["budget_profile"],
        passes_done=0,
        evidence=evidence,
        review_binding=substrate.build_review_binding(
            candidate="deliverable",
            evidence=evidence,
            fence_token_or_state="wallet-gate",
        ),
    )


def _allow_acceptance_wave(monkeypatch):
    import ouroboros.review_substrate as substrate
    from ouroboros.tools import review_helpers

    monkeypatch.setattr(
        substrate,
        "triad_delivery_slots",
        lambda **_kwargs: [ReviewSlot(slot_id="slot", model="review-model")],
    )
    monkeypatch.setattr(
        review_helpers, "review_wave_budget_gate", lambda *_args, **_kwargs: None,
    )


def test_acceptance_cancellation_recheck_precedes_wallet_claim(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_acceptance_review_state

    ctx = _root_acceptance_context(tmp_path, {"evidence": "complete"})
    request_cancel(tmp_path, ctx.task_id, source="test")
    _allow_acceptance_wave(monkeypatch)
    monkeypatch.setattr(
        substrate,
        "run_review_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reviewer must not be called")
        ),
    )

    result = loop._execute_task_acceptance_panel(ctx)

    assert result.degraded is True
    assert result.degraded_reasons == [
        "cancellation_pending (no reviewer was called)"
    ]
    state = load_task_acceptance_review_state(tmp_path, ctx.task_id)
    assert state["claims_by_binding"] == {}


def test_the_paid_claim_rechecks_the_wallet_and_never_the_floor(tmp_path, monkeypatch):
    """Owner R55: the paid claim (`_claim` in the dispatch stamp, which the
    stub fires exactly where a route does, before its first physical send)
    rechecks the WALLET and cancellation (the test above); time belongs to the
    loop gate, so `review_launch_allowed` is armed to fail the test if the
    claim asks it. The wallet is exhausted for REAL between admission and the
    send (another binding buys the tree's only cycle); the refusal is FREE —
    no claim row for this binding, no reviewer call — and typed DEGRADED."""
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros.task_results import claim_task_acceptance_review_cycle, load_task_acceptance_review_state

    ctx = _root_acceptance_context(tmp_path, {"evidence": "complete"})
    _allow_acceptance_wave(monkeypatch)
    monkeypatch.setattr(task_pacing, "review_launch_allowed", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("the launch floor is the loop gate's, never the claim's")))
    other = substrate.build_review_binding(candidate="another", evidence={}, fence_token_or_state="wallet-gate")

    def _exhaust_wallet_then_fire_stamp(_request, *, usage_ctx, **_kwargs):
        claimed = claim_task_acceptance_review_cycle(tmp_path, ctx.task_id, other, claimed_by_task_id=ctx.task_id)
        assert claimed["status"] == "claimed"  # the tree's one cycle is now spent
        usage_ctx._review_paid_stamp()  # the route's write-ahead call: refuses first
        raise AssertionError("reviewer must not be called")

    monkeypatch.setattr(substrate, "run_review_request", _exhaust_wallet_then_fire_stamp)
    result = loop._execute_task_acceptance_panel(ctx)
    assert result.degraded is True and result.degraded_reasons[0].startswith("review_cycles_exhausted")
    claims = load_task_acceptance_review_state(tmp_path, ctx.task_id)["claims_by_binding"]
    assert set(claims) == {other["binding_hash"]}  # the other binding's row, never this one's


def test_acceptance_corrupt_cancellation_projection_is_unknown_without_claim(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros.task_results import load_task_acceptance_review_state

    ctx = _root_acceptance_context(tmp_path, {"evidence": "complete"})
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "cancel_intents.json").write_text("{", encoding="utf-8")
    _allow_acceptance_wave(monkeypatch)
    monkeypatch.setattr(
        substrate,
        "run_review_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reviewer must not be called")
        ),
    )

    result = loop._execute_task_acceptance_panel(ctx)

    assert result.degraded is True
    assert result.degraded_reasons[0].startswith(
        "cancellation_state_unknown:CancelIntentProjectionCorrupt"
    )
    state = load_task_acceptance_review_state(tmp_path, ctx.task_id)
    assert state["claims_by_binding"] == {}


def test_acceptance_claim_serializes_against_concurrent_cancellation(
    tmp_path, monkeypatch,
):
    import threading

    import ouroboros.task_results as task_results
    from ouroboros.cancel_intents import request_cancel

    ctx = _root_acceptance_context(tmp_path, {"evidence": "complete"})
    entered_claim = threading.Event()
    release_claim = threading.Event()
    real_cap = task_results._root_task_acceptance_review_cap

    def paused_cap(root_result):
        entered_claim.set()
        assert release_claim.wait(timeout=2.0)
        return real_cap(root_result)

    monkeypatch.setattr(task_results, "_root_task_acceptance_review_cap", paused_cap)
    claim_result = []
    claim_thread = threading.Thread(
        target=lambda: claim_result.append(
            task_results.claim_task_acceptance_review_cycle(
                tmp_path, ctx.task_id, ctx.review_binding,
                claimed_by_task_id=ctx.task_id,
            )
        ),
    )
    claim_thread.start()
    assert entered_claim.wait(timeout=2.0)

    cancel_result = []
    cancel_started = threading.Event()

    def cancel():
        cancel_started.set()
        cancel_result.append(
            request_cancel(tmp_path, ctx.task_id, source="concurrent-test")
        )

    cancel_thread = threading.Thread(
        target=cancel,
    )
    cancel_thread.start()
    assert cancel_started.wait(timeout=2.0)
    cancel_thread.join(timeout=0.05)
    assert cancel_thread.is_alive(), "cancellation must wait for the in-flight claim"

    release_claim.set()
    claim_thread.join(timeout=2.0)
    cancel_thread.join(timeout=2.0)
    assert not claim_thread.is_alive() and not cancel_thread.is_alive()
    assert claim_result[0]["status"] == "claimed"
    assert cancel_result and cancel_result[0]["state"] == "requested"


def test_acceptance_zero_physical_refusal_does_not_claim_wallet(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros.task_results import load_task_acceptance_review_state

    evidence = {
        "__unresolved_partial_artifacts__": [{"status": "source_unavailable"}],
    }
    ctx = _root_acceptance_context(tmp_path, evidence)
    _allow_acceptance_wave(monkeypatch)

    result = loop._execute_task_acceptance_panel(ctx)

    assert result.aggregate_signal == "DEGRADED"
    assert result.actors[0]["status"] == "not_dispatched"
    state = load_task_acceptance_review_state(tmp_path, ctx.task_id)
    assert state["claims_by_binding"] == {}

    # Repairing the evidence can still spend the root's sole physical cycle;
    # the synthetic refusal above did not strand the tree at cap=1.
    physical_calls = []

    def physical_panel(request, **kwargs):
        from ouroboros.review_dispatch import stamp_review_paid_on_dispatch

        stamp_review_paid_on_dispatch(kwargs["usage_ctx"])
        physical_calls.append(request)
        return SimpleNamespace(aggregate_signal="PASS")

    monkeypatch.setattr(
        substrate,
        "run_review_request",
        physical_panel,
    )
    repaired = loop._execute_task_acceptance_panel(
        _root_acceptance_context(tmp_path, {"evidence": "complete"}),
    )
    assert repaired.aggregate_signal == "PASS"
    assert len(physical_calls) == 1
    state = load_task_acceptance_review_state(tmp_path, ctx.task_id)
    assert len(state["claims_by_binding"]) == 1


def test_acceptance_route_refusal_before_physical_send_keeps_wallet_retryable(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros import usage_accounting as usage
    from ouroboros.task_results import load_task_acceptance_review_state

    class RefusingLLM:
        def chat(self, **_kwargs):
            raise RuntimeError("route resolution failed before provider send")

    class AccountedLLM:
        def __init__(self):
            self.calls = 0

        def chat(self, **_kwargs):
            request = usage.AttemptRequest(
                model="local-review-test", provider="local", reservation_usd=0.0,
                drive_root=tmp_path, task_id="root-wallet-gate",
                root_task_id="root-wallet-gate",
            )

            def send():
                self.calls += 1
                return {"content": "[]"}, {
                    "prompt_tokens": 0, "completion_tokens": 0,
                }

            return usage.execute_physical_attempt(request, send)

    _allow_acceptance_wave(monkeypatch)
    original_run = substrate.run_review_request

    def run_with(llm):
        def _run(request, **kwargs):
            return original_run(request, llm=llm, **kwargs)

        monkeypatch.setattr(substrate, "run_review_request", _run)
        return loop._execute_task_acceptance_panel(
            _root_acceptance_context(tmp_path, {"evidence": "complete"}),
        )

    refused = run_with(RefusingLLM())
    assert refused.actors[0]["status"] == "error"
    assert refused.actors[0].get("physical_attempts", 0) == 0
    state = load_task_acceptance_review_state(tmp_path, "root-wallet-gate")
    assert state["claims_by_binding"] == {}

    accounted = AccountedLLM()
    run_with(accounted)
    state = load_task_acceptance_review_state(tmp_path, "root-wallet-gate")
    assert accounted.calls == 1
    assert len(state["claims_by_binding"]) == 1


def test_acceptance_dispatch_rechecks_cancel_before_provider_send(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as substrate
    from ouroboros import usage_accounting as usage
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import load_task_acceptance_review_state
    class CancellingLLM:
        calls = 0

        def chat(self, **_kwargs):
            request_cancel(tmp_path, "root-wallet-gate", source="dispatch-race")
            request = usage.AttemptRequest(
                model="local-review-test", provider="local", reservation_usd=0.0,
                drive_root=tmp_path, task_id="root-wallet-gate",
                root_task_id="root-wallet-gate",
            )
            return usage.execute_physical_attempt(request, self.send)
        def send(self):
            self.calls += 1
            return {"content": "[]"}, {"prompt_tokens": 0, "completion_tokens": 0}

    _allow_acceptance_wave(monkeypatch)
    original_run, llm = substrate.run_review_request, CancellingLLM()
    monkeypatch.setattr(substrate, "run_review_request", lambda request, **kwargs:
                        original_run(request, llm=llm, **kwargs))
    result = loop._execute_task_acceptance_panel(
        _root_acceptance_context(tmp_path, {"evidence": "complete"}),
    )
    assert result.actors[0]["status"] == "error" and llm.calls == 0
    state = load_task_acceptance_review_state(tmp_path, "root-wallet-gate")
    assert state["claims_by_binding"] == {}
    projection = usage.usage_projection(tmp_path, root_task_id="root-wallet-gate")
    assert projection["attempt_counts"] == {"released": 1} and projection["non_final_rows"] == 0
    assert projection["cost_final"] is True

def test_resolve_budget_profile_emits_no_deprecation_events(tmp_path):
    """ABI 7.0 (Q10=A): the alias deprecation machinery is gone - a profile
    carrying retired spellings resolves quietly, without the retired keys."""
    legacy = SimpleNamespace(
        drive_root=tmp_path,
        task_id="legacy",
        task_contract={"budget_profile": {
            "improvement_policy": "until_deadline", "stall_rounds_threshold": 2,
        }},
    )
    resolved = task_pacing.resolve_budget_profile(legacy)
    assert resolved["improvement_policy"] == "fixed"
    assert "stall_rounds_threshold" not in resolved
    assert not (tmp_path / "logs" / "events.jsonl").exists()


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
