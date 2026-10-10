"""#1536 policy boundaries with real loop, admission, review wallet and wait owners.

Inference, reviewer transport and history output are scripted. Policy tests use a
minimal agent adapter; the proactive case bootstraps OuroborosAgent. Shipped inline
round, review-cycle, packet-size and monetary limits stay unchanged. No separate
Host process or live delegated child is started here.
"""

from __future__ import annotations

import contextvars
import copy
import json
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, config, loop, review_substrate
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.presence_admission import admit_presence_turn
from ouroboros.presence_runner import (
    PresenceTurnExecutions, PresenceTurnGate, presence_event_identity, presence_turn_task_id,
    run_presence_turn,
)
from ouroboros.review_records import ReviewSlot
from ouroboros.task_results import (
    claim_task_acceptance_review_cycle, load_task_acceptance_review_state, load_task_result, write_task_result,
)
from ouroboros.tools.registry import ToolRegistry
from tests.test_host_service_api import _seed_presence_behavior
from tests.test_presence_continuation import ANSWER, call, event, finish


@pytest.fixture
def policy_turn(tmp_path, monkeypatch):
    from ouroboros import review_custody
    from ouroboros.review_execution import ReviewAttemptResult

    for key in ("OUROBOROS_MAX_ROUNDS", "OUROBOROS_REVIEW_MAX_CYCLES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    data, repo = tmp_path / "data", tmp_path / "repo"
    data.mkdir()
    repo.mkdir()
    binding = _seed_presence_behavior(data)
    admission = admit_presence_turn(drive_root=data, authenticated_transport_skill="telegram-bot",
                                   binding_id=binding, global_max_rounds=config.get_max_rounds())
    h = SimpleNamespace(data=data, repo=repo, admission=admission, script=[], calls=0, reviews=[], ctx=None,
                        inputs=[], progress=[], contract=True, before_loop=lambda _ctx: None,
                        release=threading.Event(), settled=threading.Event(), executions=PresenceTurnExecutions(),
                        gate=PresenceTurnGate(1, state_root=data / "state"), running=[])
    slots = [ReviewSlot(slot_id="policy-reviewer", model="fixture/reviewer", effort="high", timeout_sec=30)]
    monkeypatch.setattr(review_substrate, "triad_delivery_slots", lambda **_kw: slots)
    original_settle = review_custody._settle_review_attempt

    def settle(*args, **kwargs):
        try:
            return original_settle(*args, **kwargs)
        finally:
            h.settled.set()

    monkeypatch.setattr(review_custody, "_settle_review_attempt", settle)

    class Reviewer:
        def __init__(self, assignment):
            self.assignment = assignment

        def restore_custody(self, _state):
            pass

        def set_pending_invocation_checkpoint(self, _checkpoint):
            pass

        def prompt_payload(self):
            return {"messages": []}

        def prompt_chars(self):
            return 0

        def failure_custody(self):
            return {}

        def execute(self):
            from ouroboros.review_dispatch import invoke_review_paid_stamp
            from ouroboros.review_evidence_refs import acceptance_evidence_ref_vocabulary

            invoke_review_paid_stamp(self.assignment.dispatch_stamp)
            h.reviews.append(self.assignment.request)
            assert h.release.wait(30), "test did not release its reviewer"
            vocabulary = acceptance_evidence_ref_vocabulary(self.assignment.request.evidence)
            reference = next(key for key, kind in vocabulary.items() if kind in {"tool_record", "packet_section"})
            text = json.dumps({"verdict": "PASS", "summary": "Report supported", "findings": [],
                               "outcome_tier": "solved", "completion_coach": "Deliver it.",
                               "criteria_used": [{"criterion": "report", "status": "supported",
                                                  "evidence_refs": [reference]}]})
            return ReviewAttemptResult(message={"content": text}, raw_text=text,
                                       usage={"prompt_tokens": 5, "completion_tokens": 3,
                                              "physical_attempt_state": "settled"})

    monkeypatch.setattr(review_substrate, "_review_route_executor", lambda assignment, **_kw: Reviewer(assignment))

    def model(_llm, messages, *_args, **kwargs):
        h.calls += 1
        h.inputs.append(copy.deepcopy(messages))
        assert h.script, f"unexpected extra inference round: {messages[-3:]}"
        step = h.script.pop(0)
        observer = kwargs.get("model_context_observer")
        if callable(observer):
            observer(messages)
        return (step(messages) if callable(step) else step), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", model)

    class Agent:
        def handle_task(self, task):
            registry = ToolRegistry(repo_dir=repo, drive_root=data)
            ctx = h.ctx = registry._ctx
            ctx.task_id = task["id"]
            ctx.is_direct_chat = True
            ctx.inline_max_rounds = task["metadata"]["inline_max_rounds"]
            ctx.task_metadata = dict(task["metadata"])
            ctx.task_contract = dict(task["task_contract"])
            if h.contract:
                ctx.task_contract["expected_output"] = "A complete status report."
            write_task_result(data, task["id"], "running", task_contract=ctx.task_contract, root_task_id=task["id"])
            ctx.task_attempt, ctx.current_chat_id = 1, task["chat_id"]
            ctx.review_wait_callback = self.review_wait_callback
            registry.override_handler("chat_history", lambda **_kw: "History read.")
            task["_skip_post_task_synthesis"] = True
            h.before_loop(ctx)
            with task_model_wait_scope(task=task, drive_root=data, event_queue=None, worker_slot_held=False) as waiter:
                ctx.model_wait_context, waiter.tool_context = waiter, ctx
                text, usage, trace = loop.run_llm_loop(
                    [{"role": "system", "content": "Presence turn."}, {"role": "user", "content": task["text"]}],
                    registry, SimpleNamespace(default_model=lambda: "fixture/main"), data / "logs",
                    lambda text, **_kw: h.progress.append(text), queue.Queue(), task_id=task["id"], drive_root=data)
            h.usage, h.trace = usage, trace
            events = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=data, repo_dir=repo), None, None, events, task,
                                       text, usage, trace, 0.0, data / "logs", ctx=ctx)
            return events

    def start():
        turn = event()
        execution = h.executions.start_thread(
            presence_turn_task_id(binding, turn.source_event_id), identity=presence_event_identity(binding, turn),
            admit=lambda: h.gate.acquire(turn.conversation_key),
            run=lambda lease: run_presence_turn(admission=admission, event=turn, repo_dir=repo, drive_root=data,
                                               agent_factory=lambda **_kw: Agent(), admitted=lease),
            context=contextvars.copy_context())[0]
        h.running.append(execution)
        return execution

    h.start = start
    yield h
    h.release.set()
    for execution in h.running:
        execution.result.result(timeout=30)
    if h.reviews:
        assert h.settled.wait(15), "review custody did not settle"
    assert not h.executions.live()


@pytest.mark.parametrize("mode,contract,reviewed", [
    ("auto", False, False), ("auto", True, True),
    ("required", False, False), ("required", True, True),
    ("off", False, False), ("off", True, False),
])
def test_presence_eligibility_keeps_auto_required_and_off(policy_turn, monkeypatch, mode, contract, reviewed):
    h = policy_turn
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", mode)
    h.contract = contract
    h.script = [finish("nominate", message=ANSWER)]
    if contract:
        # The production one-shot verify-before-done reminder remains enabled.
        h.script.append(lambda _messages: finish("after-reminder", answer_sha256=h.ctx._delivery_candidate.content_sha256))
    if reviewed:
        h.script.append(lambda _messages: finish("select", answer_sha256=h.ctx._delivery_candidate.content_sha256))
    execution = h.start()
    initial = execution.initial.result(timeout=30)
    if reviewed:
        assert initial.status == "continuing" and initial.text == ""
        assert not execution.result.done() and h.calls == 2
        h.release.set()
    final = execution.result.result(timeout=30)
    assert final.text == ANSWER and final.outcome == "message"
    assert h.calls == 1 + int(contract) + int(reviewed) and len(h.reviews) == int(reviewed)
    assert bool(load_task_result(h.data, final.task_id).get("presence_continuation")) is reviewed
    assert not h.script


def test_default_inline_cap_is_not_reset_by_review_wait(policy_turn):
    h = policy_turn
    assert config.get_max_rounds() is None and h.admission.inline_max_rounds == 10
    h.script = [{"content": None, "tool_calls": [call("chat_history", {}, f"read-{i}")]} for i in range(9)]
    h.script += [finish("nominate", message=ANSWER), {"content": "Forced record without a delivery declaration."}]
    execution = h.start()
    initial = execution.initial.result(timeout=30)
    assert initial.status == "continuing" and initial.text == "" and h.calls == 10
    h.release.set()
    final = execution.result.result(timeout=30)
    assert final.text == "" and final.outcome in {"silent", "deferred"}
    assert h.calls == 11 and not h.script  # ten inline calls plus the existing single forced record
    assert h.usage["reason_code"] == "round_limit"
    assert len(h.reviews) == 1


def _spend_review_cycles(h, ctx, count):
    for index in range(count):
        binding = review_substrate.build_review_binding(candidate=f"Earlier distinct material {index}",
                                                       evidence={}, fence_token_or_state="earlier-review")
        claim = claim_task_acceptance_review_cycle(h.data, ctx.task_id, binding, claimed_by_task_id=ctx.task_id)
        assert claim["status"] == "claimed"


def test_last_default_review_cycle_still_allows_the_author_response(policy_turn):
    from ouroboros.review_cycles import review_max_cycles

    h = policy_turn
    assert review_max_cycles() == 2
    h.before_loop = lambda ctx: _spend_review_cycles(h, ctx, 1)
    h.script = [finish("nominate", message=ANSWER),
                lambda _messages: finish("after-reminder", answer_sha256=h.ctx._delivery_candidate.content_sha256),
                lambda messages: finish("select", answer_sha256=h.ctx._delivery_candidate.content_sha256)]
    execution = h.start()
    initial = execution.initial.result(timeout=30)
    assert initial.status == "continuing" and h.calls == 2
    assert len(load_task_acceptance_review_state(h.data, initial.task_id)["claims_by_binding"]) == 2
    h.release.set()
    final = execution.result.result(timeout=30)
    assert final.text == ANSWER and h.calls == 3 and len(h.reviews) == 1
    assert "[PRESENCE CONVERSATION RESUMED]" in str(h.inputs[-1])
    assert len(load_task_acceptance_review_state(h.data, final.task_id)["claims_by_binding"]) == 2


def test_spent_default_review_wallet_dispatches_and_lends_nothing(policy_turn):
    h = policy_turn
    h.before_loop = lambda ctx: _spend_review_cycles(h, ctx, 2)
    h.script = [finish("nominate", message=ANSWER),
                lambda _messages: finish("after-reminder", answer_sha256=h.ctx._delivery_candidate.content_sha256),
                finish("stop", message=ANSWER, author_disposition="partial", action="stop",
                       rationale="Review wallet is exhausted; retain the unaccepted report.")]
    execution = h.start()
    final = execution.result.result(timeout=30)
    assert execution.initial.result(timeout=1) is final
    assert not h.reviews and not load_task_result(h.data, final.task_id).get("presence_continuation")
    assert h.trace["review_decision"]["dispatch_refusal"]["reason"] == "review_cycles_exhausted"
    assert len(load_task_acceptance_review_state(h.data, final.task_id)["claims_by_binding"]) == 2


def test_acceptance_money_gate_refuses_a_spent_default_root_cap(policy_turn, monkeypatch):
    """Exercise monetary admission itself; no claim of Main's budget-pause lifecycle."""
    from ouroboros import pricing, usage_accounting as ua
    from ouroboros.contracts.task_contract import build_task_contract

    h = policy_turn
    root_cap = config.SETTINGS_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"]
    global_cap = config.SETTINGS_DEFAULTS["TOTAL_BUDGET"]
    task_id = "presence-spent-money"
    contract = build_task_contract({"id": task_id, "expected_output": "Report"})
    write_task_result(h.data, task_id, "running", root_task_id=task_id, task_contract=contract)
    ctx = SimpleNamespace(task_id=task_id, drive_root=h.data, repo_dir=h.repo, task_contract=contract, pending_events=[],
                          task_metadata={"root_task_id": task_id, "budget_drive_root": str(h.data)})

    class Price(tuple):
        tiers = ()

    monkeypatch.setattr(pricing, "get_pricing", lambda **_kw: {"fixture/reviewer": Price((1., .1, 1.25, 4.))})
    scope = ua.UsageScope(drive_root=h.data, task_id=task_id, root_task_id=task_id,
                          root_limit_usd=root_cap, global_limit_usd=global_cap,
                          root_limit_source="shipped_default", global_limit_source="shipped_default")
    with ua.usage_scope(scope):
        reservation = ua.reserve_attempt(ua.AttemptRequest(
            model="fixture/reviewer", provider="openrouter", reservation_usd=root_cap,
            drive_root=h.data, task_id=task_id, root_task_id=task_id,
            root_limit_usd=root_cap, global_limit_usd=global_cap))
        ua.mark_dispatched(reservation)
        ua.settle_attempt(reservation, {}, cost_usd=root_cap, cost_final=True)
        panel = loop._TaskAcceptanceContext(
            tools=SimpleNamespace(_ctx=ctx), content=ANSWER, task_id=task_id, task_type="task",
            llm_trace={"tool_calls": []}, drive_root=h.data,
            messages=[{"role": "system", "content": "Presence"}, {"role": "user", "content": "Report"}],
            emit_progress=lambda *_a, **_kw: None, mode="required", subtree_statuses=[], budget_profile={},
            passes_done=0, evidence={"goal": "Report"},
            review_binding=review_substrate.build_review_binding(candidate=ANSWER, evidence={"goal": "Report"},
                                                                 fence_token_or_state="money-gate"))
        result = loop._execute_task_acceptance_panel(panel)
    assert result.aggregate_signal == "DEGRADED" and result.degraded
    assert result.degraded_reasons[0].startswith("review_wave_budget_insufficient")
    assert not h.reviews and not load_task_acceptance_review_state(h.data, task_id).get("claims_by_binding")
    refusal = next(row for row in ctx.pending_events if row.get("type") == "review_wave_budget_insufficient")
    assert refusal["limit_usd"] == root_cap and refusal["remaining_usd"] == 0
    assert refusal["estimated_wave_usd"] > 0


def test_full_agent_review_return_keeps_the_spent_default_wallet(tmp_path, monkeypatch):
    """Real wallet across reentry; the transport fixture reserves before each reply.

    The bootstrap helper replaces call_llm_with_retry, including its physical
    reservation. Restore that ledger boundary in the fixture so an attempted
    loop call is not confused with an admitted provider dispatch. This is wallet
    owner/reentry coupling, not a qualification of the full provider transport.
    """
    from ouroboros import usage_accounting as ua
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, nominate, wait_for

    h = install_bootstrap_harness(monkeypatch, tmp_path / "data")
    scopes = []
    attempted_scopes = []
    dispatched = []
    spent = threading.Event()
    scripted_transport = loop.call_llm_with_retry

    def metered_transport(*args, **kwargs):
        attempted_scopes.append(ua.current_usage_scope())
        reservation = ua.reserve_attempt(ua.AttemptRequest(
            model="fixture/main", provider="fixture", reservation_usd=.01))
        ua.mark_dispatched(reservation)
        dispatched.append(reservation.attempt_id)
        response = scripted_transport(*args, **kwargs)
        ua.settle_attempt(reservation, {}, cost_usd=.01, cost_final=True)
        return response

    monkeypatch.setattr(loop, "call_llm_with_retry", metered_transport)

    def script(_messages):
        assert not spent.is_set(), "the author bought a model round after its wallet was exhausted"
        scope = ua.current_usage_scope()
        assert scope.root_limit_usd == config.SETTINGS_DEFAULTS["OUROBOROS_PER_TASK_COST_USD"] == 50.0
        scopes.append(scope)
        ctx = h.agents[0].tools._ctx
        if getattr(ctx, "_task_acceptance_pending", ""):
            return finish("wait", answer_sha256=ctx._delivery_candidate.content_sha256)
        return nominate()

    h.script = script
    execution = h.start(event())
    try:
        initial = execution.initial.result(timeout=40)
        assert initial.status == "continuing" and initial.text == ""
        wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "waiting",
                 what="author parked before spending its remaining wallet")
        scope = scopes[-1]
        assert scope.task_id == scope.root_task_id == initial.task_id
        calls_before_wake = h.calls
        with ua.usage_scope(scope):
            accounting = ua.refresh_root_accounting(scope.drive_root, scope.root_task_id, strict=True)
            remaining = scope.root_limit_usd - accounting["accounted_usd"]
            assert 0 < remaining < 50.0
            reservation = ua.reserve_attempt(ua.AttemptRequest(
                model="fixture/wallet-spend", provider="fixture", reservation_usd=remaining))
            ua.mark_dispatched(reservation)
            ua.settle_attempt(reservation, {}, cost_usd=remaining, cost_final=True)
        spent.set()
        h.release.set()
        final = execution.result.result(timeout=40)
        row = load_task_result(h.data, initial.task_id)
        assert h.calls == calls_before_wake and len(h.agents) == 1 and len(h.reviews) == 1
        assert len(dispatched) == calls_before_wake
        assert all(attempt.root_task_id == scope.root_task_id and attempt.root_limit_usd == 50.0
                   for attempt in attempted_scopes)
        assert final.text == "" and final.outcome in {"silent", "deferred"}
        assert row["outcome_axes"]["execution"]["reason_code"] == "budget_exhausted"
        assert row["presence_continuation"]["outputs"] == []
        with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded) as refused:
            ua.reserve_attempt(ua.AttemptRequest(model="fixture/wallet-probe", provider="fixture", reservation_usd=.01))
        assert refused.value.limit_scope == "root"
    finally:
        h.release.set()
        if h.entered.is_set():
            assert h.settled.wait(15)
        execution.result.result(timeout=40)
        assert not h.executions.live()


def test_full_agent_explicit_calendar_deadline_survives_review_park(tmp_path, monkeypatch):
    """Advance only the fixture clock; the original deadline and caps never change."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import deadline_utils
    from ouroboros.model_wait import _CALENDAR, calendar_scope
    from ouroboros.review_cycles import review_max_cycles
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, nominate, wait_for

    h = install_bootstrap_harness(monkeypatch, tmp_path / "data")
    clock = SimpleNamespace(now=datetime.now(timezone.utc))
    expires = clock.now + timedelta(minutes=10)
    deadline = expires.isoformat()
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: clock.now)
    observed_deadlines = []

    def script(_messages):
        observed_deadlines.append(_CALENDAR.get())
        assert clock.now < expires, "a model call escaped the expired calendar bound"
        ctx = h.agents[0].tools._ctx
        if getattr(ctx, "_task_acceptance_pending", ""):
            return finish("wait", answer_sha256=ctx._delivery_candidate.content_sha256)
        return nominate()

    h.script = script
    with calendar_scope(deadline):
        execution = h.start(event())
    try:
        initial = execution.initial.result(timeout=40)
        assert initial.status == "continuing" and initial.text == ""
        wait_for(lambda: (load_task_result(h.data, initial.task_id).get("owner_wait") or {}).get("state") == "waiting",
                 what="author parked with its inherited calendar deadline")
        ctx = h.agents[0].tools._ctx
        assert ctx.inline_max_rounds == 10
        assert review_max_cycles() == 2
        assert observed_deadlines and all(deadline in bounds for bounds in observed_deadlines)
        assert h.entered.is_set() and not h.release.is_set() and not execution.result.done()
        calls_before_expiry = h.calls
        clock.now = expires + timedelta(seconds=1)
        final = execution.result.result(timeout=30)
        row = load_task_result(h.data, initial.task_id)
        assert h.calls == calls_before_expiry and len(h.agents) == len(h.reviews) == 1
        assert ctx._presence_conversation_lost == "deadline"
        assert row["owner_wait"]["resume_reason"] == "control:deadline"
        assert row["outcome_axes"]["execution"]["reason_code"] == "deadline_local"
        assert final.text == "" and not final.output_ref
        assert row["presence_continuation"]["outputs"] == []
        assert not h.executions.live()
    finally:
        h.release.set()
        execution.result.result(timeout=40)
        if h.entered.is_set():
            assert h.settled.wait(15)


def test_proactive_full_agent_preserves_child_and_parent_tail(tmp_path, monkeypatch):
    """Real initiation/admission/bootstrap; promoted work is a retained scheduled fixture."""
    import functools

    from ouroboros.presence_capabilities import (
        PresenceSelection, PresenceState, PresenceToolTarget, load_presence_state,
        presence_state_fingerprint, save_presence_state,
    )
    from ouroboros.presence_profile import parse_presence_profile, presence_request_fingerprint
    from ouroboros.presence_runner import PROACTIVE_TURNS
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_review_state
    from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, nominate, wait_for
    from tests.test_presence_continuation_host import _initiate

    h = install_bootstrap_harness(monkeypatch, tmp_path / "data")
    h.binding = _seed_presence_behavior(h.data)
    skill_dir = h.data / "skills/external/community-helper"
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(skill_path.read_text().replace(
        "  capability_requests:\n", "  capability_requests:\n    - id: review\n      kind: tool\n"
        "      required: true\n      purpose: Request independent result review.\n"), encoding="utf-8")
    loaded = load_skill(skill_dir, h.data)
    save_review_state(h.data, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    state = load_presence_state(h.data, loaded.name)
    review_request = next(request for request in profile.capability_requests if request.request_id == "review")
    save_presence_state(h.data, loaded.name, PresenceState((*state.selections, PresenceSelection(
        presence_request_fingerprint(review_request), PresenceToolTarget("builtin", "task_acceptance_review")))),
        expected_state_fingerprint=presence_state_fingerprint(state))
    monkeypatch.setattr("ouroboros.presence_runner.run_presence_turn",
                        functools.partial(run_presence_turn, agent_factory=h.factory))

    def script(messages):
        ctx = h.agents[0].tools._ctx
        if not load_task_result(h.data, "work-proactive"):
            write_task_result(h.data, "work-proactive", "scheduled", delegation_role="root",
                              root_task_id="work-proactive", description="Compile Q2",
                              metadata={"presence": dict(ctx.task_metadata["presence"])})
        ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": "work-proactive"}
        resumed = next((row["content"] for row in messages if row.get("role") == "user"
                        and str(row.get("content", "")).startswith("[PRESENCE CONVERSATION RESUMED]")), "")
        if resumed:
            assert "your promoted work work-proactive is scheduled" in resumed
            return finish("final", answer_sha256=ctx._delivery_candidate.content_sha256)
        if getattr(ctx, "_task_acceptance_pending", ""):
            return finish("wait", answer_sha256=ctx._delivery_candidate.content_sha256)
        return nominate()

    h.script = script
    try:
        initial = _initiate(h, "Report status and retain the Q2 work.", "proactive-child-tail")
        assert initial["status"] == "continuing" and initial["work_ref"] == "work-proactive"
        assert initial["turn_ref"] == initial["continuation_ref"] != initial["work_ref"]
        assert PROACTIVE_TURNS.live() and load_task_result(h.data, "work-proactive")["status"] == "scheduled"
        h.release.set()
        wait_for(lambda: not PROACTIVE_TURNS.live(), timeout=40, what="proactive parent terminal")
        parent = load_task_result(h.data, initial["turn_ref"])
        assert parent["status"] == "completed" and parent["metadata"]["presence_result_text"] == ANSWER
        assert parent["metadata"]["presence_work_ref"] == "work-proactive"
        assert load_task_result(h.data, "work-proactive")["status"] == "scheduled"
        assert len(h.agents) == 1 and len(h.reviews) == 1
        write_task_result(h.data, "work-proactive", "completed", result="Q2 done.", terminal_origin="model_final")
        assert load_task_result(h.data, "work-proactive")["result"] == "Q2 done."
    finally:
        h.release.set()
        wait_for(lambda: not PROACTIVE_TURNS.live(), timeout=40, what="proactive teardown")
        if h.entered.is_set():
            assert h.settled.wait(15)
