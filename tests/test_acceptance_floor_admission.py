"""The acceptance admission RAILS (owner R52): the configured floor is the whole
time rail, the wallet buys one work-order send per paid row, the paid-cycle cap
counts, and nothing predicts how long a review takes.

The module that used to pin an estimate, its disclosures and a read-only
preview of both gates now pins the three surviving boundaries themselves — the
launch gate exactly at the floor, the improvement window exactly at floor ×
scale, the money fence per send, the count cap — plus the local predicate
equivalence with the pre-deletion tree, the paid claim that asks the wallet and
cancellation only (owner R55: the floor is evaluated ONCE, at loop admission,
so a panel whose evidence build ate the margin dispatches — the disclosed
residual, bounded by the R23 deadline clamps), and the purity of every
read-only poll (which writes nothing of its own and inherits the canonical
usage-ledger reader's own bounded maintenance — today: the torn-tail
quarantine after a single crash mid-append, pinned in that single-crash shape,
the empty `state/` directory the reader's lock lives in on a never-initialized
root, and removal of a stale `usage_attempts.lock` older than the reader's
90 s stale window (`usage_ledger._locked` →
`platform_layer.acquire_exclusive_file_lock`, whose stale-age branch unlinks
the lock file and retries) — each pinned by its own regression below; every
ledger state, absent included, goes through the canonical reader). A separate
module from the three-delivery contract suite on purpose: the subject here is
what may START and what may be SPENT, not what a delivery row receives. Every
offline fixture — the fake triads, the scripted ledger-crossing reviewer, the
seeded wallet, the priced catalog row — is the delivery suite's, imported by
name; nothing here is a copy.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_acceptance_delivery import (
    _ACCEPTANCE_PACKET,
    _CLEAN_VERDICT,
    _ROW_API,
    _ROW_NATIVE,
    _EpisodeLLM,
    _acceptance_ctx,
    _offline_env,
    _priced_offline_model,
    _real_panel,
    _root_scope,
    _roots,
    _seed_root_ledger,
    _timing,
    _tool_call,
)


class _KnownPricedLLM(_EpisodeLLM):
    """The delivery suite's scripted reviewer whose provider reports a FINAL price
    for each send: KNOWN spend, the only money a limit decides on (#1487). The
    suite's own replies carry no usable price, so they stay unresolved exposure."""

    def chat(self, **kwargs):
        from ouroboros import usage_accounting as ua

        request = ua.AttemptRequest(reservation_usd=self.reservation_usd, drive_root=self.drive_root,
                                    model="openai/fake-reviewer", provider="openrouter")
        return ua.execute_physical_attempt(request, lambda: self._reply(kwargs),
                                           extractor=lambda reply: (reply[1], self.reservation_usd, True))


def _known_charge(scope, cost):
    from ouroboros import usage_accounting as ua

    with ua.usage_scope(scope):
        held = ua.reserve_attempt(ua.AttemptRequest(model="openai/fake-reviewer", provider="openrouter",
                                                    reservation_usd=cost))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=cost, cost_final=True)


def test_a_wave_at_the_known_root_limit_is_still_refused(monkeypatch, tmp_path):
    """The floor is the admission line, not a bypass: once the root's KNOWN spend
    has reached its limit (#1487: the reservation's own rule), the panel is refused
    as before — through the real gate — and no reviewer is called."""
    from ouroboros import loop as loop_mod
    from ouroboros import usage_accounting as ua

    _offline_env(monkeypatch, _ROW_NATIVE)
    _priced_offline_model(monkeypatch)
    governance, workspace = _roots(tmp_path)
    llm = _EpisodeLLM(tmp_path, [{"content": json.dumps(_CLEAN_VERDICT)}] * 2, scoped=True)
    _real_panel(monkeypatch, llm, stub_gate=False)
    scope = _root_scope(tmp_path, root_limit_usd=0.5)
    _seed_root_ledger(scope)
    _known_charge(scope, 0.5)  # known spend is at the $0.50 limit
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    with ua.usage_scope(scope):
        refused = loop_mod._execute_task_acceptance_panel(ctx)
    assert refused.aggregate_signal == "DEGRADED"
    assert any(r.startswith("review_wave_budget_insufficient") for r in refused.degraded_reasons)
    assert llm.calls == []  # no reviewer was called
    assert any(e.get("type") == "review_wave_budget_insufficient" for e in ctx.tools._ctx.pending_events)


def _rows(events_path, event_type):
    from ouroboros.utils import iter_jsonl_objects

    return [e for e in iter_jsonl_objects(events_path) if e.get("type") == event_type]


def _floor_band_ctx(
    monkeypatch, tmp_path, *, seconds_left, claims=1, max_improvement_passes=None,
    policy="adaptive",
):
    """A REAL root task on a real deadline: the packet triad, the 200 s
    configured floor, the 120 s grace as the WHOLE reserve (pct 0), `claims`
    really claimed cycles, and one recorded 200 s panel — telemetry the rails
    read back nowhere. Nothing is patched: the same contract and wallet state
    production reads. `max_improvement_passes` defaults to NONE so the shipped
    configuration (2 review cycles → 1 improvement pass) is what these
    regressions exercise; a task-local cap would mask the count axis."""
    import queue
    from datetime import timedelta

    from ouroboros import task_pacing
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.deadline_utils import utc_now
    from ouroboros.review_substrate import build_review_binding
    from ouroboros.task_results import (
        STATUS_RUNNING,
        claim_task_acceptance_review_cycle,
        write_task_result,
    )

    _offline_env(monkeypatch, _ROW_API)  # a packet panel
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "200")  # the floor
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")  # the whole reserve (pct 0)
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    now = utc_now()
    profile = {"improvement_policy": policy, "reserve_finalization_pct": 0}
    if max_improvement_passes is not None:
        profile["max_improvement_passes"] = max_improvement_passes
    contract = build_task_contract({
        "deadline_at": (now + timedelta(seconds=seconds_left)).isoformat(),
        "budget_profile": profile,
    })
    metadata = {
        "root_task_id": "root-floor-band", "delegation_role": "root",
        "budget_drive_root": str(tmp_path), "task_contract": contract,
        "created_at": (now - timedelta(seconds=60)).isoformat(),
        "deadline_at": contract["deadline_at"],
    }
    ctx = SimpleNamespace(
        task_id="root-floor-band", root_task_id="root-floor-band", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_contract=contract, task_metadata=metadata,
        pending_events=[], event_queue=queue.Queue(),
    )
    write_task_result(tmp_path, "root-floor-band", STATUS_RUNNING, root_task_id="root-floor-band",
                      delegation_role="root", task_contract=contract)
    for index in range(claims):
        claim = claim_task_acceptance_review_cycle(
            tmp_path, "root-floor-band",
            build_review_binding(candidate=f"deliverable v{index}", evidence=dict(_ACCEPTANCE_PACKET),
                                 fence_token_or_state="floor-band"),
            claimed_by_task_id="root-floor-band",
        )
        assert claim["status"] == "claimed"  # REAL paid cycles: the wallet counts them
    events = task_pacing.acceptance_timing_events_path(ctx)
    events.parent.mkdir(parents=True, exist_ok=True)
    _timing(events, duration_sec=200, delivery="api_chat")  # telemetry only: no gate reads it
    return ctx, events


def test_projecting_an_exhausted_cap_emits_no_review_cycles_exhausted_event(monkeypatch, tmp_path):
    """The improvement gate EMITS the typed `review_cycles_exhausted`
    escalation when a ctx is supplied and the SHARED cap is exhausted under
    Required+Blocking — so the read-only wallet projection that every poll
    calls must never reach it. With the shared cap at 2 cycles and 2 cycles
    really claimed, projecting stays silent (its `review_cycles_exhausted`
    state is the remaining==0 wallet semantics, not an event), while the
    predicate called WITH this ctx does emit — the control that keeps this
    assertion from passing vacuously."""
    from ouroboros import task_pacing
    from ouroboros.outcomes import REASON_REVIEW_CYCLES_EXHAUSTED
    from ouroboros.task_results import (
        project_task_acceptance_review_capacity,
        task_acceptance_required_blocking,
    )

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "required")
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")  # 2 cycles => 1 improvement pass
    ctx, events = _floor_band_ctx(monkeypatch, tmp_path, seconds_left=620, claims=2)
    # The lane really is Required+Blocking through the ONE derivation every
    # reader shares (the control the retired `until_deadline` count-axis test
    # carried): otherwise the typed reason below could come from a default.
    assert task_acceptance_required_blocking() is True
    profile = task_pacing.resolve_budget_profile(ctx)
    snapshot = task_pacing.build_budget_snapshot(ctx, profile=profile)
    assert profile["max_improvement_passes"] is None
    # The window is wide open (spendable > 2 × the floor): only the COUNT axis
    # can refuse here, so this test cannot pass through the time rail.
    assert snapshot.spendable_sec > 2 * 200.0

    projection = project_task_acceptance_review_capacity(ctx, task_id="root-floor-band")
    assert projection["cap_cycles"] == 2 and projection["claimed_cycles"] == 2
    assert projection["state"] == "unavailable"  # remaining == 0: the projection's own semantics
    assert projection["reason"] == REASON_REVIEW_CYCLES_EXHAUSTED
    assert ctx.pending_events == [] and ctx.event_queue.empty()
    assert _rows(events, REASON_REVIEW_CYCLES_EXHAUSTED) == []  # the poll never reaches the gate

    # The paid wallet stays spent, but ordinary author work remains available.
    assert task_pacing.improvement_pass_allowed(snapshot, 1, profile, required_blocking=True, ctx=ctx) == (True, "")
    assert _rows(events, REASON_REVIEW_CYCLES_EXHAUSTED) == []
    assert project_task_acceptance_review_capacity(ctx, task_id="root-floor-band")["claimed_cycles"] == 2


# NOTE (7.0 ABI, Q10=A): upstream's `until_deadline` count-axis test was dropped here —
# the alias itself is retired (tests/test_abi5_q10_removals.py pins its absence), so the
# Required+Blocking count cap is exercised by the `fixed`/`adaptive` policies alone.
def test_the_per_send_wallet_fence_still_binds_after_an_admitted_dispatch(monkeypatch, tmp_path):
    """The per-send wallet binding at dispatch is what actually protects money,
    proven with PRICED sends: the wave gate admits while known spend is below
    the limit; the native episode's first send is reserved and settled at a
    final price that brings known spend to the root limit, so its second send
    is refused by the ledger (`budget_exhausted`). The coarse admission is a
    filter, never the fence."""
    from ouroboros import loop as loop_mod
    from ouroboros import usage_accounting as ua

    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    _offline_env(monkeypatch, _ROW_NATIVE)
    _priced_offline_model(monkeypatch)
    governance, workspace = _roots(tmp_path)
    llm = _KnownPricedLLM(
        tmp_path, [], scoped=True, reservation_usd=0.06,
        native_script=[{"tool_calls": [_tool_call("read_file", {"path": "greeting.txt"})]},
                       {"content": json.dumps(_CLEAN_VERDICT)}],
    )
    _real_panel(monkeypatch, llm, stub_gate=False)
    limit = 0.06  # the first send's known $0.06 reaches the limit; the second is refused
    scope = _root_scope(tmp_path, root_limit_usd=limit)
    _seed_root_ledger(scope)
    ctx = _acceptance_ctx(tmp_path, evidence=dict(_ACCEPTANCE_PACKET), repo_dir=str(governance),
                          workspace_root=str(workspace), workspace_mode="project")
    with ua.usage_scope(scope):
        result = loop_mod._execute_task_acceptance_panel(ctx)
    (actor,) = result.actors
    assert len(llm.calls) == 1  # the first send went out; the second never reached the model
    assert actor["usage"]["native_end_reason"] == "budget_exhausted"
    assert result.aggregate_signal == "DEGRADED"
    projection = ua.usage_projection(tmp_path, root_task_id="root-delivery")
    spent = float(projection["limit_usd"]) - float(projection["remaining_known_usd"])
    assert projection["limit_usd"] == limit and spent == pytest.approx(0.06)
    assert projection["settled_usd"] == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# The two admission boundaries themselves (owner R52): the floor, and the floor
# scaled by the policy window. Nothing else may move them.
# ---------------------------------------------------------------------------


def _snapshot(spendable, *, reserve=120.0):
    from ouroboros import task_pacing

    return task_pacing.BudgetSnapshot(
        has_deadline=True, total_sec=100000.0, elapsed_sec=0.0,
        remaining_sec=reserve + spendable, reserve_sec=reserve,
    )


def test_the_launch_gate_turns_over_exactly_at_the_configured_floor(monkeypatch):
    """Gate 1's whole rail: `spendable > floor` admits, anything at or below it
    is refused `review_skipped_deadline_reserve`. The floor comes from the
    getter — the shipped default, a raised setting, and a setting BELOW 200 s
    that the 200 s hard floor overrides — never from a literal in the call."""
    from ouroboros import task_pacing

    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", raising=False)
    for setting, expected_floor in ((None, 200.0), ("10", 200.0), ("500", 500.0)):
        if setting is None:
            monkeypatch.delenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", raising=False)
        else:
            monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", setting)
        floor = task_pacing._acceptance_floor_sec()
        assert floor == expected_floor
        assert task_pacing.review_launch_allowed(_snapshot(floor - 1)) == (
            False, "review_skipped_deadline_reserve")
        assert task_pacing.review_launch_allowed(_snapshot(floor)) == (
            False, "review_skipped_deadline_reserve")
        assert task_pacing.review_launch_allowed(_snapshot(floor + 1)) == (True, "")
    # No deadline: the count axis is the only one, whatever the floor says.
    assert task_pacing.review_launch_allowed(
        task_pacing.BudgetSnapshot(has_deadline=False)) == (True, "")


@pytest.mark.parametrize("policy", ["adaptive", "fixed"])
def test_author_work_stops_at_ordinary_reserve(monkeypatch, policy):
    from ouroboros import task_pacing

    profile = {"improvement_policy": policy, "max_improvement_passes": 2}
    assert task_pacing.improvement_pass_allowed(_snapshot(0), 0, profile) == (False, "improvement_window_inside_reserve")
    assert task_pacing.improvement_pass_allowed(_snapshot(1), 0, profile) == (True, "")


# The BEFORE half of this table was recorded by running the same probe against
# a `git worktree` of f62512b6 — the tree that still asked
# `acceptance_review_estimate_sec` for an estimate and passed it into both
# gates. Every cell's ADMIT/REFUSE is identical here; only the reason token
# differs, and only where the base said `launched_at_floor` and this tree says
# "". History (empty vs one 20 000 s recorded panel) changed nothing THERE
# either — which is why the reader could go. `spendable` is the literal window
# in seconds against the shipped 200 s floor: floor−1, floor, floor+1,
# 2×floor−1, 2×floor, 2×floor+1.
_ADMISSION_MATRIX = [
    (history, policy, spendable, spendable > 200, spendable > 0)
    for history in ("empty", "huge") for policy in ("adaptive", "fixed")
    for spendable in (0, 1, 199, 200, 201, 399, 400, 401)
]


@pytest.mark.parametrize("history,policy,spendable,launch_ok,improve_ok", _ADMISSION_MATRIX)
def test_critic_floor_and_author_reserve_are_independent(
        monkeypatch, tmp_path, history, policy, spendable, launch_ok, improve_ok):
    """Paid launch uses its floor; author work uses ordinary remaining time."""
    from ouroboros import task_pacing

    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", raising=False)
    floor = task_pacing._acceptance_floor_sec()
    assert floor == 200.0  # the shipped default the table was recorded against
    ctx = SimpleNamespace(drive_root=tmp_path, task_metadata={}, task_id="matrix")
    events = task_pacing.acceptance_timing_events_path(ctx)
    events.parent.mkdir(parents=True, exist_ok=True)
    if history == "huge":
        _timing(events, duration_sec=20000.0, delivery="api_chat")
        assert events.read_text(encoding="utf-8").count("20000") == 1  # really on disk
    profile = {"improvement_policy": policy, "reserve_finalization_pct": 0}
    snapshot = _snapshot(spendable)
    assert task_pacing.review_launch_allowed(snapshot) == (
        (True, "") if launch_ok else (False, "review_skipped_deadline_reserve"))
    assert task_pacing.improvement_pass_allowed(snapshot, 0, profile, ctx=None) == (
        (True, "") if improve_ok else (False, "improvement_window_inside_reserve"))


# ---------------------------------------------------------------------------
# The paid claim checks cancellation and the wallet only (owner R55): the
# launch floor is evaluated ONCE, at loop admission. What that leaves — the
# disclosed residual, and the R23 clamp that bounds it — is driven end to end
# on the real panel path.
# ---------------------------------------------------------------------------


def _deadline_panel_ctx(monkeypatch, tmp_path, *, over_floor):
    """A REAL packet panel on a REAL deadline: the 200 s configured floor, the
    120 s grace as the WHOLE reserve (pct 0), and a spendable window
    `over_floor` seconds ABOVE the floor when the loop gate asks. Returns
    `(ctx, clock)`; raising `clock["offset"]` moves BOTH clocks the panel reads
    — the pacing rails' and the R23 deadline clamps' — forward without touching
    the recorded deadline: an injected clock, not a rewritten task."""
    from datetime import timedelta

    from ouroboros import deadline_utils, task_pacing
    from ouroboros.deadline_utils import utc_now

    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC", "200")  # the floor
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")  # the whole reserve (pct 0)
    monkeypatch.setenv("OUROBOROS_ACCEPTANCE_RESERVE_PCT", "0")
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    _offline_env(monkeypatch, _ROW_API)
    now = utc_now()
    clock = {"offset": 0.0}
    for module in (task_pacing, deadline_utils):
        monkeypatch.setattr(module, "utc_now", lambda: utc_now() + timedelta(seconds=clock["offset"]))
    ctx = _acceptance_ctx(tmp_path, evidence={}, task_metadata={
        "created_at": (now - timedelta(seconds=60)).isoformat(),
        "deadline_at": (now + timedelta(seconds=120 + 200 + over_floor)).isoformat(),
    })
    return ctx, clock


def _evidence_build(clock, *, burns_sec):
    """The host packet builder, replaced by one that really CONSUMES `burns_sec`
    of the window it was admitted into (the step between the loop gate and the
    seam). `burns_sec=0` is the same builder consuming nothing."""

    def _build(_ctx):
        clock["offset"] += float(burns_sec)
        return dict(_ACCEPTANCE_PACKET)

    return _build


def _admitted_panel(monkeypatch, root, *, burns_sec):
    """One REAL packet panel in its own `root` (arms of one test must not share
    a wallet: the same binding claims once), admitted by the loop gate's OWN
    predicate at floor + 60 s, whose host packet builder then CONSUMES
    `burns_sec` of that window. Returns `(ctx, clock, llm)`."""
    from ouroboros import loop_acceptance_review, task_pacing

    root.mkdir(exist_ok=True)
    ctx, clock = _deadline_panel_ctx(monkeypatch, root, over_floor=60)
    llm = _EpisodeLLM(root, [{"content": json.dumps(_CLEAN_VERDICT)}])
    _real_panel(monkeypatch, llm)
    monkeypatch.setattr(
        loop_acceptance_review, "_build_host_acceptance_evidence",
        _evidence_build(clock, burns_sec=burns_sec))
    # The loop gate, on exactly what loop.py feeds it: ADMIT at floor + 60 s.
    admitted = task_pacing.build_budget_snapshot(ctx.tools._ctx, profile=ctx.budget_profile)
    assert admitted.spendable_sec > 200.0
    assert task_pacing.review_launch_allowed(admitted) == (True, "")
    return ctx, clock, llm


def test_a_panel_whose_evidence_build_ate_the_margin_dispatches_and_the_deadline_bounds_it(
        monkeypatch, tmp_path):
    """The DISCLOSED residual of owner R55, end to end on the real panel with
    an injected clock. The loop gate's own predicate admits at floor + 60 s;
    the evidence build then burns 130 s of that window, so by the time money
    is committed the spendable window is 130 s — at or below the 200 s floor —
    and the panel DISPATCHES anyway: the paid claim asks the wallet and
    cancellation only, so there is one `claims_by_binding` row and one
    physical send. That "exactly one send" is THIS FIXTURE's fact — a packet
    row whose scripted reviewer answers cleanly, so no repair/retry send is
    taken — not a bound: admission prices one work-order send per paid row,
    but a packet row may use its permitted repair/retry send and a native
    row may run several rounds, every send deadline- and wallet-fenced where
    pricing exists, so the total of an admitted panel is NOT bounded to one
    floor wave. What bounds such a panel is the R23 deadline clamp (the third
    arm): a build that eats the margin AND the reserve leaves the owner
    window exhausted before the send, and the row is cut typed and $0 —
    `not_dispatched`, no claim row, no send, a DEGRADED panel — while a panel
    that finishes (the residual arm) keeps its normal verdict; never a free
    skip. The control arm (a build that burns nothing) dispatches exactly as
    before, so neither the dispatch nor the cut can pass vacuously."""
    from ouroboros import loop as loop_mod, task_pacing
    from ouroboros.task_results import load_task_acceptance_review_state

    # CONTROL: the same fixture, the same window, a build that burns nothing.
    with monkeypatch.context() as arm:
        ctx, _clock, llm = _admitted_panel(arm, tmp_path / "control", burns_sec=0)
        dispatched = loop_mod._execute_task_acceptance_panel(ctx)
        assert dispatched.aggregate_signal == "PASS"
        assert len(llm.calls) == 1  # the reviewer really was sent the work order
        assert len(load_task_acceptance_review_state(
            tmp_path / "control", "root-delivery")["claims_by_binding"]) == 1

    # RESIDUAL (R55): the build ate the margin; the claim never asks the floor.
    with monkeypatch.context() as arm:
        ctx, _clock, llm = _admitted_panel(arm, tmp_path / "residual", burns_sec=130)
        dispatched = loop_mod._execute_task_acceptance_panel(ctx)
        # The window really shrank below the floor: asked again, the loop gate
        # would refuse — nothing asks it again.
        shrunk = task_pacing.build_budget_snapshot(ctx.tools._ctx, profile=ctx.budget_profile)
        assert shrunk.spendable_sec <= 200.0
        assert task_pacing.review_launch_allowed(shrunk) == (False, "review_skipped_deadline_reserve")
        assert dispatched.aggregate_signal == "PASS"
        assert len(llm.calls) == 1  # ONE physical send: this packet row took no repair/retry
        assert len(load_task_acceptance_review_state(
            tmp_path / "residual", "root-delivery")["claims_by_binding"]) == 1  # ONE claim row

    # R23: the build ate the margin AND the reserve; the deadline clamp cuts
    # the row before its send — typed, $0, DEGRADED — nothing about it a skip.
    with monkeypatch.context() as arm:
        ctx, _clock, llm = _admitted_panel(arm, tmp_path / "cut", burns_sec=260)
        cut = loop_mod._execute_task_acceptance_panel(ctx)
        (actor,) = cut.actors
        assert actor["status"] == "not_dispatched"
        assert cut.aggregate_signal == "DEGRADED" and cut.degraded is True
        assert llm.calls == []  # no send
        assert load_task_acceptance_review_state(
            tmp_path / "cut", "root-delivery")["claims_by_binding"] == {}  # no claim row


# ---------------------------------------------------------------------------
# Purity: a poll answers, and changes nothing it could have been asked again.
# ---------------------------------------------------------------------------


def test_an_empty_store_polls_known_zero_through_the_reader_and_writes_nothing(tmp_path):
    """Owner R56 (2): on a root whose usage store holds no attempt the
    coordination poll answers a KNOWN zero settled spend THROUGH the canonical
    reader (no fast path answers ahead of it) and changes nothing at all. The
    store exists as in production: the supervisor's lifecycle job creates it
    before any worker (and so any poll) exists. Pinned as the EXACT observed
    tree, byte for byte, across two polls; no events row, no ctx attribute.
    The relayed numbers are the reader's own projection, key by key."""
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros import usage_store
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.usage_accounting import usage_breakdown

    ctx = SimpleNamespace(
        task_id="root-empty", root_task_id="root-empty", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-empty", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    assert sorted(root.rglob("*")) == []  # genuinely fresh
    usage_store.migrate_from_journal(root)  # the lifecycle job, before any worker
    attrs_before = set(vars(ctx))
    known_zero = {
        "state": "known", "settled_usd": 0.0, "accounted_usd": 0.0,
        "cost_final": True, "unknown_unmetered": 0, "integrity_degraded": False,
    }

    def _tree():
        return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
                for path in sorted(root.rglob("*"))}

    before = _tree()
    events_before = custody.event_log_path(root).read_bytes()
    for _ in range(2):
        assert coordination_live_context(ctx)["settled_spend"] == known_zero
        assert _tree() == before  # the poll changed nothing
    assert custody.event_log_path(root).read_bytes() == events_before  # no events row
    assert set(vars(ctx)) == attrs_before
    assert ctx.event_queue.empty() and ctx.pending_events == []

    # The answer IS the reader's: the same projection, key by key.
    projection = usage_breakdown(root, root_task_id="root-empty")
    assert {key: projection[key] for key in known_zero if key != "state"} == {
        key: value for key, value in known_zero.items() if key != "state"}


@pytest.mark.parametrize("enforced", [False, True])
def test_a_stale_name_lock_is_the_readers_removal_and_the_poll_stays_known_zero(
        monkeypatch, tmp_path, enforced):
    """The maintenance a poll inherits from the canonical reader is bounded:
    on an installation without kernel file locks (the store's ``name`` tier)
    every store access runs under the money name lock, so a
    `state/usage_attempts.lock` left behind by a dead writer and older than the
    lock's 90 s stale window is REMOVED by the reader's own lock acquisition
    (`usage_ledger._named_lock` → `platform_layer.acquire_exclusive_file_lock`,
    whose stale-age branch unlinks the file and retries), never by the poll,
    which writes nothing of its own. On the ``enforced`` tier SQLite's own
    locks serialize the store and the same file is inert: left untouched.
    Pinned as the EXACT observed tree; the answer is the reader's own known
    zero; no events row; no ctx attribute; a second poll changes nothing."""
    import os
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros import platform_layer, usage_store
    from ouroboros.delegate_supervision import coordination_live_context

    ctx = SimpleNamespace(
        task_id="root-stale-lock", root_task_id="root-stale-lock", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-stale-lock", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path: enforced)
    usage_store.migrate_from_journal(root)  # the lifecycle job decides the tier once
    usage_store.forget(root)

    def _tree():
        return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
                for path in sorted(root.rglob("*"))}

    migrated = _tree()
    stale_lock = root / "state" / "usage_attempts.lock"
    stale_lock.write_text("pid=999999 ts=0\n", encoding="utf-8")
    os.utime(stale_lock, (0, 0))  # a dead writer's lock, aged to the epoch: far past 90 s
    with_stale = _tree()
    events_before = custody.event_log_path(root).read_bytes()
    attrs_before = set(vars(ctx))
    known_zero = {
        "state": "known", "settled_usd": 0.0, "accounted_usd": 0.0,
        "cost_final": True, "unknown_unmetered": 0, "integrity_degraded": False,
    }

    assert coordination_live_context(ctx)["settled_spend"] == known_zero
    after_first = _tree()
    if enforced:
        assert after_first == with_stale  # inert on the enforced tier
    else:
        assert not stale_lock.exists()  # the reader's stale-age branch removed it, then released its own
        assert after_first == migrated
    assert custody.event_log_path(root).read_bytes() == events_before  # no events row

    assert coordination_live_context(ctx)["settled_spend"] == known_zero
    assert _tree() == after_first  # the second poll changed nothing
    assert set(vars(ctx)) == attrs_before
    assert ctx.event_queue.empty() and ctx.pending_events == []


def test_a_directory_at_the_ledger_path_is_the_readers_verdict_not_known_zero(tmp_path):
    """Every ledger state goes through the canonical reader — there is no fast
    path ahead of it — so a DIRECTORY at the ledger path reports exactly the
    reader's own typed refusal (`UsageAccountingError` → `state: unknown`, no
    confident $0), on a repeat poll too — pinned against the reader called
    directly, so the fact cannot drift from the verdict it relays."""
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.usage_accounting import usage_breakdown
    from ouroboros.usage_ledger import LEDGER_REL, UsageAccountingError

    ctx = SimpleNamespace(
        task_id="root-dir", root_task_id="root-dir", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-dir", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    (root / LEDGER_REL).mkdir(parents=True)
    with pytest.raises(UsageAccountingError):  # the reader's own verdict on a directory
        usage_breakdown(root, root_task_id="root-dir")

    for _ in range(2):
        spend = coordination_live_context(ctx)["settled_spend"]
        assert spend["state"] == "unknown" and spend["reason"] == "UsageAccountingError"
        assert spend["settled_usd"] is None and spend["cost_final"] is None


def _legacy_settings(monkeypatch, tmp_path):
    """Point config at a settings file whose retired auto-Low pair a
    `load_settings()` WOULD normalize and persist — the write this batch must
    keep out of every read path. Returns the path and its (bytes, mtime_ns)."""
    import json as _json

    from ouroboros import config as cfg

    settings_path = tmp_path / "settings-legacy.json"
    settings_path.write_text(_json.dumps({"OUROBOROS_CONTEXT_MODE": "low"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    return settings_path, (settings_path.read_bytes(), settings_path.stat().st_mtime_ns)


def test_the_whole_coordination_poll_writes_nothing_and_reports_an_unlatched_task(
        monkeypatch, tmp_path):
    """The poll every subagent bootstrap and nanny wake runs writes NOTHING on a
    healthy tree, in ALL of its facts and not only the review-capacity one (the
    bounded maintenance it inherits from the canonical ledger reader — today
    the empty `state/` directory on a never-initialized root, the stale-lock
    removal, and the usage ledger's torn-tail quarantine — is pinned by the
    absent-ledger and stale-lock tests above and the torn-ledger test below).
    On a task carrying an `adaptive` profile (the `until_deadline` alias is retired) AND no
    `created_at`/`started_at` (both writes armed), with the grace env ABSENT and
    a legacy context-mode settings file (the settings write armed): the settings
    bytes and mtime, the events stream, the task result and the event queue are
    byte-identical, no ctx attribute appears, and the `time` fact honestly
    answers `not_set` for a task whose window nobody has latched yet. The two
    controls below fire both writes from the paths that OWN them, so nothing
    passes vacuously."""
    from ouroboros import config as cfg, task_pacing
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.task_results import task_result_path

    ctx, events = _floor_band_ctx(
        monkeypatch, tmp_path, seconds_left=620, policy="adaptive")  # 7.0 ABI: until_deadline retired
    del ctx.task_metadata["created_at"]  # metadata-poor: no anchor without a WRITE
    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    settings_path, settings_before = _legacy_settings(monkeypatch, tmp_path)
    result_file = task_result_path(tmp_path, "root-floor-band", create=False)
    from ouroboros import usage_store

    usage_store.migrate_from_journal(tmp_path)  # the lifecycle job, before any worker
    before = (events.read_bytes(), result_file.read_bytes())
    attrs_before = set(vars(ctx))

    for _ in range(2):
        live = coordination_live_context(ctx)
        assert live["time"]["state"] == "not_set"  # no latched anchor, and none taken
        assert live["time"] == {"state": "not_set", "remaining_sec": None,
                                "reserve_sec": None, "inside_reserve": None, "expired": None}
        assert live["review_capacity"]["state"] == "available"
        assert live["review_capacity"]["claimed_cycles"] == 1
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == settings_before
    assert (events.read_bytes(), result_file.read_bytes()) == before
    assert ctx.event_queue.empty() and ctx.pending_events == []
    assert set(vars(ctx)) == attrs_before  # the EXACT attribute set: nothing latched, nothing cached
    assert not hasattr(ctx, "_time_budget_started_at")  # the one latch a poll could take

    # Controls: each armed write really is reachable from its owning path.
    assert cfg.load_settings().get("OUROBOROS_CONTEXT_MODE") == "max"
    assert settings_path.read_bytes() != settings_before[0]  # load_settings persists
    assert task_pacing.resolve_budget_profile(ctx)["improvement_policy"] == "adaptive"
    # (the third control of the upstream test — the alias deprecation row and
    # its `_acceptance_pacing_deprecation_emitted` latch — is void: the
    # ``until_deadline`` alias and its emission are retired, 7.0 ABI Q10=A, and
    # nothing in the tree writes that attribute any more)
    assert task_pacing.build_budget_snapshot(ctx).has_deadline is True
    assert getattr(ctx, "_time_budget_started_at", None) is not None


def test_a_single_crash_torn_journal_gets_the_quarantine_its_one_time_import_performs(
        monkeypatch, tmp_path):
    """The one CONTENT write among the bounded maintenance a poll inherits from
    the canonical reader: on a root whose store was exported back to a journal
    (the downgrade step) and whose older release then crashed mid-append, the
    first read runs the one-time import, and the import repairs the journal
    exactly as every journal reader did — truncate to the intact prefix, one
    quarantine row holding the torn bytes verbatim, one
    `usage_ledger_tail_quarantined` event — before retiring it as
    ``usage_attempts.jsonl.imported``; the poll then reports the survivors as
    degraded integrity rather than final cost. Nothing else moves: the
    settings bytes and mtime, every earlier event row, the task result, the
    queue and the ctx attributes are unchanged, and a second poll writes
    nothing at all."""
    import base64

    from ouroboros import delegate_custody as custody, usage_accounting as ua, usage_store
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.task_results import task_result_path
    from ouroboros.usage_ledger import LEDGER_REL, QUARANTINE_REL

    ctx, events = _floor_band_ctx(monkeypatch, tmp_path, seconds_left=620)
    settings_path, settings_before = _legacy_settings(monkeypatch, tmp_path)
    # Seed the money the POLL itself will read (the custody root it resolves),
    # through the real attempt path, then export it as a journal.
    root = custody.custody_root(ctx)
    with ua.usage_scope(ua.UsageScope(
            drive_root=root, task_id="root-floor-band", root_task_id="root-floor-band")):
        ua.execute_physical_attempt(
            ua.AttemptRequest(model="local-review-test", provider="local", reservation_usd=0.0),
            lambda: ({"content": "seed"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}))
    usage_store.export_journal(root)
    ledger, quarantine = root / LEDGER_REL, root / QUARANTINE_REL
    intact, torn = ledger.read_bytes(), b'{"seq": 99, "attempt_id": "half-writ'
    ledger.write_bytes(intact + torn)  # the process died mid-append
    result_file = task_result_path(tmp_path, "root-floor-band", create=False)
    before = (events.read_bytes(), result_file.read_bytes(), set(vars(ctx)))

    live = coordination_live_context(ctx)

    spend = live["settled_spend"]  # it ANSWERED from the repaired prefix, honestly
    assert "reason" not in spend  # no read error: the repair is not a failure path
    assert spend["state"] == "partial" and spend["integrity_degraded"] is True
    assert spend["cost_final"] is False  # a quarantined tail never claims final cost
    assert ledger.read_bytes() == intact  # imported and kept in place, truncated to the intact prefix
    rows = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and base64.b64decode(rows[0]["raw_base64"]) == torn
    assert rows[0]["source"] == str(ledger)
    after = events.read_bytes()
    assert after.startswith(before[0])  # every earlier event row is untouched
    added = [json.loads(line) for line in after[len(before[0]):].decode("utf-8").splitlines()]
    assert [row["type"] for row in added] == ["usage_ledger_tail_quarantined"]
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == settings_before
    assert result_file.read_bytes() == before[1] and set(vars(ctx)) == before[2]
    assert ctx.event_queue.empty() and ctx.pending_events == []

    # The imported store is the healthy case again: the second poll writes nothing.
    steady = (ledger.read_bytes(), quarantine.read_bytes(), events.read_bytes(),
              (root / usage_store.STORE_REL).read_bytes())
    coordination_live_context(ctx)
    assert (ledger.read_bytes(), quarantine.read_bytes(), events.read_bytes(),
            (root / usage_store.STORE_REL).read_bytes()) == steady


def test_a_descendant_poll_gets_the_wallet_axis_and_no_time_axis_at_all(monkeypatch, tmp_path):
    """The projection a descendant embeds verbatim into its coordination
    payload carries the wallet and the cancellation state — and NOTHING about
    time. Pinned as the exact key set, so a new field cannot be added back
    without this test: the descendant reads its own window from the adjacent
    `time` fact, and the launch rule is evaluated where it belongs — once, at
    loop admission (owner R55) — never inside this projection."""
    from ouroboros.task_results import project_task_acceptance_review_capacity

    ctx, _events = _floor_band_ctx(monkeypatch, tmp_path, seconds_left=620)
    child = SimpleNamespace(
        task_id="child-1", root_task_id="root-floor-band", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), delegation_role="subagent",
        task_contract=ctx.task_contract,
        task_metadata={**ctx.task_metadata, "root_task_id": "root-floor-band",
                       "delegation_role": "subagent"},
        pending_events=[],
    )
    for ctx_under_test in (ctx, child):
        projection = project_task_acceptance_review_capacity(
            ctx_under_test, task_id=str(ctx_under_test.task_id))
        assert set(projection) == {
            "root_task_id", "cap_cycles", "claimed_cycles", "remaining_cycles",
            "binding_seen", "dedupe", "state", "reason",
        }
        assert projection["state"] == "available" and projection["claimed_cycles"] == 1


def test_reading_the_finalization_grace_never_touches_the_settings_file(monkeypatch, tmp_path):
    """`get_finalization_grace_sec` is env → the supplied settings mapping →
    the shipped default. With the env absent it must answer from the default
    without reading (and therefore migrating) settings from disk."""
    from ouroboros import config as cfg

    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    settings_path, before = _legacy_settings(monkeypatch, tmp_path)
    assert cfg.get_finalization_grace_sec() == cfg.FINALIZATION_GRACE_DEFAULT_SEC
    assert cfg.get_finalization_grace_sec({"OUROBOROS_FINALIZATION_GRACE_SEC": "45"}) == 45
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "9999")
    assert cfg.get_finalization_grace_sec() == 300  # the clamp is unchanged
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == before
    # Control: the file really is one a settings read would rewrite.
    assert cfg.load_settings().get("OUROBOROS_CONTEXT_MODE") == "max"
    assert settings_path.read_bytes() != before[0]
