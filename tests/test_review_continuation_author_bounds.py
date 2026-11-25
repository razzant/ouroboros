"""A selected review continuation answers to two different owners (#2901).

The wave's original attempt keeps its evidence and its payer, the root it was charged to, whose own recorded caps
travel with it. The deciding task (a Continue successor, on its own execution drive) owns the controls: its Stop is
read where cancellations live, and the read-only checkout put back for the send is marked for the task now sending
it. Synthetic data only; nothing reaches an engine or a provider.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, review_custody
from ouroboros import review_substrate as review
from ouroboros import usage_accounting as ua
from ouroboros.review_continuation import PREFIX, _fenced
from ouroboros.review_operation import bind_review_operation
from tests.test_billing_group import data_root as data_root  # noqa: F401 -- pytest fixture


def test_the_deciding_successors_stop_is_read_on_the_canonical_root(tmp_path):
    """A Continue successor runs on its own execution drive; Stop intents live on the canonical root, where the shared
    reader (``review_stop_cause``) reads them. Its Stop refuses the choice; the ended review task's own Stop is not the
    deciding task's control."""
    canonical = tmp_path / "data"
    child = canonical / "state" / "headless_tasks" / "successor" / "data"
    child.mkdir(parents=True)
    ctx = SimpleNamespace(task_id="successor", drive_root=child, budget_drive_root=canonical,
                          task_metadata={"root_task_id": "successor"})
    assert not _fenced(ctx, "predecessor")
    cancel_intents.request_cancel(canonical, "predecessor", reason="the review's own task ended")
    assert not _fenced(ctx, "predecessor")
    cancel_intents.request_cancel(canonical, "successor", reason="owner Stop of the deciding task")
    assert _fenced(ctx, "predecessor") and review_custody.review_stop_cause(ctx) == "stopped"


def test_a_restored_checkout_is_marked_for_the_task_sending_it(tmp_path, monkeypatch):
    """The START row's mapping names the task that decided; a later replay may be sent by another live author. The
    checkout put back for that POST is marked for the sender, whose live operation the checkout pass keeps; with no
    sending operation bound, the recorded decider stays the owner."""
    from ouroboros.review_session_custody import prepare_continuation_send
    from ouroboros.tools import review_subject

    owners = []
    monkeypatch.setattr(review_subject, "restore_released_subject",
                        lambda _root, mapping: owners.append(mapping["owner_task"]) or True)
    invocation = SimpleNamespace(operation_id=f"{PREFIX}synthetic")
    record = {"review_subject_restore": {"root": str(tmp_path / "checkout" / "repo"), "owner_task": "ended-decider",
                                         "subject": {}, "sources": []}}
    sender = SimpleNamespace(continuation_author=SimpleNamespace(task_id="live-sender"))
    with bind_review_operation(sender):
        prepare_continuation_send(invocation, tmp_path, record, "")
    prepare_continuation_send(invocation, tmp_path, record, "")
    assert owners == ["live-sender", "ended-decider"]
    assert record["review_subject_restore"]["owner_task"] == "ended-decider", "the recorded mapping is not rewritten"


def _spend(root, scope, amount):
    with ua.usage_scope(scope):
        held = ua.reserve_attempt(ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=amount))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {}, cost_usd=amount, cost_final=True)


def _root_scope(root, task, *, cap=None, group="", group_cap=None):
    return ua.UsageScope(drive_root=root, task_id=task, root_task_id=task, source="test", root_limit_usd=cap,
                         root_limit_source="test" if cap is not None else "", billing_group_id=group or task,
                         billing_group_limit_usd=group_cap, billing_group_limit_source="test")


def _continued_wave_admission(root, monkeypatch, deciding_scope=None, *, continuing=True):
    """The scope a selected continuation's wave sends under (its collector names the recorded payer root), and the
    at-send admission it meets there; ``deciding_scope`` None: no ambient usage scope at all."""
    import contextlib

    from ouroboros.tools.review_helpers import review_wave_budget_gate

    ctx = SimpleNamespace(drive_root=root, task_id="successor", task_metadata={"root_task_id": "payer"},
                          _review_receptive_continuation=continuing)
    coordinator = review.ReviewCoordinator(llm=SimpleNamespace(), drive_root=root, usage_ctx=ctx)
    observed = {}

    class Captured(Exception):
        pass

    def at_send(*_args, **_kwargs):
        observed["scope"] = ua.current_usage_scope()
        observed["admission"] = review_wave_budget_gate(SimpleNamespace(task_id="payer"), surface="scope_review",
                                                        models=["fixture"], prompt_chars=[10])
        raise Captured()
    monkeypatch.setattr(coordinator, "_run_slot", at_send)
    monkeypatch.setattr(review_custody, "run_custodied_review_slots",
                        lambda **kw: kw["run_slot"](kw["slots"][0], "continued-operation", {}, time.monotonic() + 60, None))
    ambient = ua.usage_scope(deciding_scope) if deciding_scope is not None else contextlib.nullcontext()
    with ambient, pytest.raises(Captured):
        coordinator.run(review.ReviewRequest(surface="scope_review", goal="continued", task_id="payer"),
                        [review.ReviewSlot(slot_id="slot", model="fixture")])
    return observed


def _today_default_cap(monkeypatch, value="0.01"):
    monkeypatch.setattr(review, "runtime_setting",
                        lambda key, default=None: value if key == "OUROBOROS_PER_TASK_COST_USD" else default)


@pytest.mark.parametrize("ambient", ["deciding_scope", "no_scope"])
@pytest.mark.parametrize("case", ["payer_cap_reached", "payer_has_room"])
def test_a_continued_wave_is_admitted_on_its_payers_own_recorded_cap(data_root, monkeypatch, case, ambient):  # noqa: F811
    """The payer root recorded a $1.00 cap. Its pair governs: reached, the send is refused though the deciding task
    has $50 of room; with room, it is admitted though the deciding task's own cap ($0.50) and today's default ($0.01)
    are below the payer's spend. Neither another root's cap nor today's configuration stands in for the payer's, with
    or without an ambient scope of the deciding task."""
    _today_default_cap(monkeypatch)
    _spend(data_root, _root_scope(data_root, "payer", cap=1.0, group_cap=100.0),
           1.0 if case == "payer_cap_reached" else 0.6)
    deciding_cap = 50.0 if case == "payer_cap_reached" else 0.5
    deciding = _root_scope(data_root, "successor", cap=deciding_cap, group_cap=deciding_cap)
    observed = _continued_wave_admission(data_root, monkeypatch, deciding if ambient == "deciding_scope" else None)
    scope = observed["scope"]
    assert (scope.root_task_id, scope.task_id) == ("payer", "payer")
    assert scope.root_limit_usd is None and scope.root_limit_source == "" and scope.billing_group_id == "", scope
    admission = observed["admission"]
    if case == "payer_cap_reached":
        assert admission is not None and admission["binding_axis"] == "root", admission
        assert (admission["limit_usd"], admission["known_usd"]) == (pytest.approx(1.0), pytest.approx(1.0)), admission
    else:
        assert admission is None


def test_a_payer_with_no_recorded_cap_gets_none_invented(data_root, monkeypatch):  # noqa: F811
    """The payer recorded no root cap: the continuation gets none, neither today's default nor another root's, so its
    $1.00 of known spend is no refusal on the root axis."""
    _today_default_cap(monkeypatch)
    _spend(data_root, _root_scope(data_root, "payer", group_cap=100.0), 1.0)
    observed = _continued_wave_admission(data_root, monkeypatch, None)
    assert observed["scope"].root_limit_usd is None and observed["admission"] is None, observed


@pytest.mark.parametrize("ambient", ["deciding_scope", "no_scope"])
def test_a_continued_wave_meets_its_payers_whole_work_group_with_the_groups_other_work(data_root, monkeypatch,  # noqa: F811
                                                                                       ambient):
    """The payer belongs to the whole-work group ``original`` ($2.00), whose other work spent $1.50: with the payer's
    $0.50 the group's known spend reached its cap, though the payer's own root has room and the deciding task's group
    and today's default say otherwise. That original group is the fence, never another group or today's setting."""
    _today_default_cap(monkeypatch, "1000")
    _spend(data_root, _root_scope(data_root, "payer", cap=10.0, group="original", group_cap=2.0), 0.5)
    _spend(data_root, _root_scope(data_root, "sibling", cap=10.0, group="original", group_cap=2.0), 1.5)
    deciding = _root_scope(data_root, "successor", cap=50.0, group="other", group_cap=50.0)
    observed = _continued_wave_admission(data_root, monkeypatch, deciding if ambient == "deciding_scope" else None)
    admission = observed["admission"]
    assert admission is not None and admission["binding_axis"] == "group", admission
    assert admission.get("billing_group_id") == "original", admission
    assert (admission["limit_usd"], admission["known_usd"]) == (pytest.approx(2.0), pytest.approx(2.0)), admission


def test_an_ordinary_new_wave_keeps_todays_default_cap_without_a_scope(data_root, monkeypatch):  # noqa: F811
    """The negative control of the no-scope rule: a new wave that is no selected continuation still takes the
    configured per-task cap as before."""
    _today_default_cap(monkeypatch, "2")
    observed = _continued_wave_admission(data_root, monkeypatch, None, continuing=False)
    assert observed["scope"].root_limit_usd == 2.0


def test_a_wave_charged_to_its_own_root_keeps_its_scopes_caps(data_root, monkeypatch):  # noqa: F811
    """The negative control: the ordinary wave, charged to the scope's own root, keeps that scope's cap and group."""
    ctx = SimpleNamespace(drive_root=data_root, task_id="successor", task_metadata={"root_task_id": "successor"})
    coordinator = review.ReviewCoordinator(llm=SimpleNamespace(), drive_root=data_root, usage_ctx=ctx)
    observed = {}

    class Captured(Exception):
        pass

    def at_send(*_args, **_kwargs):
        observed["scope"] = ua.current_usage_scope()
        raise Captured()
    monkeypatch.setattr(coordinator, "_run_slot", at_send)
    monkeypatch.setattr(review_custody, "run_custodied_review_slots",
                        lambda **kw: kw["run_slot"](kw["slots"][0], "operation", {}, time.monotonic() + 60, None))
    own = ua.UsageScope(drive_root=data_root, task_id="successor", root_task_id="successor", source="test",
                        root_limit_usd=3.0, root_limit_source="test", billing_group_id="original",
                        billing_group_limit_usd=7.0, billing_group_limit_source="ledger_first_row")
    with ua.usage_scope(own), pytest.raises(Captured):
        coordinator.run(review.ReviewRequest(surface="scope_review", goal="own", task_id="successor"),
                        [review.ReviewSlot(slot_id="slot", model="fixture")])
    scope = observed["scope"]
    assert (scope.root_limit_usd, scope.root_limit_source) == (3.0, "test")
    assert (scope.billing_group_id, scope.billing_group_limit_usd) == ("original", 7.0)


# ---------------------------------------------------------------------------
# End-to-end synthetic Continue-successor bounds (commit_reviewed + FakeGateway)
# ---------------------------------------------------------------------------

from tests.test_review_continue_separate_drive import (  # noqa: E402,F401 -- shared Continue helpers
    SUCCESSOR as _SUCCESSOR, _choice, _start, _successor,
)
from tests.test_review_explicit_continuation import (  # noqa: E402,F401 -- shared fixtures
    _commit_call, _commit_json, _git, _owned_gateway_uses_each_test_transport,
    _scope, author, fake_route, stopped_wave,
)


def _setup_continue_wave(a, *, successor=_SUCCESSOR):
    """Run predecessor's commit review to a stopped session seat, then switch ``a.ctx`` to a recorded Continue successor."""
    canonical, predecessor, pred_root, first = _start(a, "child")
    succ_root = _successor(a, canonical, predecessor, pred_root, successor=successor)
    a.ctx.task_metadata = {"root_task_id": successor}
    return canonical, predecessor, pred_root, succ_root, first


def _prepare_predecessor_continuation_start(a, canonical, predecessor, pred_root, ref, offer):
    """Predecessor decides and projects its continuation START onto its attempt row without sending."""
    from ouroboros.review_ledger import load_record
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    saved_id, saved_drive, saved_meta = a.ctx.task_id, a.ctx.drive_root, dict(a.ctx.task_metadata)
    a.ctx.task_id, a.ctx.drive_root, a.ctx.task_metadata = predecessor, pred_root, {"root_task_id": predecessor}
    try:
        home, attempt, record = resolve_reference(a.ctx, ref, commit=True)
        continue_or_repair(a.ctx, home, attempt, record, [_choice(offer)])
    finally:
        a.ctx.task_id, a.ctx.drive_root, a.ctx.task_metadata = saved_id, saved_drive, saved_meta
    seat = next(row for row in load_record(canonical, ref["review_record_id"])["rows"] if row["seat_id"] == "second")
    return seat["continuations"][-1]["invocation_id"], seat["continuations"][-1]["operation_id"]


@pytest.mark.parametrize("case", ["lookup_found", "lookup_not_found", "lookup_unsupported", "lookup_unavailable",
                                  "legacy_no_handoff_not_found", "lookup_found_bind_busy"])
def test_cold_successor_after_predecessor_handoff_never_reposts_but_preserves_accepted_lookup(
    stopped_wave, monkeypatch, case,  # noqa: F811
):
    """After the predecessor's POST entered handoff (or on a row with no handoff proof) and lost its reply, a cold
    live Continue successor takes a strictly non-creating lookup path: a temporary miss/unavailable/unsupported lookup,
    or a found run whose binding lock is busy, never POSTs and never permanently caches an error, so a later lookup on
    the same key and service still binds and collects the accepted run without creating a second one."""
    from ouroboros import delegate_custody, delegate_invocation
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_ledger import load_record
    from ouroboros.review_operation import START_UNBOUND
    from ouroboros.review_state import load_state
    from tests._review_session_route_shared import FakeGateway

    a = stopped_wave
    canonical, predecessor, pred_root, _succ_root, first = _setup_continue_wave(a)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    token, op_id = _prepare_predecessor_continuation_start(a, canonical, predecessor, pred_root, ref, offer)
    assert delegate_custody.invocation_record(canonical, token)["handoff"] == delegate_invocation.HANDOFF_UNSENT

    if case == "legacy_no_handoff_not_found":
        orig_record = delegate_custody.invocation_record
        orig_inv = delegate_invocation.invocation_record

        def strip_handoff(fn):
            def wrapped(root, inv_id, *args, **kwargs):
                rec = fn(root, inv_id, *args, **kwargs)
                if isinstance(rec, dict) and inv_id == token:
                    return {k: v for k, v in rec.items() if k != "handoff"}
                return rec
            return wrapped

        monkeypatch.setattr(delegate_custody, "invocation_record", strip_handoff(orig_record))
        monkeypatch.setattr(delegate_invocation, "invocation_record", strip_handoff(orig_inv))
    else:
        assert delegate_custody.emit(canonical, delegate_custody.START_HANDOFF, {
            "invocation_id": token, "task_id": predecessor})
        assert delegate_custody.invocation_record(canonical, token)["handoff"] == delegate_invocation.HANDOFF_ENTERED

    lookups = []
    store = a.engine.store or "synthetic-store"
    phase = {"mode": case}
    monkeypatch.setattr(FakeGateway, "operations",
                        lambda self: [] if phase["mode"] == "lookup_unsupported" else [{"method": "POST", "path": "/v2/runs/lookup"}],
                        raising=False)

    def fake_lookup(self, body, *, idempotency_key=""):
        lookups.append(idempotency_key)
        if phase["mode"] == "lookup_unavailable":
            raise ClaudexorUnavailable("daemon_unreachable", "lookup 503", status_code=503)
        if phase["mode"] in ("lookup_found", "lookup_found_bind_busy"):
            a.engine.fates["run-accepted"] = "answer"
            phase["busy"] = phase["mode"] == "lookup_found_bind_busy"  # the bind right after this read meets a held lock
            return {"status": "found", "storeId": store,
                    "handle": {"jobId": "run-accepted", "runId": "run-accepted", "state": "succeeded"}}
        return {"status": "not_found", "storeId": store}

    real_lock = delegate_custody.actor_decision_lock

    def decision_lock(*args, **kwargs):
        if phase.pop("busy", False):
            raise delegate_custody.ActorDecisionLockUnavailable({"reason": "contention"}, 20.0)
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(delegate_custody, "actor_decision_lock", decision_lock)
    monkeypatch.setattr(FakeGateway, "lookup_run", fake_lookup, raising=False)
    posts_before = len(a.engine.posts)
    with _scope(a, _SUCCESSOR):
        text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])

    assert len(a.engine.posts) == posts_before, f"cold successor must not POST after entered/unknown handoff: {a.engine.posts}"
    if case == "lookup_found":
        record = load_record(canonical, ref["review_record_id"])
        seat = next(row for row in record["rows"] if row["seat_id"] == "second")
        inv = delegate_custody.invocation_record(canonical, token)
        assert lookups == [token] and record["state"] == "settled"
        assert "CONTINUED-SEAT-ANSWER" in text and seat["status"] == "responded" and seat["operation_id"] == op_id
        assert inv["state"] == "started" and inv["run_id"] == "run-accepted"
        return

    # Call 1 missed/errored: zero POSTs, invocation and ledger stay pending, attempt row records a retryable
    # custody_lost (START_UNBOUND, no permanent cache).
    assert len(lookups) == (0 if case == "lookup_unsupported" else 1), lookups
    inv = delegate_custody.invocation_record(canonical, token)
    assert inv["state"] == "pending", f"must stay pending, never failed_definite: {inv}"
    assert load_record(canonical, ref["review_record_id"])["state"] == "pending"
    attempt_row = next(r for r in load_state(pred_root).latest_attempt_for(
        task_id=predecessor, attempt=ref["attempt"]).triad_raw_results if r["slot_id"] == "second")
    assert attempt_row["operation_id"] == op_id and attempt_row["operation_state"] == "custody_lost"
    assert attempt_row.get("failure_code") == ("accepted_start_unbound" if case == "lookup_found_bind_busy"
                                               else "review_continuation_handoff_unknown")
    assert (attempt_row.get("usage") or {}).get("custody_state") == START_UNBOUND, attempt_row
    assert attempt_row.get("pending_invocation_id") == token

    # While still absent, a repeat call continues to forbid POST.
    if case == "lookup_not_found":
        with _scope(a, _SUCCESSOR):
            a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        assert len(a.engine.posts) == posts_before and len(lookups) == 2

    # Later on the same key and service, lookup finds the accepted run: binds and settles without creating a second run.
    phase["mode"] = "lookup_found"
    with _scope(a, _SUCCESSOR):
        recovered_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    assert len(a.engine.posts) == posts_before, f"recovery lookup must never POST: {a.engine.posts}"
    assert "CONTINUED-SEAT-ANSWER" in recovered_text
    record_after = load_record(canonical, ref["review_record_id"])
    seat_after = next(row for row in record_after["rows"] if row["seat_id"] == "second")
    inv_after = delegate_custody.invocation_record(canonical, token)
    assert record_after["state"] == "settled" and seat_after["status"] == "responded" and seat_after["operation_id"] == op_id
    assert inv_after["state"] == "started" and inv_after["run_id"] == "run-accepted"


def test_cold_successor_sends_once_when_start_is_proven_unsent_and_joins_live_worker_without_post(
    stopped_wave,  # noqa: F811
):
    """Positive controls for send authority: a START durably marked ``unsent`` may be sent once by a cold Continue
    successor, and an in-process live worker in ``_ACTIVE`` is joined without a second POST."""
    import threading

    from ouroboros import delegate_custody, delegate_invocation
    from ouroboros.review_ledger import load_record
    from tests._review_session_route_shared import _terminal_detail
    from tests.test_receptive_commit_review import _produce

    a = stopped_wave
    canonical, predecessor, pred_root, _succ_root, first = _setup_continue_wave(a)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    token, op_id = _prepare_predecessor_continuation_start(a, canonical, predecessor, pred_root, ref, offer)
    assert delegate_custody.invocation_record(canonical, token)["handoff"] == delegate_invocation.HANDOFF_UNSENT

    # Hold run-2 open while one caller sends the proven-unsent START, so a second live Continue caller joins _ACTIVE.
    running, release, first_out = threading.Event(), threading.Event(), {}
    orig_detail = a.engine.detail

    def hold_run_2(gateway, run_id):
        if run_id == "run-2" and not release.is_set():
            running.set()
            return _terminal_detail("", state="running")
        return orig_detail(gateway, run_id)

    a.engine.detail = hold_run_2
    try:
        def sender():
            with _scope(a, _SUCCESSOR):
                first_out["text"] = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        worker = _produce(a, sender)
        assert running.wait(30), first_out
        assert len(a.engine.posts) == 2 and a.engine.posts[-1][1] == token
        # Second caller arrives while the live worker owns op_id: release after join starts; zero extra POSTs.
        import copy

        join_ctx = copy.copy(a.ctx)
        join_out = {}

        def joiner():
            with _scope(a, _SUCCESSOR):
                join_out["text"] = a.git._commit_reviewed(join_ctx, review_reference=ref, continue_seats=[_choice(offer)])
        join_thread = _produce(a, joiner)
        time.sleep(0.15)
        release.set()
        worker.join(30)
        join_thread.join(30)
        assert not worker.is_alive() and not join_thread.is_alive()
    finally:
        release.set()
        a.engine.detail = orig_detail

    assert len(a.engine.posts) == 2, f"live join must not POST again: {a.engine.posts}"
    assert "CONTINUED-SEAT-ANSWER" in first_out.get("text", "") and "CONTINUED-SEAT-ANSWER" in join_out.get("text", "")
    seat = next(row for row in load_record(canonical, ref["review_record_id"])["rows"] if row["seat_id"] == "second")
    assert seat["operation_id"] == op_id and seat["status"] == "responded"


@pytest.mark.parametrize("ambient", ["rich_deciding_scope", "no_scope"])
@pytest.mark.parametrize("fence", ["payer_root", "payer_group"])
def test_exhausted_payer_refuses_before_choice_and_preserves_free_read_and_disposition(
    stopped_wave, monkeypatch, fence, ambient,  # noqa: F811
):
    """When the original payer's own root or whole-work group cap is already exhausted, ``arm`` refuses at $0 BEFORE
    recording a choice, writing START_REQUESTED or replacing the old slot — whether the deciding successor has a rich
    ambient scope or no ambient scope at all. Free reading and ``author_disposition`` stay unblocked."""
    import contextlib

    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import load_state

    a = stopped_wave
    # Pin the predecessor's durable whole-work billing group before its first review attempt writes a ledger row.
    if fence == "payer_group":
        _spend(a.data, _root_scope(a.data, a.ctx.task_id, cap=10.0, group="orig-grp", group_cap=2.0), 0.5)
    canonical, predecessor, pred_root, _succ_root, first = _setup_continue_wave(a)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    if fence == "payer_root":
        _spend(canonical, _root_scope(canonical, predecessor, cap=1.0, group_cap=100.0), 1.0)
    else:
        _spend(canonical, _root_scope(canonical, "sibling-payer", cap=10.0, group="orig-grp", group_cap=2.0), 1.5)

    record_before = load_record(canonical, ref["review_record_id"])
    row_before = next(r for r in load_state(pred_root).latest_attempt_for(
        task_id=predecessor, attempt=ref["attempt"]).triad_raw_results if r["slot_id"] == "second")
    deciding = _root_scope(canonical, _SUCCESSOR, cap=50.0, group="other-grp", group_cap=50.0)
    ambient_ctx = lambda: ua.usage_scope(deciding) if ambient == "rich_deciding_scope" else contextlib.nullcontext()

    with ambient_ctx():
        refused = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])

    assert "budget_insufficient" in refused and "REVIEW_CONTINUATION_UNSENT" not in refused, refused[:600]
    assert len(a.engine.posts) == 1
    record_after = load_record(canonical, ref["review_record_id"])
    assert record_after["revision"] == record_before["revision"]
    seat_after = next(r for r in record_after["rows"] if r["seat_id"] == "second")
    assert seat_after["operation_id"] == offer["operation_id"] and not seat_after.get("continuations"), seat_after
    row_after = next(r for r in load_state(pred_root).latest_attempt_for(
        task_id=predecessor, attempt=ref["attempt"]).triad_raw_results if r["slot_id"] == "second")
    assert row_after["operation_id"] == row_before["operation_id"], "old slot must not be replaced/reopened"

    # Free reading and author_disposition remain unblocked on the exact same exhausted payer.
    with ambient_ctx():
        free_read = a.git._commit_reviewed(a.ctx, review_reference=ref)
    assert "recorded Continue predecessor" in free_read and "budget_insufficient" not in free_read, free_read[:400]
    head_before = _git(a.repo, "rev-parse", "HEAD")
    with ambient_ctx():
        committed = a.git._commit_reviewed(a.ctx, "Commit with exhausted payer", review_reference=ref,
                                           author_disposition={"disposition": "accepted", "rationale": "Read review."})
    assert "budget_insufficient" not in committed and _git(a.repo, "rev-parse", "HEAD") != head_before, committed[:400]
    assert len(a.engine.posts) == 1


@pytest.mark.parametrize("ambient", ["rich_deciding_scope", "no_scope"])
def test_exhausted_payer_refuses_before_start_and_roster_carry_in_project(
    stopped_wave, monkeypatch, ambient,  # noqa: F811
):
    """If a choice was recorded before the original payer hit its limit, repairing projection in ``_project`` refuses
    before writing START_REQUESTED and before replacing the old stopped slot on the attempt row."""
    import contextlib

    from ouroboros import delegate_custody, review_continuation
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import load_state

    a = stopped_wave
    canonical, predecessor, pred_root, _succ_root, first = _setup_continue_wave(a)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    # Step 1: choice is recorded while payer still has room, then projection fails before START/carry.
    _spend(canonical, _root_scope(canonical, predecessor, cap=1.0, group_cap=100.0), 0.4)
    real_start = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *_a, **_k: (_ for _ in ()).throw(OSError("interrupted")))
    first_unsent = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    monkeypatch.setattr(review_continuation, "_start_requested", real_start)
    assert "REVIEW_CONTINUATION_UNSENT" in first_unsent, first_unsent[:400]
    seat = next(r for r in load_record(canonical, ref["review_record_id"])["rows"] if r["seat_id"] == "second")
    token = seat["continuations"][-1]["invocation_id"]
    assert delegate_custody.invocation_record(canonical, token) is None

    # Step 2: original payer hits its $1.00 cap; deciding successor still has $50 (or no ambient scope).
    _spend(canonical, _root_scope(canonical, predecessor, cap=1.0, group_cap=100.0), 0.6)
    deciding = _root_scope(canonical, _SUCCESSOR, cap=50.0, group_cap=50.0)
    ctx_scope = ua.usage_scope(deciding) if ambient == "rich_deciding_scope" else contextlib.nullcontext()
    with ctx_scope:
        refused = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])

    assert "REVIEW_CONTINUATION_UNSENT" in refused and "budget_insufficient" in refused, refused[:600]
    assert delegate_custody.invocation_record(canonical, token) is None, "no START_REQUESTED written"
    row_after = next(r for r in load_state(pred_root).latest_attempt_for(
        task_id=predecessor, attempt=ref["attempt"]).triad_raw_results if r["slot_id"] == "second")
    assert row_after["operation_id"] == offer["operation_id"], "old stopped slot must stay intact"
    assert len(a.engine.posts) == 1


def test_payer_with_room_continues_despite_lower_deciding_cap_while_deciding_stop_still_fences(
    stopped_wave, monkeypatch,  # noqa: F811
):
    """Positive control: when the original payer has room ($0.60/$1.00), choice + START + send succeed even if the
    deciding task's own cap ($0.50) and today's default cap ($0.01) are lower; the deciding task's own Stop still
    refuses before choice."""
    from ouroboros.review_ledger import load_record

    a = stopped_wave
    canonical, predecessor, _pred_root, _succ_root, first = _setup_continue_wave(a)
    _today_default_cap(monkeypatch, "0.01")
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    _spend(canonical, _root_scope(canonical, predecessor, cap=2.0, group_cap=100.0), 0.6)
    deciding = _root_scope(canonical, _SUCCESSOR, cap=0.5, group_cap=0.5)

    # Deciding task's Stop still refuses before choice; a non-stopped Continue successor proceeds on the payer's room.
    cancel_intents.request_cancel(canonical, _SUCCESSOR, reason="deciding stop")
    with ua.usage_scope(deciding):
        stopped_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    assert "stopped_or_paused" in stopped_text and len(a.engine.posts) == 1, stopped_text[:400]
    assert not next(r for r in load_record(canonical, ref["review_record_id"])["rows"]
                    if r["seat_id"] == "second").get("continuations")

    # Next Continue successor (not stopped, $0.00/$0.50 own spend): payer's room ($0.60/$2.00) admits choice, START and send.
    s2 = "synthetic-continue-successor-2"
    _successor(a, canonical, _SUCCESSOR, _succ_root, successor=s2)
    a.ctx.task_metadata = {"root_task_id": s2}
    deciding2 = _root_scope(canonical, s2, cap=0.5, group_cap=0.5)
    with ua.usage_scope(deciding2):
        ok_text = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
    assert "CONTINUED-SEAT-ANSWER" in ok_text and len(a.engine.posts) == 2, ok_text[:500]
    seat = next(r for r in load_record(canonical, ref["review_record_id"])["rows"] if r["seat_id"] == "second")
    assert seat["status"] == "responded"


@pytest.mark.parametrize("stage", ["before_choice", "before_start"])
def test_deciding_authors_own_exhausted_ambient_limit_refuses_choice_and_start_while_payer_has_room(
    stopped_wave, monkeypatch, stage,  # noqa: F811
):
    """Point 1 contrast: the original payer still has room ($0.60/$2.00), while the deciding Continue successor
    reaches its OWN limit with its OWN spend ($0.50/$0.50) under its own ambient scope. That existing deciding-author
    boundary still refuses before choice (``arm``) and before START/carry (``_project``); free reads and
    ``author_disposition`` stay unblocked."""
    from ouroboros import delegate_custody, review_continuation
    from ouroboros.review_ledger import load_record
    from ouroboros.review_state import load_state

    a = stopped_wave
    canonical, predecessor, pred_root, _succ_root, first = _setup_continue_wave(a)
    ref, [offer] = first["review_reference"], first["continuable_seats"]
    _spend(canonical, _root_scope(canonical, predecessor, cap=2.0, group_cap=100.0), 0.6)
    deciding = _root_scope(canonical, _SUCCESSOR, cap=0.5, group_cap=100.0)

    token = None
    if stage == "before_start":
        # Choice is recorded while deciding successor still has room, then projection fails before START/carry.
        real_start = review_continuation._start_requested
        monkeypatch.setattr(review_continuation, "_start_requested",
                            lambda *_a, **_k: (_ for _ in ()).throw(OSError("interrupted")))
        with ua.usage_scope(deciding):
            unsent = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])
        monkeypatch.setattr(review_continuation, "_start_requested", real_start)
        assert "REVIEW_CONTINUATION_UNSENT" in unsent, unsent[:400]
        seat = next(r for r in load_record(canonical, ref["review_record_id"])["rows"] if r["seat_id"] == "second")
        token = seat["continuations"][-1]["invocation_id"]
        assert delegate_custody.invocation_record(canonical, token) is None

    # Now deciding author spends its OWN $0.50/$0.50 cap while payer still has $1.40 room.
    _spend(canonical, deciding, 0.5)
    with ua.usage_scope(deciding):
        refused = a.git._commit_reviewed(a.ctx, review_reference=ref, continue_seats=[_choice(offer)])

    assert "budget_insufficient" in refused and len(a.engine.posts) == 1, refused[:600]
    row_after = next(r for r in load_state(pred_root).latest_attempt_for(
        task_id=predecessor, attempt=ref["attempt"]).triad_raw_results if r["slot_id"] == "second")
    assert row_after["operation_id"] == offer["operation_id"], "old stopped slot must stay intact"
    if stage == "before_choice":
        assert "REVIEW_CONTINUATION_UNSENT" not in refused
        assert not next(r for r in load_record(canonical, ref["review_record_id"])["rows"]
                        if r["seat_id"] == "second").get("continuations")
        # Free reading and author_disposition remain unblocked despite deciding author's own exhausted cap.
        with ua.usage_scope(deciding):
            free_read = a.git._commit_reviewed(a.ctx, review_reference=ref)
        assert "recorded Continue predecessor" in free_read and "budget_insufficient" not in free_read, free_read[:400]
        head_before = _git(a.repo, "rev-parse", "HEAD")
        with ua.usage_scope(deciding):
            committed = a.git._commit_reviewed(a.ctx, "Commit with deciding cap spent", review_reference=ref,
                                               author_disposition={"disposition": "accepted", "rationale": "Read review."})
        assert "budget_insufficient" not in committed and _git(a.repo, "rev-parse", "HEAD") != head_before, committed[:400]
    else:
        assert "REVIEW_CONTINUATION_UNSENT" in refused
        assert delegate_custody.invocation_record(canonical, token) is None, "no START_REQUESTED written"
    assert len(a.engine.posts) == 1


def test_concurrent_first_send_handoff_is_serialized_by_actor_decision_lock(tmp_path, monkeypatch):
    """Point 3: ``continuation_send_admitted`` serializes the short read+transition (``never_handed_off`` +
    ``emit(START_HANDOFF)``) under the existing ``delegate_custody.actor_decision_lock`` without holding the lock
    across network calls: of two racing first-send callers on the same ``HANDOFF_UNSENT`` token, exactly one wins."""
    import threading

    from ouroboros import delegate_custody, delegate_invocation
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.review_session_custody import continuation_send_admitted

    root = tmp_path / "data"
    root.mkdir()
    token = "inv-race-handoff"
    assert delegate_custody.record_start_requested(
        root, gateway=None, run_id="", task_id="payer-task", idempotency_key=token, invocation_id=token,
        operation_id=f"{PREFIX}race", max_seconds=60, request={"prompt": "continue"},
        surface="multi_model_review", slot_id="second", route="fake", root_task_id="payer-task",
        handoff=delegate_invocation.HANDOFF_UNSENT)

    # Hold the first emit briefly inside actor_decision_lock so the second thread actually contends on the lock.
    real_emit = delegate_custody.emit
    entered_emit = threading.Event()
    outcomes, lock = [], threading.Lock()

    def slow_emit(drive, kind, payload, *args, **kwargs):
        if kind == delegate_custody.START_HANDOFF and not entered_emit.is_set():
            entered_emit.set()
            time.sleep(0.1)
        return real_emit(drive, kind, payload, *args, **kwargs)

    monkeypatch.setattr(delegate_custody, "emit", slow_emit)
    invocation = SimpleNamespace(operation_id=f"{PREFIX}race", task_id="payer-task", surface="multi_model_review",
                                 slot_id="second", first_send_only=True,
                                 retry_state={"pending_invocation_id": token})

    def racer():
        try:
            continuation_send_admitted(invocation, None, {"prompt": "continue"}, "fixture", root)
            with lock:
                outcomes.append("admitted")
        except ReviewRouteUnavailable as exc:
            with lock:
                outcomes.append(exc.code)

    t1 = threading.Thread(target=racer)
    t2 = threading.Thread(target=racer)
    t1.start()
    assert entered_emit.wait(10)
    t2.start()
    t1.join(10)
    t2.join(10)
    assert sorted(outcomes) == ["admitted", "review_continuation_handoff_unknown"], outcomes
    assert delegate_custody.invocation_record(root, token)["handoff"] == delegate_invocation.HANDOFF_ENTERED
    # Lock was released on return (network POST stays outside actor_decision_lock).
    with delegate_custody.actor_decision_lock(root, "payer-task", wait=False):
        pass
