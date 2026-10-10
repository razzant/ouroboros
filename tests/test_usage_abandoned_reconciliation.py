"""Existing terminal maintenance closes abandoned work without inventing prices."""

import hashlib
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import server_maintenance as maintenance
from ouroboros import usage_accounting as usage
from ouroboros.observability import persist_call
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.usage_ledger import is_abandoned_settlement
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def env(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon
    from supervisor import queue

    live = set()
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda task_id, **_kw: task_id in live)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: pytest.fail("native cleanup needs no daemon"))
    depth = [0]
    original_lock = usage._locked

    @contextmanager
    def locked(root):
        with original_lock(root) as heartbeat:
            depth[0] += 1
            try:
                yield heartbeat
            finally:
                depth[0] -= 1

    monkeypatch.setattr(usage, "_locked", locked)
    return SimpleNamespace(root=tmp_path, live=live, lock_depth=depth)


def _attempt(env, task_id="child", *, provider="openai", state="dispatched", bound=1.25):
    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="test-model", provider=provider, drive_root=env.root,
        task_id=task_id, root_task_id="root", parent_task_id="root",
        reservation_usd=bound, force_unknown_reservation=bound is None,
        global_limit_usd=100.0,
    ))
    if state != "reserved":
        usage.mark_dispatched(reservation)
    if state == "unresolved":
        usage.mark_unresolved(reservation, "response missing")
    return reservation


def _terminal(env, task_id="child", *, checkpoint=None):
    return write_task_result(
        env.root, task_id, "cancelled", result=f"Preserved {task_id} answer",
        root_task_id="root", parent_task_id="" if task_id == "root" else "root",
        delegation_role="root" if task_id == "root" else "subagent",
        **({"root_phase_checkpoint": checkpoint} if checkpoint else {}),
    )


def _final(env, reservation):
    with usage._locked(env.root):
        return {row['attempt_id']: row for row in ledger_rows(env.root)}[reservation.attempt_id]


def _events(env):
    path = env.root / "logs" / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("type") == "task_cost_finalized"] if path.exists() else []


def test_early_cancel_is_revisited_by_the_existing_maintenance_pass(env, monkeypatch):
    from ouroboros import observability
    from supervisor import task_lifecycle, terminal_delivery

    monkeypatch.setattr(task_lifecycle, "sweep_cancel_intents", lambda: {})
    monkeypatch.setattr(terminal_delivery, "replay_pending_deliveries", lambda root: None)
    monkeypatch.setattr(observability, "retry_pending_child_ref_promotions", lambda root: None)
    lock = threading.Lock()
    monkeypatch.setattr(maintenance, "_CANCEL_INTENT_SWEEP_LOCK", lock)
    _terminal(env)
    _terminal(env, "root", checkpoint={"post_task_synthesis": "completed"})
    reservation = _attempt(env)
    env.live.add("child")

    lock.acquire()
    maintenance._run_cancel_delivery_ref_sweep(env.root)
    assert _final(env, reservation)["state"] == "dispatched"
    env.live.clear()
    lock.acquire()
    maintenance._run_cancel_delivery_ref_sweep(env.root)

    assert not lock.locked()
    assert is_abandoned_settlement(_final(env, reservation))
    projected = usage.usage_projection(env.root)
    assert projected["accounted_usd"] == 1.25
    assert projected["confirmed_usd"] == 0
    assert projected["cost_final"] is False
    child = load_task_result(env.root, "child")
    root = load_task_result(env.root, "root")
    assert child["accounted_upper_bound_usd"] == 1.25 and child["cost_final"] is False
    assert root["accounted_upper_bound_usd_with_children"] == 1.25
    assert root["root_phase_checkpoint"] == {"post_task_synthesis": "completed"}
    assert root["non_final_rows"] == 1 and root["cost_final"] is False
    assert root["result"] == "Preserved root answer" and root["status"] == "cancelled"
    before = ledger_rows(env.root), _events(env)
    lock.acquire()
    maintenance._run_cancel_delivery_ref_sweep(env.root)
    assert (ledger_rows(env.root), _events(env)) == before


def test_live_review_and_post_task_work_are_spared_and_task_reads_are_cached(env, monkeypatch):
    import ouroboros.task_results as results

    _terminal(env, "live")
    _terminal(env, "review")
    _terminal(env, "synthesis", checkpoint={"post_task_synthesis": "running"})
    env.live.add("live")
    reservations = [_attempt(env, "live"), _attempt(env, "live"), _attempt(env, "synthesis")]
    with usage.usage_scope(usage.UsageScope(review_slot_id="acceptance-slot")):
        reservations.append(_attempt(env, "review"))
    reads = []
    original = results.load_task_result
    monkeypatch.setattr(results, "load_task_result", lambda root, task_id, **kw: reads.append(task_id) or original(root, task_id, **kw))

    maintenance._reconcile_abandoned_usage(env.root)

    assert all(_final(env, reservation)["state"] == "dispatched" for reservation in reservations)
    assert reads.count("live") == 1 and "review" not in reads
    assert not _events(env)


def test_a_real_receipt_winning_the_abandonment_race_is_preserved(env, monkeypatch):
    _terminal(env)
    reservation = _attempt(env)
    terminalize = usage.terminalize_abandoned_attempt

    def racing(current, **kwargs):
        usage.settle_attempt(current, cost_usd=0.4, cost_final=True)
        return terminalize(current, **kwargs)

    monkeypatch.setattr(usage, "terminalize_abandoned_attempt", racing)
    maintenance._reconcile_abandoned_usage(env.root)

    assert _final(env, reservation)["cost_usd"] == 0.4
    assert not is_abandoned_settlement(_final(env, reservation))
    assert usage.usage_projection(env.root)["confirmed_usd"] == 0.4
    assert load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 0.4


def test_cost_refresh_rechecks_post_task_ownership_under_the_result_lock(env, monkeypatch):
    from supervisor import events_task_done

    _terminal(env)
    _attempt(env)
    original = events_task_done.write_task_result

    def reopened(root, task_id, status, **fields):
        original(root, task_id, status, root_phase_checkpoint={"post_task_synthesis": "running"})
        return original(root, task_id, status, **fields)

    monkeypatch.setattr(events_task_done, "write_task_result", reopened)
    with pytest.raises(ValueError, match="lost terminal task ownership"):
        events_task_done._refresh_terminal_task_cost(env.root, "child")
    stored = load_task_result(env.root, "child")
    assert "accounted_upper_bound_usd" not in stored
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    assert stored["result"] == "Preserved child answer" and not _events(env)


def _remote_request(env, reservation, operation_id):
    persist_call(env.root, task_id="child", call_id=f"{reservation.attempt_id}_model_request",
                 call_type="llm_claudexor_request", payload={}, keep_raw=True,
                 manifest={"invocation_id": reservation.attempt_id, "operation_id": operation_id})


def _remote_receipt(env, reservation, operation_id):
    raw = json.dumps({"outcome": "completed", "usage": {}, "cost": {"knowledge": "exact", "cashUsd": 0.4}})
    persist_call(env.root, task_id="child", call_id=f"{reservation.attempt_id}_model_response",
                 call_type="llm_claudexor_response", payload={"result_json_utf8": raw}, keep_raw=True,
                 manifest={"invocation_id": reservation.attempt_id, "operation_id": operation_id,
                           "operation_state": "succeeded", "dispatch_state": "response_received",
                           "response_ref": {"sha256": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(), "sizeBytes": len(raw.encode())}})


def test_late_remote_receipt_refreshes_own_and_root_cost_without_another_charge(env, monkeypatch):
    from ouroboros import claudexor_daemon

    _terminal(env)
    _terminal(env, "root", checkpoint={"post_task_synthesis": "completed"})
    reservation = _attempt(env, provider="claudexor")
    _remote_request(env, reservation, "operation")
    calls = []

    class Gateway:
        def get_model_operation(self, operation_id, **kwargs):
            assert env.lock_depth[0] == 0
            calls.append(operation_id)
            return {"state": "cancelled", "dispatch": {"state": "unknown"}}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", Gateway)
    maintenance._reconcile_abandoned_usage(env.root)
    assert is_abandoned_settlement(_final(env, reservation))
    assert calls == ["operation", "closed"]
    _remote_receipt(env, reservation, "operation")
    maintenance._reconcile_abandoned_usage(env.root)

    assert calls == ["operation", "closed"], "the retained exact receipt requires no daemon"
    projection = usage.usage_projection(env.root)
    assert projection["accounted_usd"] == projection["confirmed_usd"] == 0.4
    assert load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 0.4
    root = load_task_result(env.root, "root")
    assert root["accounted_upper_bound_usd_with_children"] == 0.4
    assert root["root_phase_checkpoint"] == {"post_task_synthesis": "completed"}
    before = ledger_rows(env.root), _events(env)
    maintenance._reconcile_abandoned_usage(env.root)
    assert (ledger_rows(env.root), _events(env)) == before


def test_daemon_failure_stops_remote_reads_for_one_pass_but_native_work_continues(env, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _terminal(env)
    remotes = [_attempt(env, provider="claudexor") for _ in range(2)]
    for index, reservation in enumerate(remotes):
        _remote_request(env, reservation, f"op{index}")
    native = _attempt(env)
    calls = []
    fail = [True]

    class Gateway:
        def get_model_operation(self, operation_id, **kwargs):
            assert env.lock_depth[0] == 0
            calls.append(operation_id)
            if fail[0]:
                raise ClaudexorUnavailable("daemon_unreachable", "offline")
            return {"state": "running", "dispatch": {"state": "started"}}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", Gateway)
    maintenance._reconcile_abandoned_usage(env.root)
    assert calls == ["op0", "closed"]
    assert is_abandoned_settlement(_final(env, native))
    fail[0] = False
    maintenance._reconcile_abandoned_usage(env.root)
    assert calls == ["op0", "closed", "op0", "op1", "closed"]
    assert all(_final(env, reservation)["state"] == "dispatched" for reservation in remotes)


def test_reserved_and_remote_never_started_release_while_unknown_price_stays_unknown(env, monkeypatch):
    from ouroboros import claudexor_daemon

    _terminal(env)
    reserved = _attempt(env, state="reserved")
    unknown = _attempt(env, bound=None)
    remote = _attempt(env, provider="claudexor")
    _remote_request(env, remote, "not-started")
    gateway = SimpleNamespace(
        get_model_operation=lambda *_a, **_kw: {"state": "failed", "dispatch": {"state": "not_started"}},
        close=lambda: None,
    )
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: gateway)
    maintenance._reconcile_abandoned_usage(env.root)
    assert _final(env, reserved)["state"] == _final(env, remote)["state"] == "released"
    row = _final(env, unknown)
    assert is_abandoned_settlement(row) and row["reservation_upper_bound_usd"] is None
    stored = load_task_result(env.root, "child")
    assert stored["accounted_upper_bound_usd"] is None and stored["cost_final"] is False


def test_a_daemon_failure_never_starves_a_later_retained_local_receipt(env, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    _terminal(env)
    waiting = _attempt(env, provider="claudexor")
    ready = _attempt(env, provider="claudexor")
    _remote_request(env, waiting, "waiting")
    _remote_request(env, ready, "ready")
    _remote_receipt(env, ready, "ready")
    calls = []

    def unavailable(operation_id, **kwargs):
        assert env.lock_depth[0] == 0
        calls.append(operation_id)
        raise ClaudexorUnavailable("daemon_unreachable", "offline")

    gateway = SimpleNamespace(get_model_operation=unavailable, close=lambda: calls.append("closed"))
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: calls.append("attached") or gateway)
    maintenance._reconcile_abandoned_usage(env.root)
    assert _final(env, ready)["state"] == "settled"
    assert _final(env, ready)["cost_usd"] == 0.4
    assert _final(env, waiting)["state"] == "dispatched"
    assert load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 1.65
    assert calls == ["attached", "waiting", "closed"]


@pytest.mark.parametrize("failed_task", ["child", "root"])
def test_failed_projection_retries_from_settled_truth(env, monkeypatch, failed_task):
    from ouroboros import usage_store
    from supervisor import events_task_done

    _terminal(env)
    _terminal(env, "root", checkpoint={"post_task_synthesis": "completed"})
    reservation = _attempt(env, provider="claudexor")
    for task_id in ("child", "root"):
        events_task_done._refresh_terminal_task_cost(env.root, task_id)
    _remote_request(env, reservation, "receipt")
    _remote_receipt(env, reservation, "receipt")
    writer = events_task_done.write_task_result
    failed = []

    def fail_once(root, task_id, status, **fields):
        if task_id == failed_task and not failed:
            failed.append(task_id)
            raise TimeoutError("fixture result lock busy after ledger settlement")
        return writer(root, task_id, status, **fields)

    monkeypatch.setattr(events_task_done, "write_task_result", fail_once)
    maintenance._reconcile_abandoned_usage(env.root)
    assert failed == [failed_task]
    assert _final(env, reservation)["cost_usd"] == 0.4
    amount_key = "accounted_upper_bound_usd" + ("_with_children" if failed_task == "root" else "")
    assert load_task_result(env.root, failed_task)[amount_key] == 1.25
    # A new process needs no remembered refresh failure to discover this duty.
    usage_store.forget(env.root)
    ledger_before = ledger_rows(env.root)
    events_before = len(_events(env))

    maintenance._reconcile_abandoned_usage(env.root)

    assert ledger_rows(env.root) == ledger_before
    assert len(_events(env)) == events_before + 1
    for task_id, key in (("child", "accounted_upper_bound_usd"),
                         ("root", "accounted_upper_bound_usd_with_children")):
        stored = load_task_result(env.root, task_id)
        assert stored[key] == 0.4 and stored["cost_final"] is True
        assert stored["result"] == f"Preserved {task_id} answer" and stored["status"] == "cancelled"
    assert load_task_result(env.root, "root")["root_phase_checkpoint"] == {"post_task_synthesis": "completed"}
    before = ledger_before, (env.root / "logs/events.jsonl").read_bytes()
    maintenance._reconcile_abandoned_usage(env.root)
    assert (ledger_rows(env.root), (env.root / "logs/events.jsonl").read_bytes()) == before
    assert all(row.get("type") != "task_done" for row in (
        json.loads(line) for line in (env.root / "logs/events.jsonl").read_text(encoding="utf-8").splitlines()))


def test_native_late_receipt_refreshes_without_another_eligible_transition(env):
    _terminal(env)
    _terminal(env, "root")
    reservation = _attempt(env)
    maintenance._reconcile_abandoned_usage(env.root)
    assert is_abandoned_settlement(_final(env, reservation))
    assert load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 1.25

    usage.settle_attempt(reservation, {"prompt_tokens": 12, "completion_tokens": 3}, cost_usd=0.4, cost_final=True)
    ledger = ledger_rows(env.root)
    maintenance._reconcile_abandoned_usage(env.root)

    assert ledger_rows(env.root) == ledger
    child = load_task_result(env.root, "child")
    assert child["accounted_upper_bound_usd"] == 0.4 and child["cost_final"] is True
    assert (child["prompt_tokens"], child["completion_tokens"]) == (12, 3)
    assert load_task_result(env.root, "root")["accounted_upper_bound_usd_with_children"] == 0.4
    before = _events(env)
    maintenance._reconcile_abandoned_usage(env.root)
    assert _events(env) == before


def test_projection_retry_keeps_live_post_task_and_review_guards(env, monkeypatch):
    from supervisor import events_task_done

    _terminal(env, "live")
    _terminal(env, "synthesis", checkpoint={"post_task_synthesis": "running"})
    _terminal(env, "review")
    _terminal(env, "root", checkpoint={"post_task_synthesis": "running"})
    env.live.add("live")
    for task_id in ("live", "synthesis"):
        usage.settle_attempt(_attempt(env, task_id), cost_usd=0.4, cost_final=True)
    with usage.usage_scope(usage.UsageScope(review_slot_id="acceptance-slot")):
        review = _attempt(env, "review")
    usage.settle_attempt(review, cost_usd=0.4, cost_final=True)
    before = ledger_rows(env.root)
    monkeypatch.setattr(events_task_done, "_refresh_terminal_task_cost",
                        lambda *args, **kwargs: pytest.fail("ineligible owner cannot refresh"))
    maintenance._reconcile_abandoned_usage(env.root)
    assert ledger_rows(env.root) == before
    assert not _events(env)


@pytest.mark.parametrize("integrity_degraded", [False, True])
def test_projection_uses_addressed_summaries_for_distinct_owners(env, monkeypatch, integrity_degraded):
    from ouroboros import task_results, usage_store
    from ouroboros.usage_ledger import QUARANTINE_REL
    from supervisor import events_task_done

    owners = []
    for index in range(3):
        root_id, child_id = f"root-{index}", f"child-{index}"
        for task_id in (root_id, child_id):
            write_task_result(env.root, task_id, "cancelled", result="Preserved answer",
                              root_task_id=root_id, parent_task_id="" if task_id == root_id else root_id,
                              delegation_role="root" if task_id == root_id else "subagent")
            owners.append(task_id)
        for _ in range(2):
            reservation = usage.reserve_attempt(usage.AttemptRequest(
                model="test-model", provider="openai", drive_root=env.root,
                task_id=child_id, root_task_id=root_id, parent_task_id=root_id,
                reservation_usd=None if index == 1 else 1.25,
                force_unknown_reservation=index == 1, global_limit_usd=100.0,
            ))
            if index != 2:
                usage.mark_dispatched(reservation)
            if index == 0:
                usage.settle_attempt(reservation, {"prompt_tokens": 7}, cost_usd=0.4, cost_final=True)
    if integrity_degraded:
        (env.root / QUARANTINE_REL).write_text("fixture quarantine evidence\n", encoding="utf-8")
    reads, aggregations = [], []
    read_task, bucket = task_results.load_task_result, usage_store.Txn.bucket
    monkeypatch.setattr(task_results, "load_task_result",
                        lambda root, task_id, **kw: reads.append(task_id) or read_task(root, task_id, **kw))

    def aggregate(txn, scope, key=""):
        aggregations.append((scope, key))
        return bucket(txn, scope, key)

    monkeypatch.setattr(usage_store.Txn, "bucket", aggregate)
    maintenance._reconcile_abandoned_usage(env.root)
    # Recovery transactions also maintain summaries. Projection itself reads
    # only its two address scopes, never any owner/axis enumeration.
    assert all(scope in {"task", "root"} for scope, _key in aggregations[-9:])
    assert len(aggregations[-9:]) == 9
    # Recovery and projection share one result parse per owner in the pass.
    assert sorted(reads) == sorted(owners)
    for task_id in owners:
        stored = load_task_result(env.root, task_id)
        expected = events_task_done._authoritative_terminal_cost(task_id, stored, stored, {}, env.root)
        assert all(stored.get(key) == value for key, value in expected.items()), task_id
        assert stored["ledger_integrity_degraded"] is integrity_degraded
    known = load_task_result(env.root, "child-0")
    unknown = load_task_result(env.root, "child-1")
    released = load_task_result(env.root, "child-2")
    assert (known["accounted_upper_bound_usd"], known["total_rounds"], known["prompt_tokens"]) == (0.8, 2, 14)
    assert unknown["accounted_upper_bound_usd"] is None and unknown["non_final_rows"] == 2
    assert (released["accounted_upper_bound_usd"], released["total_rounds"]) == (0.0, 0)
    assert load_task_result(env.root, "root-1")["accounted_upper_bound_usd_with_children"] is None


def test_addressed_projection_failure_defers_without_fake_zero(env, monkeypatch):
    _terminal(env)
    _terminal(env, "root")
    reservation = _attempt(env)
    from supervisor import events_task_done
    probe = events_task_done._terminal_cost_probe
    calls = []

    def unavailable(root, task_id, current):
        calls.append(task_id)
        raise TimeoutError("fixture ledger projection unavailable")

    monkeypatch.setattr(events_task_done, "_terminal_cost_probe", unavailable)
    maintenance._reconcile_abandoned_usage(env.root)
    assert is_abandoned_settlement(_final(env, reservation))
    assert calls == ["child", "root"] and not _events(env)
    assert "accounted_upper_bound_usd" not in load_task_result(env.root, "child")
    before = ledger_rows(env.root)
    monkeypatch.setattr(events_task_done, "_terminal_cost_probe", probe)
    maintenance._reconcile_abandoned_usage(env.root)
    assert ledger_rows(env.root) == before
    assert load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 1.25
    assert load_task_result(env.root, "root")["accounted_upper_bound_usd_with_children"] == 1.25


@pytest.mark.parametrize("task_id", ["child", "root"])
def test_prefetched_projection_preserves_a_different_canonical_budget_root(env, task_id):
    canonical = env.root / "canonical-budget"
    _terminal(env, task_id)
    write_task_result(env.root, task_id, "cancelled", budget_drive_root=str(canonical))
    usage.settle_attempt(_attempt(env, task_id), cost_usd=0.1, cost_final=True)
    actual = usage.reserve_attempt(usage.AttemptRequest(
        model="test-model", provider="openai", drive_root=canonical,
        task_id=task_id, root_task_id="root", parent_task_id="" if task_id == "root" else "root",
        reservation_usd=1.25, global_limit_usd=100.0,
    ))
    usage.mark_dispatched(actual)
    usage.settle_attempt(actual, {"prompt_tokens": 21}, cost_usd=0.7, cost_final=True)
    before = [ledger_rows(root) for root in (env.root, canonical)]

    maintenance._reconcile_abandoned_usage(env.root)

    stored = load_task_result(env.root, task_id)
    assert stored["budget_drive_root"] == str(canonical)
    assert (stored["accounted_upper_bound_usd"], stored["prompt_tokens"]) == (0.7, 21)
    if task_id == "root":
        assert stored["accounted_upper_bound_usd_with_children"] == 0.7
    assert [ledger_rows(root) for root in (env.root, canonical)] == before


def test_a_wave_alone_keeps_a_direct_review_with_the_generic_closure(env):
    """Advisory and deep self-review send under a review wave but no reviewer slot (#1544):
    their abandoned attempts still close here, while a review-substrate slot stays with its
    own custody."""
    _terminal(env, "direct")
    _terminal(env, "substrate")
    with usage.usage_scope(usage.UsageScope(review_wave_id="wave-advisory")):
        direct = _attempt(env, "direct")
    with usage.usage_scope(usage.UsageScope(review_wave_id="commit:c1", review_slot_id="triad-slot")):
        substrate = _attempt(env, "substrate")

    maintenance._reconcile_abandoned_usage(env.root)

    assert is_abandoned_settlement(_final(env, direct))
    assert _final(env, substrate)["state"] == "dispatched"
