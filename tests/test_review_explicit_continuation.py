"""An explicit continuation of a stopped session reviewer, through the public ``review_change`` handler.

A two-seat wave (an API seat and a delegated seat) settles with the delegated run stopped by
the engine with continuable work. Nothing continues it on its own; the result offers the seat.
``review_reference`` + ``continue_seats`` then records one choice in the existing ledger record
and sends ONE successor run of the same seat in the same wave, built from the seat's original
start body. Only the Claudexor engine and the API seat's model are scripted; the wave, attempt
rows, ledger, custody, projections and collection are real.
"""
from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from ouroboros import delegate_custody, review_continuation, review_ledger
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.review_ledger import ledger_root, load_record
from ouroboros.review_state import load_state
from tests._review_session_route_shared import FakeGateway, _owned_gateway_uses_each_test_transport  # noqa: F401
from tests._review_session_route_shared import _terminal_detail, fake_route as _fake_route
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_receptive_commit_review import _git, _join, _parent, _produce
from tests.test_receptive_commit_review import author as _author

author, fake_route = _author, _fake_route
SESSION_ANSWER = "CONTINUED-SEAT-ANSWER"
RESUMABLE = {"cause": "pool_exhausted", "resetsAt": None, "limitWindow": None, "limitEvidence": None,
             "carriers": ["native", "packet"], "limitCode": None, "session": None,
             "workspace": {"kind": "none", "root": None}}


class _Engine:
    """One run per Idempotency-Key; a scripted fate per run: ``stop`` (failed with continuable work and
    partial output) or ``answer``; continuation starts may be refused or lose their reply."""

    def __init__(self):
        self.keys, self.posts, self.fates = {}, [], {"run-1": "stop"}
        self.refuse_next, self.lose_next = None, False
        self.store = ""  # "": an engine whose receipts name no store; else the store that admits every start

    def start(self, gateway, request, idempotency_key):
        gateway.start_requests.append(dict(request))
        self.posts.append((dict(request), str(idempotency_key)))
        if "continueFrom" in request and self.refuse_next is not None:
            refusal, self.refuse_next = self.refuse_next, None
            raise refusal
        run_id = self.keys.setdefault(str(idempotency_key), f"run-{len(self.keys) + 1}")
        self.fates.setdefault(run_id, "answer")
        if "continueFrom" in request and self.lose_next:
            self.lose_next = False
            raise ClaudexorUnavailable("daemon_unreachable", "accepted, reply lost")
        return {"runId": run_id, "runDir": "/tmp/fake-run", **({"storeId": self.store} if self.store else {})}

    def detail(self, gateway, run_id):
        from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

        gateway.run_gets.append(run_id)
        if self.fates.get(run_id) == "stop":
            result = _terminal_detail("PARTIAL: read three files, two left", state="failed", conformance="")
            result["summary"].update(spendUsd=0.25, failure={"code": "credential_pool_exhausted",
                                                             "safeMessage": "Account limit reached during review"},
                                     resumable=copy.deepcopy(RESUMABLE), outcomeFacts={"work_state": "unfinished"})
            return result
        answer = {"change": [], "change_clean": True, "coupling": [
            {"item": item, "verdict": "PASS", "severity": "advisory", "reason": SESSION_ANSWER}
            for item in sorted(SCOPE_REQUIRED_ITEMS)]}
        result = _terminal_detail(json.dumps(answer), conformance="passed")
        result["summary"].update(spendUsd=0.5, continuity=[{"carrier": "native", "cause": "pool_exhausted"}])
        return result


@pytest.fixture
def stopped_wave(author, fake_route, monkeypatch):
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.tools import review
    from ouroboros.triad_review import REVIEW_TWO_PART_OBJECT_CONTRACT

    a = author
    set_review_pool(monkeypatch, pool_roster(pool_seat("first", "test/m1", delivery="native"),
                                             pool_seat("second", "fake-review=fake-small", kind="agent_session")))
    routes = [ReviewRouteKind.API_CHAT, ReviewRouteKind.AGENT_SESSION]
    plan = {"models": ["m1", "fake-review=fake-small"], "routes": routes, "efforts": ["high"] * 2,
            "slot_ids": ["first", "second"], "use_local": [False] * 2, "subagent_ids": ["", ""],
            "session_targets": ["", "fake-review=fake-small"], "session_profiles": ["", ""],
            "retrieves": [True, True], "parts": [("change", "coupling")] * 2,
            "session_tasks": ["Review this candidate."] * 2,
            "session_policies": [{"output_contract": REVIEW_TWO_PART_OBJECT_CONTRACT}] * 2,
            "brief_shas": ["brief-sha"] * 2}

    def prepared(ctx, commit_message, review_rebuttal="", repo_dir=None, goal="", scope="", subject=None):
        root = subject.review_root if subject is not None else ctx.repo_dir
        return ({"prompt": "Review this candidate.", "models": list(plan["models"]), "stable_prefix_len": 0,
                 "routes": list(routes), "session_task": "", "target_repo": root,
                 "row_plan": json.loads(json.dumps(plan, default=str)) | {"routes": list(routes)},
                 "layer": "core", "brief_texts": {"brief-sha": "Review this candidate."}}, None, False)
    monkeypatch.setattr(review, "_prepare_unified_review", prepared)
    engine = _Engine()
    monkeypatch.setattr(FakeGateway, "start_run", lambda self, request, *, idempotency_key="": engine.start(
        self, request, idempotency_key))
    monkeypatch.setattr(FakeGateway, "get_run", lambda self, run_id, **_kw: engine.detail(self, run_id))
    monkeypatch.setattr(FakeGateway, "agent_capabilities", lambda self: {
        "harnesses": [dict(FakeGateway.catalog_entry)], "runControlKeys": ["continueFrom"]})
    _git(a.repo, "add", "change.py")
    a.engine = engine
    a.reviewer.release()
    return a


def _call(a, **kwargs):
    from ouroboros.tools.review_change import _handle_review_change

    outcome = {}

    def run():
        with _parent(a):
            outcome["result"] = _handle_review_change(a.ctx, **kwargs)
    _join(_produce(a, run))
    text = outcome["result"]
    return json.JSONDecoder().raw_decode(text, 0 if text.startswith("{") else text.index("\n{") + 1)[0]


def _wave(a):
    return _call(a, root="system_repo", subject="index", goal="Raise the value.", scope="change.py only")


def _seat(a, record_id, seat_id="second"):
    return {row["seat_id"]: row for row in load_record(ledger_root(a.ctx), record_id)["rows"]}[seat_id]


def _session_posts(a):
    return [(body, key) for body, key in a.engine.posts]


def test_a_chosen_stopped_seat_continues_once_in_the_same_wave_from_its_original_request(stopped_wave):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    assert first["state"] == "settled" and len(a.engine.posts) == 1  # nothing continued the stop on its own
    offer = first["continuable_seats"]
    stopped = _seat(a, record_id)
    assert [item["seat_id"] for item in offer] == ["second"] and offer[0]["operation_id"] == stopped["operation_id"]
    assert offer[0]["run_id"] == "run-1" and offer[0]["engine_stop"]["cause"] == "pool_exhausted"
    revision = load_record(ledger_root(a.ctx), record_id)["revision"]

    continued = _call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    assert continued["state"] == "settled" and continued["reused"] is False, continued
    (original, _key0), (successor, key) = a.engine.posts
    assert successor["continueFrom"] == "run-1"
    changed = {name for name in set(original) | set(successor) if original.get(name) != successor.get(name)}
    assert changed <= {"prompt", "maxSeconds", "continueFrom"}, changed  # every other control byte for byte
    assert "stopped before it finished" in successor["prompt"] and "Account limit reached" in successor["prompt"]
    seat = _seat(a, record_id)
    decision = seat["continuations"][0]
    assert key == decision["invocation_id"] and seat["operation_id"] == decision["operation_id"]
    assert seat["status"] == "responded" and seat["superseded_operation_ids"] == [stopped["operation_id"]]
    assert seat["prior_operations"][0]["operation_id"] == stopped["operation_id"]
    assert seat["prior_operations"][0]["failure_code"] == "credential_pool_exhausted"
    assert seat["outcome"]["usage_ref"]["continuity"] == [{"carrier": "native", "cause": "pool_exhausted"}]
    record = load_record(ledger_root(a.ctx), record_id)
    assert record["revision"] > revision and record["cost"] == review_ledger.record_cost(record["rows"])
    assert any(ref.get("role") == "failure" for ref in seat.get("historical_source_refs") or [])
    attempt_row = next(row for row in load_state(a.data).attempts[-1].triad_raw_results if row["slot_id"] == "second")
    assert attempt_row["operation_id"] == decision["operation_id"]

    # The same choice rejoins its one successor: no send, no new revision.
    again = _call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": stopped["operation_id"]}])
    assert again["continuation"]["continued"][0]["rejoined"] is True
    assert len(a.engine.posts) == 2 and load_record(ledger_root(a.ctx), record_id)["revision"] == record["revision"]
    refused = _call(a, review_reference=reference, continue_seats=[
        {"seat_id": "first", "operation_id": _seat(a, record_id, "first")["operation_id"]}])
    assert refused["status"] == "refused" and refused["reason"] == "not_a_session_route" and len(a.engine.posts) == 2
    answered = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": seat["operation_id"]}])
    assert answered["reason"] == "not_continuable" and len(a.engine.posts) == 2  # its usable verdict stands


def test_a_definitely_refused_successor_is_continued_only_by_a_new_choice_naming_it(stopped_wave):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    a.engine.refuse_next = ClaudexorUnavailable("predecessor_unknown", "no run owned by this daemon", status_code=404)
    refused = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    seat = _seat(a, record_id)
    successor = seat["continuations"][0]["operation_id"]
    token = seat["continuations"][0]["invocation_id"]
    assert delegate_custody.invocation_record(a.data, token)["state"] == "failed_definite"
    offer = refused["continuable_seats"]
    assert offer == [{**offer[0], "seat_id": "second", "operation_id": successor, "run_id": "", "continues_run": "run-1"}]
    rejoined = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    assert rejoined["continuation"]["continued"][0]["rejoined"] is True and len(a.engine.posts) == 2

    answered = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": successor}])
    assert answered["state"] == "settled" and _seat(a, record_id)["status"] == "responded"
    third, key = a.engine.posts[-1]
    assert third["continueFrom"] == "run-1" and key != token  # its retained target run; a new decision
    assert len(_seat(a, record_id)["prior_operations"]) == 2


def test_an_unknown_successor_start_replays_the_same_key_and_bytes_never_a_new_generation(stopped_wave):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    a.engine.lose_next = True  # the engine accepted the successor; its reply was lost
    answered = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    assert answered["state"] == "settled" and _seat(a, record_id)["status"] == "responded"
    (_first, _k), (lost, lost_key), (replayed, replay_key) = a.engine.posts
    assert replay_key == lost_key and replayed == lost and a.engine.keys[lost_key] == "run-2"
    assert len(_seat(a, record_id)["continuations"]) == 1  # one decision, one run; no new generation


def test_an_accepted_successor_whose_reply_was_lost_is_rejoined_with_the_same_key_and_bytes(stopped_wave):
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    root, attempt, record = resolve_reference(a.ctx, reference)
    continue_or_repair(a.ctx, root, attempt, record, [{"seat_id": "second", "operation_id": named}])
    decision = _seat(a, record_id)["continuations"][0]
    body = delegate_custody.invocation_record(a.data, decision["invocation_id"])["request"]
    FakeGateway().start_run(body, idempotency_key=decision["invocation_id"])  # accepted; its process ended unbound
    delegate_custody._CUSTODY.clear()
    collected = _call(a, review_reference=reference)
    assert collected["state"] == "settled" and _seat(a, record_id)["status"] == "responded"
    (_first, _k), (lost, lost_key), (replayed, replay_key) = a.engine.posts
    assert replay_key == lost_key == decision["invocation_id"] and replayed == lost == body
    assert a.engine.keys[replay_key] == "run-2" and len(_seat(a, record_id)["continuations"]) == 1


@pytest.mark.parametrize("admitted,serving", [("store-a", "store-a"), ("store-a", "store-b"), ("", "store-b")])
def test_a_successor_is_sent_only_into_the_store_that_admitted_its_stopped_run(stopped_wave, monkeypatch,
                                                                               admitted, serving):
    """#2649: the successor's START names the store that admitted run-1 (its continueFrom target), and its first send
    goes only there. A predecessor bound to no store keeps the earlier replay rule, explicitly."""
    a = stopped_wave
    a.engine.store = admitted
    monkeypatch.setattr(FakeGateway, "store_identity", admitted, raising=False)
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    assert delegate_custody.replay(a.data)["run-1"].daemon_store_id == admitted
    a.engine.store = serving  # the engine now serves another store, or the same one again
    monkeypatch.setattr(FakeGateway, "store_identity", serving, raising=False)
    _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    token = _seat(a, record_id)["continuations"][0]["invocation_id"]
    assert delegate_custody.invocation_record(a.data, token)["daemon_store_id"] == admitted
    sent = [key for body, key in a.engine.posts if "continueFrom" in body]
    if admitted and admitted != serving:  # a store that never admitted run-1 is never sent its successor
        assert sent == [] and delegate_custody.invocation_record(a.data, token)["state"] == "pending"
        seat = _seat(a, record_id)
        assert seat.get("failure_code") == "invocation_store_unproven", {k: seat.get(k) for k in (
            "status", "operation_state", "failure_code", "reported_cause", "pending_invocation_id")}
        return
    assert sent == [token] and _seat(a, record_id)["status"] == "responded"
    delegate_custody._CUSTODY.clear()
    successor = a.engine.keys[token]
    assert delegate_custody.replay(a.data)[successor].daemon_store_id == serving  # its own receipt's store


def test_a_choice_recorded_before_a_crash_is_sent_only_by_an_explicit_collection(stopped_wave, monkeypatch):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    real = review_continuation._start_requested
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: (_ for _ in ()).throw(
        OSError("process ended after the choice")))
    unsent = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    assert unsent["status"] == "decided_unsent" and "OSError" in unsent["message"], unsent  # recorded, not refused
    seat = _seat(a, record_id)
    token = seat["continuations"][0]["invocation_id"]
    assert seat["status"] == "pending" and delegate_custody.invocation_record(a.data, token) is None
    assert len(a.engine.posts) == 1  # decided, never sent
    from ouroboros import review_pause
    stale = [{"slot_id": "second", "operation_id": named, "status": "error", "operation_state": "settled",
              "error": "a stale predecessor read", "parts": ["change", "coupling"]}]
    coordinates = {"record_id": record_id, "drive_root": str(a.data), "canonical_root": str(ledger_root(a.ctx)),
                   "retry_key": load_state(a.data).attempts[-1].review_retry_key}
    assert review_pause._settle_left_record(coordinates, stale, final=False)["state"] == "pending"
    monkeypatch.setattr(review_continuation, "_start_requested", real)
    collected = _call(a, review_reference=reference)
    assert collected["state"] == "settled" and collected["continuation"]["repaired"] == ["second"]
    assert len(a.engine.posts) == 2 and a.engine.posts[1][1] == token


def _waiting_choice(a, monkeypatch, reference, choice):
    """The choice runs as its author's call and is seen to wait on a chosen operation this process still publishes."""
    import threading

    from ouroboros import review_custody
    from ouroboros.tools.review_change import _handle_review_change
    from tests.test_receptive_commit_review import _parent

    entry = review_custody.ActiveReviewAttempt(key="still-publishing", operation_id=choice[0]["operation_id"])
    with review_custody._ACTIVE_LOCK:
        review_custody._ACTIVE[entry.key] = entry
    waiting, wait, outcome = threading.Event(), review_continuation._await_publication, {}
    monkeypatch.setattr(review_continuation, "_await_publication", lambda *args: (waiting.set(), wait(*args))[1])

    def run():
        with _parent(a):
            outcome["text"] = _handle_review_change(a.ctx, review_reference=reference, continue_seats=choice)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert waiting.wait(30), "the choice never reached the publication wait"

    def finish():
        thread.join(30)
        assert not thread.is_alive(), "the waiting choice did not end"
        with review_custody._ACTIVE_LOCK:
            review_custody._ACTIVE.pop(entry.key, None)
        return json.loads(outcome["text"])
    return finish


@pytest.mark.parametrize("ending", ["pause", "window"])
def test_a_choice_waiting_on_a_publishing_operation_ends_at_pause_or_its_window_and_decides_nothing(
        stopped_wave, monkeypatch, ending):
    from datetime import timedelta

    from ouroboros import budget_pause, deadline_utils
    from ouroboros.deadline_utils import parse_deadline_ts
    from tests.test_receptive_commit_review import TASK

    a = stopped_wave
    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "600")  # the stopped run had a window to keep
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    choice = [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    before = load_record(ledger_root(a.ctx), record_id)
    finish = _waiting_choice(a, monkeypatch, reference, choice)
    if ending == "pause":
        budget_pause.begin_dispatch_fence(TASK)
    else:  # the stopped run's own window, which the choice would continue, ends while it waits
        end = parse_deadline_ts(review_continuation._window_end(
            delegate_custody.custody_root(a.ctx), review_continuation._invocation(
                delegate_custody.custody_root(a.ctx), "second", choice[0]["operation_id"], _seat(a, record_id))))
        monkeypatch.setattr(deadline_utils, "utc_now", lambda: end + timedelta(seconds=1))
    try:
        refused = finish()
    finally:
        budget_pause.end_dispatch_fence(TASK)
    assert refused["status"] == "refused", refused
    assert refused["reason"] == ("stopped_or_paused" if ending == "pause" else "window_spent"), refused
    assert load_record(ledger_root(a.ctx), record_id) == before and len(a.engine.posts) == 1  # decided nothing
    if ending == "pause":  # after Resume, and once the worker has published, the same choice continues
        continued = _call(a, review_reference=reference, continue_seats=choice)
        assert continued["state"] == "settled" and len(a.engine.posts) == 2, continued


def test_a_choice_at_a_reached_budget_fence_decides_and_sends_nothing_and_with_room_it_continues(stopped_wave, monkeypatch):
    """No Pause fence anywhere; the task's own known spend has reached its fence. The successor would be new
    generation, so the known-spend admission every review wave rides refuses it at $0 before anything is decided;
    collecting the paid record stays free, and under a raised fence the same choice continues."""
    import threading

    from ouroboros import usage_accounting as ua
    from ouroboros.tools.review_change import _handle_review_change
    from tests.test_receptive_commit_review import TASK, _parent

    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    choice = [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 0.5)  # the price catalog only; the rule is the product's

    def scope(fence):
        return ua.UsageScope(drive_root=a.data, task_id=TASK, root_task_id=TASK, root_limit_usd=fence)
    with ua.usage_scope(scope(5.0)):  # a settled row of this task brings its known spend to the $5 fence
        row = ua.reserve_attempt(ua.AttemptRequest(model="test/m1", provider="test", source="main"))
        ua.mark_dispatched(row)
        ua.settle_attempt(row, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=5.0, cost_final=True)

    def call(fence, **kwargs):  # the author's call under the task's usage scope, as the loop binds it
        out = {}

        def run():
            with _parent(a), ua.usage_scope(scope(fence)):
                out["text"] = _handle_review_change(a.ctx, review_reference=reference, **kwargs)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(60)
        text = out["text"]
        return json.JSONDecoder().raw_decode(text, 0 if text.startswith("{") else text.index("\n{") + 1)[0]
    before = load_record(ledger_root(a.ctx), record_id)
    refused = call(5.0, continue_seats=choice)
    assert refused["status"] == "refused" and refused["reason"] == "budget_insufficient", refused
    assert "per-task budget fence $5.000000" in refused["message"] and len(a.engine.posts) == 1
    assert load_record(ledger_root(a.ctx), record_id) == before  # nothing decided
    assert call(5.0)["state"] == "settled" and len(a.engine.posts) == 1  # collecting the paid record stays free
    continued = call(50.0, continue_seats=choice)
    assert continued["state"] == "settled" and len(a.engine.posts) == 2, continued


def test_an_author_that_ended_never_sends_a_recorded_choice(stopped_wave, monkeypatch):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: (_ for _ in ()).throw(OSError("ended")))
    assert _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])[
        "status"] == "decided_unsent"
    from ouroboros import review_pause
    coordinates = {"record_id": record_id, "drive_root": str(a.data), "canonical_root": str(ledger_root(a.ctx)),
                   "retry_key": load_state(a.data).attempts[-1].review_retry_key}
    settled = review_pause._settle_left_record(coordinates, [], final=True)
    seat = {row["seat_id"]: row for row in settled["rows"]}["second"]
    assert settled["state"] == "settled" and seat["operation_state"] == "not_dispatched"
    assert len(a.engine.posts) == 1


@pytest.mark.parametrize("elapsed", [5, 300])
def test_a_lost_original_start_keeps_its_window_when_recovered(stopped_wave, monkeypatch, elapsed):
    """The real public review/continue handlers: an accepted original POST loses its reply. Recovery may
    collect it after expiry, but a continuation gets only the original time remaining, never a fresh cap."""
    from datetime import datetime, timedelta, timezone

    from ouroboros import deadline_utils, review_custody, utils

    a = stopped_wave
    a.engine.store = "store-a"
    original = datetime.now(timezone.utc)
    now = {"value": original}
    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "30")
    monkeypatch.setattr(delegate_custody, "utc_now_iso", lambda: now["value"].isoformat())
    monkeypatch.setattr(utils, "utc_now_iso", lambda: now["value"].isoformat())
    monkeypatch.setattr(review_custody, "utc_now", lambda: now["value"])
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: now["value"])
    start = a.engine.start

    def lost_initial(gateway, request, key):
        handle = start(gateway, request, key)
        if len(a.engine.posts) == 1:
            now["value"] = original + timedelta(seconds=elapsed)
            raise ClaudexorUnavailable("daemon_unreachable", "accepted original review; reply lost")
        return handle

    def lookup(gateway, request, *, idempotency_key):
        if idempotency_key not in a.engine.keys:  # an explicitly chosen successor has not been submitted yet
            return {"status": "not_found", "storeId": "store-a"}
        return {"status": "found", "storeId": "store-a", "handle": {
            "jobId": "job-original", "runId": a.engine.keys[idempotency_key], "state": "failed"}}

    monkeypatch.setattr(a.engine, "start", lost_initial)
    monkeypatch.setattr(FakeGateway, "store_identity", "store-a", raising=False)
    monkeypatch.setattr(FakeGateway, "operations", lambda self: [{"method": "POST", "path": "/v2/runs/lookup"}], raising=False)
    monkeypatch.setattr(FakeGateway, "lookup_run", lookup, raising=False)
    first = _wave(a)
    if not first.get("continuable_seats"):
        first = _call(a, review_reference=first["review_reference"])
    assert len(a.engine.posts) == 1 and a.engine.posts[0][0]["maxSeconds"] == 30
    held = delegate_custody.replay(a.data)["run-1"]
    assert held.started_ts == now["value"].isoformat(), "publication time is not rewritten as the old request time"
    assert held.window_origin == original.isoformat() and held.max_seconds == 30
    choice = [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    continued = _call(a, review_reference=first["review_reference"], continue_seats=choice)
    if elapsed > 30:
        assert continued.get("reason") == "window_spent" and len(a.engine.posts) == 1, continued
    else:
        assert continued["state"] == "settled" and len(a.engine.posts) == 2, continued
        assert 0 < a.engine.posts[1][0]["maxSeconds"] <= 30 - elapsed


def test_a_spent_original_window_refuses_at_zero_and_writes_nothing(stopped_wave, monkeypatch):
    a = stopped_wave
    first = _wave(a)
    record_id = first["record_id"]
    before = load_record(ledger_root(a.ctx), record_id)
    monkeypatch.setattr(review_continuation, "_remaining_seconds", lambda *_a: 0)
    refused = _call(a, review_reference=first["review_reference"], continue_seats=[
        {"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}])
    assert refused["status"] == "refused" and refused["reason"] == "window_spent"
    assert load_record(ledger_root(a.ctx), record_id) == before and len(a.engine.posts) == 1


def test_stale_writers_keep_a_reopened_seat_and_its_successor(stopped_wave, monkeypatch):
    """The ledger reducers and the attempt-row writers take only the reopened seat's own successor."""
    from ouroboros.review_state_records import _merge_attempt
    from ouroboros.tools.review_change_custody import settle_collected_attempt

    a = stopped_wave
    first = _wave(a)
    record_id = first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    monkeypatch.setattr(review_continuation, "_start_requested", lambda *a_, **k: True)  # recorded, not sent
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference
    root, attempt, record = resolve_reference(a.ctx, first["review_reference"])
    record, attempt, _facts = continue_or_repair(a.ctx, root, attempt, record, [{"seat_id": "second", "operation_id": named}])
    successor = _seat(a, record_id)["operation_id"]
    stale = {"slot_id": "second", "operation_id": named, "status": "ok", "raw_text": "late old answer"}
    assert review_ledger.settle_from_collected(ledger_root(a.ctx), record_id, [stale])["state"] == "pending"
    assert review_ledger.settle_from_collected(ledger_root(a.ctx), record_id, [{**stale, "operation_id": ""}])["state"] == "pending"
    kept = review_ledger.settle_pending_payload(record, {**record, "rows": [
        {**row, "operation_id": ""} if row["seat_id"] == "second" else row for row in record["rows"]]})
    assert {row["seat_id"]: row for row in kept["rows"]}["second"]["operation_id"] == successor
    old_row = {**next(r for r in attempt.triad_raw_results if r["slot_id"] == "second"), "operation_id": named}
    merged = _merge_attempt(attempt, dataclasses.replace(attempt, triad_raw_results=[old_row]))
    assert next(r for r in merged.triad_raw_results if r["slot_id"] == "second")["operation_id"] == successor
    settle_collected_attempt(root, attempt, [{"slot_id": "second", "operation_id": "", "status": "ok"}], record)
    row = next(r for r in load_state(root).attempts[-1].triad_raw_results if r["slot_id"] == "second")
    assert row["operation_id"] == successor and row["operation_state"] == "in_flight"


def _commit_call(a, **kwargs):
    from tests.test_receptive_commit_review import _commit

    outcome = {}
    _join(_commit(a, outcome, **kwargs))
    return outcome["result"]


def _commit_json(text):
    return json.JSONDecoder().raw_decode(text, text.index("{\"review_reference\""))[0]


@pytest.mark.parametrize("changed", [False, True])
def test_a_commit_continuation_continues_the_old_frozen_subject_and_never_commits_by_itself(stopped_wave, changed):
    """An explicit continue_seats continues the seat of the OLD frozen subject: with the reviewed bytes staged it runs in
    the paid gate itself; with the author's later bytes staged it runs read-only (#2797), the files and index left as
    they are. Either way it is one new generation of the same wave, and nothing is committed by it."""
    a = stopped_wave
    head = _git(a.repo, "rev-parse", "HEAD")
    handback = _commit_json(_commit_call(a))
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    assert [item["seat_id"] for item in offer] == ["second"] and len(a.engine.posts) == 1
    if changed:
        (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")  # other bytes than the paid review saw
    _git(a.repo, "add", "change.py")
    index = _git(a.repo, "write-tree")
    continued = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    record_id = reference["review_record_id"]
    assert len(a.engine.posts) == 2 and a.engine.posts[1][0]["continueFrom"] == "run-1", continued
    assert _seat(a, record_id)["status"] == "responded" and len(_seat(a, record_id)["continuations"]) == 1
    assert "no commit, tag or push occurred" in continued or "nothing was staged, committed" in continued, continued
    assert _git(a.repo, "rev-parse", "HEAD") == head
    if changed:
        assert "read-only" in continued and "only the continuation you chose was sent" in continued, continued
        assert _git(a.repo, "write-tree") == index, "the author's staged bytes are untouched"


def test_two_identical_concurrent_choices_record_one_body_clock_and_token(stopped_wave, monkeypatch):
    from ouroboros.delegate_custody import custody_root
    from ouroboros.tools.review_collection import attempt_roster, resolve_reference

    a = stopped_wave
    first = _wave(a)
    record_id = first["record_id"]
    seats = [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    root, attempt, record = resolve_reference(a.ctx, first["review_reference"])
    clocks, raced = iter([111, 222]), []  # the late choice's read, then the landed one's (its START reads its instant)
    monkeypatch.setattr(review_continuation, "_remaining_seconds", lambda *_a: next(clocks))
    real_record = review_continuation._record_choice

    def racing(drive, current, choices):  # the other caller's identical choice lands between prepare and record
        if not raced:
            raced.append("first")
            review_continuation.arm(a.ctx, record, attempt_roster(root, attempt), seats,
                                    drive=ledger_root(a.ctx), custody=custody_root(a.ctx))
        return real_record(drive, current, choices)
    monkeypatch.setattr(review_continuation, "_record_choice", racing)
    late = review_continuation.arm(a.ctx, record, attempt_roster(root, attempt), seats,
                                   drive=ledger_root(a.ctx), custody=custody_root(a.ctx))
    seat = _seat(a, record_id)
    assert len(seat["continuations"]) == 1 and seat["continuations"][0]["max_seconds"] == 222
    assert late["continued"][0]["max_seconds"] == 222  # the first recorded clock, never its own 111
    body = delegate_custody.invocation_record(a.data, seat["continuations"][0]["invocation_id"])["request"]
    assert body["maxSeconds"] == 222 and len(a.engine.posts) == 1  # projected once; the collector sends it


def test_a_legacy_automatic_chain_continues_from_its_head_with_the_original_body(stopped_wave):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    stopped = _seat(a, record_id)
    anchor_token = next(row for row in load_state(a.data).attempts[-1].triad_raw_results
                        if row["slot_id"] == "second")["recovery_binding"]["pending_invocation_id"]
    anchor = delegate_custody.invocation_record(a.data, anchor_token)
    entry = delegate_custody.lookup(a.data, anchor["task_id"], "run-1")[1]
    legacy = delegate_custody.idempotency_key("review_continue", entry.invocation_id, "run-1")
    assert delegate_custody.record_start_requested(a.data, run_id="", task_id=anchor["task_id"], invocation_id=legacy,
        idempotency_key=legacy, operation_id=stopped["operation_id"], request={"continueFrom": "run-1"},
        continuation_of="run-1", surface="multi_model_review", slot_id="second", route=anchor["route"],
        root_task_id=anchor["root_task_id"], max_seconds=0)
    assert delegate_custody.record_started(a.data, delegate_custody.RunCustody(
        run_id="run-head", task_id=anchor["task_id"], invocation_id=legacy, idempotency_key=legacy, continuation_of="run-1",
        route_id=entry.route_id, root_task_id=anchor["root_task_id"], ledger_root=str(a.data)),
        shape={"surface": "multi_model_review", "slot_id": "second"})
    a.engine.fates["run-head"] = "stop"

    def legacy_head(payload):  # what the older transport reported: the chain's last run stopped
        for row in payload["rows"]:
            if row["seat_id"] == "second":
                row["outcome"] = {**row["outcome"], "usage_ref": {**row["outcome"]["usage_ref"], "delegated_run_id": "run-head"}}
        return payload
    review_ledger.revise_record(ledger_root(a.ctx), record_id, legacy_head)
    continued = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": stopped["operation_id"]}])
    assert continued["state"] == "settled"
    (original, _k), (successor, _key) = a.engine.posts
    assert successor["continueFrom"] == "run-head" and successor["prompt"] != original["prompt"]
    assert {k: v for k, v in successor.items() if k not in ("prompt", "maxSeconds", "continueFrom")} == {
        k: v for k, v in original.items() if k not in ("prompt", "maxSeconds")}  # the original body, not the bare one


def test_a_crash_before_the_ledger_write_decides_nothing_and_a_new_call_decides_once(stopped_wave, monkeypatch):
    a = stopped_wave
    first = _wave(a)
    reference, record_id = first["review_reference"], first["record_id"]
    named = first["continuable_seats"][0]["operation_id"]
    before = load_record(ledger_root(a.ctx), record_id)
    real = review_continuation._record_choice
    monkeypatch.setattr(review_continuation, "_record_choice", lambda *a_, **k: (_ for _ in ()).throw(OSError("ended")))
    with pytest.raises(OSError), _parent(a):
        from ouroboros.tools.review_change import _handle_review_change
        _handle_review_change(a.ctx, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    assert load_record(ledger_root(a.ctx), record_id) == before and len(a.engine.posts) == 1  # only sources were retained
    monkeypatch.setattr(review_continuation, "_record_choice", real)
    continued = _call(a, review_reference=reference, continue_seats=[{"seat_id": "second", "operation_id": named}])
    assert continued["state"] == "settled" and len(_seat(a, record_id)["continuations"]) == 1 and len(a.engine.posts) == 2


class _AnyRoute(dict):
    def get(self, key, default=None):  # every route declares the same text budget
        return dict(self["*"])


def _released_range_wave(a, monkeypatch, budget=1200):
    """A base..head wave read in its isolated checkout, with an engine text budget small enough that the seat's START
    pages its long fields into that checkout's ignored ``.review-drive``; its exit releases the checkout."""
    import subprocess

    monkeypatch.setattr(FakeGateway, "ask_input_limits", lambda self: _AnyRoute({"*": {"prompt_budget": budget}}),
                        raising=False)
    _git(a.repo, "commit", "-q", "-m", "candidate")
    base, head = (subprocess.run(["git", "rev-parse", rev], cwd=a.repo, capture_output=True, text=True).stdout.strip()
                  for rev in ("HEAD~1", "HEAD"))
    first = _call(a, root="system_repo", subject="base..head", base=base, head=head, goal="Raise the value.",
                  scope="change.py only")
    return first, a.engine.posts[0][0]


def _drive_files(text):
    import re

    from ouroboros.tools.review_subject import _DRIVE_POINTER
    return sorted({f".review-drive/{task}/{digest}.txt" for task, digest in re.findall(_DRIVE_POINTER, text)})


def test_a_released_read_only_checkout_is_materialized_again_from_its_record_before_the_successor_start(
        stopped_wave, monkeypatch):
    """#2796: the read-only seat's START names its wave's isolated checkout, released at the wave's end. Its explicit
    continuation materializes the recorded parent/tree at that path, with every retained ``.review-drive`` source the
    original request names (the paged turn texts), before the successor START is POSTed; today's index never enters."""
    import hashlib
    import pathlib

    a = stopped_wave
    first, original = _released_range_wave(a, monkeypatch)
    root = pathlib.Path(original["scope"]["root"])
    subject = load_record(ledger_root(a.ctx), first["record_id"])["subject"]
    named = _drive_files(json.dumps(original))
    assert named and not root.exists() and first["continuable_seats"], (named, first)
    (a.repo / "change.py").write_text("value = 99\n", encoding="utf-8")  # the author's live edits never become the subject
    _git(a.repo, "add", "change.py")
    seen, real = [], a.engine.start

    def start(gateway, request, key):
        here = pathlib.Path(request["scope"]["root"])
        seen.append({"root": str(here), "tree": _git(here, "write-tree") if here.is_dir() else "",
                     "head": _git(here, "rev-parse", "HEAD") if here.is_dir() else "",
                     "files": {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                               if (here / name).is_file() else "" for name in named}})
        return real(gateway, request, key)
    a.engine.start = start
    offer = first["continuable_seats"][0]
    out = _call(a, review_reference=first["review_reference"],
                continue_seats=[{"seat_id": "second", "operation_id": offer["operation_id"]}])
    assert out.get("status") not in {"refused", "decided_unsent"}, out
    [post] = seen
    assert post["root"] == str(root) and (post["tree"], post["head"]) == (subject["tree_sha"], subject["base"]), post
    assert all(post["files"][name] == name.rsplit("/", 1)[1][:-4] for name in named), post["files"]
    assert SESSION_ANSWER in json.dumps(_seat(a, first["record_id"]))
    marker = json.loads((root.parent / "checkout.json").read_text(encoding="utf-8"))
    assert marker["task_id"] == a.ctx.task_id, "the checkout pass retires it as any checkout of this task"


def test_a_released_checkout_whose_sources_are_gone_is_refused_before_any_choice(stopped_wave, monkeypatch):
    """The negative control: a retained source the original request names is no longer held, so the subject cannot be
    restored; the choice is refused typed at $0 before it is recorded, and nothing is POSTed."""
    import pathlib
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path

    a = stopped_wave
    first, original = _released_range_wave(a, monkeypatch)
    assert _drive_files(json.dumps(original))
    shutil.rmtree(task_artifact_dir_path(a.data, a.ctx.task_id) / "source_handles")
    posts, before = len(a.engine.posts), _seat(a, first["record_id"])
    out = _call(a, review_reference=first["review_reference"], continue_seats=[
        {"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}])
    assert (out["status"], out["reason"]) == ("refused", "subject_unavailable"), out
    assert len(a.engine.posts) == posts and _seat(a, first["record_id"]) == before
    assert not pathlib.Path(original["scope"]["root"]).exists()


def test_a_recorded_choice_whose_checkout_cannot_be_rebuilt_stays_decided_unsent_until_repeated(stopped_wave, monkeypatch):
    """After the choice is recorded, a failed rebuild is the choice's typed decided-unsent outcome: the choice stands,
    nothing is POSTed, and the same call repeated rebuilds the checkout and sends that very successor."""
    from ouroboros.tools import review_subject

    a = stopped_wave
    first, _original = _released_range_wave(a, monkeypatch)
    choice = [{"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}]
    real = review_subject._materialize_checkout

    def broken(*_args, **_kwargs):
        raise review_subject.StagedDiffUnavailable("injected materialization failure")
    monkeypatch.setattr(review_subject, "_materialize_checkout", broken)
    posts = len(a.engine.posts)
    out = _call(a, review_reference=first["review_reference"], continue_seats=choice)
    assert (out["status"], out["reason"]) == ("decided_unsent", "subject_unavailable"), out
    decided = _seat(a, first["record_id"])["continuations"]
    assert len(decided) == 1 and len(a.engine.posts) == posts
    monkeypatch.setattr(review_subject, "_materialize_checkout", real)
    again = _call(a, review_reference=first["review_reference"], continue_seats=choice)
    assert again.get("status") not in {"refused", "decided_unsent"}, again
    assert len(a.engine.posts) == posts + 1 and _seat(a, first["record_id"])["continuations"] == decided
    assert SESSION_ANSWER in json.dumps(_seat(a, first["record_id"]))


def _root_at_post(a, named):
    """Wrap the scripted engine: what each START's root holds at the moment it is POSTed."""
    import hashlib
    import pathlib

    seen, real = [], a.engine.start

    def start(gateway, request, key):
        here = pathlib.Path((request.get("scope") or {}).get("root") or "")
        seen.append({"key": key, "root": str(here), "tree": _git(here, "write-tree") if here.is_dir() else "",
                     "files": {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                               if (here / name).is_file() else "" for name in named}})
        return real(gateway, request, key)
    a.engine.start = start
    return seen


def test_a_carried_successor_start_replayed_after_a_restart_restores_its_pruned_checkout_before_the_post(
        stopped_wave, monkeypatch):
    """#2796, crash replay: the successor START was POSTed by a process that then ended unbound, and the checkout was
    removed meanwhile (a pass after the restart). The collection that replays that recorded start, the same key and
    bytes, first puts the checkout back from the START row's exact mapping; no rescan, no new generation."""
    import pathlib
    import shutil

    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    first, original = _released_range_wave(a, monkeypatch)
    subject = load_record(ledger_root(a.ctx), first["record_id"])["subject"]
    named = _drive_files(json.dumps(original))
    reference, record_id = first["review_reference"], first["record_id"]
    root, attempt, record = resolve_reference(a.ctx, reference)
    continue_or_repair(a.ctx, root, attempt, record, [
        {"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}])
    decision = _seat(a, record_id)["continuations"][0]
    started = delegate_custody.invocation_record(a.data, decision["invocation_id"])
    mapping = started["review_subject_restore"]
    assert mapping["root"] == original["scope"]["root"] and sorted(v["relative"] for v in mapping["sources"]) == named
    FakeGateway().start_run(started["request"], idempotency_key=decision["invocation_id"])  # POSTed; process ended unbound
    delegate_custody._CUSTODY.clear()
    checkout = pathlib.Path(mapping["root"])
    shutil.rmtree(checkout.parent)  # removed while no process held it
    monkeypatch.setattr("ouroboros.tools.review_subject.released_subject",
                        lambda *_a, **_k: pytest.fail("a replay restores from the recorded mapping, never a rescan"))
    seen = _root_at_post(a, named)
    collected = _call(a, review_reference=reference)
    assert collected["state"] == "settled" and SESSION_ANSWER in json.dumps(_seat(a, record_id)), collected
    [replay] = seen
    assert replay["key"] == decision["invocation_id"] and replay["tree"] == subject["tree_sha"], replay
    assert all(replay["files"][name] == name.rsplit("/", 1)[1][:-4] for name in named), replay["files"]
    assert len(_seat(a, record_id)["continuations"]) == 1


def test_a_replay_whose_checkout_cannot_be_restored_is_not_posted_and_stays_pending(stopped_wave, monkeypatch):
    """The negative control of the replay: the mapping's sources are gone, so the recorded start is not POSTed; it stays
    a decided, pending start (never a definite refusal), and nothing new is chosen."""
    import pathlib
    import shutil

    from ouroboros.artifacts import task_artifact_dir_path
    from ouroboros.tools.review_collection import continue_or_repair, resolve_reference

    a = stopped_wave
    first, _original = _released_range_wave(a, monkeypatch)
    reference, record_id = first["review_reference"], first["record_id"]
    root, attempt, record = resolve_reference(a.ctx, reference)
    continue_or_repair(a.ctx, root, attempt, record, [
        {"seat_id": "second", "operation_id": first["continuable_seats"][0]["operation_id"]}])
    decision = _seat(a, record_id)["continuations"][0]
    mapping = delegate_custody.invocation_record(a.data, decision["invocation_id"])["review_subject_restore"]
    shutil.rmtree(pathlib.Path(mapping["root"]).parent)
    shutil.rmtree(task_artifact_dir_path(a.data, a.ctx.task_id) / "source_handles")
    posts = len(a.engine.posts)
    _call(a, review_reference=reference)
    assert len(a.engine.posts) == posts, "nothing was POSTed into a missing subject"
    assert delegate_custody.invocation_record(a.data, decision["invocation_id"])["state"] == "pending"
    assert len(_seat(a, record_id)["continuations"]) == 1


def _scope(a, task_id):
    import queue

    from ouroboros import model_wait
    return model_wait.task_model_wait_scope(task={"id": task_id, "chat_id": 7, "_attempt": 1}, drive_root=a.data,
                                            event_queue=queue.Queue(), worker_slot_held=False)


def test_a_recorded_continue_successor_continues_its_predecessors_released_seat_and_owns_the_checkout(stopped_wave, monkeypatch):
    """#2796 with #2901: the predecessor ended; its recorded Continue successor explicitly continues the predecessor's
    stopped seat over the OLD frozen subject (read-only: no Git effect on its files, no landing). The released checkout
    is restored before the POST under a marker naming the deciding successor, so a real checkout pass keeps it while
    that successor lives even though the review's own task is terminal; the same checkout owned by the ended task is
    removed (the negative control of ownership)."""
    import pathlib

    from ouroboros.owner_continue import binding_sha
    from ouroboros.review_checking_prune import prune_review_checkouts
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document
    from ouroboros.tools.review_subject import CHECKOUT_MARKER

    a = stopped_wave
    monkeypatch.setattr(FakeGateway, "ask_input_limits", lambda self: _AnyRoute({"*": {"prompt_budget": 1200}}), raising=False)
    head = _git(a.repo, "rev-parse", "HEAD")
    handback = _commit_json(_commit_call(a))
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    original = a.engine.posts[0][0]
    checkout = pathlib.Path(original["scope"]["root"])
    named = _drive_files(json.dumps(original))
    assert offer and not checkout.exists(), "the commit gate released its checkout at the wave's end"
    predecessor, successor = a.ctx.task_id, "continued-task"
    write_task_result(a.data, predecessor, "completed", chat_id=7)
    write_task_result(a.data, successor, "running", chat_id=7)
    binding = {"successor_task_id": successor}
    patch_document(a.data, predecessor, lambda current: {**current, "continued_by": {
        "successor_task_id": successor, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    seen = _root_at_post(a, named)
    a.ctx.task_id = successor
    with _scope(a, successor):
        continued = a.git._commit_reviewed(a.ctx, review_reference=reference, continue_seats=[
            {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    record_id = reference["review_record_id"]
    assert "read-only" in continued and "recorded Continue predecessor" in continued, continued
    [post] = seen
    assert post["root"] == str(checkout) and post["tree"] and all(post["files"][n] for n in named), post
    seat = _seat(a, record_id)
    assert seat["status"] == "responded" and seat["continuations"][0]["decided_by_task"] == successor
    assert _git(a.repo, "rev-parse", "HEAD") == head
    marker = checkout.parent / CHECKOUT_MARKER
    assert json.loads(marker.read_text(encoding="utf-8"))["task_id"] == successor
    kept = prune_review_checkouts(a.data, {successor}, retention_days=0)
    assert checkout.is_dir() and kept["checkouts_kept"] >= 1, kept
    marker.write_text(json.dumps({"task_id": predecessor, "created_at": "2026-01-01T00:00:00+00:00", "schema_version": 1}))
    removed = prune_review_checkouts(a.data, {successor}, retention_days=0)
    assert not checkout.exists() and removed["checkouts_removed"] >= 1, removed


def test_a_pass_that_judged_the_old_checkout_keeps_one_restored_under_the_custody_lock(stopped_wave, monkeypatch):
    """The checkout pass's removal and a continuation's restore share the existing per-task custody lock
    (``owner_keep.gate``): a restore waits for a removal in progress, and a removal that judged the old marker keeps a
    checkout restored since (``rebuilt_since_judged``) or one whose owner's lock is held (``busy``)."""
    import pathlib
    import threading

    from ouroboros.review_checking_prune import _marker_text, _remove_unless_kept
    from ouroboros.task_custody import task_custody_lock
    from ouroboros.tools.review_collection import resolve_reference
    from ouroboros.tools.review_subject import released_subject, restore_released_subject

    a = stopped_wave
    first, original = _released_range_wave(a, monkeypatch)
    _root, _attempt, record = resolve_reference(a.ctx, first["review_reference"])
    from ouroboros.delegate_invocation import operation_record
    origin = delegate_custody.invocation_record(a.data, operation_record(
        a.data, first["continuable_seats"][0]["operation_id"], "second")["invocation_id"]) or {}
    mapping = released_subject(a.data, record, original["scope"]["root"], origin, "deciding-task")
    entry = pathlib.Path(mapping["root"]).parent
    entry.mkdir(parents=True)  # the old checkout a pass judged: an ended task's marker, nothing of the subject left
    (entry / "checkout.json").write_text(json.dumps({"task_id": "ended-task", "created_at": "x", "schema_version": 1}))
    judged = _marker_text(entry)
    released, done = threading.Event(), {}

    def restore():
        done["wrote"] = restore_released_subject(a.data, mapping)
    with task_custody_lock(a.data, "ended-task", timeout_sec=5) as locked:  # a removal in progress holds the old owner
        assert locked
        worker = threading.Thread(target=restore)
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not (entry / "repo.noindex").exists(), "the restore waits for that removal"
        released.set()
    worker.join(60)
    assert done["wrote"] is True and (entry / "repo.noindex").is_dir()
    assert _remove_unless_kept(a.data, entry, ((), {}), judged).endswith("rebuilt_since_judged")
    with task_custody_lock(a.data, "deciding-task", timeout_sec=5):
        assert _remove_unless_kept(a.data, entry, ((), {}), _marker_text(entry)).endswith("busy")
    assert (entry / "repo.noindex").is_dir()


@pytest.mark.parametrize("actor", ["successor", "same_owner"])
def test_an_existing_checkout_is_held_for_the_deciding_task_against_a_pass_that_already_judged_it(stopped_wave, monkeypatch, actor):
    """The checkout still EXISTS when the continuation is decided: its wave kept it (``isolated_checkout``'s own
    retention) under its task's marker. A checkout pass that already judged that marker (the predecessor ended, or the
    same task was not live when judged) runs its removal just before the start is POSTed; the continuation took the
    custody locks and rewrote the marker for the deciding task, so the removal keeps it and the POST reads the complete
    frozen subject. The normal retirement still happens once the deciding task ended."""
    import pathlib

    from ouroboros import review_session_custody
    from ouroboros.owner_continue import binding_sha
    from ouroboros.review_checking_prune import (
        _marker_text,
        _remove_unless_kept,
        _review_rounds,
        prune_review_checkouts,
    )
    from ouroboros.task_results import write_task_result
    from ouroboros.task_store import patch_document
    from ouroboros.tools import review_subject
    from ouroboros.tools.review_subject import CHECKOUT_MARKER

    a = stopped_wave
    monkeypatch.setattr(FakeGateway, "ask_input_limits", lambda self: _AnyRoute({"*": {"prompt_budget": 1200}}), raising=False)
    with pytest.MonkeyPatch.context() as kept:  # the wave ends with its custody undetermined: isolated_checkout keeps the checkout
        kept.setattr(review_subject, "checkout_retention", lambda _retain: {"custody_undetermined": "scripted"})
        handback = _commit_json(_commit_call(a))
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    original = a.engine.posts[0][0]
    checkout = pathlib.Path(original["scope"]["root"])
    named = _drive_files(json.dumps(original))
    predecessor, successor = a.ctx.task_id, "continued-task"
    assert checkout.is_dir() and named and json.loads((checkout.parent / CHECKOUT_MARKER).read_text())["task_id"] == predecessor
    judged = _marker_text(checkout.parent)  # an already-started pass judged this checkout removable
    if actor == "same_owner":
        successor = predecessor
    else:
        write_task_result(a.data, predecessor, "completed", chat_id=7)
        write_task_result(a.data, successor, "running", chat_id=7)
        binding = {"successor_task_id": successor}
        patch_document(a.data, predecessor, lambda current: {**current, "continued_by": {
            "successor_task_id": successor, "binding": binding, "binding_sha256": binding_sha(binding), "state": "admitted"}})
    removals, real_prepare = [], review_session_custody.prepare_continuation_send

    def removal_then_send(invocation, data_root, record, run_id):
        removals.append(_remove_unless_kept(a.data, checkout.parent, _review_rounds(a.data), judged))
        return real_prepare(invocation, data_root, record, run_id)
    monkeypatch.setattr(review_session_custody, "prepare_continuation_send", removal_then_send)  # the recovered seat's one send step
    seen = _root_at_post(a, named)
    a.ctx.task_id = successor
    with _scope(a, successor):
        continued = a.git._commit_reviewed(a.ctx, review_reference=reference, continue_seats=[
            {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    assert removals and removals[0].endswith("rebuilt_since_judged"), (removals, continued)
    [post] = seen
    assert post["root"] == str(checkout) and post["tree"] and all(post["files"][n] for n in named), post
    assert json.loads((checkout.parent / CHECKOUT_MARKER).read_text())["task_id"] == successor
    assert prune_review_checkouts(a.data, {successor}, retention_days=0)["checkouts_kept"] >= 1 and checkout.is_dir()
    write_task_result(a.data, successor, "completed", chat_id=7)
    assert prune_review_checkouts(a.data, set(), retention_days=0)["checkouts_removed"] >= 1 and not checkout.exists()


def test_a_gate_continuation_on_the_reviewed_bytes_posts_with_every_paged_view(stopped_wave, monkeypatch):
    """The commit gate's own path (the reviewed bytes staged again): the gate re-freezes its round's checkout from the
    index, which restores the tree but none of the paged ``.review-drive`` views the original START named; the
    continuation's send restores them from its START row's mapping before the POST."""
    import pathlib

    a = stopped_wave
    monkeypatch.setattr(FakeGateway, "ask_input_limits", lambda self: _AnyRoute({"*": {"prompt_budget": 1200}}), raising=False)
    handback = _commit_json(_commit_call(a))
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    original = a.engine.posts[0][0]
    named = _drive_files(json.dumps(original))
    assert named and not pathlib.Path(original["scope"]["root"]).exists()
    _git(a.repo, "add", "change.py")
    seen = _root_at_post(a, named)
    continued = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    [post] = seen
    assert post["root"] == original["scope"]["root"] and post["tree"], (post, continued)
    assert all(post["files"][name] == name.rsplit("/", 1)[1][:-4] for name in named), post["files"]
    assert "read-only" not in continued, "the exact bytes continue inside the paid gate itself"
