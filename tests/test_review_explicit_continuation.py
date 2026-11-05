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
        return {"runId": run_id, "runDir": "/tmp/fake-run"}

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
    from ouroboros import config
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
        from ouroboros.tools.review_helpers import review_enforcement_blocks
        root = subject.review_root if subject is not None else ctx.repo_dir
        return ({"blocking_review": review_enforcement_blocks(config.get_review_enforcement()),
                 "prompt": "Review this candidate.", "models": list(plan["models"]), "stable_prefix_len": 0,
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


def test_a_commit_continuation_arms_only_on_the_reviewed_bytes_and_never_commits_by_itself(stopped_wave):
    a = stopped_wave
    head = _git(a.repo, "rev-parse", "HEAD")
    handback = _commit_json(_commit_call(a))
    reference, offer = handback["review_reference"], handback["continuable_seats"]
    assert [item["seat_id"] for item in offer] == ["second"] and len(a.engine.posts) == 1
    (a.repo / "change.py").write_text("value = 3\n", encoding="utf-8")  # other bytes than the paid review saw
    _git(a.repo, "add", "change.py")
    moved = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    assert "REVIEW_BINDING_BLOCKED" in moved and len(a.engine.posts) == 1  # nothing armed for changed bytes
    record_id = reference["review_record_id"]
    assert "continuations" not in _seat(a, record_id)
    (a.repo / "change.py").write_text("value = 2\n", encoding="utf-8")
    _git(a.repo, "add", "change.py")
    continued = _commit_call(a, review_reference=reference, continue_seats=[
        {"seat_id": "second", "operation_id": offer[0]["operation_id"]}])
    assert len(a.engine.posts) == 2 and a.engine.posts[1][0]["continueFrom"] == "run-1"
    assert _seat(a, record_id)["status"] == "responded"
    assert "no commit, tag or push occurred" in continued and _git(a.repo, "rev-parse", "HEAD") == head


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
