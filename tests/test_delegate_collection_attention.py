"""The body observer watches held runs through the compact wait and reads detail on a reason."""
from __future__ import annotations

import logging
import threading
import time

import pytest

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody_current as current
from ouroboros.delegate_supervision import _TICK_SEC
from ouroboros.gateways import claudexor_observation as obs
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests.test_delegate_body_collection import started, terminal

TOKEN_A = "a" * 43
TOKEN_B = "b" * 43


def _summary(status="running", pending=False, token=TOKEN_A):
    return {"status": status, "pending": pending, "token": token, "problem": None}


def _wake(reason, summaries, attention=()):
    return {"wakeReason": reason, "attention": list(attention), "summaries": summaries}


def _retryable(code="store_closed"):
    exc = ClaudexorUnavailable(code, "the store is closed for maintenance", status_code=503)
    exc.problem = {"code": code, "message": "the store is closed for maintenance", "retryable": True}
    return exc


def _question(interaction_id="int-1"):
    return {"interactionId": interaction_id, "runId": "run-one", "questions": [
        {"id": "q1", "question": "Which branch?", "options": [], "multi_select": False}]}


class Engine:
    """An engine listing the compact wait: scripted per pass, details on demand; ``repaired`` says
    whether its route accepts the problem acknowledgement (the contract probe's answer)."""
    engine_version = "4.8.1"
    engine_build_sha = "sha-attention"
    store_identity = ""
    routes = [{"method": "POST", "path": "/v2/runs/wait-many"}]
    repaired = True
    probe_answer = None  # when set, the contract probe raises (an exception) or returns it instead
    wakes: list = []
    details: dict = {}
    requests: list = []
    probes: list = []
    detail_reads: list = []
    catalog: list = []
    hold = False  # honour the requested hold (an engine default stands in for ``None``)

    def operations(self, *, timeout_sec=None):
        if Engine.catalog and isinstance(Engine.catalog[0], BaseException):
            raise Engine.catalog.pop(0)
        return list(self.routes)

    def wait_run_attention(self, targets, *, wait_ms, timeout_sec):
        if targets == [{"runId": obs.PROBE_RUN_ID, "seen": None}]:
            Engine.probes.append(wait_ms)
            if isinstance(Engine.probe_answer, BaseException):
                raise Engine.probe_answer
            if Engine.probe_answer is not None:
                return Engine.probe_answer
            if not Engine.repaired:
                raise ClaudexorUnavailable("invalid_request", "targets[0].seen: expected string", status_code=400)
            return _wake("timeout", [{"status": None, "pending": None, "token": None, "problem": "not_found"}])
        Engine.requests.append({"targets": targets, "wait_ms": wait_ms})
        if Engine.hold:
            time.sleep((25_000 if wait_ms is None else wait_ms) / 1000)
        answer = Engine.wakes.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get_run(self, rid, **kwargs):
        Engine.detail_reads.append(rid)
        answer = Engine.details[rid]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def close(self):
        pass


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """The real runtime with its driver parked; ``step`` runs one tick and joins its passes."""
    monkeypatch.setattr(obs, "_SUPPORT", {})
    monkeypatch.setattr(run_events, "_SUPPORT", {})
    Engine.wakes, Engine.details, Engine.requests, Engine.detail_reads, Engine.catalog = [], {}, [], [], []
    Engine.probes, Engine.hold, Engine.repaired, Engine.probe_answer = [], False, True, None
    release, entered = threading.Event(), threading.Event()
    tick = body.BodyRuntime.tick

    def park(_runtime):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(body.BodyRuntime, "tick", park)
    instance = body.start_body_runtime(tmp_path, gateway_factory=Engine, max_workers=2)
    assert entered.wait(3)
    # Support is learned from the engine's catalog by the first read of any run
    # (``test_support_is_learned_by_the_first_read``); these tests start past that point.
    instance._compact_wait = True

    def step():
        tick(instance)
        with instance.lock:
            jobs = list(instance.reads.values()) + list(instance.collections.values())
        for job in jobs:
            job.result(timeout=5)
        with instance.lock:
            jobs = list(instance.reads.values())
        for job in jobs:  # a fallback pass submits per-run reads of its own
            job.result(timeout=5)

    try:
        yield instance, step
    finally:
        instance.close(wait=False)
        release.set()
        instance.close(wait=True)


def test_a_quiet_run_is_watched_without_a_detail_read(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.wakes.append(_wake("timeout", [_summary()]))
    step()
    assert Engine.requests[0]["targets"] == [{"runId": entry.run_id}]
    assert Engine.requests[0]["wait_ms"] == _TICK_SEC * 1000, "the hold is the observer's own tick window"
    assert Engine.detail_reads == []
    fact = instance.observation(entry.run_id)
    assert fact["answered"] is True and fact["run_state"] == "running" and fact["waiting_on_user"] is False


def test_attention_fetches_the_question_before_the_token_is_acknowledged(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    asked = {"summary": {"runId": entry.run_id, "state": "running", "waitingOnUser": True},
             "lastSeq": 2, "pendingInteractions": [_question()]}
    Engine.details[entry.run_id] = asked
    Engine.wakes.append(_wake("attention", [_summary(pending=True, token=TOKEN_B)], attention=[0]))
    step()
    assert Engine.detail_reads == [entry.run_id]
    assert current.run(tmp_path, entry.run_id).pending_interaction_ids == ["int-1"]
    assert instance.seen[entry.run_id] == TOKEN_B
    Engine.wakes.append(_wake("timeout", [_summary(pending=True, token=TOKEN_B)]))
    step()
    assert Engine.requests[1]["targets"] == [{"runId": entry.run_id, "seen": TOKEN_B}]
    assert Engine.detail_reads == [entry.run_id], "a known question is not fetched again"


def test_a_failed_fetch_leaves_the_token_unacknowledged_so_the_question_wakes_again(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
    Engine.wakes.append(_wake("attention", [_summary(pending=True, token=TOKEN_B)], attention=[0]))
    step()
    assert Engine.detail_reads == [entry.run_id]
    assert entry.run_id not in instance.seen
    assert instance.observation(entry.run_id)["error_code"] == "daemon_busy"


def test_a_terminal_wake_retains_the_source_through_one_detail_read(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = terminal(entry.run_id)
    Engine.wakes.append(_wake("attention", [_summary(status="succeeded", token=TOKEN_B)], attention=[0]))
    step()
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.terminal_state == "succeeded" and fresh.terminal_source
    assert Engine.detail_reads == [entry.run_id]


def test_a_hole_is_remembered_as_unanswered_and_never_raises_out_of_the_pass(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.wakes.append(ClaudexorUnavailable("daemon_unreachable", "read bound expired",
                                             observation_timeout=True, observation_reason="observation_read_timeout"))
    step()
    fact = instance.observation(entry.run_id)
    assert fact["answered"] is False and fact["error_code"] == "daemon_unreachable"
    assert Engine.detail_reads == []


@pytest.mark.parametrize("kind", ["not_found", "snapshot_unavailable"])
def test_a_problem_row_takes_the_addressed_read_that_may_say_absent_then_acknowledges_it(tmp_path, runtime, kind):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    Engine.wakes.append(_wake("problems", [{"status": None, "pending": None, "token": None, "problem": kind}]))
    step()
    assert Engine.detail_reads == [entry.run_id], "a problem row is not absence; the addressed read decides"
    assert instance.observation(entry.run_id)["run_state"] == "running"
    assert instance.seen[entry.run_id] == obs.ACKNOWLEDGED_PROBLEM
    Engine.wakes.append(_wake("timeout", [{"status": None, "pending": None, "token": None, "problem": kind}]))
    step()
    assert Engine.requests[1]["targets"] == [{"runId": entry.run_id, "seen": None}], "acknowledged: the engine holds again"
    assert Engine.detail_reads == [entry.run_id, entry.run_id], "the addressed read is taken again, not spun"


def test_a_problem_row_whose_addressed_read_fails_stays_unacknowledged(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
    Engine.wakes.append(_wake("problems", [{"status": None, "pending": None, "token": None, "problem": "not_found"}]))
    step()
    assert Engine.detail_reads == [entry.run_id] and entry.run_id not in instance.seen


def test_an_engine_whose_route_rejects_the_acknowledgement_keeps_todays_beat_of_one_detail_per_run(tmp_path, runtime):
    instance, step = runtime
    instance._compact_wait = False  # never learned on this engine
    Engine.repaired = False  # lists the route, serves it unrepaired: the release number says nothing
    entry = started(tmp_path)
    other = started(tmp_path, rid="run-two", tid="task-two")
    for rid in (entry.run_id, other.run_id):
        Engine.details[rid] = {"summary": {"runId": rid, "state": "running"}, "lastSeq": 1, "pendingInteractions": []}
    step()
    assert Engine.requests == [] and sorted(Engine.detail_reads) == [entry.run_id, other.run_id]
    assert Engine.probes == [0], "probed once, remembered for this engine identity"
    step()
    assert len(Engine.detail_reads) == 4 and Engine.probes == [0]


def test_support_is_learned_by_the_first_read_then_the_pass_takes_over(tmp_path, runtime):
    instance, step = runtime
    instance._compact_wait = False
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    step()
    assert Engine.detail_reads == [entry.run_id] and Engine.requests == []
    assert instance._compact_wait is True
    Engine.wakes.append(_wake("timeout", [_summary()]))
    step()
    assert Engine.detail_reads == [entry.run_id] and len(Engine.requests) == 1


def test_shutdown_during_a_hold_is_bounded_by_the_observers_tick_and_transport_bound(tmp_path, runtime):
    """Closing the client does not wake a receive blocked in the hold on every platform, so the
    pass itself must end within the tick plus the transport bound (about eight seconds), never
    the engine's window; the double honours the hold it was asked for and answers at once after it."""
    instance, step = runtime
    entry = started(tmp_path)
    Engine.hold = True
    Engine.wakes.append(_wake("timeout", [_summary()]))
    began = time.monotonic()
    pass_ = instance.observers.submit(instance._watch, [entry])
    time.sleep(0.2)  # the pass is inside its hold
    instance._drop_observer_gateway()
    pass_.result(timeout=_TICK_SEC + 2.0)
    assert time.monotonic() - began < _TICK_SEC + 1.0, "a hold that answers at its end: the tick, not the engine's window"


def test_a_catalog_hole_never_blocks_the_old_read(tmp_path, runtime):
    instance, step = runtime
    instance._compact_wait = False
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    Engine.catalog.append(ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                                               observation_reason="observation_read_timeout"))
    step()
    assert Engine.detail_reads == [entry.run_id], "the question could not be asked; the old read ran regardless"
    assert instance.observation(entry.run_id)["run_state"] == "running" and instance._compact_wait is False


@pytest.mark.parametrize("refusal", [_retryable("store_closed"),
                                     ClaudexorUnavailable("conflict", "another writer", status_code=409)],
                         ids=["503-retryable", "409"])
def test_a_persistently_refused_probe_never_blocks_the_old_read_and_is_asked_again(tmp_path, runtime, refusal):
    instance, step = runtime
    instance._compact_wait = False
    entry = started(tmp_path)
    Engine.details[entry.run_id] = terminal(entry.run_id)
    Engine.probe_answer = refusal
    step()
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.terminal_state == "succeeded" and fresh.terminal_source, "the old read reached the terminal"
    assert Engine.probes == [0] and instance._compact_wait is False
    assert obs._SUPPORT == {}, "nothing remembered of the refusal"
    other = started(tmp_path, rid="run-two", tid="task-two")
    Engine.details[other.run_id] = {"summary": {"runId": other.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    step()
    assert len(Engine.probes) == 2 and other.run_id in Engine.detail_reads, "asked again next beat; read regardless"


def test_a_retryable_refusal_of_the_pass_is_a_hole_not_a_verdict(tmp_path, runtime, caplog):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.wakes.append(_retryable("store_maintenance_failed"))
    caplog.set_level(logging.WARNING, logger=body.__name__)
    step()
    fact = instance.observation(entry.run_id)
    assert fact["answered"] is True and fact["error_code"] == "store_maintenance_failed"
    assert instance._compact_wait is True and obs._SUPPORT.get(("4.8.1", "sha-attention")) is True, "not switched off"
    assert not any("compact wait unusable" in record.message for record in caplog.records)
    Engine.wakes.append(_wake("timeout", [_summary()]))
    step()
    assert len(Engine.requests) == 2 and Engine.detail_reads == [], "the next pass holds again"


def test_a_plain_refusal_of_the_pass_forgets_once(tmp_path, runtime, caplog):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    Engine.wakes.append(ClaudexorUnavailable("conflict", "another writer", status_code=409))
    caplog.set_level(logging.WARNING, logger=body.__name__)
    step()
    assert instance._compact_wait is False and obs._SUPPORT.get(("4.8.1", "sha-attention")) is False
    assert sum("compact wait unusable" in record.message for record in caplog.records) == 1
    step()
    assert Engine.detail_reads == [entry.run_id] and len(Engine.requests) == 1


def test_a_refused_or_malformed_pass_returns_the_observer_to_its_old_beat_once(tmp_path, runtime, caplog):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    Engine.wakes.append({"wakeReason": "attention", "attention": [], "summaries": []})  # a batch of the wrong size
    caplog.set_level(logging.WARNING, logger=body.__name__)
    step()
    assert instance._compact_wait is False and Engine.detail_reads == []
    assert instance.observation(entry.run_id)["error_code"] == "malformed_response", "nothing swallowed silently"
    assert sum("compact wait unusable" in record.message for record in caplog.records) == 1
    step()
    step()
    assert Engine.detail_reads == [entry.run_id] * 2 and len(Engine.requests) == 1, "the old beat, not relearned from the cache"
    assert sum("compact wait unusable" in record.message for record in caplog.records) == 1, "logged once"


def test_an_answered_question_is_cleared_by_one_addressed_read_and_other_quiet_rows_stay_read_free(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running", "waitingOnUser": True},
                                    "lastSeq": 2, "pendingInteractions": [_question()]}
    Engine.wakes.append(_wake("attention", [_summary(pending=True, token=TOKEN_B)], attention=[0]))
    step()
    assert current.run(tmp_path, entry.run_id).pending_interaction_ids == ["int-1"]
    answered = "c" * 43
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running", "waitingOnUser": False},
                                    "lastSeq": 3, "pendingInteractions": []}
    Engine.wakes.append(_wake("timeout", [_summary(pending=False, token=answered)]))
    step()
    assert Engine.detail_reads == [entry.run_id] * 2, "the answered question earns one addressed read"
    assert current.run(tmp_path, entry.run_id).pending_interaction_ids == []
    Engine.wakes.append(_wake("timeout", [_summary(pending=False, token=answered)]))
    step()
    assert Engine.detail_reads == [entry.run_id] * 2, "every other quiet row stays read-free"


def test_an_engine_that_lost_the_route_returns_the_observer_to_its_old_beat(tmp_path, runtime):
    instance, step = runtime
    entry = started(tmp_path)
    Engine.details[entry.run_id] = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 1,
                                    "pendingInteractions": []}
    Engine.wakes.append(ClaudexorUnavailable("not_found", "no such route", status_code=404))
    step()
    assert Engine.detail_reads == [entry.run_id], "the pass that met the 404 read the detail itself"
    assert instance._compact_wait is False
    step()
    assert Engine.detail_reads == [entry.run_id, entry.run_id] and len(Engine.requests) == 1
