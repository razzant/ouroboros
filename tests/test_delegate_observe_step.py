"""The supervised wait's beat on a repaired engine: speech from the journal, attention from the engine."""
from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from ouroboros import delegate_activity, delegate_observe
from ouroboros.gateways import claudexor_observation as obs
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._delegated_transport_shared import _delegating_ctx

TOKEN_A = "a" * 43
TOKEN_B = "b" * 43
ROUTES = [{"method": "POST", "path": "/v2/runs/wait-many"}, {"method": "GET", "path": "/v2/runs/:id/events"}]


def _summary(status="running", pending=False, token=TOKEN_A):
    return {"status": status, "pending": pending, "token": token, "problem": None}


def _wake(reason, summaries, attention=()):
    return {"wakeReason": reason, "attention": list(attention), "summaries": summaries}


def _frames(events, *, end=False):
    text = ""
    for seq, value in events:
        text += f"id: {seq}\nevent: {value['type']}\ndata: {json.dumps(value)}\n\n"
    if end:
        text += "event: end\ndata: {}\n\n"
    return text


class _Stream:
    def __init__(self, text, status_code=200, code=""):
        self.status_code, self._text, self._code = status_code, text, code

    def iter_text(self):
        yield self._text

    def read(self):
        return json.dumps({"code": self._code}).encode()


class Engine:
    engine_version = "4.8.1"
    engine_build_sha = "sha-observe"
    routes = ROUTES

    repaired = True  # the route accepts the problem acknowledgement (the contract probe's answer)

    def __init__(self, *, streams=(), wakes=(), details=(), catalog=None, probe_answer=None):
        self.streams, self.wakes, self.details = list(streams), list(wakes), list(details)
        self.stream_opens, self.requests, self.detail_reads, self.probes = [], [], [], []
        self.catalog, self.probe_answer = catalog, probe_answer

    def operations(self, *, timeout_sec=None):
        if isinstance(self.catalog, BaseException):
            raise self.catalog
        return list(self.routes)

    def open_run_events(self, run_id, after_seq, timeout_sec):
        self.stream_opens.append((run_id, after_seq, timeout_sec))
        answer = self.streams.pop(0) if self.streams else ""
        if isinstance(answer, BaseException):
            raise answer
        return contextlib.nullcontext(answer if isinstance(answer, _Stream) else _Stream(answer))

    def wait_run_attention(self, targets, *, wait_ms, timeout_sec):
        if targets == [{"runId": obs.PROBE_RUN_ID, "seen": None}]:
            self.probes.append(wait_ms)
            if isinstance(self.probe_answer, BaseException):
                raise self.probe_answer
            if not self.repaired:
                raise ClaudexorUnavailable("invalid_request", "targets[0].seen: expected string", status_code=400)
            return _wake("timeout", [{"status": None, "pending": None, "token": None, "problem": "not_found"}])
        self.requests.append({"targets": targets, "wait_ms": wait_ms})
        answer = self.wakes.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get_run(self, run_id, *, timeout_sec=None):
        self.detail_reads.append(run_id)
        return self.details.pop(0)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "_SUPPORT", {})
    delegate_activity.reset_process_memo()
    context = _delegating_ctx(tmp_path, acting=False)
    frames = []
    context.emit_progress_fn = lambda text, **meta: frames.append((text, meta))
    context.frames = frames
    return context


def _speech(seq, text):
    return (seq, {"type": "harness.event", "seq": seq, "run_id": "run-s",
                  "payload": {"type": "message", "text": text, "harness_id": "codex", "attempt_id": "a1"}})


def test_a_quiet_beat_reads_no_detail_and_speech_reaches_the_owner(ctx):
    gateway = Engine(streams=[_frames([_speech(1, "hello"), _speech(2, "world")])],
                     wakes=[_wake("timeout", [_summary()])])
    step = delegate_observe.step(ctx, gateway, "run-s", window=3.0, since_seq=0, read_window=lambda: 30.0,
                                 elapsed_seconds=42, max_seconds=600)
    assert step.detail is None and gateway.detail_reads == []
    payload = json.loads(step.payload)
    assert payload["status"] == "progress" and payload["last_seq"] == 2 and payload["state"] == "running"
    assert payload["answered"] is True and payload["waiting_on_user"] is False
    assert (payload["elapsed_seconds"], payload["max_seconds"]) == (42, 600), "the facts against a premature cancel"
    assert gateway.stream_opens == [("run-s", 0, 3.0)]
    assert gateway.requests == [{"targets": [{"runId": "run-s"}], "wait_ms": 0}], "attention is a present-state read"
    assert len(ctx.frames) == 1 and "hello" in ctx.frames[0][0]
    assert ctx.frames[0][1]["delegated_activity"]["through_seq"] == 2
    again = delegate_observe.step(ctx, Engine(wakes=[_wake("timeout", [_summary()])]), "run-s",
                                  window=3.0, since_seq=2, read_window=lambda: 30.0)
    assert json.loads(again.payload)["status"] == "no_progress", "a silent beat is quiet"


def test_attention_earns_the_detail_and_acknowledges_the_token_after_it(ctx):
    asked = {"summary": {"state": "running", "waitingOnUser": True}, "lastSeq": 3, "pendingInteractions": [
        {"interactionId": "i-1", "questions": [{"id": "q", "question": "?", "options": [], "multi_select": False}]}]}
    gateway = Engine(streams=[""], wakes=[_wake("attention", [_summary(pending=True, token=TOKEN_B)], [0]),
                                          _wake("timeout", [_summary(pending=True, token=TOKEN_B)])],
                     details=[asked, asked])
    step = delegate_observe.step(ctx, gateway, "run-s", window=3.0, since_seq=0, read_window=lambda: 30.0)
    assert step.detail is asked and gateway.detail_reads == ["run-s"]
    gateway.streams.append("")
    again = delegate_observe.step(ctx, gateway, "run-s", window=3.0, since_seq=3, read_window=lambda: 30.0)
    assert gateway.requests[1]["targets"] == [{"runId": "run-s", "seen": TOKEN_B}]
    assert again.detail is asked and gateway.detail_reads == ["run-s"] * 2, \
        "a run waiting on its person is never a quiet beat: its question is projected from fresh detail"


def test_the_streams_end_earns_the_terminal_detail(ctx):
    done = {"summary": {"state": "succeeded"}, "lastSeq": 2, "pendingInteractions": []}
    gateway = Engine(streams=[_frames([_speech(1, "bye")], end=True)],
                     wakes=[_wake("attention", [_summary(status="succeeded", token=TOKEN_B)], [0])], details=[done])
    step = delegate_observe.step(ctx, gateway, "run-s", window=3.0, since_seq=0, read_window=lambda: 30.0)
    assert step.detail is done and len(ctx.frames) == 1


def test_a_hole_on_the_attention_read_raises_the_gateways_typed_hole(ctx):
    hole = ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                                observation_reason="observation_read_timeout")
    gateway = Engine(streams=[""], wakes=[hole])
    with pytest.raises(ClaudexorUnavailable) as caught:
        delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0)
    assert caught.value is hole


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), _Stream("", 503, "daemon_busy"),
                                     _Stream("", 503, "daemon_stopping"), _Stream("", 404, "not_found")],
                         ids=["connect", "busy", "stopping", "absent"])
def test_a_journal_that_could_not_be_read_hands_the_beat_to_the_detail_read(ctx, failure):
    gateway = Engine(streams=[failure], wakes=[_wake("timeout", [_summary()])])
    assert delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0) is None
    assert gateway.requests == [], "the detail beat decides this one, with its own hole rules and degraded window"


def test_a_problem_row_hands_the_beat_back_to_the_detail_read(ctx):
    gateway = Engine(streams=[""], wakes=[_wake("problems", [
        {"status": None, "pending": None, "token": None, "problem": "not_found"}])])
    assert delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0) is None


def test_the_seen_memo_is_bounded_like_the_activity_memo():
    delegate_observe.reset_process_seen()
    limit = delegate_activity._MEMO_LIMIT
    for index in range(limit + 5):
        delegate_observe._acknowledge(("t", f"run-{index}"), TOKEN_A)
    assert len(delegate_observe._SEEN) == limit
    assert ("t", "run-0") not in delegate_observe._SEEN and ("t", f"run-{limit + 4}") in delegate_observe._SEEN
    delegate_observe._acknowledge(("t", "run-5"), TOKEN_B)
    assert next(iter(delegate_observe._SEEN)) == ("t", "run-6"), "a re-acknowledged run is the newest, not evicted next"
    delegate_observe.reset_process_seen()


@pytest.mark.parametrize("routes", [[ROUTES[0]], [ROUTES[1]], []])
def test_an_engine_missing_either_surface_keeps_the_detail_beat(ctx, routes):
    gateway = Engine()
    gateway.routes = routes
    assert delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0) is None
    assert gateway.stream_opens == [] and gateway.requests == []


def _retryable(code="store_closed"):
    exc = ClaudexorUnavailable(code, "the store is closed for maintenance", status_code=503)
    exc.problem = {"code": code, "message": "the store is closed for maintenance", "retryable": True}
    return exc


@pytest.mark.parametrize("engine", [
    lambda: Engine(catalog=ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                                                observation_reason="observation_read_timeout")),
    lambda: Engine(probe_answer=_retryable("store_closed")),
    lambda: Engine(probe_answer=ClaudexorUnavailable("conflict", "another writer", status_code=409)),
], ids=["catalog-hole", "probe-503-retryable", "probe-409"])
def test_a_question_that_cannot_be_asked_hands_the_beat_to_the_detail_read(ctx, engine):
    gateway = engine()
    assert delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0) is None
    assert gateway.stream_opens == [] and gateway.requests == [] and obs._SUPPORT == {}


def test_an_engine_whose_route_rejects_the_acknowledgement_keeps_the_detail_beat(ctx):
    gateway = Engine(streams=[""], wakes=[_wake("timeout", [_summary()])])
    gateway.repaired = False
    assert delegate_observe.step(ctx, gateway, "run-s", window=1.0, since_seq=0, read_window=lambda: 30.0) is None
    assert gateway.probes == [0] and gateway.stream_opens == [] and gateway.requests == []
