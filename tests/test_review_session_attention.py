"""The review session poller holds on the engine's compact wait and reads detail on a reason."""
from __future__ import annotations

import pytest

from ouroboros import review_execution as rx
from ouroboros.gateways import claudexor_observation as obs
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.gateways.claudexor import ClaudexorUnavailable

TOKEN_A = "a" * 43
TOKEN_B = "b" * 43


def _summary(status="running", pending=False, token=TOKEN_A):
    return {"status": status, "pending": pending, "token": token, "problem": None}


def _wake(reason, summaries, attention=()):
    return {"wakeReason": reason, "attention": list(attention), "summaries": summaries}


class Engine:
    engine_version = "4.8.1"
    engine_build_sha = "sha-review"
    routes = [{"method": "POST", "path": "/v2/runs/wait-many"}]
    repaired = True  # the route accepts the problem acknowledgement (the contract probe's answer)

    def __init__(self, wakes, details):
        self.wakes, self.details = list(wakes), list(details)
        self.requests, self.detail_reads, self.catalog, self.probes = [], 0, [], []

    def operations(self, *, timeout_sec=None):
        if self.catalog and isinstance(self.catalog[0], BaseException):
            raise self.catalog.pop(0)
        return list(self.routes)

    def wait_run_attention(self, targets, *, wait_ms, timeout_sec):
        if targets == [{"runId": obs.PROBE_RUN_ID, "seen": None}]:
            self.probes.append(wait_ms)
            if not self.repaired:
                raise ClaudexorUnavailable("invalid_request", "targets[0].seen: expected string", status_code=400)
            return _wake("timeout", [{"status": None, "pending": None, "token": None, "problem": "not_found"}])
        self.requests.append({"targets": targets, "wait_ms": wait_ms})
        answer = self.wakes.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get_run(self, run_id, *, timeout_sec=None):
        self.detail_reads += 1
        answer = self.details.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _Custody:
    @staticmethod
    def is_terminal(detail):
        return (detail.get("summary") or {}).get("state") in {"succeeded", "failed", "cancelled", "interrupted"}

    @staticmethod
    def summary_of(detail):
        return detail.get("summary") or {}


class RuleEngine(Engine):
    """A 4.8.1 engine double that follows run-wait-many.ts literally: a problem row wakes the
    call at once unless its ``seen`` is null; a pending or terminal row wakes it unless ``seen``
    is its token; otherwise the call holds (shortened here), and the run finishes during it."""

    def __init__(self):
        super().__init__(wakes=[], details=[])
        self.held, self.finished = 0, False

    def wait_run_attention(self, targets, *, wait_ms, timeout_sec):
        if targets == [{"runId": obs.PROBE_RUN_ID, "seen": None}]:
            return super().wait_run_attention(targets, wait_ms=wait_ms, timeout_sec=timeout_sec)
        self.requests.append({"targets": targets, "wait_ms": wait_ms})
        assert len(self.requests) <= 50, "busy loop: the slot never let the engine hold"
        target = targets[0]
        if not self.finished:
            problem = {"status": None, "pending": None, "token": None, "problem": "snapshot_unavailable"}
            if target.get("seen", "omitted") is not None:
                return _wake("problems", [problem])
            if not wait_ms:
                return _wake("timeout", [problem])
            self.held, self.finished = self.held + 1, True  # the hold (instant here) ends when the run finishes
        row = _summary(status="succeeded", token=TOKEN_B)
        return _wake("attention", [row], [0]) if target.get("seen") != TOKEN_B else _wake("timeout", [row])

    def get_run(self, run_id, *, timeout_sec=None):
        self.detail_reads += 1
        return DONE if self.finished else RUNNING


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(obs, "_SUPPORT", {})
    monkeypatch.setattr(run_events, "_SUPPORT", {})
    monkeypatch.setattr(rx, "_SESSION_POLL_SEC", 0.01)


RUNNING = {"summary": {"state": "running"}, "lastSeq": 1, "pendingInteractions": []}
DONE = {"summary": {"state": "succeeded"}, "lastSeq": 5, "pendingInteractions": [], "primaryOutput": {"text": "ok"}}


def test_the_slot_holds_on_the_engine_and_reads_detail_only_when_the_run_ends(tmp_path, monkeypatch):
    gateway = Engine(
        wakes=[_wake("timeout", [_summary()]), _wake("attention", [_summary(status="succeeded", token=TOKEN_B)], [0])],
        details=[RUNNING, DONE])
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE
    assert gateway.detail_reads == 2, "the first read of the slot and the one the terminal wake earned"
    assert [r["targets"] for r in gateway.requests] == [[{"runId": "run-r"}], [{"runId": "run-r"}]]
    assert slept == [], "the engine held the call; the slot slept nothing of its own"


def test_the_token_is_echoed_as_seen_only_after_its_detail_was_read(tmp_path, monkeypatch):
    unread = ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
    gateway = Engine(
        wakes=[_wake("attention", [_summary(pending=False, token=TOKEN_B)], [0]),
               _wake("attention", [_summary(status="succeeded", token=TOKEN_B)], [0])],
        details=[RUNNING, unread, DONE])
    monkeypatch.setattr(rx.time, "sleep", lambda s: None)
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE
    # The first wake's detail read was refused: its token was NOT acknowledged, so the next
    # request still carries no ``seen`` and the engine wakes the slot again for the same facts.
    assert gateway.requests[1]["targets"] == [{"runId": "run-r"}]


def test_a_hole_on_the_wait_leaves_the_detail_read_to_decide(tmp_path, monkeypatch):
    hole = ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                                observation_reason="observation_read_timeout")
    gateway = Engine(wakes=[hole], details=[RUNNING, DONE])
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE and gateway.detail_reads == 2
    assert slept, "an unheld beat sleeps the session's own interval before reading, as before"


def test_a_problem_row_is_read_once_acknowledged_and_the_engine_holds_again(tmp_path, monkeypatch):
    gateway = RuleEngine()
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE
    assert [r["targets"] for r in gateway.requests] == [[{"runId": "run-r"}], [{"runId": "run-r", "seen": None}]]
    assert slept == [rx._SESSION_POLL_SEC], "an answer about an unacknowledged problem is no hold: the slot slept its beat"
    assert gateway.held == 1 and gateway.detail_reads == 3, "initial read, the addressed read, the terminal"


def test_an_unanswered_addressed_read_does_not_spin_on_the_same_notice(tmp_path, monkeypatch):
    unread = ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
    gateway = Engine(
        wakes=[_wake("attention", [_summary(status="succeeded", token=TOKEN_B)], [0])] * 3,
        details=[RUNNING, unread, unread, DONE])
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE and len(gateway.requests) == 3
    assert slept == [rx._SESSION_POLL_SEC] * 2, "each unanswered read earned the slot's own beat before the next wait"


def test_a_catalog_hole_leaves_the_detail_read_to_decide(tmp_path, monkeypatch):
    gateway = Engine(wakes=[], details=[RUNNING, DONE])
    gateway.catalog = [ClaudexorUnavailable("daemon_unreachable", "catalog timeout", observation_timeout=True,
                                            observation_reason="observation_read_timeout")]
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE and gateway.requests == [] and slept, "unobserved this beat: the slot's own beat, then the detail"


def test_an_engine_that_lost_the_route_hands_its_own_detail_read_to_the_slot(tmp_path, monkeypatch):
    gateway = Engine(wakes=[ClaudexorUnavailable("not_found", "no such route", status_code=404)], details=[RUNNING, DONE])
    monkeypatch.setattr(rx.time, "sleep", lambda s: None)
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE and gateway.detail_reads == 2, "the slot's first read and the fallback's terminal one: never a third"


def test_an_engine_without_the_wait_keeps_the_old_sleep_and_read_beat(tmp_path, monkeypatch):
    class Old(Engine):
        repaired = False  # lists the route, rejects the acknowledgement: the release number is not asked

    gateway = Old(wakes=[], details=[RUNNING, RUNNING, DONE])
    slept = []
    monkeypatch.setattr(rx.time, "sleep", lambda s: slept.append(s))
    detail = rx._poll_session_terminal(gateway, _Custody(), tmp_path, object(), "run-r", 30.0)
    assert detail is DONE and gateway.requests == [] and gateway.detail_reads == 3 and len(slept) == 2
