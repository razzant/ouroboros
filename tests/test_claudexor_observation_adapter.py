"""One observation adapter: the compact wait on a repaired engine, the detail read elsewhere."""

import pytest

from ouroboros.gateways import claudexor_observation as obs
from ouroboros.gateways import claudexor_run_events as run_events
from ouroboros.gateways.claudexor import ClaudexorUnavailable

ENGINE_TOKEN = "A" * 43
OTHER_TOKEN = "B" * 43
WAIT_MANY_ROW = {"method": "POST", "path": "/v2/runs/wait-many"}
PROBE = [{"runId": obs.PROBE_RUN_ID, "seen": None}]
# The contract probe's answers: a repaired engine places no such run; an engine without the
# repair refuses ``seen: null`` at its schema.
ACCEPTED = {"wakeReason": "timeout", "attention": [],
            "summaries": [{"status": None, "pending": None, "token": None, "problem": "not_found"}]}
REJECTED = ClaudexorUnavailable("invalid_request", "targets[0].seen: expected string", status_code=400)


class FakeGateway:
    """The two surfaces the adapter may touch, scripted per test; ``probe`` answers the contract probe."""

    def __init__(self, *, version="4.8.1", sha="sha-1", routes=(WAIT_MANY_ROW,), waits=(), details=None,
                 probe=ACCEPTED):
        self.engine_version = version
        self.engine_build_sha = sha
        self.routes = list(routes)
        self.waits = list(waits)
        self.details = dict(details or {})
        self.probe = probe
        self.catalog_reads = 0
        self.probes = []
        self.wait_requests = []
        self.detail_reads = []

    def operations(self, *, timeout_sec=None):
        self.catalog_reads += 1
        return list(self.routes)

    def wait_run_attention(self, targets, *, wait_ms, timeout_sec):
        if targets == PROBE:
            self.probes.append({"wait_ms": wait_ms, "timeout_sec": timeout_sec})
            if isinstance(self.probe, BaseException):
                raise self.probe
            return self.probe
        self.wait_requests.append({"targets": targets, "wait_ms": wait_ms, "timeout_sec": timeout_sec})
        answer = self.waits.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def get_run(self, run_id, *, timeout_sec=None):
        self.detail_reads.append((run_id, timeout_sec))
        answer = self.details[run_id]
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def _fresh_support_cache(monkeypatch):
    monkeypatch.setattr(obs, "_SUPPORT", {})
    monkeypatch.setattr(run_events, "_SUPPORT", {})


def _summary(status="running", pending=False, token=ENGINE_TOKEN):
    return {"status": status, "pending": pending, "token": token, "problem": None}


def _problem(kind):
    return {"status": None, "pending": None, "token": None, "problem": kind}


# -- support ------------------------------------------------------------------------------

def test_support_is_the_catalog_and_the_contract_probe_remembered_per_engine_identity():
    unlisted = FakeGateway(sha="sha-unlisted", routes=())
    assert obs.wait_many_supported(unlisted, timeout_sec=1) is False
    assert unlisted.probes == [], "an engine that lists no route is not probed"
    repaired = FakeGateway()
    assert obs.wait_many_supported(repaired, timeout_sec=1) is True
    assert obs.wait_many_supported(repaired, timeout_sec=1) is True
    assert repaired.catalog_reads == 1 and repaired.probes == [{"wait_ms": 0, "timeout_sec": 1}], \
        "one catalog read and one zero-wait probe per engine identity"
    same_identity = FakeGateway(probe=REJECTED)
    assert obs.wait_many_supported(same_identity, timeout_sec=1) is True and same_identity.probes == []


@pytest.mark.parametrize("answer", [
    REJECTED,
    ClaudexorUnavailable("bad_request", "unknown field seen", status_code=400),
    ClaudexorUnavailable("not_found", "no such route", status_code=404),
    {"wakeReason": "timeout", "attention": [], "summaries": [_summary()]},  # a 200 of another shape
    {"wakeReason": "problems", "attention": [], "summaries": []},
], ids=["invalid_request", "other-400", "route-404", "old-shape-row", "old-shape-empty"])
def test_a_received_refusal_of_the_acknowledgement_is_unsupported_and_remembered(answer):
    gateway = FakeGateway(sha="sha-unrepaired", probe=answer)
    assert obs.wait_many_supported(gateway, timeout_sec=1) is False
    assert obs.wait_many_supported(gateway, timeout_sec=1) is False
    assert len(gateway.probes) == 1 and gateway.catalog_reads == 1, "a received answer is remembered"


@pytest.mark.parametrize("hole", [
    ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                         observation_reason="observation_read_timeout"),
    ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503),
    ClaudexorUnavailable("daemon_stopping", "no new product request was admitted", status_code=503),
])
def test_a_hole_on_the_probe_leaves_support_unknown_and_is_asked_again(hole):
    gateway = FakeGateway(sha="sha-hole", probe=hole)
    assert obs.wait_many_supported(gateway, timeout_sec=1) is False, "this beat reads as before"
    assert obs._SUPPORT == {}, "nothing is remembered of a hole"
    gateway.probe = ACCEPTED
    assert obs.wait_many_supported(gateway, timeout_sec=1) is True
    assert len(gateway.probes) == 2 and gateway.catalog_reads == 1, "the next beat asked the contract again"


def test_a_release_number_decides_nothing_about_the_route():
    claims = FakeGateway(version="4.8.1", sha="sha-claims", probe=REJECTED)
    assert obs.wait_many_supported(claims, timeout_sec=1) is False, "lists the route, serves it unrepaired"
    predates = FakeGateway(version="4.8.0", sha="sha-predates")
    assert obs.wait_many_supported(predates, timeout_sec=1) is True, "the contract, whatever the number"
    assert not hasattr(obs, "WAIT_MANY_MIN_VERSION")


def _retryable(code="store_closed"):
    exc = ClaudexorUnavailable(code, "the store is closed for maintenance", status_code=503)
    exc.problem = {"code": code, "message": "the store is closed for maintenance", "retryable": True}
    return exc


@pytest.mark.parametrize("refusal", [
    ClaudexorUnavailable("http_401", "unauthorized", status_code=401),
    ClaudexorUnavailable("conflict", "another writer", status_code=409),
    _retryable("store_closed"),
    ClaudexorUnavailable("idempotency_status_unavailable", "not now", status_code=503),
    ClaudexorUnavailable("http_500", "boom", status_code=500),
], ids=["401", "409", "503-retryable", "503-plain", "500"])
def test_any_other_refusal_of_the_probe_is_not_supported_this_beat_and_not_remembered(refusal):
    gateway = FakeGateway(sha="sha-refused", probe=refusal)
    assert obs.wait_many_supported(gateway, timeout_sec=1) is False, "the question never blocks the old read"
    assert obs._SUPPORT == {}, "nothing is remembered: the old read says what the refusal is"
    gateway.probe = ACCEPTED
    assert obs.wait_many_supported(gateway, timeout_sec=1) is True and len(gateway.probes) == 2, "asked again"


def test_a_retryable_refusal_is_a_hole_and_a_plain_one_is_not():
    assert obs.is_observation_hole(_retryable("store_closed")) and obs.is_observation_hole(_retryable("store_maintenance_failed"))
    plain = ClaudexorUnavailable("idempotency_status_unavailable", "not now", status_code=503)
    assert not obs.is_observation_hole(plain)
    not_503 = ClaudexorUnavailable("conflict", "another writer", status_code=409)
    not_503.problem = {"code": "conflict", "retryable": True}
    assert not obs.is_observation_hole(not_503), "the engine's retryable word counts on its 503s only"
    wake = obs.wait_attention(FakeGateway(waits=[_retryable()]), [("run-a", None)], wait_sec=1.0, timeout_sec=1.0)
    assert wake.reason == "unobserved" and wake.hole is not None, "a retryable refusal of the pass is a hole, not a verdict"


def test_a_double_without_the_seam_or_a_handshake_is_unsupported_without_a_request():
    class Bare:
        engine_version = "9.9.9"

        def get_run(self, run_id, **_kw):
            return {}

    assert obs.wait_many_supported(Bare(), timeout_sec=1) is False
    assert obs.wait_many_supported(FakeGateway(version=""), timeout_sec=1) is False


# -- the engine path -----------------------------------------------------------------------

def test_engine_rows_and_attention_are_projected_and_only_engine_tokens_are_echoed():
    gateway = FakeGateway(waits=[{
        "wakeReason": "attention", "attention": [1],
        "summaries": [_summary(), _summary(pending=True, token=OTHER_TOKEN), _problem("not_found")],
    }])
    wake = obs.wait_attention(gateway, [("run-a", "local:abc"), ("run-b", ENGINE_TOKEN), ("run-c", None)],
                              wait_sec=99.0, timeout_sec=5.0)
    request = gateway.wait_requests[0]
    assert request["targets"] == [{"runId": "run-a"}, {"runId": "run-b", "seen": ENGINE_TOKEN}, {"runId": "run-c"}]
    assert request["wait_ms"] == 50_000, "the window is clamped to the engine's maximum"
    assert request["timeout_sec"] == 5.0
    assert wake.reason == "attention" and wake.waited is True and wake.source == "wait_many"
    assert wake.attention == [1] and wake.needs_attention(1) and not wake.needs_attention(0)
    assert [row.run_id for row in wake.rows] == ["run-a", "run-b", "run-c"]
    assert wake.rows[1].pending is True and wake.rows[1].token == OTHER_TOKEN and wake.rows[1].observed
    assert wake.rows[2].problem == "not_found" and not wake.rows[2].observed and wake.rows[2].token is None


@pytest.mark.parametrize("body", [
    {"wakeReason": "attention", "attention": [], "summaries": [_summary()]},  # batch of the wrong size
    {"wakeReason": None, "attention": [], "summaries": [_summary(), _summary()]},
    {"wakeReason": "attention", "attention": [5], "summaries": [_summary(), _summary()]},
    {"wakeReason": "timeout", "attention": [], "summaries": [_summary(token="short"), _summary()]},
    {"wakeReason": "problems", "attention": [], "summaries": [_problem(""), _summary()]},
    {"wakeReason": "problems", "attention": [], "summaries": [_problem(7), _summary()]},
])
def test_a_malformed_engine_answer_is_a_typed_refusal_never_a_fabricated_row(body):
    gateway = FakeGateway(waits=[body])
    with pytest.raises(ClaudexorUnavailable) as caught:
        obs.wait_attention(gateway, [("run-a", None), ("run-b", None)], wait_sec=1.0, timeout_sec=1.0)
    assert caught.value.code == "malformed_response"


def test_an_unknown_wake_reason_or_problem_kind_is_attention_never_a_refusal():
    later = FakeGateway(waits=[{"wakeReason": "later", "attention": [], "summaries": [_summary(), _summary()]}])
    wake = obs.wait_attention(later, [("run-a", None), ("run-b", None)], wait_sec=1.0, timeout_sec=1.0)
    assert wake.reason == "later" and wake.attention == [0, 1], "unknown is not no: every row gets one addressed read"
    assert wake.waited is False, "no hold is assumed of an answer this adapter cannot read"
    vanished = FakeGateway(waits=[{"wakeReason": "problems", "attention": [], "summaries": [_problem("vanished")]}])
    wake = obs.wait_attention(vanished, [("run-a", None)], wait_sec=1.0, timeout_sec=1.0)
    assert wake.rows[0].problem == "vanished" and not wake.rows[0].observed and wake.waited is False


def test_a_problem_answer_is_no_hold_and_its_acknowledgement_rides_the_wire_as_null():
    gateway = FakeGateway(waits=[{"wakeReason": "problems", "attention": [], "summaries": [_problem("snapshot_unavailable")]},
                                 {"wakeReason": "timeout", "attention": [], "summaries": [_summary()]}])
    first = obs.wait_attention(gateway, [("run-a", None)], wait_sec=25.0, timeout_sec=1.0)
    assert first.reason == "problems" and first.waited is False, "the engine answers an unacknowledged problem at once"
    again = obs.wait_attention(gateway, [("run-a", obs.ACKNOWLEDGED_PROBLEM)], wait_sec=25.0, timeout_sec=1.0)
    assert gateway.wait_requests[1]["targets"] == [{"runId": "run-a", "seen": None}]
    assert again.waited is True


@pytest.mark.parametrize("wait_sec,reason,waited", [(0.0, "timeout", False), (5.0, "aborted", False),
                                                     (5.0, "timeout", True), (None, "attention", True)])
def test_only_a_held_call_counts_as_waited(wait_sec, reason, waited):
    gateway = FakeGateway(waits=[{"wakeReason": reason, "attention": [0] if reason == "attention" else [],
                                  "summaries": [_summary(pending=True)]}])
    wake = obs.wait_attention(gateway, [("run-a", None)], wait_sec=wait_sec, timeout_sec=1.0)
    assert wake.waited is waited


def test_a_pass_over_more_targets_than_the_engine_takes_is_chunked_and_holds_past_no_notice():
    first = {"wakeReason": "attention", "attention": [3],
             "summaries": [_summary()] * 3 + [_summary(pending=True, token=OTHER_TOKEN)] + [_summary()] * 996}
    second = {"wakeReason": "timeout", "attention": [], "summaries": [_problem("not_found"), _summary()]}
    gateway = FakeGateway(waits=[first, second])
    targets = [(f"run-{i}", None) for i in range(1002)]
    wake = obs.wait_attention(gateway, targets, wait_sec=25.0, timeout_sec=1.0)
    sizes = [len(request["targets"]) for request in gateway.wait_requests]
    assert sizes == [obs.WAIT_MANY_MAX_TARGETS, 2]
    assert gateway.wait_requests[0]["wait_ms"] == 25_000 and gateway.wait_requests[1]["wait_ms"] == 0
    assert len(wake.rows) == 1002 and wake.attention == [3] and wake.reason == "attention" and wake.waited is True
    assert wake.rows[1000].problem == "not_found" and wake.rows[3].pending is True


@pytest.mark.parametrize("failure", [
    ClaudexorUnavailable("daemon_unreachable", "read bound expired", observation_timeout=True,
                         observation_reason="observation_read_timeout"),
    ClaudexorUnavailable("daemon_stopping", "no new product request was admitted", status_code=503),
    ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503),
])
def test_a_hole_is_an_unobserved_wake_not_a_raise(failure):
    gateway = FakeGateway(waits=[failure])
    wake = obs.wait_attention(gateway, [("run-a", None)], wait_sec=1.0, timeout_sec=1.0)
    assert wake.reason == "unobserved" and wake.hole is failure and wake.waited is False
    assert [row.observed for row in wake.rows] == [False]


def test_a_received_refusal_of_the_read_itself_propagates():
    gateway = FakeGateway(waits=[ClaudexorUnavailable("invalid_request", "targets required", status_code=400)])
    with pytest.raises(ClaudexorUnavailable) as caught:
        obs.wait_attention(gateway, [("run-a", None)], wait_sec=1.0, timeout_sec=1.0)
    assert caught.value.status_code == 400


def test_a_404_on_the_route_falls_back_to_detail_in_the_same_call_and_names_no_run_absent():
    detail = {"summary": {"state": "running"}, "pendingInteractions": [], "lastSeq": 4}
    gateway = FakeGateway(waits=[ClaudexorUnavailable("not_found", "no such route", status_code=404)],
                          details={"run-a": detail})
    wake = obs.wait_attention(gateway, [("run-a", None)], wait_sec=1.0, timeout_sec=2.0)
    assert wake.source == "detail" and wake.rows[0].observed and wake.rows[0].detail is detail
    assert obs.wait_many_supported(gateway, timeout_sec=1) is False, "the route's absence is remembered"
    assert gateway.wait_requests and gateway.detail_reads == [("run-a", 2.0)]


# -- the fallback path ---------------------------------------------------------------------

def _pending(interaction_id):
    return {"interactionId": interaction_id, "questions": [
        {"id": "q1", "question": "Which?", "options": [], "multi_select": False}]}


def test_detail_fallback_derives_rows_and_attention_and_keeps_todays_beat():
    live = {"summary": {"state": "running"}, "pendingInteractions": [], "lastSeq": 1}
    asked = {"summary": {"state": "running", "waitingOnUser": True}, "pendingInteractions": [_pending("i-1")]}
    done = {"summary": {"state": "succeeded"}, "pendingInteractions": [], "lastSeq": 9}
    gateway = FakeGateway(probe=REJECTED, details={"run-a": live, "run-b": asked, "run-c": done})
    wake = obs.wait_attention(gateway, [("run-a", None), ("run-b", None), ("run-c", None)],
                              wait_sec=3.0, timeout_sec=5.0)
    assert wake.source == "detail" and wake.waited is False and wake.reason == "attention"
    assert wake.attention == [1, 2]
    assert [row.detail for row in wake.rows] == [live, asked, done]
    assert wake.rows[0].status == "running" and wake.rows[0].pending is False
    assert wake.rows[1].pending is True and wake.rows[2].terminal
    assert all(row.token.startswith("local:") for row in wake.rows), "fallback tokens never look like the engine's"
    assert gateway.wait_requests == [] and len(gateway.detail_reads) == 3


def test_fallback_tokens_follow_the_compact_facts_so_seen_suppresses_a_known_question():
    asked = {"summary": {"state": "running", "waitingOnUser": True}, "pendingInteractions": [_pending("i-1")]}
    gateway = FakeGateway(probe=REJECTED, details={"run-b": asked})
    first = obs.wait_attention(gateway, [("run-b", None)], wait_sec=0, timeout_sec=1.0)
    assert first.attention == [0]
    token = first.rows[0].token
    again = obs.wait_attention(gateway, [("run-b", token)], wait_sec=0, timeout_sec=1.0)
    assert again.attention == [] and again.rows[0].token == token
    gateway.details["run-b"] = {**asked, "pendingInteractions": [_pending("i-2")]}
    replaced = obs.wait_attention(gateway, [("run-b", token)], wait_sec=0, timeout_sec=1.0)
    assert replaced.attention == [0] and replaced.rows[0].token != token


def test_fallback_keeps_a_runs_own_refusal_on_its_row_and_a_hole_for_the_whole_batch():
    gone = ClaudexorUnavailable("run_not_found", "no such run", status_code=404)
    gateway = FakeGateway(probe=REJECTED, details={
        "run-a": {"summary": {"state": "running"}, "pendingInteractions": []}, "run-b": gone})
    wake = obs.wait_attention(gateway, [("run-a", None), ("run-b", None)], wait_sec=0, timeout_sec=1.0)
    assert wake.reason == "problems" and wake.rows[1].problem == "not_found" and wake.rows[1].error is gone
    assert wake.rows[0].observed
    hole = ClaudexorUnavailable("daemon_unreachable", "socket refused", observation_timeout=True,
                                observation_reason="daemon_unreachable")
    gateway.details["run-a"] = hole
    unobserved = obs.wait_attention(gateway, [("run-a", None), ("run-b", None)], wait_sec=0, timeout_sec=1.0)
    assert unobserved.reason == "unobserved" and unobserved.hole is hole and unobserved.source == "detail"
