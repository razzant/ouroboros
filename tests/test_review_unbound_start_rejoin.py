"""A delegated reviewer whose start reply was lost after its author left only its own wait:
the engine accepted the run, the host never bound it, and the seat's in-rail retry under the
same key found no daemon, so the seat settled in flight with its token.

The left wave's notifier then read that seat cold. An attach-only read cannot bind an unbound start, but the
seat's own row still names it and its exact request is retained: while its author lives, that author's
collection (``review_reference``) replays the request under its original key and binds the accepted run.
The notifier must leave the record owed instead of settling it as failed with an "author's task ended"
cause; once the author really ended, the same seat settles as unknown under that true cause, also when it
ended after reading that notice (the maintenance pass, once no review operation of the task is live). Only
the engine (idempotent admission, one lost reply) and the daemon's availability are scripted here; the
wave, custody, attempt rows, ledger, notifier and collection are real.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import review_custody, review_operation, review_pause
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.late_review_marks import replay_owed_late_reviews
from ouroboros.owner_mailbox import write_owner_message
from ouroboros.review_ledger import ledger_root, load_record
from ouroboros.review_state import load_state
from ouroboros.task_results import write_task_result
from tests._review_session_route_shared import FakeGateway, _owned_gateway_uses_each_test_transport  # noqa: F401
from tests._review_session_route_shared import fake_route as _fake_route
from tests.test_receptive_commit_review import TASK, _join, _notices, _parent, _produce, _read_mail, until
from tests.test_receptive_commit_review import author as _author
from tests.test_receptive_session_seat_checkpoint import SESSION_ANSWER, _payload, _posts, _review_change, _session_row
from tests.test_receptive_session_seat_checkpoint import session_wave as _session_wave
from tests.test_review_operation_collection import forbid_effects as _forbid_effects

author, fake_route, session_wave, forbid_effects = _author, _fake_route, _session_wave, _forbid_effects
pytestmark = pytest.mark.serial
AUTHOR_ENDED = "no retained outcome when its author's task ended"
UNBOUND = "A start was requested but no run was ever bound to it; its outcome stays unknown and it is never re-posted."
# The maintenance pass's report over the one owed left-wave duty.
WAITING = {"settled": 0, "waiting": 1, "retired": 0, "failed": 0}
SETTLED = {"settled": 1, "waiting": 0, "retired": 0, "failed": 0}
NOTHING_OWED = {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}


class _Engine:
    """Run admission as the host sees it: one run per Idempotency-Key, an exact replay answers that run and
    the same key with another body is refused. The first reply is lost after its run was accepted, and the
    daemon then misses the host's next ensure until the test restores it."""

    def __init__(self):
        self.runs, self.posts, self.reply_lost, self.daemon_down = {}, [], False, threading.Event()
        self.on_lost_reply = lambda: None

    def start_run(self, gateway, request, *, idempotency_key=""):
        gateway.start_requests.append(dict(request))
        key, body = str(idempotency_key), json.dumps(request, sort_keys=True, default=str)
        self.posts.append((key, body))
        accepted, run_id = self.runs.setdefault(key, (body, f"run-{len(self.runs) + 1}"))
        if accepted != body:
            raise ClaudexorUnavailable("idempotency_conflict", "same key, another request", status_code=409)
        if not self.reply_lost:
            self.reply_lost = True
            self.daemon_down.set()
            self.on_lost_reply()
            raise ClaudexorUnavailable("daemon_unreachable", "Claudexor control POST /v2/runs failed after 60.000s: "
                                       "ReadTimeout", observation_timeout=True,
                                       observation_reason="observation_read_timeout")
        return {"runId": run_id, "runDir": "/tmp/fake-run"}


@pytest.fixture
def lost_reply(session_wave, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    engine = _Engine()
    monkeypatch.setattr(FakeGateway, "start_run", lambda self, request, *, idempotency_key="": engine.start_run(
        self, request, idempotency_key=idempotency_key))

    def ensure():
        if engine.daemon_down.is_set():
            raise ClaudexorUnavailable("daemon_not_answering", "owned daemon process is alive but did not answer "
                                       "in this 20.0s wait; retry joins the same daemon", status_code=503)
        return gateway_module.ClaudexorGateway()
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", ensure)
    real_left_seats = review_pause._left_seats

    def after_release(record):  # the incident's order: the cold read follows the worker's release
        until(lambda: not review_custody._ACTIVE)
        return real_left_seats(record)
    monkeypatch.setattr(review_pause, "_left_seats", after_release)
    return session_wave, engine


def _leave(a):
    """The author leaves on mail while the delegated seat still prepares its first launch."""
    first = {}
    thread = _review_change(a, first)
    assert a.reviewer.sent["m1"].wait(60) and a.preparing.wait(60)
    write_owner_message(a.data, "Is the review done?", TASK)
    _join(thread)
    assert first["result"].startswith("An addressed message reached you while the reviewers ran"), first
    return _payload(first["result"])["review_reference"]


def _reissue(a, reference):
    again = {}

    def collect():
        from ouroboros.tools.review_change import _handle_review_change
        with _parent(a):
            again["result"] = _handle_review_change(a.ctx, review_reference=reference)
    return _produce(a, collect), again


def _seat(a, reference, slot_id="second"):
    record = load_record(ledger_root(a.ctx), reference["record_id"])
    return record, {row["seat_id"]: row for row in record["rows"]}[slot_id]


def test_a_lost_start_reply_stays_owed_until_its_living_author_binds_the_accepted_run(lost_reply):
    a, engine = lost_reply
    reference = _leave(a)
    a.reviewer.release()
    a.release.set()  # the seat starts; the engine accepts the run, its reply is lost and the retry finds no daemon
    [notice] = until(lambda: _notices(a, "change"))
    token = _session_row(a)["pending_invocation_id"]
    assert token and [key for key, _body in engine.posts] == [token], "one accepted start; the notifier sent nothing"
    assert custody.invocation_record(a.data, token)["state"] == "pending"
    record, seat = _seat(a, reference)
    assert record["state"] == "pending", (seat.get("operation_state"), seat.get("reported_cause"))
    assert seat["operation_state"] != "custody_lost"
    assert not any(AUTHOR_ENDED in str(row.get("reported_cause") or "") for row in record["rows"])
    assert notice["text"].startswith("[Review custody]") and "review_reference" in notice["text"], notice["text"]

    engine.daemon_down.clear()
    _read_mail(a)
    collector, again = _reissue(a, reference)
    _join(collector)
    collected = _payload(again["result"])
    assert (collected["state"], collected["aggregate"]) == ("settled", "PASS"), again["result"]
    assert [key for key, _body in engine.posts] == [token, token], "only the exact start, under its original key"
    assert len({body for _key, body in engine.posts}) == 1 and list(engine.runs) == [token], "one accepted run"
    bound = custody.invocation_record(a.data, token)
    assert (bound["state"], bound["run_id"]) == ("started", engine.runs[token][1]), "the original run is bound"
    assert SESSION_ANSWER in str(_session_row(a).get("raw_text") or ""), "the original run's late answer collects"


def test_an_ended_author_settles_the_same_unbound_start_as_unknown_under_that_true_cause(lost_reply):
    a, engine = lost_reply
    reference = _leave(a)
    # The author's task ends after the seat's start was sent and before its notifier reads it: nobody collects it.
    engine.on_lost_reply = lambda: write_task_result(a.data, TASK, "completed", chat_id=7, result="Finished.")
    a.reviewer.release()
    a.release.set()
    until(lambda: _seat(a, reference)[0]["state"] == "settled")
    record, seat = _seat(a, reference)
    assert seat["operation_state"] == "custody_lost" and seat["reported_cause"] == AUTHOR_ENDED
    assert {row["seat_id"]: row["status"] for row in record["rows"]}["first"] == "responded"
    assert [key for key, _body in engine.posts] == [_session_row(a)["pending_invocation_id"]], "nothing re-sent"


def _producer(a):
    """The delegated seat's completed producer receipt, read as the maintenance pass reads it."""
    attempt = load_state(a.data).attempts[-1]
    kind, _slot_id, row = next(item for item in review_pause._left_rows(attempt) if item[1] == "second")
    request, slot = review_operation.recorded_review_assignment(
        a.data, task_id=TASK, surface=kind, row=row, retry_key=attempt.review_retry_key,
        root_task_id=attempt.root_task_id, review_contract=attempt.review_contract_fingerprint)
    return review_operation.recover_review_producer(a.data, request, slot, row)


def _noticed_and_ended(a, reference):
    """The living author reads its custody notice (the record stays owed); the seat's worker and its
    operation end, leaving only its completed producer receipt, still in flight. Returns the seat's token."""
    a.reviewer.release()
    a.release.set()
    until(lambda: _notices(a, "change"))
    until(lambda: not review_custody._ACTIVE and not [op for op in review_operation._LIVE.values() if op.task_id == TASK])
    assert _seat(a, reference)[0]["state"] == "pending" and _producer(a).operation_state == "in_flight"
    return _session_row(a)["pending_invocation_id"]


def _operation_elsewhere(a):
    """A review operation of the task still live in another process, as a rejoin of the same start
    records it before it sends: its durable pointer names that process and the seat's operation."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.utils import utc_now_iso

    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    pointer = {"surface": "change", "retry_key": load_state(a.data).attempts[-1].review_retry_key, "task_attempt": 1,
               "controller": {"pid": worker.pid, "birth": process_start_time(worker.pid), "session": "elsewhere"},
               "operations": {"second": _session_row(a)["operation_id"]}, "recorded_at": utc_now_iso(),
               "state": review_operation.OPERATION_DISPATCHED}
    review_operation._update_operations(a.data, TASK, lambda rows: {**rows, "review-operation-elsewhere": pointer},
                                        create=True)
    return worker


@pytest.mark.parametrize("rejoin", [None, "live_elsewhere"])
def test_an_author_ending_after_its_custody_notice_settles_the_unbound_start_once_nothing_live_can_bind_it(
        lost_reply, rejoin):
    """The living author read its custody notice and then ended without re-issuing, its controller process
    alive. Its seat's worker ended in flight and nobody can bind that start now: the maintenance pass
    settles the seat as unknown under that true cause at $0, never re-posted. A review operation of the
    task still live in another process (a rejoin of the same start) keeps the record owed until it ends."""
    a, engine = lost_reply
    reference = _leave(a)
    token = _noticed_and_ended(a, reference)
    worker = _operation_elsewhere(a) if rejoin else None
    try:
        write_task_result(a.data, TASK, "completed", chat_id=7, result="Finished without collecting.")
        if worker is not None:
            assert review_operation.task_has_live_review_operation(a.data, TASK)
            assert replay_owed_late_reviews(a.data) == WAITING
            record, seat = _seat(a, reference)
            assert record["state"] == "pending" and not seat.get("reported_cause"), seat
    finally:
        if worker is not None:
            worker.kill()
            worker.wait()
    assert replay_owed_late_reviews(a.data) == SETTLED
    record, seat = _seat(a, reference)
    assert record["state"] == "settled" and (seat["operation_state"], seat["reported_cause"]) == (
        "custody_lost", AUTHOR_ENDED)
    assert {row["seat_id"]: row["status"] for row in record["rows"]}["first"] == "responded"
    assert [key for key, _body in engine.posts] == [token], "the one accepted start, never re-posted"
    assert custody.invocation_record(a.data, token)["state"] == "pending", "nothing bound it"
    assert replay_owed_late_reviews(a.data) == NOTHING_OWED, "its duty is discharged"


def test_a_bound_run_still_running_keeps_the_ended_authors_record_owed_until_its_answer_collects(
        lost_reply, monkeypatch):
    """The same post-notice end, but the seat's own invocation is bound to the engine's accepted run
    (custody STARTED, as a worker that lost only its observation leaves it) and that run still runs. The
    maintenance pass reads it attach-only and keeps the record owed, never cancelling or re-posting it;
    once the run ends, its answer settles the record at $0."""
    from ouroboros import claudexor_daemon

    a, engine = lost_reply
    reference = _leave(a)
    token = _noticed_and_ended(a, reference)
    run_id = engine.runs[token][1]
    custody.emit(a.data, custody.STARTED, {"run_id": run_id, "invocation_id": token, "task_id": TASK,
                                           "surface": "change"})
    readers = []
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway",
                        lambda **_kw: readers.append(FakeGateway()) or readers[-1])
    FakeGateway.nonterminal = True
    write_task_result(a.data, TASK, "completed", chat_id=7, result="Finished without collecting.")
    assert replay_owed_late_reviews(a.data) == WAITING
    record, seat = _seat(a, reference)
    assert record["state"] == "pending" and not seat.get("reported_cause"), seat
    FakeGateway.nonterminal = False  # the run ends
    assert replay_owed_late_reviews(a.data) == SETTLED
    record, seat = _seat(a, reference)
    assert (record["state"], record["verdict"]["aggregate"], seat["status"]) == ("settled", "PASS", "responded")
    assert [reader.run_gets for reader in readers] == [[run_id], [run_id]]
    assert not [call for reader in readers for call in reader.start_requests + reader.cancels]
    assert [key for key, _body in engine.posts] == [token], "the one accepted start, never re-posted"
    assert replay_owed_late_reviews(a.data) == NOTHING_OWED


def test_a_reissue_while_the_start_reply_is_outstanding_waits_on_the_live_seat(session_wave, monkeypatch):
    a = session_wave
    posting, reply = threading.Event(), threading.Event()
    real_start = FakeGateway.start_run

    def held(self, request, *, idempotency_key=""):
        posting.set()
        assert reply.wait(60)
        return real_start(self, request, idempotency_key=idempotency_key)
    monkeypatch.setattr(FakeGateway, "start_run", held)
    reference = _leave(a)
    a.reviewer.release()
    a.release.set()
    assert posting.wait(60)  # START_REQUESTED and its checkpoint landed; the POST waits for its reply
    _read_mail(a)
    collector, again = _reissue(a, reference)
    collector.join(1.0)
    assert collector.is_alive(), "the re-issue waits on its live seat instead of settling it from a cold read"
    reply.set()
    _join(collector)
    collected = _payload(again["result"])
    assert (collected["state"], collected["aggregate"]) == ("settled", "PASS"), again["result"]
    assert len(_posts()) == 1, "the live seat's own start, never a second one"


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("unbound", [False, True])
def test_a_lost_seat_names_the_author_ended_cause_only_after_its_author_ended(tmp_path, final, unbound):
    from ouroboros.review_operation import START_UNBOUND
    from tests.test_review_ledger_settlement import _record
    from tests.test_review_seat_outcome import _body

    record = _record(tmp_path)
    usage = {"pending_invocation_id": "inv-b", **({"custody_state": START_UNBOUND} if unbound else {})}
    seats = [{"slot_id": "a", "parts": ["change", "coupling"], "operation_id": "a", "operation_state": "settled",
              "status": "responded", "raw_text": _body()},
             {"slot_id": "b", "parts": ["change", "coupling"], "operation_id": "b", "operation_state": "custody_lost",
              "status": "error", "late_result_pending": True, "failure_code": "review_custody_lost",
              "error": UNBOUND, "usage": usage}]
    settled = review_pause._settle_left_record({"record_id": record["record_id"], "drive_root": str(tmp_path)},
                                               seats, final=final)
    rows = {row["seat_id"]: row for row in settled["rows"]}
    assert rows["a"]["status"] == "responded"
    if unbound and not final:  # its living author's collection can still bind that exact start
        assert settled["state"] == "pending" and rows["b"]["operation_state"] == "in_flight"
        assert not rows["b"].get("reported_cause")
        return
    assert settled["state"] == "settled" and rows["b"]["operation_state"] == "custody_lost"
    assert rows["b"]["reported_cause"] == (AUTHOR_ENDED if final else UNBOUND)


@pytest.mark.parametrize("row_names", ["nothing", "its_start", "a_start_without_its_body"])
def test_only_a_start_its_own_row_names_with_a_retained_body_is_typed_as_unbound(tmp_path, forbid_effects,
                                                                                   monkeypatch, row_names):
    from ouroboros.review_operation import START_UNBOUND
    from tests.test_review_operation_collection import SURFACE, _collect, _ctx, _observe, _session_run, _started
    from tests.test_review_operation_collection import TASK as COLLECT_TASK

    calls = _observe(monkeypatch, {})
    if row_names == "a_start_without_its_body":
        custody.emit(tmp_path, custody.START_REQUESTED, {
            "invocation_id": "inv-op-session", "task_id": COLLECT_TASK, "operation_id": "op-session",
            "slot_id": "s", "surface": SURFACE, "root_task_id": COLLECT_TASK})
    else:
        _started(tmp_path)  # requested with its exact body, never bound
    run = _session_run()
    if row_names != "nothing":
        run["actors"][0]["pending_invocation_id"] = "inv-op-session"
    actor, fact = _collect(run, tmp_path, _ctx(tmp_path))
    assert (actor["operation_state"], fact) == ("custody_lost", "unavailable") and "never re-posted" in actor["error"]
    assert (actor["usage"].get("custody_state") == START_UNBOUND) is (row_names == "its_start")
    assert calls == [], "an attach-only read neither observes nor re-posts an unbound start"
