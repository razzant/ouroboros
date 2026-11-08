"""Owner Batch4 (1A/3A/4A): the Continue of an interrupted root, through its REAL consumers.

``supervisor/continuation_admission.admit_continuation`` (behind ``POST
/api/tasks/{id}/continue``) is driven on an installed queue with durable task
results and a real owner mailbox: crash boundaries, replay after the
successor ended, refusals by typed cause, exact owner sources kept apart from
peer context, the mailbox retained until a Continue copied it, the stale card
pointer, and the held unknown writer.
"""

from __future__ import annotations

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._store_shared import delete_result, seed_result

NONCE = "press-0001-abcdef"


def _interrupted(tmp_path, task_id="pred-1", *, origin=None, reason_code="", **extra):
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, write_task_result

    from ouroboros.usage_admission import original_group_limit
    prior = original_group_limit(tmp_path, task_id)
    initial = {} if prior["source"] == "ledger_first_row" else {"billing_group": {
        "billing_group_id": task_id, "billing_group_limit_usd": 20.0,
        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}}
    write_task_result(tmp_path, task_id, STATUS_RUNNING, chat_id=7, project_id="", **initial,
                      origin_message_text="Write the Friday report",
                      origin_message_ref={"chat_id": 7, "client_message_id": "m-1"},
                      root_task_id=task_id, title="Friday report", deadline_at="2099-01-01T00:00:00+00:00")
    fields = {"cancel_origin": origin if origin is not None else {"source": "snapshot_restore",
                                                                   "reason": "server_shutdown"}}
    if reason_code:
        fields = {"reason_code": reason_code}
    write_task_result(tmp_path, task_id, STATUS_CANCELLED if not reason_code else "failed",
                      result="interrupted", **fields, **extra)


def _owner_mail(tmp_path, task_id="pred-1"):
    from ouroboros.owner_mailbox import (
        KIND_OWNER_TEXT, KIND_TASK_MESSAGE, acknowledge_transcript_entry, write_owner_message,
    )

    write_owner_message(tmp_path, "Also add the charts", task_id, msg_id="owner-2", kind=KIND_OWNER_TEXT)
    acknowledge_transcript_entry(tmp_path, task_id, {"msg_id": "owner-2"})
    write_owner_message(tmp_path, "Use last week's numbers", task_id, msg_id="owner-3", kind=KIND_OWNER_TEXT)
    write_owner_message(tmp_path, "peer says hi", task_id, msg_id="peer-1", kind=KIND_TASK_MESSAGE)


# ``accounting_wait_expired``: a running direct turn whose accounting wait outlived its idle window.
@pytest.mark.parametrize("reason_code", ["", "accounting_wait_expired"], ids=["restart", "accounting_expiry"])
def test_continue_admits_a_new_root_with_exact_owner_sources_and_replays_it_forever(tmp_path, monkeypatch, reason_code):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    returns = {"relay-1": {"relay_id": "relay-1", "source": {"chat_id": 1, "address": "held-row"}}}
    held = {"held-row": {"direction": "in", "chat_id": 1, "text": "quoted words"}}
    _interrupted(tmp_path, reason_code=reason_code, open_relays=returns, held_owner_messages=held)
    _owner_mail(tmp_path)

    first = admit_continuation("pred-1", action_nonce=NONCE)

    assert first["ok"] is True and first["replay"] is False and first["held"] is False
    successor = first["successor_task_id"]
    assert successor != "pred-1" and successor.startswith("pred-1-c")
    row = next(task for task in workers.PENDING if task["id"] == successor)
    assert row["chat_id"] == 7 and row["root_task_id"] == successor and row["deadline_at"].startswith("2099")
    assert row["predecessor_task_id"] == "pred-1"
    meta = row["metadata"]
    assert meta["continuation"]["owner_sources"]["open_relays"] == returns
    assert meta["continuation"]["owner_sources"]["held_owner_messages"] == held
    assert meta["objective_author"]["kind"] == "continuation"
    corpus = [(r["source"], r["content"]) for r in meta["owner_corpus"]]
    assert corpus == [("origin_message", "Write the Friday report"),
                      ("owner_mailbox", "Also add the charts"), ("owner_mailbox", "Use last week's numbers")]
    assert [p["text"] for p in meta["continuation"]["peer_context"]] == ["peer says hi"], \
        "peer mail is context, never an owner source"
    assert "Also add the charts" in row["text"] and "not an owner instruction" in row["text"]
    stored = load_task_result(tmp_path, successor)
    assert stored["status"] == "scheduled" and stored["continuation_admission"]["binding"]["action_nonce"] == NONCE
    claim = load_task_result(tmp_path, "pred-1")["continued_by"]
    assert claim["successor_task_id"] == successor and claim["state"] == "admitted"

    # The same press again: the SAME admission, no second task.
    again = admit_continuation("pred-1", action_nonce=NONCE)
    assert again["replay"] is True and again["successor_task_id"] == successor
    assert [task["id"] for task in workers.PENDING].count(successor) == 1
    # ...even after the successor ended.
    workers.PENDING[:] = []
    write_task_result(tmp_path, successor, "completed", result="done")
    late = admit_continuation("pred-1", action_nonce=NONCE)
    assert late["replay"] is True and late["status"] == "completed" and late["successor_task_id"] == successor
    # A NEW press on the stale card is answered with the accepted successor.
    other = admit_continuation("pred-1", action_nonce="another-press-99")
    assert other == {"ok": False, "error": "already_continued", "successor_task_id": successor}


def test_the_successor_seeds_its_owner_corpus_with_the_owners_words_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    ctx = SimpleNamespace(task_metadata=row["metadata"], _owner_directives=[])
    _initialize_owner_directives(ctx, [{"role": "user", "content": row["text"]}])
    assert [d["content"] for d in ctx._owner_directives] == ["Write the Friday report"], \
        "the host-composed work order never becomes an owner directive"


def test_the_opening_names_the_predecessors_tree_and_its_reader_without_counters_or_an_owner_door(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, swarm_fanout_rollup={"children": 0, "terminal": 0})
    for child, parent, status in (("child-a", "pred-1", "completed"), ("child-b", "pred-1", "failed"),
                                  ("grand-1", "child-a", "completed")):
        write_task_result(tmp_path, child, status, parent_task_id=parent, root_task_id="pred-1", result="its final")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    tree = next((line for line in row["text"].splitlines() if line.startswith("Its helper tree")), "")
    assert "child-a (completed), child-b (failed), grand-1 (completed, under child-a)" in tree, row["text"]
    assert "get_task_result(task_id=<id>)" in tree and "include_work_order_source=True" in tree
    assert "pred-1 (" not in tree and "swarm_fanout" not in row["text"], "no counters copied, the root not its own child"
    assert [(r["source"], r["content"]) for r in row["metadata"]["owner_corpus"]] == [
        ("origin_message", "Write the Friday report")]
    ctx = SimpleNamespace(task_metadata=row["metadata"], _owner_directives=[])
    _initialize_owner_directives(ctx, [{"role": "user", "content": row["text"]}])
    assert [d["content"] for d in ctx._owner_directives] == ["Write the Friday report"], "the tree opens no owner door"


@pytest.mark.parametrize("origin,reason_code,refusal", [
    ({"source": "http_single", "reason": "Stop now"}, "", "stopped_by_owner"),
    ({"source": "panic"}, "", "stopped_by_owner"),
    ({}, "", "interruption_cause_unrecorded"),
])
def test_continue_is_refused_after_a_stop_a_panic_or_an_unrecorded_cause(tmp_path, monkeypatch, origin,
                                                                        reason_code, refusal):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, origin=origin)
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    assert outcome["ok"] is False and outcome["error"] == refusal
    assert "continued_by" not in load_task_result(tmp_path, "pred-1") and workers.PENDING == []


@pytest.mark.parametrize("terminal,refusal", [
    ({"cancel_origin": {"source": "http_single", "reason": "Stop now"}}, "stopped_by_owner"),
    ({"deadline_at": "2000-01-01T00:00:00+00:00"}, "hard_limit_reached"),
    ({"status": "completed"}, "author_finished"),
], ids=["stop", "deadline", "answered"])
def test_an_accounting_expiry_never_outranks_a_stop_or_a_whole_work_bound(tmp_path, monkeypatch, terminal, refusal):
    """The new technical cause admits only what it names: a recorded owner Stop or Panic, the work's own
    reached deadline or money, and a finished answer still refuse, write no claim and queue nothing."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code="accounting_wait_expired")
    seed_result(tmp_path, "pred-1", {**load_task_result(tmp_path, "pred-1"), **terminal})
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    assert outcome["ok"] is False and outcome["error"] == refusal
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, "pred-1")


_STAMP = "delegated_custody_unreconciled"


@pytest.mark.parametrize("stored,admitted,cause_or_refusal", [
    ({"reason_code": "provider_failure", "delegated_runs_unreconciled": ["run-open"],
      "outcome_axes": {"execution": {"status": "infra_failed", "reason_code": "provider_failure"}}},
     True, "provider_failure"),
    ({"reason_code": _STAMP, "outcome_axes": {"execution": {"status": "infra_failed", "reason_code": "provider_failure"}}},
     True, "provider_failure"),
    ({"status": "completed", "reason_code": _STAMP,
      "outcome_axes": {"execution": {"status": "best_effort", "reason_code": "round_limit"}}}, True, "round_limit"),
    ({"reason_code": _STAMP, "cancel_origin": {"source": "http_single", "reason": "Stop now"},
      "outcome_axes": {"execution": {"status": "failed", "reason_code": "provider_failure"}}}, False, "stopped_by_owner"),
    ({"reason_code": _STAMP, "outcome_axes": {"execution": {"status": "failed", "reason_code": "budget_exhausted"}}},
     False, "hard_limit_reached"),
    ({"status": "completed", "reason_code": _STAMP,
      "outcome_axes": {"execution": {"status": "ok", "reason_code": "final_message"}}}, False, "author_finished"),
    ({"reason_code": _STAMP, "outcome_axes": {"execution": {"status": "failed"}}}, False, "interruption_cause_unrecorded"),
    ({"reason_code": _STAMP, "outcome_axes": {"execution": {"status": "failed", "reason_code": _STAMP}}},
     False, "interruption_cause_unrecorded"),
], ids=["new-provider-failure", "legacy-provider-failure", "legacy-technical-rail", "legacy-owner-stop",
        "legacy-hard-bound", "legacy-completed-author", "legacy-missing-cause", "legacy-stamp-read-back-as-cause"])
def test_continue_reads_a_custody_stamp_by_the_cause_beneath_it(tmp_path, monkeypatch, stored, admitted,
                                                                cause_or_refusal):
    """#1583: an open debt never was the cause. A new row keeps its own reason; an older row's
    custody stamp is read beneath, from its typed execution reason (never prose), and every
    guard still answers first: a Stop, a hard bound and a finished answer refuse, and a stamped
    row whose cause was never recorded still refuses, naming the stamp. History keeps its bytes."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code="provider_failure")
    seed_result(tmp_path, "pred-1", {**load_task_result(tmp_path, "pred-1"), **stored})
    before = load_task_result(tmp_path, "pred-1")
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    after = load_task_result(tmp_path, "pred-1")
    assert {key: after.get(key) for key in ("reason_code", "outcome_axes")} == {
        key: before.get(key) for key in ("reason_code", "outcome_axes")}, "no migration of the stored row"
    if not admitted:
        assert outcome["ok"] is False and outcome["error"] == cause_or_refusal
        assert outcome["cause"] == {"stopped_by_owner": "http_single", "hard_limit_reached": "budget_exhausted",
                                    "author_finished": "finished", "interruption_cause_unrecorded": _STAMP}[
            cause_or_refusal], "a refusal names the guard's own fact, or the stamp it could not see beneath"
        assert workers.PENDING == [] and "continued_by" not in after
        return
    assert outcome["ok"] is True and outcome["held"] is False
    row = next(task for task in workers.PENDING if task["id"] == outcome["successor_task_id"])
    assert row["metadata"]["continuation"]["cause"] == cause_or_refusal
    assert _STAMP not in row["text"], "the successor's work order names the real cause"


def test_a_legacy_stamped_root_with_a_live_writer_is_held_not_doubled(tmp_path, monkeypatch):
    """The cause beneath the stamp admits Continue; a writer that really remains unsettled still holds it."""
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code=_STAMP,
                 outcome_axes={"execution": {"status": "infra_failed", "reason_code": "provider_failure"}})
    assert load_task_result(tmp_path, "pred-1")["reason_code"] == _STAMP
    write_task_result(tmp_path, "pred-child", STATUS_RUNNING, chat_id=7)
    workers.RUNNING["pred-child"] = {"task": {"id": "pred-child", "root_task_id": "pred-1",
                                              "parent_task_id": "pred-1"}, "worker_id": 0, "attempt": 1}
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is True
    assert ack["blockers"] == [{"kind": "running_member", "task_id": "pred-child"}]
    assert [task["id"] for task in workers.PENDING] == [ack["successor_task_id"]]


def test_a_finished_answer_and_a_live_task_refuse_while_a_technical_limit_admits(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "done-1", "completed", chat_id=7, root_task_id="done-1")
    assert admit_continuation("done-1", action_nonce=NONCE)["error"] == "author_finished"
    write_task_result(tmp_path, "live-1", STATUS_RUNNING, chat_id=7, root_task_id="live-1")
    assert admit_continuation("live-1", action_nonce=NONCE)["error"] == "predecessor_live"
    _interrupted(tmp_path, "limit-1", reason_code="round_limit")
    assert admit_continuation("limit-1", action_nonce=NONCE)["ok"] is True


def test_the_continues_barriers_run_outside_q_and_its_record_is_written_with_the_row_still_queued(tmp_path, monkeypatch):
    """Q spans memory only: the successor is enqueued under Q, its snapshot published outside it, its result row
    written under Q again (the row revalidated still queued), its final publication outside Q once more."""
    from supervisor.continuation_admission import admit_continuation

    queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    seen = []
    real = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot",
                        lambda reason="": seen.append((reason, queue._queue_lock._is_owned())) or real(reason=reason))
    answer = admit_continuation("pred-1", action_nonce=NONCE)
    assert answer["ok"] is True
    barriers = [(reason, under_q) for reason, under_q in seen if reason.startswith("owner_continue")]
    assert barriers == [("owner_continue_admitted", False), ("owner_continue_committed", False)]


def test_a_stop_that_takes_the_row_between_the_holds_leaves_the_record_unwritten_and_the_answer_unconfirmed(tmp_path, monkeypatch):
    """The window the two holds open: a Stop's custody can pop the queued successor before its record is written.
    The admission writes no scheduled record over the Stop's, answers unconfirmed (the same nonce reads the
    recorded outcome), and the reservation is released."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    real = queue.persist_queue_snapshot

    def stop_during_the_barrier(reason=""):
        answered = real(reason=reason)
        if reason == "owner_continue_admitted":  # as a Stop's custody would, under Q, after the barrier's capture
            with queue._queue_lock:
                workers.PENDING[:] = [row for row in workers.PENDING if not str(row.get("id") or "").startswith("pred-1-")]
        return answered

    monkeypatch.setattr(queue, "persist_queue_snapshot", stop_during_the_barrier)
    answer = admit_continuation("pred-1", action_nonce=NONCE)
    assert answer["ok"] is False and answer["error"] == "continuation_unconfirmed" and answer["unconfirmed"]
    assert workers.PENDING == [] and load_task_result(tmp_path, answer["successor_task_id"]) is None
    assert queue.ADMISSION_RESERVATIONS == {}


def test_crash_after_the_claim_or_after_the_snapshot_converges_on_one_identity(tmp_path, monkeypatch):
    """Claim written, then the process died: the same nonce completes THAT admission.
    Queue row persisted but no result row: the same nonce recovers it, never twice.
    A failed snapshot refuses without a queue row and keeps the claim for the retry."""
    from ouroboros.task_results import load_task_result
    from supervisor import continuation_admission as admission
    from supervisor.continuation_admission import admit_continuation

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    real_persist = queue.persist_queue_snapshot
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": False)
    failed = admit_continuation("pred-1", action_nonce=NONCE)
    assert failed["ok"] is False and failed["error"] == "queue_snapshot_persist_failed"
    assert workers.PENDING == [] and load_task_result(tmp_path, failed["successor_task_id"]) is None
    assert load_task_result(tmp_path, "pred-1")["continued_by"]["state"] == "bound"
    assert admit_continuation("pred-1", action_nonce="different-press")["error"] == "already_continued"
    monkeypatch.setattr(queue, "persist_queue_snapshot", real_persist)
    retried = admit_continuation("pred-1", action_nonce=NONCE)
    assert retried["ok"] is True and retried["successor_task_id"] == failed["successor_task_id"]

    # Queue row present, result row lost: recovered as the SAME admission.
    successor = retried["successor_task_id"]
    delete_result(tmp_path, successor)
    recovered = admit_continuation("pred-1", action_nonce=NONCE)
    assert recovered["replay"] is True and recovered.get("recovered") is True
    assert [task["id"] for task in workers.PENDING].count(successor) == 1
    assert load_task_result(tmp_path, successor)["continuation_admission"]["binding"]["action_nonce"] == NONCE
    assert admission._replay(queue, "pred-1", "nonce-of-someone-else", successor)["error"] \
        == "continuation_identity_conflict"


@pytest.mark.parametrize("reason_code", ["", "accounting_wait_expired"], ids=["restart", "accounting_expiry"])
def test_an_unsettled_writer_holds_the_same_continue_until_a_fresh_check_releases_it(tmp_path, monkeypatch, reason_code):
    """A child of the interrupted root still RUNS: the Continue is admitted but
    HELD (no diagnostic successor, no second writer). Resume refuses while it
    runs; the child's terminal reconciles and releases the SAME task."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.continuation_admission import admit_continuation, release_settled_continuations
    from supervisor.events_budget import HOLD_CONTINUATION_WRITER, budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code=reason_code)
    write_task_result(tmp_path, "pred-child", STATUS_RUNNING, chat_id=7)
    workers.RUNNING["pred-child"] = {"task": {"id": "pred-child", "root_task_id": "pred-1",
                                              "parent_task_id": "pred-1"}, "worker_id": 0, "attempt": 1}

    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is True
    assert ack["blockers"] == [{"kind": "running_member", "task_id": "pred-child"}]
    held = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert budget_hold_fact(held)["reason"] == HOLD_CONTINUATION_WRITER
    refused = queue.resume_budget_paused_task(held["id"])
    assert refused["ok"] is False and refused["error"] == "predecessor_writers_unsettled"
    assert release_settled_continuations("pred-1") == []

    workers.RUNNING.pop("pred-child")
    assert release_settled_continuations("pred-1") == [held["id"]]
    assert budget_hold_fact(held) is None


def test_cleanup_retains_acknowledged_owner_rows_before_unlink(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import load_task_result
    from ouroboros.owner_continue import owner_sources

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    _owner_mail(tmp_path)
    assert settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert not _mailbox_path(tmp_path, "pred-1").exists()
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert not sources["gaps"]
    assert [r["content"] for r in sources["later"]] == ["Also add the charts", "Use last week's numbers"]


def test_an_unreadable_owner_mailbox_refuses_instead_of_inventing_owner_text(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import _mailbox_path
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    path = _mailbox_path(tmp_path, "pred-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"msg_id": "x", "kind": "owner_text", "text": "torn', encoding="utf-8")
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    assert outcome["ok"] is False and outcome["error"] == "owner_source_missing"
    assert outcome["gaps"] == ["owner_mailbox_unreadable"]


@pytest.mark.parametrize("reason_code", ["", "accounting_wait_expired"], ids=["restart", "accounting_expiry"])
def test_the_continue_endpoint_and_the_detail_offer(tmp_path, monkeypatch, reason_code):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_continue import api_task_continue
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, reason_code=reason_code)
    assert continuation_offer(load_task_result(tmp_path, "pred-1"), "pred-1")["eligible"] is True
    client = TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/continue", api_task_continue,
                                                methods=["POST"])]))
    assert client.post("/api/tasks/pred-1/continue", json={"action_nonce": NONCE, "x": 1}).status_code == 400
    assert client.post("/api/tasks/pred-1/continue", json={"action_nonce": "bad"}).status_code == 400
    ok = client.post("/api/tasks/pred-1/continue", json={"action_nonce": NONCE})
    assert ok.status_code == 200 and ok.json()["successor_task_id"].startswith("pred-1-c")
    again = client.post("/api/tasks/pred-1/continue", json={"action_nonce": "a-new-press-77"})
    assert again.status_code == 409 and again.json()["successor_task_id"] == ok.json()["successor_task_id"]
    offer = continuation_offer(load_task_result(tmp_path, "pred-1"), "pred-1")
    assert offer == {"eligible": False, "refusal": "already_continued",
                     "successor_task_id": ok.json()["successor_task_id"], "state": "admitted"}


def _corpus(*rows, unavailable=()):
    """The retained owner corpus as ``capture_task_inputs`` freezes it on the result."""
    return {"task_inputs": {"version": 1, "owner_requirements_and_decisions": list(rows),
                            "unavailable_sections": list(unavailable)}}


def _settled_with_corpus(tmp_path, *rows, unavailable=(), late=False):
    """Mailbox rows captured on the terminal row and cleaned; the corpus retained beside them."""
    from ouroboros.owner_mailbox import _mailbox_path, write_owner_message
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import write_task_result

    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    _owner_mail(tmp_path)
    if late:
        write_owner_message(tmp_path, "Option B, please", "pred-1", msg_id="late-1",
                            late_answer={"task_id": "asker-1", "quiz_id": "q-1"})
    assert settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    assert not _mailbox_path(tmp_path, "pred-1").exists()
    write_task_result(tmp_path, "pred-1", "cancelled", review_evidence=_corpus(*rows, unavailable=unavailable))


def test_the_retained_corpus_fallback_keeps_identity_provenance_and_dedup(tmp_path, monkeypatch):
    """Regression: the corpus fallback called ``seen.add`` on a dict and crashed every
    Continue whose terminal row retained an owner corpus. Now each branch obeys ONE
    identity rule: a msg_id names one text (a duplicate is skipped, a rival text is a
    gap), the run's first text never becomes a "later" owner message, a task's words
    are context, and a row of unknown provenance is never promoted to owner text."""
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    _settled_with_corpus(
        tmp_path,
        {"source": "initial_user", "content": "Write the Friday report"},  # the original's projection
        {"source": "owner_mailbox", "content": "Also add the charts", "msg_id": "owner-2"},  # captured twin
        {"source": "direct_incoming", "content": "And a summary slide", "msg_id": "cm-9"},  # only here
        {"source": "direct_incoming", "content": "And a summary slide"},  # same words, no id
        {"source": "principal_task_message", "content": "parent: numbers attached", "msg_id": "t-1",
         "source_task_id": "parent-x"},
    )
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")

    assert sources["gaps"] == []
    assert sources["original"]["content"] == "Write the Friday report"
    assert [(row["source"], row["content"]) for row in sources["later"]] == [
        ("owner_mailbox", "Also add the charts"), ("owner_mailbox", "Use last week's numbers"),
        ("direct_incoming", "And a summary slide")]
    assert sources["later"][0]["exact_row"], "the exact mailbox row outranks its corpus projection"
    assert [(row["source_task_id"], row["text"]) for row in sources["peer_context"]] == [
        ("", "peer says hi"), ("parent-x", "parent: numbers attached")]


def test_continue_carries_my_humans_confirmed_place_words_citable_without_promoting_them(tmp_path, monkeypatch):
    """Astra finding 3 (R1 MINOR-3): the words a focus heard from my human's confirmed account are provenance rows in
    its corpus. Continue carries them to the successor whole, with their id and reply target, shows them verbatim in
    its start text and seeds them so a cite verifies there; they owe no reply and are no order. A peer task's words
    and a forwarded line stay what they were: context, uncitable. The row keeps its provider identity through the
    Continue, so once my send receipt names it (my own post through my human's account), the successor's cite is
    refused, while my human's own message in the same words stays citable."""
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives, owner_input_revision
    from ouroboros.owner_continue import owner_sources, work_order_text
    from ouroboros.owner_words import cited_owner_basis
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    words, peer, forwarded = "Проверь, пожалуйста, экономику", "parent: numbers attached", "<@U2> fwd: Deploy the hotfix now"
    target = {"address": "platform message m-1 in this room", "author": "", "text": ""}
    _settled_with_corpus(  # a forward reaches a focus only inside its frame text, never as a provenance row
        tmp_path,
        {"source": "initial_text", "content": f"[FOCUS] frame\n\n{forwarded}"},
        {"source": "place_confirmed", "content": words, "carried": True, "msg_id": "chat 7 / e1", "reply_to": target,
         "conversation_id": "C7", "message_id": "1793991999.000001"},
        {"source": "principal_task_message", "content": peer, "msg_id": "t-1", "source_task_id": "parent-x"},
    )
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert sources["gaps"] == []
    assert sources["confirmed_place"] == [{"source": "place_confirmed", "content": words, "msg_id": "chat 7 / e1",
                                           "conversation_id": "C7", "message_id": "1793991999.000001", "reply_to": target}]
    assert [row["text"] for row in sources["peer_context"]] == ["peer says hi", peer]
    text = work_order_text("pred-1", "cancelled", sources, successor_task_id="succ-1")
    assert f"- chat 7 / e1: {words} (in reply to platform message m-1 in this room)" in text
    assert peer not in text and forwarded not in text
    admitted = admit_continuation("pred-1", action_nonce=NONCE)
    assert admitted["ok"], admitted
    task = next(t for t in workers.PENDING if t["id"] == admitted["successor_task_id"])
    assert words in task["text"] and task["metadata"]["continuation"]["owner_sources"]["confirmed_place"] == sources["confirmed_place"]
    ctx = SimpleNamespace(task_metadata=task["metadata"], _owner_directives=[], drive_root=tmp_path)
    _initialize_owner_directives(ctx, [{"role": "user", "content": task["text"]}])
    assert owner_input_revision(ctx) == 0, "carried provenance owes no reply and is no order"
    basis, error = cited_owner_basis(ctx, words)
    assert error == "" and basis == {"source": "place_owner", "content": words, "cited": words, "basis": "cited", "carried": True,
                                     "msg_id": "chat 7 / e1", "reply_to": target}
    for other in (peer, forwarded):
        assert cited_owner_basis(ctx, other)[0] is None, other
    from ouroboros import own_sends
    from ouroboros.loop_messages import _record_owner_directive
    from ouroboros.owner_words import PLACE_CONFIRMED, place_row_facts
    seeded = next(row for row in ctx._owner_directives if row.get("source") == PLACE_CONFIRMED)
    assert (seeded["conversation_id"], seeded["message_id"]) == ("C7", "1793991999.000001"), "a next Continue reads it"
    receipt = f'MsgID,Channel,Text\n1793991999.000001,C7,"{words}"\n'
    assert own_sends.record_result(tmp_path, {"annotations": {"readOnlyHint": False}}, receipt, via="mcp:slack/post")
    assert cited_owner_basis(ctx, words)[0] is None, "the receipt names it: my own post, no order of theirs"
    _record_owner_directive(ctx, source=PLACE_CONFIRMED, content=words, carried=True, msg_id="chat 7 / e2",
                            **place_row_facts("C7", {"message_id": "1793992000.000002"}))
    assert cited_owner_basis(ctx, words)[0]["msg_id"] == "chat 7 / e2", "their own message in the same words"


@pytest.mark.parametrize("row,gap", [
    ({"source": "owner_mailbox", "content": "Use LAST MONTH's numbers", "msg_id": "owner-3"},
     "owner_message_identity_conflict"),
    # Shapes the recorder never writes are gaps, not silently dropped or str()-promoted.
    ({"source": "owner_mailbox", "content": "   "}, "retained_owner_corpus_row_malformed"),
    # A task id beside an unknown label is not a task's provenance either.
    ({"source": "task_local", "content": "who said this", "source_task_id": "t-9"},
     "owner_source_provenance_unknown"),
])
def test_a_corpus_row_that_cannot_be_trusted_is_a_named_gap_and_admits_nothing(tmp_path, monkeypatch, row, gap):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _settled_with_corpus(tmp_path, row)
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    assert outcome == {"ok": False, "error": "owner_source_missing", "gaps": [gap]}
    assert "continued_by" not in load_task_result(tmp_path, "pred-1") and workers.PENDING == []


def test_a_failed_corpus_capture_is_a_gap_not_an_empty_corpus(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _settled_with_corpus(tmp_path, unavailable=("owner_requirements_and_decisions",))
    outcome = admit_continuation("pred-1", action_nonce=NONCE)
    assert outcome["error"] == "owner_source_missing" and outcome["gaps"] == ["owner_corpus_capture_unavailable"]


def test_a_late_answer_frame_is_the_projection_of_its_exact_owner_row_not_a_rival(tmp_path, monkeypatch):
    """The corpus keeps a late quiz answer as the model-facing card frame; the mailbox
    row keeps the owner's own words with typed ``late_answer`` provenance. Same id,
    different bytes, one message: the exact row is kept and nothing is refused."""
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _settled_with_corpus(tmp_path, {
        "source": "owner_mailbox", "msg_id": "late-1",
        "content": "[Late answer to a question asked by task asker-1, which had finished]\nOption B, please"},
        late=True)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    corpus = [(r["source"], r["content"]) for r in row["metadata"]["owner_corpus"]]
    assert corpus == [("origin_message", "Write the Friday report"), ("owner_mailbox", "Also add the charts"),
                      ("owner_mailbox", "Use last week's numbers"), ("owner_mailbox", "Option B, please")]


def test_a_chained_continue_reads_its_seeded_corpus_once_and_flags_a_rival(tmp_path, monkeypatch):
    """A Continue of a Continue: the chained sources are the authority, the seeded
    corpus rows (``origin_message`` + the carried owner rows) are their copies — each
    message appears once in the next work order; a rival text under a carried id is a gap."""
    from ouroboros.owner_continue import owner_sources, work_order_text
    from ouroboros.task_results import load_task_result, write_task_result

    _install_queue(tmp_path, monkeypatch)
    chained = {"original": {"source": "origin_message", "content": "Write the Friday report"},
               "later": [{"source": "owner_mailbox", "content": "Also add the charts", "msg_id": "owner-2"}]}
    seeded = [{"source": "origin_message", "content": "Write the Friday report"},
              {"source": "owner_mailbox", "content": "Also add the charts", "msg_id": "owner-2"},
              {"source": "owner_mailbox", "content": "Then email it to Ann", "msg_id": "owner-7"}]
    _interrupted(tmp_path, "succ-1", metadata={"continuation": {"owner_sources": chained}},
                 review_evidence=_corpus(*seeded))
    result = load_task_result(tmp_path, "succ-1")
    sources = owner_sources(tmp_path, result, "succ-1")
    assert sources["gaps"] == []
    assert [row["content"] for row in sources["later"]] == ["Also add the charts", "Then email it to Ann"]
    text = work_order_text("succ-1", "owner_restart", sources)
    assert text.count("Also add the charts") == 1 and text.count("Write the Friday report") == 1

    write_task_result(tmp_path, "succ-1", "cancelled", review_evidence=_corpus(
        {"source": "owner_mailbox", "content": "Also add the maps", "msg_id": "owner-2"}))
    assert owner_sources(tmp_path, load_task_result(tmp_path, "succ-1"), "succ-1")["gaps"] == [
        "owner_message_identity_conflict"]


def test_malformed_mailbox_chained_and_original_shapes_are_gaps(tmp_path, monkeypatch):
    """Every owner source branch refuses a shape its writer never produces instead of
    ``str()``-promoting it: a mailbox body that is not text, a chained source row that
    is not text, and an original owner message that is not text."""
    import json

    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.task_results import load_task_result, write_task_result

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    path = _mailbox_path(tmp_path, "pred-1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"msg_id": "odd", "kind": "owner_text", "text": {"b": 1}}) + "\n", encoding="utf-8")
    assert owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")["gaps"] == [
        "owner_mailbox_row_malformed"]
    path.unlink()

    chained = {"original": {"source": "origin_message", "content": "Write the Friday report"},
               "later": [{"source": "owner_mailbox", "content": ["not", "text"], "msg_id": "o-1"}]}
    _interrupted(tmp_path, "succ-2", metadata={"continuation": {"owner_sources": chained}})
    assert owner_sources(tmp_path, load_task_result(tmp_path, "succ-2"), "succ-2")["gaps"] == [
        "chained_owner_source_malformed"]

    write_task_result(tmp_path, "odd-origin", "cancelled", chat_id=7, root_task_id="odd-origin",
                      origin_message_text={"not": "text"}, origin_message_ref={"chat_id": 7},
                      cancel_origin={"source": "owner_restart"})
    assert owner_sources(tmp_path, load_task_result(tmp_path, "odd-origin"), "odd-origin")["gaps"] == [
        "original_owner_message_text"]


# --- Batch4 F4: the typed rail, not the status word, decides a technical limit ---

def _ADMITTED_GROUP(task_id):
    """The whole-work money group a root's task admission pins (as ``_interrupted`` writes it)."""
    return {"billing_group_id": task_id, "billing_group_limit_usd": 20.0,
            "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}


def _rail_result(tmp_path, monkeypatch, rail, *, reason="absolute_ceiling"):
    """Run a REAL forced-finalization rail and derive its terminal row the way the
    pipeline does (``derive_loop_outcome`` + ``_durable_terminal_status``)."""
    from types import SimpleNamespace

    from ouroboros.agent_task_pipeline import _durable_terminal_status
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION, derive_loop_outcome
    from tests.test_delivery_forced_finalization import _forced_test_context

    loop_root = tmp_path / f"loop-{rail}"
    loop_root.mkdir()
    loop, _registry, limit_ctx, _trace = _forced_test_context(loop_root)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (
        {"role": "assistant", "content": "Best answer from the verified work so far."}, 0.0))
    if rail == "round_limit":
        text, usage, trace = loop._handle_round_limit(limit_ctx)
    elif rail == "finalization_grace":
        text, usage, trace = loop._handle_forced_finalization(limit_ctx, reason)
    elif rail == "budget_exhausted":
        limit_ctx.round_idx = 1
        text, usage, trace = loop._check_budget_limits(limit_ctx, 0.0)
    else:  # deadline_local / owner_requested_finalization through the one forced-answer rail
        code = REASON_OWNER_REQUESTED_FINALIZATION if rail == "owner_wrap_up" else rail
        text, usage, trace = loop._forced_final_answer(limit_ctx, prompt="Finish now.", fallback_text="x",
                                                      reason_code=code)
    outcome = derive_loop_outcome(text, usage, trace)
    execution = outcome["outcome_axes"]["execution"]["status"]
    status = _durable_terminal_status(SimpleNamespace(drive_root=tmp_path), {"id": "x"}, execution, existing={})
    return status, outcome


@pytest.mark.parametrize("rail", ["round_limit", "finalization_grace"])
def test_a_technical_limit_that_still_answered_is_offered_continue(tmp_path, monkeypatch, rail):
    """F4: the real rail extracts a best-effort answer, so the root settles ``completed``
    under the rail's code — it used to be read as a finished answer and refused."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    status, outcome = _rail_result(tmp_path, monkeypatch, rail)
    assert (status, outcome["reason_code"]) == ("completed", rail)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    write_task_result(tmp_path, "limited-1", "running", chat_id=7, root_task_id="limited-1",
                      origin_message_text="Write the Friday report", origin_message_ref={"chat_id": 7},
                      deadline_at="2099-01-01T00:00:00+00:00", billing_group=_ADMITTED_GROUP("limited-1"))
    write_task_result(tmp_path, "limited-1", status, result=outcome["final_text"], reason_code=outcome["reason_code"],
                      outcome_axes=outcome["outcome_axes"])
    assert continuation_offer(load_task_result(tmp_path, "limited-1"), "limited-1") == {
        "eligible": True, "cause": rail, "refusal": ""}
    ack = admit_continuation("limited-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    successor = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert successor["deadline_at"].startswith("2099"), "the hard deadline travels; no new one is minted"


@pytest.mark.parametrize("rail,status,refusal", [
    ("deadline_local", "completed", "hard_limit_reached"),
    ("budget_exhausted", "failed", "hard_limit_reached"),
    ("owner_wrap_up", "completed", "author_finished"),
])
def test_the_works_own_deadline_money_and_wrap_up_are_never_offered_continue(tmp_path, monkeypatch, rail,
                                                                          status, refusal):
    from ouroboros.owner_continue import continuation_eligibility

    _install_queue(tmp_path, monkeypatch)
    derived, outcome = _rail_result(tmp_path, monkeypatch, rail)
    assert derived == status
    row = {"status": derived, "reason_code": outcome["reason_code"], "root_task_id": "r"}
    assert continuation_eligibility(row, "r")["refusal"] == refusal
    # The same hard bounds refuse on the failed (no answer) side too.
    if rail != "owner_wrap_up":
        assert continuation_eligibility({**row, "status": "failed"}, "r")["refusal"] == "hard_limit_reached"


def test_a_grace_rail_that_was_the_hard_deadline_refuses_and_the_reaper_rails_are_classified(tmp_path, monkeypatch):
    """``finalization_grace`` is also how the supervisor's DEADLINE grace ends a task:
    once the work's own hard deadline is reached the new id would not extend it.
    Every reaper rail (``queue_timeouts.TIMEOUT_TERMINAL_REASONS``) has a verdict."""
    from ouroboros.owner_continue import continuation_eligibility
    from supervisor.queue_timeouts import TIMEOUT_TERMINAL_REASONS

    _install_queue(tmp_path, monkeypatch)
    status, outcome = _rail_result(tmp_path, monkeypatch, "finalization_grace", reason="deadline")
    row = {"status": status, "reason_code": outcome["reason_code"], "root_task_id": "r"}
    assert continuation_eligibility({**row, "deadline_at": "2000-01-01T00:00:00+00:00"}, "r")["refusal"] \
        == "hard_limit_reached"
    assert continuation_eligibility({**row, "metadata": {"deadline_at": "2000-01-01T00:00:00Z"}}, "r")["refusal"] \
        == "hard_limit_reached"
    assert continuation_eligibility({**row, "deadline_at": "2099-01-01T00:00:00+00:00"}, "r")["eligible"] is True
    verdicts = {code: continuation_eligibility({"status": "failed", "reason_code": code, "root_task_id": "r"}, "r")
                for code in TIMEOUT_TERMINAL_REASONS}
    assert {code: (v["eligible"], v["refusal"]) for code, v in verdicts.items()} == {
        "absolute_ceiling": (True, ""), "idle_timeout": (True, ""), "deadline": (False, "hard_limit_reached")}


def test_history_replays_the_offer_so_a_collapsed_card_needs_no_opening(tmp_path, monkeypatch):
    """The chat history projection carries the host's ``continuation_offer`` beside the
    terminal truth: an interrupted root's replayed rows offer Continue, a continued
    one points to its successor, a finished one says no — without a detail read."""
    import asyncio
    import json
    from types import SimpleNamespace

    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "pred-1")
    _interrupted(tmp_path, "pred-2")
    write_task_result(tmp_path, "done-1", "completed", chat_id=7, root_task_id="done-1", result="done")
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").write_text("".join(json.dumps({
        "ts": f"2026-09-27T00:00:0{i}Z", "direction": "system", "type": "task_summary", "task_id": task_id,
        "chat_id": 7, "text": "summary"}) + "\n" for i, task_id in enumerate(("pred-1", "pred-2", "done-1"))),
        encoding="utf-8")
    (tmp_path / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    successor = admit_continuation("pred-2", action_nonce=NONCE)["successor_task_id"]

    endpoint = make_chat_history_endpoint(tmp_path)
    payload = json.loads(asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "20"}))).body)["messages"]
    offers = {row["task_id"]: row.get("continuation_offer") for row in payload if row.get("task_id")}
    assert offers["pred-1"] == {"eligible": True, "cause": "snapshot_restore", "refusal": ""}
    assert offers["pred-2"]["successor_task_id"] == successor and offers["pred-2"]["eligible"] is False
    assert offers["done-1"]["refusal"] == "author_finished"


@pytest.mark.parametrize("field,value,gap", [
    ("corpus", {}, "retained_owner_corpus_malformed"),
    ("corpus", None, "retained_owner_corpus_malformed"),
    ("inputs", [], "retained_owner_corpus_malformed"),
    ("continuation", "lost chain", "chained_owner_source_malformed"),
    ("sources", [], "chained_owner_source_malformed"),
    ("later", [{"source": "principal_task_message", "content": "peer words"}], "owner_source_provenance_unknown"),
    ("later", [{"source": "owner_mailbox", "content": "first", "msg_id": "same"},
               {"source": "owner_mailbox", "content": "rival", "msg_id": "same"}], "owner_message_identity_conflict"),
    ("gaps", [{}], "chained_owner_source_malformed"),
])
def test_malformed_owner_source_containers_refuse_without_claim(tmp_path, monkeypatch, field, value, gap):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    row = load_task_result(tmp_path, "pred-1")
    chain = {"original": {"source": "origin_message", "content": "Original words"}}
    if field in {"corpus", "inputs", "unavailable"}:
        inputs = {"owner_requirements_and_decisions": value} if field == "corpus" else (
            value if field == "inputs" else {"unavailable_sections": value})
        row["review_evidence"] = {"task_inputs": inputs}
    elif field == "mailrows":
        row["owner_mailbox"] = {"read_complete": True, "rows": value}
    else:
        if field not in {"continuation", "sources"}:
            chain[field] = value
        row["metadata"] = {"continuation": value if field == "continuation" else {
            "owner_sources": value if field == "sources" else chain}}
    seed_result(tmp_path, "pred-1", row)
    result = admit_continuation("pred-1", action_nonce=NONCE)
    assert result["error"] == "owner_source_missing" and gap in result["gaps"]
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, "pred-1")


def test_chained_continue_keeps_peer_context_as_peer_context(tmp_path, monkeypatch):
    from ouroboros.owner_continue import owner_sources, owner_corpus_rows
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    peer = {"msg_id": "peer-1", "text": "Peer's exact words", "source_task_id": "child-1",
            "provenance": "peer_task", "ts": "2026-09-27"}
    chained = {"original": {"source": "origin_message", "content": "Original words"},
               "later": [], "gaps": []}
    _interrupted(tmp_path, metadata={"continuation": {"owner_sources": chained, "peer_context": [peer]}})
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert sources["gaps"] == [] and sources["peer_context"] == [peer]
    assert owner_corpus_rows(sources) == [{"source": "origin_message", "content": "Original words"}]


@pytest.mark.parametrize("source,gap", [("unknown_source", "owner_source_provenance_unknown"),
                                      ("principal_task_message", "owner_message_identity_conflict")])
def test_duplicate_identity_cannot_hide_provenance_gap(tmp_path, monkeypatch, source, gap):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, metadata={"continuation": {"owner_sources": {
        "original": {"source": "origin_message", "content": "Original words"},
        "later": [{"source": "owner_mailbox", "content": "same words", "msg_id": "same"}]}}},
        review_evidence={"task_inputs": {"owner_requirements_and_decisions": [
            {"source": source, "content": "same words", "msg_id": "same"}]}})
    result = admit_continuation("pred-1", action_nonce=NONCE)
    assert result["error"] == "owner_source_missing" and gap in result["gaps"]
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, "pred-1")


def test_equal_unidentified_owner_and_peer_words_keep_both_sources(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, review_evidence={"task_inputs": {"owner_requirements_and_decisions": [
        {"source": "principal_task_message", "content": "same words", "source_task_id": "peer"},
        {"source": "owner_mailbox", "content": "same words"}]}})
    assert admit_continuation("pred-1", action_nonce=NONCE)["ok"]
    continuation = workers.PENDING[0]["metadata"]["continuation"]
    assert continuation["owner_sources"]["later"] == [
        {"source": "owner_mailbox", "content": "same words", "msg_id": ""}]
    assert continuation["peer_context"][0]["text"] == "same words"


def test_a_predecessor_that_heard_my_human_mid_turn_continues_while_its_mailbox_is_kept(tmp_path, monkeypatch):
    """#2811: the drain records my human's mid-turn place words under the opening's id, never under their context
    mailbox row's id. One id is one text and one owner-ness, so Continue admits the predecessor while its mailbox is
    still kept, and the successor carries their words."""
    import json
    from types import SimpleNamespace

    from ouroboros.owner_continue import owner_sources
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_place_arrival_wake import _delivered

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    words = "Send the status to the leads"
    assert write_task_message(tmp_path, json.dumps({"event": words}), "pred-1", source_task_id="presence-x",
                              provenance="place_event", msg_id="place-event:ident-1",
                              sender_origin={"place_id": "p1", "conversation_key": "chat:a:r"},
                              place_arrival={"source_event_id": "e1", "message": {}, "wake": True, "owner_words": words})
    reader = SimpleNamespace(task_id="pred-1", task_attempt=1, task_metadata={}, _loop_mailbox_seen_ids=set(),
                             _owner_directives=[{"source": "initial_user", "content": "Write the Friday report"}])
    _delivered(tmp_path, reader)
    write_task_result(tmp_path, "pred-1", "cancelled", review_evidence=_corpus(
        *_accept_owner_directives(reader, tmp_path, "pred-1")))
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    # A confirmed account's later place message is provenance the run may cite, never an owner directive of its own;
    # the Continue packet's owner rows therefore do not carry it (its words stay in the kept mailbox).
    assert sources["gaps"] == [] and sources["later"] == []
    assert admit_continuation("pred-1", action_nonce=NONCE)["ok"] is True


def test_a_late_answer_in_a_kept_mailbox_reaches_the_successor_with_its_card(tmp_path, monkeypatch):
    """Audit 05.11 item 37: a late answer delivered to a head still reading is my human's words with the card it
    answers. Continue rebuilds that row from the kept mailbox with the card beside it, never the bare label."""
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    card = {"quiz_id": "q7", "question": "Publish exactly this reply?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"], "answered_after_terminal": True}
    assert write_owner_message(tmp_path, "1. Publish it", "pred-1", msg_id="owner-late", late_answer={
        "task_id": "asker", "quiz_id": "q7", "answer_snapshot": {"task_id": "asker", "quiz": card}})
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert not sources["gaps"]
    [row] = [row for row in owner_corpus_rows(sources) if row["content"] == "1. Publish it"]
    assert (row["reply_to"]["author"], row["reply_to"]["text"].split("\n")[0]) == (
        "Ouroboros, task asker", "Publish exactly this reply?")


@pytest.mark.parametrize("shape", ["quiz_row", "late_answer"])
def test_a_relayed_answer_kept_in_a_mailbox_reaches_the_successor_with_its_card(tmp_path, monkeypatch, shape):
    """Audit 05.11 item 37: a quoted answer relayed to a work is my human's words with the card it answers. Continue
    rebuilds that row from the work's kept mailbox with the card beside it (for its Supervisor and its model). A row
    the host carried to this work (``host_carried``) is the tree's: it enters neither the corpus nor the peers."""
    from ouroboros import dialogue_store
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_relay import owner_quote, write_relay
    from ouroboros.task_results import load_task_result

    _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, root_phase_checkpoint={}, child_ref_promotion={})
    card = {"quiz_id": "q7", "question": "Publish exactly this reply?", "state": "answered", "answered_index": 0,
            "options": ["Publish it", "Not now"]}
    answer = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:01:00Z", "client_message_id": "owner-answer",
              "text": "1. Publish it", **({"type": "quiz_answer", "task_id": "asker", "quiz": card} if shape == "quiz_row"
                                           else {"late_answer": {"task_id": "asker", "quiz_id": "q7", "answer_snapshot": {
                                               "task_id": "asker", "quiz": {**card, "answered_after_terminal": True}}}})}
    dialogue_store.append(tmp_path, "chat", answer)
    write_relay(tmp_path, "pred-1", relay_id="r1", note="Please act on it.", quotes=[owner_quote(tmp_path, answer)],
                chat_id=1, teller_task_id="teller")
    carried = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:02:00Z", "client_message_id": "owner-carried",
               "text": "Use the Berlin numbers"}
    dialogue_store.append(tmp_path, "chat", carried)
    address = format_address(row_address(carried))
    write_relay(tmp_path, "pred-1", relay_id=f"carry:{address}", note="", chat_id=1, teller_task_id="root",
                quotes=[owner_quote(tmp_path, carried, address=address)], canonical_root=tmp_path,
                host_carried={"went_to": ""})
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert not sources["gaps"]
    rows = owner_corpus_rows(sources)
    [row] = [row for row in rows if row["content"] == "1. Publish it"]
    assert (row["reply_to"]["author"], row["reply_to"]["text"].split("\n")[0]) == (
        "Ouroboros, task asker", "Publish exactly this reply?")
    assert "Use the Berlin numbers" not in [row["content"] for row in rows]  # the tree's row, not this corpus
    assert [peer["msg_id"] for peer in sources["peer_context"]] == ["relay:r1:pred-1"]  # the note is a peer; the carried entry none


# --- one owner image message, one identity (#2941) ---------------------------------------------------------

IMAGE_WORDS = "Keep this exact wording.\n[image: literal owner syntax]\nDo not publish yet."  # image-like words are words


def _uploads(tmp_path, names):
    from tests._owner_image_mail import png_bytes

    folder = tmp_path / "uploads"
    folder.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        (folder / name).write_bytes(png_bytes(180 + index) if name.endswith(".png") else b"owner notes\n")
    return [folder / name for name in names]


def _image_mail_read(tmp_path, files, *, legacy=False, alter=None, broken=None, monkeypatch=None):
    """The real producer rows of one owner message: the server door stages FILES and writes it, the predecessor's own
    drain reads it, its terminal captures the corpus and the mailbox, and the settled mailbox goes (captured only).
    ``legacy`` drops ``words`` from the recorded row: the exact row a drain wrote for this message before #2941, as a
    restored pre-upgrade checkpoint still holds it. ``alter`` edits the captured corpus row;
    ``broken`` makes the drain's image composition raise (the drain's
    binding only: a later rebuild composes with the real composer)."""
    import queue
    from types import SimpleNamespace

    from ouroboros import loop_round_limits
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.post_task_synthesis import capture_task_inputs
    from ouroboros.task_custody import settle_task_mailbox
    from ouroboros.task_results import write_task_result
    from tests._owner_image_mail import write_owner_images

    _interrupted(tmp_path)
    text, manifest = write_owner_images(tmp_path, "pred-1", IMAGE_WORDS, files, msg_id="owner-img",
                                        client_message_id="cm-img")
    if broken is not None:
        monkeypatch.setattr(loop_round_limits, "build_incoming_user_content", broken)
    ctx = SimpleNamespace(task_attempt=1, task_id="pred-1", drive_root=tmp_path, task_metadata={})
    messages = [{"role": "user", "content": "Write the Friday report"}]
    _initialize_owner_directives(ctx, messages)
    loop_round_limits._drain_incoming_messages(messages, queue.Queue(), tmp_path, "pred-1", None, set(), ctx)
    for row in ctx._owner_directives if legacy else ():
        row.pop("words", None)
    inputs = capture_task_inputs(ctx, {"id": "pred-1"}, tmp_path, [])
    [row] = [row for row in inputs["owner_requirements_and_decisions"] if row.get("msg_id") == "owner-img"]
    if alter is not None:
        alter(row, text)
    write_task_result(tmp_path, "pred-1", "cancelled", review_evidence={"task_inputs": inputs})
    assert settle_task_mailbox(tmp_path, "pred-1", tmp_path)
    return text, manifest, row


def _refused(ack, workers, tmp_path, gap="owner_message_identity_conflict"):
    from ouroboros.task_results import load_task_result

    assert ack == {"ok": False, "error": "owner_source_missing", "gaps": [gap]}, ack
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, "pred-1")


def _replace(old, new):
    def alter(row, _text):
        assert old(row) in row["content"]
        row["content"] = row["content"].replace(old(row), new(row))
    return alter


@pytest.mark.parametrize("case, files, admitted", [
    ("one_image", ("a.png",), True),
    ("two_images", ("a.png", "b.png"), True),
    ("manifest_ref", ("notes.txt", "a.png"), True),
    ("literal_words", (), True),
    ("altered_original_path", ("a.png",), False),
    ("altered_image_ref", ("a.png",), False),
    ("appended_words", ("a.png",), False),
    ("pixels_unavailable_tail", ("a.png",), False),
    ("not_composed_tail", ("a.png",), False),
    ("bytes_changed_since", ("a.png",), False),
    ("literal_words_with_an_image_tail", (), False),
])
def test_a_pre_upgrade_image_projection_stands_only_as_its_exact_rebuilt_rendering(tmp_path, monkeypatch, case, files,
                                                                                  admitted):
    """A record from before ``words``: the corpus row is the drain's rendering of the exact mailbox row under one id.
    It is that message only when the drain's own composer, run again from the task's own store and verified pixels,
    gives the same projection byte for byte; the text is never parsed. An altered path, image reference or tail,
    a rendering error, a failed composition, changed bytes or a rendering beside a message with no images prove
    nothing: the explicit conflict stays and nothing is written."""
    from ouroboros import artifacts
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    if case == "manifest_ref":
        monkeypatch.setattr(artifacts, "_MAX_STAGED_ATTACHMENTS", 1)
    uploads = _uploads(tmp_path, files)
    alter = {
        "altered_original_path": _replace(lambda row: "; original: ", lambda row: "; original: /elsewhere"),
        "altered_image_ref": _replace(lambda row: row["content"].split("[image ref sha256:")[1][:16],
                                      lambda row: "0123456789abcdef"),
        "appended_words": lambda row, _text: row.update(content=row["content"] + "\nand delete the old report"),
        "literal_words_with_an_image_tail": lambda row, text: row.update(
            content=text + "\n[image: a.png; original: /x/a.png]\n[image ref sha256:0123456789abcdef]"),
    }.get(case)

    def broken(*_args, **_kwargs):
        raise RuntimeError("decoder crashed")

    if case == "pixels_unavailable_tail":  # the staged bytes changed before the drain rendered them
        original = artifacts.stage_task_attachments
        monkeypatch.setattr(artifacts, "stage_task_attachments", lambda *a, **k: [
            (row, (artifacts.task_artifact_dir_path(tmp_path, "pred-1") / row["relpath"]).write_bytes(b"changed"))[0]
            for row in original(*a, **k)])
    text, manifest, row = _image_mail_read(tmp_path, uploads, legacy=True, alter=alter, monkeypatch=monkeypatch,
                                           broken=broken if case == "not_composed_tail" else None)
    assert "words" not in row and (row["content"] == text) is (case == "literal_words"), "the pre-upgrade record"
    if case == "bytes_changed_since":
        (artifacts.task_artifact_dir_path(tmp_path, "pred-1") / manifest[0]["relpath"]).write_bytes(b"changed later")

    ack = admit_continuation("pred-1", action_nonce=NONCE)

    if not admitted:
        _refused(ack, workers, tmp_path)
        return
    assert ack["ok"] is True, ack
    successor = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert [(r["content"], "words" in r) for r in successor["metadata"]["owner_corpus"] if r.get("msg_id") == "owner-img"] == [
        (text, False)], "the exact mailbox row stands, once"
    assert [image["label"] for image in successor.get("attachment_images") or []] == [
        path.name for path in uploads if path.suffix == ".png"]


def test_two_exact_rows_under_one_image_message_id_still_conflict(tmp_path, monkeypatch):
    """Two exact mailbox rows under one id with different words are rivals, whatever files either carries."""
    from ouroboros.owner_mailbox import write_owner_message
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_image_mail import write_owner_images

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_owner_images(tmp_path, "pred-1", IMAGE_WORDS, _uploads(tmp_path, ("a.png",)), msg_id="owner-img")
    assert write_owner_message(tmp_path, "Publish it now", "pred-1", msg_id="owner-img")
    _interrupted(tmp_path)

    _refused(admit_continuation("pred-1", action_nonce=NONCE), workers, tmp_path)


@pytest.mark.parametrize("carried, same", [("exact_text", True), ("words_beside_a_rendering", True), ("rival", False)])
def test_a_message_its_start_already_carried_keeps_its_files_and_a_rival_attaches_none(tmp_path, monkeypatch,
                                                                                       carried, same):
    """The start's own corpus can name a message before the mailbox pass reads its exact row: the same words keep
    that row's files as this work's inputs, once; other words under that id are a conflict and attach nothing."""
    from ouroboros.owner_continue import owner_sources
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_image_mail import write_owner_images

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    text, _manifest = write_owner_images(tmp_path, "pred-1", IMAGE_WORDS, _uploads(tmp_path, ("a.png",)),
                                         msg_id="owner-img")
    row = {"source": "owner_mailbox", "msg_id": "owner-img", **{
        "exact_text": {"content": text}, "words_beside_a_rendering": {"content": text + "\n[image: a.png]", "words": text},
        "rival": {"content": text + "\nPublish it now"}}[carried]}
    _interrupted(tmp_path, metadata={"objective_author": {"kind": "task", "task_id": "drafter"}, "owner_corpus": [row]})
    sources = owner_sources(tmp_path, load_task_result(tmp_path, "pred-1"), "pred-1")
    assert [entry["msg_id"] for entry in sources["attached"]] == (["owner-img"] if same else [])
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    if not same:
        _refused(ack, workers, tmp_path)
        return
    successor = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    assert [image["label"] for image in successor["attachment_images"]] == ["a.png"]


@pytest.mark.parametrize("carrier, gap", [("start", "start_owner_corpus_row_malformed"),
                                          ("chained", "chained_owner_source_malformed")])
def test_words_that_are_not_text_refuse_wherever_the_row_rides(tmp_path, monkeypatch, carrier, gap):
    """A present ``words`` that is not text is a malformed row on every carrier: never read as its content."""
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    row = {"source": "owner_mailbox", "content": "Use last week's numbers\n[image ref sha256:0123456789abcdef]",
           "msg_id": "owner-2", "words": None}
    metadata = ({"objective_author": {"kind": "task", "task_id": "drafter"}, "owner_corpus": [row]} if carrier == "start"
                else {"continuation": {"predecessor_task_id": "pred-0", "owner_sources": {"later": [row], "original": {
                    "source": "origin_message", "content": "Write the Friday report",
                    "origin_message_ref": {"chat_id": 7, "client_message_id": "m-1"}}}}})
    _interrupted(tmp_path, metadata=metadata)

    _refused(admit_continuation("pred-1", action_nonce=NONCE), workers, tmp_path, gap=gap)


def test_a_carried_rendering_keeps_its_reference_and_its_words_through_each_continue(tmp_path, monkeypatch):
    """A message of my human's that reached the work through a carried corpus (a routed start's, by value) has no
    mailbox row in it: each successor keeps its image reference for re-viewing and its words for identity, so a
    Continue of a Continue (C1 → C2 → C3) compares the same words on every carrier."""
    import queue
    from types import SimpleNamespace

    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_words import carried_owner_corpus
    from ouroboros.post_task_synthesis import capture_task_inputs
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._owner_image_mail import write_owner_images

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    text, _manifest = write_owner_images(tmp_path, "drafter", IMAGE_WORDS, _uploads(tmp_path, ("a.png",)),
                                         msg_id="owner-img:drafter")
    drafter = SimpleNamespace(task_attempt=1, task_id="drafter", drive_root=tmp_path, task_metadata={})
    _drain_incoming_messages([], queue.Queue(), tmp_path, "drafter", None, set(), drafter)
    [carried] = [row for row in carried_owner_corpus(drafter) if row.get("msg_id") == "owner-img:drafter"]
    assert carried["words"] == text and carried["content"] != text and "[image ref sha256:" in carried["content"]
    _interrupted(tmp_path, metadata={"objective_author": {"kind": "task", "task_id": "drafter"},
                                     "owner_corpus": [carried]})
    current, nonce = "pred-1", NONCE
    for hop in range(3):
        ack = admit_continuation(current, action_nonce=nonce)
        assert ack["ok"] is True, (hop, ack)
        successor = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
        assert [(row["content"], row["words"]) for row in successor["metadata"]["owner_corpus"]
                if row.get("msg_id") == "owner-img:drafter"] == [(carried["content"], text)], hop
        assert carried["content"] in successor["text"], "the image reference stays readable"
        ctx = SimpleNamespace(task_attempt=1, task_id=successor["id"], drive_root=tmp_path,
                              task_metadata=successor["metadata"], _owner_directives=[])
        _initialize_owner_directives(ctx, [{"role": "user", "content": successor["text"]}])
        inputs = capture_task_inputs(ctx, {"id": successor["id"]}, tmp_path, [])
        assert [row["words"] for row in inputs["owner_requirements_and_decisions"]
                if row.get("msg_id") == "owner-img:drafter"] == [text], hop
        workers.PENDING[:] = []
        write_task_result(tmp_path, successor["id"], "cancelled", result="interrupted", review_evidence={
            "task_inputs": inputs}, cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
        current, nonce = successor["id"], f"hop-{hop}-press-2941"
