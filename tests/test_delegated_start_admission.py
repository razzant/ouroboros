"""Independent starts of one task take the actor lock before the launch lock, as their claims do.

A sibling's claim delays a start; it never refuses it. A Resume choice committed first still makes
the same invocation lookup-only. Pause at the handoff and real read failures send nothing.
"""
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import owner_pause, store
from ouroboros.delegate_resume import BUSY, LOOKUP_ONLY, UNREADABLE, StartHeld, checked_start, close_unsent_start
from ouroboros.task_results import write_task_result
from tests._owner_pause_ended_helpers import REQUEST, _decide, _pending


def _source(root):
    write_task_result(root, "root-1", "running", root_task_id="root-1")
    return SimpleNamespace(drive_root=root, task_id="root-1", root_task_id="root-1", task_metadata={})


def _waits_for_actor_lock(monkeypatch, thread_name):
    """Set when THREAD_NAME asks for an actor lock; the real acquisition follows unchanged."""
    real, asked = custody.actor_decision_lock, threading.Event()

    def lock(root, task_id, **kwargs):
        if threading.current_thread().name == thread_name:
            asked.set()
        return real(root, task_id, **kwargs)
    monkeypatch.setattr(custody, "actor_decision_lock", lock)
    return asked


def _send(source, iid, sent, results):
    def submit():
        sent.append((iid, threading.current_thread().name))
        return {"runId": "run-" + iid}
    try:
        results[iid] = owner_pause.run_operation(source, checked_start(source.drive_root, "root-1", iid, submit))
    except BaseException as exc:  # the assertion names it
        results[iid] = exc


def test_a_sibling_claim_delays_an_independent_start_and_both_send_once(tmp_path, monkeypatch):
    from ouroboros.delegate_start_claims import claimed_start_request

    source = _source(tmp_path)
    _pending(tmp_path, "root-1", "b")  # B already holds its own valid pending claim
    b_asked = _waits_for_actor_lock(monkeypatch, "sender-b")
    a_holds = threading.Event()
    record = custody.record_start_requested

    def claim_while_b_waits(root, **row):
        a_holds.set()  # A is inside _claimed, holding the actor lock
        assert b_asked.wait(10), "B never contended"
        return record(root, **row)
    monkeypatch.setattr(custody, "record_start_requested", claim_while_b_waits)
    sent, results, claims = [], {}, {}

    def sender_a():
        claims["a"] = claimed_start_request(
            tmp_path, task_id="root-1", root_task_id="root-1", invocation_id="a", idempotency_key="a",
            request=REQUEST, project_id="", project_owned=False, route="some-route")
        _send(source, "a", sent, results)
    a = threading.Thread(target=sender_a, name="sender-a")
    b = threading.Thread(target=_send, args=(source, "b", sent, results), name="sender-b")
    a.start()
    assert a_holds.wait(10)
    b.start()
    a.join(30)
    b.join(30)
    assert not a.is_alive() and not b.is_alive()
    assert claims["a"] == (True, {})
    assert results == {"a": {"runId": "run-a"}, "b": {"runId": "run-b"}}, results
    assert sorted(iid for iid, _ in sent) == ["a", "b"]  # each request reached transport exactly once
    assert all(not thread.startswith("sender-") for _, thread in sent)  # I/O ran on the executor, after release
    assert {custody.invocation_record(tmp_path, iid)["state"] for iid in "ab"} == {"pending"}


def test_a_choice_committed_while_the_sender_waits_keeps_the_same_invocation_lookup_only(tmp_path, monkeypatch):
    from ouroboros import delegate_invocation

    source = _source(tmp_path)
    start = _pending(tmp_path, "root-1", "lost")
    asked = _waits_for_actor_lock(monkeypatch, "sender")
    read, holding = delegate_invocation._read, threading.Event()

    def decider_read(txn, iid):
        if threading.current_thread().name == "decider":
            holding.set()  # _record holds the actor lock here
            assert asked.wait(10), "the sender never contended"
        return read(txn, iid)
    monkeypatch.setattr(delegate_invocation, "_read", decider_read)
    decided, sent, results = {}, [], {}
    decider = threading.Thread(target=lambda: decided.update(d=_decide(tmp_path, start)), name="decider")
    decider.start()
    assert holding.wait(10)
    sender = threading.Thread(target=_send, args=(source, "lost", sent, results), name="sender")
    sender.start()
    decider.join(30)
    sender.join(30)
    assert decided["d"]["creation"] == "lookup_only"
    assert isinstance(results["lost"], StartHeld) and str(results["lost"]) == LOOKUP_ONLY
    assert sent == []  # no replay of the accepted invocation
    assert custody.invocation_record(tmp_path, "lost")["state"] == "pending"


@pytest.mark.parametrize("control", ["pause", "stop"])
def test_owner_control_at_the_handoff_sends_nothing_and_closes_the_fresh_request(tmp_path, monkeypatch, control):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.model_wait import task_model_wait_scope

    source = _source(tmp_path)
    _pending(tmp_path, "root-1", "fresh")
    asked = _waits_for_actor_lock(monkeypatch, "sender")
    sent, results = [], {}

    def owned_send():
        # The tool's own invocation, as the registry binds it: Stop is read at its effect boundary.
        with task_model_wait_scope(task={"id": "root-1", "_attempt": 1}, drive_root=tmp_path,
                                   event_queue=None, worker_slot_held=False), \
                owner_pause.tool_handoff(source, "delegate_start"):
            _send(source, "fresh", sent, results)
    sender = threading.Thread(target=owned_send, name="sender")
    with custody.actor_decision_lock(tmp_path, "root-1"):  # a sibling's claim
        sender.start()
        assert asked.wait(10)
        if control == "pause":
            owner_pause.install_fence(tmp_path, "root-1", request_id="press")
        else:
            request_cancel(tmp_path, "root-1", reason="owner stop", source="owner", requested_by="owner")
    sender.join(30)
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    refusal = owner_pause.OwnerPauseRefused if control == "pause" else PhysicalDispatchInterrupted
    assert isinstance(results["fresh"], refusal) and not isinstance(results["fresh"], StartHeld), results
    assert sent == []
    assert close_unsent_start(tmp_path, "root-1", "fresh", control)  # the sender's own closure, now uncontended
    assert custody.invocation_record(tmp_path, "fresh")["state"] != "pending"


@pytest.mark.parametrize("failure,code", [
    ("lock_contention", BUSY), ("lock_errno", UNREADABLE), ("store_busy", BUSY), ("store_lost", UNREADABLE),
])
def test_contention_and_a_real_read_failure_are_told_apart_and_send_nothing(tmp_path, monkeypatch, failure, code):
    import errno

    from ouroboros import platform_layer

    source = _source(tmp_path)
    _pending(tmp_path, "root-1", "fresh")
    if failure.startswith("lock_"):
        def refused(_path, *, outcome=None, **_kwargs):
            outcome.update(reason="contention", errno=None) if failure == "lock_contention" else \
                outcome.update(reason="kernel_refused", errno=errno.ENOSPC)
            return None
        monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", refused)
    else:
        def unavailable(*_args, **_kwargs):
            raise store.StoreBusy("contended", reason="contention") if failure == "store_busy" else store.StoreLost("gone")
        monkeypatch.setattr(store, "open_store", unavailable)
    sent, results = [], {}
    _send(source, "fresh", sent, results)
    assert isinstance(results["fresh"], StartHeld) and str(results["fresh"]) == code, results
    assert sent == []
    monkeypatch.undo()
    assert custody.invocation_record(tmp_path, "fresh")["state"] == "pending"  # nothing changed
