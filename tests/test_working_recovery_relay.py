"""A receiver lost before its first save must not erase its inherited cognition.

The receiver comes from the live same-id crash retry, or is a stored pre-upgrade new-ID timeout
successor (no producer mints one now, #1376); the loss of either retries through the crash rail.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from ouroboros import working_checkpoint as wc
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import load_task_result
from supervisor import queue, workers
from supervisor.events_budget import budget_hold_fact
from tests._silence_reaper_support import historical_successor
from tests.test_project_hold_recovery import worker
from tests.test_restart_saved_work import _ack, _crash, _stop_and_boot
from tests.test_swarm_host_admission import host  # noqa: F401
from tests.test_working_recovery_admission import _running
from tests._store_shared import read_result, seed_result

pytestmark = pytest.mark.serial


def _first_receiver(host, tmp_path, monkeypatch, scope, origin="crash"):  # noqa: F811 - the imported fixture
    original = _running(host, tmp_path, monkeypatch, scope)
    if origin == "crash":
        target_id = "held"
        _crash(queue, workers, monkeypatch, "held")
    else:
        target_id = "held-retry"
        workers.RUNNING.clear()
        historical_successor(original, 1, target_id)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    [receiver] = sent
    assert receiver["id"] == target_id and receiver["_attempt"] == 2
    ctx = SimpleNamespace(task_id=target_id, task_attempt=2, drive_root=host.root,
                          working_recovery=receiver["_working_recovery"])
    saved = wc.load_recovery(ctx)
    wc.consume_recovery(ctx, saved["_working_handoff"])
    assert not wc.checkpoint_path(host.root, target_id, 2).exists()
    assert not wc.checkpoint_path(host.root, "held", 1).exists()
    return receiver, copy.deepcopy(receiver["_working_recovery"])


@pytest.mark.parametrize("scope", ["project", "main"])
@pytest.mark.parametrize("origin", ["crash", "stored_new_id"])
def test_real_crash_retry_carries_original_source_across_two_lost_receivers(
    host, tmp_path, monkeypatch, scope, origin,  # noqa: F811
):
    receiver, original = _first_receiver(host, tmp_path, monkeypatch, scope, origin)
    exact = read_actor_source_bytes(host.root, "held", original["source_ref"])
    target_id = receiver["id"]
    _crash(queue, workers, monkeypatch, target_id)  # lost before its first save
    [retry] = host.pending
    relay = retry["_working_recovery"]
    assert relay["target_task_id"] == target_id and relay["target_attempt"] == 3
    assert relay["continued_from_task_id"] == receiver["id"] and relay["continued_from_attempt"] == 2
    for key in ("source_ref", "source_task_id", "from_attempt", "seq"):
        assert relay[key] == original[key]
    assert read_actor_source_bytes(host.root, "held", relay["source_ref"]) == exact
    assert wc.recovery_source_for_task(host.root, retry)["messages"]
    changed = {**retry, "_attempt": 4}
    assert not wc.recovery_source_for_task(host.root, changed), "a relay names one exact receiver"
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [target_id], retry
    assert load_task_result(host.root, target_id)["admitted_dispatch_attempt"] == 3


@pytest.mark.parametrize("restart", [False, True], ids=["quit", "acknowledged-restart"])
def test_real_stop_restores_a_dispatched_receiver_without_its_first_checkpoint(
    host, tmp_path, monkeypatch, restart,  # noqa: F811
):
    from supervisor.restart_retention import prepare_restart_returns

    receiver, original = _first_receiver(host, tmp_path, monkeypatch, "project")
    monkeypatch.setattr(workers, "WORKERS", {})
    if restart:
        assert prepare_restart_returns(host.root, workers.RUNNING, workers.PENDING,
                                       transaction_id="relay-return") == {"held"}
        _ack(host.root, "relay-return")
    assert _stop_and_boot(queue, workers) == 1
    [restored] = host.pending
    assert restored["_attempt"] == 3
    assert restored["_working_recovery"]["source_ref"] == original["source_ref"]
    assert restored["_working_recovery"]["target_attempt"] == 3
    sent = worker(host, monkeypatch)
    if not restart:
        workers.assign_tasks()
        assert not sent and budget_hold_fact(restored)
        assert queue.resume_budget_paused_task("held")["ok"]
    workers.assign_tasks()
    assert len(sent) == 1 and sent[0]["_attempt"] == 3


@pytest.mark.parametrize("damage", ["missing_mark", "later_mark", "bad_prior_attempt", "bad_source", "newer_torn"])
def test_a_relay_never_guesses_missing_prior_dispatch_or_replaces_a_newer_source(
    host, tmp_path, monkeypatch, damage,  # noqa: F811
):

    prior, original = _first_receiver(host, tmp_path, monkeypatch, "project")

    stored = read_result(host.root, "held")
    if damage == "missing_mark":
        stored.pop("admitted_dispatch_attempt")
    elif damage == "later_mark":
        stored["admitted_dispatch_attempt"] = 3
    elif damage == "bad_prior_attempt":
        prior["_attempt"] = 1
    elif damage == "bad_source":
        prior["_working_recovery"]["source_ref"]["sha256"] = "0" * 64
    else:
        wc.checkpoint_path(host.root, "held", 2, create=True).write_text("{")
    seed_result(host.root, "held", stored)
    next_task = {**prior, "_attempt": 3}
    assert not wc.attach_recovery(host.root, next_task, source_task_id="held", from_attempt=2,
                                  cause="worker_crash", prior_task=prior)
    assert next_task["_working_recovery"] == prior["_working_recovery"]
    assert not wc.recovery_source_for_task(host.root, next_task)
