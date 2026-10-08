"""The single admission door for the roots consciousness starts (P3, PLAN 5.5 / 5.14 п.2).

``supervisor.queue.enqueue_task`` — the ONE place every pooled root passes —
refuses a consciousness-origin ROOT under the queue lock when the concurrency
cap (``OUROBOROS_CONSCIOUSNESS_MAX_TASKS``, 0 = never) or the rolling-24h
allowance (``OUROBOROS_CONSCIOUSNESS_DAILY_USD``, 0 = may not spend) says so,
in the queue's EXISTING refusal shape (``_admission_blocked`` + the
``_admission_detail`` the depth guard already uses). Subagents are their
root's business; owner work is never gated; a snapshot restore re-admits
already-admitted work.
"""

from __future__ import annotations

import types

import pytest

from ouroboros import consciousness_authority as ca

AVAILABLE = {"status": "available", "limit_usd": 20.0, "settled_usd": 4.0, "accounted_usd": 4.0, "remaining_usd": 16.0,
             "unknown_unmetered": 0, "resets_at": ""}


@pytest.fixture
def door(tmp_path, monkeypatch):
    from supervisor import queue

    queue.init(tmp_path)
    pending: list = []
    running: dict = {}
    queue.init_queue_refs(pending, running, {"value": 0})
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 2)
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda root, now=None: dict(AVAILABLE))
    return queue, pending, running


def _root(tid, level="act", **extra):
    return {"id": tid, "type": "task", "text": tid, "chat_id": 1, "root_task_id": tid, "delegation_role": "root",
            "metadata": {"initiator": "consciousness", "usage_category": "consciousness_task",
                         "consciousness_autonomy": level, **extra}}


def test_a_consciousness_root_is_admitted_with_its_level_derived(door):
    queue, pending, _running = door
    admitted = queue.enqueue_task(_root("c1"))
    assert "_admission_blocked" not in admitted and [t["id"] for t in pending] == ["c1"]
    assert admitted["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert admitted["metadata"]["runtime_mode_cap"] == "light"
    assert queue.live_consciousness_root_count() == 1


def test_the_concurrency_cap_counts_live_pending_and_running_roots(door):
    queue, pending, running = door
    assert "_admission_blocked" not in queue.enqueue_task(_root("c1"))
    running["c2"] = {"task": _root("c2")}
    assert queue.live_consciousness_root_count() == 2
    refused = queue.enqueue_task({**_root("c3"), "_admission_token": "tok"})
    assert refused["_admission_blocked"] == "consciousness_task_limit"
    assert refused["_admission_detail"] == "2 of 2 consciousness-started tasks already live"
    assert [t["id"] for t in pending] == ["c1"]
    assert "c3" not in queue.ADMISSION_RESERVATIONS
    # A settled root frees its slot: the count is live, not historical.
    running.clear()
    assert "_admission_blocked" not in queue.enqueue_task(_root("c3"))


def test_max_tasks_zero_means_consciousness_never_starts_tasks(door, monkeypatch):
    queue, pending, _running = door
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 0)
    refused = queue.enqueue_task(_root("c1"))
    assert refused["_admission_blocked"] == "consciousness_task_limit"
    assert "OUROBOROS_CONSCIOUSNESS_MAX_TASKS=0" in refused["_admission_detail"]
    assert pending == []


def test_an_exhausted_allowance_refuses_with_the_window_facts(door, monkeypatch):
    queue, pending, _running = door
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda root, now=None: {
        **AVAILABLE, "status": "exhausted", "settled_usd": 21.5, "accounted_usd": 25.5, "remaining_usd": 0.0,
        "unknown_unmetered": 1, "resets_at": "2026-09-17T08:00:00+00:00"})
    refused = queue.enqueue_task(_root("c1"))
    assert refused["_admission_blocked"] == "consciousness_allowance_exhausted"
    # Known spend decides and is called spent; open holds are shown beside it, uncounted (#1487).
    assert refused["_admission_detail"] == (
        "$21.50 (at least) of $20.00 known spend in the last 24 h (plus $4.00 of open holds, "
        "not counted); resets at 2026-09-17T08:00:00+00:00")
    assert pending == []
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda root, now=None: {
        **AVAILABLE, "status": "exhausted", "limit_usd": 0.0, "settled_usd": 0.0, "accounted_usd": 0.0,
        "remaining_usd": 0.0})
    zero = queue.enqueue_task(_root("c2"))
    assert zero["_admission_blocked"] == "consciousness_allowance_exhausted"
    assert "OUROBOROS_CONSCIOUSNESS_DAILY_USD=0" in zero["_admission_detail"]


def test_an_unreadable_ledger_refuses_honestly_instead_of_admitting(door, monkeypatch):
    queue, pending, _running = door
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda root, now=None: {
        "status": "allowance_unknown", "error": "OSError: ledger locked", "limit_usd": 20.0,
        "accounted_usd": None, "remaining_usd": None, "resets_at": ""})
    refused = queue.enqueue_task(_root("c1"))
    assert refused["_admission_blocked"] == "consciousness_allowance_unknown"
    assert "ledger locked" in refused["_admission_detail"] and pending == []


def test_subagents_and_owner_work_pass_the_door_untouched(door, monkeypatch):
    queue, pending, running = door
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 0)
    child = {**_root("kid"), "delegation_role": "subagent", "root_task_id": "wake-1", "parent_task_id": "wake-1"}
    assert "_admission_blocked" not in queue.enqueue_task(child)
    assert child["metadata"]["initiator"] == "consciousness"  # the label rides; the door ignores children
    owner = {"id": "o1", "type": "task", "text": "owner work", "chat_id": 1, "delegation_role": "root",
             "metadata": {"client_message_id": "cm-1"}}
    assert "_admission_blocked" not in queue.enqueue_task(owner)
    assert [t["id"] for t in pending] == ["kid", "o1"]
    assert queue.live_consciousness_root_count() == 0


def test_a_snapshot_restore_is_not_gated(door, monkeypatch):
    queue, pending, _running = door
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 0)
    restored = queue.enqueue_task(_root("c1"), restoring_snapshot=True)
    assert "_admission_blocked" not in restored and [t["id"] for t in pending] == ["c1"]


def test_the_refusal_uses_the_queue_vocabulary_the_promote_path_already_reads():
    """The promote handler turns ``_admission_blocked``/``_admission_detail`` into the
    typed PROMOTE_REJECTED the tool renders (pinned in test_consciousness_authority);
    the depth guard writes the same pair, so no new refusal form exists."""
    import inspect

    from supervisor import task_admission, worker_promotion

    assert '_admission_detail' in inspect.getsource(task_admission.reject_invalid_task_depth)
    assert '"detail": str(admitted.get("_admission_detail") or "")' in inspect.getsource(worker_promotion.promote_chat_to_task)


def test_promote_refusal_carries_the_door_detail(tmp_path, monkeypatch):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: {**task, "_admission_blocked": "consciousness_task_limit",
                                   "_admission_detail": "2 of 2 consciousness-started tasks already live"},
        persist_queue_snapshot=lambda **_k: True, load_state=lambda: {"owner_chat_id": 1},
    )
    evt = {"type": "promote_chat_to_task", "task_id": "c0000002", "objective": "x", "chat_id": 1,
           "workspace": "none", "initiator": "consciousness"}
    outcome = workers.promote_chat_to_task(evt, ctx)
    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "consciousness_task_limit"
    assert outcome["detail"] == "2 of 2 consciousness-started tasks already live"
