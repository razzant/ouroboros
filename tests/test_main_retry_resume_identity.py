"""#1660 / #1730: Main uses the same current retry identity as the Resume door.

Real Project admission, idle-timeout retry publication, exact root-budget park,
HTTP state and Main's JS consumer. Only OS workers, models and the clock are
doubled by the existing retry fixtures; no identity predicate is stubbed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause
from ouroboros.owner_pause import root_actor
from ouroboros.task_results import load_task_result, resolve_task_lineage
from supervisor import queue, workers
from supervisor.events_budget import install_exact_budget_pause
from tests._budget_pause_exact_helpers import _fast_hold, _loop_ctx, _mock_pause_observation
from tests.test_execution_clock_resume_handover import _dispatch_one, _save_first_attempt, clock as _clock
from tests.test_main_state_projection import _attention_rows, _main_consumer, _member_http, _member_owner
from tests.test_project_hold_recovery import accepted
from tests.test_retry_pause_identity import _assign, _reap, _successors, reaper as _reaper
from tests.test_swarm_host_admission import host as _host

clock, reaper, host = _clock, _reaper, _host  # Register shared pytest fixtures.

pytestmark = pytest.mark.serial


def _project_retry(host, tmp_path, monkeypatch, clock, reaper):
    _member_owner(host.root, monkeypatch)
    accepted(host, tmp_path)
    first = _dispatch_one(host, monkeypatch)
    _save_first_attempt(host, first, monkeypatch, clock)
    workers.WORKERS[0] = SimpleNamespace(wid=0, busy_task_id="held", reaping=False, active_capacity=True)
    _reap(clock, reaper)
    [retry] = _successors(host)
    logical = load_task_result(host.root, "held", strict=True)
    physical = load_task_result(host.root, retry["id"], strict=True)
    assert logical["status"] == "interrupted"
    assert logical["superseded_by"] == logical["retry_task_id"] == retry["id"] != "held"
    assert physical["supersedes_task_id"] == physical["original_task_id"] == physical["timeout_retry_from"] == "held"
    assert physical["root_task_id"] == "held" and retry["project_id"] == first["project_id"]
    assert resolve_task_lineage(retry["id"], **{key: physical.get(key) for key in (
        "metadata", "root_task_id", "parent_task_id", "delegation_role", "original_task_id",
        "timeout_retry_from")})["is_retry_root_attempt"]
    return retry


def _park_retry(host, monkeypatch, *, reason="budget", child=True):
    [running] = _assign(host, monkeypatch)
    if child:
        member = queue.enqueue_task({"id": "held-child", "type": "task", "text": "child work",
            "root_task_id": "held", "parent_task_id": running["id"], "delegation_role": "subagent", "depth": 1,
            "project_id": running["project_id"], "chat_id": running["chat_id"],
            "_project_admission": dict(running["_project_admission"])})
        assert not member.get("_admission_blocked"), member
    ctx, limit = _loop_ctx(host.root, running["id"], attempt=running["_attempt"])
    ctx.root_task_id = "held"
    ctx.task_metadata = {key: running[key] for key in ("root_task_id", "original_task_id", "timeout_retry_from")}
    limit.messages = [{"role": "user", "content": "Continue the project"},
                      {"role": "assistant", "content": "Saved retry work"}]
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    if reason == "owner":
        from supervisor.owner_pause_control import request_owner_pause

        assert request_owner_pause(running["id"], request_id="retry-owner-pause")["ok"]
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        if reason == "owner":
            budget_pause.enter_owner_pause(limit)
        else:
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_DISPATCH_REFUSED,
                scope="root", reason_text="root budget cap", root_task_id="held")
    budget_pause.end_dispatch_fence(running["id"])
    install_exact_budget_pause(SimpleNamespace(DRIVE_ROOT=host.root, RUNNING=workers.RUNNING,
        PENDING=workers.PENDING, WORKERS=workers.WORKERS, sort_pending=queue.sort_pending,
        persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None), running["id"],
        budget_pause.exact_pause_marker(raised.value.pause)["checkpoint"])
    pause = budget_pause.budget_pause_row(host.root, running["id"])
    assert pause["root_task_id"] == "held" and pause["state"] == "paused" and pause["source_ref"]
    assert pause["scope"] == "root" and pause["reason"] == reason
    assert "held" in queue.BUDGET_ROOT_FENCES and running["id"] not in workers.RUNNING
    parked = next(row for row in host.pending if row["id"] == running["id"])
    assert parked["_budget_pause"]["exact_continuation"] is True
    return parked


def _snapshot(client):
    response = client.get("/api/state")
    assert response.status_code == 200, response.text
    return response.json()


def _record(label, payload, task_id):
    # Retained pytest stdout captures the real projection, not just a predicate verdict.
    print(json.dumps({"label": label, "task_id": task_id,
        "activities": payload["active_chat_activities"], "attention": payload["main_attention"],
        "complete": payload["main_attention_complete"], "gaps": payload["main_attention_gaps"]}, sort_keys=True))


def test_root_budget_retry_keeps_its_resume_in_state_and_main(host, tmp_path, monkeypatch, clock, reaper):
    retry = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    parked = _park_retry(host, monkeypatch)
    assert parked["id"] == retry["id"] and root_actor(host.root, "held", retry["id"], strict=True)
    assert not root_actor(host.root, "held", "held-child", strict=True)
    assert queue.BUDGET_ROOT_FENCES["held"]["status"] == "paused"
    with _member_http(host.root) as client:
        payload = _snapshot(client)
        _record("root-budget-retry", payload, retry["id"])
        dom = _main_consumer([payload], retry["id"])
        print(json.dumps({"main_consumer": dom}, sort_keys=True))
        assert payload["active_chat_activities_complete"] and payload["main_attention_complete"]
        assert not _attention_rows(payload, "held-child"), "the genuine member waits for its root"
        assert "held-child" not in {row["activity_id"] for row in payload["active_chat_activities"]}
        [activity] = [row for row in payload["active_chat_activities"] if row["activity_id"] == retry["id"]]
        assert activity["phase"] == "budget_paused" and activity["can_resume"] is True
        [action] = _attention_rows(payload, retry["id"])
        assert action["attention_reason"] == "budget" and action["can_resume"] is True
        assert dom["card"] and dom["resume"] and dom["enabled"]
        assert dom["posted"] == [[f'/api/tasks/{retry["id"]}/resume', "POST"]]
        response = client.post(f'/api/tasks/{retry["id"]}/resume')
        assert response.status_code == 200, response.text
        assert "held" not in queue.BUDGET_ROOT_FENCES
        assert not _attention_rows(_snapshot(client), retry["id"])


@pytest.mark.parametrize("fault", ["unreadable", "nonreciprocal"])
def test_unknown_retry_identity_is_incomplete_and_disables_a_shown_resume(
        host, tmp_path, monkeypatch, clock, reaper, fault):
    from tests._store_shared import seed_result, unreadable_result

    retry = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    _park_retry(host, monkeypatch, child=False)
    with _member_http(host.root) as client:
        shown = _snapshot(client)  # also warms the fence/display memo before identity becomes unreadable
        if fault == "unreadable":
            unreadable_result(monkeypatch, "held", root=host.root)
        else:
            root = load_task_result(host.root, "held", strict=True)
            seed_result(host.root, "held", {**root, "retry_task_id": "unconfirmed-successor"})
        unknown = _snapshot(client)
        _record(fault, unknown, retry["id"])
        assert unknown["active_chat_activities_complete"] is False
        assert unknown["main_attention_complete"] is False
        assert not _attention_rows(unknown, retry["id"])
        dom = _main_consumer([shown, unknown], retry["id"])
        assert dom["card"] and dom["resume"] and not dom["enabled"] and not dom["posted"]
        assert client.post(f'/api/tasks/{retry["id"]}/resume').status_code != 200


def test_owner_pause_of_a_retry_stays_quiet_in_main(host, tmp_path, monkeypatch, clock, reaper):
    retry = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    _park_retry(host, monkeypatch, reason="owner")
    with _member_http(host.root) as client:
        payload = _snapshot(client)
        _record("own-pause", payload, retry["id"])
        assert payload["main_attention_complete"] is True
        assert not payload["main_attention"]
        assert not _main_consumer([payload], retry["id"])["card"]


def test_a_superseded_retry_is_not_the_current_root_actor(host, tmp_path, monkeypatch, clock, reaper):
    from supervisor.events_budget import explicit_resume_required

    first = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    [running] = _assign(host, monkeypatch)
    _reap(clock, reaper, running["id"])
    [current] = [row for row in host.pending if row.get("timeout_retry_from") == first["id"]]
    stored = load_task_result(host.root, first["id"], strict=True)
    assert stored["superseded_by"] == stored["retry_task_id"] == current["id"]
    parked = _park_retry(host, monkeypatch, child=False)
    # A stale consumer carrier cannot acquire root authority merely from its lineage IDs.
    stale = {**first, "_budget_pause": dict(parked["_budget_pause"])}
    assert not root_actor(host.root, "held", first["id"], strict=True)
    assert root_actor(host.root, "held", current["id"], strict=True)
    assert explicit_resume_required(queue, stale, owner_paused=False) is False


@pytest.mark.parametrize("identity", ["logical-root", "current-retry", "child", "unreadable-chain"])
def test_interrupted_member_resume_uses_the_existing_root_actor_boundary(
        host, tmp_path, monkeypatch, clock, reaper, identity):
    """O1 is a guard contract, not a claim that the reaper terminalizes its old root.

    Both root identities must be excluded before reading member-grant facts;
    unreadable member facts/identity remain unknown for an actual descendant.
    """
    from ouroboros.owner_pause import interrupted_member_resume
    from tests._store_shared import seed_result

    retry = _project_retry(host, tmp_path, monkeypatch, clock, reaper)
    member = queue.enqueue_task({"id": "held-child", "type": "task", "text": "child work",
        "root_task_id": "held", "parent_task_id": retry["id"], "delegation_role": "subagent", "depth": 1,
        "project_id": retry["project_id"], "chat_id": retry["chat_id"],
        "_project_admission": dict(retry["_project_admission"])})
    assert not member.get("_admission_blocked"), member
    if identity == "unreadable-chain":
        root = load_task_result(host.root, "held", strict=True)
        seed_result(host.root, "held", {**root, "retry_task_id": "unconfirmed-successor"})
    reads = []

    def unavailable_member_grant(root, task_id):
        reads.append(task_id)
        raise OSError("member-grant facts unavailable")

    monkeypatch.setattr(budget_pause, "budget_pause_row", unavailable_member_grant)
    task_id = {"logical-root": "held", "child": "held-child"}.get(identity, retry["id"])
    expected = None if identity in {"child", "unreadable-chain"} else False
    assert interrupted_member_resume(host.root, "held", task_id, attempt=retry["_attempt"]) is expected
    assert reads == (["held-child"] if identity == "child" else [])
