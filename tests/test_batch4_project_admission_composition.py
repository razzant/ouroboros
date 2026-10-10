"""Batch4 owner controls composed with published Project admission (#1441).

A fresh Continue of a Project task is a fresh host admission: its successor must
carry the prepared Project basis, never the legacy marker the queue gives a row
that arrives without one. Otherwise a temporary authority outage across a
restart would hold the same successor forever, after the authority returned.
Project hold recovery releases only its own hold: the owner's Pause latch and
Restart hold stay until the owner's own Resume.
"""
from __future__ import annotations

import copy

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import STATUS_CANCELLED, STATUS_RUNNING, load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_project_hold_recovery import accepted, restore_unreadable, resume_after_app_stop, worker
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial

NONCE = "press-0001-abcdef"


def _interrupted_project_task(host, tmp_path, task_id="pred-1"):  # noqa: F811
    folder = (tmp_path / "prepared").resolve()
    folder.mkdir(exist_ok=True)
    registry.create_project(host.root, "target", name="Original Project", working_dir=str(folder))
    chat_id = registry.get_project(host.root, "target")["chat_id"]
    write_task_result(host.root, task_id, STATUS_RUNNING, chat_id=chat_id, project_id="target",
                      workspace_root=str(folder), root_task_id=task_id, title="Friday report",
                      billing_group={"billing_group_id": task_id, "billing_group_limit_usd": 20.0,
                                     "billing_group_limit_source": "initial_task_admission",
                                     "billing_group_limit_revision": "admission-1"},
                      origin_message_text="Write the Friday report",
                      origin_message_ref={"chat_id": chat_id, "client_message_id": "m-1"})
    write_task_result(host.root, task_id, STATUS_CANCELLED, result="interrupted",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    return folder


def test_fresh_project_continue_recovers_once_after_temporary_authority_outage(host, tmp_path, monkeypatch):  # noqa: F811
    from supervisor.continuation_admission import admit_continuation

    folder = _interrupted_project_task(host, tmp_path)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True and ack["held"] is False, ack
    successor = ack["successor_task_id"]
    admitted = copy.deepcopy(next(row for row in host.pending if row["id"] == successor))
    basis = admitted["_project_admission"]
    assert basis["project_id"] == "target" and basis["frozen"] is True
    assert not basis.get("legacy_basis"), "a fresh Continue is prepared, never a legacy basis"
    assert admitted["workspace_root"] == str(folder) and admitted["admitted_dispatch"] == "none"

    # An unacknowledged app restart during the outage holds the same accepted id.
    path, original = restore_unreadable(host)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_project_admission_restore_hold"], "missing authority keeps the hold"

    path.write_bytes(original)  # the authority returns
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent, "authority recovery cannot grant Resume after an app stop"
    assert not host.pending[0].get("_project_admission_restore_hold")
    resume_after_app_stop(host, successor)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [successor], "same successor, dispatched exactly once"
    for key in ("workspace_root", "chat_id", "project_id", "text", "metadata", "_project_admission"):
        assert sent[0][key] == admitted[key], key
    continuation = sent[0]["metadata"]["continuation"]
    assert continuation["billing_group_id"] == "pred-1" and continuation["billing_group_limit_usd"] == 20.0
    assert [row["content"] for row in sent[0]["metadata"]["owner_corpus"]] == ["Write the Friday report"]
    assert load_task_result(host.root, successor)["admitted_dispatch"] == "possible"  # durable before the put

    # The same press is still the same admission; no second root or handoff.
    again = admit_continuation("pred-1", action_nonce=NONCE)
    assert again["ok"] is True and again["replay"] is True and again["successor_task_id"] == successor
    assert not host.pending and list(queue.RUNNING) == [successor]


def test_unreadable_registry_refuses_the_fresh_continue_without_spending_its_claim(host, tmp_path, monkeypatch):  # noqa: F811
    from supervisor.continuation_admission import admit_continuation

    _interrupted_project_task(host, tmp_path)
    path = registry._registry_path(host.root)
    original = path.read_bytes()
    path.write_text("{torn", encoding="utf-8")
    refused = admit_continuation("pred-1", action_nonce=NONCE)
    assert refused["ok"] is False and refused["error"] == "project_routing_fence_lookup_failed", refused
    assert not host.pending and not queue.ADMISSION_RESERVATIONS
    path.write_bytes(original)
    retried = admit_continuation("pred-1", action_nonce=NONCE)  # the same nonce retries its own claim
    assert retried["ok"] is True and retried["successor_task_id"] == refused["successor_task_id"]
    assert not next(row for row in host.pending if row["id"] == retried["successor_task_id"])[
        "_project_admission"].get("legacy_basis")


@pytest.mark.parametrize("owner_control", ["pause", "restart"])
def test_project_recovery_keeps_owner_pause_and_restart_holds(host, tmp_path, monkeypatch, owner_control):  # noqa: F811
    from supervisor.events_budget import budget_hold_fact
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import resume_budget_paused_task

    prepared = copy.deepcopy(accepted(host, tmp_path))
    flag = host.root / "state" / "owner_restart_no_resume.flag"
    if owner_control == "pause":
        assert request_owner_pause("held", request_id="owner-pause")["ok"]
    else:
        flag.write_text("restart", encoding="utf-8")  # the owner's Restart marker, before its boot
    path, original = restore_unreadable(host)
    flag.unlink(missing_ok=True)  # the boot consumed the marker after the holds were persisted
    if owner_control == "restart":
        assert budget_hold_fact(host.pending[0])["reason"] == "owner_restart_hold"
    path.write_bytes(original)  # Project authority returns
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert not sent, "Project recovery never releases the owner's own control"
    assert not host.pending[0].get("_project_admission_restore_hold")
    resumed = resume_budget_paused_task("held")  # the owner's explicit Resume
    assert resumed["ok"], resumed
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    for key in ("_project_admission", "workspace_root", "drive_root", "text"):
        assert sent[0][key] == prepared[key], key
