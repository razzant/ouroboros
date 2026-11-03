"""Persistent model choices: application, Settings writes and owner stopping are separate outcomes.

These HTTP/remote-process cases share the established operation-lifetime fixture, but
own the Settings contention contract rather than the paid operation's lifetime.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from ouroboros import config, model_wait
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_review_operation_lifetime import (
    MODEL, OTHER_MODEL, TASK, _REMOTE_OPERATION, _decision, _forge, _lines,
    _rows, _stop_remote, until,
)
from tests import test_review_operation_lifetime as operation_cases

env = operation_cases.env


_PERSIST = dict(action="switch", model=OTHER_MODEL, credential_profile_id="", use_local=False, persist_role=True)


def _record_stop(root, kind, monkeypatch):
    """Wrap up, Stop now, current-generation Restart, or an unreadable stop state."""
    from ouroboros.cancel_intents import STOP_POLICY_FINALIZE, request_cancel

    (root / "state").mkdir(exist_ok=True)
    if kind == "graceful":
        request_cancel(root, TASK, reason="owner requested finalize-then-stop", source="test",
                       requested_stop_policy=STOP_POLICY_FINALIZE, allow_settled_target=True)
    elif kind == "immediate":
        request_cancel(root, TASK, reason="owner stop", source="test", allow_settled_target=True)
    elif kind == "owner_restart":
        from tests._work_stop_helpers import live_restart
        live_restart(root, monkeypatch)
    elif kind == "unknown":
        (root / "state" / "cancel_intents.json").write_text("{not json", encoding="utf-8")


@contextlib.contextmanager
def _persist_remote(f, monkeypatch, surface, *, launcher=False):
    """A real remote review controller parked on its reviewer wait, its seat saved in Settings,
    and the real decision ingress; ``post`` answers while a concurrent Settings Save holds the lock."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.owner_settings import (
        _owner_update_settings, fields_on_saved_document, settings_document_mutation)
    from ouroboros.gateway.task_decision import api_decision_answer
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_custody import current_custody_session_id
    from supervisor import queue as task_queue

    monkeypatch.setattr(task_queue, "RUNNING", {})
    path = f.root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    seat = {"subagent_id": "one", "recommended_use": "Reviews.", "review_eligible": True,
            "route": {"kind": "api_model", "target_id": MODEL}}
    path.write_text(json.dumps({"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [seat]})}))
    web = Starlette(routes=[Route("/api/decisions", api_decision_answer, methods=["POST"])])
    web.state.drive_root = f.root
    r = SimpleNamespace(path=path)
    r.canonical = lambda wait_id: (load_task_result(f.root, TASK).get("model_waits") or {}).get(wait_id) or {}
    r.saved_target = lambda: json.loads(
        json.loads(path.read_text())["OUROBOROS_SUBAGENTS"])["items"][0]["route"]["target_id"]

    def post(body, meanwhile, *, hold_until_answered=False):
        """POST ``body`` while a Settings Save holds the document lock until ``meanwhile`` ran
        (and, for a lock that must outlast its bound, until the answer arrived)."""
        held, answered, failures = threading.Event(), threading.Event(), []

        def save():
            with settings_document_mutation():
                held.set()
                try:
                    meanwhile(body["decision_id"].rsplit(":", 1)[1], body["request_id"])
                    if hold_until_answered:
                        assert answered.wait(30)
                    _owner_update_settings(fields_on_saved_document({"OUROBOROS_TOOL_TIMEOUT_SEC": 77}))
                except BaseException as error:  # reported below, after the lock is released
                    failures.append(error)

        holder = threading.Thread(target=save, daemon=True)
        holder.start()
        assert held.wait(10)
        try:
            return r.client.post("/api/decisions", json=body)
        finally:
            answered.set()
            holder.join(60)
            assert not holder.is_alive() and not failures, failures

    def claimed_then(close):
        def meanwhile(wait_id, request_id):
            until(lambda: (r.canonical(wait_id).get("pending_action") or {}).get("request_id") == request_id,
                  timeout=30)
            close(wait_id, request_id)
        return meanwhile

    def applied_then(close):
        def meanwhile(wait_id, request_id):
            until(lambda: r.canonical(wait_id).get("applied_request_id") == request_id, timeout=30)
            close(wait_id, request_id)
        return meanwhile

    def forge(wait_id):
        """A row of the same LIVE controller under its own open pointer: it stays waiting."""
        owner_id = f"forged-{wait_id}"
        forged = {**r.row, "wait_id": wait_id, "model_wait_owner_id": owner_id,
                  "review_operation": {**r.block, "owner_id": owner_id}}
        _forge(f.root, wait_id, forged, dict(r.pointer))
        return forged

    r.post, r.claimed_then, r.applied_then, r.forge = post, claimed_then, applied_then, forge
    r.unforge = lambda wait_id, forged: _forge(f.root, wait_id, forged, dict(r.pointer), remove=True)
    command = [sys.executable, "-c", _REMOTE_OPERATION, str(f.root), current_custody_session_id(), TASK,
               surface, str(os.getpid())]
    if launcher:  # Exercise Windows venv topology on every host: the interpreter is a grandchild.
        command = [sys.executable, "-c",
                   "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))", *command]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=os.getcwd())
    r.out, reader = _lines(child.stdout)
    executing = {}
    try:
        banner = json.loads(r.out.get(timeout=60))
        executing.update(banner, birth=process_start_time(banner["pid"]))
        r.row = until(lambda: [row for row in _rows(f.root) if row.get("credential_harness")], timeout=30)[0]
        write_task_result(f.root, TASK, "completed", result="answer")
        r.block = r.row["review_operation"]
        r.pointer = load_task_result(f.root, TASK)["review_operations"][r.block["owner_id"]]
        with TestClient(web) as client:
            r.client = client
            yield r
    finally:
        intents = f.root / "state" / "cancel_intents.json"
        if intents.exists() and intents.read_text(encoding="utf-8", errors="replace") == "{not json":
            intents.unlink()  # the synthetic unreadable stop state never outlives its case
        teardown = _stop_remote(child, reader, executing)
    assert teardown["interpreter_exited"] and not teardown["reader_running"], teardown


@pytest.mark.serial
@pytest.mark.parametrize("surface, stop", [("task_acceptance", ""), ("plan_review", ""),
                                           ("task_acceptance", "immediate"), ("plan_review", "owner_restart"),
                                           ("task_acceptance", "graceful"), ("plan_review", "unknown")])
def test_a_role_save_that_waits_for_settings_meets_its_own_remote_receipt(env, monkeypatch, surface, stop):
    """A legitimate Settings Save holds the document lock while the owner's switch with
    "save for this role" is claimed through the real ingress. The remote controller
    consumes and applies that exact claim meanwhile, so its row stops waiting and the
    controller reads closed by the time the role save runs. Its own receipt answers:
    applied once, never refused. Wrap up lets the reviewer finish and keeps the separately
    confirmed save (owner choice, 2026-10-14); a recorded Stop now or Panic, or an unreadable
    state, leaves it applied, not saved. A closure that did not apply this exact claim still
    refuses and saves nothing."""
    from ouroboros.owner_mailbox import KIND_MODEL_WAIT, drain_owner_entries
    from ouroboros.task_store import patch_document

    f = env

    def end_without_applying(wait_id, _request_id):
        owner_id = r.canonical(wait_id)["model_wait_owner_id"]
        patch_document(f.root, TASK, lambda current: {**current, "review_operations": {
            key: value for key, value in current["review_operations"].items() if key != owner_id}})

    def resolved_with(receipt):
        def close(wait_id, request_id):
            model_wait.mutate_wait(f.root, TASK, wait_id, lambda previous: {
                **{key: value for key, value in previous.items() if key != "pending_action"},
                "state": "resolved", **receipt(request_id)})
        return close

    with _persist_remote(f, monkeypatch, surface) as r:
        if not stop:
            # Rows of the same LIVE controller under their own open pointers: a closure
            # while the save waited that did not apply THIS claim refuses and saves nothing.
            for wait_id, close in (
                    ("ended-between", end_without_applying),
                    ("applied-another", resolved_with(lambda _request_id: {"applied_request_id": "another"})),
                    ("foreign-owner", resolved_with(lambda request_id: {
                        "applied_request_id": request_id, "model_wait_owner_id": "another-owner"}))):
                forged = r.forge(wait_id)
                try:
                    refused = r.post(_decision(forged, f"persist-{wait_id}", **_PERSIST), r.claimed_then(close))
                    answer = refused.json()
                    assert refused.status_code == 409 and answer["reason_code"] == "task_not_live", answer
                    assert (answer["applied"], answer["saved"]) == (False, False) and r.saved_target() == MODEL, answer
                    assert answer["wait"].get("wait_id") == wait_id, "the refusal carries the row as it now is"
                    assert "pending_action" not in answer["wait"] and "pending_action" not in r.canonical(wait_id)
                finally:
                    r.unforge(wait_id, forged)
        # The real operation consumes and applies the exact claim while the save waits.
        switch = _decision(r.row, "persist-remote", **_PERSIST)
        response = r.post(switch, r.applied_then(lambda *_: _record_stop(f.root, stop, monkeypatch)))
        answer = response.json()
        assert response.status_code == 202, answer
        assert (answer["applied"], answer["duplicate"], answer["state"]) == (True, False, "resolved")
        assert answer["wait"]["applied_request_id"] == "persist-remote"
        # Applied and saved are separate facts: Stop now, Panic or an unknown state leave Settings as they were.
        kept = stop in ("", "graceful")
        assert answer["saved"] is kept
        assert r.saved_target() == (OTHER_MODEL if kept else MODEL)
        assert (r.canonical(r.row["wait_id"]).get("saved_request_id") == "persist-remote") is kept
        assert json.loads(r.path.read_text())["OUROBOROS_TOOL_TIMEOUT_SEC"] == 77, "the concurrent Save landed too"
        closed = json.loads(r.out.get(timeout=60))
        assert closed["operation"] == "closed" and closed["calls"].count(MODEL) == 1
        assert closed["calls"] == [MODEL, OTHER_MODEL] if stop in ("", "graceful", "unknown") else \
            closed["calls"].count(OTHER_MODEL) <= 1, "one application by the operation itself, never a repeated action"
        assert drain_owner_entries(f.root, TASK, set(), kinds={KIND_MODEL_WAIT}) == []
        # The same request after the operation closed is refused before any claim or save.
        stamp = r.path.stat().st_mtime_ns
        late = r.client.post("/api/decisions", json=switch)
        assert late.status_code == 409 and late.json()["reason_code"] == "task_not_live"
        assert r.path.stat().st_mtime_ns == stamp


@pytest.mark.serial
@pytest.mark.parametrize("stop,launcher", [("", False), ("graceful", False), ("immediate", False),
                                         ("owner_restart", False), ("unknown", False),
                                         ("owner_restart", True)])
def test_a_role_save_for_a_still_waiting_claim_obeys_the_same_stop_record(env, monkeypatch, stop, launcher):
    """The controller is live and has not consumed the claim (a row under its own open pointer)
    when a stop record lands while the Settings lock is held. Clear: the role is saved and the
    claim stays pending for its controller. Wrap up (finalize-then-stop) leaves the operation
    live — the claim stays accepted, the reviewer still finishes and the confirmed role save
    lands, exactly as on the applied path. Stop now, Restart or an unreadable stop state leave
    no provably live controller: refused, discarded, nothing saved."""
    from ouroboros.owner_mailbox import KIND_MODEL_WAIT, drain_owner_entries

    f = env
    with _persist_remote(f, monkeypatch, "task_acceptance", launcher=launcher) as r:
        wait_id = "live-claim"
        forged = r.forge(wait_id)
        try:
            response = r.post(_decision(forged, "persist-live", **_PERSIST),
                              r.claimed_then(lambda *_: _record_stop(f.root, stop, monkeypatch)))
            answer = response.json()
            kept = stop in ("", "graceful")
            if kept:
                assert response.status_code == 202, answer
                assert (answer["applied"], answer["saved"], answer["state"]) == (False, True, "waiting"), answer
                assert answer["wait"]["pending_action"]["request_id"] == "persist-live", "accepted, not applied"
            else:
                assert response.status_code == 409 and answer["reason_code"] == "task_not_live", answer
                assert (answer["applied"], answer["saved"]) == (False, False), answer
                assert "pending_action" not in r.canonical(wait_id)
            assert r.saved_target() == (OTHER_MODEL if kept else MODEL)
            assert (r.canonical(wait_id).get("saved_request_id") == "persist-live") is kept
            assert json.loads(r.path.read_text())["OUROBOROS_TOOL_TIMEOUT_SEC"] == 77
            if stop == "graceful":
                # Under Wrap up the identical request is the same claim (no second write or application),
                # and a stale observation is refused before any claim or Settings write.
                stamp = r.path.stat().st_mtime_ns
                repeat = r.client.post("/api/decisions", json=_decision(forged, "persist-live", **_PERSIST))
                assert repeat.status_code == 200, repeat.json()
                assert (repeat.json()["duplicate"], repeat.json()["saved"], repeat.json()["applied"]) == (True, True, False)
                stale_row = r.forge("stale-claim")
                try:
                    stale = r.client.post("/api/decisions", json={
                        **_decision(stale_row, "persist-stale", **_PERSIST), "revision": stale_row["revision"] + 7})
                    refused = stale.json()
                    assert stale.status_code == 409 and refused["reason_code"] == "stale_model_wait", refused
                    assert (refused["applied"], refused["saved"]) == (False, False)
                    assert "pending_action" not in r.canonical("stale-claim")
                finally:
                    r.unforge("stale-claim", stale_row)
                assert r.path.stat().st_mtime_ns == stamp and r.saved_target() == OTHER_MODEL, "no second Settings write"
        finally:
            r.unforge(wait_id, forged)
        if stop == "unknown":
            (f.root / "state" / "cancel_intents.json").unlink()
        if stop in ("", "graceful", "unknown"):
            # The reviewer still finishes (under Wrap up too): a task-only switch is applied once.
            finish = r.client.post("/api/decisions", json=_decision(r.row, "finish", **{**_PERSIST, "persist_role": False}))
            assert finish.status_code == 202, finish.json()
            assert json.loads(r.out.get(timeout=60)) == {"operation": "closed", "calls": [MODEL, OTHER_MODEL]}
        else:
            closed = json.loads(r.out.get(timeout=60))
            assert closed["operation"] == "closed" and closed["calls"] == [MODEL], "Stop now and Panic end the reviewer"
        assert drain_owner_entries(f.root, TASK, set(), kinds={KIND_MODEL_WAIT}) == []


@pytest.mark.serial
def test_a_role_save_whose_settings_lock_times_out_still_reports_its_applied_claim(env, monkeypatch):
    """The Settings lock outlasts its bound while the remote controller applies the exact claim:
    the save failure stays a 503 with saved false, the current receipt says applied, and
    nothing is written or dispatched again."""
    from ouroboros.owner_mailbox import KIND_MODEL_WAIT, drain_owner_entries

    f = env
    monkeypatch.setenv("OUROBOROS_SETTINGS_DOCUMENT_LOCK_TIMEOUT_SEC", "1")
    with _persist_remote(f, monkeypatch, "task_acceptance") as r:
        response = r.post(_decision(r.row, "persist-busy", **_PERSIST), r.applied_then(lambda *_: None),
                          hold_until_answered=True)
        answer = response.json()
        assert response.status_code == 503 and answer["reason_code"] == "model_wait_decision_failed", answer
        assert "another settings write is still in progress" in answer["error"]
        assert (answer["applied"], answer["saved"]) == (True, False), answer
        assert answer["wait"]["applied_request_id"] == "persist-busy" and answer["wait"]["state"] == "resolved"
        assert r.saved_target() == MODEL and "saved_request_id" not in r.canonical(r.row["wait_id"])
        assert json.loads(r.out.get(timeout=60)) == {"operation": "closed", "calls": [MODEL, OTHER_MODEL]}
        assert drain_owner_entries(f.root, TASK, set(), kinds={KIND_MODEL_WAIT}) == []
