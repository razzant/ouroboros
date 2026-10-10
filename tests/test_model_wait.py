"""Model-wait ownership, controls and execution clocks preserve the live task."""

import contextvars
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import queue
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros import owner_mailbox as mailbox
from ouroboros import model_wait
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_llm_claudexor import MODEL, result, ledger, setup as gateway_fixture

setup = gateway_fixture


def test_wait_controls_never_enter_owner_dialogue_or_steal_other_kinds(tmp_path):
    mailbox.write_owner_message(tmp_path, "change the answer", "task-one", msg_id="owner-one")
    mailbox.write_owner_message(tmp_path, '{"wait_id":"wait-one"}', "task-one",
                                msg_id="wait-action", kind=mailbox.KIND_MODEL_WAIT)
    mailbox.write_owner_message(tmp_path, "finalize", "task-one", msg_id="stop-one",
                                kind=mailbox.KIND_FINALIZE_NOW)
    waiter_seen = set()
    waits = mailbox.drain_owner_entries(tmp_path, "task-one", waiter_seen,
                                        kinds={mailbox.KIND_MODEL_WAIT})
    assert [entry["msg_id"] for entry in waits] == ["wait-action"]
    assert waiter_seen == {"wait-action"}
    normal_seen = set()
    normal = mailbox.drain_owner_entries(tmp_path, "task-one", normal_seen)
    assert [entry["msg_id"] for entry in normal] == ["owner-one", "stop-one"]
    assert normal_seen == {"owner-one", "stop-one"}
    assert mailbox.drain_owner_entries(tmp_path, "task-one", waiter_seen,
                                        kinds={mailbox.KIND_MODEL_WAIT}) == []


def test_revoked_wait_control_is_withheld_by_selective_reader(tmp_path):
    mailbox.write_owner_message(tmp_path, "request", "task-one", msg_id="wait-action",
                                kind=mailbox.KIND_MODEL_WAIT)
    assert mailbox.revoke_owner_control(tmp_path, "task-one", "wait-action")
    assert mailbox.drain_owner_entries(tmp_path, "task-one", kinds={mailbox.KIND_MODEL_WAIT}) == []


def test_task_retry_retires_old_wait_control_and_keeps_owner_text(tmp_path):
    mailbox.write_owner_message(tmp_path, "owner instruction", "task-one", msg_id="owner-one")
    mailbox.write_owner_message(tmp_path, "request", "task-one", msg_id="wait-action",
                                kind=mailbox.KIND_MODEL_WAIT)
    assert mailbox.reset_attempt_controls_for_retry(tmp_path, "task-one") == 1
    assert mailbox.reset_attempt_controls_for_retry(tmp_path, "task-one") == 0
    assert mailbox.drain_owner_entries(tmp_path, "task-one", kinds={mailbox.KIND_MODEL_WAIT}) == []
    assert mailbox.drain_owner_messages(tmp_path, "task-one") == ["owner instruction"]


def test_parallel_quota_waits_charge_union_to_task_and_own_duration_to_each_slot(tmp_path):
    controller = model_wait.TaskModelWait(task={"id": "task-one"}, drive_root=tmp_path,
                                         event_queue=None, worker_slot_held=False)
    controller.quota_enter("wait-a", "slot-a", now=10.0)
    controller.quota_enter("wait-b", "slot-b", now=15.0)
    controller.quota_enter("wait-a", "slot-a", now=17.0)
    controller.quota_leave("wait-a", "slot-a", now=20.0)
    controller.quota_leave("unknown", "slot-b", now=22.0)
    assert controller.paused_seconds(now=25.0) == 15.0
    assert controller.paused_seconds("slot-a", now=25.0) == 10.0
    assert controller.paused_seconds("slot-b", now=25.0) == 10.0
    controller.quota_leave("wait-b", "slot-b", now=30.0)
    assert controller.paused_seconds(now=100.0) == 20.0
    assert controller.paused_seconds("slot-b", now=100.0) == 15.0


def test_wait_context_is_task_scoped_and_shared_by_copied_threads(tmp_path):
    assert model_wait.current_model_wait() is None
    with model_wait.task_model_wait_scope(task={"id": "task-one"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False) as controller:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(contextvars.copy_context().run, model_wait.current_model_wait)
            second = pool.submit(contextvars.copy_context().run, model_wait.current_model_wait)
            assert first.result() is second.result() is controller
        assert not controller.closed
    assert controller.closed and model_wait.current_model_wait() is None


def test_reprepare_callback_is_lexical_and_does_not_mutate_supplied_kwargs(tmp_path):
    with model_wait.task_model_wait_scope(task={"id": "task-one"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False) as controller:
        kwargs = {"messages": [{"role": "user", "content": "old"}]}

        def prepare(values):
            values["messages"][0]["content"] = "new"
            return values

        with controller.register_reprepare("main", prepare):
            assert controller.reprepare("main", kwargs)["messages"][0]["content"] == "new"
            assert kwargs["messages"][0]["content"] == "old"
        assert controller.reprepare("main", kwargs) == kwargs


def test_existing_physical_context_requires_its_reprepare_callback(tmp_path, monkeypatch):
    from ouroboros import usage_accounting

    monkeypatch.setattr(usage_accounting, "current_physical_attempt_context", lambda: object())
    controller = model_wait.TaskModelWait(task={"id": "task-one"}, drive_root=tmp_path,
                                         event_queue=None, worker_slot_held=False)
    with pytest.raises(model_wait.ModelWaitInterrupted) as raised:
        controller.reprepare("main", {"messages": []})
    assert raised.value.control_reason == "model_wait_reprepare_required"


@pytest.fixture
def live_wait(setup, monkeypatch):
    from supervisor import queue as task_queue
    from ouroboros.gateway import task_model_wait as gateway

    root, transport, client = setup
    task = {"id": "task-one", "_attempt": 1, "drive_root": str(root)}
    monkeypatch.setattr(task_queue, "RUNNING", {"task-one": {"task": task, "attempt": 1}})
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(client, "claudexor_model_sources", lambda: {
        "sources": [{"id": "codex", "credentialHarness": "actual-harness", "label": "Fixture"}]})
    catalog = {"source": "codex", "credentialProfileId": "account-a", "models": [{"id": "exact-model"}]}
    monkeypatch.setattr(client, "claudexor_model_catalog", lambda *_, **_kw: catalog)
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(monotonic=__import__("time").monotonic,
                                                            time=__import__("time").time,
                                                            sleep=lambda _seconds: None))
    write_task_result(root, "task-one", "running")
    events = queue.Queue()
    with model_wait.task_model_wait_scope(task=task, drive_root=root, event_queue=events,
                                          worker_slot_held=True) as controller:
        yield root, transport, client, controller, events, lambda body: gateway._decide(root, body)


def _decision_clients(root):
    """Real Web/Host ingress sharing one root and the installation's live getter."""
    from contextlib import ExitStack, contextmanager
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.gateway.task_decision import api_decision_answer
    from tests.test_host_service_api import _seed_token

    @contextmanager
    def clients():
        _seed_token(root, permissions=["inject_chat"])
        web = Starlette(routes=[Route("/api/decisions", api_decision_answer, methods=["POST"])])
        web.state.drive_root = root
        host = create_host_service_app(root)
        with ExitStack() as stack:
            web_client = stack.enter_context(TestClient(web))
            host_client = stack.enter_context(TestClient(host, headers={"x-skill-token": "token"}))
            yield {"web": lambda body: web_client.post("/api/decisions", json=body),
                   "host": lambda body: host_client.post("/chat/decision", json=body)}
    return clients()


@pytest.fixture
def elapsed_quota_wait(live_wait, monkeypatch):
    """Advance catalog time explicitly; an immediate lookup may share one OS tick."""
    _root, _transport, client, controller, _events, _decide = live_wait
    clock = SimpleNamespace(now=controller.started_monotonic)
    monkeypatch.setattr(model_wait.time, "monotonic", lambda: clock.now)
    catalog = client.claudexor_model_catalog

    def elapsed_catalog(*args, **kwargs):
        clock.now += 2.5
        return catalog(*args, **kwargs)

    monkeypatch.setattr(client, "claudexor_model_catalog", elapsed_catalog)
    return live_wait


def _refusal(code="subscription_window_exhausted"):
    return result(outcome="failed", problem={"code": code, "message": "fixture resource refusal",
                                             "context": {"resetsAt": "2099-01-01T00:00:00Z"}})


def _action_for(event, action, **fields):
    return {"request_id": "request-one", "decision_id": f"model_wait:{event['task_id']}:{event['wait_id']}",
            "revision": event["revision"], "action": action, **fields}


def test_quota_wait_rejoins_call_without_replaying_tools_and_keeps_ledger(elapsed_quota_wait):
    root, transport, client, controller, events, _decide = elapsed_quota_wait
    transport.results = [_refusal(), result()]
    transport.dispatch = ["not_started", "response_received"]
    messages = [{"role": "user", "content": "request"}, result()["message"],
                {"role": "tool", "tool_call_id": "a", "content": "already executed"},
                {"role": "tool", "tool_call_id": "b", "content": "already executed too"}]
    answer, usage = client.chat(messages, MODEL, model_role="main")
    assert answer == result()["message"]
    assert transport.uploads[0][0]["messages"] == transport.uploads[1][0]["messages"]
    # The never-started release retains its failure evidence before the answered repeat.
    attempts = ledger(root)
    assert [(row["state"], row["revision"]) for row in attempts] == [("released", 4), ("settled", 3)]
    assert attempts[0]["physical_failure"]["stage"] == "raised_exception"
    assert len(usage["ledger_attempt_ids"]) == 2
    rows = list(events.queue)
    assert rows[0]["state"] == "waiting" and rows[-1]["state"] == "resolved"
    assert rows[-1]["resolution"] == "resource_available" and rows[-1]["credential_harness"] == "actual-harness"
    assert rows[-1]["revision"] > rows[0]["revision"]
    assert all(row["worker_slot_held"] is True and row["is_progress"] is False for row in rows)
    assert load_task_result(root, "task-one")["model_waits"][rows[0]["wait_id"]]["state"] == "resolved"
    assert controller.paused_seconds() == pytest.approx(2.5)


@pytest.mark.parametrize("advisory", [False, True])
def test_auto_wait_requests_its_model_and_resumes_when_compatible_second_account_recovers(live_wait, monkeypatch, advisory):
    _root, transport, client, _controller, events, _decide = live_wait
    recovered = result()
    recovered["route"].update(credentialProfileId="account-b", accountFingerprint="identity-b")
    transport.results = [_refusal(), recovered]
    transport.dispatch = ["not_started", "response_received"]
    calls = []

    def catalog(source, profile=None, *, requested_model=None):
        calls.append((source, profile, requested_model))
        assert requested_model == "exact-model"
        return {"source": source, "credentialProfileId": "account-b",
                "accountFingerprint": "identity-b", "models": [] if advisory else [{"id": "exact-model"}],
                "admission": {"requestedModel": requested_model, "inventoryAbsence": "advisory" if advisory else "authoritative"}}

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    _, usage = client.chat([{"role": "user", "content": "continue exact model"}], MODEL, model_role="main")
    assert calls == [("codex", None, "exact-model")]
    assert list(events.queue)[-1]["resolution"] == "resource_available"
    assert len(transport.uploads) == 2
    assert usage["model_role_route"]["credential_profile_id"] == ""  # Auto stays unpinned.
    assert usage["claudexor"]["route"]["credentialProfileId"] == "account-b"


def test_owner_switch_is_local_to_role_and_persists_only_when_checked(live_wait, monkeypatch):
    root, transport, client, controller, events, decide = live_wait
    transport.results = [_refusal(), result(), result(), result()]
    transport.dispatch = ["not_started", "response_received", "response_received", "response_received"]
    acted = False

    def catalog(*_, **_kw):
        nonlocal acted
        if not acted:
            current = list(events.queue)[-1]
            response = decide(_action_for(current, "switch", model=MODEL, credential_profile_id="account-b",
                                          use_local=False, persist_role=False))
            assert response.status_code == 202 and json.loads(response.body)["saved"] is False
            acted = True
        raise ClaudexorUnavailable("subscription_window_exhausted", "not ready")

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    _, usage = client.chat([{"role": "user", "content": "same model"}], MODEL, model_role="light")
    client.chat([{"role": "user", "content": "next"}], MODEL, model_role="light")
    client.chat([{"role": "user", "content": "main unchanged"}], MODEL, model_role="main")
    assert [entry[0]["account"] for entry in transport.uploads] == [
        {"mode": "auto"}, {"mode": "pin", "profileId": "account-b"},
        {"mode": "pin", "profileId": "account-b"}, {"mode": "auto"}]
    assert usage["model_role_route"]["role"] == "light"
    assert usage["model_role_route"]["credential_profile_id"] == "account-b"
    assert controller.overrides["light"]["model_account_override"] == "account-b"
    assert not (root / "settings.json").exists()


def test_owner_decision_revision_pending_and_replay_fences(live_wait):
    root, _transport, _client, controller, _events, decide = live_wait
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "main"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    event = {**row, "task_id": "task-one"}
    body = _action_for(event, "auto_continue", auto_continue=False)
    first = decide(body)
    assert first.status_code == 202 and json.loads(first.body)["applied"] is False
    replay = decide(body)
    assert replay.status_code == 200 and json.loads(replay.body)["duplicate"] is True
    competing = decide({**body, "request_id": "another-request"})
    assert competing.status_code == 409 and json.loads(competing.body)["reason_code"] == "model_wait_action_pending"
    controller.waits["wait-one"] = dict(row)
    controller.revision = 1
    controller._drain_controls()
    assert controller.waits["wait-one"]["auto_continue"] is False
    applied = decide(body)
    assert applied.status_code == 200 and json.loads(applied.body)["applied"] is True
    stale = decide({**body, "request_id": "new-request"})
    assert stale.status_code == 409 and json.loads(stale.body)["reason_code"] == "stale_model_wait"


def test_wait_projection_refuses_corrupt_existing_task_result(tmp_path):
    directory = tmp_path / "task_results"
    directory.mkdir()
    path = directory / "task-one.json"
    path.write_bytes(b'{"broken":')
    with pytest.raises(ValueError):
        model_wait.mutate_wait(tmp_path, "task-one", "wait-one", lambda _: {"state": "waiting"})
    assert path.read_bytes() == b'{"broken":'


def test_dated_pool_of_unknown_cause_waits_visibly_as_unavailable_never_as_quota(elapsed_quota_wait):
    """#1409 case 1: an engine-dated pool refusal is neither quota nor sign-in. With no later
    route it takes the visible, interruptible live wait under its own reason and heals on the
    SAME call; it never sleeps to the reset and never reads as quota."""
    root, transport, client, controller, events, _decide = elapsed_quota_wait
    transport.results = [_refusal("credential_pool_exhausted"), result()]
    transport.dispatch = ["not_started", "response_received"]
    answer, usage = client.chat([{"role": "user", "content": "hi"}], MODEL, model_role="main")
    assert answer == result()["message"] and len(usage["ledger_attempt_ids"]) == 2
    rows = list(events.queue)
    assert rows[0]["reason"] == rows[-1]["reason"] == "unavailable"
    assert rows[0]["reset_at"] == "2099-01-01T00:00:00Z" and rows[0]["quota_clock"]["active"] is False
    assert rows[-1]["resolution"] == "resource_available" and controller.paused_seconds() == 0
    assert load_task_result(root, "task-one")["model_waits"][rows[0]["wait_id"]]["reason"] == "unavailable"


def test_async_cancellation_resolves_only_its_wait_and_keeps_shared_task(live_wait, monkeypatch):
    root, transport, client, controller, events, _decide = live_wait
    transport.results = [_refusal()]
    transport.dispatch = ["not_started"]
    entered = threading.Event()

    def catalog(*_, **_kw):
        entered.set()
        raise ClaudexorUnavailable("subscription_window_exhausted", "not ready")

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)

    async def run():
        task = asyncio.create_task(client.chat_async([{"role": "user", "content": "hi"}], MODEL, model_role="main"))
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        terminal = None
        while terminal is None:
            event = await asyncio.to_thread(events.get, True, 2)
            if event["state"] == "resolved":
                terminal = event
        assert terminal["resolution"] == "caller_cancelled"

    asyncio.run(run())
    assert not controller.closed and len(transport.accepted_operations) == 1
    assert all(row["state"] == "resolved" for row in load_task_result(root, "task-one")["model_waits"].values())


def test_auth_wait_does_not_pause_execution_clock(live_wait, monkeypatch):
    _root, transport, client, controller, _events, _decide = live_wait
    transport.results = [_refusal("auth_required"), result()]
    transport.dispatch = ["not_started", "response_received"]
    original = client.claudexor_model_catalog

    def catalog(*args, **kwargs):
        assert controller.paused_seconds() == 0
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    client.chat([{"role": "user", "content": "hi"}], MODEL, model_role="light")
    assert controller.paused_seconds() == 0


def test_quota_pause_cannot_move_explicit_calendar_deadline(tmp_path):
    task = {"id": "task-one", "metadata": {"deadline_at": "2000-01-01T00:00:00Z"}}
    controller = model_wait.TaskModelWait(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    controller.quota_enter("wait-a", "")
    assert controller.control_reason() == "deadline"
    controller.task["metadata"].clear()
    with model_wait.calendar_scope("2000-01-01T00:00:00Z"):
        assert controller.control_reason() == "deadline"
    assert controller.control_reason() is None


def test_prepared_send_binds_new_main_authority_and_restores_outer_context():
    from ouroboros import usage_accounting as accounting

    first = accounting.PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="cold_estimate", route_fp="first",
        round_id="round-one", target_total_tokens=None, capacity_total_tokens=None,
        context_target_miss=False, automatic_pass_used=False)
    from dataclasses import replace
    second = replace(first, route_fp="second")
    original_predicate, new_predicate = lambda _: True, lambda _: False
    with accounting.bind_physical_attempt_context(first, original_predicate):
        with model_wait.prepared_call_scope(model_wait.PreparedModelCall({"model": "new"}, second, new_predicate)) as values:
            assert values == {"model": "new"}
            assert accounting.current_physical_attempt_context() is second
            assert accounting.current_physical_attempt_predicate() is new_predicate
        assert accounting.current_physical_attempt_context() is first
        assert accounting.current_physical_attempt_predicate() is original_predicate


def test_same_future_keeps_execution_budget_during_quota_pause(tmp_path, monkeypatch):
    from concurrent.futures import TimeoutError as FutureTimeout

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(monotonic=lambda: clock.now))
    with model_wait.task_model_wait_scope(task={"id": "task-one"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False) as controller:
        controller.quota_enter("wait-a", "")

        class Future:
            timeouts = []

            def result(self, timeout):
                self.timeouts.append(timeout)
                if len(self.timeouts) == 1:
                    clock.now = 20.0
                    raise FutureTimeout()
                return "done"

            def done(self):
                return False

        future = Future()
        assert model_wait.future_result(future, 10.0) == "done"
        assert future.timeouts == [10.0, 10.0]


def test_two_waits_can_receive_same_request_id_without_control_collision(live_wait):
    root, _transport, _client, controller, _events, decide = live_wait
    for index, role in enumerate(("main", "light"), 1):
        row = {"wait_id": f"wait-{index}", "revision": index, "task_attempt": 1, "state": "waiting", "role": role}
        model_wait.mutate_wait(root, "task-one", row["wait_id"], lambda _, row=row: row)
        controller.waits[row["wait_id"]] = dict(row)
        response = decide(_action_for({**row, "task_id": "task-one"}, "auto_continue", auto_continue=False))
        assert response.status_code == 202
    controller.revision = 2
    controller._drain_controls()
    assert all(row["auto_continue"] is False for row in controller.waits.values())
    assert controller.auto_continue == {"main": False, "light": False}


def test_permanent_switch_uses_existing_owner_writer_once(live_wait, monkeypatch):
    from ouroboros import config

    root, _transport, _client, controller, _events, decide = live_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "light"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    controller.waits["wait-one"] = dict(row)
    controller.revision = 1
    body = _action_for({**row, "task_id": "task-one"}, "switch", model=MODEL,
                       credential_profile_id="account-b", use_local=False, persist_role=True)
    response = decide(body)
    assert response.status_code == 202 and json.loads(response.body)["saved"] is True
    settings = json.loads((root / "settings.json").read_text())
    assert settings["OUROBOROS_MODEL_LIGHT"] == MODEL
    assert json.loads(settings["OUROBOROS_MODEL_ACCOUNTS"])["light"] == "account-b"
    stamp = (root / "settings.json").stat().st_mtime_ns
    replay = decide(body)
    assert replay.status_code == 200 and json.loads(replay.body)["saved"] is True
    assert (root / "settings.json").stat().st_mtime_ns == stamp


def test_invalid_persistent_choice_and_failed_revalidation_do_not_wedge_action(live_wait, monkeypatch):
    from ouroboros import config, model_slots

    root, _transport, _client, controller, _events, decide = live_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "light"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    body = _action_for({**row, "task_id": "task-one"}, "switch", model="openai::model",
                       credential_profile_id="invalid-api-pin", use_local=False, persist_role=True)
    assert decide(body).status_code == 400
    assert "pending_action" not in load_task_result(root, "task-one")["model_waits"]["wait-one"]
    original = model_slots.apply_model_role_override
    calls = []

    def changed_setting(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("role changed between preview and atomic save")
        return original(*args, **kwargs)

    monkeypatch.setattr(model_slots, "apply_model_role_override", changed_setting)
    valid = {**body, "model": MODEL, "credential_profile_id": "account-a"}
    refused = decide(valid)
    assert refused.status_code == 503 and json.loads(refused.body)["saved"] is False
    assert "pending_action" not in load_task_result(root, "task-one")["model_waits"]["wait-one"]
    assert not (root / "memory/owner_mailbox/task-one.jsonl").exists()
    corrected = decide({**valid, "request_id": "corrected"})
    assert corrected.status_code == 202


def test_metadata_revision_drift_does_not_discard_accepted_action(live_wait):
    root, _transport, _client, controller, _events, decide = live_wait
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "main"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    controller.waits["wait-one"] = dict(row)
    controller.revision = 1
    assert decide(_action_for({**row, "task_id": "task-one"}, "auto_continue", auto_continue=False)).status_code == 202
    controller.waits["wait-one"]["credential_harness"] = "newly-discovered-harness"
    controller._publish(controller.waits["wait-one"])
    assert controller.waits["wait-one"]["revision"] == 2
    controller._drain_controls()
    stored = load_task_result(root, "task-one")["model_waits"]["wait-one"]
    assert stored["auto_continue"] is False and stored["applied_request_id"] == "request-one"
    assert stored["revision"] == 3 and "pending_action" not in stored


def test_wait_context_copy_does_not_import_parents_physical_capture(tmp_path):
    from ouroboros import usage_accounting as accounting

    request = accounting.AttemptRequest(model=MODEL, provider="claudexor", drive_root=tmp_path,
                                         force_unknown_reservation=True)
    accounting.execute_physical_attempt(request, lambda: {}, extractor=lambda _: ({}, None, False))
    assert accounting.last_physical_attempt_capture() is not None
    with model_wait.task_model_wait_scope(task={"id": "task-one"}, drive_root=tmp_path,
                                          event_queue=None, worker_slot_held=False) as controller:
        def inspect_context():
            return model_wait.current_model_wait(), accounting.last_physical_attempt_capture()

        with ThreadPoolExecutor(max_workers=1) as pool:
            actual, capture = pool.submit(model_wait.copy_wait_context().run, inspect_context).result()
        assert actual is controller and capture is None


def test_generic_client_parameter_flattens_kwargs_and_keeps_string_product(live_wait):
    _root, _transport, client, _controller, _events, _decide = live_wait
    calls = []

    @model_wait.model_waitable(client_parameter="client")
    def child_call(client, *, model_role="vision", **kwargs):
        calls.append((client, model_role, kwargs))
        return "image description", {"ledger_attempt_ids": ["existing-attempt"]}

    product, usage = child_call(client, model=MODEL, prompt="describe", images=[])
    assert product == "image description" and len(calls) == 1
    assert calls[0][0] is client and calls[0][1] == "vision" and calls[0][2]["prompt"] == "describe"
    assert "kwargs" not in calls[0][2]
    assert usage["model_role_route"]["role"] == "vision"


@pytest.mark.parametrize("delta", [{"revision": True}, {"action": []}, {"model": 3},
                                    {"credential_profile_id": None}, {"use_local": "false"},
                                    {"persist_role": "true"}, {"comment": "not this family"}])
def test_malformed_switch_is_refused_before_effects(live_wait, delta):
    root, _transport, _client, _controller, _events, decide = live_wait
    body = {"request_id": "request-one", "decision_id": "model_wait:task-one:wait-one", "revision": 1,
            "action": "switch", "model": MODEL, "credential_profile_id": "", "use_local": False, "persist_role": False}
    response = decide({**body, **delta})
    assert response.status_code == 400 and json.loads(response.body)["ok"] is False
    assert not (root / "settings.json").exists()
    assert not (root / "memory/owner_mailbox/task-one.jsonl").exists()


def test_terminal_and_cancel_pending_tasks_refuse_stale_controls(live_wait, monkeypatch):
    from supervisor import queue as task_queue
    from ouroboros import cancel_intents

    _root, _transport, _client, _controller, _events, decide = live_wait
    body = _action_for({"task_id": "task-one", "wait_id": "wait-one", "revision": 1}, "retry")
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda *_: True)
    assert json.loads(decide(body).body)["reason_code"] == "cancel_pending"
    task_queue.RUNNING.clear()
    assert json.loads(decide(body).body)["reason_code"] == "task_not_live"


def test_supervisor_reordering_keeps_latest_union_clock_and_never_marks_progress(live_wait):
    from supervisor import task_model_wait as projection
    from supervisor import queue as task_queue

    root, _transport, _client, _controller, _events, _decide = live_wait
    pushed, persisted = [], []
    ctx = SimpleNamespace(RUNNING=task_queue.RUNNING, DRIVE_ROOT=root,
                          append_jsonl=lambda _path, row: persisted.append(row),
                          bridge=SimpleNamespace(push_log=pushed.append))
    first = {"type": "task_model_wait", "task_id": "task-one", "wait_id": "wait-a", "revision": 1,
             "task_attempt": 1, "state": "waiting", "is_progress": False,
             "quota_clock": {"revision": 1, "elapsed_sec": 0.0, "active": True, "observed_at": 100.0}}
    last = {**first, "revision": 3, "state": "resolved",
            "quota_clock": {"revision": 3, "elapsed_sec": 20.0, "active": False, "observed_at": 120.0}}
    sibling = {**first, "wait_id": "wait-b", "revision": 2}
    for event in (last, first, sibling):
        projection.handle_task_model_wait(event, ctx)
    meta = task_queue.RUNNING["task-one"]
    assert meta["task"]["model_waits"]["wait-a"]["state"] == "resolved"
    assert meta["task"]["model_waits"]["wait-b"]["state"] == "waiting"
    assert projection.quota_waited_seconds(meta, 200.0) == 20.0
    assert "last_progress_at" not in meta and len(pushed) == len(persisted) == 2


def test_history_preserves_typed_wait_map_without_counting_progress_or_human_rows(tmp_path):
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.utils import append_jsonl

    stamp = datetime.now(timezone.utc)
    for seconds, wait_id, revision, state in ((0, "wait-a", 1, "waiting"), (1, "wait-b", 2, "waiting"), (2, "wait-a", 3, "resolved")):
        append_jsonl(tmp_path / "logs/progress.jsonl", {
            "type": "task_model_wait", "task_id": "task-one", "chat_id": 1,
            "wait_id": wait_id, "revision": revision, "task_attempt": 1, "state": state,
            "ts": (stamp + timedelta(seconds=seconds)).isoformat(), "is_progress": False,
        })
    write_task_result(tmp_path, "task-one", "completed")
    data = json.loads(_assemble_history_response(tmp_path, 1, 0, 5))
    rows = [row for row in data["messages"] if row.get("system_type") == "task_model_wait"]
    assert len(rows) == 1 and rows[0]["is_progress"] is False
    assert rows[0]["model_waits"]["wait-a"]["revision"] == 3
    assert rows[0]["model_waits"]["wait-b"]["revision"] == 2
    assert rows[0]["task_terminal_status"] == "completed"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("dispatch", ["not_started", "response_received"])
def test_confirmed_mixed_pool_waits_and_heals_on_the_same_live_call(elapsed_quota_wait, dispatch, asynchronous):
    root, transport, client, controller, events, _decide = elapsed_quota_wait
    transport.results = [result(outcome="failed", problem={
        "code": "credential_pool_exhausted", "retryable": False,
        "message": "No account can serve this model request",
        "context": {"poolCause": "mixed", "resetsAt": None},
    }), result()]
    transport.dispatch = [dispatch, "response_received"]
    messages = [{"role": "user", "content": "Keep completed work"}]
    answer, usage = (asyncio.run(client.chat_async(messages, MODEL, model_role="light")) if asynchronous
                     else client.chat(messages, MODEL, model_role="light"))
    assert answer == result()["message"]
    assert transport.uploads[0][0]["messages"] == transport.uploads[1][0]["messages"]
    assert len(usage["ledger_attempt_ids"]) == 2
    rows = list(events.queue)
    assert rows[0]["reason"] == rows[-1]["reason"] == "auth_quota"
    assert rows[0]["credential_profile_id"] == "", "A mixed pool has no single login target"
    assert rows[0]["quota_clock"]["active"] is True
    assert rows[-1]["state"] == "resolved" and rows[-1]["resolution"] == "resource_available"
    assert rows[-1]["quota_clock"]["active"] is False and controller.paused_seconds() == pytest.approx(2.5)
    assert load_task_result(root, "task-one")["model_waits"][rows[0]["wait_id"]]["reason"] == "auth_quota"


@pytest.mark.parametrize("pool_context", [{}, {"poolCause": "unavailable"}, {"poolCause": ""},
                                         {"poolCause": "unknown"}, {"poolCause": ["mixed"]},
                                         {"poolCauses": ["auth", "quota"]}])
def test_unproved_pool_cause_never_enters_resource_wait(live_wait, pool_context):
    from ouroboros.llm_claudexor import ClaudexorModelError

    _root, transport, client, controller, events, _decide = live_wait
    transport.results = [result(outcome="failed", problem={
        "code": "credential_pool_exhausted", "message": "Unavailable", "context": pool_context,
    })]
    transport.dispatch = ["not_started"]
    with pytest.raises(ClaudexorModelError, match="credential_pool_exhausted"):
        client.chat([{"role": "user", "content": "Do not infer quota"}], MODEL, model_role="main")
    assert not controller.waits and events.empty() and len(transport.accepted_operations) == 1


def test_mixed_pool_cannot_bypass_unknown_physical_custody(live_wait):
    from ouroboros.llm_claudexor import ClaudexorModelError

    _root, transport, client, controller, events, _decide = live_wait
    transport.results = [result(outcome="unknown", problem={
        "code": "credential_pool_exhausted", "context": {"poolCause": "mixed"},
    })]
    transport.dispatch = ["unknown"]
    with pytest.raises(ClaudexorModelError, match="model_outcome_unknown"):
        client.chat([{"role": "user", "content": "No duplicate generation"}], MODEL, model_role="main")
    assert not controller.waits and events.empty() and len(transport.accepted_operations) == 1


def test_mixed_pool_wait_keeps_calendar_deadline_and_existing_quota_union(live_wait, monkeypatch):
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched

    _root, _transport, client, controller, events, _decide = live_wait
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=__import__("time").time, sleep=lambda _seconds: None))
    controller.started_monotonic = 90.0
    controller.quota_enter("other-reviewer", "other-slot", now=90.0)

    def expired_calendar():
        clock.now = 110.0
        controller.task["deadline_at"] = "2000-01-01T00:00:00Z"
        return controller.control_reason()

    error = ClaudexorModelNotDispatched({
        "code": "credential_pool_exhausted", "context": {"poolCause": "mixed"},
    }, model_role="main")
    with pytest.raises(model_wait.ModelWaitInterrupted) as raised:
        controller.wait(client, error, {"model": MODEL, "model_role": "main",
                        "model_poll_control": expired_calendar})
    assert raised.value.control_reason == "deadline"
    controller.quota_leave("other-reviewer", "other-slot", now=110.0)
    assert controller.paused_seconds(now=120.0) == 20.0, "Concurrent pauses count their union, not 30 seconds"
    assert list(events.queue)[-1]["resolution"] == "deadline"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("code", ["auth_required", "subscription_window_exhausted", "credential_pool_exhausted"])
def test_call_can_decline_resource_wait_without_losing_task_binding(live_wait, code, asynchronous):
    from ouroboros.llm_claudexor import ClaudexorModelError

    root, transport, client, controller, events, _decide = live_wait
    failure = _refusal(code)
    if code == "credential_pool_exhausted":
        failure["problem"]["context"]["poolCause"] = "mixed"
    transport.results = [failure, failure, result()]
    transport.dispatch = ["not_started", "not_started", "response_received"]
    controller.overrides["light"] = {"model": MODEL, "use_local": False, "model_account_override": ""}
    messages = [{"role": "user", "content": "Return an unavailable advisory, retaining the owner"}]

    def call(**kwargs):
        return (asyncio.run(client.chat_async(messages, MODEL, model_role="light", **kwargs)) if asynchronous
                else client.chat(messages, MODEL, model_role="light", **kwargs))

    with pytest.raises(ClaudexorModelError) as caught:
        call(wait_for_resources=False)
    error = caught.value
    assert error.code == code and error.operation_id == "op-0"
    assert error.physical_attempt_capture.state == "released"
    assert error.model_role_route == {"role": "light", "model": MODEL, "use_local": False,
                                     "credential_profile_id": ""}
    assert not controller.waits and events.empty() and len(transport.accepted_operations) == 1
    assert transport.uploads[0][0]["account"] == {"mode": "auto"}
    assert "wait_for_resources" not in json.dumps(transport.uploads[0][0])
    assert model_wait.current_model_wait() is controller and not controller.closed
    attempts = ledger(root)
    assert [(row["state"], row["revision"]) for row in attempts] == [("released", 4)]
    assert attempts[0]["physical_failure"]["stage"] == "raised_exception"

    answer, usage = call()
    assert answer == result()["message"] and len(transport.accepted_operations) == 3
    assert len(usage["ledger_attempt_ids"]) == 2
    assert any(row["state"] == "waiting" for row in list(events.queue))
    assert controller.overrides["light"]["model_account_override"] == ""


@pytest.mark.parametrize("asynchronous", [False, True])
def test_declining_resource_wait_still_honors_owner_control(live_wait, monkeypatch, asynchronous):
    _root, transport, client, controller, events, _decide = live_wait
    monkeypatch.setattr(controller, "control_reason", lambda: "finalize_requested")
    with pytest.raises(model_wait.ModelWaitInterrupted) as caught:
        args = ([{"role": "user", "content": "Respect Stop"}], MODEL)
        kwargs = {"model_role": "light", "wait_for_resources": False}
        if asynchronous:
            asyncio.run(client.chat_async(*args, **kwargs))
        else:
            client.chat(*args, **kwargs)
    assert caught.value.control_reason == "finalize_requested"
    assert not transport.accepted_operations and not controller.waits and events.empty()
