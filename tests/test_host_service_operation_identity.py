"""Real ingress and task-authority controls for repeated skill deliveries."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.host_notice import STEPS
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import message_bus
from tests._dialogue import chat_rows
from tests.test_host_service_operations import (
    CHAT, MSG, _client, _headers, _chat_row, _inbound, _isolate_queue, _origin_ref, _receipt,
)


@pytest.fixture
def direct_budget_refusal(tmp_path, monkeypatch):
    """The lane with one carrier and an actor that finds the budget exhausted before any model work."""
    from queue import SimpleQueue
    from types import SimpleNamespace
    from supervisor import state
    from tests._agent_entry_helpers import task_entry_agent
    from tests.conversation_carrier_support import isolate_lane_pool

    agent = task_entry_agent(tmp_path, lambda _task: pytest.fail("exhausted budget entered the task body"))
    agent.env.drive_path = lambda name: tmp_path / name
    [spare] = isolate_lane_pool(monkeypatch, tmp_path, event_q=SimpleQueue())
    monkeypatch.setattr(state, "load_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda _state, *, strict: 0)
    return SimpleNamespace(agent=agent, spare=spare)


def _carried_refusal(workers, world):
    """Run the admitted turn's body in its carrier: the pre-start refusal unwinds it before any model work."""
    from tests.conversation_carrier_support import carry

    with pytest.raises(Exception, match="budget_exhausted"):
        carry(workers, world.spare, world.agent)


def test_atomic_acceptance_before_dequeue_and_exact_source_consumption(tmp_path, monkeypatch):
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
            "accepted_source_ref": {"chat_id": 999}, "task_metadata": {"origin_message_ref": {"chat_id": 999}}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: client.post("/chat/inject", headers=_headers(), json=body), range(4)))
    assert all(reply.status_code == 202 for reply in replies)
    assert bridge._inbox.qsize() == 1
    assert sum(reply.json().get("rejoined", False) for reply in replies) == 3
    rows = chat_rows(tmp_path)
    assert len(rows) == 1
    update = bridge.get_updates(0, timeout=0)[0]["message"]
    ref = message_bus.record_inbound_message(
        bridge, update, chat_id=CHAT, user_id=0, client_message_id=MSG, text="one message", ts="later",
    )
    from ouroboros.owner_relay import owner_row_source

    # The one accepted row by its physical identity: its four older fields and the digest of that very row.
    assert ref == owner_row_source(rows[0])[0]
    assert {key: ref[key] for key in ("chat_id", "client_message_id", "ts", "text_sha256")} == build_owner_message_ref(
        chat_id=CHAT, client_message_id=MSG, ts=rows[0]["ts"], text="one message")
    assert len(chat_rows(tmp_path)) == 1
    assert not update.get("task_metadata", {}).get("origin_message_ref")
    replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert replay.json()["rejoined"] is True and bridge._inbox.empty()
    distinct = client.post("/chat/inject", headers=_headers(), json={**body, "client_message_id": "message-two"})
    assert distinct.status_code == 202 and bridge._inbox.qsize() == 1


@pytest.mark.parametrize("step", ["row", "task", "accepted_http", "accepted_ingress"])
def test_c1_transient_reply_reads_return_not_accepted_before_any_write(tmp_path, monkeypatch, step):
    from starlette.testclient import TestClient
    from ouroboros import dialogue_store, task_status
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.gateway.host_service import create_host_service_app
    from ouroboros.store import StoreBusy, StoreMigrating
    from tests.test_host_service_api import _seed_token

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    _seed_token(tmp_path, skill="telegram", permissions=["inject_chat"], bundled=True)
    original = message_bus.log_chat("out", 42, 0, "Previous answer")
    write_task_result(tmp_path, "reply-task", "running", chat_id=42)
    address = "task:reply-task" if step == "task" else format_address(row_address(original))
    body = {"chat_id": 42, "text": "A follow-up", "client_message_id": "tg:123:42:50", "reply_to": address,
            "transport": {"kind": "telegram", "conversation_id": "42", "bot_id": "123", "owner_message_id": 50}}
    client = TestClient(create_host_service_app(tmp_path, bridge_getter=lambda: bridge))
    headers = {"X-Skill-Token": "token"}
    carrier, name, at = {
        "row": (dialogue_store, "rows_by_identity", 1), "task": (task_status, "load_task_result", 1),
        "accepted_http": (dialogue_store, "accepted_row", 1), "accepted_ingress": (dialogue_store, "accepted_row", 2),
    }[step]
    real, calls = getattr(carrier, name), 0

    def read(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == at:
            raise (StoreMigrating if step.startswith("accepted") else StoreBusy)("fixture unavailable")
        return real(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(carrier, name, read)
        response = client.post("/chat/inject", headers=headers, json=body)
    assert response.status_code == 503 and response.json()["code"] == "not_accepted", response.text
    assert bridge._inbox.empty() and len(chat_rows(tmp_path)) == 1
    accepted = client.post("/chat/inject", headers=headers, json=body)
    assert accepted.status_code == 202 and bridge._inbox.qsize() == 1, accepted.text
    assert chat_rows(tmp_path)[-1]["reply_to"]["address"] == address


def _telegram_injection(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from ouroboros.gateway.host_service import create_host_service_app
    from tests.test_host_service_api import _seed_token

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    _seed_token(tmp_path, skill="telegram", permissions=["inject_chat"], bundled=True)
    client = TestClient(create_host_service_app(tmp_path, bridge_getter=lambda: bridge))

    def post():
        return client.post("/chat/inject", headers={"X-Skill-Token": "token"}, json={
            "chat_id": 42, "text": "About the report", "client_message_id": "tg:123:42:50", "reply_to": "task:helper",
            "transport": {"kind": "telegram", "conversation_id": "42", "bot_id": "123", "owner_message_id": 50},
        })

    return bridge, post


def test_d1_reply_to_task_after_store_export_accepts_an_unresolved_quote(tmp_path, monkeypatch):
    from ouroboros import store, store_import, store_switch
    from tests.test_chat_references import _reply_world, _route

    _reply_world(tmp_path)
    bridge, post = _telegram_injection(tmp_path, monkeypatch)
    with store.open_store(tmp_path, write=True, service=True) as txn:
        txn.set_meta("phase", store.EXPORTING)
    store_import.export_facts(tmp_path)
    store_switch._retire(tmp_path)
    with pytest.raises(store.StoreRetired):
        with store.open_store(tmp_path, write=False):
            pass

    accepted = post()
    assert accepted.status_code == 202, accepted.text
    [row] = chat_rows(tmp_path)
    assert row["chat_id"] == 42 and row["text"] == "About the report"
    assert row["reply_to"] == {"address": "task:helper", "unresolved": True}
    [(args, kwargs)], followups, receipts = _route(bridge, tmp_path, monkeypatch, row)
    assert args[0] == 42 and not kwargs["task_metadata"].get("project_id")
    assert followups == [42] and not receipts


@pytest.mark.parametrize("unavailable", ["importing", "noncontention-busy"])
def test_d2_project_lineage_unavailable_retries_before_acceptance(tmp_path, monkeypatch, unavailable):
    from ouroboros import projects_registry as pr, store
    from tests.test_chat_references import _reply_world

    project = _reply_world(tmp_path)
    bridge, post = _telegram_injection(tmp_path, monkeypatch)
    real_open = pr.open_store
    importing_root = tmp_path / "lineage-import"
    importing_root.mkdir()

    def lineage_store(_root, **kwargs):
        if unavailable == "noncontention-busy":
            raise store.StoreBusy("lineage unavailable without lock contention")
        # The task lookup above sees the published fixture, while the indexed lineage read sees an
        # importing store that has not published its database yet.
        with store.importing(importing_root):
            with real_open(importing_root, **kwargs):
                pytest.fail("strict lineage opened a store inside its import window")

    with monkeypatch.context() as scoped:
        scoped.setattr(pr, "open_store", lineage_store)
        refused = post()
    assert refused.status_code == 503 and refused.json()["code"] == "not_accepted", refused.text
    assert bridge._inbox.empty() and chat_rows(tmp_path) == []
    accepted = post()
    assert accepted.status_code == 202 and bridge._inbox.qsize() == 1, accepted.text
    [row] = chat_rows(tmp_path)
    assert row["reply_to"]["telegram_room_chat_id"] == project["chat_id"]


def test_c9_only_the_bundled_bridge_accepts_the_optional_decision_surface(tmp_path):
    import asyncio
    from ouroboros.gateway.task_decision import answer_decision
    from tests.test_chat_references import seed_bundled_bridge
    from tests.test_host_service_api import _seed_token

    _seed_token(tmp_path, skill="telegram")
    body = {"request_id": "answer:1", "decision_id": "quiz:test:history", "option_index": 0}
    event = {**body, "surface": {"kind": "telegram", "bot_id": "123", "conversation_id": "42"}}
    status, refused = asyncio.run(answer_decision(tmp_path, event, source="skill:telegram"))
    assert status == 400 and refused["reason_code"] == "unexpected_fields"
    ordinary = asyncio.run(answer_decision(tmp_path, body, source="skill:telegram"))
    seed_bundled_bridge(tmp_path)
    external = tmp_path / "skills/external/telegram"
    external.rename(tmp_path / "unused-external-fixture")
    assert asyncio.run(answer_decision(tmp_path, event, source="skill:telegram")) == ordinary


def test_failed_acceptance_never_queues_and_restart_never_requeues(tmp_path, monkeypatch):
    from ouroboros.utils import atomic_write_json

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    atomic_write_json(tmp_path / "state/state.json", {"session_id": "before"})
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message"}
    with monkeypatch.context() as scoped:
        scoped.setattr(message_bus, "accept_chat", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("canonical message acceptance could not be persisted")))
        assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 500
    assert bridge._inbox.empty()
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    atomic_write_json(tmp_path / "state/state.json", {"session_id": "after"})
    restarted = message_bus.LocalChatBridge()
    restarted_client = _client(tmp_path, restarted)
    assert restarted_client.post("/chat/inject", headers=_headers(), json=body).json()["rejoined"]
    state = restarted_client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert state["status"] == "lost" and restarted._inbox.empty()


@pytest.mark.parametrize("mismatch", ["chat_id", "text_sha256", "source"])
def test_annotation_collision_never_authorizes_foreign_task(tmp_path, monkeypatch, mismatch):
    client = _client(tmp_path)
    _inbound(tmp_path, "own source")
    ref = _origin_ref(tmp_path)
    if mismatch == "source":
        _chat_row(tmp_path, "in", "foreign source", chat_id=1, client_message_id=MSG, source="web")
        ref = build_owner_message_ref(chat_id=1, client_message_id=MSG, ts="foreign", text="foreign source")
    else:
        ref[mismatch] = 1 if mismatch == "chat_id" else "0" * 64 if mismatch == "text_sha256" else "foreign"
    _, pending = _isolate_queue(monkeypatch, tmp_path, [{"id": "foreign", "chat_id": 1, "origin_message_ref": ref}])
    write_task_result(tmp_path, "foreign", "scheduled", origin_message_ref=ref, result="foreign answer")
    _receipt(tmp_path, "promote_chat_to_task", "scheduled", "foreign")
    state = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert state["status"] == "pending" and "task_id" not in state and "text" not in state
    result = client.post("/chat/cancel", headers=_headers(), json={"operation_ref": f"{CHAT}:{MSG}"})
    assert result.status_code == 409 and len(pending) == 1
    assert not (tmp_path / "state/cancel_intents.json").exists()


def test_interleaved_same_chat_answers_are_matched_by_task_origin(tmp_path):
    client = _client(tmp_path)
    _inbound(tmp_path, "request A")
    ref_a = _origin_ref(tmp_path)
    _chat_row(tmp_path, "in", "request B", client_message_id="B")
    write_task_result(tmp_path, "task-a", "completed", origin_message_ref=ref_a, result="answer A")
    _chat_row(tmp_path, "out", "answer A", task_id="task-a")
    a = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    b = client.get(f"/chat/operations/{CHAT}:B", headers=_headers()).json()
    assert a["status"] == "completed" and a["text"] == "answer A"
    assert b["status"] == "pending" and "text" not in b


def test_pre_task_budget_refusal_remains_an_exact_failed_operation(tmp_path, monkeypatch, direct_budget_refusal):
    from supervisor import worker_chat_lane, workers

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    monkeypatch.setattr(workers, "send_with_budget", message_bus.send_with_budget)
    client = _client(tmp_path, bridge)
    assert client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "do work",
    }).status_code == 202
    ref = bridge.get_updates(0, timeout=0)[0]["message"]["accepted_source_ref"]
    worker_chat_lane.handle_chat_direct(CHAT, "do work", task_metadata={"origin_message_ref": ref, "_host_operation": True})
    _carried_refusal(workers, direct_budget_refusal)
    # The carrier's typed refusal is the host operation's exact failed terminal, correlated by its source.
    [task_id] = list(workers.RUNNING)
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == "failed"  # its record keeps the step
    assert stored["result"] == f"The total budget is spent, so this turn did not start. {STEPS['raise_total_budget']}"
    result = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    assert result["status"] == "failed" and result["text"] == stored["result"], "the operation answers from its record"


@pytest.mark.parametrize("chat_id", [1, 123456])
def test_ordinary_main_and_transport_refusals_keep_their_original_envelope(monkeypatch, chat_id, direct_budget_refusal):
    """A registered refusal is keyed, but only host operations retain a terminal origin."""
    from supervisor import worker_chat_lane, workers

    frames = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    finals = []
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **kw: finals.append((a, kw)))
    worker_chat_lane.handle_chat_direct(chat_id, "hello", task_metadata={
        "origin_message_ref": build_owner_message_ref(chat_id=chat_id, client_message_id="ordinary", ts="now", text="hello"),
    })
    receipt = next(row for row in frames if row.get("type") == "typing")
    assert receipt["client_message_id"] == "ordinary" and receipt["chat_id"] == chat_id
    assert receipt["activity_id"]
    assert finals == [], "the admission refused nothing"
    _carried_refusal(workers, direct_budget_refusal)
    # The carrier's keyed final: the registered turn's id, no durable origin for an ordinary turn.
    [(args, kwargs)] = finals
    assert kwargs["task_id"] == receipt["activity_id"] and kwargs["system_type"] == "task_not_started"
    assert args[1] == "The total budget is spent, so this turn did not start."
    assert kwargs["progress_meta"]["notice"]["step"] == "raise_total_budget" and "origin_message_ref" not in kwargs["progress_meta"]
    # Keyed by the registered turn's id, in the owner's closed-class words; an ordinary turn keeps no
    # durable terminal origin (its client id stays on the live receipt).
    stored = load_task_result(workers.DRIVE_ROOT, receipt["activity_id"])
    assert stored["status"] == "failed", stored
    assert stored["result"] == f"The total budget is spent, so this turn did not start. {STEPS['raise_total_budget']}"
    assert [row for row in chat_rows(workers.DRIVE_ROOT) if row.get("origin_message_ref")] == []


@pytest.mark.parametrize("host_operation", [False, True])
def test_chat_crash_preserves_only_the_host_accepted_operation(tmp_path, monkeypatch, host_operation):
    from queue import SimpleQueue
    from supervisor import worker_chat_lane, workers

    from types import SimpleNamespace
    from tests.conversation_carrier_support import carry, isolate_lane_pool

    class CrashingAgent:
        env = SimpleNamespace(drive_root=tmp_path, drive_path=lambda name: tmp_path / name)

        def handle_task(self, task):
            raise RuntimeError("test chat execution failed")

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    [spare] = isolate_lane_pool(monkeypatch, tmp_path, event_q=SimpleQueue())
    monkeypatch.setattr(workers, "send_with_budget", message_bus.send_with_budget)
    client = _client(tmp_path, bridge)
    assert client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "do work",
    }).status_code == 202
    ref = bridge.get_updates(0, timeout=0)[0]["message"]["accepted_source_ref"]
    worker_chat_lane.handle_chat_direct(
        CHAT, "do work", task_metadata={"origin_message_ref": ref, "_host_operation": host_operation},
    )
    # The carrier records the crash as the turn's own terminal before it unwinds (owner decision 1A).
    with pytest.raises(RuntimeError, match="test chat execution failed"):
        carry(workers, spare, CrashingAgent())
    [task_id] = list(workers.RUNNING)
    stored = load_task_result(tmp_path, task_id)
    assert stored["status"] == "failed" and "test chat execution failed" in stored["result"]
    # The owner reads the failure once, keyed to the turn; only the host-accepted operation keeps its source.
    rows = chat_rows(tmp_path)
    final = rows[-1]
    # Its row says the phrase; the raw error is its notice's detail and stays in the record above.
    assert final["text"] == "This turn stopped on an unexpected error."
    assert "test chat execution failed" in final["notice"]["details"]
    assert final["task_terminal_status"] == "failed"
    assert final.get("origin_message_ref") == (ref if host_operation else None)
    # The lane row carries the turn's complete ingress origin: the operation's authority, host op or not.
    result = client.get(f"/chat/operations/{CHAT}:{MSG}", headers=_headers()).json()
    # The raw error stays reachable: a host-accepted row answers with its phrase and its notice, any other turn
    # from its record.
    assert result["status"] == "failed" and "test chat execution failed" in (
        result["notice"]["details"] if host_operation else result["text"])


def test_real_server_consumes_one_source_and_authors_only_its_operation_marker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import server

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    routed = []
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, row: routed.append(row))
    ctx = SimpleNamespace(load_state=lambda: {"owner_id": 1}, update_state=lambda mutator: mutator({"owner_id": 1}))
    client = _client(tmp_path, bridge)
    client.post("/chat/inject", headers=_headers(), json={"chat_id": CHAT, "client_message_id": MSG, "text": "hello"})
    from ouroboros.owner_relay import owner_row_source

    ref = _origin_ref(tmp_path)
    server._process_bridge_updates(bridge, 0, ctx)
    [stored] = chat_rows(tmp_path)
    assert routed[-1]["origin_message_ref"] == owner_row_source(stored)[0]  # the one source, by its physical row
    assert {key: routed[-1]["origin_message_ref"][key] for key in ref} == ref
    assert routed[-1]["task_metadata"]["_host_operation"] is True
    assert len(chat_rows(tmp_path)) == 1
    bridge.enqueue_local_message("ordinary", task_metadata={"_host_operation": True})
    server._process_bridge_updates(bridge, 1, ctx)
    assert "_host_operation" not in routed[-1]["task_metadata"]


def test_the_archive_import_between_an_acceptance_and_its_replay_cannot_turn_the_replay_into_new_work(tmp_path, monkeypatch):
    """The accepted row keeps its identity through the one-time archive import (a legacy chain
    imported underneath it) and through a cutover of a legacy-written live file: a replay rejoins
    the same source, nothing is enqueued twice."""
    from ouroboros import dialogue_import

    (tmp_path / "archive").mkdir(parents=True, exist_ok=True)
    (tmp_path / "archive" / "chat_20261001T000000.jsonl").write_text(
        '{"direction": "out", "chat_id": 1, "text": "older history", "ts": "2026-10-01T00:00:00+00:00"}\n', encoding="utf-8")
    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "hello"}
    accepted = client.post("/chat/inject", headers=_headers(), json=body)
    assert accepted.status_code == 202
    source = _origin_ref(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["pending"] == 1, "the legacy segment is not imported yet"
    replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert replay.status_code == 202 and replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source
    dialogue_import.run_dialogue_import(tmp_path)
    assert dialogue_import.coverage(tmp_path, "chat")["imported"]
    with (tmp_path / "logs" / "chat.jsonl").open("a", encoding="utf-8") as handle:  # an older body appended
        handle.write('{"direction": "out", "chat_id": 1, "text": "from an older body", "ts": "2026-10-17T00:00:00+00:00"}\n')
    dialogue_import.cutover(tmp_path)
    archived_replay = client.post("/chat/inject", headers=_headers(), json=body)
    assert archived_replay.status_code == 202 and archived_replay.json()["rejoined"]
    assert bridge._inbox.qsize() == 1 and _origin_ref(tmp_path) == source
    assert [row["text"] for row in chat_rows(tmp_path)] == ["older history", "hello", "from an older body"]


def test_racing_named_upload_rejoin_keeps_only_the_accepted_copy(tmp_path, monkeypatch):
    import pathlib
    import threading
    from ouroboros.gateway import host_service

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    barrier = threading.Barrier(2)
    original = host_service.store_upload
    def copy(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=5)
        return result
    monkeypatch.setattr(host_service, "store_upload", copy)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
            "attachments": [{"path": str(source)}]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post("/chat/inject", headers=_headers(), json=body), range(2)))
    assert all(response.status_code == 202 for response in responses)
    assert sum(response.json().get("rejoined", False) for response in responses) == 1
    updates = bridge.get_updates(0, timeout=0)
    assert len(updates) == 1 and bridge._inbox.empty()
    stored = pathlib.Path(updates[0]["message"]["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert list((tmp_path / "uploads").iterdir()) == [stored]
    assert stored.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("failure", ["write_unknown_empty", "write_unknown_landed", "queue_unknown"])
def test_named_upload_custody_survives_unknown_write_or_queue_outcome(tmp_path, monkeypatch, failure):
    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    original = message_bus.accept_chat
    def fail(*args, **kwargs):
        if failure == "write_unknown_landed":
            original(*args, **kwargs)
        raise OSError("controlled admission failure")
    if failure.startswith("write_unknown"):
        monkeypatch.setattr(message_bus, "accept_chat", fail)
    else:
        monkeypatch.setattr(bridge, "enqueue_local_message", fail)
    response = client.post("/chat/inject", headers=_headers(), json={
        "chat_id": CHAT, "client_message_id": MSG, "text": "one message",
        "attachments": [{"path": str(source)}],
    })
    assert response.status_code == 500 and bridge._inbox.empty()
    copies = list((tmp_path / "uploads").iterdir())
    assert len(copies) == 1
    assert copies[0].read_bytes() == source.read_bytes()
    row = message_bus.accepted_chat_message(tmp_path, CHAT, MSG)
    assert bool(row) is (failure != "write_unknown_empty")
    assert source.read_bytes() == b"complete attachment"


@pytest.mark.parametrize("cancel_mode", ["asyncio", "anyio"])
def test_cancelled_named_acceptance_settles_before_upload_cleanup(tmp_path, monkeypatch, cancel_mode):
    import asyncio
    import threading
    import anyio
    from types import SimpleNamespace
    from ouroboros.gateway import host_service

    bridge = message_bus.LocalChatBridge()
    client = _client(tmp_path, bridge)
    source = tmp_path / "state/skills/a2a/input.pdf"
    source.write_bytes(b"complete attachment")
    entered, release = threading.Event(), threading.Event()
    original = message_bus.accept_chat
    def held(*args, **kwargs):
        receipt = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return receipt
    monkeypatch.setattr(message_bus, "accept_chat", held)
    async def body():
        return {"chat_id": CHAT, "client_message_id": MSG, "text": "one message",
                "attachments": [{"path": str(source)}]}
    request = SimpleNamespace(app=client.app, headers={k.lower(): v for k, v in _headers().items()}, json=body)
    ctx = client.app.state.host_service_context
    async def run():
        scope = anyio.CancelScope()
        async def call():
            with scope:
                return await host_service._api_chat_inject(request)
        task = asyncio.create_task(call())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            scope.cancel() if cancel_mode == "anyio" else task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and ctx._inflight["a2a"] == 1
        finally:
            release.set()
            if cancel_mode == "anyio":
                await task
                assert scope.cancelled_caught
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert ctx._inflight["a2a"] == 0
    asyncio.run(run())
    message = bridge.get_updates(0, timeout=0)[0]["message"]
    from pathlib import Path
    stored = Path(message["task_metadata"]["chat_attachment_uploads"][0]["path"])
    assert stored.read_bytes() == source.read_bytes()


def test_solitary_inline_photo_retains_a_staged_original_after_acceptance(tmp_path, monkeypatch):
    import base64
    from pathlib import Path
    from ouroboros.artifacts import stage_task_attachments
    from ouroboros.context import build_user_content
    from tests.test_live_image_delivery import pixels, image_blocks, decode_block

    raw = pixels(size=(9000, 24))
    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    body = {"chat_id": CHAT, "client_message_id": MSG, "image_base64": base64.b64encode(raw).decode(),
            "image_mime": "image/png", "text": ""}
    response = client.post("/chat/inject", headers=_headers(), json=body)
    assert response.status_code == 202, response.text
    message = bridge.get_updates(0, timeout=0)[0]["message"]
    uploads = message["task_metadata"]["chat_attachment_uploads"]
    assert len(uploads) == 1 and Path(uploads[0]["path"]).read_bytes() == raw
    manifest = stage_task_attachments(tmp_path, "photo-task", uploads)
    content = build_user_content({"id": "photo-task", "drive_root": tmp_path, "attachment_images": manifest})
    block = image_blocks([{"content": content}])[0]
    assert Path(block["_source_path"]).read_bytes() == raw
    assert decode_block(block) == raw
    assert client.post("/chat/inject", headers=_headers(), json=body).json()["rejoined"] is True
    assert bridge._inbox.empty()


@pytest.mark.parametrize("refusal", ["busy", "migrating"])
def test_a_room_decision_refused_before_acceptance_answers_not_accepted_and_the_same_delivery_lands_once(
        tmp_path, monkeypatch, refusal):
    """The exact Reply's room decision finds the store busy or still migrating: the host answers ``503
    not_accepted`` and this attempt writes nothing; the same delivery sent again is accepted once and queued once,
    and a third rejoins it."""
    from ouroboros import chat_references
    from ouroboros.store import StoreBusy, StoreMigrating

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    tries = []

    def busy_once(_root, _reply, _chat):
        tries.append(1)
        if len(tries) == 1:
            raise (StoreBusy("room facts busy", reason="contention") if refusal == "busy"
                   else StoreMigrating("room facts still importing"))
        return 0, True

    monkeypatch.setattr(chat_references, "exact_reply_ingress", lambda *_args: True)
    monkeypatch.setattr(chat_references, "telegram_reply_room", busy_once)
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "About this", "reply_to": "task:helper"}
    refused = client.post("/chat/inject", headers=_headers(), json=body)
    assert refused.status_code == 503 and refused.json()["code"] == "not_accepted"
    assert message_bus.accepted_chat_message(tmp_path, CHAT, MSG) is None and bridge._inbox.empty()
    accepted = client.post("/chat/inject", headers=_headers(), json=body)
    assert accepted.status_code == 202 and not accepted.json().get("rejoined")
    again = client.post("/chat/inject", headers=_headers(), json=body)
    assert again.status_code == 202 and again.json()["rejoined"] is True
    assert len(chat_rows(tmp_path)) == 1 and bridge._inbox.qsize() == 1 and len(tries) == 2


def test_a_failure_after_the_decision_is_no_not_accepted_answer(tmp_path, monkeypatch):
    """A busy store met while accepting (the write attempted) is an unknown outcome: never ``not_accepted``."""
    from ouroboros.store import StoreBusy

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    monkeypatch.setattr(message_bus, "accept_chat",
                        lambda *a, **kw: (_ for _ in ()).throw(StoreBusy("busy while writing", reason="contention")))
    reply = client.post("/chat/inject", headers=_headers(), json={"chat_id": CHAT, "client_message_id": MSG,
                                                                  "text": "one message"})
    assert reply.status_code == 500 and reply.json().get("code") != "not_accepted"


def test_named_skill_reply_rejoins_only_the_same_passage_after_source_loss(tmp_path, monkeypatch):
    from ouroboros.chat_chain import format_address, row_address

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    original = message_bus.log_chat("out", CHAT, 0, "Harbor, harbor.")
    excerpt = {"text": "harbor", "start_utf16": 8}
    body = {"chat_id": CHAT, "client_message_id": MSG, "text": "The second pier.",
            "reply_to": format_address(row_address(original)), "reply_excerpt": excerpt}
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    assert chat_rows(tmp_path)[-1]["reply_to"]["excerpt"] == excerpt
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *args: (None, {}))
    rejoined = client.post("/chat/inject", headers=_headers(), json=body)
    assert rejoined.status_code == 202 and rejoined.json()["rejoined"]
    changed = {**body, "reply_excerpt": {**excerpt, "start_utf16": 0}}
    assert client.post("/chat/inject", headers=_headers(), json=changed).status_code == 409
    assert bridge._inbox.qsize() == 1


@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "named"])
@pytest.mark.parametrize("source", ["question", "missing"])
def test_a_refused_passage_is_one_bad_request_named_or_not(tmp_path, monkeypatch, named, source):
    """A passage with no readable authored source is the request's fault on both inject paths; nothing is saved or
    queued, and 409 stays the answer to a reused id that names a different message."""
    from ouroboros.chat_chain import format_address, row_address

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    address = format_address(row_address(message_bus.log_chat("out", CHAT, 0, "Which pier?", record_type="quiz")))
    if source == "missing":
        address = address[:-1] + ("0" if address[-1] != "0" else "1")
    saved = len(chat_rows(tmp_path))
    body = {"chat_id": CHAT, "text": "The second pier.", "reply_to": address,
            "reply_excerpt": {"text": "pier", "start_utf16": 6}, **({"client_message_id": MSG} if named else {})}
    refused = client.post("/chat/inject", headers=_headers(), json=body)
    assert refused.status_code == 400 and "quote" in refused.text
    assert len(chat_rows(tmp_path)) == saved and bridge._inbox.qsize() == 0


def test_unnamed_skill_queues_the_host_resolution_of_an_authored_notice(tmp_path, monkeypatch):
    from ouroboros.chat_chain import format_address, row_address

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    client = _client(tmp_path, bridge)
    original = message_bus.log_chat("out", CHAT, 0, "Harbor, harbor.", record_type="proactive_message")
    address = format_address(row_address(original))
    excerpt = {"text": "harbor", "start_utf16": 8}
    forged = {"address": address, "snippet": "Wrong", "author_label": "Someone else"}
    body = {"chat_id": CHAT, "text": "The second pier.", "reply_to": address, "reply_excerpt": excerpt,
            "accepted_reply": forged, "reply_reference": forged, "task_metadata": {"reply_reference": forged}}
    assert client.post("/chat/inject", headers=_headers(), json=body).status_code == 202
    monkeypatch.setattr("ouroboros.chat_references.resolve_row", lambda *_: pytest.fail("second source read"))
    [update] = bridge.get_updates(0, timeout=0)
    message_bus.record_inbound_message(bridge, update["message"], chat_id=CHAT, user_id=0,
                                       client_message_id="queued", text=body["text"], ts="later")
    saved = chat_rows(tmp_path)[-1]["reply_to"]
    assert (saved["author_label"], saved["snippet"], saved["excerpt"]) == ("Ouroboros", original["text"], excerpt)


@pytest.mark.parametrize("mode", ["named", "unnamed", "canonical_row_only"])
def test_skill_injection_routes_the_accepted_reply_after_dequeue(tmp_path, monkeypatch, mode):
    from types import SimpleNamespace

    import server
    from ouroboros.chat_chain import format_address, row_address

    bridge = message_bus.LocalChatBridge()
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    routed = []
    monkeypatch.setattr(server, "_route_owner_message", lambda bridge, ctx, row: routed.append(row))
    ctx = SimpleNamespace(load_state=lambda: {"owner_id": 1}, update_state=lambda mutator: mutator({"owner_id": 1}))
    original = message_bus.log_chat("out", CHAT, 0, "Copper lantern")
    excerpt = {"text": "lantern", "start_utf16": 7}
    body = {"chat_id": CHAT, "text": "Which lantern?", "reply_excerpt": excerpt,
            "reply_to": format_address(row_address(original))}
    if mode != "unnamed":
        body["client_message_id"] = MSG
    assert _client(tmp_path, bridge).post("/chat/inject", headers=_headers(), json=body).status_code == 202
    if mode == "canonical_row_only":
        updates = bridge.get_updates(0, timeout=0)
        updates[0]["message"].pop("task_metadata")  # a carrier older than the canonical reply fact
        monkeypatch.setattr(bridge, "get_updates", lambda *_a, **_k: updates)
    server._process_bridge_updates(bridge, 0, ctx)
    assert len(routed) == 1
    reply = routed[0]["task_metadata"]["reply_reference"]
    # The dispatch carrier may add a verified full source; the durable row keeps
    # the compact canonical reference used for routing and history.
    assert {key: value for key, value in reply.items() if key != "text"} == chat_rows(tmp_path)[-1]["reply_to"]
    if mode in {"named", "unnamed"}:
        assert reply["text"] == original["text"]
    else:
        assert "text" not in reply
    assert reply["address"] == body["reply_to"] and reply["excerpt"] == excerpt
