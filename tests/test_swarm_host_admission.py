"""Swarm admission preserves owner input and uses no routing model.

Queue, reservation, promotion, snapshot, result and annotation writers are real.
Only model/process/bridge boundaries and isolated roots differ.
"""
import asyncio
import copy
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

import server
from ouroboros import config
from ouroboros.gateway.routing_decision import _derived_identity
from ouroboros.llm import LLMClient
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import create_project, project_binding_for_task
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.utils import append_jsonl
from supervisor import git_ops, queue, state, worker_chat_lane as lane, workers
from supervisor.events import _handle_promote_chat_to_task

pytestmark = pytest.mark.serial
EXACT_OWNER = "  Исследуй варианты A и B.\nСначала уточни ограничения.  \n"
WORDLESS = "(image attached)"
SURFACE = {"channel": "web", "viewport": {"width": 1280, "height": 800}}
CONSTRAINT = {"mode": "normal"}
REFUSALS = {
    "update": "🔒 An update is using the repository. Try this message again when it finishes.",
    "budget": "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.",
    "ledger": "⚠️ Cost accounting is unavailable. Task was not dispatched; retry after ledger recovery.",
}


class ForbiddenModelEntry(AssertionError):
    pass


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, **_options):
        self.target, self.args, self.kwargs = target, args, kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


class Bridge:
    def __init__(self):
        self.acks, self.frames = [], []

    def send_routing_ack(self, chat_id, **payload):
        self.acks.append({"chat_id": chat_id, **copy.deepcopy(payload)})

    def broadcast(self, payload):
        self.frames.append(copy.deepcopy(payload))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


@pytest.fixture
def host(tmp_path, monkeypatch, request):
    root = tmp_path / "data"
    root.mkdir()
    repo_root = tmp_path / "isolated-repo"
    repo_root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("OUROBOROS_REPO_DIR", str(repo_root))
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    state.init(root, 0.0)
    queue.init(root)
    pending, running, sequence = [], {}, {"value": 0}
    pool = {0: SimpleNamespace(active_capacity=True, readiness_exhausted=False, reaping=False, busy_task_id=None)}
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", root)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
        monkeypatch.setattr(module, "QUEUE_SEQ_COUNTER_REF", sequence)
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", root)
    monkeypatch.setattr(git_ops, "REPO_DIR", repo_root)
    monkeypatch.setattr(workers, "REPO_DIR", repo_root)
    monkeypatch.setattr(workers, "WORKERS", pool)
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "_repo_writer_gate_reason", "")
    notices, model_attempts = [], []
    bridge = Bridge()
    send = lambda chat_id, text, **metadata: notices.append({"chat_id": chat_id, "text": text, **metadata})
    monkeypatch.setattr(workers, "send_with_budget", send)
    monkeypatch.setattr("supervisor.message_bus.get_bridge", lambda: bridge)
    monkeypatch.setattr("ouroboros.server_owner_routing.threading", SimpleNamespace(Thread=ImmediateThread))

    def forbidden_model(*args, **kwargs):
        model_attempts.append({"entry": "LLMClient", "kwargs_keys": sorted(kwargs)})
        raise ForbiddenModelEntry("No model call is permitted in the admission probe")

    async def forbidden_model_async(*args, **kwargs):
        return forbidden_model(*args, **kwargs)

    def forbidden_lane(*args, **kwargs):
        model_attempts.append({"entry": "_run_chat_task", "ephemeral": kwargs.get("ephemeral"),
                               "text": args[2] if len(args) > 2 else None,
                               "task_metadata": copy.deepcopy(kwargs.get("task_metadata"))})
        raise ForbiddenModelEntry("Swarm entered the model-owning chat lane before managed admission")

    # Windows creates its asyncio socketpair through connect. Prepare the loop
    # before sealing sockets; API calls still run with every network/model guard.
    event_loop = asyncio.new_event_loop()
    def close_loop():
        try:
            event_loop.run_until_complete(event_loop.shutdown_asyncgens())
            event_loop.run_until_complete(event_loop.shutdown_default_executor())
        finally:
            event_loop.close()
    request.addfinalizer(close_loop)

    monkeypatch.setattr(LLMClient, "chat", forbidden_model)
    monkeypatch.setattr(LLMClient, "chat_async", forbidden_model_async)
    monkeypatch.setattr(lane, "_run_chat_task", forbidden_lane)
    monkeypatch.setattr("ouroboros.server_owner_routing._decision_turn_metadata", forbidden_model)
    monkeypatch.setattr(socket.socket, "connect", forbidden_model)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_model)
    ctx = SimpleNamespace(
        DRIVE_ROOT=root, PENDING=pending, RUNNING=running, WORKERS=pool, bridge=bridge,
        enqueue_task=queue.enqueue_task, persist_queue_snapshot=queue.persist_queue_snapshot,
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1}, append_jsonl=append_jsonl,
        consciousness=SimpleNamespace(inject_observation=lambda *_: None, pause=lambda: None, resume=lambda: None),
        handle_chat_direct=lane.handle_chat_direct,
        send_with_budget=send, get_chat_agent=lambda: SimpleNamespace(_busy=False),
    )
    for actual in (state.DRIVE_ROOT, queue.DRIVE_ROOT, git_ops.DRIVE_ROOT, workers.DRIVE_ROOT):
        assert Path(actual).resolve() == root.resolve()
    assert state.STATE_PATH == root / "state/state.json"
    assert queue.QUEUE_SNAPSHOT_PATH == root / "state/queue_snapshot.json"
    assert config.SETTINGS_PATH == root / "settings.json"
    assert not config.SETTINGS_PATH.exists()
    assert os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") != "1"
    assert os.environ.get("OUROBOROS_ALLOW_LIVE_REPO_TESTS") != "1"
    return SimpleNamespace(root=root, ctx=ctx, bridge=bridge, pending=pending, running=running,
                           notices=notices, attempts=model_attempts, monkeypatch=monkeypatch,
                           run_async=event_loop.run_until_complete)


def incoming_case(host, room, *, caption_only=False, wordless=False):
    project_id, chat_id = "", 1
    if room == "project":
        project_id = "room-project"
        project = create_project(host.root, project_id, name="Room Project")
        chat_id = int(project["chat_id"])
    kind = "wordless" if wordless else "caption" if caption_only else "text"
    client_message_id = f"swarm-probe-{room}-{kind}"
    # A wordless send's canonical row holds the host placeholder, not owner words.
    logged = WORDLESS if wordless else EXACT_OWNER
    ref = build_owner_message_ref(chat_id=chat_id, client_message_id=client_message_id,
                                  ts="2026-09-12T12:00:00Z", text=logged)
    append_jsonl(host.root / "logs/chat.jsonl", {"ts": ref["ts"], "role": "user", "chat_id": chat_id,
                 "client_message_id": client_message_id, "text": logged,
                 **({"text_placeholder": True} if wordless else {})})
    source = host.root / "input.txt"
    source.write_bytes(b"exact-owner-attachment\n")
    incoming = {
        "chat_id": chat_id, "text": "" if caption_only or wordless else EXACT_OWNER,
        "image_caption": EXACT_OWNER if caption_only else "", "log_text": logged,
        "client_message_id": client_message_id, "origin_message_ref": ref,
        "source": "web", "task_constraint": CONSTRAINT,
        "task_metadata": {"force_plan": True, "force_plan_source": "swarm",
                          "project_id": "untrusted-other-project", "client_surface": SURFACE,
                          "chat_attachment_uploads": [{"path": str(source), "label": "Owner input"}]},
    }
    token, task_id = _derived_identity(client_message_id, "swarm", 0)
    return SimpleNamespace(incoming=incoming, chat_id=chat_id, project_id=project_id, source=source,
                           client_message_id=client_message_id, token=token, task_id=task_id, ref=ref)


def promotion_event(case):
    """Existing internal event shape, not a replacement router implementation."""
    return {
        "type": "promote_chat_to_task", "task_id": case.task_id, "routing_token": case.token,
        "objective": case.incoming["text"] or case.incoming["image_caption"],
        "chat_id": case.chat_id, "project_id": case.project_id, "client_message_id": case.client_message_id,
        "task_constraint": CONSTRAINT, "source_ref": case.ref, "source_text": EXACT_OWNER,
        "client_surface": SURFACE, "attachment_uploads": case.incoming["task_metadata"]["chat_attachment_uploads"],
        "force_plan": True, "force_plan_source": "swarm", "workspace": "none",
    }


def capture(host, case, **extra):
    return {
        "annotations": rows(host.root / "logs/chat_annotations.jsonl"),
        "scheduled_result": load_task_result(host.root, case.task_id),
        "queue_snapshot": (json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
                           if queue.QUEUE_SNAPSHOT_PATH.exists() else None),
        **extra,
    }


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("caption_only", [False, True], ids=["text", "caption"])
def test_swarm_press_admits_root_without_any_model_call(host, room, caption_only):
    case = incoming_case(host, room, caption_only=caption_only)
    if room == "project":
        host.running["existing-project-root"] = {"task": {"id": "existing-project-root", "root_task_id": "existing-project-root",
            "delegation_role": "root", "type": "task", "project_id": case.project_id, "chat_id": case.chat_id,
            "text": "An existing task", "drive_root": str(host.root)}}
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    fact = capture(host, case)
    assert host.attempts == [], "Swarm admission must not enter a model-owning chat lane"
    assert len(host.pending) == 1
    task = host.pending[0]
    assert task["id"] == case.task_id and task["root_task_id"] == case.task_id
    assert task["chat_id"] == case.chat_id and str(task.get("project_id") or "") == case.project_id
    assert task["objective"] == EXACT_OWNER
    assert task["metadata"]["force_plan"] is True and task["metadata"]["force_plan_source"] == "swarm"
    assert task["origin_message_ref"] == case.ref and task["origin_message_text"] == EXACT_OWNER
    assert task["metadata"]["client_surface"] == SURFACE
    assert len(task["attachment_manifest"]) == 1
    assert Path(task["attachment_manifest"][0]["abs_path"]).read_bytes() == case.source.read_bytes()
    assert fact["queue_snapshot"]["pending"][0]["task"]["origin_message_ref"] == case.ref
    assert len(fact["annotations"]) == 1 and fact["annotations"][0]["status"] == "scheduled"
    assert fact["scheduled_result"]["status"] == "scheduled"
    assert not host.notices
    assert not (host.root / "memory/owner_mailbox/existing-project-root.jsonl").exists()


@pytest.mark.parametrize("room", ["main", "project"])
def test_swarm_press_admits_a_wordless_attachment_message(host, room):
    # The web gateway no longer invents a caption for an uploaded photo: a send with
    # attachments and no words reaches routing with neither, under the row's placeholder.
    case = incoming_case(host, room, wordless=True)
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    fact = capture(host, case)
    assert host.attempts == [] and not host.notices
    assert len(host.pending) == 1
    task = host.pending[0]
    assert task["id"] == case.task_id and str(task.get("project_id") or "") == case.project_id
    assert task["objective"] == task["origin_message_text"] == WORDLESS
    assert task["origin_message_ref"] == case.ref
    assert task["metadata"]["force_plan"] is True and task["metadata"]["force_plan_source"] == "swarm"
    assert len(task["attachment_manifest"]) == 1
    assert Path(task["attachment_manifest"][0]["abs_path"]).read_bytes() == case.source.read_bytes()
    assert [row["status"] for row in fact["annotations"]] == ["scheduled"]
    assert fact["scheduled_result"]["status"] == "scheduled"


@pytest.mark.parametrize("room", ["main", "project"])
def test_internal_admission_preserves_scope_origin_attachments_force_plan(host, room):
    case = incoming_case(host, room)
    outcome = _handle_promote_chat_to_task(promotion_event(case), host.ctx)
    fact = capture(host, case, outcome=outcome)
    assert outcome["status"] == "scheduled"
    assert len(host.pending) == 1 and len(fact["annotations"]) == 1
    task = host.pending[0]
    assert task["id"] == case.task_id and task["root_task_id"] == case.task_id
    assert task["chat_id"] == case.chat_id and str(task.get("project_id") or "") == case.project_id
    assert task["origin_message_ref"] == case.ref and task["origin_message_text"] == EXACT_OWNER
    assert task["metadata"]["force_plan"] is True and task["metadata"]["force_plan_source"] == "swarm"
    assert task["metadata"]["client_surface"] == SURFACE
    manifest = task["attachment_manifest"]
    assert len(manifest) == 1 and manifest[0]["status"] == "staged"
    assert Path(manifest[0]["abs_path"]).read_bytes() == case.source.read_bytes()
    assert fact["scheduled_result"]["status"] == "scheduled"
    assert fact["scheduled_result"]["description"] == EXACT_OWNER
    assert fact["queue_snapshot"]["pending"][0]["task"]["origin_message_ref"] == case.ref
    if room == "project":
        assert project_binding_for_task(host.root, case.task_id)["source_ref"] == case.ref
    assert not host.attempts


@pytest.mark.parametrize("room", ["main", "project"])
def test_internal_admission_keeps_exact_objective_bytes(host, room):
    case = incoming_case(host, room)
    outcome = _handle_promote_chat_to_task(promotion_event(case), host.ctx)
    fact = capture(host, case, outcome=outcome)
    assert outcome["status"] == "scheduled"
    assert fact["scheduled_result"]["description"] == EXACT_OWNER
    assert host.pending[0]["objective"] == EXACT_OWNER, "queued objective lost owner whitespace before managed execution"
    assert host.pending[0]["description"] == EXACT_OWNER
    assert host.pending[0]["text"].startswith(EXACT_OWNER)
    # Frozen task-contract normalization stays separate from the exact owner input.
    assert host.pending[0]["task_contract"]["objective"] == EXACT_OWNER.strip()


@pytest.mark.parametrize("room", ["main", "project"])
@pytest.mark.parametrize("phase", ["pending", "running", "terminal"])
def test_replayed_client_message_id_admits_nothing_new(host, room, phase):
    case = incoming_case(host, room)
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    assert len(host.pending) == 1
    if phase != "pending":
        task = host.pending.pop()
        if phase == "running":
            host.running[case.task_id] = {"task": task}
        write_task_result(host.root, case.task_id, "running" if phase == "running" else "completed")
    before = capture(host, case)
    before_pending, before_running = copy.deepcopy(host.pending), copy.deepcopy(host.running)
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    assert host.pending == before_pending and host.running == before_running
    assert capture(host, case) == before
    assert len(before["annotations"]) == 1 and before["annotations"][0]["status"] == "scheduled"
    assert len(host.bridge.acks) == 1
    admissions = [row for row in rows(host.root / "logs/supervisor.jsonl")
                  if row.get("type") == "promote_chat_to_task_admitted"]
    assert len(admissions) == 1
    assert not host.attempts and not host.notices


@pytest.mark.parametrize("collision", ["reserved", "different_token", "unreadable", "live_only"])
def test_replay_identity_requires_original_durable_token(host, collision):
    case = incoming_case(host, "main")
    host.pending.append({"id": case.task_id})
    if collision in {"reserved", "different_token"}:
        write_task_result(host.root, case.task_id, "scheduled", promotion_admission={
            "routing_token": case.token if collision == "reserved" else "other-token",
            "status": "scheduled",
        })
    if collision == "reserved":
        queue.ADMISSION_RESERVATIONS[case.task_id] = "other-token"
    if collision == "unreadable":
        path = host.root / "task_results" / f"{case.task_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text("{broken")
    result = queue.reserve_task_admission(case.task_id, case.token, drive_root=host.root)
    assert result == {"status": "blocked", "reason": (
        "task_id_lookup_failed" if collision == "unreadable" else "duplicate_task_id"
    )}
    assert host.pending == [{"id": case.task_id}]
    assert not host.attempts


@pytest.mark.parametrize("failure", ["update", "budget", "ledger"])
def test_existing_admission_refusal_text_and_order(host, failure):
    case = incoming_case(host, "main")
    budget_calls = []
    if failure == "update":
        host.monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "destructive_test_window")

    def budget(_state, *, strict, **_display_read):
        budget_calls.append({"strict": strict})
        if failure == "update":
            pytest.fail("budget floor ran before the rejecting owner-conversation floor")
        if failure == "ledger":
            raise OSError("isolated ledger unavailable")
        return 0

    host.monkeypatch.setattr(state, "budget_remaining", budget)
    server._route_owner_message(host.bridge, host.ctx, case.incoming)
    fact = capture(host, case, budget_calls=budget_calls)
    assert [row["text"] for row in host.notices] == [REFUSALS[failure]]
    assert budget_calls == ([] if failure == "update" else [{"strict": True}])
    assert not host.pending and not fact["annotations"] and not fact["scheduled_result"]
    assert not host.attempts
