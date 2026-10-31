"""#1545: a review the Skills page or an install starts leaves the llm_usage row a task's review leaves.

Consumer checks over the real doors and the real supervisor path. ``POST /api/skills/<skill>/review``
and the marketplace auto-review run a real review wave (``run_review_request``) whose transport is
faked at the physical send, charging the ledger once per send as the real client does. The
substrate's own emitter hands the row to the attached server bus when the seat reports it, and the
supervisor's ordinary drain and ``llm_usage`` handler write ``events.jsonl``, serve
``GET /api/logs/events`` and push one live frame addressed to the panel (inner ``chat_id`` 0, which
Main rejects). Publishing charges nothing, never creates a bus, and only usage rows take this way.
The last two run the shipped ``review_skill`` itself (``review_change``, the coordinator's own seats),
so the skill a seat's row and attempt name comes from the skill boundary, never from the test.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros import skill_review
from ouroboros import usage_accounting as ua
from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request
from ouroboros.skill_review import SkillReviewOutcome
from ouroboros.tools.review_helpers import emit_review_event, emit_review_usage
from supervisor import workers
from tests._usage_store_testing import ledger_rows
from tests.test_extensions_api import _make_client, _stop_patches


class _PhysicalSend:
    """A reviewer transport faked at the physical send: one settled ledger attempt per send."""

    def __init__(self, *, content=None, gate=None):
        self.content = content if content is not None else json.dumps({"verdict": "PASS", "findings": []})
        self.gate, self.entered, self.attempts = gate, threading.Event(), []

    def chat(self, **kwargs):
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(30), "the test never released the seat"
        held = ua.reserve_attempt(ua.AttemptRequest(model=kwargs["model"], provider="fixture", reservation_usd=0.05))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {}, cost_usd=0.02, cost_final=True)
        self.attempts.append(held.attempt_id)
        return {"content": self.content}, {"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 0,
                                           "cost": 0.02, "provider": "fixture", "ledger_attempt_ids": [held.attempt_id]}


def _wave(ctx, skill_name, transport):
    run_review_request(
        ReviewRequest(surface="skill_review", goal=f"review {skill_name}", task_id=ctx.task_id,
                      usage_attribution={"review_skill": skill_name, "review_wave_id": f"wave-{skill_name}"}),
        slots=[ReviewSlot(slot_id="seat-a", model="fixture/reviewer")],
        drive_root=ctx.drive_root, llm=transport, usage_ctx=ctx)


def _reviewer(transport, seen, *, leave=False, reused=False, verdict="clean"):
    """``review_skill`` running one real wave. ``leave``: the author leaves it and its seat answers
    later; ``reused``: a settled record of the same bytes answers at $0, no panel sits."""
    def review(ctx, skill_name, **_kwargs):
        seen["ctx"] = ctx
        if reused:
            return SkillReviewOutcome(skill_name=skill_name, status="clean", reused=True)
        if leave:
            seen["seat"] = threading.Thread(target=_wave, args=(ctx, skill_name, transport), daemon=True)
            seen["seat"].start()
            return SkillReviewOutcome(skill_name=skill_name, status="pending", pending=True)
        _wave(ctx, skill_name, transport)
        return SkillReviewOutcome(skill_name=skill_name, status=verdict, paid=True,
                                  error="" if verdict == "clean" else "the seat returned no verdict")
    return review


@pytest.fixture
def server_bus(monkeypatch):
    """The server process's attached event bus (``workers.get_event_q``'s process-lifetime queue)."""
    bus = queue.Queue()
    monkeypatch.setattr(workers, "_EVENT_Q", bus)
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", False)
    return bus


@pytest.fixture
def door(tmp_path, monkeypatch):
    skill = tmp_path / "skills" / "alpha"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: alpha\ndescription: Test skill.\nversion: 0.1.0\n"
                                    "type: instruction\n---\nbody\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "skills"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    client, data, patches = _make_client(tmp_path, monkeypatch)
    try:
        yield SimpleNamespace(client=client, data=data, repo=tmp_path / "repo")
    finally:
        _stop_patches(patches)


def _supervise(data, bus, monkeypatch):
    """One pass of the supervisor loop's real drain over the bus; returns the live frames it pushed."""
    from ouroboros.server_liveness import drain_worker_events
    from supervisor import message_bus
    from supervisor import state as supervisor_state

    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    bridge, frames = message_bus.LocalChatBridge(), []
    bridge._broadcast_fn = frames.append
    ctx = SimpleNamespace(DRIVE_ROOT=data, RUNNING={}, append_jsonl=supervisor_state.append_jsonl, bridge=bridge,
                          budget_projection_dirty=False)
    drain_worker_events(bus, ctx, [time.monotonic(), {}, time.thread_time(), None],
                        on_restart=lambda *_a: pytest.fail("no restart was asked"))
    assert bus.empty()
    return [frame for frame in frames if frame.get("type") == "log"]


def _usage_rows(data):
    path = data / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "llm_usage"]


def _assert_review_row(row, attempts, *, task_id):
    assert row["task_id"] == task_id and row["chat_id"] == 0
    assert (row["review_skill"], row["review_wave_id"], row["review_slot_id"]) == ("alpha", "wave-alpha", "seat-a")
    assert row["ledger_attempt_ids"] == attempts and row["accounting_authority"] == "physical_attempt_ledger"
    assert (row["prompt_tokens"], row["completion_tokens"], row["cached_tokens"], row["cache_write_tokens"]) == (
        10, 5, 0, None), "a reported zero stays zero and an unreported counter stays unknown"
    assert (row["provider"], row["category"], row["record_contract_version"]) == ("fixture", "review", 2)


def test_a_skills_page_review_writes_its_usage_row_through_the_supervisor(door, server_bus, monkeypatch):
    transport, seen = _PhysicalSend(), {}
    monkeypatch.setattr(skill_review, "review_skill", _reviewer(transport, seen))
    response = door.client.post("/api/skills/alpha/review", json={})
    assert response.status_code == 200 and response.json()["status"] == "clean", response.text
    spent = ledger_rows(door.data)
    assert [row["attempt_id"] for row in spent] == transport.attempts and len(spent) == 1
    assert [event["type"] for event in server_bus.queue] == ["llm_usage"]
    assert {event["type"] for event in seen["ctx"].pending_events} == {"cognitive_operation"}, \
        "only the usage row leaves the carrier"
    frames = _supervise(door.data, server_bus, monkeypatch)
    assert ledger_rows(door.data) == spent, "publishing the row charges nothing"
    [row] = _usage_rows(door.data)
    _assert_review_row(row, transport.attempts, task_id="api_skill_review")
    tail = door.client.get("/api/logs/events?limit=50").json()["entries"]
    assert [entry["ledger_attempt_ids"] for entry in tail if entry.get("type") == "llm_usage"] == [transport.attempts]
    [frame] = frames
    assert frame["data"]["type"] == "llm_usage" and frame["data"]["chat_id"] == 0 and frame["chat_id"] == 0, \
        "the panel's own address, which Main rejects (web/tests/chat_thread_routing.test.js)"


@pytest.mark.parametrize("ending", ["seat-after-response", "client-cancelled"])
def test_usage_a_review_spends_after_its_request_still_reaches_the_supervisor(door, server_bus, monkeypatch, ending):
    """No drain waits on the request: the row leaves when the seat reports it, which may be after the
    response or after the client went away."""
    from ouroboros.gateway import extensions

    gate, seen = threading.Event(), {}
    transport = _PhysicalSend(gate=gate)
    monkeypatch.setattr(skill_review, "review_skill", _reviewer(transport, seen, leave=ending == "seat-after-response"))
    if ending == "seat-after-response":
        response = door.client.post("/api/skills/alpha/review", json={})
        assert response.status_code == 200 and response.json()["status"] == "pending", response.text
        assert server_bus.empty()
        gate.set()
        seen["seat"].join(30)
    else:
        monkeypatch.setattr(extensions, "_request_drive_root", lambda _request: door.data)
        monkeypatch.setattr(extensions, "_request_repo_dir", lambda _request: door.repo)

        async def cancelled_client():
            request = asyncio.create_task(extensions.api_skill_review(SimpleNamespace(path_params={"skill": "alpha"})))
            while not transport.entered.is_set():
                await asyncio.sleep(0.01)
            request.cancel()
            await asyncio.sleep(0.05)
            assert server_bus.empty()
            gate.set()
            return await request

        assert json.loads(asyncio.run(cancelled_client()).body)["status"] == "clean"
    spent = ledger_rows(door.data)
    assert [event["type"] for event in server_bus.queue] == ["llm_usage"]
    _supervise(door.data, server_bus, monkeypatch)
    assert ledger_rows(door.data) == spent
    [row] = _usage_rows(door.data)
    _assert_review_row(row, transport.attempts, task_id="api_skill_review")


@pytest.mark.parametrize("result", ["unreadable-verdict", "reused-verdict"])
def test_a_failed_review_keeps_its_spent_row_and_a_reused_one_adds_none(door, server_bus, monkeypatch, result):
    seen = {}
    transport = _PhysicalSend(content="not a verdict" if result == "unreadable-verdict" else None)
    monkeypatch.setattr(skill_review, "review_skill", _reviewer(
        transport, seen, reused=result == "reused-verdict", verdict="pending"))
    response = door.client.post("/api/skills/alpha/review", json={})
    assert response.status_code == 200 and response.json()["status"] == (
        "clean" if result == "reused-verdict" else "pending"), response.text
    spent = ledger_rows(door.data)
    _supervise(door.data, server_bus, monkeypatch)
    assert ledger_rows(door.data) == spent
    if result == "reused-verdict":
        assert spent == [] and _usage_rows(door.data) == []
    else:
        [row] = _usage_rows(door.data)
        _assert_review_row(row, transport.attempts, task_id="api_skill_review")


def test_a_marketplace_auto_review_writes_its_row_too(tmp_path, server_bus, monkeypatch):
    from ouroboros.marketplace import install

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    data = tmp_path / "data"
    data.mkdir()
    transport, seen = _PhysicalSend(), {}
    monkeypatch.setattr(skill_review, "review_skill", _reviewer(transport, seen))
    assert install._run_skill_review(data, tmp_path / "repo", "alpha") == ("clean", [], "")
    _supervise(data, server_bus, monkeypatch)
    [row] = _usage_rows(data)
    _assert_review_row(row, transport.attempts, task_id="marketplace_install")


def _carrier(tmp_path, **attributes):
    from ouroboros.gateway.extensions import _ApiReviewCtx

    ctx = _ApiReviewCtx(tmp_path, tmp_path / "repo")
    for key, value in attributes.items():
        setattr(ctx, key, value)
    return ctx


def test_only_a_queue_less_server_carrier_routes_usage_to_the_server_bus(tmp_path, server_bus):
    usage = {"prompt_tokens": 3, "completion_tokens": None, "ledger_attempt_ids": ["a-1"]}
    # A task's own queue keeps every row, exactly as before.
    task_queue = queue.Queue()
    task_carrier = _carrier(tmp_path, event_queue=task_queue)
    emit_review_usage(task_carrier, model="m", usage=usage, source="review")
    assert [event["type"] for event in task_queue.queue] == ["llm_usage"] and task_carrier.pending_events == []
    # A context that did not declare itself a server carrier keeps its own fallback list.
    plain = SimpleNamespace(task_id="t", event_queue=None, pending_events=[])
    emit_review_usage(plain, model="m", usage=usage, source="review")
    assert [event["type"] for event in plain.pending_events] == ["llm_usage"]
    # Other review events of a server carrier stay on its list: only usage is a supervisor row.
    server = _carrier(tmp_path)
    emit_review_event(server, {"type": "cognitive_operation", "task_id": server.task_id, "phase": "started"})
    assert [event["type"] for event in server.pending_events] == ["cognitive_operation"]
    assert server_bus.empty()
    # An explicit producer address is kept; only a missing one takes the carrier's.
    emit_review_usage(server, model="m", usage=usage, source="review", extra={"chat_id": 7})
    emit_review_usage(server, model="m", usage=usage, source="review")
    assert [event["chat_id"] for event in server_bus.queue] == [7, 0]


@pytest.mark.parametrize("bus", ["never-attached", "shutting-down", "put-fails"])
def test_without_a_working_server_bus_the_row_stays_on_the_carrier(tmp_path, monkeypatch, bus):
    """Best effort, like every telemetry row: no bus is created and nothing raises; the ledger keeps the money."""
    class Broken:
        def put(self, _event):
            raise BrokenPipeError("manager connection lost")

    monkeypatch.setattr(workers, "_EVENT_Q", None if bus == "never-attached" else
                        Broken() if bus == "put-fails" else queue.Queue())
    monkeypatch.setattr(workers, "_EVENT_Q_SHUTDOWN", bus == "shutting-down")
    monkeypatch.setattr(workers, "_EVENT_Q_MANAGER", None)
    server = _carrier(tmp_path)
    emit_review_usage(server, model="m", usage={"prompt_tokens": 1}, source="review")
    assert [event["type"] for event in server.pending_events] == ["llm_usage"]
    assert workers._EVENT_Q_MANAGER is None and (workers._EVENT_Q is None) is (bus == "never-attached")


# --- The shipped skill boundary: skill_review -> review_change(subject="skill") -> the coordinator's seats ---

CLEAN_SKILL_ANSWER = "[]\nNO_FINDINGS"


@pytest.fixture
def skill_door(tmp_path, monkeypatch, server_bus):
    """``POST /api/skills/weather/review`` over the shipped wave: a reviewable skill in the drive, the
    installed body as its governance root and a review pool. Nothing names the skill to the seats or the
    ledger by hand; only the provider send (``_PhysicalSend``) and, for a session seat, the engine are fakes."""
    import server as srv

    from tests import _contributor_packet_shared as shared
    from tests.review_pool_rosters import FACTORY_MODELS, pool_roster, pool_seat, set_review_pool
    from tests.test_skill_exec import _build_skill

    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", "")
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    system = shared.init_installed_body(tmp_path / "ouroboros")["repo"]
    drive = tmp_path / "ouroboros" / "data"
    for sub in ("logs", "locks", "state"):
        (drive / sub).mkdir(parents=True)
    _build_skill(drive / "skills" / "external", "weather")
    transport = _PhysicalSend(content=CLEAN_SKILL_ANSWER)
    monkeypatch.setattr("ouroboros.tools.review.LLMClient", lambda: transport)
    client, _unused, patches = _make_client(tmp_path, monkeypatch)
    srv.app.app.state.drive_root, srv.app.app.state.repo_dir = drive, system

    def pool(*session_rows, api=True):
        api_rows = [pool_seat(f"review-{index}", model) for index, model in enumerate(FACTORY_MODELS[:2], start=1)]
        set_review_pool(monkeypatch, pool_roster(*(api_rows if api else []), *session_rows))

    pool()
    try:
        yield SimpleNamespace(client=client, drive=drive, transport=transport, bus=server_bus, pool=pool)
    finally:
        _stop_patches(patches)


def _review_record(drive, record_id):
    from ouroboros import review_ledger

    return review_ledger.load_record(drive, record_id) or {}


def test_a_skills_page_review_names_its_skill_on_every_row_and_attempt_of_the_shipped_wave(skill_door, monkeypatch):
    """#1545 F1: the real door, the real ``review_skill`` and the real seats. Each physical send's ledger
    attempt and its ``llm_usage`` row carry the reviewed skill beside the record's wave and their own seat."""
    door = skill_door
    response = door.client.post("/api/skills/weather/review", json={})
    payload = response.json()
    assert response.status_code == 200 and payload["status"] == "clean", payload
    record = _review_record(door.drive, payload["record_id"])
    spent = ledger_rows(door.drive)
    _supervise(door.drive, door.bus, monkeypatch)
    rows = _usage_rows(door.drive)
    seats = sorted(str(seat.get("seat_id") or "") for seat in record.get("rows") or [])
    assert seats == ["review-1", "review-2"] and len(spent) == len(rows) == 2, (seats, spent, rows)
    assert sorted(a for row in rows for a in row["ledger_attempt_ids"]) == sorted(door.transport.attempts) == sorted(
        attempt["attempt_id"] for attempt in spent), "one row per physical send, 1:1 with the ledger"
    assert sorted(row["review_slot_id"] for row in rows) == sorted(attempt["review_slot_id"] for attempt in spent) == seats
    for item in (*rows, *spent):
        assert (item.get("review_skill"), item.get("review_wave_id")) == ("weather", record["review_wave_id"]), item
    assert {row["task_id"] for row in rows} == {"api_skill_review"} and {row["chat_id"] for row in rows} == {0}


def _held_engine(tmp_path, monkeypatch):
    """The engine as the shared fake /v2 surface: a session run still running through its seat window, whose
    deadline cancel the engine answers only when the test says so; the verify read then finds that the run
    had finished on its own (completion wins, #1547)."""
    from ouroboros import delegate_custody as custody
    from tests._review_session_route_shared import FakeGateway, _terminal_detail

    class Engine(FakeGateway):
        run_id = f"run-{uuid.uuid4().hex[:12]}"  # never a latched id of another test in this process
        cancel_reached, release = threading.Event(), threading.Event()

        def start_run(self, request, *, idempotency_key=""):
            super().start_run(request, idempotency_key=idempotency_key)
            return {"runId": Engine.run_id, "runDir": str(FakeGateway.run_dir)}

        def cancel_run(self, run_id, *, reason=""):
            Engine.cancel_reached.set()
            assert Engine.release.wait(60), "the test never let the engine answer the cancel"
            FakeGateway.nonterminal = False
            return super().cancel_run(run_id, reason=reason)

    FakeGateway.reset()
    detail = _terminal_detail('{"findings": []}', conformance="passed")
    detail["summary"].update(spendUsd=0.04, inputTokens=1200, outputTokens=300)
    for name, value in (("run_dir", tmp_path / "review-run"), ("detail", detail), ("nonterminal", True)):
        monkeypatch.setattr(FakeGateway, name, value)
    monkeypatch.setattr("ouroboros.gateways.claudexor.ClaudexorGateway", Engine)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: Engine())
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: Engine())
    custody._CUSTODY.clear()
    return Engine


def _until(predicate, timeout=60.0, what="the condition"):
    deadline = time.monotonic() + timeout
    while not (value := predicate()):
        assert time.monotonic() < deadline, f"{what} did not happen in {timeout:g}s"
        time.sleep(0.05)
    return value


def test_a_session_seat_that_settles_after_the_skills_page_answered_names_its_skill(skill_door, monkeypatch, tmp_path):
    """In its own process a Skills-page review's late session row comes from the seat's worker, which outlives
    the request (pending) and writes the row when the engine answers; the re-issue then collects the wave at $0.
    That row reaches the supervisor naming the skill, as do the session's subscription attempt and its start
    custody (the attribution a cold collection in another process reads, ``review_operation._attach_only_run``)."""
    from ouroboros import delegate_custody as custody
    from tests.review_pool_rosters import pool_seat

    door = skill_door
    monkeypatch.setenv("OUROBOROS_REVIEW_MODEL_TIMEOUT_SEC", "2")
    # One read-only reviewer row (the api seats are the test above): its session runs in the engine's own
    # envelope; a full-access row would need an engine that binds a private checking root, which this fake lacks.
    door.pool(pool_seat("review-s", "fake-review=fake-small", kind="agent_session", access="readonly"), api=False)
    engine = _held_engine(tmp_path, monkeypatch)
    response = door.client.post("/api/skills/weather/review", json={})
    payload = response.json()
    assert response.status_code == 200 and payload["status"] == "pending", (payload.get("panel") or {}).get("not_reached")
    assert engine.cancel_reached.wait(30), [(len(g.start_requests), len(g.run_gets), g.cancels) for g in engine.instances]
    assert door.bus.empty(), "the seat had not answered when the request did"
    engine.release.set()
    _until(lambda: not door.bus.empty(), what="the session row")
    spent = ledger_rows(door.drive)
    _supervise(door.drive, door.bus, monkeypatch)
    [session_row] = _usage_rows(door.drive)
    # The re-issue collects the wave once the worker has settled the seat (until then it answers pending):
    # no second start, attempt or row.
    again = _until(lambda: (lambda answer: answer if answer.get("status") != "pending" else None)(
        door.client.post("/api/skills/weather/review", json={}).json()), what="the re-issue's collection")
    assert again["status"] == "clean" and again["record_id"] == payload["record_id"], again
    record = _review_record(door.drive, payload["record_id"])
    _supervise(door.drive, door.bus, monkeypatch)
    assert record["state"] == "settled" and ledger_rows(door.drive) == spent and _usage_rows(door.drive) == [session_row]
    assert [len(gateway.start_requests) for gateway in engine.instances if gateway.start_requests] == [1]
    [session_attempt] = spent
    started = custody.replay(door.drive)[engine.run_id]
    assert (session_row["task_id"], session_row["chat_id"], session_row["review_slot_id"]) == ("api_skill_review", 0, "review-s")
    # A session row names its run; the run's subscription attempt is keyed by that run's hash.
    assert session_row["claudexor"]["run_id"] == engine.run_id and session_attempt["session_id_sha256"] == \
        hashlib.sha256(engine.run_id.encode("utf-8")).hexdigest(), (session_row, session_attempt)
    for item in (session_row, session_attempt, vars(started)):
        assert (item.get("review_skill"), item.get("review_wave_id")) == ("weather", record["review_wave_id"]), item
