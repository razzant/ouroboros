"""#1578 B3: one reviewer failure is one durable, canonical ``review_model_error`` and one live frame.

Driven through the real coordinator, the manual collector, a replay or recovery of
a keyed cycle, the parser path and the system (deep) review's direct executor. The
owner (``triad_review._append_review_model_error``) writes the CANONICAL owner
root's ``logs/events.jsonl``; only a successful append is acknowledged on the
actor (``error_event_logged``), so a failed append is written by a later
collection, and a logged one never twice.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import triad_review, utils
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot

pytestmark = pytest.mark.serial


def _ctx(root, **extra):
    return SimpleNamespace(
        drive_root=root, task_id="child", task_metadata={"root_task_id": "root", **extra},
        drive_logs=lambda: root / "logs", event_queue=queue.Queue(), pending_events=[],
    )


def _failed_actor():
    return ReviewActorRecord(
        slot_id="slot-stable", model="requested-model", status="error",
        error="Delegated run failed: provider unavailable", reported_cause="upstream socket closed",
        failure_code="provider_unavailable", http_status=503,
        operation_id="operation-7", transport_status="provider_transport_error",
        usage={"provider": "claudexor", "delegated_run_id": "run-real"},
        prompt_ref={"call_id": "prompt-1"}, response_ref={"call_id": "response-1"},
    )


def _rows(root):
    path = root / "logs" / "events.jsonl"
    return [row for line in path.read_text().splitlines()
            if (row := json.loads(line)).get("type") == "review_model_error"] if path.exists() else []


def _run(root, monkeypatch, actor, *, surface="multi_model_review", ctx=None):
    from ouroboros.review_substrate import ReviewCoordinator

    monkeypatch.setattr("ouroboros.review_custody.run_custodied_review_slots", lambda **kw: [actor])
    monkeypatch.setattr("ouroboros.reviewer_slot_config.record_reviewer_slot_executions", lambda *a, **kw: None)
    request = ReviewRequest(surface=surface, goal="review test", task_id="child", retry_key="wave-known")
    coordinator = ReviewCoordinator(llm=object(), drive_root=root, usage_ctx=ctx)
    return coordinator.run(request, [ReviewSlot(slot_id=actor.slot_id, model="requested-model")])


@pytest.mark.parametrize("surface", [
    "multi_model_review", "skill_review", "plan_review", "scope_review", "advisory_review",
])
def test_a_run_failure_has_cause_identity_and_one_live_frame(tmp_path, monkeypatch, surface):
    from supervisor.events_worker_reports import _handle_log_event

    ctx = _ctx(tmp_path)
    # The production worker sink is this log_event envelope; its supervisor handler
    # forwards it live without appending a second event row.
    monkeypatch.setattr(utils, "_log_sink", lambda row: utils.emit_log_event(ctx.event_queue, row))
    run = _run(tmp_path, monkeypatch, _failed_actor(), surface=surface, ctx=ctx)
    [row] = _rows(tmp_path)
    assert row["type"] == "review_model_error" and row["source"] == surface
    assert row["error"] == "Delegated run failed: provider unavailable"
    assert row["reported_cause"] == "upstream socket closed"
    assert {key: row[key] for key in ("task_id", "root_task_id", "review_wave_id", "slot_id")} == {
        "task_id": "child", "root_task_id": "root", "review_wave_id": "wave-known", "slot_id": "slot-stable"}
    assert row["provider"] == "claudexor" and row["delegated_run_id"] == "run-real"
    assert row["failure_code"] == "provider_unavailable" and row["http_status"] == 503
    assert row["prompt_ref"] == {"call_id": "prompt-1"} and row["response_ref"] == {"call_id": "response-1"}
    frames = []
    supervisor = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}, bridge=SimpleNamespace(push_log=frames.append))
    envelopes = []
    while not ctx.event_queue.empty():
        envelopes.append(ctx.event_queue.get_nowait())
    [envelope] = [row for row in envelopes if row.get("data", {}).get("type") == "review_model_error"]
    assert envelope["type"] == "log_event"
    _handle_log_event(envelope, supervisor)
    assert len(frames) == 1 and frames[0]["error"] == row["error"]
    assert ctx.event_queue.empty() and len(_rows(tmp_path)) == 1
    assert run.actors[0]["error_event_logged"] is True


def test_preflight_control_timeout_survives_the_existing_review_error_record(tmp_path, monkeypatch):
    import httpx

    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as cx
    from ouroboros.observability import read_call_payload
    from ouroboros.review_substrate import ReviewCoordinator
    from tests._review_session_route_shared import _agent_request, _agent_slot

    calls = []

    def fail(request):
        calls.append((request.method, request.url.path))
        raise httpx.ReadTimeout("private provider details", request=request)

    gateway = cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(fail))
    clock = iter((10.0, 70.125))
    monkeypatch.setattr(cx, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setattr(utils, "_log_sink", None)
    request = _agent_request(task_id="child", session_root=str(tmp_path))
    slot = _agent_slot(session_target="codex=gpt-6-astra", session_access="full")
    actor = ReviewCoordinator(llm=object(), drive_root=tmp_path)._run_slot(request, slot)
    expected = "Claudexor control GET /v2/agent-capabilities failed after 60.125s: ReadTimeout"
    assert calls == [("GET", "/v2/agent-capabilities")], "no reviewer POST follows failed preparation"
    assert actor.error == expected and actor.failure_code == "daemon_unreachable"
    assert actor.operation_state == "not_dispatched" and actor.http_status is None
    assert actor.reported_cause == "" and actor.usage["delegated_start_unrequested"] is True
    manifest, payload, _ = read_call_payload(tmp_path, task_id="child", call_id=actor.operation_id + "_error")
    assert manifest["producer_complete"] is True
    assert payload["error"] == payload["producer_outcome"]["error"] == expected
    assert payload["producer_outcome"]["operation_state"] == "not_dispatched"
    projected = _run(tmp_path, monkeypatch, actor, ctx=_ctx(tmp_path))
    assert projected.actors[0]["error"] == _rows(tmp_path)[0]["error"] == expected


def test_a_child_drive_failure_lands_in_the_canonical_owner_root(tmp_path, monkeypatch):
    """A delegated child's task drive is not the log the owner reads: the canonical root is."""
    canonical = tmp_path / "data"
    child = canonical / "task_drives" / "child"
    child.mkdir(parents=True)
    monkeypatch.setattr(utils, "_log_sink", None)
    ctx = _ctx(child)
    _run(child, monkeypatch, _failed_actor(), ctx=ctx)
    assert _rows(child) == []
    [row] = _rows(canonical)
    assert row["task_id"] == "child" and row["root_task_id"] == "root"
    budget_root = tmp_path / "budget"
    budget_root.mkdir()
    _run(child, monkeypatch, _failed_actor(), ctx=_ctx(child, budget_drive_root=str(budget_root)))
    assert len(_rows(budget_root)) == 1 and len(_rows(canonical)) == 1  # the task's budget root wins


def test_the_manual_collector_reuses_the_logged_run_failure(tmp_path, monkeypatch):
    from ouroboros.tools.review import _collect_review_findings
    from ouroboros.tools.review_response import parse_model_response, review_operation_fields
    from supervisor.log_addressing import make_server_log_sink

    frames = []
    monkeypatch.setattr(utils, "_log_sink", make_server_log_sink(SimpleNamespace(push_log=frames.append),
                                                                 tmp_path, running={}))
    ctx = _ctx(tmp_path)
    actor = _run(tmp_path, monkeypatch, _failed_actor(), ctx=ctx).actors[0]
    envelope = parse_model_response("requested-model", {**actor, **review_operation_fields(actor)}, None)
    _critical, _advisory, errored, raw = _collect_review_findings(ctx, [envelope])
    assert errored and raw[0]["error_event_logged"] is True
    assert len(_rows(tmp_path)) == len([row for row in frames if row.get("type") == "review_model_error"]) == 1
    assert ctx.pending_events == [] and ctx.event_queue.empty()


def test_a_failed_append_is_not_acknowledged_and_a_later_collection_writes_it_once(tmp_path, monkeypatch):
    ctx, frames = _ctx(tmp_path), []
    actor = _failed_actor()
    request = ReviewRequest(surface="multi_model_review", goal="review test", task_id="child", resolved_wave_id="wave")
    monkeypatch.setattr(utils, "_log_sink", frames.append)
    append = triad_review.append_jsonl
    monkeypatch.setattr(triad_review, "append_jsonl", lambda *a, **kw: False)
    triad_review.emit_review_run_errors(ctx, request, [actor], drive_root=tmp_path)
    assert actor.error_event_logged is False
    assert frames == _rows(tmp_path) == ctx.pending_events == [] and ctx.event_queue.empty()
    monkeypatch.setattr(triad_review, "append_jsonl", append)
    triad_review.emit_review_run_errors(ctx, request, [actor], drive_root=tmp_path)
    triad_review.emit_review_run_errors(ctx, request, [actor], drive_root=tmp_path)
    assert actor.error_event_logged is True
    assert len(frames) == len(_rows(tmp_path)) == 1


class _TerminalProviderFailure:
    """The real coordinator's executor seam answered by a typed terminal provider HTTP 503."""

    def __init__(self, assignment, release=None):
        self.slot_id, self.release = assignment.slot.slot_id, release

    def restore_custody(self, _state):
        return None

    def set_pending_invocation_checkpoint(self, _checkpoint):
        return None

    def prompt_payload(self):
        return {"messages": []}

    def prompt_chars(self):
        return 0

    def failure_custody(self):
        return {}

    def execute(self):
        if self.release is not None and self.slot_id == "late":
            self.release.wait(5)
        error = RuntimeError(f"{self.slot_id}: provider HTTP 503")
        error.physical_attempt_capture = SimpleNamespace(state="unresolved", provider_status_code=503)
        raise error


def _model_errors(root):
    return [row for row in _rows(root) if row["type"] == "review_model_error"]


@pytest.mark.parametrize("append_ok", [True, False])
def test_same_cycle_replay_keeps_the_settled_operations_append_receipt(tmp_path, monkeypatch, append_ok):
    """A keyed plan cycle replays a fast slot's settled API failure from exact-operation
    custody while its sibling is still in flight. The coordinator's successful append
    must reach that settled copy, so the replay never appends it again; a failed append
    stays unacknowledged, and the next replay writes it exactly once."""
    import threading
    import time

    from ouroboros.review_custody import _ACTIVE
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import run_review_request

    monkeypatch.setattr(utils, "_log_sink", None)
    release = threading.Event()
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda assignment, **_kw: _TerminalProviderFailure(assignment, release))
    request = ReviewRequest(surface="plan_review", goal="review", task_id="child", retry_key="plan_review:child:1")
    slots = [ReviewSlot(slot_id="fast", model="test/model", route=ReviewRouteKind.API_CHAT, timeout_sec=2.0),
             ReviewSlot(slot_id="late", model="test/model", route=ReviewRouteKind.API_CHAT, timeout_sec=0.05)]
    ctx, append = SimpleNamespace(), triad_review.append_jsonl
    try:
        if not append_ok:
            monkeypatch.setattr(triad_review, "append_jsonl", lambda *a, **kw: False)
        runs = [run_review_request(request, slots=slots, drive_root=tmp_path, usage_ctx=ctx)]
        monkeypatch.setattr(triad_review, "append_jsonl", append)
        runs += [run_review_request(request, slots=slots, drive_root=tmp_path, usage_ctx=ctx) for _ in range(2)]
    finally:
        release.set()
    late = runs[0].actors[1]["operation_id"]
    deadline = time.monotonic() + 5
    while any(entry.operation_id == late for entry in list(_ACTIVE.values())) and time.monotonic() < deadline:
        time.sleep(0.01)
    fast = [run.actors[0] for run in runs]
    assert len({actor["operation_id"] for actor in fast}) == 1  # one paid operation, replayed
    assert all(run.actors[1]["late_result_pending"] for run in runs)  # its sibling never settled meanwhile
    assert sorted(row["slot_id"] for row in _model_errors(tmp_path)) == ["fast", "late"]
    assert [actor["error_event_logged"] for actor in fast] == [append_ok, True, True]


def test_cold_recovery_of_a_logged_operation_keeps_the_recorded_receipt(tmp_path, monkeypatch):
    """After a restart the settled copy is gone: the recorded plan wave row and the
    operation's complete producer CAS are its custody. The CAS outcome was stamped
    before the failure was logged, so the recorded row's positive receipt for that
    SAME operation must survive recovery."""
    from dataclasses import replace

    from ouroboros.review_custody import _freeze_roster_rows
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_substrate import run_review_request
    from ouroboros.tools.plan_review_runtime import _plan_row_from_actor

    monkeypatch.setattr(utils, "_log_sink", None)
    monkeypatch.setattr("ouroboros.review_substrate._review_route_executor",
                        lambda assignment, **_kw: _TerminalProviderFailure(assignment))
    key = "plan_review:" + "a" * 64 + ":1"
    request = ReviewRequest(
        surface="plan_review", goal="review", task_id="child", retry_key=key,
        reconciliation_identity={"subject_hash": "a" * 64, "epoch": key, "roster_hash": "roster",
                                 "health_epoch": [], "root_task_id": "root", "task_attempt": 1,
                                 "review_contract": "frozen-contract"})
    slot = ReviewSlot(slot_id="fast", model="test/model", route=ReviewRouteKind.API_CHAT)
    [actor] = run_review_request(request, slots=[slot], drive_root=tmp_path,
                                 usage_ctx=SimpleNamespace(drive_root=tmp_path)).actors
    assert actor["error_event_logged"] is True and len(_model_errors(tmp_path)) == 1
    restarted = SimpleNamespace(drive_root=tmp_path)  # a new process: no settled copy
    restarted._review_frozen_rows = {"plan_review": _freeze_roster_rows(
        restarted, "plan_review", [_plan_row_from_actor(actor, slot)])}
    [recovered] = run_review_request(replace(request, reconcile_only=True), slots=[slot],
                                     drive_root=tmp_path, usage_ctx=restarted).actors
    assert recovered["operation_id"] == actor["operation_id"]
    assert recovered["operation_state"] == "late_settled"  # rebuilt from the producer CAS
    assert recovered["error_event_logged"] is True
    assert len(_model_errors(tmp_path)) == 1


def test_a_receipt_marks_only_its_own_settled_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "_log_sink", None)
    older, same = (ReviewActorRecord(slot_id="slot-stable", model="m", status="error", operation_id=op)
                   for op in ("operation-6", "operation-7"))
    ctx = _ctx(tmp_path)
    ctx._review_settled_attempts = {"older-cycle": older, "this-cycle": same}
    request = ReviewRequest(surface="plan_review", goal="review test", task_id="child")
    triad_review.emit_review_run_errors(ctx, request, [_failed_actor()], drive_root=tmp_path)
    assert (older.error_event_logged, same.error_event_logged) == (False, True)


def test_awaited_valid_and_report_answers_write_no_error(tmp_path, monkeypatch):
    from ouroboros.tools.scope_review_contract import SCOPE_REQUIRED_ITEMS

    monkeypatch.setattr(utils, "_log_sink", None)
    for actor, surface in (
        (ReviewActorRecord(slot_id="slot-stable", model="m", status="error", operation_state="pending_dispatch"),
         "multi_model_review"),
        (ReviewActorRecord(slot_id="slot-stable", model="m", status="ok", raw_text=json.dumps([
            {"item": item, "verdict": "PASS", "severity": "advisory", "reason": "checked"}
            for item in sorted(SCOPE_REQUIRED_ITEMS)])),
         "scope_review"),
        (ReviewActorRecord(slot_id="slot-stable", model="m", status="ok", raw_text="# Review\nAll checks passed."),
         "deep_self_review"),
    ):
        _run(tmp_path, monkeypatch, actor, surface=surface)
    assert _rows(tmp_path) == []


@pytest.mark.parametrize("text,status", [("unparseable", "parse_failure"), ("I cannot review this. []", "parse_failure")])
def test_the_skill_parser_records_its_specific_failure_without_invented_ids(tmp_path, monkeypatch, text, status):
    monkeypatch.setattr(utils, "_log_sink", None)
    parsed = triad_review.parse_seat_answers({"results": [{"model": "m", "text": text}]}, {"": ("skill",)})
    ctx = SimpleNamespace(drive_logs=lambda: tmp_path / "logs")  # a context exposing only its log dir
    triad_review.emit_review_model_error_events(ctx, parsed, source="skill_review", skill_name="demo")
    [row] = _rows(tmp_path)
    assert row["status"] == status and row["error"] and row["skill"] == "demo"
    assert not any(row.get(key) for key in ("task_id", "root_task_id", "slot_id", "review_wave_id"))


def test_the_recorded_operation_binding_outranks_the_collecting_context(tmp_path, monkeypatch):
    """A result collected under another context keeps its producer's lineage (``review_operation_binding``)."""
    monkeypatch.setattr(utils, "_log_sink", None)
    binding = {"task_id": "producer-child", "root_task_id": "producer-root", "parent_task_id": "producer-parent"}
    parsed = triad_review.parse_seat_answers({"results": [{
        "model": "m", "text": "provider refused", "verdict": "ERROR", "recovery_binding": binding}]}, {})
    triad_review.emit_review_model_error_events(_ctx(tmp_path), parsed, source="multi_model_review")
    [row] = _rows(tmp_path)
    assert all(row[key] == value for key, value in binding.items())


class _DirectExecutor:
    def __init__(self, assignment, **kwargs):
        self.assignment = assignment

    def prompt_payload(self):
        return {"prompt": "review evidence"}

    def failure_custody(self):
        return {"provider": "claudexor", "delegated_run_id": "run-direct", "delegated_run_started": True}

    def execute(self):
        from ouroboros.review_execution import ReviewRouteUnavailable

        error = ReviewRouteUnavailable("direct provider failure", code="provider_unavailable")
        error.reported_cause = "upstream service stopped"
        raise error


@pytest.fixture
def direct_review_boundaries(monkeypatch):
    """A session availability probe must never prepare an owned engine or use a socket."""
    import socket
    from ouroboros import claudexor_daemon

    def forbidden(*_args, **_kwargs):
        pytest.fail("system review test crossed an unmocked daemon/network boundary")

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.mark.parametrize("empty", [False, True])
def test_the_system_reviews_direct_failure_reaches_the_canonical_log(
    tmp_path, monkeypatch, direct_review_boundaries, empty,
):
    """``/review`` (surface=system) runs deep_self_review's direct executor, outside the
    coordinator: its failure (exception or empty product) still writes the one error."""
    from ouroboros import deep_self_review
    from ouroboros import claudexor_daemon, subagents
    from ouroboros.review_execution import ReviewAttemptResult
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_change import run_review_change
    from tests.review_pool_rosters import mixed_pool_rows, pool_roster, pool_seat, set_review_pool

    repo, canonical = tmp_path / "repo", tmp_path / "data"
    drive = canonical / "task_drives" / "child"
    repo.mkdir()
    (drive / "memory").mkdir(parents=True)
    (repo / "BIBLE.md").write_text("# BIBLE\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id = "child"
    ctx.task_metadata = {"root_task_id": "root"}
    set_review_pool(monkeypatch, pool_roster(*mixed_pool_rows(), pool_seat(
        "deep-session", "codex=gpt-5.6-sol", kind="agent_session", effort="xhigh", marked=False)))
    availability, executed = [], []
    gateway = SimpleNamespace(close=lambda: availability.append("closed"))

    def ensure_gateway(**kwargs):
        assert kwargs == {"admission_wait_sec": 0}
        availability.append("gateway")
        return gateway

    def healthy_route(actual_gateway, route, shape, **kwargs):
        assert actual_gateway is gateway and route == "codex"
        assert kwargs["route_model"] == "gpt-5.6-sol"
        availability.append("healthy")
        return "", ""

    class DirectFailure(_DirectExecutor):
        def execute(self):
            executed.append(self.assignment)
            if empty:
                return ReviewAttemptResult(message={"content": ""}, raw_text="", usage=self.failure_custody())
            return super().execute()

    # Availability precedes the executor: declare that route healthy without
    # starting the daemon. Keep real route selection and deep error handling.
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", ensure_gateway)
    monkeypatch.setattr(subagents, "route_health", healthy_route)
    frames = []
    monkeypatch.setattr(utils, "_log_sink", frames.append)
    monkeypatch.setattr("ouroboros.review_execution._review_route_executor", DirectFailure)
    monkeypatch.setattr("ouroboros.review_native_episode.review_native_transcript_bound", lambda *a, **kw: 400_000)
    monkeypatch.setattr(deep_self_review, "_record_execution", lambda *a, **kw: None)
    monkeypatch.setattr(deep_self_review, "_retrieving_task", lambda *a, **kw: ("review", {
        "memory": {"inlined": 0, "total": 0, "dispositions": {}}, "bible_chars": 0,
        "required_sources": [], "governance_manifest": []}))
    result = run_review_change(ctx, subject="system", surface="system", reviewers=["deep-session"])
    assert availability == ["gateway", "healthy", "closed"]
    [assignment] = executed
    assert assignment.slot.slot_id == "deep-session" and assignment.slot.route.value == "agent_session"
    assert result["usage"]["reason_code"] == "deep_self_review_error"
    assert result["usage"]["execution_status"] == "infra_failed"
    assert result["report_delivered"] is False
    [seat] = result["rows"]  # the failed seat keeps its reason on the record row itself
    assert seat["status"] == "error" and seat["failure_code"] == "deep_self_review_error"
    assert seat["reported_cause"].strip(), seat
    assert _rows(drive) == []  # a child's task drive is not the owner's canonical log
    [row] = _rows(canonical)
    assert row["type"] == "review_model_error" and row["source"] == "deep_self_review"
    assert row["error"] == ("empty response" if empty else "direct provider failure")
    if not empty:
        assert row["reported_cause"] == "upstream service stopped" and row["failure_code"] == "provider_unavailable"
    assert row["slot_id"] == "deep-session" and row["task_id"] == "child" and row["root_task_id"] == "root"
    assert row["delegated_run_id"] == "run-direct" and row["provider"] == "claudexor"
    assert row["review_wave_id"].startswith("wave-")  # the system review's own round
    assert row["prompt_ref"].get("call_id", "").endswith("_prompt")
    assert len(frames) == 1
    assert frames[0]["type"] == row["type"] and frames[0]["error"] == row["error"]
    assert not (drive / "memory" / "deep_review.md").exists()
