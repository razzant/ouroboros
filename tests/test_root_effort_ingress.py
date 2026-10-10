"""#1539: an explicit starting effort for a new ROOT, through its real producers.

Each root birth — a conversation's promote and route, ``POST /api/tasks`` (and
the CLI over it), a schedule template fired by the scheduler (the owner's API
and ``schedule_followup``), and the owner's Continue — carries the optional
``reasoning_effort`` from the producer through the supervisor's admission row
to the task the worker is handed, and that task's first ordinary round
REQUESTS it. Omission leaves the field absent so the task type's configured
effort applies; a supplied blank, unknown or non-string value is refused
before any effect. What the provider applied is a separate fact
(``request_wire_attempt``); nothing here claims it equals the request.
"""

from __future__ import annotations

import json
import queue
import types

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_restart_retention import _pool_events, _restart_door


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


def _pool_ready(monkeypatch, workers):
    """One idle worker slot: admission requires a live pool, as on a ready server."""
    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace(busy_task_id=None, reaping=False)})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")


def _supervisor_ctx(root, workers):
    """The supervisor side the promote handler runs against (its real queue)."""
    from supervisor import queue as q

    return types.SimpleNamespace(
        DRIVE_ROOT=root, WORKERS=workers.WORKERS, RUNNING=workers.RUNNING,
        PENDING=workers.PENDING,
        bridge=types.SimpleNamespace(send_routing_ack=lambda *a, **k: None, broadcast=lambda *a, **k: None),
        enqueue_task=q.enqueue_task, persist_queue_snapshot=lambda **_k: True,
        load_state=lambda: {"owner_chat_id": 7}, append_jsonl=lambda *a, **k: None,
    )


def _tool_ctx(root, sup_ctx, **metadata):
    """A conversation turn whose control events reach the REAL supervisor handler."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.events import _handle_promote_chat_to_task

    words = "please research this properly"
    ref = build_owner_message_ref(chat_id=7, client_message_id="owner-msg-1", ts="2026-10-06T12:00:00+00:00",
                                  text=words)
    return types.SimpleNamespace(
        current_chat_id=7, drive_root=root, pending_events=[], task_id="owner-turn",
        task_metadata={"origin_message_ref": ref, "origin_message_text": words, **metadata},
        event_queue=types.SimpleNamespace(put_nowait=lambda evt: _handle_promote_chat_to_task(evt, sup_ctx)),
    )


def _promoted(tmp_path, monkeypatch, **promote_kwargs):
    """Promote through the tool, the event, the supervisor admission and its receipt.
    A root Ouroboros creates itself takes an explicit effort in Cyber Pro only."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _promote_chat_to_task

    if "reasoning_effort" in promote_kwargs:
        monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    sup = _supervisor_ctx(tmp_path, workers)
    out = _promote_chat_to_task(_tool_ctx(tmp_path, sup), "Research the market", workspace="none",
                                predecessor_task_id="", **promote_kwargs)
    assert out.startswith("OK: task"), out
    [task] = workers.PENDING
    return task, load_task_result(tmp_path, task["id"]), workers


def _first_round(tmp_path, monkeypatch, task, *, wire=False):
    """Hand the ADMITTED task to the worker's real entry and loop.

    The provider is stubbed only at the physical executor seam. The real agent,
    loop, candidate preparation, wire projection and accounting still run, so the
    first request's requested/sent/applied/reported effort facts remain observable.
    """
    from ouroboros import agent as agent_module
    from ouroboros import llm as llm_module
    from ouroboros import llm_attempt
    from ouroboros.llm import LLMClient
    from tests.test_tree_cost_ceiling import _patch_execute_candidate

    monkeypatch.setenv("OUROBOROS_MODEL", "deepseek::deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "low")
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", True, raising=False)
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages",
                        lambda **_kw: ([{"role": "user", "content": "go"}], {}))
    captured = []
    real_execute = llm_attempt._execute_candidate

    class _Response:
        def model_dump(self):
            return {
                "id": "stub-wire-response",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    def execute(request, _send, before_dispatch):
        captured.append(request)
        # Keep the real reservation, manifest, dispatch and settlement; only the
        # final HTTP callable is replaced, so no provider or paid call occurs.
        return real_execute(request, lambda: _Response(), before_dispatch)

    _patch_execute_candidate(monkeypatch, llm_module, execute)
    events = queue.Queue()
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=repo, drive_root=tmp_path), event_queue=events)
    agent._handle_task_scoped({**task, "drive_root": str(tmp_path)})
    assert captured, "the real first round never reached the physical executor"
    first = captured[0]
    if not wire:
        return [str((first.effort or {}).get("requested") or "")]
    usage_events = []
    while not events.empty():
        event = events.get_nowait()
        if event.get("type") == "llm_usage":
            usage_events.append(event)
    assert usage_events, "the real send emitted no usage receipt"
    return first, usage_events[0]["usage"]


def test_promote_carries_an_explicit_effort_to_the_admission_row_and_the_first_round(tmp_path, monkeypatch):
    task, row, _workers = _promoted(tmp_path, monkeypatch, reasoning_effort="XHigh ")
    assert task["reasoning_effort"] == "xhigh"
    # Admission authority before any worker exists: the scheduled row names it.
    assert row["status"] == "scheduled" and row["reasoning_effort"] == "xhigh"
    requested = _first_round(tmp_path, monkeypatch, task)
    assert requested and requested[0] == "xhigh", "round 1 requests the explicit start, not the Task default"


def test_omission_keeps_the_task_default_and_leaves_no_field(tmp_path, monkeypatch):
    task, row, _workers = _promoted(tmp_path, monkeypatch)
    assert "reasoning_effort" not in task and not row.get("reasoning_effort")
    assert _first_round(tmp_path, monkeypatch, task)[0] == "low"  # OUROBOROS_EFFORT_TASK


@pytest.mark.parametrize("bad", ["", "  ", "turbo", 3, ["high"], {"tier": "high"}])
def test_an_invalid_effort_is_refused_before_any_promote_or_route_effect(tmp_path, monkeypatch, bad):
    from ouroboros.tools.control import _promote_chat_to_task, _route_to_project

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    ctx = _tool_ctx(tmp_path, _supervisor_ctx(tmp_path, workers))
    ctx.event_queue = types.SimpleNamespace(put_nowait=lambda evt: pytest.fail(f"emitted {evt}"))
    out = _promote_chat_to_task(ctx, "x", predecessor_task_id="", reasoning_effort=bad)
    assert out.startswith("⚠️ TOOL_ARG_ERROR (promote_chat_to_task): reasoning_effort must be one of")
    routed = _route_to_project(ctx, project_id="racer", message="x", predecessor_task_id="", reasoning_effort=bad)
    assert routed.startswith("⚠️ TOOL_ARG_ERROR (route_to_project): reasoning_effort must be one of")
    assert workers.PENDING == [] and ctx.pending_events == []
    assert not (tmp_path / "task_results").exists() or not list((tmp_path / "task_results").iterdir())


def test_route_to_project_carries_it_and_the_picker_discloses_new_vs_existing_behavior(tmp_path, monkeypatch):
    from ouroboros.projects_registry import create_project
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _route_to_project

    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")  # applied in Cyber Pro only
    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    create_project(tmp_path, "racer", name="Racer")
    ctx = _tool_ctx(tmp_path, _supervisor_ctx(tmp_path, workers))
    out = _route_to_project(ctx, project_id="racer", message="continue the racer", predecessor_task_id="",
                            reasoning_effort="max")
    assert out.startswith("✉️ Routed to project"), out
    [task] = workers.PENDING
    assert task["project_id"] == "racer" and task["reasoning_effort"] == "max"
    assert load_task_result(tmp_path, task["id"])["reasoning_effort"] == "max"
    # An owner turn with no resolvable project gets the owner's picker; a New task
    # uses the request, while an existing-task steer leaves its effort unchanged.
    monkeypatch.setattr("ouroboros.tools.control_routing._emit_and_wait_for_routing",
                        lambda _ctx, evt: ("live", {"status": "needs_manual_target", "options": []}))
    ctx.is_direct_chat = True
    picker = _route_to_project(ctx, project_id="", message="which one?", predecessor_task_id="",
                               reasoning_effort="max")
    assert "A New task picked from it starts on reasoning_effort=max" in picker
    assert "picking an existing task delivers the message there and leaves that task's effort unchanged" in picker


def test_first_round_physical_wire_keeps_requested_submitted_clamped_and_observed_separate(tmp_path, monkeypatch):
    task, _row, _workers = _promoted(tmp_path, monkeypatch, reasoning_effort="xhigh")
    request, usage = _first_round(tmp_path, monkeypatch, task, wire=True)
    effort = request.effort
    wire = usage["request_wire"]
    clamp = usage["reasoning_effort_clamped"]
    # Requested is the root's canonical choice; submitted is the provider-shaped
    # field in the physical candidate; the provider mapping is a separate clamp;
    # the stub response reports no served tier, so the test makes no exact-serving claim.
    assert effort["requested"] == "xhigh"
    assert effort["sent"] == {"reasoning_effort": "high"}
    # The candidate's requested value is the provider-shaped wire tier; the
    # original root request remains separately visible on the physical attempt.
    assert wire["requested_effort"] == "high"
    assert wire["original_requested_effort"] == "xhigh"
    assert wire["applied_effort"] == "high"
    assert clamp["requested"] == "xhigh" and clamp["applied"] == "high"
    assert wire["reported_effort"] is None and wire["reported_effort_source"] is None


def test_api_task_create_validates_records_and_hands_over_the_explicit_effort(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.task_results import load_task_result

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    app = Starlette(routes=[Route("/api/tasks", api_tasks_create, methods=["POST"])])
    app.state.drive_root, app.state.repo_dir = tmp_path, tmp_path / "repo"
    (tmp_path / "repo").mkdir()
    client = TestClient(app)
    for bad in ("", "turbo", None, 5):
        refused = client.post("/api/tasks", json={"description": "x", "reasoning_effort": bad})
        assert refused.status_code == 400, (bad, refused.text)
    shadow = client.post("/api/tasks", json={"description": "x", "metadata": {"reasoning_effort": "high"}})
    assert shadow.status_code == 400 and "top-level field" in shadow.json()["error"]
    assert workers.PENDING == [], "a refused body admits nothing"
    created = client.post("/api/tasks", json={"description": "Run the audit", "reasoning_effort": "high"})
    assert created.status_code == 200, created.text
    task_id = created.json()["task_id"]
    [task] = workers.PENDING
    assert task["id"] == task_id and task["reasoning_effort"] == "high"
    assert load_task_result(tmp_path, task_id)["reasoning_effort"] == "high"
    plain = client.post("/api/tasks", json={"description": "Run another"})
    assert plain.status_code == 200 and "reasoning_effort" not in workers.PENDING[-1]
    assert _first_round(tmp_path, monkeypatch, task)[0] == "high"


def test_cli_run_and_schedule_send_the_flag_and_refuse_an_unknown_tier_before_contact(monkeypatch):
    from ouroboros import cli

    sent = []

    class Client:
        def request(self, method, path, body=None):
            sent.append((method, path, body))
            return {"task_id": "abc123"}

    monkeypatch.setattr(cli, "_client", lambda *_a, **_k: Client())
    monkeypatch.setattr(cli, "_print_json", lambda *_a, **_k: None)
    assert cli.main(["run", "--detach", "--reasoning-effort", "xhigh", "audit it"]) == 0
    assert sent[-1][2]["reasoning_effort"] == "xhigh"
    assert cli.main(["run", "--detach", "audit it"]) == 0
    assert "reasoning_effort" not in sent[-1][2]
    assert cli.main(["schedule", "add", "--name", "n", "--cron", "0 9 * * *", "--reasoning-effort", "high",
                     "daily", "check"]) == 0
    assert sent[-1][1] == "/api/schedules" and sent[-1][2]["task"]["reasoning_effort"] == "high"
    count = len(sent)
    monkeypatch.setattr(cli, "_client", lambda *_a, **_k: pytest.fail("no client before validation"))
    with pytest.raises(cli.CLIError, match="--reasoning-effort"):
        cli._run_command(cli.build_parser().parse_args(["run", "--detach", "--reasoning-effort", "turbo", "x"]))
    with pytest.raises(cli.CLIError, match="--reasoning-effort"):
        cli._schedule_command(cli.build_parser().parse_args(
            ["schedule", "add", "--name", "n", "--cron", "0 9 * * *", "--reasoning-effort", "", "x"]))
    assert len(sent) == count


def _fired(tmp_path, record):
    """Write a schedule through the owners' writer, then build the occurrence's task."""
    from supervisor.queue_schedules import _task_from_schedule
    from supervisor.schedule_lifecycle import upsert_scheduled_task

    stored = upsert_scheduled_task(record, drive_root=tmp_path, actor="owner:test")
    return stored, _task_from_schedule(stored, task_id="fired-1")


def test_a_schedule_template_effort_is_checked_at_write_and_reaches_the_fired_root(tmp_path, monkeypatch):
    from supervisor.queue_schedules import ScheduleRefused, load_schedule_store

    _install_queue(tmp_path, monkeypatch)
    base = {"id": "daily", "name": "daily", "trigger": {"type": "cron", "expr": "0 9 * * *"}}
    stored, fired = _fired(tmp_path, {**base, "task": {"type": "task", "text": "check", "reasoning_effort": " MAX"}})
    assert stored["task"]["reasoning_effort"] == "max" and fired["reasoning_effort"] == "max"
    _stored, plain = _fired(tmp_path, {**base, "id": "plain", "task": {"type": "task", "text": "check"}})
    assert "reasoning_effort" not in plain
    for template in ({"type": "task", "text": "x", "reasoning_effort": "turbo"},
                     {"type": "task", "text": "x", "reasoning_effort": ""},
                     {"type": "task", "text": "x", "metadata": {"reasoning_effort": "high"}}):
        with pytest.raises(ScheduleRefused, match="reasoning_effort"):
            _fired(tmp_path, {**base, "id": "bad", "task": template})
    assert "bad" not in {row["id"] for row in load_schedule_store(tmp_path)["tasks"]}, "refused before effect"
    # A hand-edited table keeps firing: an unknown tier stays the default, never a crash.
    from supervisor.queue_schedules import _task_from_schedule

    assert "reasoning_effort" not in _task_from_schedule(
        {**stored, "task": {**stored["task"], "reasoning_effort": "turbo"}}, task_id="fired-2")


def test_the_owner_schedule_door_refuses_a_bad_tier_with_400_and_stores_a_good_one(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import api_schedules_upsert
    from supervisor.queue_schedules import load_schedule_store

    _install_queue(tmp_path, monkeypatch)
    app = Starlette(routes=[Route("/api/schedules", api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = tmp_path
    client = TestClient(app)
    body = {"id": "daily", "name": "daily", "trigger": {"type": "cron", "expr": "0 9 * * *"}}
    bad = client.post("/api/schedules", json={**body, "task": {"type": "task", "text": "x", "reasoning_effort": "turbo"}})
    assert bad.status_code == 400 and "reasoning_effort must be one of" in bad.json()["error"]
    assert load_schedule_store(tmp_path)["tasks"] == []
    good = client.post("/api/schedules", json={**body, "task": {"type": "task", "text": "x", "reasoning_effort": "high"}})
    assert good.status_code == 200, good.text
    assert load_schedule_store(tmp_path)["tasks"][0]["task"]["reasoning_effort"] == "high"


def test_a_followup_fires_its_explicit_effort_through_the_real_occurrence(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import followup
    from supervisor import queue_schedules

    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")  # applied in Cyber Pro only
    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    monkeypatch.setattr(followup, "_is_delegated_subagent", lambda _ctx: False)
    ctx = types.SimpleNamespace(task_id="origin-root", root_task_id="origin-root", current_chat_id=7,
                                drive_root=tmp_path, budget_drive_root=tmp_path, project_id="",
                                task_metadata={"root_task_id": "origin-root",
                                               "resource_intent": {"kind": "system_repo"}},
                                task_contract={}, is_direct_chat=False, workspace_root=None)
    write_task_result(tmp_path, "origin-root", "running", root_task_id="origin-root", chat_id=7)
    due = "2000-01-01T00:00:00Z"  # already due: the next scheduler pass fires it
    refused = followup._handle_schedule_followup(ctx, relation="independent", run_at=due,
                                                 objective="look again", reasoning_effort="turbo")
    assert "FOLLOWUP_EFFORT_INVALID" in refused and not queue_schedules.load_schedule_store(tmp_path)["tasks"]
    registered = followup._handle_schedule_followup(ctx, relation="independent", run_at=due,
                                                    objective="look again", reasoning_effort="xhigh")
    assert registered.startswith("FOLLOWUP_SCHEDULED"), registered
    [row] = queue_schedules.load_schedule_store(tmp_path)["tasks"]
    assert row["task"]["reasoning_effort"] == "xhigh"
    queue_schedules.check_scheduled_tasks()  # the scheduler's real claim → prepare → admit
    fired = [task for task in workers.PENDING if (task.get("metadata") or {}).get("schedule_id") == row["id"]]
    assert fired and fired[0]["reasoning_effort"] == "xhigh", workers.PENDING
    assert load_task_result(tmp_path, fired[0]["id"])["reasoning_effort"] == "xhigh", "the admission receipt row"
    assert _first_round(tmp_path, monkeypatch, fired[0])[0] == "xhigh"


@pytest.mark.parametrize("effort", ["xhigh", None], ids=["explicit", "omitted"])
def test_preworker_admission_keeps_effort_but_preserves_baseline_continue_refusal(tmp_path, monkeypatch, effort):
    """A real admission row can outlive a worker that never wrote its start.

    Restart settles that row with the explicit effort intact, but the existing
    Continue eligibility still refuses because no owner source was captured.
    The test deliberately does not fabricate an origin or owner corpus.
    """
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    task, admission_row, workers = _promoted(
        tmp_path, monkeypatch, **({"reasoning_effort": effort} if effort else {}))
    assert admission_row["status"] == "scheduled" and admission_row.get("reasoning_effort") == effort
    assert "origin_message_ref" not in admission_row and "owner_corpus" not in admission_row
    monkeypatch.setattr(workers, "WORKERS", {})
    _pool_events(workers, monkeypatch)
    workers.PENDING.clear()
    workers.RUNNING[task["id"]] = {"task": dict(task), "worker_id": 0, "attempt": 1}
    _restart_door(tmp_path, monkeypatch, workers)
    interrupted = load_task_result(tmp_path, task["id"])
    assert interrupted["status"] == "cancelled"
    assert interrupted["cancel_origin"]["source"] == "owner_restart"
    assert interrupted.get("reasoning_effort") == effort
    assert "origin_message_ref" not in interrupted and "owner_corpus" not in interrupted
    outcome = admit_continuation(task["id"], action_nonce="preworker-continue-0001")
    assert outcome["ok"] is False and outcome["error"] == "owner_source_missing"
    assert workers.PENDING == []


def test_continue_keeps_the_explicit_effort_including_a_successor_that_never_reached_a_worker(
        tmp_path, monkeypatch):
    """The predecessor rows are the ones real writers produced: the promotion
    admission plus the worker's own first write, then a Continue successor's
    admission row interrupted before any worker wrote — Continue reads the value
    from the result row, outside the SHA-bound binding (BINDING_VERSION unchanged)."""
    from ouroboros import agent as agent_module
    from ouroboros.owner_continue import BINDING_VERSION
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    task, _row, workers = _promoted(tmp_path, monkeypatch, reasoning_effort="xhigh")
    monkeypatch.setattr(workers, "WORKERS", {})  # the slot is the Restart door's to kill, not ours
    _pool_events(workers, monkeypatch)
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.env = types.SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    agent._persist_running_record(dict(task))  # the worker's first write, unchanged
    workers.PENDING.clear()
    workers.RUNNING[task["id"]] = {"task": dict(task), "worker_id": 0, "attempt": 1}
    _restart_door(tmp_path, monkeypatch, workers)
    first = admit_continuation(task["id"], action_nonce="continue-press-0001")
    assert first["ok"] is True, first
    [successor] = [row for row in workers.PENDING if row["id"] == first["task_id"]]
    assert successor["reasoning_effort"] == "xhigh"
    stored = load_task_result(tmp_path, successor["id"])
    assert stored["reasoning_effort"] == "xhigh"
    binding = stored["continuation_admission"]["binding"]
    assert "reasoning_effort" not in binding and binding["binding_version"] == BINDING_VERSION
    # The successor is interrupted BEFORE any worker write: its admission row is all there is.
    workers.PENDING.clear()
    workers.RUNNING[successor["id"]] = {"task": dict(successor), "worker_id": 0, "attempt": 1}
    _restart_door(tmp_path, monkeypatch, workers)
    second = admit_continuation(successor["id"], action_nonce="continue-press-0002")
    assert second["ok"] is True, second
    [grandchild] = [row for row in workers.PENDING if row["id"] == second["task_id"]]
    assert grandchild["reasoning_effort"] == "xhigh"


def test_continue_of_a_root_started_without_a_choice_inherits_nothing(tmp_path, monkeypatch):
    from ouroboros import agent as agent_module
    from supervisor.continuation_admission import admit_continuation

    task, _row, workers = _promoted(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "WORKERS", {})
    _pool_events(workers, monkeypatch)
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.env = types.SimpleNamespace(drive_root=tmp_path, budget_drive_root=tmp_path)
    # A derived default or a switch_model is never stored on a root's row (dispatch
    # stamps children alone), so nothing becomes the successor's start.
    agent._persist_running_record(dict(task))
    workers.PENDING.clear()
    workers.RUNNING[task["id"]] = {"task": dict(task), "worker_id": 0, "attempt": 1}
    _restart_door(tmp_path, monkeypatch, workers)
    admitted = admit_continuation(task["id"], action_nonce="continue-press-0003")
    assert admitted["ok"] is True, admitted
    [successor] = [row for row in workers.PENDING if row["id"] == admitted["task_id"]]
    assert "reasoning_effort" not in successor


def test_a_control_seeding_an_absent_row_keeps_the_queued_explicit_effort(tmp_path, monkeypatch):
    from supervisor.queue import ensure_control_task_result

    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    workers.PENDING.append({"id": "queued-1", "type": "task", "chat_id": 7, "root_task_id": "queued-1",
                            "delegation_role": "root", "reasoning_effort": "high", "_attempt": 1})
    workers.PENDING.append({"id": "queued-2", "type": "task", "chat_id": 7, "root_task_id": "queued-2",
                            "delegation_role": "root", "_attempt": 1})
    assert ensure_control_task_result("queued-1")["reasoning_effort"] == "high"
    assert not ensure_control_task_result("queued-2").get("reasoning_effort")


def test_configured_children_keep_their_profile_and_ask_for_effort_through_their_own_argument():
    from ouroboros.subagents import LEGACY_SUBAGENT_FIELDS
    from ouroboros.tools.control import get_tools

    schemas = {entry.name: entry.schema for entry in get_tools()}
    assert "reasoning_effort" in schemas["promote_chat_to_task"]["parameters"]["properties"]
    assert "reasoning_effort" in schemas["route_to_project"]["parameters"]["properties"]
    child = schemas["schedule_subagent"]["parameters"]["properties"]
    assert "reasoning_effort" not in child and child["effort"]["default"] == "auto"
    assert "reasoning_effort" in LEGACY_SUBAGENT_FIELDS  # a pre-record stored child value stays ignored
    from ouroboros.config import EFFORT_SCALE

    assert child["effort"]["enum"] == ["auto", *EFFORT_SCALE]

    assert schemas["promote_chat_to_task"]["parameters"]["properties"]["reasoning_effort"]["enum"] == \
        list(EFFORT_SCALE)
    assert json.dumps(schemas["promote_chat_to_task"])  # serializable as sent
