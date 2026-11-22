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
import threading
import types

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._tariff_guard import _no_unscripted_tariff_reads  # noqa: F401  (autouse: DeepSeek sends here)
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


OPENING = "please research this properly"  # the owner's words every start here answers; their message lives in room 7


def _tool_ctx(root, sup_ctx, *, chat_id=7, **metadata):
    """A conversation turn whose control events reach the REAL supervisor handler.
    ``chat_id``: the room the turn answers in (the opening message stays in room 7)."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from supervisor.events import _handle_promote_chat_to_task

    ref = build_owner_message_ref(chat_id=7, client_message_id="owner-msg-1", ts="2026-10-06T12:00:00+00:00",
                                  text=OPENING)
    return types.SimpleNamespace(
        current_chat_id=chat_id, drive_root=root, pending_events=[], task_id="owner-turn",
        task_metadata={"origin_message_ref": ref, "origin_message_text": OPENING, **metadata},
        event_queue=types.SimpleNamespace(put_nowait=lambda evt: _handle_promote_chat_to_task(evt, sup_ctx)),
    )


def _promoted(tmp_path, monkeypatch, *, folder=None, chat_id=7, **promote_kwargs):
    """Promote through the tool, the event, the supervisor admission and its receipt.
    A root Ouroboros creates itself takes an explicit effort in Cyber Pro only. ``folder``: an allowed
    folder outside the data drive the start names (else it opts out of any folder); ``chat_id``: its room."""
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _start_task

    if "reasoning_effort" in promote_kwargs:
        monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    _q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    sup = _supervisor_ctx(tmp_path, workers)
    out = _start_task(_tool_ctx(tmp_path, sup, chat_id=chat_id), "Research the market", title="Market research",
                      **({"workspace": "none"} if folder is None else {"workspace_root": str(folder)}), **promote_kwargs)
    assert out.startswith("OK: task"), out
    [task] = workers.PENDING
    return task, load_task_result(tmp_path, task["id"]), workers


_POST_TASK_WAIT_SEC = 60.0  # a bound for a stubbed post-task pass, not a judgment about its speed


def _wait_for_post_task(started):
    """Outside a worker process a root's post-task synthesis runs on a detached thread
    (``agent_task_pipeline``). The round this test drove started it, so the test waits for it
    instead of leaving it to the thread-hygiene grace, which a slow runner outlasts."""
    from tests.conftest import _DETACHED_THREAD_NAME_PREFIXES

    for thread in set(threading.enumerate()) - started:
        if not thread.name.startswith(_DETACHED_THREAD_NAME_PREFIXES):
            thread.join(_POST_TASK_WAIT_SEC)
            assert not thread.is_alive(), f"{thread.name} still runs {_POST_TASK_WAIT_SEC}s after the round"


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
    fitted = []  # the effort the start fit sized the request with (``context_fit._request_shapes``)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages",
                        lambda **kw: fitted.append(kw["reasoning_effort"]) or ([{"role": "user", "content": "go"}], {}))
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
    started = set(threading.enumerate())
    agent._handle_task_scoped({**task, "drive_root": str(tmp_path)})
    _wait_for_post_task(started)
    assert captured, "the real first round never reached the physical executor"
    first = captured[0]
    assert fitted == [str((first.effort or {}).get("requested") or "")], "the start fit sizes the effort round 1 sends"
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


BINDING = ("chat_id", "workspace_root", "workspace_mode", "memory_mode", "drive_root", "budget_drive_root")


def test_a_promoted_receipt_keeps_the_admitted_room_folder_and_drive_before_any_worker_writes(
        tmp_path, tmp_path_factory, monkeypatch):
    """The scheduled receipt is the only durable row of a root no worker started. It names the room, the folder,
    the drive and the authored title the admission itself set, verbatim, never the opening message's room (7) or
    a default; the title is its own field beside ``suggested_name``."""
    from ouroboros.headless import task_state_dir

    folder = tmp_path_factory.mktemp("allowed-folder")  # outside the data drive and the system repo
    task, row, _workers = _promoted(tmp_path, monkeypatch, folder=folder, chat_id=9)
    assert row["status"] == "scheduled" and task["origin_message_ref"]["chat_id"] == 7
    keys = ("title", *BINDING)
    assert {key: row[key] for key in keys} == {key: task[key] for key in keys}
    assert row["title"] == "Market research" and row["suggested_name"] == task["suggested_name"]
    assert (row["chat_id"], row["workspace_root"], row["workspace_mode"], row["memory_mode"]) == (
        9, str(folder.resolve()), "external", "forked")
    assert (row["drive_root"], row["budget_drive_root"]) == (
        str(task_state_dir(tmp_path, task["id"]) / "data"), str(tmp_path))


def test_a_start_without_a_folder_keeps_its_room_and_records_no_folder_or_drive(tmp_path, monkeypatch):
    task, row, _workers = _promoted(tmp_path, monkeypatch, chat_id=9)
    assert row["chat_id"] == task["chat_id"] == 9
    assert {key: row.get(key) for key in BINDING} == {key: task.get(key) for key in BINDING}
    assert not any(row.get(key) for key in BINDING[1:])


def _restarted_before_any_worker_write(tmp_path, tmp_path_factory, monkeypatch, **promote_kwargs):
    """A start into an allowed folder, ended by the REAL owner Restart before its worker wrote anything."""
    from ouroboros.task_results import load_task_result

    folder = tmp_path_factory.mktemp("allowed-folder")
    task, admission_row, workers = _promoted(tmp_path, monkeypatch, folder=folder, chat_id=9, **promote_kwargs)
    assert admission_row["status"] == "scheduled"
    monkeypatch.setattr(workers, "WORKERS", {})
    _pool_events(workers, monkeypatch)
    workers.PENDING.clear()
    workers.RUNNING[task["id"]] = {"task": dict(task), "worker_id": 0, "attempt": 1}
    _restart_door(tmp_path, monkeypatch, workers)
    interrupted = load_task_result(tmp_path, task["id"])
    assert interrupted["status"] == "cancelled" and interrupted["cancel_origin"]["source"] == "owner_restart"
    return task, admission_row, interrupted, folder, workers


@pytest.mark.parametrize("effort", ["xhigh", None], ids=["explicit", "omitted"])
def test_preworker_admission_keeps_effort_and_is_continued_where_it_was_admitted(
        tmp_path, tmp_path_factory, monkeypatch, effort):
    """A real admission row can outlive a worker that never wrote its start.

    The real Restart settles that row with the explicit effort intact; the owner's explicit Continue then admits
    one successor from the receipt alone: the room it was admitted to, the allowed folder, its own fork of the
    drive (never the predecessor's), the owner's exact opening and the effort the start named, or none."""
    from pathlib import Path

    from ouroboros.headless import prepared_drive_mode, task_state_dir
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    task, admission_row, interrupted, folder, workers = _restarted_before_any_worker_write(
        tmp_path, tmp_path_factory, monkeypatch, **({"reasoning_effort": effort} if effort else {}))
    opener = task["origin_message_ref"]
    assert admission_row.get("reasoning_effort") == effort and interrupted.get("reasoning_effort") == effort
    assert admission_row["origin_message_ref"] == opener and "owner_corpus" not in admission_row
    assert interrupted["origin_message_ref"] == opener and interrupted["chat_id"] == 9

    ack = admit_continuation(task["id"], action_nonce="preworker-continue-0001")
    assert ack["ok"] is True, ack
    [successor] = [row for row in workers.PENDING if row["id"] == ack["successor_task_id"]]
    assert (successor["chat_id"], successor["workspace_root"], successor["workspace_mode"]) == (
        9, str(folder.resolve()), "external")
    own = task_state_dir(tmp_path, successor["id"]) / "data"
    assert (successor["memory_mode"], successor["drive_root"], successor["budget_drive_root"]) == (
        "forked", str(own), str(tmp_path))
    assert successor["drive_root"] != task["drive_root"] and prepared_drive_mode(Path(successor["drive_root"]),
                                                                                 successor["id"]) == "forked"
    assert successor.get("reasoning_effort") == effort
    assert f"Owner's original instruction (verbatim):\n{OPENING}" in successor["text"]
    assert successor["metadata"]["origin_message_ref"] == opener and successor["metadata"]["origin_message_text"] == OPENING
    assert [(row["source"], row["content"]) for row in successor["metadata"]["owner_corpus"]] == [
        ("origin_message", OPENING)]
    binding = load_task_result(tmp_path, task["id"])["continued_by"]["binding"]
    assert (binding["chat_id"], binding["workspace_root"], binding["workspace_mode"], binding["memory_mode"]) == (
        9, str(folder.resolve()), "external", "forked")
    stored = load_task_result(tmp_path, successor["id"])
    assert (stored["chat_id"], stored["workspace_root"], stored["drive_root"]) == (
        9, str(folder.resolve()), str(own))
    assert admit_continuation(task["id"], action_nonce="preworker-continue-0001")["successor_task_id"] == successor["id"]
    assert [row["id"] for row in workers.PENDING].count(successor["id"]) == 1


def test_preworker_admission_without_its_opening_still_refuses_continue(tmp_path, tmp_path_factory, monkeypatch):
    """The admitted room and folder never stand in for the owner's opening: a root whose receipt lost it, and
    whose producer recorded no absence, still names the gap and admits nothing."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests._store_shared import seed_result

    task, _admission, interrupted, _folder, workers = _restarted_before_any_worker_write(
        tmp_path, tmp_path_factory, monkeypatch)
    seed_result(tmp_path, task["id"], {key: value for key, value in interrupted.items()
                                       if key not in {"origin_message_ref", "origin_message_text"}})
    outcome = admit_continuation(task["id"], action_nonce="preworker-continue-0002")
    assert outcome["ok"] is False and outcome["error"] == "owner_source_missing"
    assert "original_owner_message_text" in outcome["gaps"]
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, task["id"])


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
    assert "reasoning_effort" in schemas["start_task"]["parameters"]["properties"]
    assert "route_to_project" not in schemas
    child = schemas["schedule_subagent"]["parameters"]["properties"]
    assert "reasoning_effort" not in child and child["effort"]["default"] == "auto"
    assert "reasoning_effort" in LEGACY_SUBAGENT_FIELDS  # a pre-record stored child value stays ignored
    from ouroboros.config import EFFORT_SCALE

    assert child["effort"]["enum"] == ["auto", *EFFORT_SCALE]

    assert schemas["start_task"]["parameters"]["properties"]["reasoning_effort"]["enum"] == \
        list(EFFORT_SCALE)
    assert json.dumps(schemas["start_task"])  # serializable as sent
