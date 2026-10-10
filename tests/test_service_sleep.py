"""#1539: a WARM sleep woken by the exit of one of the task's own services.

``await_messages(mode="warm", services=[...])`` pins each selected service to
the start it has now — its ``task_id:name`` id is only the lookup key a
stop/start reuses, so the start time and process identity tell a replacement
apart — and the existing park loop wakes on that start's exit with its real
return code (host and local-executor Popen) or ``None`` (Docker: the probe's
own exit status is not the service's). A start the registry no longer holds
(stopped, replaced, lost to a Restart) or an inconclusive probe wakes as
``unknown``, never as success, and no model round runs while parked.

Real subprocesses throughout; serial because they are real processes.
"""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from tests.test_model_sleep import _ctx, _mail, _result

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def _services_reaped(monkeypatch):
    from ouroboros import config as cfg
    from ouroboros.tools import services

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    yield
    monkeypatch.undo()  # the wake-path tripwires must not fire inside the reaper's own status reads
    services.kill_all_services(durable=False)


def _registry(tmp_path, task_id="sleeper", *, executor=False):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    workspace, data = tmp_path / "workspace", tmp_path / "service-data"  # disjoint from every drive
    workspace.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=data)
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=data, workspace_root=workspace,
                      workspace_mode="external", task_id=task_id)
    if executor:
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": "/workspace"}
    registry.set_context(ctx)
    return registry


def _start(registry, name, *, exit_code=0, flag=None):
    """A real service: runs until ``flag`` exists (or exits at once), then exits ``exit_code``."""
    body = (f"import pathlib, sys, time\nflag = {str(flag)!r}\n"
            "while flag != 'None' and not pathlib.Path(flag).exists():\n    time.sleep(0.05)\n"
            f"sys.exit({exit_code})\n")
    return json.loads(registry.execute("start_service", {"name": name, "cmd": [sys.executable, "-c", body]}))


def _arm(ctx, **selectors):
    from ouroboros.owner_wait import checkpoint_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    armed = json.loads(_await_messages(ctx, mode="warm", **selectors))
    assert armed["reason"] == "sleep_armed", armed
    return checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())


def _no_readiness_work(monkeypatch):
    """The wake reader reads execution facts only: no readiness scan, no model round."""
    from ouroboros.tools import services
    from ouroboros import workspace_executor

    monkeypatch.setattr(services, "_refresh_ready", lambda _r: pytest.fail("readiness refreshed by a wake"))
    monkeypatch.setattr(workspace_executor, "_refresh_executor_service_readiness",
                        lambda _r: pytest.fail("readiness refreshed by a wake"))
    monkeypatch.setattr("ouroboros.loop.call_llm_with_retry", lambda *a, **k: pytest.fail("model round while parked"))


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("executor", [False, True], ids=["host", "local_executor"])
def test_the_selected_start_exits_and_the_park_wakes_with_its_real_return_code(tmp_path, monkeypatch,
                                                                               exit_code, executor):
    from ouroboros import model_sleep
    from ouroboros.owner_wait import direct_owner_wait

    _result(tmp_path, "sleeper")
    registry = _registry(tmp_path, executor=executor)
    flag = tmp_path / "go"
    started = _start(registry, "build", exit_code=exit_code, flag=flag)
    assert started["state"] == "running"
    ctx = _ctx(tmp_path)
    checkpoint = _arm(ctx, services=["build"])
    [pin] = checkpoint["sleep"]["services"]
    assert pin["service_id"] == "sleeper:build" and pin["started_at"]
    assert ("backend_pid" in pin) is executor and ("pid" in pin) is not executor
    assert checkpoint["sleep"]["any_mail"] is False
    _no_readiness_work(monkeypatch)
    flag.write_text("go")  # the exit lands after the park was armed: the recheck sees it
    outcome = direct_owner_wait(ctx, checkpoint)
    assert outcome == f"service:build:exited:{exit_code}"
    notice = model_sleep.wake_notice(checkpoint["sleep"], outcome, 1.0)["content"]
    assert f"service build exiting with return code {exit_code}" in notice and "not a verdict" in notice


def test_an_already_exited_start_answers_at_once_and_nothing_parks(tmp_path):
    from ouroboros.tools import services
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _start(_registry(tmp_path), "build", exit_code=3)
    record = services._SERVICES["sleeper:build"]
    deadline = time.time() + 10
    while record.proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    ctx = _ctx(tmp_path)
    ready = json.loads(_await_messages(ctx, mode="warm", services=["build"]))
    assert ready == {"reason": "ready", "woke_by": "service:build:exited:3", "slept": False, "mode": "warm"}
    assert not getattr(ctx, "_model_sleep", None)


def test_a_same_name_restart_is_unknown_for_the_original_selection_never_its_success(tmp_path, monkeypatch):
    from ouroboros.owner_wait import direct_owner_wait
    from ouroboros.tools import services

    _result(tmp_path, "sleeper")
    registry = _registry(tmp_path)
    _start(registry, "build", flag=tmp_path / "never")
    ctx = _ctx(tmp_path)
    checkpoint = _arm(ctx, services=["build"])
    original = dict(checkpoint["sleep"]["services"][0])
    json.loads(registry.execute("stop_service", {"name": "build"}))
    replacement = _start(registry, "build", flag=tmp_path / "never")
    # The lookup key is the same; only the start identity tells them apart.
    assert replacement["service_id"] == original["service_id"] == "sleeper:build"
    assert services.service_execution_facts("sleeper:build")["state"] == "running"
    _no_readiness_work(monkeypatch)
    assert direct_owner_wait(ctx, checkpoint) == "service:build:unknown"


def test_a_stopped_selection_wakes_unknown_and_foreign_or_missing_names_are_refused(tmp_path, monkeypatch):
    from ouroboros.owner_wait import direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _start(_registry(tmp_path, "other-task"), "build", flag=tmp_path / "never")  # another task's "build"
    ctx = _ctx(tmp_path)
    assert "services: build is not a service of this task" in _await_messages(ctx, mode="warm", services=["build"])
    assert "services: ghost is not a service of this task" in _await_messages(ctx, mode="warm", services=["ghost"])
    assert "services must be a list of ids" in _await_messages(ctx, mode="warm", services="build")
    assert not getattr(ctx, "_model_sleep", None)
    registry = _registry(tmp_path)
    _start(registry, "web", flag=tmp_path / "never")
    checkpoint = _arm(ctx, services=["web"])
    registry.execute("stop_service", {"name": "web"})
    _no_readiness_work(monkeypatch)
    assert direct_owner_wait(ctx, checkpoint) == "service:web:unknown"


def test_a_cold_sleep_cannot_select_services_and_the_cold_custody_is_unchanged(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _start(_registry(tmp_path), "build", flag=tmp_path / "never")
    ctx = _ctx(tmp_path)
    refused = _await_messages(ctx, mode="cold", services=["build"])
    assert "services wake only a warm sleep" in refused and not getattr(ctx, "_model_sleep", None)
    # Without the selector the existing cold refusal still names the live service.
    with pytest.raises(ValueError, match="service build"):
        model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=60), "cold")


def test_owner_controls_and_other_selected_sources_still_wake_a_service_sleep(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.owner_wait import direct_owner_wait

    _result(tmp_path, "sleeper")
    _result(tmp_path, "child-1")
    _start(_registry(tmp_path), "build", flag=tmp_path / "never")
    ctx = _ctx(tmp_path)
    checkpoint = _arm(ctx, services=["build"], tasks=["child-1"])
    _result(tmp_path, "child-1", "completed")
    assert direct_owner_wait(ctx, checkpoint) == "task:child-1:completed"
    ctx = _ctx(tmp_path)
    checkpoint = _arm(ctx, services=["build"])
    _mail(tmp_path, "sleeper", "stop and summarise", kind="owner_text")
    assert direct_owner_wait(ctx, checkpoint) == "owner_text"
    # A sleep that selects no service keeps its existing shape.
    assert "services" not in model_sleep.selectors(ctx, tasks=["child-1"])


def test_a_docker_backend_has_no_return_code_and_an_inconclusive_probe_wakes_unknown(tmp_path, monkeypatch):
    """Docker state comes from its liveness probe, whose own exit status is not
    the service's: an exited backend wakes with an unknown code, an inconclusive
    probe of a present record wakes as unknown for the model to judge once."""
    from ouroboros import model_sleep, workspace_executor
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    record = SimpleNamespace(service_id="sleeper:db", name="db", task_id="sleeper", started_at=time.time(),
                             backend_pid="4242", local_proc=None,
                             executor=SimpleNamespace(kind="docker_exec", container_name="svc"))
    monkeypatch.setitem(workspace_executor._SERVICES, "sleeper:db", record)
    states = iter(["running", "running", "exited", "unknown"])
    monkeypatch.setattr(workspace_executor, "_service_state", lambda _record: next(states))
    ctx = _ctx(tmp_path)
    assert json.loads(_await_messages(ctx, mode="warm", services=["db"]))["reason"] == "sleep_armed"
    [pin] = ctx._model_sleep["services"]
    assert pin == {"name": "db", "service_id": "sleeper:db", "started_at": record.started_at, "backend_pid": "4242"}
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "service:db:exited:unknown"
    assert model_sleep.wake_reason(ctx, ctx._model_sleep) == "service:db:unknown"
    notice = model_sleep.wake_notice(ctx._model_sleep, "service:db:exited:unknown", 2.0)["content"]
    assert "a return code this backend does not observe" in notice


def test_a_retained_warm_service_sleep_wakes_unknown_after_restart_through_sleep_wake(tmp_path, monkeypatch):
    """A manual Restart retains the warm sleep as an exact pause; the supervisor's
    sleep-wake pass then reads readiness with a plain namespace and a NEW process's
    empty registries. The selected start is gone: unknown, recorded, held by the
    Restart until the owner's Resume — never a raise and never an endless sleep."""
    from ouroboros import model_sleep, workspace_executor
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from ouroboros.tools import services
    from supervisor import sleep_wake
    from supervisor.events_budget import budget_hold_fact
    from supervisor.restart_retention import pause_retention
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
    from tests.test_batch4_repair_compositions import _running
    from tests.test_restart_retention import _pool_events, _restart_door

    q, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    _start(_registry(tmp_path, "root"), "build", flag=tmp_path / "never")
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, services=["build"]), "warm")
    model_sleep.begin(ctx)
    checkpoint = checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    assert pause_retention(tmp_path, "root", 1)
    _restart_door(tmp_path, monkeypatch, workers)
    task = workers.PENDING[0]
    assert budget_hold_fact(task)["reason"] == "owner_restart_hold"
    row = budget_pause_row(tmp_path, "root")
    assert row["sleep"]["services"][0]["service_id"] == "root:build"
    # The next server generation: neither registry carried the service.
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    assert sleep_wake._readiness(q, task, row) == "service:build:unknown"
    [outcome] = sleep_wake.wake_ready_sleepers(q)
    assert outcome["error"] == "sleep_wake_vetoed" and outcome["veto"] == "owner_restart_hold"
    assert budget_pause_row(tmp_path, "root")["sleep_ready"]["reason"] == "service:build:unknown"
    # An older retained sleep with no services selector keeps working unchanged.
    legacy = {key: value for key, value in row["sleep"].items() if key != "services"}
    assert sleep_wake._readiness(q, task, {**row, "sleep": {**legacy, "wake_at": ""}}) == ""
