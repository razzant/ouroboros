"""Saved settings are compared with actual component startup/health facts."""

from ouroboros.local_model import RUNTIME_PROBE
import asyncio
import io
import json
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from starlette.requests import Request

from ouroboros import config, local_model, server_process
from ouroboros.gateway import models, settings
from ouroboros.server_entrypoint import bound_service_socket


@pytest.fixture
def applied(monkeypatch):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    config.initialize_runtime_mode_baseline("advanced")
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    saved = dict(config.SETTINGS_DEFAULTS)
    server_process.record_applied_restart_settings({key: saved[key] for key in (
        "OUROBOROS_SERVER_HOST", "OUROBOROS_HOST_SERVICE_PORT",
        "OUROBOROS_SKILLS_REPO_PATH", "OUROBOROS_UI_DIR",  # the last: server_web.ServedUI.apply
    )}, server_host_source="settings")
    return saved, manager


def test_reader_imported_under_the_applied_mode_sees_the_next_boot(applied, monkeypatch):
    from ouroboros.config import get_runtime_mode

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    assert get_runtime_mode() == "advanced"
    config.reset_runtime_mode_baseline_for_tests()
    config.initialize_runtime_mode_baseline()
    assert get_runtime_mode() == "pro"


def test_server_pending_truth_survives_reload_revert_and_environment_save(applied, monkeypatch):
    saved, _manager = applied
    for key, value in {
        "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_SERVER_HOST": "0.0.0.0", "OUROBOROS_HOST_SERVICE_PORT": 9123,
        "OUROBOROS_SKILLS_REPO_PATH": "/example/skills", "OUROBOROS_UI_DIR": "none",
    }.items():
        changed = {**saved, key: value}
        monkeypatch.setenv(key, str(value))  # a Save is not application
        state = settings._build_restart_state(changed)
        assert state["restart_keys"] == [key]
        assert settings._build_restart_state({**changed, "TOTAL_BUDGET": 300}) == state
        assert settings._build_restart_state(saved)["restart_required"] is False


def test_missing_component_facts_stay_unknown(applied, monkeypatch):
    saved, _manager = applied
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    state = settings._build_restart_state(saved)
    assert len(state["unknown_keys"]) == 4  # the owner's worker cap is read live on every pass: never a restart key
    assert "not reported" in state["summary"]
    changed = {**saved, "OUROBOROS_MAX_WORKERS": 19}
    assert "OUROBOROS_MAX_WORKERS" not in settings._build_restart_state(changed)["restart_keys"]


def test_socket_owners_publish_only_successfully_bound_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(server_process, "_applied_restart_settings", {})
    with bound_service_socket(tmp_path, "host_service", "127.0.0.1", 0) as sock:
        port = sock.getsockname()[1]
        assert server_process.applied_restart_settings()["OUROBOROS_HOST_SERVICE_PORT"] == port
        # Exercise failed-bind publication without OS-specific port reuse rules.
        with patch("socket.socket.bind", side_effect=OSError("fixture bind refused")):
            with pytest.raises(OSError, match="fixture bind refused"):
                with bound_service_socket(tmp_path, "main", "127.0.0.1", port):
                    pass
        assert "OUROBOROS_SERVER_HOST" not in server_process.applied_restart_settings()


def test_every_local_setting_is_compared_only_after_confirmed_health(applied):
    saved, manager = applied
    saved.update(LOCAL_MODEL_SOURCE="fixture/model", LOCAL_MODEL_FILENAME="fixture.gguf")
    manager._launch_settings = local_model.local_model_settings(saved)
    manager._status = "loading"
    assert manager.settings_application(saved)["pending_keys"] == []
    assert "not applied" in manager.settings_application(saved)["summary"]
    manager._status = "ready"
    manager._applied_settings = dict(manager._launch_settings)
    for key, value in {
        "LOCAL_MODEL_SOURCE": "another/model", "LOCAL_MODEL_FILENAME": "other.gguf",
        "LOCAL_MODEL_PORT": 9345, "LOCAL_MODEL_N_GPU_LAYERS": 17,
        "LOCAL_MODEL_CONTEXT_LENGTH": 32768, "LOCAL_MODEL_CHAT_FORMAT": "other-format",
    }.items():
        state = settings._build_restart_state({**saved, key: value})
        assert state["restart_required"] is False
        assert state["local_model"]["pending_keys"] == [key]
    assert manager.settings_application({**saved, "LOCAL_MODEL_CONTEXT_LENGTH": 0})["pending_keys"] == []
    manager._status = "error"
    assert "not applied" in manager.settings_application(saved)["summary"]
    assert manager.settings_application(saved)["status"] == "error"


def _request(body):
    async def receive():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
    return Request({"type": "http", "method": "POST", "path": "/api/local-model/start",
                    "headers": [], "app": SimpleNamespace(state=SimpleNamespace())}, receive)


@pytest.mark.parametrize("force_exits", [False, True])
def test_model_stop_retains_custody_until_exit_is_observed(applied, monkeypatch, force_exits):
    saved, manager = applied
    child = SimpleNamespace(pid=712, code=None)
    child.poll = lambda: child.code
    binding = {"pid": child.pid}
    manager._proc, manager._service_binding = child, binding
    manager._status, manager._runtime_status = "ready", "ok"
    manager._applied_settings = local_model.local_model_settings(saved)
    entered, release = threading.Event(), threading.Event()
    waits, signals, cleared, panic_targets = [], [], [], []

    def wait(*, timeout):
        waits.append(timeout)
        if timeout == 10:
            entered.set()
            assert release.wait(5)
        elif force_exits:
            child.code = -9
            return child.code
        raise subprocess.TimeoutExpired("fixture", timeout)

    child.wait = wait
    monkeypatch.setattr(local_model, "terminate_process_tree", lambda proc: signals.append(("term", proc)))
    monkeypatch.setattr(local_model, "kill_process_tree", lambda proc: signals.append(("kill", proc)))
    monkeypatch.setattr(server_process, "clear_service_binding", lambda _, service, row: cleared.append((service, row)))
    monkeypatch.setattr(manager, "_probe_runtime", Mock(side_effect=AssertionError("unexpected runtime probe")))
    monkeypatch.setattr(manager, "_run_install", Mock(side_effect=AssertionError("unexpected installation")))

    def panic_request(proc):
        panic_targets.append(proc)
        return {"pid": proc.pid, "requested": True}

    monkeypatch.setattr("ouroboros.platform_layer.request_process_tree_kill", panic_request)

    async def assert_start_and_install_refused():
        start = await models.api_local_model_start(_request({"source": "fixture.gguf"}))
        install = await models.api_local_model_install_runtime(_request({}))
        assert start.status_code == install.status_code == 409

    async def exercise():
        stopping = asyncio.create_task(models.api_local_model_stop(_request({})))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            status = await models.api_local_model_status(_request({}))
            assert json.loads(status.body)["status"] == "stopping"
            assert manager._proc is child and manager._service_binding is binding and not cleared
            assert not manager.is_running
            await assert_start_and_install_refused()
            duplicate = await models.api_local_model_stop(_request({}))
            assert duplicate.status_code == 409
            assert signals == [("term", child)]
        finally:
            release.set()
            response = await stopping

        assert waits == [10, 5] and signals == [("term", child), ("kill", child)]
        if not force_exits:
            assert response.status_code == 409 and json.loads(response.body)["code"] == "local_model_stop_unconfirmed"
            status = manager.status_dict()
            assert status["status"] == "stopping" and "not confirmed" in status["error"]
            assert manager._proc is child and manager._service_binding is binding and not cleared
            await assert_start_and_install_refused()
            with patch.object(manager, "_lock") as lock:
                lock.__enter__.side_effect = AssertionError("Panic must not take the manager lock")
                assert manager.panic_stop(request_only=True) == [{"pid": child.pid, "requested": True}]
            assert panic_targets == [child] and manager._proc is child
            child.code = -9
            response = await models.api_local_model_stop(_request({}))
            assert waits == [10, 5] and signals == [("term", child), ("kill", child)]

        assert response.status_code == 200 and json.loads(response.body) == {"status": "stopped"}
        assert manager._proc is None and manager._service_binding is None and manager.get_status() == "offline"
        assert cleared == [("local_model", binding)]

    asyncio.run(exercise())


@pytest.mark.parametrize("interrupt", ["stop", "panic", "exit"])
def test_late_health_cannot_publish_the_same_child_after_interruption(applied, monkeypatch, interrupt):
    _, manager = applied
    child = SimpleNamespace(pid=713, code=None, wait=Mock(side_effect=subprocess.TimeoutExpired("fixture", 5)))
    child.poll = lambda: child.code
    manager._proc, manager._status = child, "loading"
    panic_signal = Mock(return_value={"requested": True})

    def health(*, port, launch_id):
        assert (port, launch_id) == (manager.port, "fixture-launch")
        # The request is outside the manager lock. Its matching answer can arrive too late.
        if interrupt == "stop":
            assert not manager.stop_server()
        elif interrupt == "panic":
            with patch.object(manager, "_lock") as lock:
                lock.__enter__.side_effect = AssertionError("Panic must not take the manager lock")
                assert manager.panic_stop(request_only=True) == [{"requested": True}]
        else:
            child.code = 17
        return {"ok": True, "context_length": 4096, "model_name": "fixture"}

    monkeypatch.setattr(manager, "health_check", health)
    monkeypatch.setattr(local_model, "terminate_process_tree", Mock())
    monkeypatch.setattr(local_model, "kill_process_tree", Mock())
    monkeypatch.setattr("ouroboros.platform_layer.request_process_tree_kill", panic_signal)
    publish = Mock()
    monkeypatch.setattr(server_process, "record_service_binding", publish)
    manager._wait_for_healthy(proc=child, port=manager.port, launch_id="fixture-launch", timeout=1)
    if interrupt == "stop":
        assert manager.get_status() == "stopping" and manager._proc is child
    elif interrupt == "panic":
        panic_signal.assert_called_once_with(child)
        assert manager.get_status() == "loading" and manager._proc is child
    else:
        assert manager._proc is None and manager.status_dict()["error_stage"] == "startup_exit"
    assert manager._applied_settings == {} and manager._service_binding is None
    publish.assert_not_called()


@pytest.mark.parametrize(("http_status", "stamp"), [(200, None), (200, "earlier-launch"), (307, "current-launch")])
def test_a_model_list_without_current_launch_evidence_never_applies_settings(
    applied, monkeypatch, http_status, stamp,
):
    import requests
    from ouroboros.local_model_server import LOCAL_LAUNCH_HEADER

    saved, manager = applied
    saved["LOCAL_MODEL_SOURCE"] = "fixture/model"
    child = SimpleNamespace(pid=714, code=None)
    child.poll = lambda: child.code
    manager._proc, manager._status, manager._runtime_status = child, "loading", "ok"
    manager._launch_settings = local_model.local_model_settings(saved)
    response = Mock(status_code=http_status, headers={LOCAL_LAUNCH_HEADER: stamp} if stamp else {})
    response.json.return_value = {"data": [{"id": "fixture", "context_window": 4096}]}
    exchange = MagicMock()
    session = exchange.__enter__.return_value
    session.get.return_value = response
    monkeypatch.setattr(requests, "Session", lambda: exchange)
    publish = Mock()
    monkeypatch.setattr(server_process, "record_service_binding", publish)

    def child_exits_after_reply(_delay):
        assert manager.get_status() == "loading"
        assert manager._service_binding is None and manager._applied_settings == {}
        child.code = 17
        manager._stderr_buf.extend(b"Fixture could not bind its port")

    monkeypatch.setattr(local_model, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=child_exits_after_reply))
    manager._wait_for_healthy(proc=child, port=manager.port, launch_id="current-launch", timeout=1)
    session.get.assert_called_once_with(f"http://127.0.0.1:{manager.port}/v1/models", timeout=5, allow_redirects=False)
    publish.assert_not_called()
    status = json.loads(asyncio.run(models.api_local_model_status(_request({}))).body)
    assert status["status"] == "error" and status["error_stage"] == "startup_exit"
    assert status["runtime_status"] == "ok" and status["error"].endswith("(code 17): Fixture could not bind its port")
    assert "not applied" in manager.settings_application(saved)["summary"]


@pytest.mark.parametrize("poll_first", [True, False])
def test_status_records_startup_exit_and_its_stderr_in_either_poll_order(applied, poll_first, tmp_path):
    _, manager = applied
    proc = SimpleNamespace(poll=lambda: 17)
    manager._proc, manager._status, manager._runtime_status = proc, "loading", "ok"
    if poll_first:
        assert manager.get_status() == "error"
    manager._wait_for_healthy(proc=proc, port=manager.port, launch_id="fixture-launch")
    # The pipe may drain after either observer sees the process exit.
    manager._stderr_buf.extend(b"Fixture header cannot be loaded")
    status = json.loads(asyncio.run(models.api_local_model_status(_request({}))).body)
    assert status["status"] == "error" and status["error_stage"] == "startup_exit"
    assert status["error"] == "Server process exited during startup (code 17): Fixture header cannot be loaded"
    next_model = tmp_path / "next.gguf"
    next_model.write_bytes(b"fixture")
    assert manager.download_model(str(next_model)) == str(next_model)
    assert "error_stage" not in manager.status_dict()


def test_health_timeout_and_exit_after_ready_have_no_startup_exit_stage(applied):
    _, manager = applied
    manager._proc = SimpleNamespace(poll=lambda: None)
    manager._status = "loading"
    manager._wait_for_healthy(proc=manager._proc, port=manager.port, launch_id="fixture-launch", timeout=0)
    status = manager.status_dict()
    assert "failed to become healthy" in status["error"] and "error_stage" not in status
    manager._status, manager._proc = "ready", SimpleNamespace(poll=lambda: 17)
    status = manager.status_dict()
    assert status["error"] == "Server exited with code 17" and "error_stage" not in status
    manager.stop_server()
    assert manager.get_status() == "offline" and "error_stage" not in manager.status_dict()


def test_start_while_a_child_is_loading_keeps_its_exit_and_late_stderr(applied, monkeypatch):
    from ouroboros.local_model_autostart import auto_start_local_model

    _, manager = applied
    child = SimpleNamespace(code=None)
    child.poll = lambda: child.code
    manager._proc, manager._status, manager._runtime_status = child, "loading", "ok"
    tail = manager._stderr_buf
    monkeypatch.setattr(manager, "_probe_runtime", lambda: pytest.fail("overlapping Start began preparation"))
    assert asyncio.run(models.api_local_model_start(_request({"source": "another/model"}))).status_code == 409
    auto_start_local_model({"LOCAL_MODEL_SOURCE": "another/model"})
    assert manager._proc is child and manager.get_status() == "loading"
    child.code = 17
    manager._wait_for_healthy(proc=child, port=manager.port, launch_id="fixture-launch")
    tail.extend(b"first child stderr")
    result = manager.status_dict()
    assert result["error_stage"] == "startup_exit" and result["error"].endswith("(code 17): first child stderr")
    monkeypatch.setattr(manager, "_probe_runtime", lambda: subprocess.CompletedProcess([], 1))
    assert asyncio.run(models.api_local_model_start(_request({"source": "another/model"}))).status_code == 412
    assert "error_stage" not in manager.status_dict()


@pytest.mark.parametrize("late", ["progress", "return", "failure"])
def test_overlapping_start_during_download_is_refused_and_stop_prevents_a_late_spawn(applied, monkeypatch, tmp_path, late):
    _, manager = applied
    second = tmp_path / "available.gguf"
    second.touch()
    downloading, release_download = threading.Event(), threading.Event()
    launching, release_launch = threading.Event(), threading.Event()
    shards, probes = [], []

    def download(filename, tqdm_class, **kwargs):
        shards.append(filename)
        with tqdm_class(total=8, file=io.StringIO(), disable=False) as bar:
            bar.update(2)
            downloading.set()
            assert release_download.wait(10)
            if late == "progress":
                bar.update(4)
            elif late == "failure":
                raise OSError("cancelled transfer failed late")
        return "first-shard.gguf"

    def probe():
        probes.append(1)
        if len(probes) == 3:  # The second model has its reservation and is verifying its local file.
            launching.set()
            assert release_launch.wait(10)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(manager, "_probe_runtime", probe)
    child = SimpleNamespace(pid=712, stderr=None, poll=lambda: None)
    spawn = Mock(return_value=child)
    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": spawn}))
    monkeypatch.setattr("ouroboros.process_custody.record_process", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_wait_for_healthy", lambda **kwargs: None)

    async def exercise():
        def start(source, filename=""):
            return models.api_local_model_start(_request({"source": source, "filename": filename}))

        first = asyncio.create_task(start("fixture/repo", "quant/a-01-of-02.gguf"))
        newest = None
        try:
            assert await asyncio.to_thread(downloading.wait, 10)
            assert manager.status_dict()["download_progress"] == 0.125
            assert (await start(str(second))).status_code == 409
            assert (await models.api_local_model_install_runtime(_request({}))).status_code == 409
            assert (await models.api_local_model_stop(_request({}))).status_code == 200
            assert manager.get_status() == "offline"

            newest = asyncio.create_task(start(str(second)))
            assert await asyncio.to_thread(launching.wait, 10)
            release_download.set()
            cancelled = await first
            assert cancelled.status_code == 409 and "cancelled by Stop" in json.loads(cancelled.body)["error"]
            state = manager.status_dict()
            assert state["status"] == "loading" and state["error"] is None and state["download_progress"] == 0
            assert (await start("third/model")).status_code == 409
            release_launch.set()
            assert (await newest).status_code == 200
            assert manager.status_dict()["model_path"] == str(second)
            assert shards == ["a-01-of-02.gguf"]
            assert spawn.call_count == 1 and str(second) in spawn.call_args.args[0]
        finally:
            release_download.set()
            release_launch.set()
            await asyncio.gather(first, *([newest] if newest else []))

    asyncio.run(exercise())


@pytest.mark.parametrize("held_probe", [1, 2], ids=["before-download", "before-spawn"])
def test_stopped_import_cannot_replace_the_next_starts_runtime_evidence(applied, tmp_path, monkeypatch, held_probe):
    _, manager = applied
    model = tmp_path / "cached.gguf"
    model.touch()
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def probe():
        attempts.append(1)
        attempt = len(attempts)
        if attempt == held_probe:
            entered.set()
            assert release.wait(10)
        return subprocess.CompletedProcess([], int(attempt > held_probe))

    monkeypatch.setattr(manager, "_probe_runtime", probe)
    spawn = Mock(side_effect=AssertionError("a cancelled Start reached Popen"))
    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": spawn}))

    async def exercise():
        def start():
            return models.api_local_model_start(_request({"source": str(model)}))

        first = asyncio.create_task(start())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert (await models.api_local_model_stop(_request({}))).status_code == 200
            assert (await start()).status_code == 412
        finally:
            release.set()
            cancelled = await first
        assert cancelled.status_code == 409
        status = manager.status_dict()
        assert status["status"] == "error" and status["runtime_status"] == "missing"
        spawn.assert_not_called()

    asyncio.run(exercise())


@pytest.mark.parametrize("stop_at", ["spawn", "running", "verification"])
def test_runtime_install_keeps_ownership_until_its_cancelled_worker_finishes(applied, monkeypatch, stop_at):
    _, manager = applied
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    children, stopped = [], []

    def spawn(*args, **kwargs):
        child = Mock(stdout=None, returncode=0, pid=811)
        child.poll.return_value = None
        children.append(child)
        if stop_at == "spawn":
            entered.set()
            assert release.wait(10)
        if stop_at == "running":
            def wait(timeout=None):
                if timeout is None:  # Hold the install worker while Stop requests termination.
                    entered.set()
                    assert release.wait(10)
                return 0
            child.wait.side_effect = wait
        return child

    def probe():
        if stop_at == "verification":
            entered.set()
            assert release.wait(10)
        return subprocess.CompletedProcess([], 0)

    run = manager._run_install

    def worker():
        try:
            run()
        finally:
            done.set()

    monkeypatch.setattr(manager, "_run_install", worker)
    monkeypatch.setattr(manager, "_probe_runtime", probe)
    monkeypatch.setattr(local_model, "_get_install_command", lambda: ["fixture-installer"])
    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": spawn}))
    monkeypatch.setattr(local_model, "terminate_process_tree", lambda p: stopped.append(p.pid))

    async def install():
        return json.loads((await models.api_local_model_install_runtime(_request({}))).body)["status"]

    async def exercise():
        assert await install() == "installing"
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            for stop in (False, True):
                if stop:
                    assert (await models.api_local_model_stop(_request({}))).status_code == 200
                status = json.loads((await models.api_local_model_status(_request({}))).body)
                assert status["runtime_status"] == "installing"
                assert (await models.api_local_model_start(_request({"source": "another/model"}))).status_code == 409
                assert await install() == "already_installing"
                assert len(children) == 1
        finally:
            release.set()
            assert await asyncio.to_thread(done.wait, 10)
        assert stopped == ([] if stop_at == "verification" else [811])
        assert manager.status_dict()["runtime_status"] == "unknown"
        done.clear()
        assert await install() == "installing"
        assert await asyncio.to_thread(done.wait, 10)
        assert len(children) == 2 and manager.status_dict()["runtime_status"] == "install_ok"

    asyncio.run(exercise())


def test_status_polls_share_the_import_and_start_can_retry_a_timeout(applied, monkeypatch):
    _, manager = applied
    entered, release, joined = threading.Event(), threading.Event(), threading.Event()
    attempts = []

    class ObservedProbe(local_model.Future):
        def result(self, timeout=None):
            if not self.done():
                joined.set()
            return super().result(timeout)

    def probe():
        attempts.append(1)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(10)
            raise subprocess.TimeoutExpired("fixture import", 15)
        return subprocess.CompletedProcess([], 1)

    monkeypatch.setattr(local_model, "Future", ObservedProbe)
    monkeypatch.setattr(manager, "_probe_runtime", probe)

    async def poll():
        return json.loads((await models.api_local_model_status(_request({}))).body)

    async def exercise():
        first = asyncio.create_task(poll())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            second = asyncio.create_task(poll())
            assert await asyncio.to_thread(joined.wait, 10)
        finally:
            release.set()
        for result in [await first, await second, await poll()]:
            assert result["runtime_status"] == "unknown" and "timed out" in result["error"]
            assert "hint" not in result
        assert len(attempts) == 1
        retry = await models.api_local_model_start(_request({"source": "fixture/model"}))
        assert retry.status_code == 412 and json.loads(retry.body)["error"] == "runtime_missing"
        assert (await poll())["runtime_status"] == "missing"
        assert len(attempts) == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("after_download", [False, True])
def test_an_unavailable_runtime_probe_offers_no_install(applied, tmp_path, after_download):
    _, manager = applied
    model = tmp_path / "probe.gguf"
    model.touch()
    probes = [subprocess.CompletedProcess([], 0)] if after_download else []
    with patch.object(manager, "_probe_runtime", side_effect=[*probes, OSError("cannot run probe")]):
        response = asyncio.run(models.api_local_model_start(_request({"source": str(model)})))
    body = json.loads(response.body)
    assert response.status_code == 500 and "cannot run probe" in body["error"] and "hint" not in body
    status = manager.status_dict()
    assert status["status"] == "error" and status["runtime_status"] == "unknown"


def test_a_finished_runtime_install_keeps_the_startup_exit_evidence_until_a_new_start(applied, tmp_path):
    """The update offer retires with the runtime state, not with the error: ``install_ok`` (a successful update the
    page may have missed by being away) keeps the last start's qualified exit as evidence, so no second Update is
    offered without a new start, and only that start's own lifecycle clears it."""
    _, manager = applied
    manager._status, manager._runtime_status = "startup_exit", "install_ok"
    manager._error = "Server process exited during startup (code 17)"
    manager._stderr_buf.extend(b"Fixture header cannot be loaded")
    status = manager.status_dict()
    assert status["runtime_status"] == "install_ok"
    assert status["status"] == "error" and status["error_stage"] == "startup_exit"
    assert "Fixture header cannot be loaded" in status["error"]
    next_model = tmp_path / "next.gguf"
    next_model.write_bytes(b"fixture")
    assert manager.download_model(str(next_model)) == str(next_model)
    after = manager.status_dict()
    assert after["status"] == "offline" and "error_stage" not in after and after["error"] is None


@pytest.fixture
def lifecycle(applied, tmp_path, monkeypatch):
    """The real manager over benign doubles: an installer that succeeds (``hold`` keeps it running) and a model child
    that stays up. Nothing is downloaded, signalled or started on the host."""
    _, manager = applied
    model = tmp_path / "deferred.gguf"
    model.touch()
    installed, hold = threading.Event(), threading.Event()
    hold.set()
    run = manager._run_install

    def settle():
        try:
            run()
        finally:
            installed.set()

    installer = Mock(stdout=None, returncode=0, pid=811, wait=lambda timeout=None: hold.wait(10))
    child = SimpleNamespace(pid=712, stderr=None, poll=lambda: None)
    spawn = Mock(side_effect=lambda command, **kwargs: installer if command == ["fixture-installer"] else child)
    monkeypatch.setattr(manager, "_run_install", settle)
    monkeypatch.setattr(manager, "_probe_runtime", lambda: subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(manager, "_wait_for_healthy", lambda **kwargs: None)
    monkeypatch.setattr(local_model, "_get_install_command", lambda: ["fixture-installer"])
    monkeypatch.setattr(local_model, "subprocess", SimpleNamespace(**{**vars(subprocess), "Popen": spawn}))
    monkeypatch.setattr("ouroboros.process_custody.record_process", lambda *args, **kwargs: None)
    return SimpleNamespace(manager=manager, model=model, spawn=spawn, installed=installed, hold=hold)


@pytest.mark.parametrize("then", ["deferred", "explicit"])
def test_a_stop_ends_the_start_an_install_offered_for_good_but_not_a_later_one(lifecycle, then):
    """The reply that admits an install carries its ``install_id``, the one receipt its caller's deferred Start may
    present. A Stop revokes it and no later install makes it current again; the newest id, or none, still starts."""
    manager, spawn, installed = lifecycle.manager, lifecycle.spawn, lifecycle.installed

    async def install():
        reply = json.loads((await models.api_local_model_install_runtime(_request({}))).body)
        assert await asyncio.to_thread(installed.wait, 10)
        installed.clear()
        return reply

    def start(**receipt):
        return models.api_local_model_start(_request({"source": str(lifecycle.model), **receipt}))

    async def exercise():
        first = await install()
        assert set(first) == {"status", "install_id"} and first["status"] == "installing"
        assert manager.status_dict()["runtime_status"] == "install_ok"
        assert (await models.api_local_model_stop(_request({}))).status_code == 200
        assert (await start(install_id=first["install_id"])).status_code == 409  # the Stop itself revoked it
        second = await install()
        assert second["install_id"] != first["install_id"]
        late = await start(install_id=first["install_id"])  # a later install does not make it current again
        assert late.status_code == 409 and "cancelled" in json.loads(late.body)["error"]
        assert spawn.call_count == 2 and manager.get_status() == "offline"
        assert manager.status_dict()["runtime_status"] == "install_ok"  # the installs' own result stays
        current = await start(**({"install_id": second["install_id"]} if then == "deferred" else {}))
        assert current.status_code == 200 and spawn.call_count == 3 and manager.get_status() == "loading"

    asyncio.run(exercise())


def test_a_joined_install_gets_no_receipt_and_the_admitting_callers_start_follows_it(lifecycle):
    """Joining a running install admits nothing, so its reply carries no ``install_id``; the admitting caller's own
    deferred Start is refused while pip holds the interpreter and goes through once the install has finished."""
    manager, installed, hold = lifecycle.manager, lifecycle.installed, lifecycle.hold
    hold.clear()  # the installer keeps the interpreter until released

    def start(install_id):
        return models.api_local_model_start(_request({"source": str(lifecycle.model), "install_id": install_id}))

    async def post():
        return json.loads((await models.api_local_model_install_runtime(_request({}))).body)

    async def exercise():
        try:
            admitted, joined = await post(), await post()
            assert set(admitted) == {"status", "install_id"} and joined == {"status": "already_installing"}
            during = await start(admitted["install_id"])
            assert during.status_code == 409 and "being installed" in json.loads(during.body)["error"]
        finally:
            hold.set()
        assert await asyncio.to_thread(installed.wait, 10)
        assert manager.status_dict()["runtime_status"] == "install_ok"
        after = await start(admitted["install_id"])
        assert after.status_code == 200 and lifecycle.spawn.call_count == 2

    asyncio.run(exercise())


@pytest.mark.parametrize("then", ["deferred", "explicit"])
def test_a_stop_after_the_boot_decision_cancels_the_staged_autostart(lifecycle, monkeypatch, then):
    """The lifespan decides on the local autostart before its boot coordinator stages it, so the receipt is minted at
    the decision: a Stop in between refuses the staged Start, another manager never honours the receipt, and a live
    receipt or an explicit Start still goes through."""
    from ouroboros.local_model_autostart import auto_start_local_model
    from ouroboros.server_runtime import local_model_boot_decision

    manager, spawn = lifecycle.manager, lifecycle.spawn
    settings = {"USE_LOCAL_MAIN": True, "LOCAL_MODEL_SOURCE": str(lifecycle.model)}
    assert local_model_boot_decision({}) == (False, None)
    needed, stale = local_model_boot_decision(settings)
    assert needed and stale
    assert asyncio.run(models.api_local_model_stop(_request({}))).status_code == 200  # before the staged step runs
    auto_start_local_model(settings, grant=stale)
    assert spawn.call_count == 0 and manager.get_status() == "offline"
    _, live = local_model_boot_decision(settings)
    assert live != stale
    with monkeypatch.context() as replaced:
        other = local_model.LocalModelManager()
        replaced.setattr(local_model, "_manager", other)
        probe = Mock(side_effect=AssertionError("a receipt crossed to another manager"))
        replaced.setattr(other, "_probe_runtime", probe)
        auto_start_local_model(settings, grant=live)
        assert other.get_status() == "offline" and spawn.call_count == 0
    if then == "deferred":
        auto_start_local_model(settings, grant=live)
    else:
        assert asyncio.run(models.api_local_model_start(_request({"source": str(lifecycle.model)}))).status_code == 200
    assert spawn.call_count == 1 and manager.get_status() == "loading"


def test_boot_hands_the_decisions_own_grant_to_the_autostart_it_stages(monkeypatch):
    from ouroboros import desktop_autostart, execution_domain, local_model_autostart, server_boot

    staged, handed = threading.Event(), []
    monkeypatch.setattr(local_model_autostart, "auto_start_local_model",
                        lambda settings, **kwargs: (handed.append(kwargs), staged.set()))
    monkeypatch.setattr(execution_domain, "measure_generation_identity", lambda: None)
    monkeypatch.setattr(desktop_autostart, "follow_moved_app", lambda: False)  # no OS sign-in entry is read or written
    server_boot.start_background_services({"LOCAL_MODEL_SOURCE": "fixture/model"}, local_model=True,
                                          local_grant="decision-grant", boot_managed_update=lambda: None,
                                          warm_daemon=False)
    assert staged.wait(5) and handed == [{"grant": "decision-grant"}]  # the decision's receipt, not one read at this step


@pytest.fixture
def healthy_model_process(tmp_path, monkeypatch):
    """The real manager starts a lightweight /v1/models child, never downloads a model."""
    manager = local_model.LocalModelManager()
    monkeypatch.setattr(local_model, "_manager", manager)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    script = tmp_path / "model_server.py"
    script.write_text(
        'import http.server,json,os,socket,socketserver,sys\n'
        'from ouroboros.local_model_server import LOCAL_LAUNCH_ENV, LOCAL_LAUNCH_HEADER\n'
        'def no_reverse_dns(*args):\n'
        ' raise AssertionError("loopback fixture must not resolve hostnames")\n'
        'socket.getfqdn=no_reverse_dns\n'
        'class LoopbackHTTPServer(http.server.HTTPServer):\n'
        ' def server_bind(self):\n'
        '  socketserver.TCPServer.server_bind(self)\n'
        '  self.server_name,self.server_port=self.server_address[:2]\n'
        'class Handler(http.server.BaseHTTPRequestHandler):\n'
        ' def do_GET(self):\n'
        '  body=json.dumps({"data":[{"id":"fixture","context_window":9999}]}).encode()\n'
        '  self.send_response(200);self.send_header(LOCAL_LAUNCH_HEADER,os.environ[LOCAL_LAUNCH_ENV])\n'
        '  self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)\n'
        ' def log_message(self,*args):pass\n'
        'LoopbackHTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()\n', encoding="utf-8")
    original_popen, original_run = subprocess.Popen, subprocess.run
    commands = []

    def popen(command, **kwargs):
        if "ouroboros.local_model_server" in command:
            commands.append(list(command))
            command = [command[0], str(script), str(command[command.index("--port") + 1])]
        return original_popen(command, **kwargs)

    def run(command, **kwargs):
        if command[-1] == RUNTIME_PROBE:
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    settled = threading.Event()
    original_health = manager._wait_for_healthy

    def health(**launch):
        try:
            original_health(timeout=8, **launch)
        finally:
            settled.set()

    monkeypatch.setattr(local_model.subprocess, "Popen", popen)
    monkeypatch.setattr(local_model.subprocess, "run", run)
    monkeypatch.setattr(manager, "_wait_for_healthy", health)
    monkeypatch.setattr(manager, "download_model", lambda source, filename, **kwargs: str(tmp_path / filename))
    try:
        yield manager, settled, commands
    finally:
        manager.stop_server()


def test_actual_local_start_health_and_restart_clear_saved_mismatch(healthy_model_process, monkeypatch, free_tcp_port):
    manager, settled, commands = healthy_model_process
    saved = {**config.SETTINGS_DEFAULTS, "LOCAL_MODEL_SOURCE": "fixture/model",
             "LOCAL_MODEL_FILENAME": "fixture.gguf", "LOCAL_MODEL_PORT": free_tcp_port,
             "LOCAL_MODEL_CONTEXT_LENGTH": 0, "LOCAL_MODEL_N_GPU_LAYERS": 3}
    monkeypatch.setattr(models, "load_settings", lambda: saved)
    body = {"source": "fixture/model", "filename": "fixture.gguf", "port": free_tcp_port,
            "n_ctx": 0, "n_gpu_layers": 3, "chat_format": ""}
    response = asyncio.run(models.api_local_model_start(_request(body)))
    assert response.status_code == 200
    assert settled.wait(10)
    assert manager.is_running, manager.status_dict()
    # 0 starts the runtime with the factory window.
    assert commands[-1][commands[-1].index("--n_ctx") + 1] == str(config.SETTINGS_DEFAULTS["LOCAL_MODEL_CONTEXT_LENGTH"])
    assert manager.settings_application(saved)["pending_keys"] == []
    saved["LOCAL_MODEL_CONTEXT_LENGTH"] = 8192
    assert manager.settings_application(saved)["pending_keys"] == ["LOCAL_MODEL_CONTEXT_LENGTH"]
    asyncio.run(models.api_local_model_stop(_request({})))
    assert manager.settings_application(saved)["status"] == "offline"
    settled.clear()
    response = asyncio.run(models.api_local_model_start(_request({**body, "n_ctx": 8192})))
    assert response.status_code == 200
    assert settled.wait(10)
    status = json.loads(asyncio.run(models.api_local_model_status(_request({}))).body)
    assert status["status"] == "ready"
    assert status["settings_application"]["pending_keys"] == []
    # Provider training metadata does not replace the effective launch window.
    assert status["context_length"] == 9999
    assert manager._applied_settings["LOCAL_MODEL_CONTEXT_LENGTH"] == 8192
