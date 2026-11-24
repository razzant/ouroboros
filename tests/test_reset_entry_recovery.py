"""Defensive Reset admission and recovery seams, with invented hermetic fixtures only.

No real profile, listener, worker pool or restart is started. The live consumer is a passive double;
the pool-never-started case supplies the actual empty census shape produced by ``kill_workers``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from supervisor import queue, state, state_initialization


@pytest.fixture
def root(tmp_path, monkeypatch):
    prior_state, prior_queue = state.DRIVE_ROOT, queue.DRIVE_ROOT
    state.init(tmp_path)
    queue.init(tmp_path)
    monkeypatch.setattr(state_initialization, "_requested_reset", None)
    yield tmp_path
    state.init(prior_state)
    queue.init(prior_queue)


def _request(root, body, *, path="/api/reset/v1", alive=True, restarts=None, with_callback=True):
    async def payload():
        return body

    app_state = {
        "drive_root": root,
        "restart_consumer": SimpleNamespace(is_alive=lambda: alive),
        "reset_launch_protection": lambda _root: "local",
    }
    if with_callback:
        app_state["request_restart"] = lambda *, owner: restarts.append(owner) if restarts is not None else None
    return SimpleNamespace(method="POST", json=payload, url=SimpleNamespace(path=path),
                           app=SimpleNamespace(state=SimpleNamespace(**app_state)))


def _call(request):
    from ouroboros.gateway import control

    return asyncio.run(control.api_reset(request))


@pytest.mark.parametrize("body", [{}, {"confirm": "reset_all_data"}, None, "reset_all_data"])
def test_only_the_versioned_post_can_request_reset(root, monkeypatch, body):
    from ouroboros.gateway import control

    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: pytest.fail("a refused body takes no writer"))
    monkeypatch.setattr(state_initialization, "request_runtime_reset",
                        lambda *_: pytest.fail("a legacy body recorded Reset intent"))
    response = _call(_request(root, body, path="/api/reset"))
    assert response.status_code == 400
    assert json.loads(response.body)["reason"] == "reset_confirmation_required"
    assert state_initialization.reset_status(root) == {"status": "none"}


def test_the_previous_confirmation_is_not_accepted_by_the_versioned_route(root, monkeypatch):
    from ouroboros.gateway import control

    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: pytest.fail("a refused token takes no writer"))
    monkeypatch.setattr(state_initialization, "request_runtime_reset",
                        lambda *_: pytest.fail("a previous token recorded Reset intent"))
    response = _call(_request(root, {"confirm": "reset_all_data"}))
    assert response.status_code == 400
    assert state_initialization.reset_status(root) == {"status": "none"}


@pytest.mark.parametrize("alive,with_callback", [(False, True), (True, False)])
def test_no_live_restart_consumer_means_no_intent(root, monkeypatch, alive, with_callback):
    from ouroboros.gateway import control

    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: pytest.fail("an unacknowledged Reset takes no writer"))
    monkeypatch.setattr(state_initialization, "request_runtime_reset",
                        lambda *_: pytest.fail("an unacknowledged Reset recorded intent"))
    response = _call(_request(root, {"confirm": "reset_all_data:v1"}, alive=alive, with_callback=with_callback))
    assert response.status_code == 409
    assert state_initialization.reset_status(root) == {"status": "none"}


def test_the_live_restart_consumer_acknowledges_and_get_reads_the_same_status(root, monkeypatch):
    from ouroboros.gateway import control

    restarts = []
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    response = _call(_request(root, {"confirm": "reset_all_data:v1"}, restarts=restarts))
    assert response.status_code == 200
    # The acknowledgement names the Reset's own identity, as the durable write recorded it: what a page that asked
    # for this Reset matches a later read by.
    identity = state_initialization.read_witness(root)[1]["initialization_id"]
    assert json.loads(response.body) == {"status": "requested", "restarting": True, "reset_id": identity}
    assert restarts == [True]
    assert state_initialization.reset_status(root) == {"status": "requested", "reset_id": identity}
    request = _request(root, None)
    request.method = "GET"
    assert json.loads(_call(request).body) == {"status": "requested", "reset_id": identity}


def test_a_repeat_of_an_unfinished_reset_is_accepted_under_the_same_identity(root, monkeypatch):
    """What the Settings page posts again for an interrupted Reset: the same identity comes back, so one receipt
    covers every pass of that Reset and its read names the very Reset the page was answered about."""
    from ouroboros.gateway import control

    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    first = json.loads(_call(_request(root, {"confirm": "reset_all_data:v1"})).body)
    identity = first["reset_id"]
    assert identity == state_initialization.read_witness(root)[1]["initialization_id"]
    monkeypatch.setattr(state_initialization, "_requested_reset", None)  # the generation that asked it has ended
    assert state_initialization.reset_status(root) == {
        "status": "interrupted", "error": state_initialization._NO_REASON, "reset_id": identity}
    second = json.loads(_call(_request(root, {"confirm": "reset_all_data:v1"})).body)
    assert second == {"status": "requested", "restarting": True, "reset_id": identity}
    assert state_initialization.reset_status(root) == {"status": "requested", "reset_id": identity}


def test_a_pool_that_never_started_has_an_empty_census_not_an_unknown_one(root):
    from ouroboros import store_switch

    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    cleanup = {"worker_exits": {"doomed": [], "dead": [], "unconfirmed": []},
               "live_children": [], "owned_stop": {"state": "completed", "unconfirmed": []}}
    assert store_switch.switch_at_restart(root, None, cleanup, restart_drained=True) == {"status": "reset_requested"}


def test_the_real_cleanup_publishes_the_empty_pool_census_the_reset_handoff_needs(root, monkeypatch):
    """A boot whose pool never started (no held Reset is simulated): the real cleanup still reaches
    the empty kill census, with inert process surfaces."""
    import server
    from ouroboros import store_switch
    from supervisor import workers

    # Test-owned empty pool maps, bound reversibly at their authoritative owner (workers.py) instead
    # of clearing the live module maps in place. The queue sees the SAME objects, exactly as
    # workers.init binds them in production via init_queue_refs; conftest's autouse
    # _restore_supervisor_queue_refs fixture restores the queue's refs after the test.
    empty_workers: dict = {}
    empty_pending: list = []
    empty_running: dict = {}
    monkeypatch.setattr(workers, "WORKERS", empty_workers)
    monkeypatch.setattr(workers, "PENDING", empty_pending)
    monkeypatch.setattr(workers, "RUNNING", empty_running)
    monkeypatch.setattr(queue, "PENDING", empty_pending)
    monkeypatch.setattr(queue, "RUNNING", empty_running)
    queue.init_queue_refs(empty_pending, empty_running, {"value": 0})
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(workers, "_LAST_WORKER_EXIT_CENSUS", None)
    # _partition_slots_for_stop arms the conversation lane even for an empty pool (lanes=None).
    monkeypatch.setattr("supervisor.conversation_lane.arm", lambda *_a, **_k: None)
    monkeypatch.setattr(workers, "_reconcile_review_owners", lambda *_a, **_k: None)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda **_kw: True)
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {})
    monkeypatch.setattr(server, "_restart_requested", SimpleNamespace(is_set=lambda: False))
    monkeypatch.setattr(server, "_owner_restart_requested", SimpleNamespace(is_set=lambda: False))
    monkeypatch.setattr(server, "begin_owned_stop", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "stop_owned_work", lambda *_a, **_k: {"state": "completed", "unconfirmed": []})
    monkeypatch.setattr("ouroboros.maintenance.child.stop", lambda: None)
    monkeypatch.setattr("ouroboros.maintenance.child.live_children", lambda: [])
    monkeypatch.setattr("ouroboros.storage.count.stop", lambda: None)
    monkeypatch.setattr("ouroboros.storage.count.live_children", lambda: [])
    monkeypatch.setattr("ouroboros.storage_free.free.stop", lambda: None)
    monkeypatch.setattr("ouroboros.storage_free.free.live_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])

    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    cleanup = server._emergency_process_cleanup()

    census = cleanup["worker_exits"]
    assert census is not None and census["ts"]
    assert census["doomed"] == census["dead"] == census["unconfirmed"] == []
    assert census["cleanup_ok"] is True and census["snapshot_ok"] is True
    assert cleanup["live_children"] == []
    assert store_switch.switch_at_restart(root, None, cleanup, restart_drained=True) == {"status": "reset_requested"}


def test_a_missing_worker_census_stays_unknown_on_retry(root):
    from ouroboros import store_switch

    state_initialization.request_runtime_reset(root, launch_protection=lambda: "local")
    cleanup = {"worker_exits": None, "live_children": [], "owned_stop": {"state": "completed", "unconfirmed": []}}
    assert store_switch.switch_at_restart(root, None, cleanup, restart_drained=True) == {"status": "reset_interrupted"}
    assert state_initialization.reset_status(root)["status"] == "interrupted"
    assert "worker_exits_unconfirmed" in state_initialization.reset_status(root)["error"]


@pytest.mark.parametrize("status,expected", [("interrupted", False), ("none", True)])
def test_interrupted_reset_does_not_force_the_first_run_wizard(monkeypatch, status, expected):
    from ouroboros import launcher_onboarding

    monkeypatch.setattr(launcher_onboarding, "load_settings", lambda: {})
    monkeypatch.setattr(launcher_onboarding, "apply_runtime_provider_defaults", lambda settings: (settings, False, set()))
    monkeypatch.setattr(launcher_onboarding, "_apply_settings_to_env", lambda settings: None)
    monkeypatch.setattr(launcher_onboarding, "has_startup_ready_provider", lambda settings: False)
    monkeypatch.setattr(state_initialization, "reset_status", lambda _: {"status": status})
    _settings, required = launcher_onboarding.prepare_first_run_settings()
    assert required is expected


@pytest.fixture
def reset_launch_env(root, monkeypatch):
    import os
    from ouroboros import config

    monkeypatch.setattr(os, "environ", {"OUROBOROS_DATA_DIR": str(root)})
    monkeypatch.setattr(config, "DATA_DIR", root)
    document = root / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_PATH", document)
    return document


def test_reset_through_the_real_cli_callback_from_outside_checkout(root, reset_launch_env, tmp_path, monkeypatch):
    import logging
    import os
    import pathlib
    import sys
    from ouroboros import cli, config, server_control, settings_integrity as integrity
    from ouroboros.gateway import control
    from ouroboros.launch_roots import RESET_PROTECTION_KEY
    from ouroboros.server_entrypoint import ResetLaunch

    repo = pathlib.Path(cli.__file__).resolve().parents[1]
    outside = tmp_path / "caller"
    outside.mkdir()
    monkeypatch.chdir(outside)
    tail = ["server", "--host", "127.0.0.1", "--port", "8765", "--no-ui"]
    monkeypatch.setattr(sys, "argv", ["ouroboros", *tail])
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "3")
    reset_launch_env.write_text('{"OUROBOROS_MAX_WORKERS": 3}', encoding="utf-8")
    calls, replies, restarts = [], [], []
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", lambda: (None, None))
    monkeypatch.setattr(server_control, "IS_WINDOWS", False)
    monkeypatch.setattr(server_control, "_restart_stop_requested", False)
    monkeypatch.setattr(os, "execvpe", lambda _exe, argv, env: calls.append((argv, env)))

    def entry():
        assert sys.argv == ["ouroboros"]
        launch = ResetLaunch(repo, False)
        config.apply_settings_to_env(config.load_settings())
        req = _request(root, {"confirm": "reset_all_data:v1"}, restarts=restarts)
        req.app.state.reset_launch_protection = lambda path: launch.approve("127.0.0.1", path)
        replies.append(_call(req))
        assert server_control.restart_arguments(os.environ) == ("-m", "ouroboros.cli", *tail)
        launch.select_reset_reexec({"status": "reset_requested"})
        server_control.restart_current_process("127.0.0.1", 8765, repo_dir=repo,
                                               log=logging.getLogger(__name__), owner_initiated=True)
        return 0

    monkeypatch.setitem(sys.modules, "server", SimpleNamespace(__file__=str(repo / "server.py"), executable_main=entry))
    assert cli.main() == 0
    assert len(replies) == 1 and replies[0].status_code == 200 and restarts == [True]
    assert state_initialization.read_witness(root)[1][RESET_PROTECTION_KEY] == "local"
    argv, successor_env = calls[0]
    assert argv[1:] == [str(repo / "server.py"), "--host=127.0.0.1", "--port=8765"]
    monkeypatch.setattr(os, "environ", successor_env)
    monkeypatch.setattr(sys, "argv", argv[1:])
    assert ResetLaunch(repo, False).at_boot(root) == "local"
    integrity.drop_projection(successor_env)
    assert successor_env["OUROBOROS_MAX_WORKERS"] == "3"
    assert reset_launch_env.read_text(encoding="utf-8") == '{"OUROBOROS_MAX_WORKERS": 3}'
    # An unrecorded legacy mirror is still recognized, not promoted to operator input.
    legacy = {"OUROBOROS_MAX_WORKERS": "3"}
    config.apply_settings_to_env(config.load_settings(), environ=legacy)
    integrity.drop_projection(legacy)
    assert "OUROBOROS_MAX_WORKERS" not in legacy


@pytest.mark.parametrize("proven", [True, False])
def test_managed_reset_uses_the_launchers_input_not_its_document(root, reset_launch_env, monkeypatch, proven):
    import os
    import sys
    import launcher
    from ouroboros import config, settings_integrity as integrity
    from ouroboros.gateway import control
    from ouroboros.launch_roots import RESET_PROTECTION_KEY
    from ouroboros.server_entrypoint import ResetLaunch
    from ouroboros.utils import atomic_write_json

    host, password = "192.0.2.21", "fixture-password"
    if proven:
        monkeypatch.setenv(integrity.NETWORK_PASSWORD_KEY, password)
    monkeypatch.setenv(integrity.PROJECTED_SETTINGS_ENV, "{}")  # the fresh launch boundary
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", host)
    saved = json.dumps({integrity.NETWORK_PASSWORD_KEY: password})
    reset_launch_env.write_text(saved, encoding="utf-8")
    spawn = []
    monkeypatch.setattr(launcher, "DATA_DIR", root)
    monkeypatch.setattr(launcher, "SETTINGS_PATH", reset_launch_env)
    monkeypatch.setattr(launcher, "REPO_DIR", root)
    monkeypatch.setattr(launcher, "EMBEDDED_PYTHON", sys.executable)
    monkeypatch.setattr(launcher, "_reset_legacy_input", False)
    monkeypatch.setattr(launcher, "_load_settings", config.load_settings)
    monkeypatch.setattr(launcher, "_apply_settings_to_env", config.apply_settings_to_env)
    monkeypatch.setattr(launcher, "_shutdown_event", SimpleNamespace(is_set=lambda: False))
    monkeypatch.setattr(launcher, "threading", SimpleNamespace(Thread=lambda **_: SimpleNamespace(start=lambda: None)))
    monkeypatch.setattr(launcher, "_write_server_process_record", lambda *_a, **_kw: None)
    monkeypatch.setattr(launcher, "_shell_facts", lambda: {"bundle_version": ""})
    monkeypatch.setattr(launcher, "IS_WINDOWS", False)
    monkeypatch.setattr(launcher, "_agent_proc", None)
    monkeypatch.setattr(launcher, "_agent_start", None)

    def capture(argv, **kwargs):
        spawn.append((argv[1:], kwargs["env"]))
        return SimpleNamespace(pid=42, stdout=None)

    def writer():
        assert proven, "unproved admission opened the mutation lock"
        return None, None

    monkeypatch.setattr(launcher, "_hidden_popen", capture)
    monkeypatch.setattr(control, "_acquire_repo_mutation_lock", writer)
    launcher.start_agent(port=8765)
    argv, child_env = spawn[-1]
    restarts = []
    with monkeypatch.context() as child:
        child.setattr(os, "environ", child_env)
        child.setattr(sys, "argv", argv)
        launch = ResetLaunch(root, True)
        config.apply_settings_to_env(config.load_settings())
        req = _request(root, {"confirm": "reset_all_data:v1"}, restarts=restarts)
        req.app.state.reset_launch_protection = lambda path: launch.approve(host, path)
        response = _call(req)
    assert reset_launch_env.read_text(encoding="utf-8") == saved
    if not proven:
        assert response.status_code == 409 and json.loads(response.body)["reason"] == "reset_protection_unproved"
        assert restarts == [] and state_initialization.reset_status(root) == {"status": "none"}
        return
    assert response.status_code == 200 and restarts == [True]
    _, witness = state_initialization.read_witness(root)
    assert witness[RESET_PROTECTION_KEY] == "network_password" and password not in json.dumps(witness)
    atomic_write_json(state_initialization.witness_path(root), {**witness, "reset_handed_off": True})
    launcher.start_agent(port=8765)
    argv, env = spawn[-1]
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(sys, "argv", argv)
    assert ResetLaunch(root, True).at_boot(root) == "network_password"


def test_unproved_boot_keeps_confirmed_handoff_without_cleanup(root, reset_launch_env, monkeypatch):
    from ouroboros import server_entrypoint
    from ouroboros.utils import atomic_write_json

    reset_launch_env.write_text("{}", encoding="utf-8")
    state_initialization.request_runtime_reset(root, launch_protection=lambda: "network_password")
    _, witness = state_initialization.read_witness(root)
    path = state_initialization.witness_path(root)
    atomic_write_json(path, {**witness, "reset_handed_off": True})
    before = path.read_bytes()
    monkeypatch.setattr(server_entrypoint, "_admission", {})
    monkeypatch.setattr(state_initialization, "_clear_runtime_data", lambda *_: pytest.fail("cleanup started"))
    with pytest.raises(server_entrypoint.ServerAdmissionRefused, match="left intact"):
        server_entrypoint.admit_data_root(root, wait_sec=0)
    assert path.read_bytes() == before and reset_launch_env.read_text(encoding="utf-8") == "{}"
