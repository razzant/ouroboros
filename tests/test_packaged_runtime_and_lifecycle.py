"""Regressions for packaged-runtime resolution and terminal-path symmetry.

Covers the v6.87.10 fixes: bundled payloads reachable from the server process,
pip kept out of the signed bundle (and its exit code honored), the Windows
python download checksum, the merge-aware Update Now action, the
finalization-grace latch, the settings->env export derivation, and the
cancellation path's partial-result rescue — plus the v6.87.34 grace EPISODE:
the latch and its durable finalize_now control are withdrawn together, and
only the task's OWN progress withdraws them.
"""

import logging
import pathlib
import subprocess
import sys
import types

import pytest

import ouroboros.launcher_bootstrap as bootstrap_module
from ouroboros import platform_layer


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _log_stub(sink):
    def _record(level):
        def _log(msg, *args, **kwargs):
            sink.append((level, msg % args if args else msg))
        return _log

    return types.SimpleNamespace(
        info=_record("info"), warning=_record("warning"), error=_record("error"),
        debug=_record("debug"),
    )


# --------------------------------------------------------------------------
# F1: bundled node/ripgrep must be visible to the SERVER process, which runs
# out of the managed repo with no sys._MEIPASS.
# --------------------------------------------------------------------------

def _make_bundled_rg(base: pathlib.Path) -> pathlib.Path:
    candidate = platform_layer.embedded_ripgrep_candidates(base)[0]
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    return candidate


def test_bundled_payload_resolves_through_bundle_dir_env(tmp_path, monkeypatch):
    """A process with no _MEIPASS and a repo-root elsewhere still finds the payload."""
    bundle = tmp_path / "bundle"
    expected = _make_bundled_rg(bundle)
    monkeypatch.delenv(platform_layer.BUNDLE_DIR_ENV, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    # The resolver also keeps the repo root as the dev-layout base and, for an
    # older launcher, recovers immutable resources from the embedded Python
    # path. Only assert the historical miss when neither fallback exists.
    fallback_payload_present = any(
        candidate.exists()
        for base in [REPO_ROOT, *platform_layer.bundled_resource_ancestor_bases()]
        for candidate in platform_layer.embedded_ripgrep_candidates(base)
    )
    if not fallback_payload_present:
        assert platform_layer.resolve_bundled_ripgrep() is None  # the pre-fix behaviour

    monkeypatch.setenv(platform_layer.BUNDLE_DIR_ENV, str(bundle))
    # The env base is searched FIRST, so this holds with or without a dev payload.
    assert platform_layer.resolve_bundled_ripgrep() == str(expected)


def test_bundled_node_uses_the_same_bases(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    candidate = platform_layer.embedded_node_candidates(bundle)[0]
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv(platform_layer.BUNDLE_DIR_ENV, str(bundle))
    assert platform_layer.resolve_bundled_node() == str(candidate)


def test_bundle_dir_env_is_exported_to_the_server_and_cli():
    """The two spawn seams must hand the bundle root down; nothing else can."""
    launcher_src = (REPO_ROOT / "launcher.py").read_text(encoding="utf-8")
    start_agent = launcher_src.split("def start_agent", 1)[1].split("\ndef ", 1)[0]
    assert "BUNDLE_DIR_ENV" in start_agent

    cli_src = (REPO_ROOT / "ouroboros" / "packaged_cli.py").read_text(encoding="utf-8")
    inner_env = cli_src.split("def _inner_cli_env", 1)[1].split("\ndef ", 1)[0]
    assert "BUNDLE_DIR_ENV: str(runtime.bundle_root)" in inner_env


# --------------------------------------------------------------------------
# F4 / F3: pip must not write inside the signed bundle, and its exit code must
# not be swallowed.
# --------------------------------------------------------------------------

def test_embedded_python_env_redirects_both_bundle_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    env = bootstrap_module.embedded_python_env(tmp_path)
    state = tmp_path / "state"
    assert env["PYTHONPYCACHEPREFIX"] == str(state / "pycache")
    assert env["PYTHONUSERBASE"] == str(state / "python-userbase")
    # An inherited no-user-site would install to the user site then refuse to import it.
    assert "PYTHONNOUSERSITE" not in env


def test_pip_install_target_args_only_for_the_embedded_interpreter(tmp_path):
    embedded = tmp_path / "python-standalone" / "bin" / "python3"
    embedded.parent.mkdir(parents=True)
    embedded.write_text("", encoding="utf-8")
    assert platform_layer.pip_install_target_args(str(embedded)) == ["--user"]
    # A dev venv refuses --user; a blanket flag would break it. Pin the ANSWER to the
    # interpreter's own location, never to whichever interpreter runs the suite: under
    # the bundled python `sys.executable` IS embedded, so asserting [] for it failed on
    # every packaged install — and the tests preflight is fail-closed, so that took the
    # self-modification commit gate down with it.
    venv = tmp_path / ".venv" / "bin" / "python3"
    venv.parent.mkdir(parents=True)
    venv.write_text("", encoding="utf-8")
    assert platform_layer.pip_install_target_args(str(venv)) == []


def _install_deps_context(tmp_path, interpreter, returncode, sink):
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "requirements-runtime.lock").write_text("anyio\n", encoding="utf-8")
    calls = []

    def _run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode, stdout=b"", stderr=b"boom")

    context = bootstrap_module.BootstrapContext(
        bundle_dir=tmp_path, repo_dir=tmp_path / "repo", data_dir=tmp_path / "data",
        settings_path=tmp_path / "settings.json", embedded_python=str(interpreter),
        app_version="6.87.10", hidden_run=_run, save_settings=lambda s: None,
        log=_log_stub(sink),
    )
    return context, calls


def test_install_deps_targets_the_user_site_for_a_bundled_interpreter(tmp_path):
    embedded = tmp_path / "python-standalone" / "bin" / "python3"
    embedded.parent.mkdir(parents=True)
    embedded.write_text("", encoding="utf-8")
    sink = []
    context, calls = _install_deps_context(tmp_path, embedded, 0, sink)
    assert bootstrap_module.install_deps(context) is True
    assert "--user" in calls[0]


def test_install_deps_reports_a_failing_pip(tmp_path):
    """Reported at WARNING, not ERROR: this failure is RETURNED as the function's own
    value and every caller consumes it (the three tests below exist to prove that), so
    it is a handled outcome, not an unhandled fault. Logging it at error level made an
    ordinary offline/mirror-flaky bootstrap — which continues, and may well have every
    package it needs already — indistinguishable from a crash in the launcher log."""
    sink = []
    context, _calls = _install_deps_context(tmp_path, sys.executable, 1, sink)
    assert bootstrap_module.install_deps(context) is False
    assert [msg for level, msg in sink if level == "error"] == []
    warnings = [msg for level, msg in sink if level == "warning"]
    assert any("pip exited 1" in msg and "boom" in msg for msg in warnings)


# --------------------------------------------------------------------------
# XG-7B.3: install_deps' False must reach the decisions that consume it — the
# launcher wrapper, bootstrap, and the restart loop — not die in a bare call.
# --------------------------------------------------------------------------

def test_the_launcher_wrapper_propagates_the_install_result(monkeypatch):
    import launcher

    monkeypatch.setattr(launcher, "_bootstrap_context", lambda: None)
    monkeypatch.setattr(launcher, "_install_deps_impl", lambda ctx: False)
    assert launcher._install_deps() is False
    monkeypatch.setattr(launcher, "_install_deps_impl", lambda ctx: True)
    assert launcher._install_deps() is True


def test_bootstrap_repo_reports_a_failed_dependency_install(tmp_path, monkeypatch):
    sink = []
    context = bootstrap_module.BootstrapContext(
        bundle_dir=tmp_path, repo_dir=tmp_path / "repo", data_dir=tmp_path / "data",
        settings_path=tmp_path / "settings.json", embedded_python=sys.executable,
        app_version="6.87.35",
        hidden_run=lambda *a, **k: subprocess.CompletedProcess([], 0, b"", b""),
        save_settings=lambda s: None, log=_log_stub(sink),
    )
    monkeypatch.setattr(bootstrap_module, "ensure_managed_repo", lambda c: "updated")
    monkeypatch.setattr(bootstrap_module, "bootstrap_native_skills", lambda c: None)
    monkeypatch.setattr(bootstrap_module, "install_deps", lambda c: False)
    assert bootstrap_module.bootstrap_repo(context) is False
    # WARNING, not ERROR, for the same reason as `install_deps` itself: bootstrap
    # RETURNS False here and the caller decides what to do about it. The line is the
    # explanation attached to a handled outcome, and the assertion below is what
    # actually proves the outcome was not swallowed.
    assert [msg for level, msg in sink if level == "error"] == []
    assert any("FAILED dependency install" in msg for level, msg in sink if level == "warning")

    monkeypatch.setattr(bootstrap_module, "ensure_managed_repo", lambda c: "unchanged")
    assert bootstrap_module.bootstrap_repo(context) is True  # no install needed, no failure


def test_a_restart_with_a_failed_dependency_install_pauses_and_discloses(
    monkeypatch, tmp_path, caplog,
):
    """XG-7B.3 through the REAL caller: agent_lifecycle_loop used to discard
    _install_deps' result and restart an evolved checkout without the packages
    its reviewed commit added. It must retry once, visibly, and name the
    consequence when the retry also fails."""
    import logging
    import threading

    import launcher

    monkeypatch.setattr(launcher, "_shutdown_event", threading.Event())
    monkeypatch.setattr(launcher, "_cleanup_recorded_server_process", lambda reason: None)
    monkeypatch.setattr(launcher, "_kill_stale_runtime_ports", lambda port: None)
    # The per-generation stray sweep must never signal a real process from a test.
    monkeypatch.setattr(launcher, "_reap_same_install_strays", lambda reason: [])
    monkeypatch.setattr(
        launcher, "_cleanup_recorded_server_group_for_pid", lambda pid, reason: None,
    )
    monkeypatch.setattr(launcher, "_update_server_process_record_port", lambda pid, port: None)
    monkeypatch.setattr(launcher, "_poll_port_file", lambda timeout=30: 52123)
    monkeypatch.setattr(launcher, "_wait_for_server", lambda port, timeout=45: True)
    monkeypatch.setattr(launcher, "PORT_FILE", tmp_path / "port")
    monkeypatch.setattr(launcher, "_sync_existing_repo_from_bundle", lambda: None)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    install_calls = []

    def _failing_install():
        install_calls.append(1)
        return False

    monkeypatch.setattr(launcher, "_install_deps", _failing_install)

    starts = []

    def _fake_start_agent(port):
        starts.append(port)
        if len(starts) == 1:
            return types.SimpleNamespace(
                pid=1, returncode=launcher.RESTART_EXIT_CODE, wait=lambda: None,
            )
        launcher._shutdown_event.set()
        return types.SimpleNamespace(pid=2, returncode=0, wait=lambda: None)

    monkeypatch.setattr(launcher, "start_agent", _fake_start_agent)

    with caplog.at_level(logging.ERROR, logger="launcher"):
        launcher.agent_lifecycle_loop(port=52123)

    assert len(install_calls) == 2, "no visible retry: the failed install was discarded"
    assert len(starts) == 2, "the restart must still proceed under the crash fuse"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("Dependency install failed" in m for m in errors)
    assert any("crash fuse" in m for m in errors)


# --------------------------------------------------------------------------
# F2: no unverified download may become the packaged runtime.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "script",
    sorted(p.name for p in (REPO_ROOT / "scripts").glob("download_*_standalone.*")),
)
def test_every_standalone_download_verifies_a_checksum(script):
    text = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8").lower()
    assert "sha256" in text, f"{script} downloads a runtime without verifying it"
    assert any(token in text for token in ("mismatch", "verified")), (
        f"{script} computes a hash but never refuses on mismatch"
    )


def test_windows_python_download_pins_the_release_checksum():
    text = (REPO_ROOT / "scripts" / "download_python_standalone.ps1").read_text(encoding="utf-8")
    assert "Get-FileHash -Algorithm SHA256" in text
    assert "throw" in text.split("Get-FileHash", 1)[1]


def test_windows_python_download_checks_native_dependency_install_exits():
    text = (REPO_ROOT / "scripts" / "download_python_standalone.ps1").read_text(encoding="utf-8")
    assert "Agent dependency installation failed with exit code" in text
    assert "llama-cpp-python installation failed with exit code" in text


# --------------------------------------------------------------------------
# F5: Update Now must go through the merge plan, not the hard-reset hatch.
# --------------------------------------------------------------------------

def test_update_now_posts_the_merge_aware_strategy():
    """Update Now decides its strategy FROM THE MERGE PLAN (v6.88.0 flow:
    preflight -> verifiedUpdatePlan -> typed apply; the shared verifier wraps
    updateStrategyForPlan since the 2026-08-31 redesign); the legacy
    'replace'/'stash' escape hatch must stay out of the normal path (it lives
    only behind the explicit Recovery confirmation)."""
    text = (REPO_ROOT / "web" / "modules" / "updates.js").read_text(encoding="utf-8")
    apply_fn = text.split("async function applyUpdate", 1)[1].split("\n    }", 1)[0]
    code = "\n".join(
        line for line in apply_fn.splitlines() if not line.strip().startswith("//")
    )
    assert "updatePreflight" in code
    assert "verifiedUpdatePlan" in code
    for legacy in ("'replace'", "'stash'"):
        assert legacy not in code, f"Update Now still reaches for the legacy {legacy} path"
    assert "assisted_started" in code


# --------------------------------------------------------------------------
# G2: a settings key that never reaches the environment is a silent no-op.
# --------------------------------------------------------------------------

def test_every_settings_key_is_exported_unless_named():
    from ouroboros import config

    exported = set(config.settings_env_keys())
    missing = set(config.SETTINGS_DEFAULTS) - exported - config.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    assert not missing, f"settings keys accepted but never exported to env: {sorted(missing)}"
    assert config.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV <= set(config.SETTINGS_DEFAULTS)


def test_skill_lifecycle_timeout_setting_reaches_the_queue(monkeypatch):

    from ouroboros import config
    from ouroboros import skill_lifecycle_queue

    # XG-7B.4: apply_settings_to_env pops every other settings-default key and
    # injects review defaults. The autouse os.environ snapshot in conftest
    # restores the live environ afterwards, so nothing leaks into sibling tests.
    monkeypatch.delenv("OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC", raising=False)
    assert skill_lifecycle_queue._lifecycle_deadline_sec() == float(
        config.SETTINGS_DEFAULTS["OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC"]
    )
    monkeypatch.setattr(config, "_DISK_AUTHORED_SETTINGS", ())
    config.apply_settings_to_env({"OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC": 42})
    assert skill_lifecycle_queue._lifecycle_deadline_sec() == 42.0


# --------------------------------------------------------------------------
# E1: the finalization-grace latch belongs to one episode, not to the task.
#
# The latch-only version of this scenario is gone: it asserted the metadata half
# and nothing else, which is exactly how the split state (latch cleared, durable
# control still readable) read as correct — and, once the clear grew a real
# withdrawal, its unstubbed event bus spawned a live SyncManager per run. The
# harness below covers the same resume through the real path, plus the mailbox.
# --------------------------------------------------------------------------

def _enforce_harness(monkeypatch, tmp_path, running, *, idle=900, grace=300):
    """Drive the real enforce loop the way the supervisor loop actually drives it.

    The bus is PUMPED: server.py drains the event queue through ``dispatch_event``
    immediately before every ``enforce_task_timeouts()``, and that dispatch writes
    into the very RUNNING rows the enforce loop then reads. A harness that leaves
    the bus undrained measures a world the wire never produces — which is how a
    grace episode that the supervisor's own toast revoked 0.5s later passed as
    correct. ``ctx.RUNNING`` is the SAME dict (production invariant: workers.init
    -> queue.init_queue_refs), so the pump is not an approximation.
    """
    import queue as _stdqueue

    from supervisor import events as events_mod
    from supervisor import queue as queue_mod
    from supervisor import task_reaper, workers as workers_mod

    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "RUNNING", running)
    monkeypatch.setattr(queue_mod, "PENDING", [])
    monkeypatch.setattr(queue_mod, "get_task_idle_timeout_sec", lambda: idle)
    monkeypatch.setattr(queue_mod, "get_per_call_timeout_ceiling_sec", lambda: 60)
    monkeypatch.setattr(queue_mod, "get_task_abs_ceiling_sec", lambda: 10_000_000)
    monkeypatch.setattr(queue_mod, "FINALIZATION_GRACE_SEC", grace)
    monkeypatch.setattr(queue_mod, "persist_queue_snapshot", lambda **_k: True)
    monkeypatch.setattr(queue_mod, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue_mod, "_reap_queue", _stdqueue.Queue())
    bus = _stdqueue.Queue()
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: bus)
    monkeypatch.setattr(task_reaper, "send_with_budget", lambda *_a, **_k: True)

    clock = [0.0]
    delivered = []
    monkeypatch.setattr(events_mod, "time", types.SimpleNamespace(time=lambda: clock[0]))
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], WORKERS={},
        send_with_budget=lambda _cid, text, **_k: delivered.append(str(text)),
        append_jsonl=lambda *_a, **_k: None,
    )

    def _tick(now):
        clock[0] = now
        while True:
            try:
                evt = bus.get_nowait()
            except _stdqueue.Empty:
                break
            events_mod.dispatch_event(evt, ctx)
        queue_mod._enforce_task_timeouts_locked(types.SimpleNamespace(WORKERS={}), now, 7, {})

    _tick.delivered = delivered  # what the pump actually put on the owner's wire
    return _tick


def _live_finalize_controls(drive, task_id):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, drain_owner_entries

    return [
        entry for entry in drain_owner_entries(drive, task_id)
        if entry.get("kind") == KIND_FINALIZE_NOW
    ]


def test_resuming_task_has_its_finalize_now_control_withdrawn(monkeypatch, tmp_path):
    """The latch and the mailbox control are ONE episode: clearing the latch while
    the durable finalize_now stays readable let the task drain a stale kill order
    and finalize early with no terminal condition pending."""
    task_id = "t-episode"
    meta = {
        "task": {"id": task_id, "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    tick = _enforce_harness(monkeypatch, tmp_path, {task_id: meta})

    tick(2000.0)  # idle past the timeout: the grace episode opens
    assert meta["finalization_requested_at"] == 2000.0
    assert len(_live_finalize_controls(tmp_path, task_id)) == 1

    meta["last_progress_at"] = 2100.0  # the task itself went back to work
    tick(2110.0)
    assert "finalization_requested_at" not in meta
    assert _live_finalize_controls(tmp_path, task_id) == []


def test_wedged_orchestrator_with_a_flickering_subtree_still_reaches_the_kill(
    monkeypatch, tmp_path,
):
    """A DESCENDANT's progress spares the orchestrator from the kill but is not the
    orchestrator answering its own grace request. Revoking the episode on subtree
    activity re-armed it on every flicker: the window never elapsed and one
    finalize_now control piled up per tick."""
    orch, child = "orch1", "child1"
    orch_meta = {
        "task": {"id": orch, "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    child_meta = {
        "task": {"id": child, "chat_id": 7, "parent_task_id": orch, "root_task_id": orch},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 1,
    }
    from supervisor import queue as queue_mod

    running = {orch: orch_meta, child: child_meta}
    tick = _enforce_harness(monkeypatch, tmp_path, running)

    now = 1000.0
    for step in range(20):
        now += 1000.0
        if step % 2 == 0:  # the child progresses every other tick
            child_meta["last_progress_at"] = now
        tick(now)
        if orch not in queue_mod.RUNNING:
            break

    assert orch not in queue_mod.RUNNING, "wedged orchestrator was never reaped"
    assert len(_live_finalize_controls(tmp_path, orch)) == 1
    assert child in queue_mod.RUNNING  # the progressing child is untouched


def test_supervisor_grace_toast_is_not_the_task_answering_it(monkeypatch, tmp_path):
    """The grace toast is addressed to the task's card but AUTHORED by the supervisor.

    Counting it as the task's own work made the supervisor answer its own question:
    the toast stamped last_progress_at, the next 0.5s tick read the task as resumed,
    and the episode was withdrawn — taking the finalize_now with it, so cooperative
    finalization never reached a wedged task at all.
    """
    task_id = "t-narration"
    meta = {
        "task": {"id": task_id, "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    tick = _enforce_harness(monkeypatch, tmp_path, {task_id: meta}, grace=120)

    tick(2000.0)  # the episode opens and queues its owner toast
    assert meta["finalization_requested_at"] == 2000.0
    tick(2000.5)  # the toast is dispatched here, exactly as the loop does it

    assert any("reached idle_timeout" in line for line in tick.delivered), (
        "the harness never drained the bus — the toast under test was never dispatched"
    )
    assert meta["last_progress_at"] == 1000.0, "host narration counted as the task's work"
    assert meta.get("finalization_requested_at") == 2000.0
    assert len(_live_finalize_controls(tmp_path, task_id)) == 1, "control revoked undelivered"

    tick(2121.0)  # the window elapses with the task still silent
    from supervisor import queue as queue_mod

    assert task_id not in queue_mod.RUNNING, "wedged task was never reaped"


def test_a_spared_task_still_gets_its_whole_grace_window(monkeypatch, tmp_path):
    """Sparing suspends the stop, so it suspends the window.

    An orchestrator blocked in wait_tasks makes no own progress and drains no
    mailbox. If its episode's clock runs down while the subtree keeps it
    deliberately alive, it is killed at the first tick the subtree goes quiet —
    with the finalize_now still unread. It must get the full window from the
    moment sparing ends, and still only ONE episode and ONE control.
    """
    orch, child = "orch2", "child2"
    orch_meta = {
        "task": {"id": orch, "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    child_meta = {
        "task": {"id": child, "chat_id": 7, "parent_task_id": orch, "root_task_id": orch},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 1,
    }
    from supervisor import queue as queue_mod

    running = {orch: orch_meta, child: child_meta}
    tick = _enforce_harness(monkeypatch, tmp_path, running, grace=120)

    tick(2000.0)  # both silent: the orchestrator's episode opens
    assert orch_meta["finalization_requested_at"] == 2000.0

    child_meta["last_progress_at"] = 2050.0  # the child resumes: orchestrator spared
    for now in (2060.0, 2500.0, 2900.0):  # long past 2000 + 120
        tick(now)
        assert orch in queue_mod.RUNNING, f"spared orchestrator was killed at {now}"

    running.pop(child)  # the child finishes; the subtree goes quiet
    tick(2900.5)
    assert orch in queue_mod.RUNNING, "killed at the first quiet tick with zero grace"
    assert len(_live_finalize_controls(tmp_path, orch)) == 1, "a second episode was opened"

    tick(3021.0)  # a full window after sparing ended
    assert orch not in queue_mod.RUNNING
    assert len(_live_finalize_controls(tmp_path, orch)) == 1


def test_child_settlement_stamps_parent_activity_and_withdraws_grace(
    monkeypatch, tmp_path,
):
    """Q5 (slime saga): a coordinator waiting on children was idle-killed 120s
    after its last child DELIVERED its result — delivery did not count as parent
    activity, so the parent died exactly when integration should start. The
    child's terminal dispatch now stamps the PARENT's own progress, so a parent
    inside a finalization-grace episode is spared and the episode is withdrawn
    whole by the EXISTING spare machinery (own progress answers the request)."""
    from supervisor import events as events_mod
    from supervisor import queue as queue_mod

    orch, child = "orch3", "child3"
    orch_meta = {
        "task": {"id": orch, "chat_id": 7},
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 0,
    }
    child_task = {
        "id": child, "chat_id": 7, "parent_task_id": orch, "root_task_id": orch,
        "delegation_role": "subagent",
    }
    child_meta = {
        "task": child_task,
        "started_at": 1000.0, "last_progress_at": 1000.0, "worker_id": 1,
    }
    running = {orch: orch_meta, child: child_meta}
    tick = _enforce_harness(monkeypatch, tmp_path, running, grace=120)

    tick(2000.0)  # both idle: the orchestrator's grace episode opens
    assert orch_meta["finalization_requested_at"] == 2000.0
    assert len(_live_finalize_controls(tmp_path, orch)) == 1

    tick(2050.0)  # inside the grace window; also moves the clock the stamp reads
    # The child's terminal result is DELIVERED — the settled task_done dispatch.
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], WORKERS={},
        send_with_budget=lambda _cid, _text, **_k: None,
        append_jsonl=lambda *_a, **_k: None,
        persist_queue_snapshot=lambda **_k: True,
        bridge=types.SimpleNamespace(push_log=lambda _e: None),
    )
    events_mod._finish_task_done_dispatch(
        {}, ctx, task_id=child, worker_id=1, task=child_task,
        final_task_result={}, task_done_event={"type": "task_done", "task_id": child},
    )

    assert child not in running
    assert orch_meta["last_progress_at"] == 2050.0, "settlement did not stamp the parent"

    tick(2055.0)  # own progress: the episode is withdrawn whole, parent spared
    assert orch in queue_mod.RUNNING
    assert "finalization_requested_at" not in orch_meta
    assert _live_finalize_controls(tmp_path, orch) == []

    tick(3000.0)  # the stamp is one-shot: a genuinely idle parent still reaches
    assert orch_meta.get("finalization_requested_at") == 3000.0  # a fresh episode


def test_provider_outage_root_terminal_notifies_owner_chat(tmp_path):
    """Q7 (slime saga): a root task terminalized by a provider outage must tell
    the owner immediately that it was NOT completed — the historical shape was
    95 minutes of silence behind a result claiming "completed (best effort)"."""
    from supervisor import events as events_mod

    sent = []

    def _ctx(running):
        return types.SimpleNamespace(
            DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], WORKERS={},
            send_with_budget=lambda cid, text, **_k: sent.append((cid, str(text))),
            append_jsonl=lambda *_a, **_k: None,
            persist_queue_snapshot=lambda **_k: True,
            bridge=types.SimpleNamespace(push_log=lambda _e: None),
        )

    root_task = {"id": "root9", "chat_id": 7}
    events_mod._finish_task_done_dispatch(
        {}, _ctx({"root9": {"task": root_task, "worker_id": 0}}),
        task_id="root9", worker_id=0, task=root_task, final_task_result={},
        task_done_event={
            "type": "task_done", "task_id": "root9", "chat_id": 7,
            "status": "failed", "reason_code": "provider_unavailable",
        },
    )
    outage_lines = [t for _c, t in sent if "provider outage" in t]
    assert outage_lines and "NOT completed" in outage_lines[0]

    # A CHILD's provider death keeps the ordinary subagent toast only — the
    # parent absorbs child failures; no second owner ping per child.
    sent.clear()
    child_task = {
        "id": "kid9", "chat_id": 7, "parent_task_id": "root9",
        "root_task_id": "root9", "delegation_role": "subagent",
    }
    events_mod._finish_task_done_dispatch(
        {"status": "failed"}, _ctx({"kid9": {"task": child_task, "worker_id": 1}}),
        task_id="kid9", worker_id=1, task=child_task, final_task_result={},
        task_done_event={
            "type": "task_done", "task_id": "kid9", "chat_id": 7,
            "status": "failed", "reason_code": "provider_unavailable",
        },
    )
    assert not [t for _c, t in sent if "provider outage" in t]
    assert [t for _c, t in sent if "Subagent kid9 failed" in t]


def test_every_host_authored_progress_frame_declares_itself():
    """The gate only works if host emitters declare themselves, so make that
    structural rather than a habit: any supervisor-side event-bus frame that
    carries both ``is_progress`` and a ``task_id`` is narration ABOUT a task and
    must be marked, or it silently becomes evidence the task is working."""
    import ast

    from supervisor.events import HOST_NARRATION

    offenders = []
    for path in sorted((REPO_ROOT / "supervisor").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "put"):
                continue
            producer = node.func.value
            if not isinstance(producer, ast.Call):
                continue
            producer_fn = producer.func
            name = getattr(producer_fn, "attr", None) or getattr(producer_fn, "id", None)
            if name != "get_event_q":
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Dict):
                    continue
                literal = {k.value for k in arg.keys if isinstance(k, ast.Constant)}
                symbolic = {k.id for k in arg.keys if isinstance(k, ast.Name)}
                if not ({"is_progress", "task_id"} <= literal):
                    continue
                if HOST_NARRATION not in literal and "HOST_NARRATION" not in symbolic:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "supervisor-authored progress frames missing the HOST_NARRATION marker "
        f"(they would stamp the task's last_progress_at): {offenders}"
    )


def test_grace_latch_survives_a_control_revocation_that_did_not_persist(
    monkeypatch, tmp_path,
):
    """Fail closed: half a withdrawal is the same split state as half a clear."""
    from supervisor import task_reaper

    monkeypatch.setattr(
        "ouroboros.owner_mailbox.revoke_owner_control", lambda *_a, **_k: False,
    )
    meta = {
        "finalization_requested_at": 2000.0,
        "finalization_reason": "idle_timeout",
        "finalization_control_msg_id": "c1",
    }
    assert task_reaper.withdraw_finalization_grace(tmp_path, "t1", meta, chat_id=0) is False
    assert meta["finalization_requested_at"] == 2000.0
    assert meta["finalization_control_msg_id"] == "c1"


def test_revoked_mailbox_control_is_never_delivered(tmp_path):
    """Revocation is resolved over the WHOLE append-only mailbox, so a control
    retracted before its reader drained it is never seen; a later episode's
    control still is, and the revocation line is protocol, never owner prose."""
    from ouroboros import owner_mailbox as omb

    omb.write_owner_message(tmp_path, "idle_timeout", "t1", msg_id="c1",
                            kind=omb.KIND_FINALIZE_NOW)
    omb.write_owner_message(tmp_path, "hello", "t1", msg_id="m1")
    assert omb.revoke_owner_control(tmp_path, "t1", "c1") is True
    omb.write_owner_message(tmp_path, "deadline", "t1", msg_id="c2",
                            kind=omb.KIND_FINALIZE_NOW)

    entries = omb.drain_owner_entries(tmp_path, "t1")
    assert [(e["msg_id"], e["kind"]) for e in entries] == [
        ("m1", omb.KIND_OWNER_TEXT), ("c2", omb.KIND_FINALIZE_NOW),
    ]
    assert omb.drain_owner_messages(tmp_path, "t1") == ["hello"]


# --------------------------------------------------------------------------
# B5: cancellation must rescue the partial result the way a timeout does.
# --------------------------------------------------------------------------

def test_cancel_and_timeout_paths_share_one_salvage_helper():
    lifecycle = (REPO_ROOT / "supervisor" / "task_lifecycle.py").read_text(encoding="utf-8")
    reaper = (REPO_ROOT / "supervisor" / "task_reaper.py").read_text(encoding="utf-8")
    delivery = (REPO_ROOT / "supervisor" / "terminal_delivery.py").read_text(encoding="utf-8")
    assert "salvaged_output_note" in reaper
    running = lifecycle.split("def _finish_captured_running", 1)[1].split("\ndef ", 1)[0]
    # Phase A: the running kill path salvages through the shared delivery-seam
    # helper (terminal_delivery.salvage_cancelled_output), which wraps the SAME
    # underlying salvaged_output_note the reaper uses.
    assert "_salvage_cancelled_output(" in running
    helper = delivery.split("def salvage_cancelled_output", 1)[1].split("\ndef ", 1)[0]
    assert "salvaged_output_note" in helper
    # The rescue must precede the write, which precedes the drive removal.
    assert running.index("_salvage_cancelled_output(") < running.index("write_task_result(")


def test_cancelled_result_carries_the_salvaged_output(tmp_path, monkeypatch):
    from ouroboros import observability

    drive = tmp_path / "drive"
    monkeypatch.setattr(
        observability, "latest_llm_response_text", lambda root, tid: "partial finding X",
    )
    note = observability.salvaged_output_note(drive, "task-1")
    assert "partial finding X" in note
    assert "salvaged best-effort" in note


def test_salvage_note_is_empty_without_evidence(tmp_path):
    from ouroboros import observability

    assert observability.salvaged_output_note(tmp_path / "missing", "task-1") == ""


def test_cancelling_a_subagent_rescues_its_partial_result(monkeypatch, tmp_path):
    """End to end: the drive is deleted by publication, so the note must already be
    in the terminal result — the asymmetry with the timeout path was exactly this."""
    from ouroboros import observability
    from ouroboros.task_results import load_task_result
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)

    state = {"alive": True}
    proc = types.SimpleNamespace(
        pid=4242, is_alive=lambda: state["alive"],
        join=lambda timeout=None: None, terminate=lambda: state.__setitem__("alive", False),
    )
    worker = types.SimpleNamespace(wid=0, proc=proc, busy_task_id="live-salvage", reaping=False)
    monkeypatch.setattr(workers, "WORKERS", {0: worker}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )
    monkeypatch.setattr(q, "RUNNING", {
        "live-salvage": {
            "task": {"id": "live-salvage", "delegation_role": "subagent"}, "worker_id": 0,
        },
    }, raising=False)
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: None)
    monkeypatch.setattr(
        observability, "latest_llm_response_text", lambda root, tid: "half-written answer",
    )

    assert q.cancel_task_custody("live-salvage") == q.CANCEL_CANCELLED
    result = load_task_result(tmp_path, "live-salvage")
    assert result["status"] == "cancelled"
    assert "half-written answer" in result["result"]


def _seed_llm_response(drive: pathlib.Path, task_id: str, text: str) -> None:
    """Persist a REAL observability manifest+blob the salvage reader consumes."""
    from ouroboros import observability

    blob = observability.write_blob(drive, {"message": {"content": text}})
    observability.write_call_manifest(
        drive, task_id=task_id, call_id="llm_0001_response",
        manifest={"full_payload_ref": blob},
    )


def test_salvage_preserves_the_full_output_before_the_child_drive_dies(tmp_path):
    """XG-7B.1: the note is a bounded preview, but the FULL text must land on the
    canonical drive — the one that outlives the child drive — before deletion."""
    from ouroboros import observability

    canonical = tmp_path / "canonical"
    child = tmp_path / "child"
    canonical.mkdir()
    child.mkdir()
    full = ("cognitive output line\n" * 400) + "TAIL-SENTINEL"
    assert len(full) > observability.SALVAGED_OUTPUT_NOTE_LIMIT
    _seed_llm_response(child, "sub-1", full)

    note = observability.salvaged_output_note(child, "sub-1", preserve_root=canonical)
    assert "OMISSION NOTE" in note  # the owner-facing preview stays bounded
    assert "full copy preserved at " in note
    preserved = pathlib.Path(note.split("full copy preserved at ", 1)[1].split(")", 1)[0])
    assert canonical in preserved.parents

    import shutil

    shutil.rmtree(child)  # what publication does to a cancelled subagent's drive
    assert preserved.read_text(encoding="utf-8") == full  # nothing was lost


def test_salvage_without_a_durable_copy_keeps_everything_in_the_note(tmp_path):
    """If no full copy can be preserved, the terminal result IS the only copy —
    so it must carry the whole text rather than silently dropping the tail."""
    from ouroboros import observability

    child = tmp_path / "child"
    child.mkdir()
    full = ("cognitive output line\n" * 400) + "TAIL-SENTINEL"
    _seed_llm_response(child, "sub-1", full)

    note = observability.salvaged_output_note(child, "sub-1")
    assert "TAIL-SENTINEL" in note


def test_cancelling_a_subagent_preserves_the_full_output_on_the_canonical_drive(
    monkeypatch, tmp_path,
):
    """End to end through the REAL cancel path: publication deletes the child
    drive, so the full blob must already have a copy on the canonical drive and
    the terminal result must point at it (XG-7B.1, BIBLE P1)."""
    from ouroboros import observability
    from ouroboros.headless import HEADLESS_TASKS_DIR
    from ouroboros.task_results import load_task_result
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)

    task_id = "full-salvage"
    child_drive = tmp_path / HEADLESS_TASKS_DIR / task_id / "data"
    child_drive.mkdir(parents=True)
    full = ("half-written answer line\n" * 400) + "TAIL-SENTINEL"
    assert len(full) > observability.SALVAGED_OUTPUT_NOTE_LIMIT
    _seed_llm_response(child_drive, task_id, full)

    state = {"alive": True}
    proc = types.SimpleNamespace(
        pid=4242, is_alive=lambda: state["alive"],
        join=lambda timeout=None: None, terminate=lambda: state.__setitem__("alive", False),
    )
    worker = types.SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    monkeypatch.setattr(workers, "WORKERS", {0: worker}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **k: state.__setitem__("alive", False),
    )
    monkeypatch.setattr(q, "RUNNING", {
        task_id: {
            "task": {
                "id": task_id, "delegation_role": "subagent",
                "child_drive_root": str(child_drive),
            },
            "worker_id": 0,
        },
    }, raising=False)
    monkeypatch.setattr(q, "_emit_cancel_task_done", lambda *a, **k: None)

    assert q.cancel_task_custody(task_id) == q.CANCEL_CANCELLED

    assert not child_drive.exists(), "publication no longer deletes the child drive?"
    result = load_task_result(tmp_path, task_id)
    assert result["status"] == "cancelled"
    assert "full copy preserved at " in result["result"]
    preserved = pathlib.Path(
        result["result"].split("full copy preserved at ", 1)[1].split(")", 1)[0]
    )
    assert tmp_path in preserved.parents, "the full copy must live on the canonical drive"
    assert preserved.read_text(encoding="utf-8") == full


def test_the_bind_host_is_never_stamped_from_settings_over_the_environment(monkeypatch):
    """`--host 0.0.0.0` lives in the environment, and both readers check env FIRST.

    Deriving the env-export list from SETTINGS_DEFAULTS swept this key in, so
    `apply_settings_to_env` wrote the settings value — normally the shipped 127.0.0.1
    default that no owner ever authored — back over the operator's explicit choice. The
    server bound correctly once, then `restart_current_process` handed the overwritten
    env to `execvpe` and the LAN-reachable server became loopback for good.
    """
    import os

    from ouroboros.config import (
        SETTINGS_DEFAULTS,
        SETTINGS_KEYS_NOT_EXPORTED_TO_ENV,
        apply_settings_to_env,
        settings_env_keys,
    )

    assert "OUROBOROS_SERVER_HOST" in SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    assert "OUROBOROS_SERVER_HOST" not in settings_env_keys()

    monkeypatch.setenv("OUROBOROS_SERVER_HOST", "0.0.0.0")
    apply_settings_to_env(dict(SETTINGS_DEFAULTS))
    assert os.environ["OUROBOROS_SERVER_HOST"] == "0.0.0.0"


def test_start_agent_never_overwrites_an_operator_env_host(monkeypatch, tmp_path):
    """XG-7B.2: the export exclusion made ENV the bind-host authority, but
    start_agent re-stamped the MERGED settings value (usually the shipped
    127.0.0.1 default no owner ever authored) AFTER apply_settings_to_env —
    reintroducing for launcher-managed runs exactly the regression the
    exclusion closed. Driven THROUGH start_agent, not a re-implementation."""
    import io
    import os

    import launcher

    # Environ restored by the autouse conftest snapshot: apply_settings_to_env mutates the live mapping.
    captured = {}

    def _capture_popen(command, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return types.SimpleNamespace(pid=4242, stdout=io.BytesIO(b""), kill=lambda: None)

    monkeypatch.setattr(launcher, "_hidden_popen", _capture_popen)
    monkeypatch.setattr(launcher, "_write_server_process_record", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        launcher, "_load_settings", lambda: {"OUROBOROS_SERVER_HOST": "127.0.0.1"},
    )

    os.environ["OUROBOROS_SERVER_HOST"] = "0.0.0.0"  # the operator's explicit choice
    launcher.start_agent(port=52123)
    assert captured["env"]["OUROBOROS_SERVER_HOST"] == "0.0.0.0"

    os.environ.pop("OUROBOROS_SERVER_HOST", None)  # silence: settings may stand in
    launcher.start_agent(port=52123)
    assert captured["env"]["OUROBOROS_SERVER_HOST"] == "127.0.0.1"


def test_an_owner_restart_re_execs_without_the_inherited_runtime_mode_pin(monkeypatch, tmp_path):
    """The re-exec env is the whole mechanism, so assert on the real env it hands execvpe.

    ``OUROBOROS_BOOT_RUNTIME_MODE`` exists so a CHILD inherits the parent's ratchet
    baseline. Carried across the OWNER's own restart it also pinned the mode the
    owner had just raised in Settings: the replacement re-pinned the old baseline
    from this env and the new mode never took effect. An agent- or supervisor-
    initiated restart is the case the pin is for and must keep inheriting it.
    """
    import os

    from ouroboros.config import BOOT_RUNTIME_MODE_ENV_KEY
    from ouroboros.server_control import restart_current_process

    captured = {}

    def _capture_exec(_executable, _argv, env):
        captured["env"] = dict(env)

    monkeypatch.setattr(os, "execvpe", _capture_exec)
    monkeypatch.setenv(BOOT_RUNTIME_MODE_ENV_KEY, "light")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")

    restart_current_process(
        "127.0.0.1", 8765, repo_dir=tmp_path, log=logging.getLogger("test"),
        owner_initiated=True,
    )
    assert BOOT_RUNTIME_MODE_ENV_KEY not in captured["env"], \
        "an owner restart must let the child re-pin from load_settings()"
    # Only the PIN is dropped: the mode itself is re-authored from settings by
    # apply_settings_to_env before the child pins its baseline.
    assert captured["env"]["OUROBOROS_RUNTIME_MODE"] == "advanced"

    captured.clear()
    restart_current_process(
        "127.0.0.1", 8765, repo_dir=tmp_path, log=logging.getLogger("test"),
    )
    assert captured["env"][BOOT_RUNTIME_MODE_ENV_KEY] == "light", \
        "agent/supervisor restarts keep inheriting the owner-pinned baseline"
